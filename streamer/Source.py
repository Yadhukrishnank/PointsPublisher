# Source.py
from dataclasses import dataclass
from collections import deque
import time
import numpy as np
import cv2

# DepthAI is optional (dummy will work even if it's missing)
try:
    import depthai as dai
except Exception:
    dai = None

from Datasources import CameraConfig


# -------------------- Data containers --------------------

@dataclass
class Frame:
    rgb: np.ndarray           # HxWx3 uint8 (BGR, aligned)
    depth: np.ndarray         # HxW uint16 (mm)
    ts_host_s: float          # host timestamp seconds
    w: int
    h: int

@dataclass
class CamBundle:
    cam_id: int
    frame: Frame
    intr: CameraConfig

@dataclass
class PairedBundle:
    A: CamBundle
    B: CamBundle
    dt_ms: float


# -------------------- OAK devices (unchanged) --------------------

class OAKSingleDevice:
    """One OAK: RGB (preview) + depth aligned to RGB size."""
    def __init__(self, mx_id: str, rgb_w=1280, rgb_h=720, fps=30):
        if dai is None:
            raise RuntimeError("depthai not available; install 'depthai' or use the --dummy source.")
        self.mx = mx_id; self.rgb_w=rgb_w; self.rgb_h=rgb_h; self.fps=fps
        self.dev = dai.Device(dai.DeviceInfo(mx_id))
        self.pipeline = self._build_pipeline()
        self.dev.startPipeline(self.pipeline)
        self.qRgb = self.dev.getOutputQueue("rgb",   maxSize=4, blocking=False)
        self.qDep = self.dev.getOutputQueue("depth", maxSize=4, blocking=False)
        self._latest_rgb = None
        # intrinsics aligned to RGB output size
        calib = self.dev.readCalibration()
        K = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, self.rgb_w, self.rgb_h)
        self.intr = CameraConfig(fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2])

    def _build_pipeline(self):
        p = dai.Pipeline()
        camRgb = p.createColorCamera()
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camRgb.setFps(self.fps)
        camRgb.setPreviewSize(self.rgb_w, self.rgb_h)
        camRgb.setInterleaved(False)

        monoL = p.createMonoCamera(); monoR = p.createMonoCamera()
        monoL.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        monoR.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo = p.createStereoDepth()
        stereo.initialConfig.setConfidenceThreshold(180)
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
        stereo.setLeftRightCheck(True); stereo.setExtendedDisparity(True); stereo.setSubpixel(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(self.rgb_w, self.rgb_h)

        monoL.out.link(stereo.left); monoR.out.link(stereo.right)

        xRgb = p.createXLinkOut(); xRgb.setStreamName("rgb");  camRgb.preview.link(xRgb.input)
        xDep = p.createXLinkOut(); xDep.setStreamName("depth"); stereo.depth.link(xDep.input)
        return p

    def try_get(self):
        fRgb = self.qRgb.tryGet()
        if fRgb is not None:
            self._latest_rgb = fRgb.getCvFrame()
        fDep = self.qDep.tryGet()
        if fDep is None: return None
        depth = fDep.getFrame()  # uint16 mm
        ts = float(fDep.getTimestamp().total_seconds())
        rgb = self._latest_rgb if self._latest_rgb is not None else np.zeros((self.rgb_h, self.rgb_w, 3), np.uint8)
        H,W = depth.shape[:2]
        if (rgb.shape[0], rgb.shape[1]) != (H,W):
            rgb = cv2.resize(rgb, (W,H), interpolation=cv2.INTER_LINEAR)
        return Frame(rgb=rgb, depth=depth, ts_host_s=ts, w=W, h=H)

    def close(self):
        try: self.dev.close()
        except: pass


class MultiOAKSource:
    """Pairs two OAK devices by host timestamp with tolerance."""
    def __init__(self, mx_a: str, mx_b: str, tol_ms: float = 25.0, **kwargs):
        self.A = OAKSingleDevice(mx_a, **kwargs)
        self.B = OAKSingleDevice(mx_b, **kwargs)
        self.tol = tol_ms / 1000.0
        self.bufA, self.bufB = deque(maxlen=120), deque(maxlen=120)

    @staticmethod
    def _best_match(ts, dq):
        if not dq: return None, None, 1e9
        best_i, best_dt = None, 1e9
        for i,(t,_) in enumerate(dq):
            dt = abs(ts - t)
            if dt < best_dt: best_dt, best_i = dt, i
        return (best_i, dq[best_i][1], best_dt) if best_i is not None else (None, None, 1e9)

    def try_pair(self):
        fA = self.A.try_get()
        if fA is not None:
            i, fb, dt = self._best_match(fA.ts_host_s, self.bufB)
            if i is not None and dt <= self.tol:
                tsb, fB = self.bufB[i]; del self.bufB[i]
                return PairedBundle(
                    A=CamBundle(0, fA, self.A.intr),
                    B=CamBundle(1, fB, self.B.intr),
                    dt_ms = dt*1000.0
                )
            self.bufA.append((fA.ts_host_s, fA))

        fB = self.B.try_get()
        if fB is not None:
            i, fa, dt = self._best_match(fB.ts_host_s, self.bufA)
            if i is not None and dt <= self.tol:
                tsa, fA = self.bufA[i]; del self.bufA[i]
                return PairedBundle(
                    A=CamBundle(0, fA, self.A.intr),
                    B=CamBundle(1, fB, self.B.intr),
                    dt_ms = dt*1000.0
                )
            self.bufB.append((fB.ts_host_s, fB))
        return None

    def close(self):
        self.A.close(); self.B.close()


# -------------------- Dummy devices (tinted so you can tell A/B apart) --------------------

class DummySingleDevice:
    """
    Synthetic RGB-D generator with a base gradient RGB, optional tint, border & label,
    and a sloped/animated depth plane. Timestamp advances at 'fps'.
    """
    def __init__(self,
                 width=640, height=360, fps=30,
                 fx=591.4, fy=591.4,
                 # BGR tint (because OpenCV is BGR)
                 tint_bgr=(255, 200, 0),    # default cyan-ish
                 tint_alpha=0.35,
                 label_text=None,
                 border_px=8):
        self.w, self.h = int(width), int(height)
        self.fps = fps
        self.t = 0
        self.dt = 1.0 / float(max(1, fps))
        self.next_ts = time.time()
        self.intr = CameraConfig(fx=fx, fy=fy, cx=self.w/2.0, cy=self.h/2.0)

        self.tint_bgr = np.array(tint_bgr, dtype=np.uint8)
        self.tint_alpha = float(np.clip(tint_alpha, 0.0, 1.0))
        self.label_text = label_text
        self.border_px = int(max(0, border_px))

        # Precompute ramps for speed
        xs = np.linspace(0, 1, self.w, dtype=np.float32)[None, :]
        ys = np.linspace(0, 1, self.h, dtype=np.float32)[:, None]

        base = np.zeros((self.h, self.w, 3), np.float32)
        base[..., 0] = xs                               # B
        base[..., 1] = ys                               # G
        base[..., 2] = 0.25 + 0.5 * (xs * ys)           # R
        self.base_rgb = np.clip(base * 255.0, 0, 255).astype(np.uint8)

        self.col_ramp = (xs * 255.0).astype(np.uint8)   # used for subtle animation if desired

        # Depth base plane (in mm); we perturb with a small sinusoidal wobble per frame
        self.depth_base = ( (0.80 + 1.70 * (0.25 * xs + 0.75 * ys)) * 1000.0 ).astype(np.uint16)  # 0.8..2.5 m

        # Border mask
        if self.border_px > 0:
            m = np.zeros((self.h, self.w), np.uint8)
            m[:self.border_px, :] = 1
            m[-self.border_px:, :] = 1
            m[:, :self.border_px] = 1
            m[:, -self.border_px:] = 1
            self.border_mask = m.astype(bool)
        else:
            self.border_mask = None

    def _apply_tint(self, img_bgr: np.ndarray) -> np.ndarray:
        if self.tint_alpha <= 0.0001:
            return img_bgr
        # cv2.addWeighted handles rounding/saturation; broadcast a solid tint image
        return cv2.addWeighted(img_bgr, 1.0 - self.tint_alpha,
                               np.full_like(img_bgr, self.tint_bgr, dtype=np.uint8),
                               self.tint_alpha, 0.0)

    def _decorate(self, img_bgr: np.ndarray):
        if self.border_mask is not None:
            img_bgr[self.border_mask] = self.tint_bgr
        if self.label_text:
            cv2.putText(img_bgr, self.label_text, (18, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, tuple(int(x) for x in self.tint_bgr),
                        thickness=3, lineType=cv2.LINE_AA)

    def try_get(self):
        # RGB: start from base gradient
        rgb = self.base_rgb.copy()
        # Optionally add a tiny time animation (green shift)
        rgb[..., 1] = (rgb[..., 1].astype(np.int32) + (self.t % 255)).clip(0, 255).astype(np.uint8)
        # Apply tint & decorations
        rgb = self._apply_tint(rgb)
        self._decorate(rgb)

        # Depth: base plane + gentle wobble (in mm)
        wobble = int(40 * np.sin(self.t * 0.15))  # +/- 40 mm
        depth = (self.depth_base.astype(np.int32) + wobble).clip(0, 65535).astype(np.uint16)

        ts = self.next_ts
        self.t += 1
        self.next_ts += self.dt
        return Frame(rgb=rgb, depth=depth, ts_host_s=ts, w=self.w, h=self.h)

    def close(self): pass


class MultiDummySource:
    """
    Two synthetic cameras A/B, paired by host timestamp (like MultiOAKSource),
    with distinguishable RGB streams (tints, borders, labels).
    """
    def __init__(self, width=640, height=360, fps=30, tol_ms=25.0):
        # Cam 0 = cyan, Cam 1 = magenta (BGR tuples)
        tintA = (255, 200, 0)   # cyan-ish in BGR
        tintB = (255, 0, 255)   # magenta in BGR

        self.A = DummySingleDevice(width, height, fps, tint_bgr=tintA, tint_alpha=0.35, label_text="CAM 0", border_px=8)
        self.B = DummySingleDevice(width, height, fps, tint_bgr=tintB, tint_alpha=0.35, label_text="CAM 1", border_px=8)

        # Introduce a slight phase to B so pairing logic is exercised
        self.B.next_ts += 0.005

        self.tol = tol_ms / 1000.0
        self.bufA, self.bufB = deque(maxlen=120), deque(maxlen=120)

    @staticmethod
    def _best_match(ts, dq):
        if not dq: return None, None, 1e9
        best_i, best_dt = None, 1e9
        for i,(t,_) in enumerate(dq):
            dt = abs(ts - t)
            if dt < best_dt: best_dt, best_i = dt, i
        return (best_i, dq[best_i][1], best_dt) if best_i is not None else (None, None, 1e9)

    def try_pair(self):
        fA = self.A.try_get()
        if fA is not None:
            i, fb, dt = self._best_match(fA.ts_host_s, self.bufB)
            if i is not None and dt <= self.tol:
                tsb, fB = self.bufB[i]; del self.bufB[i]
                return PairedBundle(
                    A=CamBundle(0, fA, self.A.intr),
                    B=CamBundle(1, fB, self.B.intr),
                    dt_ms=dt*1000.0
                )
            self.bufA.append((fA.ts_host_s, fA))

        fB = self.B.try_get()
        if fB is not None:
            i, fa, dt = self._best_match(fB.ts_host_s, self.bufA)
            if i is not None and dt <= self.tol:
                tsa, fA = self.bufA[i]; del self.bufA[i]
                return PairedBundle(
                    A=CamBundle(0, fA, self.A.intr),
                    B=CamBundle(1, fB, self.B.intr),
                    dt_ms=dt*1000.0
                )
            self.bufB.append((fB.ts_host_s, fB))
        return None

    def close(self):
        self.A.close(); self.B.close()
