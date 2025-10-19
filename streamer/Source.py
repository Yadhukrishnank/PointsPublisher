# Source.py  (DepthAI v2, two OAKs, plus two-camera dummy)
from dataclasses import dataclass
from collections import deque
import time
import numpy as np
import cv2

try:
    import depthai as dai  # DepthAI v2
except Exception:
    dai = None

from Datasources import CameraConfig

# -------------------- Data containers --------------------

@dataclass
class Frame:
    rgb: np.ndarray          # HxWx3 (BGR)
    depth: np.ndarray        # HxW   (uint16 mm)
    ts_host_s: float         # host timestamp (for cross-device pairing)
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

# -------------------- Single OAK (DepthAI v2) --------------------

class OAKSingleDevice:
    """
    One device: ColorCamera + Mono L/R + StereoDepth aligned to CAM_A (color).
    Exposes color intrinsics at (rgb_w, rgb_h).
    """
    def __init__(self, mx_id: str, rgb_w=1280, rgb_h=720, fps=30):
        if dai is None:
            raise RuntimeError("depthai not installed. Please install DepthAI v2.")
        self.mx = mx_id
        self.rgb_w = int(rgb_w)
        self.rgb_h = int(rgb_h)
        self.fps   = int(fps)

        self.dev = dai.Device(dai.DeviceInfo(mx_id))
        self.pipeline = self._build_pipeline()
        self.dev.startPipeline(self.pipeline)

        self.qRgb = self.dev.getOutputQueue("rgb",   maxSize=4, blocking=False)
        self.qDep = self.dev.getOutputQueue("depth", maxSize=4, blocking=False)

        self._intr = self._get_color_intrinsics()

    def _build_pipeline(self):
        p = dai.Pipeline()

        # Color
        cam = p.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setFps(self.fps)
        cam.setVideoSize(self.rgb_w, self.rgb_h)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # Mono + StereoDepth
        monoL = p.create(dai.node.MonoCamera)
        monoR = p.create(dai.node.MonoCamera)
        monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoL.setFps(self.fps)
        monoR.setFps(self.fps)

        stereo = p.create(dai.node.StereoDepth)
        stereo.setConfidenceThreshold(200)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        stereo.setRectifyEdgeFillColor(0)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align to color

        monoL.out.link(stereo.left)
        monoR.out.link(stereo.right)

        # Outputs
        xout_rgb = p.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam.video.link(xout_rgb.input)

        xout_depth = p.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        return p

    def _get_color_intrinsics(self) -> CameraConfig:
        calib = self.dev.readCalibration()
        K = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, self.rgb_w, self.rgb_h)
        fx = float(K[0][0]); fy = float(K[1][1]); cx = float(K[0][2]); cy = float(K[1][2])
        return CameraConfig(fx=fx, fy=fy, cx=cx, cy=cy)

    def try_get(self) -> Frame | None:
        fRgb = self.qRgb.tryGet()
        fDep = self.qDep.tryGet()
        if fRgb is None or fDep is None:
            return None
        rgb = fRgb.getCvFrame()   # BGR
        depth = fDep.getFrame()   # uint16 (mm), aligned to color
        if rgb is None or depth is None:
            return None
        if depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        ts = time.time()  # host time for pairing across devices
        return Frame(rgb=rgb, depth=depth, ts_host_s=ts, w=rgb.shape[1], h=rgb.shape[0])

    @property
    def intr(self) -> CameraConfig:
        return self._intr

    def close(self):
        try:
            self.qRgb.close(); self.qDep.close()
        except: pass
        try:
            self.dev.close()
        except: pass

# -------------------- Two-device wrapper --------------------

class MultiOAKSource:
    def __init__(self, mx_id_A: str, mx_id_B: str, rgb_w=1280, rgb_h=720, fps=30, tol_ms=25.0):
        self.A = OAKSingleDevice(mx_id_A, rgb_w, rgb_h, fps)
        self.B = OAKSingleDevice(mx_id_B, rgb_w, rgb_h, fps)
        self.tol = float(tol_ms) / 1000.0
        self.bufA, self.bufB = deque(maxlen=120), deque(maxlen=120)

    @staticmethod
    def _best_match(ts, dq):
        if not dq: return (None, None, 1e9)
        best_i, best_dt = None, 1e9
        for i, (t, fr) in enumerate(dq):
            dt = abs(ts - t)
            if dt < best_dt: best_dt, best_i = dt, i
        return (best_i, dq[best_i][1], best_dt) if best_i is not None else (None, None, 1e9)

    def try_get_pair(self) -> PairedBundle | None:
        fA = self.A.try_get()
        fB = self.B.try_get()

        if fA is not None:
            j, fb, dt = self._best_match(fA.ts_host_s, self.bufB)
            if j is not None and dt <= self.tol:
                _, fB2 = self.bufB[j]; del self.bufB[j]
                return PairedBundle(
                    A=CamBundle(0, fA, self.A.intr),
                    B=CamBundle(1, fB2, self.B.intr),
                    dt_ms=dt*1000.0
                )
            self.bufA.append((fA.ts_host_s, fA))

        if fB is not None:
            i, fa, dt = self._best_match(fB.ts_host_s, self.bufA)
            if i is not None and dt <= self.tol:
                _, fA2 = self.bufA[i]; del self.bufA[i]
                return PairedBundle(
                    A=CamBundle(0, fA2, self.A.intr),
                    B=CamBundle(1, fB,  self.B.intr),
                    dt_ms=dt*1000.0
                )
            self.bufB.append((fB.ts_host_s, fB))
        return None

    def close(self):
        self.A.close(); self.B.close()

# -------------------- Two-camera dummy --------------------

class MultiDummySource:
    """
    Two synthetic cameras that mimic real devices (for bandwidth / Unity testing).
    """
    def __init__(self, rgb_w=1280, rgb_h=720, fps=30, tol_ms=25.0):
        self.w, self.h, self.fps = int(rgb_w), int(rgb_h), int(fps)
        self.dt = 1.0 / max(1, self.fps)
        now = time.time()
        self.next_ts_A = now + 0.01
        self.next_ts_B = now + 0.015
        self.tol = float(tol_ms) / 1000.0
        fx = fy = 900.0
        cx, cy = self.w * 0.5, self.h * 0.5
        self.intrA = CameraConfig(fx, fy, cx, cy)
        self.intrB = CameraConfig(fx, fy, cx, cy)
        self.bufA, self.bufB = deque(maxlen=60), deque(maxlen=60)
        
    def _gen_rgb(self, phase: float) -> np.ndarray:
        """
        Return HxWx3 BGR uint8 image with all channels having shape (h, w).
        Previously we had (h,w), (h,1), (1,w) which breaks np.dstack.
        """
        h, w = self.h, self.w

        # R channel: horizontal gradient (0..255 across width)
        rx = np.linspace(0, 255, w, dtype=np.uint8)
        r = np.tile(rx, (h, 1))  # (h, w)

        # G channel: vertical gradient (0..255 across height)
        gy = np.linspace(0, 255, h, dtype=np.uint8)
        g = np.tile(gy.reshape(h, 1), (1, w))  # (h, w)

        # B channel: time-varying solid
        b_val = int((0.5 + 0.5 * np.sin(phase)) * 255) & 255
        b = np.full((h, w), b_val, dtype=np.uint8)  # (h, w)

        # BGR
        img = np.dstack([b, g, r]).astype(np.uint8, copy=False)  # (h, w, 3)
        return img


    def _gen_depth(self, z0_mm: int) -> np.ndarray:
        xv = np.linspace(0, 1, self.w, dtype=np.float32)
        yv = np.linspace(0, 1, self.h, dtype=np.float32)[:, None]
        z = z0_mm + 800.0*xv + 400.0*yv
        return z.astype(np.uint16)

    def _emit_one(self, cam: int) -> Frame:
        now = time.time()
        if cam == 0:
            self.next_ts_A = max(self.next_ts_A, now) + self.dt
            ts = self.next_ts_A
            rgb = self._gen_rgb(phase=ts*2.0)
            depth = self._gen_depth(1200)
        else:
            self.next_ts_B = max(self.next_ts_B, now) + self.dt
            ts = self.next_ts_B
            rgb = self._gen_rgb(phase=ts*2.0+1.0)
            depth = self._gen_depth(1400)
        return Frame(rgb=rgb, depth=depth, ts_host_s=ts, w=self.w, h=self.h)

    def try_get_pair(self) -> PairedBundle | None:
        fA = self._emit_one(0) if (time.time() >= self.next_ts_A) else None
        fB = self._emit_one(1) if (time.time() >= self.next_ts_B) else None
        if fA is not None:
            self.bufA.append((fA.ts_host_s, fA))
        if fB is not None:
            self.bufB.append((fB.ts_host_s, fB))
        if self.bufA and self.bufB:
            ta, fa = self.bufA[0]
            i, fb, dt = 0, None, 1e9
            for j, (tb, gb) in enumerate(self.bufB):
                d = abs(tb - ta)
                if d < dt: i, fb, dt = j, gb, d
            if dt <= self.tol:
                self.bufA.popleft()
                del self.bufB[i]
                return PairedBundle(
                    A=CamBundle(0, fa, self.intrA),
                    B=CamBundle(1, fb, self.intrB),
                    dt_ms=dt*1000.0
                )
        return None

    def close(self): pass
