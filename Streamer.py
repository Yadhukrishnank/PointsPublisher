# Streamer.py — unified N-camera entrypoint
# - Reuses your repo's CameraContext + Strategy classes in streamer.Source
# - Optional OpenCV preview windows (YAML or CLI flags)
# - Sends v2 packets: header + optional intrinsics + optional pose
# - Applies ProcessingStep chain (Clamp → Median → ROI → Downsample)
# - Adjusts intrinsics (fx,fy,cx,cy) for ROI/stride before sending
# - Adds open-stagger per cam to improve multi-device boot (OAK)

import sys, time, threading, yaml, inspect, argparse
from pathlib import Path
import numpy as np
import cv2
import zmq

# bring in your processing steps
from streamer.ProcessingStep import build_default_steps

# ========================= Packet v2 =========================

MAGIC = 0xABCD1234
VERSION = 2
FLAG_POSE = 1   # bit0
FLAG_INTR = 2   # bit1

class PacketV2Writer:
    """
    Layout (little-endian):
    [Header 36B]: u32 MAGIC, u16 VERSION, u16 flags, u32 camera_id, u64 ts_us, u32 w, u32 h, u32 rgb_len, u32 depth_len
    [Intrinsics 16B] if flags&FLAG_INTR: float32[4] = fx,fy,cx,cy
    [RGB JPEG rgb_len B]
    [Depth U16 depth_len B] (aligned to color, mm)
    [Pose 4x4 64B] if flags&FLAG_POSE: float32[16] row-major world<-camera (T_wc)
    """
    def __init__(self, send_intrinsics: bool = True):
        self.send_intrinsics = send_intrinsics

    def pack(self, camera_id: int, timestamp_us: int, width: int, height: int,
             rgb_jpeg_bytes: bytes, depth_u16: np.ndarray,
             intrinsics: np.ndarray | None, pose_Twc: np.ndarray | None) -> bytes:
        import struct
        flags = 0
        if pose_Twc is not None:
            flags |= FLAG_POSE
        if self.send_intrinsics and intrinsics is not None:
            flags |= FLAG_INTR

        depth_u16 = np.asarray(depth_u16, dtype=np.uint16)
        depth_bytes = depth_u16.tobytes(order="C")

        header = struct.pack("<IHHI Q I I I I",
                             MAGIC, VERSION, flags, int(camera_id),
                             int(timestamp_us), int(width), int(height),
                             len(rgb_jpeg_bytes), len(depth_bytes))
        parts = [header]
        if flags & FLAG_INTR:
            intr = np.asarray(intrinsics, dtype=np.float32).reshape(4)
            parts.append(struct.pack("<4f", *intr))
        parts.append(rgb_jpeg_bytes)
        parts.append(depth_bytes)
        if flags & FLAG_POSE:
            T = np.asarray(pose_Twc, dtype=np.float32).reshape(16)
            parts.append(struct.pack("<16f", *T))
        return b"".join(parts)

# ====================== Pose loader (optional) ======================

def load_pose_4x4(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[Pose] Not found: {p}")
        return None
    try:
        if p.suffix.lower() == ".npz":
            data = np.load(p)
            for key in ("T_wc", "pose", "T", "matrix", "M"):
                if key in data and data[key].shape == (4, 4):
                    return data[key].astype(np.float32)
            if "R" in data and "t" in data:
                T = np.eye(4, dtype=np.float32)
                T[:3,:3] = data["R"].reshape(3,3)
                T[:3,  3] = data["t"].reshape(3)
                return T
            print(f"[Pose] {p.name}: no 4x4 key (expected T_wc/pose/T/matrix/M)")
            return None
        try:
            T = np.loadtxt(p, dtype=np.float32).reshape(4,4)
            return T
        except Exception:
            T = np.load(p).astype(np.float32).reshape(4,4)
            return T
    except Exception as e:
        print(f"[Pose] Failed to load {p}: {e}")
        return None

# =================== Repo strategies / fallbacks ===================

USE_EXISTING = True
CameraContext = None
AzureKinectStrategy = None

try:
    from streamer.Source import CameraContext as _CC
    CameraContext = _CC
except Exception as e:
    print(f"[Import] streamer.Source.CameraContext not available: {e}")
    USE_EXISTING = False

if USE_EXISTING:
    try:
        from streamer.Source import AzureKinectCameraStrategy as _AKS
        AzureKinectStrategy = _AKS
        print("[Import] Using repo strategy class: AzureKinectCameraStrategy")
    except Exception as e:
        print(f"[Import] Could not import AzureKinectCameraStrategy from streamer.Source: {e}")
        USE_EXISTING = False

# ---- Azure fallback (only if your repo strategy import fails) ----
class _FallbackAzureK4A:
    def __init__(self, device_index=0, color_res=(1280,720), align_to_color=True):
        try:
            from pyk4a import PyK4A, Config, ColorResolution, DepthMode, ImageFormat, CalibrationType
        except Exception as e:
            raise RuntimeError("pyk4a not installed and no Azure strategy found.") from e
        self.PyK4A = PyK4A
        self.Config = Config
        self.ColorResolution = ColorResolution
        self.DepthMode = DepthMode
        self.ImageFormat = ImageFormat
        self.CalibrationType = CalibrationType
        self.device_index = device_index
        self.color_res = color_res
        self.align = align_to_color
        self.k4a = None
        self._intr = None
        self._trans = None

    def open(self):
        res_map = {
            (1280,720): self.ColorResolution.RES_720P,
            (1920,1080): self.ColorResolution.RES_1080P,
            (2560,1440): self.ColorResolution.RES_1440P,
            (3840,2160): self.ColorResolution.RES_2160P,
        }
        cfg = self.Config(
            color_resolution=res_map.get(tuple(self.color_res), self.ColorResolution.RES_720P),
            depth_mode=self.DepthMode.NFOV_UNBINNED,
            color_format=self.ImageFormat.COLOR_BGRA32,
            synchronized_images_only=True
        )
        self.k4a = self.PyK4A(cfg, device_id=self.device_index)
        self.k4a.start()
        K = self.k4a.calibration.get_camera_matrix(self.CalibrationType.COLOR)
        fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
        self._intr = np.array([fx, fy, cx, cy], dtype=np.float32)
        self._trans = getattr(self.k4a, "transformation", None)
        if self._trans is None and self.align:
            print("[Azure] WARNING: 'transformation' not available; sending UNALIGNED depth.")
            self.align = False

    def close(self):
        if self.k4a:
            self.k4a.stop()
            self.k4a = None

    def get_frame(self):
        cap = self.k4a.get_capture()
        if cap is None or cap.color is None or cap.depth is None:
            return None
        depth = cap.depth
        if self.align and self._trans is not None:
            depth = self._trans.depth_image_to_color_camera(depth)
        bgr = cv2.cvtColor(cap.color, cv2.COLOR_BGRA2BGR)
        h, w = bgr.shape[:2]
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        class _Cfg: pass
        cfg = _Cfg(); cfg.fx, cfg.fy, cfg.cx, cfg.cy = self._intr
        return bgr, depth.astype(np.uint16), cfg

# ---- OAK fallback (used only if your repo lacks LuxonisCameraStrategy) ----
class _FallbackLuxonisOAK:
    def __init__(self, mxid=None, device_index=0, color_res=(1280,720), align_to_color=True, usb2mode=False):
        import depthai as dai
        self.dai = dai
        self.mxid = mxid
        self.device_index = int(device_index)
        self.w, self.h = map(int, color_res)
        self.align = bool(align_to_color)
        self.usb2 = bool(usb2mode)
        self.dev = None
        self.qRgb = None
        self.qDepth = None
        self._intr = None

    def open(self):
        dai = self.dai
        p = dai.Pipeline()

        cam_rgb = p.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setPreviewSize(self.w, self.h)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(30)

        mono_l = p.create(dai.node.MonoCamera)
        mono_r = p.create(dai.node.MonoCamera)
        mono_l.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_r.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        mono_l.setFps(30); mono_r.setFps(30)

        stereo = p.create(dai.node.StereoDepth)
        try: PM = dai.node.StereoDepth.PresetMode
        except AttributeError: PM = dai.StereoDepth.PresetMode
        stereo.setDefaultProfilePreset(PM.MEDIUM_DENSITY)
        stereo.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(False)
        stereo.setLeftRightCheck(False)
        if self.align:
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(self.w, self.h)

        mono_l.out.link(stereo.left)
        mono_r.out.link(stereo.right)

        xout_rgb = p.create(dai.node.XLinkOut); xout_rgb.setStreamName("rgb")
        xout_d   = p.create(dai.node.XLinkOut); xout_d.setStreamName("depth")
        cam_rgb.preview.link(xout_rgb.input)
        stereo.depth.link(xout_d.input)

        if self.mxid:
            info = dai.DeviceInfo(self.mxid)
            self.dev = dai.Device(p, info, usb2Mode=self.usb2)
        else:
            devs = dai.Device.getAllAvailableDevices()
            if not devs:
                raise RuntimeError("No OAK devices found.")
            info = devs[min(self.device_index, len(devs)-1)]
            self.dev = dai.Device(p, info, usb2Mode=self.usb2)

        self.qRgb   = self.dev.getOutputQueue("rgb",   maxSize=2, blocking=False)
        self.qDepth = self.dev.getOutputQueue("depth", maxSize=2, blocking=False)

        calib = self.dev.readCalibration()
        K = calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, self.w, self.h)
        fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
        self._intr = np.array([fx, fy, cx, cy], dtype=np.float32)

    def close(self):
        if self.dev is not None:
            self.dev.close()
            self.dev = None

    def get_frame(self):
        rgb_pkt = self.qRgb.tryGet()
        d_pkt   = self.qDepth.tryGet()
        if rgb_pkt is None or d_pkt is None:
            return None
        bgr   = rgb_pkt.getCvFrame()
        depth = d_pkt.getFrame().copy()
        if depth.shape[:2] != bgr.shape[:2]:
            depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        class _Cfg: pass
        cfg = _Cfg(); cfg.fx, cfg.fy, cfg.cx, cfg.cy = self._intr
        return bgr, depth.astype(np.uint16), cfg

# ====================== Preview (RGB + Depth) ======================

def _resolve_colormap_code(name_or_code):
    if isinstance(name_or_code, int):
        return int(name_or_code)
    name = str(name_or_code).strip().upper()
    return getattr(cv2, f"COLORMAP_{name}", cv2.COLORMAP_JET)

def _colorize_depth_mm(depth_u16: np.ndarray, dmin: int, dmax: int, cmap_code: int) -> np.ndarray:
    d = np.asarray(depth_u16, dtype=np.float32)
    valid = d > 0
    lo, hi = float(dmin), float(max(dmax, dmin + 1))
    d = np.clip(d, lo, hi)
    norm = (d - lo) * (255.0 / (hi - lo))
    norm[~valid] = 0.0
    img8 = norm.astype(np.uint8)
    cm = cv2.applyColorMap(img8, cmap_code)
    cm[~valid] = (0, 0, 0)
    return cm

class PreviewHub(threading.Thread):
    """Single UI thread showing latest RGB and Depth per camera. ESC closes windows."""
    def __init__(self):
        super().__init__(daemon=True)
        self.enabled_rgb = False
        self.enabled_depth = False
        self._run = False
        self._lock = threading.Lock()
        self._rgb = {}
        self._depth = {}

    def enable(self, rgb: bool, depth: bool):
        self.enabled_rgb = bool(rgb)
        self.enabled_depth = bool(depth)
        if (self.enabled_rgb or self.enabled_depth) and not self._run:
            self._run = True
            self.start()

    def update(self, cam_id: int, bgr: np.ndarray | None, depth_bgr: np.ndarray | None):
        if not self._run:
            return
        with self._lock:
            if self.enabled_rgb and bgr is not None:
                self._rgb[cam_id] = bgr
            if self.enabled_depth and depth_bgr is not None:
                self._depth[cam_id] = depth_bgr

    def run(self):
        while self._run:
            imgs = []
            with self._lock:
                if self.enabled_rgb:
                    imgs.extend(("RGB", cid, img) for cid, img in self._rgb.items())
                if self.enabled_depth:
                    imgs.extend(("Depth", cid, img) for cid, img in self._depth.items())
            for kind, cid, img in imgs:
                cv2.imshow(f"{kind} - Cam {cid}", img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                self._run = False
                break
        cv2.destroyAllWindows()

PREVIEW = PreviewHub()

# =================== ZMQ publisher (v2) ===================

class ZmqPublisherV2:
    def __init__(self, port: int, camera_id: int, pose_4x4: np.ndarray | None,
                 jpeg_quality: int = 80, send_intrinsics: bool = True,
                 bind_host: str = "0.0.0.0"):
        self.camera_id = int(camera_id)
        self.pose_4x4 = pose_4x4
        self.jpeg_quality = int(jpeg_quality)
        self.packet = PacketV2Writer(send_intrinsics=send_intrinsics)
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, 2)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.addr = f"tcp://{bind_host}:{int(port)}"
        self.sock.bind(self.addr)
        print(f"[ZmqPublisherV2] Cam{self.camera_id} PUSH {self.addr} pose={'yes' if pose_4x4 is not None else 'no'}")
        if self.pose_4x4 is not None:
            T = np.asarray(self.pose_4x4, dtype=np.float32).reshape(4,4)
            pretty = "\n".join("   " + " ".join(f"{v: .6f}" for v in row) for row in T)
            print(f"[Cam{self.camera_id}] T_wc (row-major, meters):\n{pretty}")

    def _extract(self, frame):
        """Return (bgr, rgb_jpeg_or_None, depth_u16, (w,h), intr[4] or None, ts_us)."""
        bgr = None; jpg = None; depth = None; size = None; intr = None; ts = None

        if isinstance(frame, tuple) and len(frame) == 3:
            bgr, depth, cfg = frame
            if bgr is not None:
                h, w = bgr.shape[:2]; size = (w, h)
            elif depth is not None and hasattr(depth, "shape"):
                h, w = depth.shape[:2]; size = (w, h)
            if hasattr(cfg, "fx"):
                intr = np.array([cfg.fx, cfg.fy, cfg.cx, cfg.cy], dtype=np.float32)
            if hasattr(cfg, "timestamp_us"):
                ts = int(cfg.timestamp_us)

        if isinstance(frame, dict):
            bgr   = frame.get("bgr") or frame.get("rgb") or frame.get("color") or bgr
            jpg   = frame.get("rgb_jpeg") or frame.get("jpeg") or jpg
            depth = frame.get("depth_u16") or frame.get("depth") or depth
            wh    = frame.get("size")
            size  = size or (wh if wh is not None else (frame.get("width"), frame.get("height")))
            intr  = frame.get("intrinsics") or frame.get("intr") or intr
            ts    = frame.get("timestamp_us") or frame.get("ts_us") or ts

        if size is None:
            if bgr is not None: h, w = bgr.shape[:2]; size = (w, h)
            elif depth is not None and hasattr(depth, "shape"): h, w = depth.shape[:2]; size = (w, h)
            else: raise ValueError("No size could be inferred from frame.")
        w, h = int(size[0]), int(size[1])

        if depth is None:
            return None, None, None, (w, h), intr, int(ts if ts is not None else time.time()*1e6)

        depth = np.asarray(depth, dtype=np.uint16).reshape(h, w)
        intr  = None if intr is None else np.asarray(intr, dtype=np.float32).reshape(4)
        ts    = int(ts if ts is not None else time.time() * 1e6)
        return bgr, jpg, depth, (w, h), intr, ts

    def push(self, frame):
        bgr, jpg, depth, (w, h), intr, ts = self._extract(frame)
        if depth is None:
            return
        if jpg is None:
            if bgr is None:
                return
            ok, enc = cv2.imencode(".jpg", np.asarray(bgr), [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                return
            jpg = enc.tobytes()
        payload = self.packet.pack(
            camera_id=self.camera_id, timestamp_us=ts,
            width=w, height=h, rgb_jpeg_bytes=jpg, depth_u16=depth,
            intrinsics=intr, pose_Twc=self.pose_4x4
        )
        try:
            self.sock.send(payload, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

# ================= Intrinsics helpers (ROI / stride adjustment) =================

def _intr_get(i):
    if i is None: return None
    try:
        return float(i.fx), float(i.fy), float(i.cx), float(i.cy), "obj"
    except AttributeError:
        return float(i["fx"]), float(i["fy"]), float(i["cx"]), float(i["cy"]), "dict"

def _intr_set(i, fx, fy, cx, cy, kind):
    if i is None: return None
    if kind == "obj":
        i.fx, i.fy, i.cx, i.cy = fx, fy, cx, cy
    else:
        i["fx"], i["fy"], i["cx"], i["cy"] = fx, fy, cx, cy
    return i

def _adjust_intrinsics_for_roi_stride(intr, global_cfg):
    """Apply ROI shift and downsample scaling to intrinsics in-place."""
    tup = _intr_get(intr)
    if tup is None:
        return intr
    fx, fy, cx, cy, kind = tup
    roi = global_cfg.get("roi_xywh")
    s   = int(global_cfg.get("downsample_stride", 1) or 1)
    if roi:
        x0, y0 = int(roi[0]), int(roi[1])
        cx -= x0
        cy -= y0
    if s > 1:
        fx /= s; fy /= s; cx /= s; cy /= s
    return _intr_set(intr, fx, fy, cx, cy, kind)

# ================= Strategy factory =================

def _safe_construct(cls, **kwargs):
    sig = inspect.signature(cls.__init__)
    allowed = {k:v for k,v in kwargs.items() if k in sig.parameters}
    try:
        return cls(**allowed) if allowed else cls()
    except Exception:
        return cls()

def make_strategy(cam_cfg: dict, global_cfg: dict):
    cam_type = str(cam_cfg.get("type", "azure")).lower()
    color_res = tuple(cam_cfg.get("color_res", [1280, 720]))
    align_to_color = bool(global_cfg.get("align_to_color", True))
    usb2mode = bool(cam_cfg.get("usb2mode", False))

    if cam_type in ("azure", "k4a"):
        if USE_EXISTING and CameraContext and AzureKinectStrategy:
            w, h = color_res
            return _safe_construct(
                AzureKinectStrategy,
                width=w, height=h,
                color_res=color_res,
                device_index=cam_cfg.get("device", 0),
                device=cam_cfg.get("device", 0),
                align_to_color=align_to_color,
                align=align_to_color
            )
        return _FallbackAzureK4A(device_index=cam_cfg.get("device", 0),
                                 color_res=color_res, align_to_color=align_to_color)

    if cam_type in ("oak", "oakd", "luxonis"):
        try:
            from streamer.Source import LuxonisCameraStrategy as _LXS
            w, h = color_res
            # Prefer MXID for multi-device
            return _safe_construct(
                _LXS,
                width=w, height=h,
                color_res=color_res,
                mxid=cam_cfg.get("mxid", None),
                device=cam_cfg.get("device", 0),
                device_index=cam_cfg.get("device", 0),
                usb2mode=usb2mode,
            )
        except Exception:
            return _FallbackLuxonisOAK(
                mxid=cam_cfg.get("mxid", None),
                device_index=cam_cfg.get("device", 0),
                color_res=color_res,
                align_to_color=align_to_color,
                usb2mode=usb2mode,
            )

    if cam_type in ("dummy", "sim", "test"):
        from streamer.Source import DummyCameraStrategy as _DCS
        w, h = color_res
        return _safe_construct(
            _DCS,
            width=w, height=h,
            color_res=color_res,
            fov_deg=cam_cfg.get("fov_deg", 70.0),
            pattern=cam_cfg.get("pattern", "checker"),
            z_mm=cam_cfg.get("z_mm", 1500),
            amp_mm=cam_cfg.get("amp_mm", 250),
            period_s=cam_cfg.get("period_s", 4.0),
            seed=cam_cfg.get("seed", None),
            align_to_color=align_to_color
        )

    raise ValueError(f"Unknown camera type: {cam_type}")

def resolve_port(cam_cfg: dict, base_port: int) -> int:
    p = cam_cfg.get("port")
    return int(p) if p is not None else int(base_port) + (int(cam_cfg["id"]) - 1)

# ======================== Pipeline worker ========================

class CameraPipeline(threading.Thread):
    def __init__(self, cam_cfg: dict, global_cfg: dict,
                 preview_rgb: bool, preview_depth: bool,
                 depth_min_mm: int, depth_max_mm: int, depth_cmap_code: int):
        super().__init__(daemon=True)
        self.cam_id = int(cam_cfg["id"])
        self.port = resolve_port(cam_cfg, int(global_cfg.get("base_port", 5555)))
        self.fps_max = int(global_cfg.get("fps_max", 30))
        self.frame_period = 1.0 / self.fps_max if self.fps_max > 0 else 0.0

        self.preview_rgb = bool(preview_rgb)
        self.preview_depth = bool(preview_depth)
        self.depth_min_mm = int(depth_min_mm)
        self.depth_max_mm = int(depth_max_mm)
        self.depth_cmap_code = int(depth_cmap_code)

        pose = None
        if cam_cfg.get("pose_file"):
            pose_dir = Path(global_cfg.get("pose_dir", "."))
            p = Path(cam_cfg["pose_file"])
            if not p.is_absolute():
                p = pose_dir / p
            pose = load_pose_4x4(str(p))
        self.pose = pose

        self.global_cfg = global_cfg
        self.strategy = make_strategy(cam_cfg, global_cfg)

        # Decide how to interact (context or direct)
        if CameraContext and hasattr(CameraContext, "__call__"):
            self.ctx = CameraContext(self.strategy)
            self._grab  = self.ctx.get_frame
            self._open  = getattr(self.ctx, "init", getattr(self.ctx, "connect", getattr(self.ctx, "open", None)))
            self._close = getattr(self.ctx, "close", lambda: None)
        else:
            self.ctx = self.strategy
            self._grab  = getattr(self.ctx, "get_frame", None)
            self._open  = getattr(self.ctx, "connect", getattr(self.ctx, "open", None))
            self._close = getattr(self.ctx, "close", lambda: None)

        # Build processing steps chain from YAML
        self.steps = build_default_steps(global_cfg)

        self.pub = ZmqPublisherV2(
            port=self.port, camera_id=self.cam_id, pose_4x4=self.pose,
            jpeg_quality=int(global_cfg.get("jpeg_quality", 80)),
            send_intrinsics=bool(global_cfg.get("send_intrinsics", True)),
        )
        self._fps_t0 = time.time()
        self._fps_n = 0
        self._next_deadline = time.time()
        self.startup_delay_s = float(global_cfg.get("open_stagger_ms", 600)) * 0.001 * (self.cam_id - 1)

    def run(self):
        try:
            # stagger opening per cam to avoid multi-device boot contention
            if self.startup_delay_s > 0:
                time.sleep(self.startup_delay_s)
            if self._open: self._open()
        except Exception as e:
            print(f"[Cam{self.cam_id}] FAILED to open device on port {self.port}: {e}")
            return

        try:
            while True:
                tup = self._grab()
                if tup is None:
                    continue

                if isinstance(tup, tuple) and len(tup) == 3:
                    rgb, depth, cfg = tup

                    # Steps (Clamp → Median → ROI → Downsample)
                    for s in self.steps:
                        try:
                            rgb, depth = s.process(rgb, depth)
                        except Exception:
                            pass

                    # Adjust intrinsics for ROI/stride (must match what the steps did)
                    try:
                        cfg = _adjust_intrinsics_for_roi_stride(cfg, self.global_cfg)
                    except Exception:
                        pass

                    # Depth preview
                    depth_bgr = None
                    if self.preview_depth and depth is not None:
                        depth_bgr = _colorize_depth_mm(depth, self.depth_min_mm, self.depth_max_mm, self.depth_cmap_code)

                    # RGB preview
                    if self.preview_rgb:
                        PREVIEW.update(self.cam_id, rgb, depth_bgr)

                    # Send
                    self.pub.push((rgb, depth, cfg))
                else:
                    self.pub.push(tup)

                self._fps_n += 1

                # Optional FPS throttle
                if self.frame_period > 0:
                    self._next_deadline += self.frame_period
                    now = time.time()
                    if self._next_deadline > now:
                        time.sleep(self._next_deadline - now)
                    else:
                        self._next_deadline = now

                # Periodic FPS log
                now = time.time()
                if now - self._fps_t0 >= 2.0:
                    fps = self._fps_n / (now - self._fps_t0)
                    print(f"[Cam{self.cam_id}] port={self.port} ~{fps:.1f} fps")
                    self._fps_t0 = now
                    self._fps_n = 0

        finally:
            try: self._close()
            except: pass

# ============================= Main =============================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="Show OpenCV RGB preview")
    ap.add_argument("--preview-depth", action="store_true", help="Show colorized depth preview")
    ap.add_argument("--depth-min", type=int, default=None, help="Depth min (mm) for preview normalization")
    ap.add_argument("--depth-max", type=int, default=None, help="Depth max (mm) for preview normalization")
    ap.add_argument("--depth-cmap", type=str, default=None, help="OpenCV colormap name (e.g., JET, TURBO, INFERNO, VIRIDIS, MAGMA)")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg_path = Path("config/multicam.yaml")
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        print("Example:\n"
              "global:\n"
              "  base_port: 5555\n  jpeg_quality: 80\n  fps_max: 30\n  send_intrinsics: true\n  align_to_color: true\n"
              "  pose_dir: ./poses\n  preview: false\n  preview_depth: false\n  depth_min_mm: 400\n  depth_max_mm: 6000\n  depth_colormap: JET\n"
              "cameras:\n"
              "  - id: 1\n    type: oak\n    mxid: \"YOUR_DEVICE_MXID\"\n    port: 5555\n    color_res: [1280, 720]\n")
        sys.exit(1)

    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    global_cfg = cfg.get("global", {})
    cameras = cfg.get("cameras", [])
    if not cameras:
        print("No cameras in config."); sys.exit(1)

    # Preview options (YAML or CLI)
    preview_rgb = bool(global_cfg.get("preview", False) or args.preview)
    preview_depth = bool(global_cfg.get("preview_depth", False) or args.preview_depth)
    dmin = int(args.depth_min if args.depth_min is not None else global_cfg.get("depth_min_mm", 400))
    dmax = int(args.depth_max if args.depth_max is not None else global_cfg.get("depth_max_mm", 6000))
    cmap_name_or_code = args.depth_cmap if args.depth_cmap is not None else global_cfg.get("depth_colormap", "JET")
    depth_cmap_code = _resolve_colormap_code(cmap_name_or_code)

    if preview_rgb or preview_depth:
        PREVIEW.enable(preview_rgb, preview_depth)

    print("=== Streamer (N-cam unified) ===")
    for c in cameras:
        port = resolve_port(c, int(global_cfg.get("base_port", 5555)))
        print(f"  - Cam{c['id']}: type={c.get('type','azure')} port={port} pose={'yes' if c.get('pose_file') else 'no'}")

    workers = [CameraPipeline(c, global_cfg, preview_rgb, preview_depth, dmin, dmax, depth_cmap_code) for c in cameras]
    for w in workers: w.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    main()
