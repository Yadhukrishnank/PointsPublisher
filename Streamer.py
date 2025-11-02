# Entry point
import sys, time, threading, yaml, inspect, argparse
from pathlib import Path
import numpy as np
import cv2
import zmq
import socket, struct, json

from streamer.ProcessingStep import build_default_steps
from streamer.net.PacketV2 import PacketV2Writer, MAGIC, VERSION, FLAG_POSE, FLAG_INTR
from streamer.Actions import PREVIEW, ZmqPublisherV2, _resolve_colormap_code, _colorize_depth_mm

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
            # try common keys
            for key in ("T_wc", "pose", "T", "matrix", "M"):
                if key in data and data[key].shape == (4, 4):
                    return data[key].astype(np.float32)
            # handle R / t case
            if "R" in data and "t" in data:
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = data["R"].reshape(3, 3)
                T[:3,  3] = data["t"].reshape(3)
                return T
            print(f"[Pose] {p.name}: no 4x4 key (expected T_wc/pose/T/matrix/M)")
            return None
        # fallback: txt or raw .npy
        try:
            T = np.loadtxt(p, dtype=np.float32).reshape(4, 4)
            return T
        except Exception:
            T = np.load(p).astype(np.float32).reshape(4, 4)
            return T
    except Exception as e:
        print(f"[Pose] Failed to load {p}: {e}")
        return None

def pose_to_unity_coords(T_cam_to_cal: np.ndarray | None) -> np.ndarray | None:
    """
    Convert pose from calibration world (X right, Y forward, Z up)
    into Unity world (X right, Y up, Z forward).

    We do this by swapping Y and Z axes of the world frame before sending.
    """
    if T_cam_to_cal is None:
        return None

    axis_swap = np.array([
        [1,0,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,0,1],
    ], dtype=np.float32)

    # camera -> unityWorld = axis_swap * (camera -> calibWorld)
    T_cam_to_unity = axis_swap @ T_cam_to_cal
    return T_cam_to_unity.astype(np.float32)


# ===================================================================
# =========== Quest/Unity auto-discovery responder ==================
# ===================================================================
#
# The Quest/Unity side (MultiZmqFrameReceiver.cs) will:
#  1. Broadcast UDP "DISCOVER_ZMQ_SERVER" to discoveryPort (default 5554)
#  2. Wait for a reply "ZMQ_SERVER_HERE"
#  3. Use the sender IP of that reply as 'host' for all ZMQ PULL sockets
#
# This thread is what answers that broadcast so the Quest never needs
# a hardcoded PC IP.

def _udp_discovery_worker(listen_port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # allow quick restart without "address already in use"
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", listen_port))
    print(f"[discovery] UDP responder listening on *:{listen_port}")
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            msg = data.decode("ascii", errors="ignore").strip()
            if msg == "DISCOVER_ZMQ_SERVER":
                # reply directly (unicast back to Quest/Unity caller)
                sock.sendto(b"ZMQ_SERVER_HERE", addr)
                print(f"[discovery] reply -> {addr[0]}:{addr[1]}")
        except Exception as e:
            # don't kill the whole responder on random socket hiccups
            print(f"[discovery] error: {e}")
            time.sleep(0.1)

def start_udp_discovery_responder(port: int):
    t = threading.Thread(
        target=_udp_discovery_worker,
        args=(port,),
        daemon=True
    )
    t.start()
    return t

# ------------------ IRIS / scene auto-spawn helpers ------------------
# (kept as-is; harmless if you don't use IRIS)

MCAST_GRP  = "239.255.10.10"
MCAST_PORT = 7720

def discover_iris_unity(timeout: float = 3.0, iface_ip: str = "0.0.0.0") -> dict:
    """
    Listen for one IRISXRNode heartbeat and return
    {'ip','port','node_id','node_info_id'}.
    Unity/Quest broadcasts: nodeID(36) + nodeInfoID(36) + port(as text).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", MCAST_PORT))

    # join multicast group
    mreq = struct.pack("4s4s", socket.inet_aton(MCAST_GRP), socket.inet_aton(iface_ip))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.settimeout(timeout)

    try:
        data, (ip, _src_port) = sock.recvfrom(1024)
    finally:
        sock.close()

    s = data.decode("utf-8", errors="ignore")
    node_id      = s[:36]
    node_info_id = s[36:72]
    port         = int(s[72:])
    return {"ip": ip, "port": port, "node_id": node_id, "node_info_id": node_info_id}

def spawn_sim_scene(ip: str, port: int, scene_name: str = "PointClouds") -> None:
    ctx  = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 3000)   # 3s timeout
    sock.connect(f"tcp://{ip}:{int(port)}")
    payload = json.dumps({"name": scene_name}).encode("utf-8")
    print(f"[IRIS] -> SpawnSimScene to {ip}:{port} name={scene_name}")
    sock.send_multipart([b"SpawnSimScene", payload])
    reply = sock.recv()                   # raises if timeout
    print(f"[IRIS] <- {reply.decode('utf-8', 'ignore')}")

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

# ---- Azure fallback (only used if repo strategy can't be imported) ----
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

        # intrinsics
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

# ---- OAK fallback (only used if repo Luxonis strategy can't be imported) ----
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

        # RGB camera node
        cam_rgb = p.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setPreviewSize(self.w, self.h)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(30)

        # Mono cameras for stereo depth
        mono_l = p.create(dai.node.MonoCamera)
        mono_r = p.create(dai.node.MonoCamera)
        mono_l.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_r.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        mono_l.setFps(30)
        mono_r.setFps(30)

        # Stereo depth
        stereo = p.create(dai.node.StereoDepth)
        try:
            PM = dai.node.StereoDepth.PresetMode
        except AttributeError:
            # depthai 2.x style
            PM = dai.StereoDepth.PresetMode

        stereo.setDefaultProfilePreset(PM.MEDIUM_DENSITY)
        stereo.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(False)
        stereo.setLeftRightCheck(False)

        # Align to RGB if requested
        if self.align:
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

        # Force output size == rgb preview size, so rgb/depth match
        stereo.setOutputSize(self.w, self.h)

        # Link mono → stereo
        mono_l.out.link(stereo.left)
        mono_r.out.link(stereo.right)

        # XLink outputs for rgb/depth
        xout_rgb = p.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        xout_d = p.create(dai.node.XLinkOut)
        xout_d.setStreamName("depth")
        stereo.depth.link(xout_d.input)

        # Create device
        if self.mxid:
            info = dai.DeviceInfo(self.mxid)
            self.dev = dai.Device(p, info, usb2Mode=self.usb2)
        else:
            devs = dai.Device.getAllAvailableDevices()
            if not devs:
                raise RuntimeError("No OAK devices found.")
            info = devs[min(self.device_index, len(devs) - 1)]
            self.dev = dai.Device(p, info, usb2Mode=self.usb2)

        # Output queues
        self.qRgb   = self.dev.getOutputQueue("rgb",   maxSize=2, blocking=False)
        self.qDepth = self.dev.getOutputQueue("depth", maxSize=2, blocking=False)

        # Intrinsics from calibration
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
            depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

        class _Cfg: pass
        cfg = _Cfg(); cfg.fx, cfg.fy, cfg.cx, cfg.cy = self._intr
        return bgr, depth.astype(np.uint16), cfg

# ================= Intrinsics helpers (ROI / stride) =================

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
    """
    Apply ROI shift + downsample scaling to intrinsics in-place
    so Unity gets numbers that match the actually-sent images.
    """
    tup = _intr_get(intr)
    if tup is None:
        return intr
    fx, fy, cx, cy, kind = tup

    roi = global_cfg.get("roi_xywh")
    stride = int(global_cfg.get("downsample_stride", 1) or 1)

    if roi:
        x0, y0 = int(roi[0]), int(roi[1])
        cx -= x0
        cy -= y0

    if stride > 1:
        fx /= stride; fy /= stride; cx /= stride; cy /= stride

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
        return _FallbackAzureK4A(
            device_index=cam_cfg.get("device", 0),
            color_res=color_res,
            align_to_color=align_to_color
        )

    if cam_type in ("oak", "oakd", "luxonis"):
        try:
            from streamer.Source import LuxonisCameraStrategy as _LXS
            w, h = color_res
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

# ======================== One camera pipeline thread ========================

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


        # Load pose from file (camera -> calibration world)
        pose_raw = None
        if cam_cfg.get("pose_file"):
            pose_dir = Path(global_cfg.get("pose_dir", "."))
            p = Path(cam_cfg["pose_file"])
            if not p.is_absolute():
                p = pose_dir / p
            pose_raw = load_pose_4x4(str(p))

        # Convert that pose into Unity coordinates (camera -> Unity world)
        pose_unity = pose_to_unity_coords(pose_raw)
        self.pose = pose_unity


        if self.pose is not None:
            print(f"[Cam{self.cam_id}] T_cam_to_unity (row-major):")
            for r in range(4):
                print("    " + " ".join(f"{self.pose[r,c]: .6f}" for c in range(4)))
        else:
            print(f"[Cam{self.cam_id}] No pose file, using identity in publisher.")


        self.global_cfg = global_cfg
        self.strategy = make_strategy(cam_cfg, global_cfg)

        # Wrap strategy in CameraContext if available, else use it directly
        if CameraContext and hasattr(CameraContext, "__call__"):
            self.ctx = CameraContext(self.strategy)
            self._grab  = self.ctx.get_frame
            self._open  = getattr(self.ctx, "init",
                          getattr(self.ctx, "connect",
                          getattr(self.ctx, "open", None)))
            self._close = getattr(self.ctx, "close", lambda: None)
        else:
            self.ctx = self.strategy
            self._grab  = getattr(self.ctx, "get_frame", None)
            self._open  = getattr(self.ctx, "connect",
                          getattr(self.ctx, "open", None))
            self._close = getattr(self.ctx, "close", lambda: None)

        # Pre / post steps (Clamp → Median → ROI crop → Downsample, etc.)
        self.steps = build_default_steps(global_cfg)

        # This publisher creates and binds a PUSH socket on tcp://0.0.0.0:<port>
        # Unity connects with PULL to that same port.
        self.pub = ZmqPublisherV2(
            port=self.port,
            camera_id=self.cam_id,
            pose_4x4=self.pose,
            jpeg_quality=int(global_cfg.get("jpeg_quality", 80)),
            send_intrinsics=bool(global_cfg.get("send_intrinsics", True)),
        )

        self._fps_t0 = time.time()
        self._fps_n = 0
        self._next_deadline = time.time()

        # optional stagger so multiple devices don't all init at once
        self.startup_delay_s = float(global_cfg.get("open_stagger_ms", 600)) * 0.001 * (self.cam_id - 1)

    def run(self):
        try:
            if self.startup_delay_s > 0:
                time.sleep(self.startup_delay_s)
            if self._open:
                self._open()
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

                    # Apply processing steps
                    for s in self.steps:
                        try:
                            rgb, depth = s.process(rgb, depth)
                        except Exception:
                            # don't break stream if one step fails
                            pass

                    # Fix intrinsics to match crop/downsample
                    try:
                        cfg = _adjust_intrinsics_for_roi_stride(cfg, self.global_cfg)
                    except Exception:
                        pass

                    # Optional window preview
                    depth_bgr = None
                    if self.preview_depth and depth is not None:
                        depth_bgr = _colorize_depth_mm(
                            depth, self.depth_min_mm, self.depth_max_mm, self.depth_cmap_code
                        )
                    if self.preview_rgb:
                        PREVIEW.update(self.cam_id, rgb, depth_bgr)

                    # Push to ZMQ
                    self.pub.push((rgb, depth, cfg))
                else:
                    # If strategy returns a dict-like frame already shaped for PacketV2Writer
                    self.pub.push(tup)

                # FPS accounting
                self._fps_n += 1

                # Throttle to fps_max if requested
                if self.frame_period > 0:
                    self._next_deadline += self.frame_period
                    now = time.time()
                    if self._next_deadline > now:
                        time.sleep(self._next_deadline - now)
                    else:
                        self._next_deadline = now

                # Log ~every 2s
                now = time.time()
                if now - self._fps_t0 >= 2.0:
                    fps = self._fps_n / (now - self._fps_t0)
                    print(f"[Cam{self.cam_id}] port={self.port} ~{fps:.1f} fps")
                    self._fps_t0 = now
                    self._fps_n = 0

        finally:
            try:
                self._close()
            except:
                pass

# ============================= Main =============================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="Show OpenCV RGB preview")
    ap.add_argument("--preview-depth", action="store_true", help="Show colorized depth preview")
    ap.add_argument("--depth-min", type=int, default=None, help="Depth min (mm) for preview normalization")
    ap.add_argument("--depth-max", type=int, default=None, help="Depth max (mm) for preview normalization")
    ap.add_argument("--depth-cmap", type=str, default=None,
                    help="OpenCV colormap name (e.g., JET, TURBO, INFERNO, VIRIDIS, MAGMA)")
    return ap.parse_args()

def main():
    args = parse_args()

    cfg_path = Path("config/multicam.yaml")
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        print("Example:\n"
              "global:\n"
              "  base_port: 5555\n"
              "  jpeg_quality: 80\n"
              "  fps_max: 30\n"
              "  send_intrinsics: true\n"
              "  align_to_color: true\n"
              "  pose_dir: ./poses\n"
              "  preview: false\n"
              "  preview_depth: false\n"
              "  depth_min_mm: 400\n"
              "  depth_max_mm: 6000\n"
              "  depth_colormap: JET\n"
              "  discovery_port: 5554\n"
              "cameras:\n"
              "  - id: 1\n"
              "    type: oak\n"
              "    mxid: \"YOUR_DEVICE_MXID\"\n"
              "    port: 5555\n"
              "    color_res: [1280, 720]\n")
        sys.exit(1)

    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    global_cfg = cfg.get("global", {})
    cameras = cfg.get("cameras", [])
    if not cameras:
        print("No cameras in config.")
        sys.exit(1)

    # Preview settings (CLI overrides YAML)
    preview_rgb = bool(global_cfg.get("preview", False) or args.preview)
    preview_depth = bool(global_cfg.get("preview_depth", False) or args.preview_depth)

    dmin = int(args.depth_min if args.depth_min is not None else global_cfg.get("depth_min_mm", 400))
    dmax = int(args.depth_max if args.depth_max is not None else global_cfg.get("depth_max_mm", 6000))

    cmap_name_or_code = (
        args.depth_cmap
        if args.depth_cmap is not None
        else global_cfg.get("depth_colormap", "JET")
    )
    depth_cmap_code = _resolve_colormap_code(cmap_name_or_code)

    if preview_rgb or preview_depth:
        PREVIEW.enable(preview_rgb, preview_depth)

    # ------------------------------------------------------------
    # Start UDP discovery responder so Quest can auto-find this PC
    # ------------------------------------------------------------
    try:
        disc_port = int(global_cfg.get("discovery_port", 5554))
        start_udp_discovery_responder(disc_port)
    except Exception as e:
        print(f"[discovery] Failed to start responder: {e}")

    # ------------------------------------------------------------
    # Optional: IRIS auto-spawn (safe to leave, no effect if unused)
    # ------------------------------------------------------------
    try:
        do_autospawn = bool(global_cfg.get("iris_autospawn", True))
        if do_autospawn:
            timeout_s  = float(global_cfg.get("iris_timeout_s", 3.0))
            scene_name = str(global_cfg.get("iris_scene_name", "PointCloudViz"))
            print(f"[IRIS] Discovering Unity IRIS node (timeout {timeout_s:.1f}s)...")
            info = discover_iris_unity(timeout=timeout_s)
            print(f"[IRIS] Found node at {info['ip']}:{info['port']} (node {info['node_id']})")
            spawn_sim_scene(info["ip"], info["port"], scene_name=scene_name)
    except Exception as e:
        print(f"[IRIS] Discovery/spawn skipped or failed: {e}")

    # ------------------------------------------------------------
    # Spin up each camera capture + publisher thread
    # ------------------------------------------------------------
    print("=== Streamer (N-cam unified) ===")
    for c in cameras:
        port = resolve_port(c, int(global_cfg.get("base_port", 5555)))
        has_pose = "yes" if c.get("pose_file") else "no"
        print(f"  - Cam{c['id']}: type={c.get('type','azure')} port={port} pose={has_pose}")

    workers = [
        CameraPipeline(
            c,
            global_cfg,
            preview_rgb,
            preview_depth,
            dmin,
            dmax,
            depth_cmap_code
        )
        for c in cameras
    ]

    for w in workers:
        w.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    main()

# ============================= EOF =============================
#  python .\Streamer.py --preview --preview-depth --depth-min 400 --depth-max 6000 --depth-cmap TURBO

