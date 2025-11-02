# streamer/Actions.py
import time, threading
from typing import Optional, Tuple
import numpy as np
import cv2
import zmq

from streamer.net.PacketV2 import PacketV2Writer

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
            if key == 27:  # ESC aborts preview loop
                self._run = False
                break

        cv2.destroyAllWindows()


PREVIEW = PreviewHub()

# =================== ZMQ publisher (per camera) ===================
#
# We bind a PUSH socket on tcp://0.0.0.0:<port>.
# Unity/Quest creates a PULL socket and connects to tcp://<discovered_pc_ip>:<port>.
# That matches the new auto-discovery flow (no manual IP in headset).

class ZmqPublisherV2:
    def __init__(
        self,
        port: int,
        camera_id: int,
        pose_4x4: np.ndarray | None,
        jpeg_quality: int = 80,
        send_intrinsics: bool = True,
        bind_host: str = "0.0.0.0",
    ):
        self.camera_id = int(camera_id)
        self.pose_4x4 = pose_4x4
        self.jpeg_quality = int(jpeg_quality)
        self.packet = PacketV2Writer(send_intrinsics=send_intrinsics)

        # PUSH socket (sender binds, Quest connects with PULL)
        self.ctx  = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, 1)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.addr = f"tcp://{bind_host}:{int(port)}"
        self.sock.bind(self.addr)
        print(f"[ZmqPublisherV2] Cam{self.camera_id} PUSH {self.addr} "
              f"pose={'yes' if pose_4x4 is not None else 'no'}")
    # publisher--------------------------------------------------------
        # ctx = zmq.Context.instance()
        # self.sock = ctx.socket(zmq.PUB)    
        # self.sock.setsockopt(zmq.SNDHWM, 1)
        # self.sock.setsockopt(zmq.LINGER, 0)
        # self.addr = f"tcp://{bind_host}:{int(port)}"
        # self.sock.bind(self.addr)
        # time.sleep(0.25)                        
        # print(f"[ZmqPublisherV2] Cam{self.camera_id} PUB  {self.addr} "
        #       f"pose={'yes' if pose_4x4 is not None else 'no'}")

        if self.pose_4x4 is not None:
            T = np.asarray(self.pose_4x4, dtype=np.float32).reshape(4, 4)
            pretty = "\n".join(
                "   " + " ".join(f"{v: .6f}" for v in row) for row in T
            )
            print(f"[Cam{self.camera_id}] T_wc (row-major, meters):\n{pretty}")

    def _extract(self, frame):
        """
        Return (bgr, rgb_jpeg_or_None, depth_u16, (w,h), intr[4] or None, ts_us)
        in a predictable shape for PacketV2Writer.
        """
        bgr = None
        jpg = None
        depth = None
        size = None
        intr = None
        ts = None

        # tuple style: (bgr, depth_u16, cfg_with_intrinsics)
        if isinstance(frame, tuple) and len(frame) == 3:
            bgr, depth, cfg = frame
            if bgr is not None:
                h, w = bgr.shape[:2]
                size = (w, h)
            elif depth is not None and hasattr(depth, "shape"):
                h, w = depth.shape[:2]
                size = (w, h)

            if hasattr(cfg, "fx"):
                intr = np.array([cfg.fx, cfg.fy, cfg.cx, cfg.cy], dtype=np.float32)

            if hasattr(cfg, "timestamp_us"):
                ts = int(cfg.timestamp_us)

        # dict style (for future flexibility)
        if isinstance(frame, dict):
            bgr   = frame.get("bgr") or frame.get("rgb") or frame.get("color") or bgr
            jpg   = frame.get("rgb_jpeg") or frame.get("jpeg") or jpg
            depth = frame.get("depth_u16") or frame.get("depth") or depth
            wh    = frame.get("size")
            size  = size or (
                wh if wh is not None
                else (frame.get("width"), frame.get("height"))
            )
            intr  = frame.get("intrinsics") or frame.get("intr") or intr
            ts    = frame.get("timestamp_us") or frame.get("ts_us") or ts

        # final size resolution
        if size is None:
            if bgr is not None:
                h, w = bgr.shape[:2]; size = (w, h)
            elif depth is not None and hasattr(depth, "shape"):
                h, w = depth.shape[:2]; size = (w, h)
            else:
                raise ValueError("No size could be inferred from frame.")
        w, h = int(size[0]), int(size[1])

        if depth is None:
            return None, None, None, (w, h), intr, int(ts if ts is not None else time.time() * 1e6)

        depth = np.asarray(depth, dtype=np.uint16).reshape(h, w)
        intr  = None if intr is None else np.asarray(intr, dtype=np.float32).reshape(4)
        ts    = int(ts if ts is not None else time.time() * 1e6)
        return bgr, jpg, depth, (w, h), intr, ts

    def push(self, frame):
        import cv2

        bgr, jpg, depth, (w, h), intr, ts = self._extract(frame)
        if depth is None:
            return

        # If we haven't been given a pre-encoded JPEG, encode now.
        if jpg is None:
            if bgr is None:
                return
            ok, enc = cv2.imencode(
                ".jpg",
                np.asarray(bgr),
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if not ok:
                return
            jpg = enc.tobytes()

        payload = self.packet.pack(
            camera_id=self.camera_id,
            timestamp_us=ts,
            width=w,
            height=h,
            rgb_jpeg_bytes=jpg,
            depth_u16=depth,
            intrinsics=intr,
            pose_Twc=self.pose_4x4
        )

        try:
            self.sock.send(payload, flags=zmq.NOBLOCK)
        except zmq.Again:
            # drop frame instead of blocking
            pass
