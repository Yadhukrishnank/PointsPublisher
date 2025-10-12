# Actions.py
import struct
import socket
import threading
from typing import Optional

import numpy as np
import cv2
import zmq

from Datasources import CameraConfig, Culling, scale_intrinsics


class UdpDiscoveryServer:
    """Responds to UDP broadcasts so the Quest can discover the ZMQ endpoint."""
    def __init__(self, port: int = 5556, response: bytes = b"ZMQ_SERVER_HERE"):
        self.port = port
        self.response = response
        self._th: Optional[threading.Thread] = None
        self._stop = False

    def start(self):
        def loop():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", self.port))
            while not self._stop:
                try:
                    data, addr = s.recvfrom(1024)
                    if data == b"DISCOVER_ZMQ_SERVER":
                        s.sendto(self.response, addr)
                except Exception:
                    break
            s.close()

        self._th = threading.Thread(target=loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop = True


class ZMQPublishMuxAction:
    """
    Single PUSH socket (one port) that carries frames for multiple cameras.
    Packet layout (little-endian):
      uint8  cam_id
      uint8  flags (0)
      uint16 reserved (0)
      uint32 width, height
      float  fx, fy, cx, cy
      float  zmin, zmax, xCull, yCull
      float  pose[16]  // row-major 4x4: world_from_camera
      uint64 timestamp_us
      uint32 rgb_len
      uint8  rgb_jpeg[rgb_len]
      uint32 depth_len
      uint8  depth_z16[depth_len]
    """
    def __init__(self, port: int = 5555, discovery_port: int = 5556):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, 1)  # drop old frames if client is slow
        self.sock.bind(f"tcp://*:{port}")
        self.discovery = UdpDiscoveryServer(discovery_port)
        self.discovery.start()
        print(f"[ZMQ] PUSH bound tcp://*:{port}, UDP discovery on {discovery_port}")

    def close(self):
        try:
            self.sock.close(0)
        finally:
            try:
                self.discovery.stop()
            except Exception:
                pass

    def send_frame(
        self,
        cam_id: int,
        width: int,
        height: int,
        cfg_full: CameraConfig,
        cull: Culling,
        pose_4x4: np.ndarray,
        timestamp_us: int,
        rgb_jpeg_bytes: bytes,
        depth_u16: np.ndarray,
        ds_block: int = 1,
        roi_off=(0, 0),
    ):
        # Scale intrinsics for downsample and ROI (if any)
        scale = 1.0 / float(ds_block) if ds_block > 1 else 1.0
        cfg = scale_intrinsics(cfg_full, scale, roi_off)

        # Header
        header = (
            struct.pack("<BBH", cam_id, 0, 0)
            + struct.pack("<2I", width, height)
            + struct.pack("<4f", cfg.fx, cfg.fy, cfg.cx, cfg.cy)
            + struct.pack("<4f", cull.zcullmin, cull.zcullmax, cull.x_cull, cull.y_cull)
            + struct.pack("<16f", *pose_4x4.astype(np.float32).reshape(-1).tolist())
            + struct.pack("<Q", int(timestamp_us))
        )

        # Payloads
        rgb_bytes = rgb_jpeg_bytes if isinstance(rgb_jpeg_bytes, (bytes, bytearray)) else bytes(rgb_jpeg_bytes)
        depth_bytes = depth_u16.astype(np.uint16, copy=False).tobytes()

        packet = (
            header
            + struct.pack("<I", len(rgb_bytes))
            + rgb_bytes
            + struct.pack("<I", len(depth_bytes))
            + depth_bytes
        )

        try:
            self.sock.send(packet, zmq.NOBLOCK)
        except zmq.Again:
            # Drop if consumer is slow — keeps latency low
            pass


class PreviewAction:
    """
    OpenCV preview for 2-camera processed outputs.
    Call show(cam_id, rgb_jpeg_bytes, depth_u16, info_text) for A (0) and B (1).
    It renders a mosaic (A row, B row) with RGB | Depth side-by-side.
    """
    def __init__(self, window_name: str = "MultiCam Preview", zmin_m: float = 0.25, zmax_m: float = 4.0):
        self.win = window_name
        self.zmin = float(zmin_m)
        self.zmax = float(zmax_m)
        self._rows = {0: None, 1: None}  # cam_id -> row image
        self._last_key = -1
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)

    def _decode_rgb(self, rgb_jpeg_bytes: bytes) -> Optional[np.ndarray]:
        if not rgb_jpeg_bytes:
            return None
        arr = np.frombuffer(rgb_jpeg_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR

    def _colorize_depth(self, depth_u16: np.ndarray) -> np.ndarray:
        d = depth_u16.astype(np.float32) * 0.001  # mm -> m
        mask0 = d <= 0
        # normalize to [0..255]
        lo, hi = self.zmin, self.zmax
        rng = max(hi - lo, 1e-6)
        d = np.clip((d - lo) / rng, 0.0, 1.0)
        d[mask0] = 0.0
        gray = (d * 255.0).astype(np.uint8)
        col = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        col[mask0] = (0, 0, 0)
        return col

    def show(self, cam_id: int, rgb_jpeg_bytes: bytes, depth_u16: np.ndarray, info_text: Optional[str] = None) -> bool:
        rgb = self._decode_rgb(rgb_jpeg_bytes)
        dep = self._colorize_depth(depth_u16) if depth_u16 is not None else None
        if rgb is None and dep is None:
            return False

        # size match & side-by-side
        if rgb is None:
            rgb = np.zeros_like(dep)
        if dep is None:
            dep = np.zeros_like(rgb)
        if rgb.shape[:2] != dep.shape[:2]:
            dep = cv2.resize(dep, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        row = np.concatenate([rgb, dep], axis=1)

        # overlays
        h, w = row.shape[:2]
        cam_label = f"Cam {'A' if cam_id == 0 else 'B'}  {w//2}x{h}"
        cv2.putText(row, cam_label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        if info_text:
            cv2.putText(row, info_text, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 230, 50), 2, cv2.LINE_AA)

        self._rows[cam_id] = row

        # mosaic with rows for A and B
        rows = [r for r in [self._rows.get(0), self._rows.get(1)] if r is not None]
        if not rows:
            return False
        if len(rows) == 1:
            mosaic = rows[0]
        else:
            w0, w1 = rows[0].shape[1], rows[1].shape[1]
            if w0 != w1:
                tgt = max(w0, w1)
                def pad(img):
                    pad = tgt - img.shape[1]
                    if pad <= 0: return img
                    return np.pad(img, ((0, 0), (0, pad), (0, 0)), mode="constant")
                rows = [pad(rows[0]), pad(rows[1])]
            mosaic = np.vstack(rows)

        cv2.imshow(self.win, mosaic)
        self._last_key = cv2.waitKey(1) & 0xFF
        return self._last_key == ord('q')

    def close(self):
        try:
            cv2.destroyWindow(self.win)
        except Exception:
            pass
