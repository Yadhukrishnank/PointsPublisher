# Actions.py
import struct
import socket
import threading
from typing import Optional, Tuple

import numpy as np
import cv2
import zmq

from Datasources import CameraConfig, Culling, scale_intrinsics


class UdpDiscoveryServer:
    """Responds to UDP broadcasts so the Quest/Unity can discover the ZMQ endpoint."""
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
        # ------- IMPORTANT: allow two back-to-back frames (cam0, cam1) without dropping -------
        self.sock.setsockopt(zmq.SNDHWM, 8)   # more headroom than 1; keeps latency low but avoids constant drops
        self.sock.setsockopt(zmq.LINGER, 0)   # don't stall on shutdown
        self.sock.bind(f"tcp://*:{port}")
        self.discovery = UdpDiscoveryServer(discovery_port)
        self.discovery.start()
        self._drops = [0, 0]  # per-cam drop counters
        print(f"[ZMQ] PUSH bound tcp://*:{port}, UDP discovery on {discovery_port} | SNDHWM=8, LINGER=0")

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
        roi_off: Tuple[int, int] = (0, 0),
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
            # drop if consumer is slow — keeps latency low
            if 0 <= cam_id < 2:
                self._drops[cam_id] += 1
                if (self._drops[cam_id] % 30) == 1:
                    print(f"[ZMQ] DROP cam{cam_id}: total={self._drops[cam_id]} (consider raising SNDHWM)")
            else:
                print("[ZMQ] DROP (unknown cam id)")

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

    def _decode_rgb(self, jpg_bytes: bytes) -> np.ndarray:
        if isinstance(jpg_bytes, (bytes, bytearray)) and len(jpg_bytes) > 0:
            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        # fallback
        return np.zeros((1, 1, 3), np.uint8)

    def _depth_vis(self, depth_u16: np.ndarray) -> np.ndarray:
        d = depth_u16.astype(np.float32)
        d[d <= 0] = np.nan
        zmin = self.zmin * 1000.0
        zmax = self.zmax * 1000.0
        d = np.clip((d - zmin) / (zmax - zmin), 0.0, 1.0)
        d = (255.0 * (1.0 - d)).astype(np.uint8)  # near=bright
        return cv2.applyColorMap(d, cv2.COLORMAP_TURBO)

    def show(self, cam_id: int, rgb_jpg: bytes, depth_u16: np.ndarray, info_text: Optional[str] = None) -> bool:
        rgb = self._decode_rgb(rgb_jpg)
        dep = self._depth_vis(depth_u16)
        H = max(rgb.shape[0], dep.shape[0])
        rgb = cv2.resize(rgb, (dep.shape[1], dep.shape[0]), interpolation=cv2.INTER_LINEAR)
        row = np.hstack([rgb, dep])
        if info_text:
            cv2.putText(row, info_text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        self._rows[cam_id] = row

        if self._rows[0] is not None and self._rows[1] is not None:
            # stack rows
            w = max(self._rows[0].shape[1], self._rows[1].shape[1])
            r0 = cv2.resize(self._rows[0], (w, self._rows[0].shape[0]))
            r1 = cv2.resize(self._rows[1], (w, self._rows[1].shape[0]))
            mosaic = np.vstack([r0, r1])
            cv2.imshow(self.win, mosaic)

        self._last_key = cv2.waitKey(1) & 0xFF
        return self._last_key == ord('q')

    def close(self):
        try:
            cv2.destroyWindow(self.win)
        except Exception:
            pass
