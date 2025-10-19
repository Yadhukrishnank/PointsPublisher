# Actions.py
import struct
import socket
import threading
from typing import Optional, Tuple

import numpy as np
import cv2
import zmq

from Datasources import CameraConfig, Culling, scale_intrinsics

# -------------------- UDP discovery server --------------------

class UdpDiscoveryServer:
    """Responds to UDP broadcasts so the Quest/Unity can discover the ZMQ endpoint."""
    def __init__(self, port: int = 5556, response: bytes = b"ZMQ_SERVER_HERE"):
        self.port = int(port)
        self.response = response
        self._stop = False
        self._th: Optional[threading.Thread] = None

    def start(self):
        def loop():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", self.port))
            s.settimeout(0.25)
            while not self._stop:
                try:
                    data, addr = s.recvfrom(1024)
                    if data == b"DISCOVER_ZMQ_SERVER":
                        s.sendto(self.response, addr)
                except socket.timeout:
                    continue
                except Exception:
                    continue
            s.close()
        self._th = threading.Thread(target=loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop = True
        if self._th: self._th.join(timeout=0.2)

# -------------------- ZMQ single-port multiplex publisher --------------------

class ZMQPublishMuxAction:
    """
    Packet layout per frame (same head as single-cam; then pose + cam_id):
      uint32  width
      uint32  height
      uint32  rgb_len
      bytes   rgb_jpeg[rgb_len]
      uint32  depth_len
      bytes   depth_u16[depth_len]     // mm, width*height*2
      float32 fx, fy, cx, cy
      float32 zCullMin, zCullMax
      float32 xCull, yCull
      float32 pose_4x4[16]             // row-major T_world_from_camera
      uint8   cam_id
    """
    def __init__(self, port: int, cull: Culling, start_discovery_port: Optional[int] = 5556):
        self.cull = cull
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, 8)
        self.sock.bind(f"tcp://*:{int(port)}")

        self.discovery = None
        if start_discovery_port is not None and start_discovery_port > 0:
            self.discovery = UdpDiscoveryServer(port=start_discovery_port)
            self.discovery.start()

    @staticmethod
    def _pack_header_and_payload(width: int, height: int,
                                 rgb_jpeg: bytes, depth_u16: np.ndarray,
                                 intr: CameraConfig, cull: Culling,
                                 pose_4x4: np.ndarray, cam_id: int) -> bytes:
        if depth_u16 is None:
            depth_u16 = np.zeros((height, width), np.uint16)
        depth_u16 = np.ascontiguousarray(depth_u16.astype(np.uint16, copy=False))
        depth_bytes = depth_u16.tobytes(order='C')

        rgb_len = len(rgb_jpeg)
        depth_len = len(depth_bytes)

        assert pose_4x4.shape == (4, 4)
        pose = pose_4x4.astype(np.float32).reshape(-1)  # row-major 16

        parts = [
            struct.pack('<2I', width, height),
            struct.pack('<I', rgb_len), rgb_jpeg,
            struct.pack('<I', depth_len), depth_bytes,
            struct.pack('<4f', float(intr.fx), float(intr.fy), float(intr.cx), float(intr.cy)),
            struct.pack('<4f', float(cull.zcullmin), float(cull.zcullmax), float(cull.x_cull), float(cull.y_cull)),
            struct.pack('<16f', *pose),
            struct.pack('<B', int(cam_id) & 0xFF),
        ]
        return b''.join(parts)

    def send_frame(self, *, cam_id: int, width: int, height: int,
                   cfg_full: CameraConfig, cfg_scale: float, roi_off: Tuple[float, float],
                   rgb_jpeg_bytes: bytes, depth_u16: np.ndarray,
                   pose_4x4: np.ndarray):
        intr_eff = scale_intrinsics(cfg_full, cfg_scale, roi_off=roi_off)
        pkt = self._pack_header_and_payload(width, height,
                                            rgb_jpeg_bytes, depth_u16,
                                            intr_eff, self.cull, pose_4x4, cam_id)
        try:
            self.sock.send(pkt, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self):
        try:
            if self.discovery: self.discovery.stop()
        except: pass
        try:
            self.sock.close(0)
        except: pass

# -------------------- Hardened preview (2× cams) --------------------

class PreviewMosaic:
    def __init__(self, win_name="Preview 2×"):
        self.win = str(win_name)
        self._rows = [None, None]
        self._last_key = -1
        self._ver = "PreviewMosaic v2.2"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        print(f"[INFO] {self._ver} ready")

    @staticmethod
    def _decode_rgb(jpg_bytes: bytes) -> np.ndarray:
        if not jpg_bytes:
            return np.zeros((240, 320, 3), np.uint8)
        arr = np.frombuffer(jpg_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            img = np.zeros((240, 320, 3), np.uint8)
        return img

    @staticmethod
    def _reshape_depth_to_hw(depth_any, w: int, h: int) -> np.ndarray:
        """Force uint16 depth of shape (h,w) no matter what comes in."""
        a = np.asarray(depth_any)
        if a.ndim == 2 and a.shape == (h, w):
            d = a
        else:
            a = np.squeeze(a)
            if a.ndim == 1:
                if a.size == w * h:
                    d = a.reshape(h, w)
                else:
                    flat = a.reshape(-1)
                    need = w * h
                    if flat.size < need:
                        pad = np.pad(flat, (0, need - flat.size), mode='edge')
                        d = pad.reshape(h, w)
                    else:
                        d = flat[:need].reshape(h, w)
            elif a.ndim == 2:
                d = cv2.resize(a.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint16)
            else:
                flat = a.reshape(-1)
                need = w * h
                if flat.size < need:
                    flat = np.pad(flat, (0, need - flat.size), mode='edge')
                d = flat[:need].reshape(h, w)
        return np.ascontiguousarray(d.astype(np.uint16, copy=False))

    def _depth_vis(self, depth_u16: np.ndarray, w: int, h: int) -> np.ndarray:
        d = self._reshape_depth_to_hw(depth_u16, w, h)
        d_ = d.copy()
        d_[d_ == 0] = 65535
        d_ = np.clip(d_, 200, 6000)
        d8 = ((d_ - 200) * (255.0 / (6000 - 200))).astype(np.uint8)
        vis = cv2.applyColorMap(d8, cv2.COLORMAP_JET)  # (h,w,3)
        return vis

    def show(self, cam_id: int, rgb_jpg: bytes, depth_u16: np.ndarray, text: str = None, size=None) -> bool:
        """
        size MUST be (w,h) of the processed frame. We force both RGB & Depth to (h,w,3).
        """
        if size is None:
            rgb_guess = self._decode_rgb(rgb_jpg)
            size = (rgb_guess.shape[1], rgb_guess.shape[0])
        w, h = int(size[0]), int(size[1])

        rgb = self._decode_rgb(rgb_jpg)
        if rgb.shape[0] != h or rgb.shape[1] != w:
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        dep = self._depth_vis(depth_u16, w, h)

        row = cv2.hconcat([rgb, dep])
        if text:
            cv2.putText(row, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        self._rows[cam_id] = row

        if self._rows[0] is not None and self._rows[1] is not None:
            wmax = max(self._rows[0].shape[1], self._rows[1].shape[1])
            r0 = cv2.resize(self._rows[0], (wmax, self._rows[0].shape[0]))
            r1 = cv2.resize(self._rows[1], (wmax, self._rows[1].shape[0]))
            mosaic = cv2.vconcat([r0, r1])
            cv2.imshow(self.win, mosaic)

        self._last_key = cv2.waitKey(1) & 0xFF
        return self._last_key == ord('q')

    def close(self):
        try:
            cv2.destroyWindow(self.win)
        except:
            pass
