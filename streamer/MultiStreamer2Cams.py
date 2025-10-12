# MultiStreamer2Cams.py
# Streams two camera feeds (RGB JPEG + z16 depth + intrinsics + 4x4 pose)
# over a single ZMQ PUSH socket. Includes UDP discovery responder.
# Uses hardcoded extrinsics (can still be overridden by flags if you want).

import argparse
import time
import threading
import socket
import struct
import os
import glob
from typing import Optional, Tuple

import numpy as np
import cv2
import zmq

# Local modules
from Source import MultiDummySource, MultiOAKSource, PairedBundle, CamBundle


# === Hardcoded extrinsics (edit these paths/keys) ============================
USE_HARDCODED_EXTRINSICS = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Put your files here (absolute or relative paths are fine):
HARDCODED_EXTR_A = os.path.join(SCRIPT_DIR, "extrinsics_A.npz")
HARDCODED_EXTR_B = os.path.join(SCRIPT_DIR, "extrinsics_B.npz")

# If your .npz has multiple arrays, name the one that is 4x4 T_world_from_camera.
# If single 4x4 in the file, set to None.
HARDCODED_POSE_KEY = "cam_to_world"  # or None
# ============================================================================


# --------------- CLI -----------------

def parse_args():
    ap = argparse.ArgumentParser("MultiStreamer2Cams")
    gsrc = ap.add_mutually_exclusive_group(required=False)
    gsrc.add_argument("--dummy", action="store_true",
                      help="Use two synthetic cameras with distinct tints.")
    gsrc.add_argument("--mx-a", type=str, default=None,
                      help="OAK device MXID for camera A")
    ap.add_argument("--mx-b", type=str, default=None,
                    help="OAK device MXID for camera B (required if --mx-a is given)")

    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=int, default=30)

    ap.add_argument("--port", type=int, default=5555, help="TCP port for ZMQ PUSH")
    ap.add_argument("--disc-port", type=int, default=5556, help="UDP discovery port")

    # You can still override hardcoded extrinsics via flags, if desired:
    ap.add_argument("--extr-a", type=str, default=None, help="NPZ extrinsics for cam A (4x4)")
    ap.add_argument("--extr-b", type=str, default=None, help="NPZ extrinsics for cam B (4x4)")
    ap.add_argument("--pose-key", type=str, default=None, help="Array name inside NPZ (e.g. cam_to_world)")

    ap.add_argument("--cull-min", type=float, default=0.05, help="Min depth (m)")
    ap.add_argument("--cull-max", type=float, default=4.0, help="Max depth (m)")
    ap.add_argument("--xcull", type=float, default=1.0, help="Unused ROI param")
    ap.add_argument("--ycull", type=float, default=1.0, help="Unused ROI param")

    ap.add_argument("--jpeg-q", type=int, default=85, help="JPEG quality (1..100)")

    ap.add_argument("--preview", action="store_true", help="OpenCV preview windows for A/B")

    ap.add_argument("--send-mode", choices=["drop", "block"], default="drop",
                    help="On backpressure: drop frame or block until receiver ready")

    ap.add_argument("--no-discovery", action="store_true", help="Disable UDP discovery responder")

    return ap.parse_args()


# --------------- UDP discovery responder -----------------

class DiscoveryResponder(threading.Thread):
    """
    Responds 'ZMQ_SERVER_HERE' to 'DISCOVER_ZMQ_SERVER' UDP broadcasts.
    """
    def __init__(self, port: int):
        super().__init__(daemon=True)
        self.port = port
        self._stop = threading.Event()

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        sock.bind(("0.0.0.0", self.port))
        sock.settimeout(0.5)
        print(f"[DISC] Listening on UDP :{self.port}")
        try:
            while not self._stop.is_set():
                try:
                    data, addr = sock.recvfrom(1024)
                    if data and data.decode(errors="ignore").startswith("DISCOVER_ZMQ_SERVER"):
                        sock.sendto(b"ZMQ_SERVER_HERE", addr)
                except socket.timeout:
                    pass
                except Exception as e:
                    print(f"[DISC] Error: {e}")
                    time.sleep(0.2)
        finally:
            sock.close()
            print("[DISC] Stopped")

    def stop(self):
        self._stop.set()


# --------------- Extrinsics loading -----------------

def _npz_first_4x4(path: str, key_hint: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        data = np.load(path)
        if key_hint and key_hint in data and getattr(data[key_hint], "shape", None) == (4, 4):
            return data[key_hint].astype(np.float32), key_hint
        for k in data.files:
            A = data[k]
            if getattr(A, "shape", None) == (4, 4):
                return A.astype(np.float32), k
        print(f"[EXTR] No 4x4 matrix in {path}")
    except Exception as e:
        print(f"[EXTR] Failed to load {path}: {e}")
    return None, None


def _log_pose(label: str, path: Optional[str], key: Optional[str], T: np.ndarray):
    t = T[0:3, 3]
    detR = float(np.linalg.det(T[0:3, 0:3]))
    pth = os.path.basename(path) if path else "(default)"
    print(f"[EXTR] {label}: {pth}  key={key or '-'}  t=[{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}] m  det(R)={detR:+.4f}")


def load_extrinsics(args) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads T_world_from_cam for A and B. Preference:
      1) hardcoded paths (if USE_HARDCODED_EXTRINSICS)
      2) --extr-a / --extr-b
      3) defaults: A = I, B = translate +0.5m on X
    """
    # defaults
    T_A = np.eye(4, dtype=np.float32)
    T_B = np.array([[1,0,0,0.50],
                    [0,1,0,0.00],
                    [0,0,1,0.00],
                    [0,0,0,1   ]], dtype=np.float32)
    pathA = None
    pathB = None
    keyA = None
    keyB = None

    # (1) Hardcoded override
    if USE_HARDCODED_EXTRINSICS:
        args.extr_a = HARDCODED_EXTR_A
        args.extr_b = HARDCODED_EXTR_B
        args.pose_key = HARDCODED_POSE_KEY

    # (2) Flags
    if args.extr_a and os.path.isfile(args.extr_a):
        A, keyA = _npz_first_4x4(args.extr_a, args.pose_key)
        if A is not None:
            T_A = A.astype(np.float32)
            pathA = args.extr_a
    if args.extr_b and os.path.isfile(args.extr_b):
        B, keyB = _npz_first_4x4(args.extr_b, args.pose_key)
        if B is not None:
            T_B = B.astype(np.float32)
            pathB = args.extr_b

    # Logs (and sanity on rotations)
    _log_pose("A", pathA, keyA, T_A)
    _log_pose("B", pathB, keyB, T_B)

    # Relative baseline & quick sanity
    try:
        T_BA = np.linalg.inv(T_A) @ T_B
        tBA = T_BA[0:3, 3]
        baseline = float(np.linalg.norm(tBA))
        print(f"[EXTR] Relative B_from_A: t=[{tBA[0]:+.3f} {tBA[1]:+.3f} {tBA[2]:+.3f}] m  |  baseline={baseline:.3f} m")
    except Exception as e:
        print(f"[EXTR] Warning: could not compute B_from_A: {e}")

    return T_A, T_B


# --------------- ZMQ helpers -----------------

def make_socket_and_bind(port: int, send_mode: str):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.LINGER, 0)                              # don't hang on exit
    sock.setsockopt(zmq.SNDHWM, 4 if send_mode == "drop" else 1000)
    sock.bind(f"tcp://*:{port}")
    print(f"[ZMQ] PUSH bound at tcp://*:{port}")
    return sock


def wait_for_peer(sock, timeout_s=5.0):
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLOUT)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        events = dict(poller.poll(200))
        if sock in events and (events[sock] & zmq.POLLOUT):
            print("[ZMQ] Downstream ready.")
            return True
    print("[ZMQ] No downstream yet. Will stream and drop until it connects.")
    return False


class SafeSender:
    def __init__(self, sock, mode="drop"):
        self.sock = sock
        self.drop = (mode == "drop")
        self.dropped = 0
        self._last_warn = 0.0

    def send(self, data: bytes, tag: str):
        if not self.drop:
            self.sock.send(data)   # blocking
            return True
        try:
            self.sock.send(data, flags=zmq.DONTWAIT)
            return True
        except zmq.Again:
            self.dropped += 1
            now = time.time()
            if now - self._last_warn > 1.0:
                print(f"[ZMQ] Backpressure: dropping frame '{tag}' (dropped={self.dropped})")
                self._last_warn = now
            return False


# --------------- Packet packing -----------------

def pack_pose_rowmajor_16f(M: np.ndarray) -> bytes:
    # Ensure float32, row-major
    Mf = np.array(M, dtype=np.float32).reshape(4, 4)
    return struct.pack("<16f", *Mf.flatten(order="C"))

def jpeg_bytes(bgr: np.ndarray, quality: int) -> bytes:
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return bgr.tobytes()
    return enc.tobytes()

def build_packet(cam_id: int,
                 bundle: CamBundle,
                 T_wc: np.ndarray,
                 cull_min_m: float,
                 cull_max_m: float,
                 x_cull: float,
                 y_cull: float,
                 jpeg_q: int) -> bytes:
    """
    Multiplexed packet layout (little-endian), matching ZmqFrameReceiver (two-cam):
    [1] cam_id
    [1] flags (0)
    [2] reserved
    [4] width (int32)
    [4] height (int32)
    [16] intrinsics fx,fy,cx,cy (4 floats)
    [16] culling   zmin,zmax, xCull, yCull (4 floats)
    [64] pose 4x4 row-major (16 floats)  -- T_world_from_camera
    [8] timestamp_us (uint64)
    [4] rgb_len (int32) + [rgb_len] bytes (JPEG)
    [4] depth_len (int32) + [depth_len] bytes (z16)
    """
    f = bundle.frame
    intr = bundle.intr
    w, h = int(f.w), int(f.h)

    rgb_jpg = jpeg_bytes(f.rgb, jpeg_q)
    depth_bytes = f.depth.tobytes(order="C")

    ts_us = int(f.ts_host_s * 1e6) if f.ts_host_s is not None else int(time.time() * 1e6)

    header = bytearray()
    header += struct.pack("<BBH", cam_id & 0xFF, 0, 0)          # cam_id, flags, reserved
    header += struct.pack("<ii", w, h)                          # width, height
    header += struct.pack("<ffff", float(intr.fx), float(intr.fy), float(intr.cx), float(intr.cy))
    header += struct.pack("<ffff", float(cull_min_m), float(cull_max_m), float(x_cull), float(y_cull))
    header += pack_pose_rowmajor_16f(T_wc)
    header += struct.pack("<Q", np.uint64(ts_us))

    payload = struct.pack("<i", len(rgb_jpg)) + rgb_jpg + struct.pack("<i", len(depth_bytes)) + depth_bytes
    return bytes(header) + payload


# --------------- Preview -----------------

class Previewer:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.last = 0.0

    def show(self, tag: str, bgr: np.ndarray):
        if not self.enabled: return
        cv2.imshow(tag, bgr)
        if time.time() - self.last > 0.2:
            cv2.waitKey(1)
            self.last = time.time()


# --------------- Main -----------------

def main():
    args = parse_args()

    # Override with hardcoded extrinsics (no flags needed)
    if USE_HARDCODED_EXTRINSICS:
        args.extr_a = HARDCODED_EXTR_A
        args.extr_b = HARDCODED_EXTR_B
        args.pose_key = HARDCODED_POSE_KEY
        print(f"[EXTR] Hardcoded paths enabled.")
        print(f"[EXTR]  A path = {args.extr_a}")
        print(f"[EXTR]  B path = {args.extr_b}")
        print(f"[EXTR]  key    = {args.pose_key or '(first 4x4)'}")

    # Discovery responder
    disc = None
    if not args.no_discovery:
        disc = DiscoveryResponder(args.disc_port)
        disc.start()

    # Choose source
    if args.dummy or (args.mx_a is None and args.mx_b is None):
        print("[SRC] Using DUMMY sources (A & B)")
        src = MultiDummySource(width=args.width, height=args.height, fps=args.fps)
    else:
        if args.mx_a is None or args.mx_b is None:
            raise SystemExit("Both --mx-a and --mx-b are required for OAK mode.")
        print(f"[SRC] Using OAK devices: A={args.mx_a}  B={args.mx_b}")
        src = MultiOAKSource(args.mx_a, args.mx_b, rgb_w=args.width, rgb_h=args.height, fps=args.fps)

    # Extrinsics (logs show whether loaded correctly)
    T_A, T_B = load_extrinsics(args)

    # ZMQ
    sock = make_socket_and_bind(args.port, args.send_mode)
    wait_for_peer(sock, timeout_s=5.0)
    sender = SafeSender(sock, mode=args.send_mode)

    # Preview
    prev = Previewer(args.preview)

    # Loop
    t0 = time.time()
    sent_pairs = 0
    try:
        while True:
            pair: Optional[PairedBundle] = src.try_pair()
            if pair is None:
                time.sleep(0.001)
                if args.preview:
                    if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                        break
                continue

            A: CamBundle = pair.A
            B: CamBundle = pair.B

            # Build packets
            pktA = build_packet(0, A, T_A, args.cull_min, args.cull_max, args.xcull, args.ycull, args.jpeg_q)
            pktB = build_packet(1, B, T_B, args.cull_min, args.cull_max, args.xcull, args.ycull, args.jpeg_q)

            # Preview (just RGB)
            prev.show("A", A.frame.rgb)
            prev.show("B", B.frame.rgb)

            # Send with backpressure handling
            sender.send(pktA, "cam0")
            sender.send(pktB, "cam1")

            sent_pairs += 1
            now = time.time()
            if now - t0 >= 1.0:
                print(f"[STAT] Pairs/s={sent_pairs}  dt_match={pair.dt_ms:.2f} ms  dropped={sender.dropped}")
                t0 = now
                sent_pairs = 0

    except KeyboardInterrupt:
        print("[MAIN] Ctrl+C")
    except Exception as e:
        print(f"[MAIN] Error: {e}")
    finally:
        if disc: disc.stop()
        try:
            sock.close(0)
        except Exception:
            pass
        try:
            src.close()
        except Exception:
            pass
        if args.preview:
            try: cv2.destroyAllWindows()
            except Exception: pass
        print("[MAIN] Exit.")


if __name__ == "__main__":
    main()
