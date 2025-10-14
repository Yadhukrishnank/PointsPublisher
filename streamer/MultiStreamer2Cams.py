#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiStreamer2Cams.py  (hardcoded MX IDs, simple extrinsics loading)
Two RGB-D cameras (or --dummy) → processing chain → single-port ZMQ publisher + preview.

Extrinsics (no flags):
- Looks for two files in the same folder as this script:
    extrinsics_cam0.npz   (pose for camera 0)
    extrinsics_cam1.npz   (pose for camera 1)
- Dummy mode: missing -> identity (logged)
- Real mode: missing -> error and exit
"""

import time
import argparse
import signal
import sys
from pathlib import Path
import numpy as np

from Source import MultiOAKSource, MultiDummySource
from ProcessingStep import (
    DepthClampAndMask,
    LocalMedianReject,
    CropROI,
    DownSampling,
    EncodeRGBAsJPEG,
)
from Actions import ZMQPublishMuxAction, PreviewAction
from Datasources import Culling

# --------- Hardcoded OAK MX IDs (edit these to your devices) ----------
MX_ID_CAM0 = "184430102111900E00"   # camera 0 (A)
MX_ID_CAM1 = "19443010B11CEF1200"   # camera 1 (B)

# --------- Simple defaults ----------
DEFAULTS = dict(
    rgb_w=1280, rgb_h=720, fps=30,
    pair_tol_ms=25.0,
    zmq_port=5555, discovery_port=5556,
    use_roi=False, roi_x0=160, roi_y0=90, roi_w=640, roi_h=360,
    downsample_block=2,      # 1 = off ; 2 → 640x360 from 1280x720
    z_min_m=0.25, z_max_m=4.0,
    cull_zmin=0.05, cull_zmax=4.0, cull_x=1.0, cull_y=1.0,
    jpeg_quality=80,
)

# --------- Extrinsics files (next to this script) ----------
SCRIPT_DIR = Path(__file__).resolve().parent
EXTR_A = SCRIPT_DIR / "extrinsics_cam0.npz"
EXTR_B = SCRIPT_DIR / "extrinsics_cam1.npz"

def _log_T(tag: str, T: np.ndarray, src: str):
    t = T[:3, 3]
    print(f"[Extrinsics] {tag}: {src}")
    print(f"             T_wc translation (m) = [{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}]")

def _load_T_wc_or_identity(path: Path, allow_identity: bool, tag: str) -> np.ndarray:
    """
    Load a 4x4 world_from_camera matrix from a .npz with common keys.
    If missing and allow_identity=True, returns identity (and logs).
    """
    if not path.exists():
        if allow_identity:
            T = np.eye(4, dtype=np.float32)
            print(f"[Extrinsics] {tag}: MISSING -> using IDENTITY (dummy fallback)")
            _log_T(tag, T, "IDENTITY")
            return T
        else:
            print(f"[ERROR] Missing extrinsics for {tag}: expected '{path.name}' next to this script.")
            sys.exit(1)

    try:
        data = np.load(path)
        # Try common keys
        for k in ("T_wc", "T", "world_from_camera", "matrix"):
            if k in data:
                T = np.array(data[k], dtype=np.float32)
                break
        else:
            # If only one array in the file, take it
            keys = list(data.keys())
            if len(keys) == 1:
                T = np.array(data[keys[0]], dtype=np.float32)
            else:
                raise KeyError("no suitable key found (expected T_wc/T/world_from_camera/matrix)")
        T = T.reshape(4, 4).astype(np.float32)
        _log_T(tag, T, f"file: {path.name}")
        return T
    except Exception as e:
        if allow_identity:
            T = np.eye(4, dtype=np.float32)
            print(f"[Extrinsics] {tag}: ERROR reading '{path.name}' -> using IDENTITY (dummy fallback) [{e}]")
            _log_T(tag, T, "IDENTITY")
            return T
        else:
            print(f"[ERROR] Failed to read extrinsics for {tag} from '{path.name}': {e}")
            sys.exit(1)

# --------- Processing chain ----------
def build_chain(use_roi, roi_x0, roi_y0, roi_w, roi_h, downsample_block, z_min_m, z_max_m, jpeg_quality):
    root = DepthClampAndMask(z_min_m=z_min_m, z_max_m=z_max_m)
    root.set_next(LocalMedianReject(win=3, thr_mm=120))
    if use_roi:
        root.set_next(CropROI(roi_x0, roi_y0, roi_w, roi_h))
    root.set_next(DownSampling(downsample_block)) \
        .set_next(EncodeRGBAsJPEG(quality=jpeg_quality))
    return root

# --------- Signal handling ----------
class _StopFlag:
    def __init__(self): self.stop = False

def _install_signal_handlers(flag: _StopFlag):
    def handler(sig, frame):
        print("\n[MultiStreamer2Cams] Caught signal, shutting down …")
        flag.stop = True
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

# --------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Two-camera RGB-D streamer (single-port, hardcoded MX IDs).")
    ap.add_argument("--dummy", action="store_true", help="Use synthetic 2-camera source (no hardware needed)")
    ap.add_argument("--rgb-w", type=int, default=DEFAULTS["rgb_w"])
    ap.add_argument("--rgb-h", type=int, default=DEFAULTS["rgb_h"])
    ap.add_argument("--fps", type=int, default=DEFAULTS["fps"])
    ap.add_argument("--pair-tol-ms", type=float, default=DEFAULTS["pair_tol_ms"])
    ap.add_argument("--port", type=int, default=DEFAULTS["zmq_port"], help="ZMQ TCP port (PUSH bind)")
    ap.add_argument("--disc-port", type=int, default=DEFAULTS["discovery_port"], help="UDP discovery port")
    ap.add_argument("--use-roi", action="store_true", default=DEFAULTS["use_roi"])
    ap.add_argument("--roi-x0", type=int, default=DEFAULTS["roi_x0"])
    ap.add_argument("--roi-y0", type=int, default=DEFAULTS["roi_y0"])
    ap.add_argument("--roi-w", type=int, default=DEFAULTS["roi_w"])
    ap.add_argument("--roi-h", type=int, default=DEFAULTS["roi_h"])
    ap.add_argument("--ds-block", type=int, default=DEFAULTS["downsample_block"])
    ap.add_argument("--zmin", type=float, default=DEFAULTS["z_min_m"])
    ap.add_argument("--zmax", type=float, default=DEFAULTS["z_max_m"])
    ap.add_argument("--cull-zmin", type=float, default=DEFAULTS["cull_zmin"])
    ap.add_argument("--cull-zmax", type=float, default=DEFAULTS["cull_zmax"])
    ap.add_argument("--cull-x", type=float, default=DEFAULTS["cull_x"])
    ap.add_argument("--cull-y", type=float, default=DEFAULTS["cull_y"])
    ap.add_argument("--jpeg-quality", type=int, default=DEFAULTS["jpeg_quality"])
    args = ap.parse_args()

    print("=== MultiStreamer2Cams: configuration ===")
    print(f"MX IDs: cam0={MX_ID_CAM0}, cam1={MX_ID_CAM1}")
    for k in vars(args).keys():
        if k not in ("dummy",):  # keep it short
            pass
    print(f"dummy={args.dummy}, rgb={args.rgb_w}x{args.rgb_h}@{args.fps}fps, pair_tol={args.pair_tol_ms}ms, ds={args.ds_block}")

    # -------- Extrinsics (simple, from next to script) --------
    T_wc_A = _load_T_wc_or_identity(EXTR_A, allow_identity=args.dummy, tag="Cam 0")
    T_wc_B = _load_T_wc_or_identity(EXTR_B, allow_identity=args.dummy, tag="Cam 1")

    # -------- Sources --------
    try:
        if args.dummy:
            src = MultiDummySource(width=args.rgb_w, height=args.rgb_h, fps=args.fps, tol_ms=args.pair_tol_ms)
            print("[INFO] Using Dummy source.")
        else:
            src = MultiOAKSource(
                MX_ID_CAM0, MX_ID_CAM1,
                tol_ms=args.pair_tol_ms,
                rgb_w=args.rgb_w, rgb_h=args.rgb_h,
                fps=args.fps
            )
            print(f"[INFO] Using OAK devices: cam0={MX_ID_CAM0}, cam1={MX_ID_CAM1}")
    except Exception as e:
        print(f"[ERROR] Opening source: {e}")
        sys.exit(2)

    # -------- Publisher + Preview --------
    try:
        pub = ZMQPublishMuxAction(port=args.port, discovery_port=args.disc_port)
    except Exception as e:
        print(f"[ERROR] Creating ZMQ publisher: {e}")
        src.close()
        sys.exit(3)

    preview = PreviewAction(window_name="MultiCam Preview", zmin_m=args.zmin, zmax_m=args.zmax)

    # -------- Processing chain --------
    chain = build_chain(
        use_roi=args.use_roi,
        roi_x0=args.roi_x0, roi_y0=args.roi_y0, roi_w=args.roi_w, roi_h=args.roi_h,
        downsample_block=max(1, args.ds_block),
        z_min_m=args.zmin, z_max_m=args.zmax,
        jpeg_quality=args.jpeg_quality
    )
    roi_off = (args.roi_x0, args.roi_y0) if args.use_roi else (0, 0)
    ds_block = max(1, args.ds_block)
    cull = Culling(zcullmin=args.cull_zmin, zcullmax=args.cull_zmax, x_cull=args.cull_x, y_cull=args.cull_y)

    # -------- Main loop --------
    flag = _StopFlag()
    _install_signal_handlers(flag)

    fps_pairs = 0
    t_last = time.time()

    print("\nMultiStreamer2Cams: streaming… (Ctrl+C or 'q' in preview to quit)")
    try:
        while not flag.stop:
            pair = src.try_pair()
            if pair is None:
                if preview._last_key == ord('q'):
                    break
                time.sleep(0.001)
                continue

            # === A ===
            rgbA_jpg, depthA = chain.process(pair.A.frame.rgb, pair.A.frame.depth)
            hA, wA = depthA.shape[:2]
            tsA_us = int(pair.A.frame.ts_host_s * 1e6)

            # === B ===
            rgbB_jpg, depthB = chain.process(pair.B.frame.rgb, pair.B.frame.depth)
            hB, wB = depthB.shape[:2]
            tsB_us = int(pair.B.frame.ts_host_s * 1e6)

            # Preview
            quitA = preview.show(0, rgbA_jpg, depthA, info_text=f"Δt≈{pair.dt_ms:.1f} ms")
            quitB = preview.show(1, rgbB_jpg, depthB, info_text=None)
            if quitA or quitB:
                break

            # Publish (two messages per pair on one port)
            pub.send_frame(
                cam_id=0, width=wA, height=hA,
                cfg_full=pair.A.intr, cull=cull,
                pose_4x4=T_wc_A, timestamp_us=tsA_us,
                rgb_jpeg_bytes=rgbA_jpg, depth_u16=depthA,
                ds_block=ds_block, roi_off=roi_off
            )
            pub.send_frame(
                cam_id=1, width=wB, height=hB,
                cfg_full=pair.B.intr, cull=cull,
                pose_4x4=T_wc_B, timestamp_us=tsB_us,
                rgb_jpeg_bytes=rgbB_jpg, depth_u16=depthB,
                ds_block=ds_block, roi_off=roi_off
            )

            # Simple throughput log
            fps_pairs += 1
            now = time.time()
            if now - t_last >= 1.0:
                print(f"[Pairs/s] {fps_pairs} | Δt~{pair.dt_ms:.1f} ms | A {wA}x{hA}, B {wB}x{hB}")
                fps_pairs = 0
                t_last = now

    except Exception as e:
        print(f"[ERROR] Runtime: {e}")

    finally:
        try: preview.close()
        except: pass
        try: pub.close()
        except: pass
        try: src.close()
        except: pass
        print("MultiStreamer2Cams: stopped.")

if __name__ == "__main__":
    main()
