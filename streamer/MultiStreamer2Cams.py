#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiStreamer2Cams.py
Two OAK cameras (or a 2-cam dummy generator with --dummy) → processing chain →
single-port multiplexed ZMQ publisher + preview.

Layers:
- Source.py          : MultiOAKSource / MultiDummySource (two devices, timestamp pairing, aligned RGB-D)
- ProcessingStep.py  : DepthClampAndMask → LocalMedianReject → (optional) CropROI → DownSampling → EncodeRGBAsJPEG
- Actions.py         : UDP discovery + multiplex ZMQ publisher (+ PreviewAction for RGB|Depth mosaic)
- Datasources.py     : CameraConfig/Culling helpers + extrinsics loader

Unity side:
- One PULL socket + UDP discovery.
- Receiver demuxes by cam_id and reads pose[16].
- Renderer dispatches the same compute twice per frame into the same buffers.
"""

import time
import argparse
import signal
import sys
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
from Datasources import Culling, load_extrinsics_npz

# -------------------------------
# Default configuration
# -------------------------------

DEFAULTS = dict(
    mx_a="184430102111900E00",
    mx_b="19443010B11CEF1200",
    rgb_w=1280,
    rgb_h=720,
    fps=30,
    pair_tol_ms=25.0,
    zmq_port=5555,
    discovery_port=5556,
    use_roi=False,
    roi_x0=160,
    roi_y0=90,
    roi_w=640,
    roi_h=360,
    downsample_block=2,          # 1 = off ; 2 → 640x360
    z_min_m=0.25,
    z_max_m=4.0,
    cull_zmin=0.05,
    cull_zmax=4.0,
    cull_x=1.0,
    cull_y=1.0,
    extr_a="extrinsics_184430102111900E00.npz",
    extr_b="extrinsics_19443010B11CEF1200.npz",
    jpeg_quality=80,
)

# -------------------------------
# Build processing chain
# -------------------------------

def build_chain(use_roi: bool,
                roi_x0: int, roi_y0: int, roi_w: int, roi_h: int,
                downsample_block: int,
                z_min_m: float, z_max_m: float,
                jpeg_quality: int):
    root = DepthClampAndMask(z_min_m=z_min_m, z_max_m=z_max_m)
    root.set_next(LocalMedianReject(win=3, thr_mm=120))
    if use_roi:
        root.set_next(CropROI(roi_x0, roi_y0, roi_w, roi_h))
    root.set_next(DownSampling(downsample_block)) \
        .set_next(EncodeRGBAsJPEG(quality=jpeg_quality))
    return root

# -------------------------------
# Graceful shutdown helpers
# -------------------------------

class _StopFlag:
    def __init__(self): self.stop = False

def _install_signal_handlers(flag: _StopFlag):
    def handler(sig, frame):
        print("\n[MultiStreamer2Cams] Caught signal, shutting down …")
        flag.stop = True
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

# -------------------------------
# Main runner
# -------------------------------

def main():
    # -------- args --------
    ap = argparse.ArgumentParser(description="Two-camera OAK streamer (single-port, multiplexed)")
    ap.add_argument("--dummy", action="store_true", help="Use synthetic 2-camera source (no hardware needed)")
    ap.add_argument("--mx-a", default=DEFAULTS["mx_a"], help="MX id for camera A")
    ap.add_argument("--mx-b", default=DEFAULTS["mx_b"], help="MX id for camera B")
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
    ap.add_argument("--extr-a", default=DEFAULTS["extr_a"])
    ap.add_argument("--extr-b", default=DEFAULTS["extr_b"])
    ap.add_argument("--jpeg-quality", type=int, default=DEFAULTS["jpeg_quality"])
    args = ap.parse_args()

    print("=== MultiStreamer2Cams: config ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # -------- load extrinsics --------
    T_wc_A = None
    T_wc_B = None
    if args.dummy:
        try:
            T_wc_A = load_extrinsics_npz(args.extr_a)
            T_wc_B = load_extrinsics_npz(args.extr_b)
        except Exception as e:
            print(f"[WARN] Dummy mode: extrinsics load failed ({e}). Using identity poses.")
            T_wc_A = np.eye(4, dtype=np.float32)
            T_wc_B = np.eye(4, dtype=np.float32)
    else:
        try:
            T_wc_A = load_extrinsics_npz(args.extr_a)
            T_wc_B = load_extrinsics_npz(args.extr_b)
        except Exception as e:
            print(f"[ERROR] Loading extrinsics: {e}")
            sys.exit(1)

    # -------- source (two devices + pairing) --------
    try:
        if args.dummy:
            src = MultiDummySource(width=args.rgb_w, height=args.rgb_h, fps=args.fps, tol_ms=args.pair_tol_ms)
            print("[INFO] Using Dummy source.")
        else:
            src = MultiOAKSource(
                args.mx_a, args.mx_b,
                tol_ms=args.pair_tol_ms,
                rgb_w=args.rgb_w, rgb_h=args.rgb_h,
                fps=args.fps
            )
            print("[INFO] Using OAK devices.")
    except Exception as e:
        print(f"[ERROR] Opening source: {e}")
        sys.exit(2)

    # -------- actions: publisher + preview --------
    try:
        pub = ZMQPublishMuxAction(port=args.port, discovery_port=args.disc_port)
    except Exception as e:
        print(f"[ERROR] Creating ZMQ publisher: {e}")
        src.close()
        sys.exit(3)

    preview = PreviewAction(window_name="MultiCam Preview", zmin_m=args.zmin, zmax_m=args.zmax)

    # -------- processing chain --------
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

    # -------- loop --------
    flag = _StopFlag()
    _install_signal_handlers(flag)

    fps_pairs = 0
    t_last = time.time()

    print("\nMultiStreamer2Cams: streaming… (press Ctrl+C or 'q' in preview to quit)")
    try:
        while not flag.stop:
            pair = src.try_pair()
            if pair is None:
                # keep UI responsive
                if preview._last_key == ord('q'):
                    break
                time.sleep(0.001)
                continue

            # === Process A ===
            rgbA_jpg, depthA = chain.process(pair.A.frame.rgb, pair.A.frame.depth)
            hA, wA = depthA.shape[:2]
            tsA_us = int(pair.A.frame.ts_host_s * 1e6)

            # === Process B ===
            rgbB_jpg, depthB = chain.process(pair.B.frame.rgb, pair.B.frame.depth)
            hB, wB = depthB.shape[:2]
            tsB_us = int(pair.B.frame.ts_host_s * 1e6)

            # --- Preview (press 'q' to quit) ---
            quitA = preview.show(0, rgbA_jpg, depthA, info_text=f"Δt≈{pair.dt_ms:.1f} ms")
            quitB = preview.show(1, rgbB_jpg, depthB, info_text=None)
            if quitA or quitB:
                break

            # --- Publish on one port (two messages per pair) ---
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

            # --- Simple FPS/log ---
            fps_pairs += 1
            now = time.time()
            if now - t_last >= 1.0:
                print(f"[Pairs/s] {fps_pairs} | Δt~{pair.dt_ms:.1f} ms | sizes A {wA}x{hA}, B {wB}x{hB}")
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

# -------------------------------
# Entrypoint
# -------------------------------

if __name__ == "__main__":
    main()
