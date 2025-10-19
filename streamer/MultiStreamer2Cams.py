#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
import cv2

from Datasources import Culling, load_extrinsics_npz
from ProcessingStep import (
    ProcessingStep, DepthClampAndMask, LocalMedianReject, CropROI, DownSampling, EncodeRGBAsJPEG
)
from Source import MultiOAKSource, MultiDummySource
from Actions import ZMQPublishMuxAction, PreviewMosaic

# --------- Replace with your MX IDs ----------
MX_ID_CAM0 = "184430102111900E00"   # camera 0
MX_ID_CAM1 = "19443010B11CEF1200"   # camera 1

DEFAULTS = dict(
    rgb_w=1280, rgb_h=720, fps=30,
    pair_tol_ms=25.0,
    zmq_port=5555, discovery_port=5556,
    use_roi=False, roi_x0=160, roi_y0=90, roi_w=640, roi_h=360,
    downsample_block=4,
    z_min_m=0.25, z_max_m=4.0,
    cull_zmin=0.05, cull_zmax=4.0, cull_x=1.0, cull_y=1.0,
    jpeg_quality=80
)

def build_chain(use_roi, roi_x0, roi_y0, roi_w, roi_h, downsample_block, z_min_m, z_max_m, jpeg_quality) -> ProcessingStep:
    root = DepthClampAndMask(z_min_m=z_min_m, z_max_m=z_max_m)
    tail = root
    tail = tail.set_next(LocalMedianReject(win=3, thr_mm=120))
    if use_roi:
        tail = tail.set_next(CropROI(roi_x0, roi_y0, roi_w, roi_h))
    tail = tail.set_next(DownSampling(downsample_block))
    tail = tail.set_next(EncodeRGBAsJPEG(quality=jpeg_quality))
    return root

def _normalize(rgb_jpg: bytes, depth_u16, w, h):
    # depth: ensure (h,w) uint16, C-contiguous
    if depth_u16 is None:
        depth_u16 = np.zeros((h, w), np.uint16)
    else:
        d = np.asarray(depth_u16)
        if d.ndim == 1 and d.size == w*h:
            d = d.reshape(h, w)
        elif d.ndim != 2:
            d = d.reshape(-1, 1)
        depth_u16 = np.ascontiguousarray(d.astype(np.uint16, copy=False))
    # rgb: ensure non-empty JPEG
    if not rgb_jpg:
        blank = np.zeros((h, w, 3), np.uint8)
        ok, buf = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        rgb_jpg = buf.tobytes() if ok else b""
    return rgb_jpg, depth_u16

def _find_extrinsics(here: str, mxid: str, fallback_name: str, allow_identity: bool, tag: str) -> np.ndarray:
    p1 = os.path.join(here, f"extrinsics_{mxid}.npz")
    p2 = os.path.join(here, fallback_name)
    if os.path.exists(p1):
        M = load_extrinsics_npz(p1)
        print(f"[Extrinsics] {tag}: {os.path.basename(p1)}")
        return M
    if os.path.exists(p2):
        M = load_extrinsics_npz(p2)
        print(f"[Extrinsics] {tag}: {os.path.basename(p2)}")
        return M
    if allow_identity:
        print(f"[Extrinsics] {tag}: {os.path.basename(p1)} / {os.path.basename(p2)} MISSING → identity")
        return np.eye(4, dtype=np.float32)
    raise FileNotFoundError(f"Missing extrinsics for {tag}: {p1} or {p2}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true", help="Use two dummy cameras instead of OAK devices")
    ap.add_argument("--mx0", default=MX_ID_CAM0, help="MX ID for camera 0 (ignored in --dummy)")
    ap.add_argument("--mx1", default=MX_ID_CAM1, help="MX ID for camera 1 (ignored in --dummy)")
    ap.add_argument("--rgb_w", type=int, default=DEFAULTS["rgb_w"])
    ap.add_argument("--rgb_h", type=int, default=DEFAULTS["rgb_h"])
    ap.add_argument("--fps",   type=int, default=DEFAULTS["fps"])
    ap.add_argument("--pair_tol_ms", type=float, default=DEFAULTS["pair_tol_ms"])
    ap.add_argument("--zmq_port", type=int, default=DEFAULTS["zmq_port"])
    ap.add_argument("--discovery_port", type=int, default=DEFAULTS["discovery_port"])
    ap.add_argument("--use_roi", action="store_true", default=DEFAULTS["use_roi"])
    ap.add_argument("--roi", nargs=4, type=int, metavar=("x0","y0","w","h"), default=None)
    ap.add_argument("--downsample_block", type=int, default=DEFAULTS["downsample_block"])
    ap.add_argument("--z_min_m", type=float, default=DEFAULTS["z_min_m"])
    ap.add_argument("--z_max_m", type=float, default=DEFAULTS["z_max_m"])
    ap.add_argument("--cull_zmin", type=float, default=DEFAULTS["cull_zmin"])
    ap.add_argument("--cull_zmax", type=float, default=DEFAULTS["cull_zmax"])
    ap.add_argument("--cull_x", type=float, default=DEFAULTS["cull_x"])
    ap.add_argument("--cull_y", type=float, default=DEFAULTS["cull_y"])
    ap.add_argument("--jpeg_quality", type=int, default=DEFAULTS["jpeg_quality"])
    ap.add_argument("--no-preview", action="store_true", help="Disable OpenCV preview windows")
    args = ap.parse_args()

    rgb_w, rgb_h, fps = args.rgb_w, args.rgb_h, args.fps
    roi = args.roi if args.roi is not None else (DEFAULTS["roi_x0"], DEFAULTS["roi_y0"], DEFAULTS["roi_w"], DEFAULTS["roi_h"])
    use_roi = bool(args.use_roi)
    ds_block = int(max(1, args.downsample_block))
    roi_off = (float(roi[0]) if use_roi else 0.0, float(roi[1]) if use_roi else 0.0)
    cfg_scale = 1.0 / float(ds_block)

    print("=== MultiStreamer2Cams: configuration ===")
    print(f"DepthAI v2, dummy={args.dummy}, rgb={rgb_w}x{rgb_h}@{fps}, pair_tol={args.pair_tol_ms:.1f}ms, ds={ds_block}")
    print(f"ZMQ tcp://*:{args.zmq_port}, discovery UDP :{args.discovery_port}")
    if use_roi:
        print(f"ROI = x0={roi[0]} y0={roi[1]} w={roi[2]} h={roi[3]} (intrinsics will be offset+scaled)")

    # Build processing chain + actions
    chain = build_chain(use_roi, *roi, ds_block, args.z_min_m, args.z_max_m, args.jpeg_quality)
    cull = Culling(args.cull_zmin, args.cull_zmax, args.cull_x, args.cull_y)
    pub  = ZMQPublishMuxAction(port=args.zmq_port, cull=cull, start_discovery_port=args.discovery_port)

    # Extrinsics (row-major T_world_from_camera)
    here = os.path.dirname(os.path.abspath(__file__))
    T_wc_A = _find_extrinsics(here, args.mx0, "extrinsics_cam0.npz", allow_identity=args.dummy, tag="Cam 0")
    T_wc_B = _find_extrinsics(here, args.mx1, "extrinsics_cam1.npz", allow_identity=args.dummy, tag="Cam 1")

    # Sources
    if args.dummy:
        src = MultiDummySource(rgb_w=rgb_w, rgb_h=rgb_h, fps=fps, tol_ms=args.pair_tol_ms)
    else:
        src = MultiOAKSource(args.mx0, args.mx1, rgb_w=rgb_w, rgb_h=rgb_h, fps=fps, tol_ms=args.pair_tol_ms)

    use_preview = not args.no_preview
    preview = PreviewMosaic("Preview (two cameras)") if use_preview else None

    frames = 0
    t0 = time.time()

    try:
        while True:
            pair = src.try_get_pair()
            if pair is None:
                time.sleep(0.001)
                continue

            # Cam A
            rgbA, depthA = pair.A.frame.rgb, pair.A.frame.depth
            rgbA_jpg, depthA_ds = chain.process(rgbA, depthA)
            hA, wA = depthA_ds.shape[:2] if depthA_ds is not None else (pair.A.frame.h, pair.A.frame.w)
            rgbA_jpg, depthA_ds = _normalize(rgbA_jpg, depthA_ds, wA, hA)

            # Cam B
            rgbB, depthB = pair.B.frame.rgb, pair.B.frame.depth
            rgbB_jpg, depthB_ds = chain.process(rgbB, depthB)
            hB, wB = depthB_ds.shape[:2] if depthB_ds is not None else (pair.B.frame.h, pair.B.frame.w)
            rgbB_jpg, depthB_ds = _normalize(rgbB_jpg, depthB_ds, wB, hB)

            if use_preview and preview is not None:
                quitA = preview.show(0, rgbA_jpg, depthA_ds, text=f"Cam 0  {wA}x{hA}", size=(wA, hA))
                quitB = preview.show(1, rgbB_jpg, depthB_ds, text=f"Cam 1  {wB}x{hB}   Δt={pair.dt_ms:.1f} ms", size=(wB, hB))
                if quitA or quitB:
                    break

            # Publish two frames on the same port
            pub.send_frame(
                cam_id=0, width=wA, height=hA,
                cfg_full=pair.A.intr, cfg_scale=cfg_scale, roi_off=roi_off,
                rgb_jpeg_bytes=rgbA_jpg, depth_u16=depthA_ds,
                pose_4x4=T_wc_A
            )
            pub.send_frame(
                cam_id=1, width=wB, height=hB,
                cfg_full=pair.B.intr, cfg_scale=cfg_scale, roi_off=roi_off,
                rgb_jpeg_bytes=rgbB_jpg, depth_u16=depthB_ds,
                pose_4x4=T_wc_B
            )

            frames += 2
            if frames % 60 == 0:
                dt = time.time() - t0
                print(f"[SENDER] {frames/dt:.1f} fps (2 cams) — A:{wA}x{hA} B:{wB}x{hB}")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] Runtime: {e}")

    finally:
        try:
            if preview: preview.close()
        except: pass
        try: pub.close()
        except: pass
        try: src.close()
        except: pass
        print("MultiStreamer2Cams: stopped.")

if __name__ == "__main__":
    main()
