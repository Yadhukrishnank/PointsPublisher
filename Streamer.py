import numpy as np
import streamer.Source as s
import streamer.ProcessingStep as p
import streamer.Actions as a
import streamer.Datasources as ds
import cv2
import time
import logging

"""
Streams Azure Kinect color-aligned depth to Meta Quest:
Clamp → Median → ROI crop → Downsample → JPEG → Show + ZMQ publish
"""

# ---------- logging helpers ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

def human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ["KB", "MB", "GB", "TB"]:
        n /= 1024.0
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} PB"

# ---------- camera setup ----------
strategy = s.AzureKinectCameraStrategy(1280, 720)
camera = s.CameraContext(strategy)
camera.init()
intrinsics = camera.get_intrinsics()
config = ds.CameraConfig(
    fx=intrinsics["fx"],
    fy=intrinsics["fy"],
    cx=intrinsics["ppx"],
    cy=intrinsics["ppy"],
)
orig_w, orig_h = 1280, 720  # Azure color / transformed_depth

# ---------- processing chain ----------
# ROI in *full-res* pixel coords (before downsample)
roi_x0, roi_y0 = 160, 90
roi_w, roi_h = 640, 360

blocksize = 1

processing = p.DepthClampAndMask(z_min_m=0.25, z_max_m=1.0)
processing.set_next(p.LocalMedianReject(win=3, thr_mm=120)) \
          .set_next(p.CropROI(roi_x0, roi_y0, roi_w, roi_h)) \
          .set_next(p.DownSampling(blocksize=blocksize)) \
          .set_next(p.EncodeRGBAsJPEG())

print("[OpenCV] CUDA devices:", getattr(cv2, "cuda", None) and cv2.cuda.getCudaEnabledDeviceCount())

# ---------- actions / publisher ----------
culling = ds.Culling(
    zcullmin=0.05,
    zcullmax=1.0,
    x_cull=1.0,
    y_cull=1.0,
)

actions = a.ActionPipeline()
actions.add_action(a.ShowImageAction())

zmq_pub = a.ZMQPublishAction(culling, config_scaling=1.0)
actions.add_action(zmq_pub)

last_log = time.time()
frames_sec = 0
valid_sec = 0
after_cull_sec = 0

# cache for telemetry meshgrid to avoid per-frame realloc
_cached_shape = None
_cached_uu = None
_cached_vv = None

try:
    while True:
        # 1) Grab a frame
        rgb, depth, cam_cfg = camera.get_frame()

        # 2) Azure color can be BGRA; convert to BGR for JPEG safety
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR)

        # 3) Process: Clamp → Median → ROI → Downsample → JPEG
        rgb_proc, depth_proc = processing.process(rgb, depth)

        # 4) Packet header: processed size (W,H after ROI+DS)
        h_proc, w_proc = depth_proc.shape[:2]
        zmq_pub.set_width_height(w_proc, h_proc)

        # 5) Intrinsics: shift principal point by crop; scale by DS in publisher
        cam_cfg_cropped = ds.CameraConfig(
            fx=cam_cfg.fx,
            fy=cam_cfg.fy,
            cx=cam_cfg.cx - roi_x0,
            cy=cam_cfg.cy - roi_y0,
        )
        zmq_pub.config_scaling = (1.0 / float(blocksize)) if blocksize > 1 else 1.0
        zmq_pub.set_config(cam_cfg_cropped)

        # 6) Send to actions (local preview + ZMQ)
        actions.execute_all(rgb_proc, depth_proc)

        # ---------- diagnostics: processing + estimated wire sizes ----------
        # rgb_proc is a JPEG byte buffer (np.ndarray), depth_proc is uint16 image
        rgb_len = int(rgb_proc.nbytes) if hasattr(rgb_proc, "nbytes") else len(bytes(rgb_proc))
        depth_len = int(depth_proc.size * depth_proc.dtype.itemsize)  # W*H*2

        logging.info(
            f"[PROC] ROI=({roi_x0},{roi_y0},{roi_w}x{roi_h})  DS={blocksize}  OUT={w_proc}x{h_proc}  "
            f"JPEG={human_bytes(rgb_len)}  DEPTH={human_bytes(depth_len)}"
        )

        # 7) Telemetry (mirrors GPU math) — uses CROPPED+SCALED intrinsics
        z = depth_proc.astype(np.float32) * 0.001  # meters
        valid_mask = z > 0.0

        fx_s = cam_cfg_cropped.fx * zmq_pub.config_scaling
        fy_s = cam_cfg_cropped.fy * zmq_pub.config_scaling
        cx_s = cam_cfg_cropped.cx * zmq_pub.config_scaling
        cy_s = cam_cfg_cropped.cy * zmq_pub.config_scaling

        if _cached_shape != (h_proc, w_proc):
            _cached_shape = (h_proc, w_proc)
            _cached_uu, _cached_vv = np.meshgrid(
                np.arange(w_proc, dtype=np.float32),
                np.arange(h_proc, dtype=np.float32),
            )
        uu, vv = _cached_uu, _cached_vv

        X = (uu - cx_s) * z / fx_s
        Y = (vv - cy_s) * z / fy_s

        c = culling
        cull_mask = (
            valid_mask
            & (z >= c.zcullmin)
            & (z <= c.zcullmax)
            & (np.abs(X) <= c.x_cull)
            & (np.abs(Y) <= c.y_cull)
        )

        frames_sec += 1
        valid_sec += int(valid_mask.sum())
        after_cull_sec += int(cull_mask.sum())

        now = time.time()
        if now - last_log >= 1.0:
            total_pts = w_proc * h_proc
            fps_send = frames_sec / (now - last_log)
            logging.info(
                f"[SENDER] fps={fps_send:.1f}  frame={w_proc}x{h_proc} ({total_pts})  "
                f"valid/s={valid_sec}  est_after_cull/s={after_cull_sec}  "
                f"est_size/frame≈{human_bytes(rgb_len + depth_len)}"
            )
            last_log = now
            frames_sec = valid_sec = after_cull_sec = 0

except KeyboardInterrupt:
    pass
finally:
    camera.close()
    cv2.destroyAllWindows()
