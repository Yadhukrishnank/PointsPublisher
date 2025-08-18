import numpy as np
import streamer.Source as s
import streamer.ProcessingStep as p
import streamer.Actions as a
import streamer.Datasources as ds
import cv2
import time

"""
In this example a InterneCamera is used as source. The Data is then send to the Meta Quest
"""


#Setup Source Camera example
strategy = s.AzureKinectCameraStrategy(1280, 720)
camera = s.CameraContext(strategy)
camera.init()
intrinsics = camera.get_intrinsics()
config = ds.CameraConfig(
    fx=intrinsics["fx"],
    fy=intrinsics["fy"],
    cx=intrinsics["ppx"],
    cy=intrinsics["ppy"]
)
orig_w, orig_h = 1280, 720   # matches Azure color / transformed_depth

# --- Processing chain ---

processing = p.DepthClampAndMask(z_min_m=0.25, z_max_m=1)
processing.set_next(p.LocalMedianReject(win=5, thr_mm=60))
processing.set_next(p.DownSampling(blocksize=3))
processing.set_next(p.EncodeRGBAsJPEG()) 

print("[OpenCV] CUDA devices:", getattr(cv2, "cuda", None) and cv2.cuda.getCudaEnabledDeviceCount())


# Setup Aktionen

# Culling, for the rendering in the MetaQuest
culling = ds.Culling(
    zcullmin = 0.05,
    zcullmax = 1.0,
    x_cull = 1.0,
    y_cull = 1.0
)

actions = a.ActionPipeline()
# a1 = a.ShowImageAction()
actions.add_action(a.ShowImageAction())

zmq_pub = a.ZMQPublishAction(culling, config_scaling=1.0)


# a2 = a.ZMQPublishAction(culling, config_scaling=0.33) # Scaling because image is downsampled

# Not always the same as the read picture from the camera, because of the processingssteps
# At the moment must be set manually
# a2.set_width_height(426, 240)
actions.add_action(zmq_pub)

last_log = time.time()
frames_sec = 0
valid_sec = 0
after_cull_sec = 0

# Run
try:
    while True:
        rgb, depth, cam_cfg = camera.get_frame()

        #rgb, depth = camera.get_frame() # If Camera is used
        # rgb, depth, config = camera.get_frame() # if internet is used as source
        # rgb, depth = processing.process(rgb, depth)
        # a2.set_config(config)
        # actions.execute_all(rgb, depth)


 # Azure color can be BGRA; convert to BGR for JPEG safety
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR)

        # Process (downsample + JPEG)
        rgb_proc, depth_proc = processing.process(rgb, depth)

        # Use processed size for the packet header
        h_proc, w_proc = depth_proc.shape[:2]
        zmq_pub.set_width_height(w_proc, h_proc)

        # Accurate intrinsics scaling (uniform because blocksize is square)
        scale_x = w_proc / float(orig_w)
        zmq_pub.config_scaling = float(scale_x)

        # Send ORIGINAL intrinsics; the action scales them internally
        zmq_pub.set_config(cam_cfg)

        # (Optional) sanity print to see what you’re sending
        # print(f"[TX] {w_proc}x{h_proc}, scale={scale_x:.6f}")

        actions.execute_all(rgb_proc, depth_proc)

        # --- compute sender-side density (mirrors compute shader math) ---
        # depth in meters
        z = (depth_proc.astype(np.float32) * 0.001)

        # valid depth mask
        valid_mask = z > 0.0

        # scaled intrinsics (same as what you send)
        fx_s = cam_cfg.fx * zmq_pub.config_scaling
        fy_s = cam_cfg.fy * zmq_pub.config_scaling
        cx_s = cam_cfg.cx * zmq_pub.config_scaling
        cy_s = cam_cfg.cy * zmq_pub.config_scaling

        h_proc, w_proc = depth_proc.shape[:2]
        uu, vv = np.meshgrid(np.arange(w_proc, dtype=np.float32),
                            np.arange(h_proc, dtype=np.float32))
        X = (uu - cx_s) * z / fx_s
        Y = (vv - cy_s) * z / fy_s

        c = culling
        cull_mask = (valid_mask &
                    (z >= c.zcullmin) & (z <= c.zcullmax) &
                    (np.abs(X) <= c.x_cull) & (np.abs(Y) <= c.y_cull))

        frames_sec += 1
        valid_sec += int(valid_mask.sum())
        after_cull_sec += int(cull_mask.sum())

        now = time.time()
        if now - last_log >= 1.0:
            total_pts = w_proc * h_proc
            fps_send = frames_sec / (now - last_log)
            print(f"[SENDER] fps={fps_send:.1f}  frame={w_proc}x{h_proc} ({total_pts})  "
                f"valid/s={valid_sec}  est_after_cull/s={after_cull_sec}")
            last_log = now
            frames_sec = valid_sec = after_cull_sec = 0



except KeyboardInterrupt:
    pass
finally:
    camera.close()
    cv2.destroyAllWindows()