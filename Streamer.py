import numpy as np
import streamer.Source as s
import streamer.ProcessingStep as p
import streamer.Actions as a
import streamer.Datasources as ds
import cv2

"""
In this example a InterneCamera is used as source. The Data is then send to the Meta Quest
"""


#Setup Source Camera example
# strategy = s.AzureKinectCameraStrategy(640, 480)
# camera = s.CameraContext(strategy)
# camera.init()
# intrinsics = camera.get_intrinsics()
# config = ds.CameraConfig(
#     fx=intrinsics["fx"],
#     fy=intrinsics["fy"],
#     cx=intrinsics["ppx"],
#     cy=intrinsics["ppy"]
# )

#--- Setup Source (RealSense if available, else Dummy) ---
try:
    strategy = s.RealSenseCameraStrategy(640, 480)
    camera = s.CameraContext(strategy)
    camera.init()
    intrinsics = camera.get_intrinsics()
    config = ds.CameraConfig(
        fx=intrinsics["fx"], fy=intrinsics["fy"],
        cx=intrinsics["ppx"], cy=intrinsics["ppy"]
    )
    orig_w, orig_h = intrinsics["width"], intrinsics["height"]
    print("[INFO] RealSense connected.")
except Exception as e:
    print("[WARN] No camera, using DummySource:", e)

    class DummySource(s.Source):
        def __init__(self, width=640, height=480):
            self.w, self.h = width, height
            self.t = 0
        def connect(self): pass
        def get_frame(self):
            # simple moving gradient RGB + sloped depth
            rgb = np.zeros((self.h, self.w, 3), np.uint8)
            rgb[..., 1] = (self.t % 255)
            rgb[..., 2] = np.linspace(0, 255, self.w, dtype=np.uint8)[None, :]
            depth = np.linspace(500, 2000, self.w*self.h, dtype=np.uint16).reshape(self.h, self.w)
            self.t += 1
            cfg = ds.CameraConfig(fx=591.4, fy=591.4, cx=self.w/2.0, cy=self.h/2.0)
            return rgb, depth, cfg
        def close(self): pass

    strategy = DummySource(640, 480)
    camera = s.CameraContext(strategy)
    camera.init()
    config = ds.CameraConfig(591.4, 591.4, 320.0, 240.0)
    orig_w, orig_h = 640, 480


# Setup Processing
processing = p.DownSampling(blocksize=3)
processing.set_next(p.EncodeRGBAsJPEG())


# Setup Aktionen

# Culling, for the rendering in the MetaQuest
culling = ds.Culling(
    zcullmin = 0.01,
    zcullmax = 2.0,
    x_cull = 1.0,
    y_cull = 1.0
)

actions = a.ActionPipeline()
a1 = a.ShowImageAction()
actions.add_action(a1)
a2 = a.ZMQPublishAction(culling, config_scaling=0.33) # Scaling because image is downsampled

# Not always the same as the read picture from the camera, because of the processingssteps
# At the moment must be set manually
a2.set_width_height(214, 160)
actions.add_action(a2)


# Run
try:
    while True:
        #rgb, depth = camera.get_frame() # If Camera is used
        rgb, depth, config = camera.get_frame() # if internet is used as source
        rgb, depth = processing.process(rgb, depth)
        a2.set_config(config)
        actions.execute_all(rgb, depth)
except KeyboardInterrupt:
    pass
finally:
    camera.close()
    cv2.destroyAllWindows()