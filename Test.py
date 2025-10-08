#!/usr/bin/env python3
import time
from collections import deque
import cv2
import numpy as np
import depthai as dai

# ========= YOUR DEVICES =========
MX_A = "184430102111900E00"
MX_B = "19443010B11CEF1200"

# ========= TUNABLES =========
# RGB width must be a multiple of 16 when aligning depth to RGB
RGB_W, RGB_H = 1280, 720
FPS = 30

# Stereo input resolution (Mono cameras). 400p is a safe, fast default.
STEREO_RES = dai.MonoCameraProperties.SensorResolution.THE_400_P

# Timestamp pairing tolerance (host time). Start wide, then tighten.
PAIR_TOL_MS = 50.0

# ========= TIMESTAMP PAIRER (HOST TIME) =========
class Pairer:
    def __init__(self, tol_ms=10.0, maxlen=120):
        self.tol = tol_ms / 1000.0
        self.A = deque(maxlen=maxlen)  # list[(ts, frame)]
        self.B = deque(maxlen=maxlen)
        self.last_nearest_ms = None

    @staticmethod
    def ts_host(frame) -> float:
        ts = frame.getTimestamp()  # host timestamp (shared clock)
        return ts.total_seconds() if hasattr(ts, "total_seconds") else float(ts)

    def _match(self, mine, other, f, a_is_new=True):
        ts = self.ts_host(f)
        best_i, best_dt = None, 1e9
        for i, (ts_o, _) in enumerate(other):
            dt = abs(ts - ts_o)
            if dt < best_dt:
                best_dt, best_i = dt, i
        # remember nearest distance for diagnostics
        self.last_nearest_ms = None if best_dt == 1e9 else best_dt * 1000.0

        mine.append((ts, f))
        if best_i is not None and best_dt <= self.tol:
            ts_o, f_o = other[best_i]
            del other[best_i]
            mine.pop()
            return (f, f_o, best_dt) if a_is_new else (f_o, f, best_dt)
        return None

    def pushA(self, f): return self._match(self.A, self.B, f, True)
    def pushB(self, f): return self._match(self.B, self.A, f, False)

# ========= PIPELINE (DepthAI 3 host I/O; MonoCamera for L/R) =========
def build_rgb_depth_pipeline(p: dai.Pipeline):
    """
    - Camera(CAM_A) -> RGB to host (BGR888p)
    - MonoCamera(CAM_B/C) -> StereoDepth -> Depth to host (uint16 mm)
    - Depth aligned to RGB (CAM_A), output size forced to 1280x720 (width /16)
    """
    # RGB camera (unified Camera node)
    camColor = p.create(dai.node.Camera)
    camColor.build(dai.CameraBoardSocket.CAM_A)
    outColor = camColor.requestOutput(
        size=(RGB_W, RGB_H),
        type=dai.ImgFrame.Type.BGR888p,
        resizeMode=dai.ImgResizeMode.STRETCH,  # enforce exact size
        fps=FPS
    )
    qColor = outColor.createOutputQueue()

    # Stereo inputs (Mono cameras for maximum compatibility)
    left  = p.create(dai.node.MonoCamera)
    right = p.create(dai.node.MonoCamera)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    left.setResolution(STEREO_RES);  right.setResolution(STEREO_RES)
    left.setFps(FPS);                right.setFps(FPS)

    stereo = p.create(dai.node.StereoDepth)
    left.out.link(stereo.left)
    right.out.link(stereo.right)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)

    # Align to RGB and force a valid output size (width divisible by 16)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(RGB_W, RGB_H)

    qDepth = stereo.depth.createOutputQueue()  # uint16 (mm), aligned to RGB

    return qColor, qDepth

def put_label(img, text, org=(8,24)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

# ========= MAIN =========
def main():
    print("DepthAI version:", dai.__version__)

    devA = dai.Device(dai.DeviceInfo(MX_A))
    devB = dai.Device(dai.DeviceInfo(MX_B))

    with dai.Pipeline(devA) as pA, dai.Pipeline(devB) as pB:
        qColorA, qDepthA = build_rgb_depth_pipeline(pA)
        qColorB, qDepthB = build_rgb_depth_pipeline(pB)
        pA.start(); pB.start()

        pairDepth = Pairer(PAIR_TOL_MS)
        lastLog = time.time()
        fps = {"rgbA":0,"rgbB":0,"dA":0,"dB":0}
        latestA = None  # latest unpaired depthA for fallback view
        latestB = None  # latest unpaired depthB for fallback view

        print("Running. Windows: A/B RGB and A/B Depth. Press 'q' to quit.")
        while pA.isRunning() and pB.isRunning():
            # RGB previews
            if qColorA.has():
                fA = qColorA.get(); fps["rgbA"] += 1
                cv2.imshow("A: RGB", fA.getCvFrame())
            if qColorB.has():
                fB = qColorB.get(); fps["rgbB"] += 1
                cv2.imshow("B: RGB", fB.getCvFrame())

            # Depth streams (pair by HOST time, show fallback if unpaired)
            if qDepthA.has():
                dA = qDepthA.get(); fps["dA"] += 1
                latestA = dA  # keep latest for fallback
                m = pairDepth.pushA(dA)
                if m:
                    dA_sync, dB_sync, dt = m
                    depthA = dA_sync.getFrame()
                    depthB = dB_sync.getFrame()
                    visA = cv2.applyColorMap(
                        cv2.normalize(depthA, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                        cv2.COLORMAP_PLASMA
                    )
                    visB = cv2.applyColorMap(
                        cv2.normalize(depthB, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                        cv2.COLORMAP_PLASMA
                    )
                    put_label(visA, f"PAIRED  Δt={dt*1000:.1f} ms")
                    put_label(visB, f"PAIRED  Δt={dt*1000:.1f} ms")
                    cv2.imshow("A: Depth", visA)
                    cv2.imshow("B: Depth", visB)
                else:
                    # show unpaired fallback for A if B hasn’t matched yet
                    if latestA is not None:
                        da = latestA.getFrame()
                        vis = cv2.applyColorMap(
                            cv2.normalize(da, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                            cv2.COLORMAP_PLASMA
                        )
                        put_label(vis, f"UNPAIRED  nearest≈{pairDepth.last_nearest_ms or -1:.1f} ms")
                        cv2.imshow("A: Depth", vis)

            if qDepthB.has():
                dB = qDepthB.get(); fps["dB"] += 1
                latestB = dB
                m = pairDepth.pushB(dB)
                if m:
                    dA_sync, dB_sync, dt = m
                    depthA = dA_sync.getFrame()
                    depthB = dB_sync.getFrame()
                    visA = cv2.applyColorMap(
                        cv2.normalize(depthA, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                        cv2.COLORMAP_PLASMA
                    )
                    visB = cv2.applyColorMap(
                        cv2.normalize(depthB, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                        cv2.COLORMAP_PLASMA
                    )
                    put_label(visA, f"PAIRED  Δt={dt*1000:.1f} ms")
                    put_label(visB, f"PAIRED  Δt={dt*1000:.1f} ms")
                    cv2.imshow("A: Depth", visA)
                    cv2.imshow("B: Depth", visB)
                else:
                    if latestB is not None:
                        db = latestB.getFrame()
                        vis = cv2.applyColorMap(
                            cv2.normalize(db, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                            cv2.COLORMAP_PLASMA
                        )
                        put_label(vis, f"UNPAIRED  nearest≈{pairDepth.last_nearest_ms or -1:.1f} ms")
                        cv2.imshow("B: Depth", vis)

            # simple FPS log
            now = time.time()
            if now - lastLog > 1.0:
                print(f"[FPS] A rgb/depth {fps['rgbA']}/{fps['dA']} | "
                      f"B rgb/depth {fps['rgbB']}/{fps['dB']}  | "
                      f"nearest Δt≈{pairDepth.last_nearest_ms if pairDepth.last_nearest_ms is not None else -1:.1f} ms")
                fps = {k:0 for k in fps}; lastLog = now

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


