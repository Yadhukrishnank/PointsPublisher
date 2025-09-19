from abc import ABC, abstractmethod
import numpy as np
from skimage.measure import block_reduce
import cv2



class ProcessingStep(ABC):
    def __init__(self, next_step=None):
        self.next_step = next_step

    def set_next(self, next_step):
        self.next_step = next_step
        return next_step  # Für einfaches Verkettung

    def process(self, rgb_frame, depth_frame):
        # Prozessiere den aktuellen Schritt
        rgb_out, depth_out = self._process(rgb_frame, depth_frame)

        # Übergib weiter, falls noch weitere Steps
        if self.next_step:
            return self.next_step.process(rgb_out, depth_out)
        else:
            return rgb_out, depth_out

    @abstractmethod
    def _process(self, rgb_frame, depth_frame):
        """Implementiere die Verarbeitung des aktuellen Schritts"""
        pass

class DepthClampAndMask(ProcessingStep):
    """
    Clamp depth to a valid metric range and set everything else to 0 (invalid).
    Assumes incoming depth is uint16 in millimeters (Azure/RealSense z16).
    Do this BEFORE any denoise/downsample.
    """
    def __init__(self, z_min_m=0.25, z_max_m=3.5):
        super().__init__()
        # store as integers in millimeters to avoid float work per frame
        self.min_mm = int(max(0, round(z_min_m * 1000.0)))
        self.max_mm = int(max(0, round(z_max_m * 1000.0)))

    def _process(self, rgb_frame, depth_frame):
        if depth_frame is None:
            return rgb_frame, depth_frame

        # If depth came as float meters by accident, convert to uint16 mm first
        if np.issubdtype(depth_frame.dtype, np.floating):
            depth_mm = (np.clip(depth_frame, 0.0, 65.535) * 1000.0).astype(np.uint16)
        else:
            depth_mm = depth_frame

        # Keep original invalids (zeros) as zeros
        valid = depth_mm > 0

        # Clamp: anything outside [min_mm, max_mm] becomes invalid (0)
        out = depth_mm.copy()
        # below min or above max → 0
        out[valid & (out < self.min_mm)] = 0
        out[valid & (out > self.max_mm)] = 0

        return rgb_frame, out
    





class LocalMedianReject(ProcessingStep):
    """
    Suppress isolated depth spikes by comparing to a local median.
    Assumes depth is uint16 in millimeters (0 = invalid).
    win: odd kernel size (3,5,7), thr_mm: reject if |d - median| > threshold.
    """
    def __init__(self, win=5, thr_mm=60):
        super().__init__()
        self.win = int(win if win % 2 == 1 else win + 1)
        self.thr = int(thr_mm)

    def _process(self, rgb, depth_u16):
        if depth_u16 is None:
            return rgb, depth_u16
        # Median of local neighborhood
        med = cv2.medianBlur(depth_u16, self.win)
        # Absolute difference
        diff = cv2.absdiff(depth_u16, med)
        out = depth_u16.copy()
        # Only invalidate valid pixels that deviate too much from local median
        valid = depth_u16 > 0
        out[valid & (diff > self.thr)] = 0
        return rgb, out


# --- add to ProcessingStep.py ---

class CropROI(ProcessingStep):
    """
    Crop a rectangular ROI in pixel coordinates: (x0, y0, w, h).
    Works on both RGB (H×W×C) and depth (H×W). 2d cropping only
    """
    def __init__(self, x0: int, y0: int, w: int, h: int):
        super().__init__()
        self.x0 = int(max(0, x0))
        self.y0 = int(max(0, y0))
        self.w  = int(max(1, w))
        self.h  = int(max(1, h))

    # IMPORTANT: override _process, not process
    def _process(self, rgb, depth):
        # infer frame size from whichever is present
        if depth is not None:
            H, W = depth.shape[:2]
        elif rgb is not None:
            H, W = rgb.shape[:2]
        else:
            return rgb, depth

        # clamp ROI to bounds
        x1 = min(self.x0 + self.w, W)
        y1 = min(self.y0 + self.h, H)
        x0 = min(self.x0, x1 - 1)
        y0 = min(self.y0, y1 - 1)

        # crop
        if rgb is not None:
            rgb = rgb[y0:y1, x0:x1].copy()
        if depth is not None:
            depth = depth[y0:y1, x0:x1].copy()

        # return cropped pair; the base class will pass it to next step
        return rgb, depth



class EncodeRGBAsJPEG(ProcessingStep):
    def _process(self, rgb_frame, depth_frame):
        # make sure it's 8-bit
        if rgb_frame.dtype != np.uint8:
            rgb_frame = rgb_frame.astype(np.uint8)
        # higher quality
        ret, rgb_buf = cv2.imencode('.jpg', rgb_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return rgb_buf, depth_frame



class DownSampling(ProcessingStep):
    def __init__(self, blocksize):
        super().__init__()
        self.block_size = blocksize

    def _process(self, rgb_frame, depth_frame):
        depth = self.downsample(depth_frame, mode='min')
        rgb = self.downsample(rgb_frame, mode='avg')

        return rgb, depth

    def downsample(self, img, mode='avg'):
        func = np.mean if mode == 'avg' else np.min

        if img.ndim == 3:  # RGB oder multi-channel
            # block_reduce erwartet tuple mit Länge ndim, hier z.B. (block_size, block_size, 1)
            down = block_reduce(img, block_size=(self.block_size, self.block_size, 1), func=func)
        else:
            down = block_reduce(img, block_size=(self.block_size, self.block_size), func=func)

        return down
