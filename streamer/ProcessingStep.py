# ProcessingStep.py
import numpy as np
import cv2

class ProcessingStep:
    def __init__(self):
        self._next = None

    def set_next(self, nxt):
        self._next = nxt
        return nxt

    def process(self, rgb, depth_u16):
        out = (rgb, depth_u16)
        return self._next.process(*out) if self._next else out

class DepthClampAndMask(ProcessingStep):
    def __init__(self, z_min_m=0.25, z_max_m=4.0):
        super().__init__()
        self.min_mm = int(max(0.0, z_min_m) * 1000.0 + 0.5)
        self.max_mm = int(max(0.0, z_max_m) * 1000.0 + 0.5)

    def process(self, rgb, depth):
        if depth is not None and depth.size:
            depth = depth.copy()
            mask = (depth < self.min_mm) | (depth > self.max_mm)
            depth[mask] = 0
        return super().process(rgb, depth)

class LocalMedianReject(ProcessingStep):
    """Rejects outliers vs local median by threshold in mm."""
    def __init__(self, win=3, thr_mm=120):
        super().__init__()
        self.win = int(win) if int(win) % 2 == 1 else int(win) + 1
        self.thr = int(thr_mm)

    def process(self, rgb, depth):
        if depth is not None and depth.size:
            med = cv2.medianBlur(depth, self.win)
            bad = (depth > 0) & (np.abs(depth.astype(np.int32) - med.astype(np.int32)) > self.thr)
            depth = depth.copy()
            depth[bad] = 0
        return super().process(rgb, depth)

class CropROI(ProcessingStep):
    def __init__(self, x0, y0, w, h):
        super().__init__()
        self.x0, self.y0, self.w, self.h = map(int, (x0, y0, w, h))

    def process(self, rgb, depth):
        if rgb is None and depth is None:
            return super().process(rgb, depth)
        H = depth.shape[0] if depth is not None else rgb.shape[0]
        W = depth.shape[1] if depth is not None else rgb.shape[1]
        x1 = max(0, min(self.x0 + self.w, W))
        y1 = max(0, min(self.y0 + self.h, H))
        x0 = max(0, min(self.x0, x1))
        y0 = max(0, min(self.y0, y1))
        rgb2   = rgb[y0:y1, x0:x1, :] if rgb is not None else None
        depth2 = depth[y0:y1, x0:x1]  if depth is not None else None
        return super().process(rgb2, depth2)

class DownSampling(ProcessingStep):
    """Block min-pool for depth (preserves nearest), avg pool for RGB."""
    def __init__(self, block=2):
        super().__init__()
        self.b = int(max(1, block))

    def _down_rgb(self, rgb):
        if self.b == 1: return rgb
        H, W = rgb.shape[:2]
        h2, w2 = (H // self.b) * self.b, (W // self.b) * self.b
        if h2 == 0 or w2 == 0: return rgb
        r = rgb[:h2, :w2].reshape(h2//self.b, self.b, w2//self.b, self.b, 3)
        return r.mean(axis=(1, 3)).astype(np.uint8, copy=False)

    def _down_depth(self, depth):
        if self.b == 1: return depth
        H, W = depth.shape[:2]
        h2, w2 = (H // self.b) * self.b, (W // self.b) * self.b
        if h2 == 0 or w2 == 0: return depth
        d = depth[:h2, :w2].reshape(h2//self.b, self.b, w2//self.b, self.b)
        return d.min(axis=(1, 3)).astype(np.uint16, copy=False)

    def process(self, rgb, depth):
        if rgb is not None:  rgb  = self._down_rgb(rgb)
        if depth is not None: depth = self._down_depth(depth)
        return super().process(rgb, depth)

class EncodeRGBAsJPEG(ProcessingStep):
    def __init__(self, quality=80):
        super().__init__()
        self.quality = int(np.clip(quality, 10, 100))

    def process(self, rgb, depth):
        if rgb is None:
            rgb_bytes = b""
        else:
            ok, buf = cv2.imencode('.jpg', rgb, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            rgb_bytes = buf.tobytes() if ok else b""
        depth_u16 = depth.astype(np.uint16, copy=False) if depth is not None else None
        return super().process(rgb_bytes, depth_u16)
