# ProcessingStep.py
import numpy as np
import cv2

class ProcessingStep:
    def __init__(self): self._next = None
    def set_next(self, nxt):
        self._next = nxt
        return nxt
    def process(self, rgb, depth_u16):
        """Override in subclasses. Must return (rgb, depth_u16)."""
        out = (rgb, depth_u16)
        return self._next.process(*out) if self._next else out

class DepthClampAndMask(ProcessingStep):
    def __init__(self, z_min_m=0.25, z_max_m=4.0):
        super().__init__()
        self.min_mm = int(z_min_m * 1000.0)
        self.max_mm = int(z_max_m * 1000.0)
    def process(self, rgb, depth):
        depth = depth.copy()
        mask = (depth < self.min_mm) | (depth > self.max_mm)
        depth[mask] = 0
        return super().process(rgb, depth)

class LocalMedianReject(ProcessingStep):
    """Reject outliers vs local median by threshold in mm."""
    def __init__(self, win=3, thr_mm=120):
        super().__init__()
        self.win = int(max(3, win) | 1)  # odd
        self.thr = int(thr_mm)
    def process(self, rgb, depth):
        if depth.size == 0: return super().process(rgb, depth)
        med = cv2.medianBlur(depth, self.win)
        bad = (depth > 0) & (np.abs(depth.astype(np.int32) - med.astype(np.int32)) > self.thr)
        depth = depth.copy(); depth[bad] = 0
        return super().process(rgb, depth)

class CropROI(ProcessingStep):
    def __init__(self, x0, y0, w, h):
        super().__init__()
        self.x0, self.y0, self.w, self.h = int(x0), int(y0), int(w), int(h)
    def process(self, rgb, depth):
        x0,y0,w,h = self.x0, self.y0, self.w, self.h
        rgb   = rgb[y0:y0+h, x0:x0+w].copy()
        depth = depth[y0:y0+h, x0:x0+w].copy()
        return super().process(rgb, depth)

class DownSampling(ProcessingStep):
    def __init__(self, blocksize=2):
        super().__init__(); self.b = int(max(1, blocksize))
    def process(self, rgb, depth):
        if self.b == 1: return super().process(rgb, depth)
        # RGB: area resize
        h,w = rgb.shape[:2]
        rgb = cv2.resize(rgb, (w//self.b, h//self.b), interpolation=cv2.INTER_AREA)
        # Depth: min-pooling (keeps nearest surface)
        H,W = depth.shape[:2]
        h2, w2 = (H//self.b)*self.b, (W//self.b)*self.b
        d = depth[:h2, :w2].reshape(h2//self.b, self.b, w2//self.b, self.b)
        depth_ds = d.min(axis=(1,3)).astype(np.uint16, copy=False)
        return super().process(rgb, depth_ds)

class EncodeRGBAsJPEG(ProcessingStep):
    def __init__(self, quality=80):
        super().__init__(); self.quality = int(np.clip(quality, 10, 100))
    def process(self, rgb, depth):
        ok, buf = cv2.imencode('.jpg', rgb, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok: buf = np.array([], dtype=np.uint8)
        return super().process(buf.tobytes(), depth.astype(np.uint16, copy=False))
