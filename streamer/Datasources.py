# Datasources.py
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class CameraConfig:
    fx: float
    fy: float
    cx: float
    cy: float

@dataclass
class Culling:
    zcullmin: float = 0.05
    zcullmax: float = 4.0
    x_cull:   float = 1.0
    y_cull:   float = 1.0

def scale_intrinsics(cfg: CameraConfig, scale: float, roi_off=(0.0, 0.0)) -> CameraConfig:
    """
    Adjust intrinsics after ROI and downsample.
    roi_off: (x0, y0) in original pixels; scale: 1/downsample_block
    """
    return CameraConfig(
        fx=float(cfg.fx * scale),
        fy=float(cfg.fy * scale),
        cx=float((cfg.cx - roi_off[0]) * scale),
        cy=float((cfg.cy - roi_off[1]) * scale),
    )

def load_extrinsics_npz(path: str) -> np.ndarray:
    """
    Load a 4x4 row-major world_from_camera matrix from .npz. Accepts:
      - 'cam_to_world' (4x4)
      - 'world_to_cam' (4x4) -> inverted
      - 'rot_vec' (3,) + 'trans_vec'(3,) -> Rodrigues + t
    """
    f = np.load(path)
    if "cam_to_world" in f:
        M = np.array(f["cam_to_world"], dtype=np.float32)
    elif "world_to_cam" in f:
        M = np.linalg.inv(np.array(f["world_to_cam"], dtype=np.float32))
    else:
        r = f.get("rot_vec", None)
        t = f.get("trans_vec", None)
        if r is None or t is None:
            raise ValueError(
                f"Extrinsics {path}: expected 'cam_to_world' or 'world_to_cam' "
                f"or ('rot_vec','trans_vec')"
            )
        R, _ = cv2.Rodrigues(r.astype(np.float32))
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = R.astype(np.float32)
        M[:3,  3] = t.reshape(3).astype(np.float32)

    if M.shape != (4, 4):
        raise ValueError(f"Extrinsics {path}: got {M.shape}, expected 4x4")
    return M.astype(np.float32)
