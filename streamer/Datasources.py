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
    """Apply (optional) crop offset then scale (for downsample)."""
    return CameraConfig(
        fx = cfg.fx * scale,
        fy = cfg.fy * scale,
        cx = (cfg.cx - roi_off[0]) * scale,
        cy = (cfg.cy - roi_off[1]) * scale,
    )

def load_extrinsics_npz(path: str) -> np.ndarray:
    """
    Loads a 4x4 world_from_camera matrix (row-major) from .npz.
    Accepts keys: 'cam_to_world' or 'world_to_cam' (inverted) or (rot_vec, trans_vec).
    """
    f = np.load(path)
    if "cam_to_world" in f:
        M = np.array(f["cam_to_world"], dtype=np.float32)
    elif "world_to_cam" in f:
        M = np.linalg.inv(np.array(f["world_to_cam"], dtype=np.float32))
    else:
        r = f.get("rot_vec", None); t = f.get("trans_vec", None)
        if r is None or t is None:
            raise ValueError(f"Extrinsics in {path}: need cam_to_world/world_to_cam or rot_vec+trans_vec")
        R, _ = cv2.Rodrigues(r.astype(np.float32))
        M = np.eye(4, dtype=np.float32); M[:3,:3] = R; M[:3,3] = t.reshape(3).astype(np.float32)
    return M.astype(np.float32)
