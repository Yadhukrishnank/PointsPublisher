# streamer/utils/poses.py
from pathlib import Path
import numpy as np

def load_pose_4x4(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[Pose] Not found: {p}")
        return None

    if p.suffix.lower() == ".npz":
        data = np.load(p)
        for key in ("T_wc", "pose", "T", "matrix", "M"):
            if key in data and data[key].shape == (4, 4):
                return data[key].astype(np.float32)
        if "R" in data and "t" in data:
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = data["R"].reshape(3, 3)
            T[:3, 3] = data["t"].reshape(3)
            return T
        print(f"[Pose] No 4x4 key in {p.name} (expected T_wc/pose/T/matrix/M)")
        return None

    # Plain text / npy / csv
    try:
        T = np.loadtxt(p, dtype=np.float32).reshape(4, 4)
        return T
    except Exception:
        T = np.load(p).astype(np.float32).reshape(4, 4)
        return T
