"""Stage 1C: Cross-Camera Broadcast Synchronization Matrix (Procrustes / Optical Axis Matching)."""

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, asset_paths  # noqa: E402
from src.physics_matrices import compute_homography_pose_matrices  # noqa: E402


def synchronize_broadcast_to_tactical(stage1_data, game=GAME, data_dir=None):
    """Align valid broadcast homography segments to the tactical timeline via camera-pose matching."""
    n_tac = stage1_data["n_frames"]
    b_paths = asset_paths(game, data_dir, source="broadcast")
    b_hom_path = b_paths.get("homography", "")

    if not b_hom_path or not os.path.isfile(b_hom_path):
        return {
            "available": False,
            "mapped_broadcast_frame": np.full(n_tac, -1, dtype=np.int32),
            "broadcast_valid_on_tac": np.zeros(n_tac, dtype=bool),
        }

    b_df = pd.read_csv(b_hom_path)
    mat_cols = [f"h{r}{c}" for r in range(3) for c in range(3)]
    H_b = b_df[mat_cols].values.astype(np.float64).reshape(-1, 3, 3)
    valid_b = b_df["valid"].values.astype(bool) & (b_df["reprojection_error"].values < 3.0)
    pitch = stage1_data["pitch"]
    fps = stage1_data["fps"]

    pose_b = compute_homography_pose_matrices(H_b, valid_b, pitch[0], pitch[1], fps)

    # Match continuous valid broadcast shots to tactical camera look-point trajectory
    tac_cx = stage1_data["pose"]["cam_x_m"]
    tac_cy = stage1_data["pose"]["cam_y_m"]
    b_cx = pose_b["cam_x_m"]
    b_cy = pose_b["cam_y_m"]

    mapped = np.full(n_tac, -1, dtype=np.int32)
    b_valid_tac = np.zeros(n_tac, dtype=bool)

    # Coarse linear scaling + local sliding-window alignment per valid broadcast stretch
    edges = np.diff(np.concatenate(([0], valid_b.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    scale_ratio = float(n_tac) / float(max(len(H_b), 1))
    for s_b, e_b in zip(starts, ends):
        length = e_b - s_b
        if length < 50:
            continue
        est_tac_start = int(round(s_b * scale_ratio))
        search_lo = max(0, est_tac_start - 1500)
        search_hi = min(n_tac - length, est_tac_start + 1500)
        if search_hi <= search_lo:
            continue

        best_offset = est_tac_start
        best_err = np.inf
        step = 10
        sub_bx = b_cx[s_b:e_b:step]
        sub_by = b_cy[s_b:e_b:step]
        for cand in range(search_lo, search_hi, 25):
            sub_tx = tac_cx[cand:cand + length:step]
            sub_ty = tac_cy[cand:cand + length:step]
            if len(sub_tx) == len(sub_bx):
                err = float(np.mean((sub_tx - sub_bx) ** 2 + (sub_ty - sub_by) ** 2))
                if err < best_err:
                    best_err = err
                    best_offset = cand

        if best_err < 120.0:
            tac_slice = slice(best_offset, min(n_tac, best_offset + length))
            actual_len = tac_slice.stop - tac_slice.start
            mapped[tac_slice] = np.arange(s_b, s_b + actual_len, dtype=np.int32)
            b_valid_tac[tac_slice] = True

    return {
        "available": True,
        "mapped_broadcast_frame": mapped,
        "broadcast_valid_on_tac": b_valid_tac,
    }
