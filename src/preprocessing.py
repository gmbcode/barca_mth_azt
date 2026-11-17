"""Stage 1: Data Loading, Metric Scaling, Kalman-RTS Filtering & Matrix Extraction."""

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, asset_paths, half_video_blocks, source_frames  # noqa: E402
from src.physics_matrices import (  # noqa: E402
    clean_atd_trajectories,
    compute_anisotropic_pitch_control,
    compute_carry_and_shooting_matrices,
    compute_homography_pose_matrices,
    compute_multiscale_contention_matrix,
    compute_optical_flow_and_jacobian_matrices,
    compute_player_interaction_matrices,
    compute_press_and_pinning_matrices,
    compute_team_spatial_inertia_tensors,
    kalman_rts_smoother_2d,
)

# Constants from Proposed_Plan.md Section 3.1
SECOND_HALF_PRE_KICKOFF_START = 110808
SECOND_HALF_KICKOFF_FRAME = 113742
FIRST_HALF_KICKOFF_FRAME = 11185


def parse_atd_metadata(meta_path):
    """Extract team_of_track (0=Home, 1=Away) and goalkeeper slot indices from ATD XML."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0]
    slots = []
    gk_slots = []
    for idx, player in enumerate(root.iter("Player")):
        slots.append(0 if player.get("teamId") == home_id else 1)
        for val in player.iter("Value"):
            if val.text and "goalkeeper" in val.text.strip().lower():
                gk_slots.append(idx)
    if not gk_slots:
        gk_slots = [23, 45]
    return np.array(slots, dtype=np.int8), tuple(gk_slots)


def stream_parse_atd_raw(
    raw_path,
    team_of_track,
    n_frames,
    pitch=(105.3, 68.0),
    gk_slots=(23, 45),
    half_split_frame=90652,
    fps=25.0,
):
    """Stream-parse BAR-ATH_tactical_atd.txt, clean GK swaps, and compute collective kinematics."""
    length, width = pitch
    n_slots = len(team_of_track)

    ball_x_m = np.full(n_frames, np.nan, dtype=np.float64)
    ball_y_m = np.full(n_frames, np.nan, dtype=np.float64)
    raw_codes = np.full(n_frames, 2, dtype=np.int8)  # 0=H, 1=A, 2=NaN

    d_home_1 = np.full(n_frames, np.inf, dtype=np.float64)
    d_home_2 = np.full(n_frames, np.inf, dtype=np.float64)
    d_away_1 = np.full(n_frames, np.inf, dtype=np.float64)
    d_away_2 = np.full(n_frames, np.inf, dtype=np.float64)
    nearest_slot = np.full(n_frames, -1, dtype=np.int16)
    px_all = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py_all = np.full((n_frames, n_slots), np.nan, dtype=np.float32)

    mean_speed_all = np.full(n_frames, np.nan, dtype=np.float64)
    p90_speed_all = np.full(n_frames, np.nan, dtype=np.float64)
    mean_speed_h = np.full(n_frames, np.nan, dtype=np.float64)
    mean_speed_a = np.full(n_frames, np.nan, dtype=np.float64)
    centroid_hx = np.full(n_frames, 52.65, dtype=np.float64)
    centroid_hy = np.full(n_frames, 34.0, dtype=np.float64)
    centroid_ax = np.full(n_frames, 52.65, dtype=np.float64)
    centroid_ay = np.full(n_frames, 34.0, dtype=np.float64)
    area_h = np.full(n_frames, 150.0, dtype=np.float64)
    area_a = np.full(n_frames, 150.0, dtype=np.float64)

    home_mask = (team_of_track == 0)
    away_mask = (team_of_track == 1)

    with open(raw_path, "r") as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            f_idx = int(parts[0]) - 1
            if f_idx < 0 or f_idx >= n_frames:
                continue

            # Parse ball
            b_str = parts[2]
            b_seen = (b_str[0] != "N")
            if b_seen:
                c1 = b_str.index(",")
                c2 = b_str.index(",", c1 + 1)
                bx = float(b_str[:c1]) * length
                by = float(b_str[c1 + 1:c2]) * width
                ball_x_m[f_idx] = bx
                ball_y_m[f_idx] = by

            # Parse players
            px = np.full(n_slots, np.nan, dtype=np.float64)
            py = np.full(n_slots, np.nan, dtype=np.float64)
            ps = np.full(n_slots, np.nan, dtype=np.float64)
            for s_idx, grp in enumerate(parts[1].split(";")):
                if s_idx >= n_slots or grp[0] == "N":
                    continue
                c1 = grp.index(",")
                c2 = grp.index(",", c1 + 1)
                px[s_idx] = float(grp[:c1]) * length
                py[s_idx] = float(grp[c1 + 1:c2]) * width
                s_val = grp[c2 + 1:]
                if s_val and s_val[0] != "N":
                    ps[s_idx] = float(s_val)

            px_all[f_idx] = px
            py_all[f_idx] = py
            valid_s = np.isfinite(ps)
            if np.any(valid_s):
                mean_speed_all[f_idx] = float(np.mean(ps[valid_s]))
                p90_speed_all[f_idx] = float(np.percentile(ps[valid_s], 90))
            if np.any(valid_s & home_mask):
                mean_speed_h[f_idx] = float(np.mean(ps[valid_s & home_mask]))
            if np.any(valid_s & away_mask):
                mean_speed_a[f_idx] = float(np.mean(ps[valid_s & away_mask]))

            valid_p = np.isfinite(px) & np.isfinite(py)
            vh = valid_p & home_mask
            va = valid_p & away_mask
            if np.sum(vh) >= 3:
                chx = float(np.mean(px[vh]))
                chy = float(np.mean(py[vh]))
                centroid_hx[f_idx] = chx
                centroid_hy[f_idx] = chy
                dx_h, dy_h = px[vh] - chx, py[vh] - chy
                det_h = np.var(dx_h) * np.var(dy_h) - (np.mean(dx_h * dy_h)) ** 2
                area_h[f_idx] = float(np.sqrt(max(det_h, 1.0)))
            if np.sum(va) >= 3:
                cax = float(np.mean(px[va]))
                cay = float(np.mean(py[va]))
                centroid_ax[f_idx] = cax
                centroid_ay[f_idx] = cay
                dx_a, dy_a = px[va] - cax, py[va] - cay
                det_a = np.var(dx_a) * np.var(dy_a) - (np.mean(dx_a * dy_a)) ** 2
                area_a[f_idx] = float(np.sqrt(max(det_a, 1.0)))

    # Clean ATD wrong-end goalkeeper ID swaps and compute velocity with teleportation rejection
    px64, py64, vx64, vy64 = clean_atd_trajectories(
        px_all.astype(np.float64),
        py_all.astype(np.float64),
        team_of_track,
        gk_slots=gk_slots,
        half_split_frame=half_split_frame,
        pitch_length=length,
        fps=fps,
    )

    ball_seen_mask = np.isfinite(ball_x_m) & np.isfinite(ball_y_m)
    chunk_sz = 4096
    for s0 in range(0, n_frames, chunk_sz):
        s1 = min(n_frames, s0 + chunk_sz)
        b_vis_c = ball_seen_mask[s0:s1]
        if not np.any(b_vis_c):
            continue
        px_c = px64[s0:s1]
        py_c = py64[s0:s1]
        vp_c = np.isfinite(px_c) & np.isfinite(py_c)
        vh_c = vp_c & home_mask[None, :]
        va_c = vp_c & away_mask[None, :]
        d_c = np.hypot(px_c - ball_x_m[s0:s1, None], py_c - ball_y_m[s0:s1, None])
        vd_c = np.where(vp_c, d_c, np.inf)
        dh_c = np.sort(np.where(vh_c, d_c, np.inf), axis=1)
        da_c = np.sort(np.where(va_c, d_c, np.inf), axis=1)
        has_p = b_vis_c & np.any(vp_c, axis=1)

        ns_c = np.where(has_p, np.argmin(vd_c, axis=1), -1).astype(np.int16)
        dh1_c = dh_c[:, 0]
        dh2_c = dh_c[:, 1] if dh_c.shape[1] > 1 else np.full(s1 - s0, np.inf)
        da1_c = da_c[:, 0]
        da2_c = da_c[:, 1] if da_c.shape[1] > 1 else np.full(s1 - s0, np.inf)
        rc_c = np.where(has_p, np.where(dh1_c <= da1_c, 0, 1), 2).astype(np.int8)

        nearest_slot[s0:s1] = ns_c
        d_home_1[s0:s1] = np.where(has_p, dh1_c, np.inf)
        d_home_2[s0:s1] = np.where(has_p, dh2_c, np.inf)
        d_away_1[s0:s1] = np.where(has_p, da1_c, np.inf)
        d_away_2[s0:s1] = np.where(has_p, da2_c, np.inf)
        raw_codes[s0:s1] = rc_c

    # Unsupervised algorithmic player ball-retention propensity per track slot (zero human labels):
    # fraction of frames a track keeps team possession continuity +20f after initial touch
    slot_retention = np.full(n_slots, 0.5, dtype=np.float64)
    valid_slots = (nearest_slot >= 0) & (raw_codes < 2)
    if np.any(valid_slots):
        fut_idx = np.minimum(np.arange(n_frames) + 20, n_frames - 1)
        retained = valid_slots & (raw_codes[fut_idx] == raw_codes)
        for s_idx in range(n_slots):
            ms = valid_slots & (nearest_slot == s_idx)
            cnt = int(np.sum(ms))
            if cnt >= 25:
                slot_retention[s_idx] = (float(np.sum(retained & ms)) + 5.0) / (float(cnt) + 10.0)
    track_retention_prior = np.where(nearest_slot >= 0, slot_retention[np.maximum(nearest_slot, 0)], 0.5)

    return {
        "has_full_atd": True,
        "ball_x_m": ball_x_m,
        "ball_y_m": ball_y_m,
        "raw_codes": raw_codes,
        "nearest_slot": nearest_slot,
        "track_retention_prior": track_retention_prior,
        "d_home_1": d_home_1,
        "d_home_2": d_home_2,
        "d_away_1": d_away_1,
        "d_away_2": d_away_2,
        "mean_speed_all": pd.Series(mean_speed_all).interpolate(limit_direction="both").values,
        "p90_speed_all": pd.Series(p90_speed_all).interpolate(limit_direction="both").values,
        "mean_speed_h": pd.Series(mean_speed_h).interpolate(limit_direction="both").values,
        "mean_speed_a": pd.Series(mean_speed_a).interpolate(limit_direction="both").values,
        "centroid_hx": centroid_hx,
        "centroid_hy": centroid_hy,
        "centroid_ax": centroid_ax,
        "centroid_ay": centroid_ay,
        "area_h": area_h,
        "area_a": area_a,
        "log_area_ratio": np.log(np.maximum(area_h, 1.0) / np.maximum(area_a, 1.0)),
        "px_all": px64,
        "py_all": py64,
        "vx_all": vx64,
        "vy_all": vy64,
    }


def load_stage1_data(game=GAME, data_dir=None):
    """Load and compute Stage 1 preprocessed trajectories and all core physics matrices."""
    paths = asset_paths(game, data_dir, source="tactical")
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    fps = float(GAMES[game]["fps"])

    # 1. Load Homography CSV (Matrix 1 & Matrix 2)
    hom_path = paths["homography"]
    if not os.path.isfile(hom_path):
        alt_hom = "/Users/shashwatagrawal/Downloads/BAR-ATH_tactical_homography.csv"
        if os.path.isfile(alt_hom):
            hom_path = alt_hom
        else:
            raise FileNotFoundError(f"Homography file not found at {hom_path}")

    hom_df = pd.read_csv(hom_path)
    mat_cols = [f"h{r}{c}" for r in range(3) for c in range(3)]
    H = hom_df[mat_cols].values.astype(np.float64).reshape(-1, 3, 3)
    reproj_error = hom_df["reprojection_error"].values.astype(np.float64)
    valid_hom = hom_df["valid"].values.astype(bool)

    pose = compute_homography_pose_matrices(H, valid_hom, pitch[0], pitch[1], fps)

    # Apply Matrix 3 (6D Kalman-RTS Smoother) to the camera optical look-point trajectory
    cam_kalman = kalman_rts_smoother_2d(
        pose["cam_x_m"],
        pose["cam_y_m"],
        valid_hom,
        reproj_error=reproj_error,
        dt=1.0 / fps,
        drag_gamma=0.94,
        q_acc=4.5,
        r_base=0.12,
        max_jump_ms=35.0,
    )

    flow_jac = compute_optical_flow_and_jacobian_matrices(
        H, pose["H_inv"], pose["cam_x_norm"], pose["cam_y_norm"]
    )

    half_blocks = half_video_blocks(game)
    half_split_frame = half_blocks[1][1] if len(half_blocks) > 1 else 90652

    # 2. Load ATD raw + XML if present, else fallback to baseline_raw.csv contact stream
    if os.path.isfile(paths["atd_meta"]) and os.path.isfile(paths["atd_raw"]):
        team_of_track, gk_slots = parse_atd_metadata(paths["atd_meta"])
        atd_data = stream_parse_atd_raw(
            paths["atd_raw"],
            team_of_track,
            n_frames,
            pitch,
            gk_slots=gk_slots,
            half_split_frame=half_split_frame,
            fps=fps,
        )
        ball_seen = np.isfinite(atd_data["ball_x_m"]) & np.isfinite(atd_data["ball_y_m"])
        ball_kalman = kalman_rts_smoother_2d(
            np.nan_to_num(atd_data["ball_x_m"], nan=pose["cam_x_m"]),
            np.nan_to_num(atd_data["ball_y_m"], nan=pose["cam_y_m"]),
            ball_seen,
            reproj_error=reproj_error,
            dt=1.0 / fps,
            drag_gamma=0.90,
            q_acc=15.0,
            r_base=0.15,
            max_jump_ms=38.0,
        )
        atd_data["ball_kalman"] = ball_kalman

        # Matrix 5 (Shape Tensors), Matrix 9 (Pressing + Pinning), & Matrix 10 (Carry/TTI/Shooting)
        px64 = atd_data.pop("px_all").astype(np.float64)
        py64 = atd_data.pop("py_all").astype(np.float64)
        vx64 = atd_data.pop("vx_all").astype(np.float64)
        vy64 = atd_data.pop("vy_all").astype(np.float64)
        inertia = compute_team_spatial_inertia_tensors(px64, py64, team_of_track, gk_slots=gk_slots)
        press_pin = compute_press_and_pinning_matrices(
            px64,
            py64,
            vx64,
            vy64,
            team_of_track,
            ball_kalman["x_smooth"],
            ball_kalman["y_smooth"],
            pitch_length=pitch[0],
            half_split_frame=half_split_frame,
            gk_slots=gk_slots,
        )
        carry_shot = compute_carry_and_shooting_matrices(
            px64,
            py64,
            vx64,
            vy64,
            team_of_track,
            atd_data["ball_x_m"],
            atd_data["ball_y_m"],
            ball_kalman["x_smooth"],
            ball_kalman["y_smooth"],
            ball_kalman["vx_smooth"],
            ball_kalman["vy_smooth"],
            ball_kalman["speed_smooth"],
            cam_kalman["x_smooth"],
            cam_kalman["y_smooth"],
            pitch_length=pitch[0],
            pitch_width=pitch[1],
            half_split_frame=half_split_frame,
            gk_slots=gk_slots,
        )
        atd_data.update({
            "log_area_ratio": inertia["log_area_ratio"],
            "log_width_ratio": np.log(np.maximum(inertia["eig2_h"], 0.5) / np.maximum(inertia["eig2_a"], 0.5)),
            "log_depth_ratio": np.log(np.maximum(inertia["eig1_h"], 0.5) / np.maximum(inertia["eig1_a"], 0.5)),
            "eig1_h": inertia["eig1_h"],
            "eig2_h": inertia["eig2_h"],
            "eig1_a": inertia["eig1_a"],
            "eig2_a": inertia["eig2_a"],
            "low_line_h": inertia["low_line_h"],
            "high_line_h": inertia["high_line_h"],
            "low_line_a": inertia["low_line_a"],
            "high_line_a": inertia["high_line_a"],
            **press_pin,
            **carry_shot,
        })
    else:
        raw_csv_path = os.path.join(ROOT, "evaluation", "results", "baseline_raw.csv")
        raw_df = pd.read_csv(raw_csv_path)
        lbl = raw_df["label"].values
        raw_codes = np.where(lbl == "H", 0, np.where(lbl == "A", 1, 2)).astype(np.int8)
        atd_data = {
            "has_full_atd": False,
            "raw_codes": raw_codes,
            "ball_x_m": cam_kalman["x_smooth"],
            "ball_y_m": cam_kalman["y_smooth"],
            "ball_kalman": cam_kalman,
            "mean_speed_all": np.clip(0.8 * cam_kalman["speed_smooth"] + 1.5, 0.8, 6.0),
            "p90_speed_all": np.clip(1.4 * cam_kalman["speed_smooth"] + 2.5, 1.2, 9.0),
            "mean_speed_h": np.clip(0.8 * cam_kalman["speed_smooth"] + 1.5, 0.8, 6.0),
            "mean_speed_a": np.clip(0.8 * cam_kalman["speed_smooth"] + 1.5, 0.8, 6.0),
            "d_home_1": np.full(n_frames, np.inf, dtype=np.float64),
            "d_home_2": np.full(n_frames, np.inf, dtype=np.float64),
            "d_away_1": np.full(n_frames, np.inf, dtype=np.float64),
            "d_away_2": np.full(n_frames, np.inf, dtype=np.float64),
            "centroid_hx": cam_kalman["x_smooth"],
            "centroid_hy": cam_kalman["y_smooth"],
            "centroid_ax": cam_kalman["x_smooth"],
            "centroid_ay": cam_kalman["y_smooth"],
            "area_h": np.full(n_frames, 150.0, dtype=np.float64),
            "area_a": np.full(n_frames, 150.0, dtype=np.float64),
            "log_area_ratio": np.zeros(n_frames, dtype=np.float64),
            "log_width_ratio": np.zeros(n_frames, dtype=np.float64),
            "log_depth_ratio": np.zeros(n_frames, dtype=np.float64),
            "eig1_h": np.full(n_frames, 150.0, dtype=np.float64),
            "eig2_h": np.full(n_frames, 100.0, dtype=np.float64),
            "eig1_a": np.full(n_frames, 150.0, dtype=np.float64),
            "eig2_a": np.full(n_frames, 100.0, dtype=np.float64),
            "low_line_h": cam_kalman["x_smooth"],
            "high_line_h": cam_kalman["x_smooth"],
            "low_line_a": cam_kalman["x_smooth"],
            "high_line_a": cam_kalman["x_smooth"],
            "pin_mean_a": np.zeros(n_frames, dtype=np.float64),
            "pin_mean_h": np.zeros(n_frames, dtype=np.float64),
            "press_cap_on_h": np.zeros(n_frames, dtype=np.float64),
            "press_cap_on_a": np.zeros(n_frames, dtype=np.float64),
            "bait_safety_h": np.full(n_frames, 0.5, dtype=np.float64),
            "bait_safety_a": np.full(n_frames, 0.5, dtype=np.float64),
            "n_free_outlets_h": np.zeros(n_frames, dtype=np.float64),
            "n_free_outlets_a": np.zeros(n_frames, dtype=np.float64),
            "net_press_hazard_h": np.zeros(n_frames, dtype=np.float64),
            "tti_margin_ha": np.zeros(n_frames, dtype=np.float64),
            "knock_on_h": np.zeros(n_frames, dtype=bool),
            "knock_on_a": np.zeros(n_frames, dtype=bool),
            "open_carry_h": np.zeros(n_frames, dtype=np.float64),
            "open_carry_a": np.zeros(n_frames, dtype=np.float64),
            "space_ahead_h": np.full(n_frames, 12.0, dtype=np.float64),
            "space_ahead_a": np.full(n_frames, 12.0, dtype=np.float64),
            "d_goal_h": np.full(n_frames, 52.65, dtype=np.float64),
            "d_goal_a": np.full(n_frames, 52.65, dtype=np.float64),
            "n_def_cone_h": np.zeros(n_frames, dtype=np.float64),
            "n_def_cone_a": np.zeros(n_frames, dtype=np.float64),
            "shot_takeon_open_h": np.zeros(n_frames, dtype=np.float64),
            "shot_takeon_open_a": np.zeros(n_frames, dtype=np.float64),
            "shot_crowded_risk_h": np.zeros(n_frames, dtype=np.float64),
            "shot_crowded_risk_a": np.zeros(n_frames, dtype=np.float64),
            "shot_flight": np.zeros(n_frames, dtype=bool),
            "box_occ_h": np.zeros(n_frames, dtype=np.float64),
            "box_occ_a": np.zeros(n_frames, dtype=np.float64),
            "att_third_occ_diff": np.zeros(n_frames, dtype=np.float64),
            "pc_ball_log_odds": np.zeros(n_frames, dtype=np.float64),
            "pc_cam_log_odds": np.zeros(n_frames, dtype=np.float64),
            "nearest_slot": np.full(n_frames, -1, dtype=np.int16),
            "track_retention_prior": np.full(n_frames, 0.5, dtype=np.float64),
        }

    # 3. Compute Matrix 7: Multi-Scale Temporal Evidence & Contention Entropy Matrix (E_t)
    ms_contention = compute_multiscale_contention_matrix(
        atd_data["raw_codes"], half_windows=(8, 20, 40, 75, 150)
    )

    return {
        "n_frames": n_frames,
        "pitch": pitch,
        "fps": fps,
        "half_blocks": half_video_blocks(game),
        "reproj_error": reproj_error,
        "valid_hom": valid_hom,
        "pose": pose,
        "cam_kalman": cam_kalman,
        "flow_jac": flow_jac,
        "atd": atd_data,
        "ms": ms_contention,
    }
