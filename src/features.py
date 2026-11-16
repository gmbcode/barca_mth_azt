"""Stage 2: Multi-Modal Physics & Matrix Feature Engineering."""

import numpy as np


def extract_physics_feature_dict(stage1_data, st_priors):
    """Assemble the complete physics & multi-matrix feature dictionary over all 199,319 frames."""
    n = stage1_data["n_frames"]
    pose = stage1_data["pose"]
    cam_k = stage1_data["cam_kalman"]
    flow_jac = stage1_data["flow_jac"]
    atd = stage1_data["atd"]
    ms = stage1_data["ms"]
    reproj = stage1_data["reproj_error"]

    # Half-normalized attack direction: Home attacks +X in Half 1, -X in Half 2
    half_sign = np.ones(n, dtype=np.float64)
    half_sign[90652:] = -1.0
    cam_x_attack_h = np.where(half_sign > 0, pose["cam_x_norm"], 1.0 - pose["cam_x_norm"])
    vx_attack_h = cam_k["vx_smooth"] * half_sign

    # Rolling camera kinematics (2s = 50 frames, 5s = 125 frames)
    idx = np.arange(n)
    v_cam = cam_k["speed_smooth"]
    cum_v = np.concatenate(([0.0], np.cumsum(v_cam)))
    lo50, hi50 = np.maximum(0, idx - 25), np.minimum(n, idx + 26)
    lo125, hi125 = np.maximum(0, idx - 62), np.minimum(n, idx + 63)
    v_cam_mean_2s = (cum_v[hi50] - cum_v[lo50]) / np.maximum(hi50 - lo50, 1)
    v_cam_mean_5s = (cum_v[hi125] - cum_v[lo125]) / np.maximum(hi125 - lo125, 1)

    fov_x = pose["fov_x_m"]
    cum_fov = np.concatenate(([0.0], np.cumsum(fov_x)))
    fov_mean_2s = (cum_fov[hi50] - cum_fov[lo50]) / np.maximum(hi50 - lo50, 1)

    # Boundary proximity of camera focal center
    dist_goal_line_norm = np.minimum(pose["cam_x_norm"], 1.0 - pose["cam_x_norm"])
    dist_touchline_norm = np.minimum(pose["cam_y_norm"], 1.0 - pose["cam_y_norm"])

    # Tripod-lock indicator (camera stationary < 0.14 m/s)
    tripod_locked = (v_cam < 0.14)

    features = {
        "raw_codes": atd["raw_codes"],
        "cam_x_norm": pose["cam_x_norm"],
        "cam_y_norm": pose["cam_y_norm"],
        "cam_x_m": pose["cam_x_m"],
        "cam_y_m": pose["cam_y_m"],
        "cam_x_attack_h": cam_x_attack_h,
        "vx_attack_h": vx_attack_h,
        "fov_x_m": fov_x,
        "fov_y_m": pose["fov_y_m"],
        "dfov_x_ms": pose["dfov_x_ms"],
        "fov_mean_2s": fov_mean_2s,
        "v_cam": v_cam,
        "v_cam_mean_2s": v_cam_mean_2s,
        "v_cam_mean_5s": v_cam_mean_5s,
        "acc_cam": cam_k["acc_mag_smooth"],
        "turn_angle_cam": cam_k["turn_angle_rad"],
        "mahalanobis_cam": cam_k["mahalanobis_sq"],
        "frob_M": flow_jac["frob_M"],
        "det_J": flow_jac["det_J"],
        "cond_J": flow_jac["cond_J"],
        "reproj_error": reproj,
        "dist_goal_line_norm": dist_goal_line_norm,
        "dist_touchline_norm": dist_touchline_norm,
        "tripod_locked": tripod_locked,
        "E_matrix": ms["E_matrix"],
        "alive_home_share": ms["alive_home_share"],
        "duel_flip_rate_2s": ms["duel_flip_rate_2s"],
        "contention_entropy_2s": ms["contention_entropy_2s"],
        "run_length_same": ms["run_length_same"],
        "mean_speed_all": atd["mean_speed_all"],
        "p90_speed_all": atd["p90_speed_all"],
        "log_area_ratio": atd["log_area_ratio"],
        "st_net_log_odds": st_priors["st_net_log_odds"],
        "set_piece_pre_kick_d": st_priors["set_piece_pre_kick_d"],
        "set_piece_restart_team": st_priors["set_piece_restart_team"],
        "goal_dead_mask": st_priors["goal_dead_mask"],
    }
    return features
