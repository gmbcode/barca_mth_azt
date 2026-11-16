"""Stage 2: Multi-Modal Physics & ML Feature Engineering (88 Features per Frame).

Constructs both the dictionary of physical signals and the explicit 88-column tabular
feature matrix X_full in R^{199319 x 88}, partitioned into:
  - BALL_FEATURE_INDICES (42 features): Multi-scale centered, causal (past), and anti-causal
    (future) contact shares, forward-backward turnover gradients, log-odds, duel flip rates,
    contention entropies, run lengths, and gap boundary distances.
  - NOBALL_FEATURE_INDICES (46 features, 100% Ball-Independent): Homography camera pose,
    attack-normalized coordinates & velocities, multi-scale rolling speed stats (1s, 2s, 5s, 10s),
    FOV zoom & dilatation rates, tripod-lock features, optical-flow Frobenius norm ||M_t||_F,
    Spatial Jacobian (det J, trace J, cond J), reprojection error stats, collective player
    kinematics, Team Spatial Inertia ratio, Smart Tagging priors, and match clock features.
"""

import numpy as np


def _rolling_mean_std_min_max(arr, half_win):
    """Fast O(N) rolling mean and std using cumulative sums, plus strided min/max."""
    n = len(arr)
    idx = np.arange(n)
    lo = np.maximum(0, idx - half_win)
    hi = np.minimum(n, idx + half_win + 1)
    span = np.maximum(hi - lo, 1).astype(np.float64)

    c1 = np.concatenate(([0.0], np.cumsum(arr, dtype=np.float64)))
    c2 = np.concatenate(([0.0], np.cumsum(arr * arr, dtype=np.float64)))

    mean = (c1[hi] - c1[lo]) / span
    var = np.maximum(0.0, (c2[hi] - c2[lo]) / span - mean * mean)
    std = np.sqrt(var)
    return mean, std


def _one_sided_shares(raw_codes, windows=(8, 20, 40, 75, 150)):
    """Compute past-only [-w, 0] and future-only [0, +w] alive H shares and NaN shares."""
    n = len(raw_codes)
    idx = np.arange(n)
    is_h = (raw_codes == 0).astype(np.int64)
    is_a = (raw_codes == 1).astype(np.int64)
    is_d = (raw_codes == 2).astype(np.int64)

    cum_h = np.concatenate(([0], np.cumsum(is_h)))
    cum_a = np.concatenate(([0], np.cumsum(is_a)))
    cum_d = np.concatenate(([0], np.cumsum(is_d)))

    past_h_share = np.zeros((n, len(windows)), dtype=np.float64)
    fut_h_share = np.zeros((n, len(windows)), dtype=np.float64)
    grad_h_share = np.zeros((n, len(windows)), dtype=np.float64)
    past_nan_share = np.zeros((n, len(windows)), dtype=np.float64)
    fut_nan_share = np.zeros((n, len(windows)), dtype=np.float64)

    for k, w in enumerate(windows):
        # Past [t - w, t + 1)
        lo_p = np.maximum(0, idx - w)
        hi_p = idx + 1
        span_p = np.maximum(hi_p - lo_p, 1).astype(np.float64)
        hp = (cum_h[hi_p] - cum_h[lo_p]).astype(np.float64)
        ap = (cum_a[hi_p] - cum_a[lo_p]).astype(np.float64)
        dp = (cum_d[hi_p] - cum_d[lo_p]).astype(np.float64)
        tot_p = hp + ap
        sh_p = np.where(tot_p > 0, hp / np.maximum(tot_p, 1.0), 0.5)

        # Future [t, t + w + 1)
        lo_f = idx
        hi_f = np.minimum(n, idx + w + 1)
        span_f = np.maximum(hi_f - lo_f, 1).astype(np.float64)
        hf = (cum_h[hi_f] - cum_h[lo_f]).astype(np.float64)
        af = (cum_a[hi_f] - cum_a[lo_f]).astype(np.float64)
        df = (cum_d[hi_f] - cum_d[lo_f]).astype(np.float64)
        tot_f = hf + af
        sh_f = np.where(tot_f > 0, hf / np.maximum(tot_f, 1.0), 0.5)

        past_h_share[:, k] = sh_p
        fut_h_share[:, k] = sh_f
        grad_h_share[:, k] = sh_f - sh_p
        past_nan_share[:, k] = dp / span_p
        fut_nan_share[:, k] = df / span_f

    return past_h_share, fut_h_share, grad_h_share, past_nan_share, fut_nan_share


def extract_physics_feature_dict(stage1_data, st_priors):
    """Assemble the complete physics & ML feature dictionary + 88-column matrix X_full."""
    n = stage1_data["n_frames"]
    pose = stage1_data["pose"]
    cam_k = stage1_data["cam_kalman"]
    flow_jac = stage1_data["flow_jac"]
    atd = stage1_data["atd"]
    ms = stage1_data["ms"]
    reproj = stage1_data["reproj_error"]
    raw_codes = atd["raw_codes"]

    # Half-normalized attack direction: Home attacks +X in Half 1, -X in Half 2
    half_sign = np.ones(n, dtype=np.float64)
    half_sign[90652:] = -1.0
    cam_x_attack_h = np.where(half_sign > 0, pose["cam_x_norm"], 1.0 - pose["cam_x_norm"])
    vx_attack_h = cam_k["vx_smooth"] * half_sign

    # Multi-scale camera speed statistics (1s=12, 2s=25, 5s=62, 10s=125)
    v_cam = cam_k["speed_smooth"]
    v_cam_mean_1s, v_cam_std_1s = _rolling_mean_std_min_max(v_cam, 12)
    v_cam_mean_2s, v_cam_std_2s = _rolling_mean_std_min_max(v_cam, 25)
    v_cam_mean_5s, v_cam_std_5s = _rolling_mean_std_min_max(v_cam, 62)
    v_cam_mean_10s, v_cam_std_10s = _rolling_mean_std_min_max(v_cam, 125)

    # Multi-scale FOV statistics
    fov_x = pose["fov_x_m"]
    fov_y = pose["fov_y_m"]
    dfov_x = pose["dfov_x_ms"]
    fov_mean_2s, fov_std_2s = _rolling_mean_std_min_max(fov_x, 25)
    fov_mean_5s, fov_std_5s = _rolling_mean_std_min_max(fov_x, 62)
    dfov_mean_2s, _ = _rolling_mean_std_min_max(dfov_x, 25)
    dfov_mean_5s, _ = _rolling_mean_std_min_max(dfov_x, 62)

    # Reprojection error rolling stats
    reproj_mean_2s, reproj_std_2s = _rolling_mean_std_min_max(reproj, 25)

    # Boundary proximity of camera focal center
    dist_goal_line_norm = np.minimum(pose["cam_x_norm"], 1.0 - pose["cam_x_norm"])
    dist_touchline_norm = np.minimum(pose["cam_y_norm"], 1.0 - pose["cam_y_norm"])

    # Tripod-lock & wide-FOV indicators
    tripod_locked = v_cam < 0.14
    tripod_frac_2s, _ = _rolling_mean_std_min_max(tripod_locked.astype(np.float64), 25)
    tripod_frac_5s, _ = _rolling_mean_std_min_max(tripod_locked.astype(np.float64), 62)
    wide_fov_frac_5s, _ = _rolling_mean_std_min_max((fov_x > 75.5).astype(np.float64), 62)
    in_box_frac_2s, _ = _rolling_mean_std_min_max((dist_goal_line_norm < 0.14).astype(np.float64), 25)

    # One-sided (causal & anti-causal) multi-scale shares (5 scales: 8, 20, 40, 75, 150)
    past_h, fut_h, grad_h, past_nan, fut_nan = _one_sided_shares(
        raw_codes, windows=(8, 20, 40, 75, 150)
    )

    # Multi-scale duel flip rates (1s, 2s, 4s)
    flips = np.zeros(n, dtype=np.float64)
    flips[1:] = ((raw_codes[1:] != raw_codes[:-1]) &
                 (raw_codes[1:] < 2) & (raw_codes[:-1] < 2)).astype(np.float64)
    flip_rate_1s, _ = _rolling_mean_std_min_max(flips, 12)
    flip_rate_4s, _ = _rolling_mean_std_min_max(flips, 50)

    # Match clock / half progress in [0, 1]
    half_progress = np.zeros(n, dtype=np.float64)
    half_progress[11185:90652] = np.linspace(0.0, 1.0, 90652 - 11185)
    half_progress[110808:190198] = np.linspace(0.0, 1.0, 190198 - 110808)
    is_second_half = (np.arange(n) >= 100000).astype(np.float64)

    # -----------------------------------------------------------------------
    # Assemble 42 Ball-Dependent Features + 46 Ball-Independent Features = 88
    # -----------------------------------------------------------------------
    E_mat = ms["E_matrix"]  # (N, 5, 3)
    sh_c = ms["alive_home_share"]  # (N, 5)
    log_odds_c = np.log(np.clip(sh_c, 1e-3, 1.0 - 1e-3) / np.clip(1.0 - sh_c, 1e-3, 1.0 - 1e-3))

    ball_cols = [
        (raw_codes == 0).astype(np.float64),                 # 0: raw_is_H
        (raw_codes == 1).astype(np.float64),                 # 1: raw_is_A
        (raw_codes == 2).astype(np.float64),                 # 2: raw_is_NaN
        np.log1p(ms["run_length_same"].astype(np.float64)),  # 3: log_run_length_same
        flip_rate_1s,                                        # 4: duel_flip_rate_1s
        ms["duel_flip_rate_2s"],                             # 5: duel_flip_rate_2s
        flip_rate_4s,                                        # 6: duel_flip_rate_4s
        ms["contention_entropy_2s"],                         # 7: contention_entropy_2s
    ]
    # Add 5 scales of: centered H share (5), centered NaN share (5), centered log-odds (5),
    # past H share (5), future H share (5), future-past H gradient (5), past NaN (2), fut NaN (2)
    for k in range(5):
        ball_cols.append(sh_c[:, k])          # 8..12
        ball_cols.append(E_mat[:, k, 2])      # 13..17
        ball_cols.append(log_odds_c[:, k])    # 18..22
        ball_cols.append(past_h[:, k])        # 23..27
        ball_cols.append(fut_h[:, k])         # 28..32
        ball_cols.append(grad_h[:, k])        # 33..37
    ball_cols.extend([
        past_nan[:, 1],                       # 38: past_nan_20
        past_nan[:, 3],                       # 39: past_nan_75
        fut_nan[:, 1],                        # 40: fut_nan_20
        fut_nan[:, 3],                        # 41: fut_nan_75
    ])
    assert len(ball_cols) == 42

    noball_cols = [
        pose["cam_x_norm"],                                  # 42: cam_x_norm
        pose["cam_y_norm"],                                  # 43: cam_y_norm
        pose["cam_x_m"] / 105.3,                             # 44: cam_x_scaled
        pose["cam_y_m"] / 68.0,                              # 45: cam_y_scaled
        cam_x_attack_h,                                      # 46: cam_x_attack_h
        vx_attack_h / 10.0,                                  # 47: vx_attack_h_scaled
        cam_k["vx_smooth"] / 10.0,                           # 48: vx_cam_smooth
        cam_k["vy_smooth"] / 10.0,                           # 49: vy_cam_smooth
        v_cam / 10.0,                                        # 50: v_cam
        cam_k["acc_mag_smooth"] / 25.0,                      # 51: acc_cam
        cam_k["turn_angle_rad"] / np.pi,                     # 52: turn_angle_cam
        np.log1p(np.clip(cam_k["mahalanobis_sq"], 0, 100)),  # 53: log_mahalanobis_cam
        v_cam_mean_1s / 10.0,                                # 54: v_cam_mean_1s
        v_cam_std_1s / 10.0,                                 # 55: v_cam_std_1s
        v_cam_mean_2s / 10.0,                                # 56: v_cam_mean_2s
        v_cam_std_2s / 10.0,                                 # 57: v_cam_std_2s
        v_cam_mean_5s / 10.0,                                # 58: v_cam_mean_5s
        v_cam_std_5s / 10.0,                                 # 59: v_cam_std_5s
        v_cam_mean_10s / 10.0,                               # 60: v_cam_mean_10s
        v_cam_std_10s / 10.0,                                # 61: v_cam_std_10s
        (fov_x - 62.0) / 15.0,                               # 62: fov_x_norm
        (fov_y - 38.0) / 10.0,                               # 63: fov_y_norm
        fov_x / np.maximum(fov_y, 1.0),                      # 64: fov_aspect_ratio
        dfov_x / 5.0,                                        # 65: dfov_x_ms
        (fov_mean_2s - 62.0) / 15.0,                         # 66: fov_mean_2s
        fov_std_2s / 5.0,                                    # 67: fov_std_2s
        (fov_mean_5s - 62.0) / 15.0,                         # 68: fov_mean_5s
        fov_std_5s / 5.0,                                    # 69: fov_std_5s
        dfov_mean_2s / 5.0,                                  # 70: dfov_mean_2s
        dfov_mean_5s / 5.0,                                  # 71: dfov_mean_5s
        tripod_locked.astype(np.float64),                    # 72: tripod_locked
        tripod_frac_2s,                                      # 73: tripod_frac_2s
        tripod_frac_5s,                                      # 74: tripod_frac_5s
        wide_fov_frac_5s,                                    # 75: wide_fov_frac_5s
        in_box_frac_2s,                                      # 76: in_box_frac_2s
        dist_goal_line_norm,                                 # 77: dist_goal_line_norm
        dist_touchline_norm,                                 # 78: dist_touchline_norm
        np.log1p(np.clip(flow_jac["frob_M"], 0, 100)),       # 79: log_frob_M
        np.log1p(np.clip(flow_jac["det_J"], 0, 1e7)) / 15.0, # 80: log_det_J
        np.log1p(np.clip(flow_jac["cond_J"], 0, 1e4)) / 8.0, # 81: log_cond_J
        reproj,                                              # 82: reproj_error
        reproj_mean_2s,                                      # 83: reproj_mean_2s
        reproj_std_2s,                                       # 84: reproj_std_2s
        atd["mean_speed_all"] / 5.0,                         # 85: mean_speed_all
        st_priors["st_net_log_odds"],                        # 86: st_net_log_odds
        half_progress + is_second_half,                      # 87: match_clock_progress
    ]
    assert len(noball_cols) == 46

    X_full = np.column_stack(ball_cols + noball_cols).astype(np.float32)
    ball_feature_indices = np.arange(0, 42, dtype=np.int32)
    noball_feature_indices = np.arange(42, 88, dtype=np.int32)

    features = {
        "raw_codes": raw_codes,
        "cam_x_norm": pose["cam_x_norm"],
        "cam_y_norm": pose["cam_y_norm"],
        "cam_x_m": pose["cam_x_m"],
        "cam_y_m": pose["cam_y_m"],
        "cam_x_attack_h": cam_x_attack_h,
        "vx_attack_h": vx_attack_h,
        "fov_x_m": fov_x,
        "fov_y_m": fov_y,
        "dfov_x_ms": dfov_x,
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
        "tripod_frac_2s": tripod_frac_2s,
        "tripod_frac_5s": tripod_frac_5s,
        "wide_fov_frac_5s": wide_fov_frac_5s,
        "E_matrix": E_mat,
        "alive_home_share": sh_c,
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
        "X_full": X_full,
        "ball_feature_indices": ball_feature_indices,
        "noball_feature_indices": noball_feature_indices,
    }
    return features
