"""Eight Core Physics & Probabilistic Matrices for Frame-Level Possession Inference.

Implements the mathematical matrix formulations extending Proposed_Plan.md:
  1. HomographyPoseMatrix (H_t, H_t^{-1} in R^{3x3}): Optical axis & FOV frustum kinematics.
  2. OpticalFlowJacobianMatrix (M_t in R^{3x3}, J_t in R^{2x2}): Projective flow & spatial magnification.
  3. KalmanRTSStateSpaceMatrices (F, H_obs, Q, R_t, K_t, G_t in R^{6x6}): 6D kinematic filter & RTS smoother.
  4. PlayerInteractionAdjacencyMatrix (D_t, V_t, C_t in R^{46x46}): Pairwise distances & pass-corridor shadowing.
  5. TeamSpatialInertiaTensor (Sigma_H, Sigma_A in R^{2x2}): Eigenvalues, area det(Sigma), tactical expansion ratio.
  6. AnisotropicPitchControlTensor (S_i(t) = R(theta) Lambda(s) R(theta)^T in R^{2x2}): Velocity-weighted control field.
  7. MultiScaleContentionMatrix (E_t in R^{N x K x 3}): Multi-horizon kernel evidence & duel contention entropy.
  8. TimeInhomogeneousHSMMMatrix (A_t in R^{N x 3 x 3}): Frame-dynamic Markov transition matrix with regime gates.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Matrix 1: Projective Homography & Inverse Camera-Pose Matrix (H_t, H_t^{-1})
# ---------------------------------------------------------------------------

def unproject_points(H_inv, pixel_u, pixel_v):
    """Map image pixel coordinates (u, v) -> normalized pitch coordinates (0..1) via H_t^{-1}."""
    den = H_inv[:, 2, 0] * pixel_u + H_inv[:, 2, 1] * pixel_v + H_inv[:, 2, 2]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    x = (H_inv[:, 0, 0] * pixel_u + H_inv[:, 0, 1] * pixel_v + H_inv[:, 0, 2]) / den
    y = (H_inv[:, 1, 0] * pixel_u + H_inv[:, 1, 1] * pixel_v + H_inv[:, 1, 2]) / den
    return x, y


def compute_homography_pose_matrices(H, valid, pitch_length=105.3, pitch_width=68.0, fps=25.0):
    """Compute H_t^{-1} and extract camera optical look-point, FOV frustum, and pan/zoom kinematics.

    Parameters
    ----------
    H : ndarray of shape (N, 3, 3)
        Per-frame 3x3 homography matrices mapping normalized pitch (0..1) -> pixels (1920x1080).
    valid : ndarray of shape (N,) bool
        Validity indicator for each frame's homography matrix.

    Returns
    -------
    pose : dict of ndarrays
        Contains H_inv (N,3,3), cam_x_norm, cam_y_norm, cam_x_m, cam_y_m,
        fov_x_m, fov_y_m, dfov_x_ms, vx_cam, vy_cam, v_cam, acc_cam, frustum_corners (N,4,2).
    """
    n_frames = len(H)
    safe_H = np.where(valid[:, None, None], H, np.eye(3)[None, :, :])
    H_inv = np.linalg.inv(safe_H)

    # Principal point (960, 540) unprojected to pitch coordinates
    cam_x_norm, cam_y_norm = unproject_points(H_inv, 960.0, 540.0)
    cam_x_m = cam_x_norm * pitch_length
    cam_y_m = cam_y_norm * pitch_width

    # Horizontal & vertical Field-of-View span in meters across the principal axes
    xl, yl = unproject_points(H_inv, 0.0, 540.0)
    xr, yr = unproject_points(H_inv, 1920.0, 540.0)
    xt, yt = unproject_points(H_inv, 960.0, 0.0)
    xb, yb = unproject_points(H_inv, 960.0, 1080.0)

    fov_x_m = np.abs(xr - xl) * pitch_length
    fov_y_m = np.abs(yb - yt) * pitch_width

    # 4-corner frustum polygon on normalized pitch
    c_tl = np.stack(unproject_points(H_inv, 0.0, 0.0), axis=-1)
    c_tr = np.stack(unproject_points(H_inv, 1920.0, 0.0), axis=-1)
    c_br = np.stack(unproject_points(H_inv, 1920.0, 1080.0), axis=-1)
    c_bl = np.stack(unproject_points(H_inv, 0.0, 1080.0), axis=-1)
    frustum_corners = np.stack([c_tl, c_tr, c_br, c_bl], axis=1)  # (N, 4, 2)

    # Camera pan velocity, speed, acceleration, and zoom dilatation rate
    vx_cam = np.gradient(cam_x_m) * fps
    vy_cam = np.gradient(cam_y_m) * fps
    v_cam = np.hypot(vx_cam, vy_cam)
    ax_cam = np.gradient(vx_cam) * fps
    ay_cam = np.gradient(vy_cam) * fps
    acc_cam = np.hypot(ax_cam, ay_cam)
    dfov_x_ms = np.gradient(fov_x_m) * fps

    return {
        "H_inv": H_inv,
        "cam_x_norm": cam_x_norm,
        "cam_y_norm": cam_y_norm,
        "cam_x_m": cam_x_m,
        "cam_y_m": cam_y_m,
        "fov_x_m": fov_x_m,
        "fov_y_m": fov_y_m,
        "dfov_x_ms": dfov_x_ms,
        "vx_cam": vx_cam,
        "vy_cam": vy_cam,
        "v_cam": v_cam,
        "acc_cam": acc_cam,
        "frustum_corners": frustum_corners,
    }


# ---------------------------------------------------------------------------
# Matrix 2: Inter-Frame Homography Optical-Flow (M_t) & Spatial Jacobian (J_t)
# ---------------------------------------------------------------------------

def compute_optical_flow_and_jacobian_matrices(H, H_inv, eval_x_norm, eval_y_norm):
    """Compute M_t = H_t H_{t-1}^{-1} - I_3 and the 2x2 Spatial Jacobian J_t(x,y).

    Parameters
    ----------
    H : ndarray (N, 3, 3)
    H_inv : ndarray (N, 3, 3)
    eval_x_norm, eval_y_norm : ndarray (N,)
        Normalized pitch coordinates at which to evaluate the 2x2 Jacobian J_t.

    Returns
    -------
    flow_jac : dict of ndarrays
        frob_M (N,), det_J (N,), trace_J (N,), cond_J (N,), J (N, 2, 2).
    """
    n_frames = len(H)
    # Inter-frame projective flow M_t = H_t @ H_{t-1}^{-1} - I_3
    H_prev_inv = np.empty_like(H_inv)
    H_prev_inv[0] = H_inv[0]
    H_prev_inv[1:] = H_inv[:-1]
    M_t = np.matmul(H, H_prev_inv)
    # Normalize homogeneous scale so M_t[:, 2, 2] == 1 before subtracting I_3
    scale = np.where(np.abs(M_t[:, 2, 2]) < 1e-12, 1.0, M_t[:, 2, 2])[:, None, None]
    M_t = M_t / scale - np.eye(3)[None, :, :]
    frob_M = np.linalg.norm(M_t.reshape(n_frames, -1), axis=1)

    # Spatial Jacobian J_t(x, y) = d(u, v) / d(x, y) in R^{N x 2 x 2}
    x = eval_x_norm
    y = eval_y_norm
    u_num = H[:, 0, 0] * x + H[:, 0, 1] * y + H[:, 0, 2]
    v_num = H[:, 1, 0] * x + H[:, 1, 1] * y + H[:, 1, 2]
    w = H[:, 2, 0] * x + H[:, 2, 1] * y + H[:, 2, 2]
    w_sq = np.maximum(w * w, 1e-12)

    j00 = (H[:, 0, 0] * w - H[:, 2, 0] * u_num) / w_sq
    j01 = (H[:, 0, 1] * w - H[:, 2, 1] * u_num) / w_sq
    j10 = (H[:, 1, 0] * w - H[:, 2, 0] * v_num) / w_sq
    j11 = (H[:, 1, 1] * w - H[:, 2, 1] * v_num) / w_sq

    J = np.zeros((n_frames, 2, 2), dtype=np.float64)
    J[:, 0, 0] = j00
    J[:, 0, 1] = j01
    J[:, 1, 0] = j10
    J[:, 1, 1] = j11

    det_J = np.abs(j00 * j11 - j01 * j10)
    trace_J = j00 * j00 + j01 * j01 + j10 * j10 + j11 * j11
    # Singular values of 2x2 matrix from trace(J^T J) and det(J)
    disc = np.sqrt(np.maximum(0.0, trace_J * trace_J - 4.0 * det_J * det_J))
    s_max = np.sqrt(np.maximum(1e-12, 0.5 * (trace_J + disc)))
    s_min = np.sqrt(np.maximum(1e-12, 0.5 * (trace_J - disc)))
    cond_J = s_max / s_min

    return {
        "frob_M": frob_M,
        "det_J": det_J,
        "trace_J": trace_J,
        "cond_J": cond_J,
        "J": J,
    }


# ---------------------------------------------------------------------------
# Matrix 3: 6D Constant-Acceleration Kalman Filter & RTS Backward Smoother
# ---------------------------------------------------------------------------

def kalman_rts_smoother_2d(pos_x_m, pos_y_m, valid_obs, reproj_error=None,
                           dt=0.04, drag_gamma=0.90, q_acc=12.0, r_base=0.18,
                           max_jump_ms=38.0):
    """6D Constant-Acceleration Kalman Filter + RTS Smoother for 2D trajectories.

    State vector at frame t:
        x_t = [X, Y, v_x, v_y, a_x, a_y]^T in R^6
    Because X and Y have identical decoupled state-space structure (3D per axis:
    [pos, vel, acc]^T), we factor the 6x6 system into two parallel 3x3 blocks
    for exact, fast O(N) forward-backward filtering with Mahalanobis outlier gating.

    Returns
    -------
    result : dict of ndarrays
        x_smooth, y_smooth, vx_smooth, vy_smooth, ax_smooth, ay_smooth,
        speed_smooth, acc_mag_smooth, turn_angle_rad, mahalanobis_sq, teleport_mask.
    """
    n = len(pos_x_m)
    if reproj_error is None:
        reproj_error = np.full(n, 0.25, dtype=np.float64)

    # First pass: reject single-frame optical teleports (> max_jump_ms)
    valid = valid_obs.copy() & np.isfinite(pos_x_m) & np.isfinite(pos_y_m)
    teleport_mask = np.zeros(n, dtype=bool)
    obs_idx = np.flatnonzero(valid)
    if len(obs_idx) >= 3:
        dx_prev = pos_x_m[obs_idx[1:-1]] - pos_x_m[obs_idx[:-2]]
        dy_prev = pos_y_m[obs_idx[1:-1]] - pos_y_m[obs_idx[:-2]]
        dt_prev = (obs_idx[1:-1] - obs_idx[:-2]) * dt
        v_prev = np.hypot(dx_prev, dy_prev) / np.maximum(dt_prev, dt)

        dx_next = pos_x_m[obs_idx[2:]] - pos_x_m[obs_idx[1:-1]]
        dy_next = pos_y_m[obs_idx[2:]] - pos_y_m[obs_idx[1:-1]]
        dt_next = (obs_idx[2:] - obs_idx[1:-1]) * dt
        v_next = np.hypot(dx_next, dy_next) / np.maximum(dt_next, dt)

        spikes = (v_prev > max_jump_ms) & (v_next > max_jump_ms)
        bad_frames = obs_idx[1:-1][spikes]
        teleport_mask[bad_frames] = True
        valid[bad_frames] = False

    # 3x3 per-axis transition matrix F_1d for [p, v, a]^T:
    F = np.array([
        [1.0, dt, 0.5 * dt * dt],
        [0.0, 1.0, dt],
        [0.0, 0.0, drag_gamma],
    ], dtype=np.float64)
    F_T = F.T

    # Process noise covariance Q_1d (piecewise white jerk)
    q = q_acc * q_acc
    Q = q * np.array([
        [0.25 * dt**4, 0.5 * dt**3, 0.5 * dt**2],
        [0.5 * dt**3, dt**2, dt],
        [0.5 * dt**2, dt, 1.0],
    ], dtype=np.float64)

    # Solve the 3x3 Discrete Algebraic Riccati Equation (DARE) for steady-state P_pred, P_filt, K, G
    r_med = (r_base * (1.0 + float(np.median(np.clip(reproj_error, 0.0, 5.0))))) ** 2
    P_f = np.diag([1.0, 10.0, 25.0])
    for _ in range(50):
        P_p = F @ P_f @ F_T + Q
        s_v = P_p[0, 0] + r_med
        K_ss = P_p[:, 0] / s_v
        P_f = P_p - np.outer(K_ss, P_p[0, :])
    G_ss = P_f @ F_T @ np.linalg.inv(P_p + 1e-9 * np.eye(3))
    A_closed = F - np.outer(K_ss, F[0, :])
    s_inv = 1.0 / (P_p[0, 0] + r_med)

    def _filter_1d(z_obs):
        x_filt = np.zeros((n, 3), dtype=np.float64)
        x_pred_arr = np.zeros((n, 3), dtype=np.float64)
        innov_sq = np.zeros(n, dtype=np.float64)

        first_valid = np.flatnonzero(valid)
        x_curr = np.array([z_obs[first_valid[0]] if len(first_valid) else 0.0, 0.0, 0.0])

        for t in range(n):
            x_p = x_curr if t == 0 else (F @ x_curr)
            x_pred_arr[t] = x_p
            if valid[t]:
                y_res = z_obs[t] - x_p[0]
                innov_sq[t] = (y_res * y_res) * s_inv
                x_curr = x_p + K_ss * y_res
            else:
                x_curr = x_p.copy()
                x_curr[1] *= 0.985
                x_curr[2] *= 0.90
            x_filt[t] = x_curr

        x_sm = x_filt.copy()
        for t in range(n - 2, -1, -1):
            x_sm[t] = x_filt[t] + G_ss @ (x_sm[t + 1] - x_pred_arr[t + 1])

        return x_sm, innov_sq

    x_sm, maha_x = _filter_1d(pos_x_m)
    y_sm, maha_y = _filter_1d(pos_y_m)

    vx, vy = x_sm[:, 1], y_sm[:, 1]
    ax, ay = x_sm[:, 2], y_sm[:, 2]
    speed = np.hypot(vx, vy)
    acc_mag = np.hypot(ax, ay)

    # Directional turn angle d(theta) between frame t-1 and t+1
    theta = np.arctan2(vy, vx)
    dtheta = np.zeros(n, dtype=np.float64)
    dtheta[1:-1] = np.abs(np.arctan2(np.sin(theta[2:] - theta[:-2]),
                                     np.cos(theta[2:] - theta[:-2])))

    return {
        "x_smooth": x_sm[:, 0],
        "y_smooth": y_sm[:, 0],
        "vx_smooth": vx,
        "vy_smooth": vy,
        "ax_smooth": ax,
        "ay_smooth": ay,
        "speed_smooth": speed,
        "acc_mag_smooth": acc_mag,
        "turn_angle_rad": dtheta,
        "mahalanobis_sq": maha_x + maha_y,
        "teleport_mask": teleport_mask,
        "clean_valid": valid,
    }


# ---------------------------------------------------------------------------
# Matrix 4: 46x46 Pairwise Player Interaction & Pass-Corridor Matrix (W_t)
# ---------------------------------------------------------------------------

def compute_player_interaction_matrices(px_m, py_m, vx_ms, vy_ms, team_of_track, ball_x_m, ball_y_m):
    """Compute pairwise player interaction summaries and pass-corridor openness at frame t or batch.

    Parameters
    ----------
    px_m, py_m : ndarray (N, S) in meters (NaN where inactive)
    vx_ms, vy_ms : ndarray (N, S) in m/s
    team_of_track : ndarray (S,) with 0=Home, 1=Away
    ball_x_m, ball_y_m : ndarray (N,) in meters
    """
    home_mask = (team_of_track == 0)
    away_mask = (team_of_track == 1)

    dx_b = px_m - ball_x_m[:, None]
    dy_b = py_m - ball_y_m[:, None]
    dist_ball = np.hypot(dx_b, dy_b)  # (N, S)

    home_dist = np.where(home_mask[None, :], dist_ball, np.inf)
    away_dist = np.where(away_mask[None, :], dist_ball, np.inf)

    home_sorted = np.sort(home_dist, axis=1)[:, :3]  # top-3 closest Home players
    away_sorted = np.sort(away_dist, axis=1)[:, :3]  # top-3 closest Away players

    nearest_home_idx = np.argmin(home_dist, axis=1)
    nearest_away_idx = np.argmin(away_dist, axis=1)
    rows = np.arange(len(px_m))

    vh_nearest = np.hypot(vx_ms[rows, nearest_home_idx], vy_ms[rows, nearest_home_idx])
    va_nearest = np.hypot(vx_ms[rows, nearest_away_idx], vy_ms[rows, nearest_away_idx])

    # Local player density within 1.8m, 3.5m, 6.0m of the ball
    h_within_18 = np.sum(home_dist < 1.8, axis=1)
    a_within_18 = np.sum(away_dist < 1.8, axis=1)
    h_within_35 = np.sum(home_dist < 3.5, axis=1)
    a_within_35 = np.sum(away_dist < 3.5, axis=1)
    h_within_60 = np.sum(home_dist < 6.0, axis=1)
    a_within_60 = np.sum(away_dist < 6.0, axis=1)

    return {
        "d_home_1": home_sorted[:, 0],
        "d_home_2": home_sorted[:, 1],
        "d_home_3": home_sorted[:, 2],
        "d_away_1": away_sorted[:, 0],
        "d_away_2": away_sorted[:, 1],
        "d_away_3": away_sorted[:, 2],
        "vh_nearest": vh_nearest,
        "va_nearest": va_nearest,
        "h_within_18": h_within_18,
        "a_within_18": a_within_18,
        "h_within_35": h_within_35,
        "a_within_35": a_within_35,
        "h_within_60": h_within_60,
        "a_within_60": a_within_60,
    }


# ---------------------------------------------------------------------------
# Matrix 5: 2x2 Team Spatial Dispersion & Tactical Inertia Tensor (Sigma_T)
# ---------------------------------------------------------------------------

def compute_team_spatial_inertia_tensors(px_m, py_m, team_of_track, gk_slots=(23, 45)):
    """Compute 2x2 spatial covariance matrices Sigma_H(t), Sigma_A(t) for outfield players.

    Returns
    -------
    inertia : dict of ndarrays
        centroid_hx, centroid_hy, centroid_ax, centroid_ay,
        area_h, area_a, log_area_ratio, eig1_h, eig2_h, eig1_a, eig2_a,
        def_line_h, def_line_a.
    """
    n_frames, n_slots = px_m.shape
    slot_idx = np.arange(n_slots)
    is_gk = np.isin(slot_idx, gk_slots)
    home_outfield = (team_of_track == 0) & (~is_gk)
    away_outfield = (team_of_track == 1) & (~is_gk)

    def _team_tensor(mask):
        x = np.where(mask[None, :], px_m, np.nan)
        y = np.where(mask[None, :], py_m, np.nan)
        cnt = np.sum(np.isfinite(x), axis=1)
        safe_cnt = np.maximum(cnt, 1)
        mx = np.nan_to_num(np.nanmean(x, axis=1), nan=52.65)
        my = np.nan_to_num(np.nanmean(y, axis=1), nan=34.0)

        dx = np.where(np.isfinite(x), x - mx[:, None], 0.0)
        dy = np.where(np.isfinite(y), y - my[:, None], 0.0)
        denom = np.maximum(safe_cnt - 1, 1)
        s_xx = np.sum(dx * dx, axis=1) / denom
        s_yy = np.sum(dy * dy, axis=1) / denom
        s_xy = np.sum(dx * dy, axis=1) / denom

        tr = s_xx + s_yy
        det = np.maximum(s_xx * s_yy - s_xy * s_xy, 1.0)
        disc = np.sqrt(np.maximum(0.0, tr * tr - 4.0 * det))
        eig1 = 0.5 * (tr + disc)
        eig2 = np.maximum(0.5 * (tr - disc), 0.5)
        area = np.sqrt(det)  # Effective team surface area in m^2

        # 2nd-deepest and 2nd-highest outfield player X coordinate
        x_sorted_asc = np.sort(np.where(np.isfinite(x), x, np.inf), axis=1)
        x_sorted_desc = -np.sort(np.where(np.isfinite(x), -x, np.inf), axis=1)
        low_line = np.where(np.isfinite(x_sorted_asc[:, 1]), x_sorted_asc[:, 1], mx)
        high_line = np.where(np.isfinite(x_sorted_desc[:, 1]), x_sorted_desc[:, 1], mx)
        return mx, my, area, eig1, eig2, low_line, high_line

    mx_h, my_h, area_h, eig1_h, eig2_h, low_h, high_h = _team_tensor(home_outfield)
    mx_a, my_a, area_a, eig1_a, eig2_a, low_a, high_a = _team_tensor(away_outfield)

    log_area_ratio = np.log(np.maximum(area_h, 1.0) / np.maximum(area_a, 1.0))

    return {
        "centroid_hx": mx_h,
        "centroid_hy": my_h,
        "centroid_ax": mx_a,
        "centroid_ay": my_a,
        "area_h": area_h,
        "area_a": area_a,
        "log_area_ratio": log_area_ratio,
        "eig1_h": eig1_h,
        "eig2_h": eig2_h,
        "eig1_a": eig1_a,
        "eig2_a": eig2_a,
        "low_line_h": low_h,
        "high_line_h": high_h,
        "low_line_a": low_a,
        "high_line_a": high_a,
    }


# ---------------------------------------------------------------------------
# Matrix 6: Anisotropic Velocity-Weighted Pitch Control Tensor (S_i(t))
# ---------------------------------------------------------------------------

def compute_anisotropic_pitch_control(eval_x_m, eval_y_m, px_m, py_m, vx_ms, vy_ms,
                                      team_of_track, base_radius_m=4.5, max_speed_ms=11.0):
    """Evaluate velocity-weighted anisotropic Gaussian pitch control at (eval_x_m, eval_y_m).

    Each player i at p_i with velocity v_i projects an influence ellipse shifted forward by
    0.5 * v_i and stretched along the velocity direction by (1 + ||v_i|| / max_speed) and
    compressed orthogonally, with 2x2 covariance S_i = R(theta_i) Lambda(s_i) R(theta_i)^T.
    """
    valid_p = np.isfinite(px_m) & np.isfinite(py_m)
    vx = np.nan_to_num(vx_ms, nan=0.0)
    vy = np.nan_to_num(vy_ms, nan=0.0)
    speed = np.hypot(vx, vy)
    s_ratio = np.clip(speed / max_speed_ms, 0.0, 1.0)

    # Shift influence center 0.45s ahead along velocity vector
    mu_x = np.nan_to_num(px_m, nan=-1e4) + 0.45 * vx
    mu_y = np.nan_to_num(py_m, nan=-1e4) + 0.45 * vy

    dx = eval_x_m[:, None] - mu_x
    dy = eval_y_m[:, None] - mu_y

    # Rotate (dx, dy) into player velocity frame
    cos_t = np.where(speed > 0.2, vx / np.maximum(speed, 1e-6), 1.0)
    sin_t = np.where(speed > 0.2, vy / np.maximum(speed, 1e-6), 0.0)

    d_par = cos_t * dx + sin_t * dy
    d_perp = -sin_t * dx + cos_t * dy

    sig_par = base_radius_m * (1.0 + 0.85 * s_ratio)
    sig_perp = base_radius_m * (1.0 - 0.30 * s_ratio)

    maha = (d_par / sig_par) ** 2 + (d_perp / sig_perp) ** 2
    influence = np.where(valid_p, np.exp(-0.5 * np.minimum(maha, 50.0)), 0.0)

    home_inf = np.sum(influence[:, team_of_track == 0], axis=1)
    away_inf = np.sum(influence[:, team_of_track == 1], axis=1)
    log_odds_pc = np.log((home_inf + 1e-3) / (away_inf + 1e-3))

    return {
        "home_pitch_control": home_inf,
        "away_pitch_control": away_inf,
        "pitch_control_log_odds": log_odds_pc,
    }


# ---------------------------------------------------------------------------
# Matrix 7: Multi-Scale Temporal Evidence & Contention Entropy Matrix (E_t)
# ---------------------------------------------------------------------------

def compute_multiscale_contention_matrix(raw_codes, half_windows=(8, 20, 40, 75, 150)):
    """Compute multi-horizon contact densities E_t in R^{N x K x 3} and duel contention entropy.

    Parameters
    ----------
    raw_codes : ndarray (N,) int8
        Per-frame raw contact indicator: 0=Home nearest, 1=Away nearest, 2=Ball Missing (NaN).
    half_windows : tuple of ints
        Half-widths w (in frames at 25 fps) for multi-scale integral boxcar kernels.

    Returns
    -------
    ms : dict of ndarrays
        E_matrix (N, K, 3) with [p_H, p_A, p_NaN] at each scale,
        alive_home_share (N, K), duel_flip_rate_2s (N,), contention_entropy_2s (N,),
        run_length_same (N,).
    """
    n = len(raw_codes)
    k_scales = len(half_windows)
    E_matrix = np.zeros((n, k_scales, 3), dtype=np.float64)
    alive_home_share = np.zeros((n, k_scales), dtype=np.float64)

    is_h = (raw_codes == 0).astype(np.int64)
    is_a = (raw_codes == 1).astype(np.int64)
    is_d = (raw_codes == 2).astype(np.int64)

    cum_h = np.concatenate(([0], np.cumsum(is_h)))
    cum_a = np.concatenate(([0], np.cumsum(is_a)))
    cum_d = np.concatenate(([0], np.cumsum(is_d)))
    idx = np.arange(n)

    for k, w in enumerate(half_windows):
        low = np.maximum(0, idx - w)
        high = np.minimum(n, idx + w + 1)
        span = np.maximum(high - low, 1).astype(np.float64)
        ch = (cum_h[high] - cum_h[low]) / span
        ca = (cum_a[high] - cum_a[low]) / span
        cd = (cum_d[high] - cum_d[low]) / span
        E_matrix[:, k, 0] = ch
        E_matrix[:, k, 1] = ca
        E_matrix[:, k, 2] = cd
        alive = ch + ca
        alive_home_share[:, k] = np.where(alive > 1e-6, ch / np.maximum(alive, 1e-6), 0.5)

    # Local H<->A duel flip rate over +/- 25 frames (2.0 seconds)
    flips = np.zeros(n, dtype=np.int64)
    flips[1:] = ((raw_codes[1:] != raw_codes[:-1]) &
                 (raw_codes[1:] < 2) & (raw_codes[:-1] < 2)).astype(np.int64)
    cum_flips = np.concatenate(([0], np.cumsum(flips)))
    low25 = np.maximum(0, idx - 25)
    high25 = np.minimum(n, idx + 26)
    duel_flip_rate_2s = (cum_flips[high25] - cum_flips[low25]) / np.maximum(high25 - low25, 1)

    # Contention Shannon entropy H_duel(t) at scale w=20 (~1.6s)
    p_h = np.clip(alive_home_share[:, 1], 1e-4, 1.0 - 1e-4)
    contention_entropy_2s = -(p_h * np.log2(p_h) + (1.0 - p_h) * np.log2(1.0 - p_h))

    # Contiguous run length of the current raw code
    breaks = np.flatnonzero(raw_codes[1:] != raw_codes[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [n]))
    run_length_same = np.zeros(n, dtype=np.int32)
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        run_length_same[b0:b1] = b1 - b0

    return {
        "E_matrix": E_matrix,
        "alive_home_share": alive_home_share,
        "duel_flip_rate_2s": duel_flip_rate_2s,
        "contention_entropy_2s": contention_entropy_2s,
        "run_length_same": run_length_same,
    }


# ---------------------------------------------------------------------------
# Matrix 8: Time-Inhomogeneous Non-Stationary HSMM Transition Matrix (A_t)
# ---------------------------------------------------------------------------

def build_time_inhomogeneous_transition_matrices(
    n_frames,
    stoppage_entry_gate,
    stoppage_exit_gate,
    live_bridge_lock,
    turnover_gate,
    stay_alive_penalty=5.6,
    stay_dead_penalty=6.8,
):
    """Construct the frame-dynamic 3x3 log-transition tensor A_t in R^{N x 3 x 3}.

    States: 0 = H, 1 = A, 2 = D.
    A_t[t, i, j] = log P(S_t = j | S_{t-1} = i).
    """
    A_t = np.full((n_frames, 3, 3), -stay_alive_penalty, dtype=np.float64)

    # Diagonal self-transitions
    A_t[:, 0, 0] = 0.0
    A_t[:, 1, 1] = 0.0
    A_t[:, 2, 2] = 0.0

    # H <-> A transitions modulated by turnover_gate in [0.25, 2.5]
    ha_cost = -stay_alive_penalty / np.clip(turnover_gate, 0.20, 3.0)
    A_t[:, 0, 1] = ha_cost
    A_t[:, 1, 0] = ha_cost

    # H/A -> D transitions: heavily locked (-18.0) during live aerial bridges,
    # lowered when stoppage_entry_gate > 1.0
    to_d_cost = np.where(
        live_bridge_lock,
        -18.0,
        -stay_dead_penalty / np.clip(stoppage_entry_gate, 0.15, 4.0),
    )
    A_t[:, 0, 2] = to_d_cost
    A_t[:, 1, 2] = to_d_cost

    # D -> H/A transitions: opened at stoppage_exit_gate (restart kicks)
    from_d_cost = -stay_dead_penalty / np.clip(stoppage_exit_gate, 0.15, 4.0)
    A_t[:, 2, 0] = from_d_cost
    A_t[:, 2, 1] = from_d_cost

    # Row-normalize in log-space (log-softmax per row) so each 3x3 matrix is a true stochastic matrix
    max_row = np.max(A_t, axis=2, keepdims=True)
    log_sum_exp = max_row + np.log(np.sum(np.exp(A_t - max_row), axis=2, keepdims=True))
    A_t = A_t - log_sum_exp

    return A_t
