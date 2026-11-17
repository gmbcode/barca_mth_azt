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
        valid_xy = np.isfinite(x) & np.isfinite(y)
        cnt = np.sum(valid_xy, axis=1)
        safe_cnt = np.maximum(cnt, 1)
        sum_x = np.sum(np.where(valid_xy, x, 0.0), axis=1)
        sum_y = np.sum(np.where(valid_xy, y, 0.0), axis=1)
        mx = np.where(cnt > 0, sum_x / safe_cnt, 52.65)
        my = np.where(cnt > 0, sum_y / safe_cnt, 34.0)

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
    turnover_gate_ha=None,
    turnover_gate_ah=None,
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

    # H -> A and A -> H transitions (supports symmetric turnover_gate or Flick asymmetric gates)
    g_ha = turnover_gate if turnover_gate_ha is None else turnover_gate_ha
    g_ah = turnover_gate if turnover_gate_ah is None else turnover_gate_ah
    A_t[:, 0, 1] = -stay_alive_penalty / np.clip(g_ha, 0.20, 3.0)
    A_t[:, 1, 0] = -stay_alive_penalty / np.clip(g_ah, 0.20, 3.0)

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


# ---------------------------------------------------------------------------
# Matrix 9: Pressing Commitment, Defender Pinning & Blind-Side Risk Tensor
# ---------------------------------------------------------------------------

def compute_press_and_pinning_matrices(
    px_m,
    py_m,
    vx_ms,
    vy_ms,
    team_of_track,
    ball_x_m,
    ball_y_m,
    pitch_length=105.3,
    half_split_frame=90652,
    gk_slots=(23, 45),
    sigma_mark_m=5.5,
    chunk_size=4096,
):
    """Compute vectorized tactical pressing commitment & attacker-behind-defender pinning metrics.

    A defender cannot safely jump to press the ball if jumping leaves an opponent attacker
    unmarked on their goal-side / blind shoulder without cover behind.

    Returns dict of (N,) float64 arrays:
      - pin_mean_a / pin_mean_h      : Mean goal-side pinning load on nearby defenders in [0, 1]
      - press_cap_on_h / press_cap_on_a : Unpinned closing-speed-weighted press pressure on carrier
      - bait_safety_h / bait_safety_a : Fraction of nearby defender pressure neutralized by pinning
      - n_free_outlets_h / _a         : Number of unguarded open passing outlets (4m+ separation)
      - net_press_hazard_h            : Net turnover hazard favoring Away steal (+) vs Home hold (-)
    """
    n_frames, n_slots = px_m.shape
    slot_idx = np.arange(n_slots)
    is_gk = np.isin(slot_idx, gk_slots)
    h_out = (team_of_track == 0) & (~is_gk)
    a_out = (team_of_track == 1) & (~is_gk)

    pin_mean_a = np.zeros(n_frames, dtype=np.float64)
    pin_mean_h = np.zeros(n_frames, dtype=np.float64)
    press_cap_on_h = np.zeros(n_frames, dtype=np.float64)
    press_cap_on_a = np.zeros(n_frames, dtype=np.float64)
    bait_safety_h = np.full(n_frames, 0.5, dtype=np.float64)
    bait_safety_a = np.full(n_frames, 0.5, dtype=np.float64)
    n_free_outlets_h = np.zeros(n_frames, dtype=np.float64)
    n_free_outlets_a = np.zeros(n_frames, dtype=np.float64)

    two_sig_sq = 2.0 * (sigma_mark_m ** 2)
    two_cov_sq = 2.0 * (5.0 ** 2)

    for s0 in range(0, n_frames, chunk_size):
        s1 = min(n_frames, s0 + chunk_size)
        px = px_m[s0:s1]
        py = py_m[s0:s1]
        vx = np.nan_to_num(vx_ms[s0:s1], nan=0.0)
        vy = np.nan_to_num(vy_ms[s0:s1], nan=0.0)
        bx = ball_x_m[s0:s1, None]
        by = ball_y_m[s0:s1, None]

        # Flip X in Half 2 so +x_att always points toward Away's goal
        h2_mask = (np.arange(s0, s1) >= half_split_frame)[:, None]
        x_att = np.where(h2_mask, pitch_length - px, px)

        valid = np.isfinite(px) & np.isfinite(py)
        vh = valid & h_out[None, :]
        va = valid & a_out[None, :]

        # Distance & closing speed toward ball
        dbx = bx - px
        dby = by - py
        d_ball = np.hypot(dbx, dby)
        u_bx = dbx / np.maximum(d_ball, 0.5)
        u_by = dby / np.maximum(d_ball, 0.5)
        v_close = np.maximum(0.0, vx * u_bx + vy * u_by)
        w_ball_prox = np.where(valid, np.exp(-np.nan_to_num(d_ball, nan=1e3) / 6.5), 0.0)

        # Pairwise distances (B, S, S) safely masked where invalid
        px_s = np.nan_to_num(px, nan=1e4)
        py_s = np.nan_to_num(py, nan=1e4)
        dx_ij = px_s[:, :, None] - px_s[:, None, :]
        dy_ij = py_s[:, :, None] - py_s[:, None, :]
        d2_ij = dx_ij * dx_ij + dy_ij * dy_ij
        x_att_s = np.nan_to_num(x_att, nan=0.0)

        # --- Away defenders pinned by Home attackers behind them (x_att[k] > x_att[j] - 0.5) ---
        # Cover on Home attacker k from OTHER Away defender m: x_att[m] > x_att[k] + 0.8 (implies m != j)
        h_goal_side_of_a = vh[:, None, :] & va[:, :, None] & (x_att_s[:, None, :] > (x_att_s[:, :, None] - 0.5))
        cov_a_k = (
            va[:, :, None]
            & vh[:, None, :]
            & (x_att_s[:, :, None] > (x_att_s[:, None, :] + 0.8))
        )
        cov_w_a = np.where(cov_a_k, np.exp(-d2_ij / two_cov_sq), 0.0)  # (B, m, k)
        max_cov_on_k_by_a = np.max(cov_w_a, axis=1)  # (B, k)
        pin_pair_a = np.where(
            h_goal_side_of_a & (d2_ij < 144.0),
            np.exp(-d2_ij / two_sig_sq) * (1.0 - 0.65 * max_cov_on_k_by_a[:, None, :]),
            0.0,
        )
        pin_j_a = np.clip(np.max(pin_pair_a, axis=2), 0.0, 1.0)  # (B, j in Away)

        # --- Home defenders pinned by Away attackers behind them (x_att[k] < x_att[j] + 0.5) ---
        # Cover on Away attacker k from OTHER Home defender m: x_att[m] < x_att[k] - 0.8 (implies m != j)
        a_goal_side_of_h = va[:, None, :] & vh[:, :, None] & (x_att_s[:, None, :] < (x_att_s[:, :, None] + 0.5))
        cov_h_k = (
            vh[:, :, None]
            & va[:, None, :]
            & (x_att_s[:, :, None] < (x_att_s[:, None, :] - 0.8))
        )
        cov_w_h = np.where(cov_h_k, np.exp(-d2_ij / two_cov_sq), 0.0)
        max_cov_on_k_by_h = np.max(cov_w_h, axis=1)
        pin_pair_h = np.where(
            a_goal_side_of_h & (d2_ij < 144.0),
            np.exp(-d2_ij / two_sig_sq) * (1.0 - 0.65 * max_cov_on_k_by_h[:, None, :]),
            0.0,
        )
        pin_j_h = np.clip(np.max(pin_pair_h, axis=2), 0.0, 1.0)

        # Nearby proximity weights (only pressers near ball zone)
        wa_zone = np.where(va & (d_ball < 18.0), w_ball_prox, 0.0)
        wh_zone = np.where(vh & (d_ball < 18.0), w_ball_prox, 0.0)

        pin_mean_a[s0:s1] = np.sum(wa_zone * pin_j_a, axis=1) / np.maximum(np.sum(wa_zone, axis=1), 1e-3)
        pin_mean_h[s0:s1] = np.sum(wh_zone * pin_j_h, axis=1) / np.maximum(np.sum(wh_zone, axis=1), 1e-3)

        # Closing-speed weighted unpinned press capacity
        eff_a = np.where(va, w_ball_prox * (1.0 - pin_j_a) * (0.35 + np.clip(v_close / 5.0, 0.0, 1.5)), 0.0)
        eff_h = np.where(vh, w_ball_prox * (1.0 - pin_j_h) * (0.35 + np.clip(v_close / 5.0, 0.0, 1.5)), 0.0)
        press_cap_on_h[s0:s1] = np.sum(eff_a, axis=1)
        press_cap_on_a[s0:s1] = np.sum(eff_h, axis=1)

        # Press-bait safety: fraction of nearby opponent pressure neutralized by pinning
        raw_a_p = np.sum(wa_zone, axis=1)
        raw_h_p = np.sum(wh_zone, axis=1)
        bait_safety_h[s0:s1] = np.where(raw_a_p > 1e-3, np.sum(wa_zone * pin_j_a, axis=1) / (raw_a_p + 1e-3), 0.5)
        bait_safety_a[s0:s1] = np.where(raw_h_p > 1e-3, np.sum(wh_zone * pin_j_h, axis=1) / (raw_h_p + 1e-3), 0.5)

        # Open passing outlets: teammates 3m..35m from ball with >=4.0m min distance to nearest defender
        d_ij = np.sqrt(np.maximum(d2_ij, 0.0))
        min_opp_to_h = np.min(np.where(va[:, None, :], d_ij, np.inf), axis=2)  # (B, i)
        min_opp_to_a = np.min(np.where(vh[:, None, :], d_ij, np.inf), axis=2)
        h_outlets = vh & (d_ball >= 3.0) & (d_ball <= 35.0) & (min_opp_to_h >= 4.0)
        a_outlets = va & (d_ball >= 3.0) & (d_ball <= 35.0) & (min_opp_to_a >= 4.0)
        n_free_outlets_h[s0:s1] = np.sum(h_outlets, axis=1)
        n_free_outlets_a[s0:s1] = np.sum(a_outlets, axis=1)

    net_press_hazard_h = (press_cap_on_h - 0.85 * bait_safety_h) - (press_cap_on_a - 0.85 * bait_safety_a)

    return {
        "pin_mean_a": pin_mean_a,
        "pin_mean_h": pin_mean_h,
        "press_cap_on_h": press_cap_on_h,
        "press_cap_on_a": press_cap_on_a,
        "bait_safety_h": bait_safety_h,
        "bait_safety_a": bait_safety_a,
        "n_free_outlets_h": n_free_outlets_h,
        "n_free_outlets_a": n_free_outlets_a,
        "net_press_hazard_h": net_press_hazard_h,
    }


# ---------------------------------------------------------------------------
# Matrix 10: ATD Re-ID Sanity, Open-Space Ball Carrying (TTI) & Shooting Range
# ---------------------------------------------------------------------------

def clean_atd_trajectories(
    px_m,
    py_m,
    team_of_track,
    gk_slots=(23, 45),
    half_split_frame=90652,
    pitch_length=105.3,
    fps=25.0,
):
    """Mask impossible opponent-box goalkeeper ID swaps and zero out teleportation velocity spikes.

    In broadcast/tactical ATD tracking, when the camera zooms tightly into one penalty box
    leaving only one keeper visible, Metrica ATD occasionally assigns the wrong team's GK slot
    inside that box (> 62m from their own goal). Masking wrong-end GK slots (outside brief
    final-3% late-game corner kicks) prevents phantom 1-12s possession hijackings.
    """
    n_frames, n_slots = px_m.shape
    px_clean = px_m.copy()
    py_clean = py_m.copy()

    is_h2 = (np.arange(n_frames) >= half_split_frame)[:, None]
    gk_arr = [int(s) for s in gk_slots if 0 <= int(s) < n_slots]
    if gk_arr:
        # Late-match window (final 3% of each half) allows genuine GK corners if box is packed
        half1_late = int(half_split_frame * 0.96)
        half2_len = max(n_frames - half_split_frame, 1)
        half2_late = half_split_frame + int(half2_len * 0.96)
        f_idx = np.arange(n_frames)
        late_window = ((~is_h2[:, 0]) & (f_idx >= half1_late)) | (is_h2[:, 0] & (f_idx >= half2_late))

        for s in gk_arr:
            sx = px_clean[:, s]
            sy = py_clean[:, s]
            valid_s = np.isfinite(sx) & np.isfinite(sy)
            # Own defended goal: Home defends x=0 in H1, pitch_length in H2; Away opposite
            if team_of_track[s] == 0:
                d_own = np.where(is_h2[:, 0], pitch_length - sx, sx)
                in_opp_box_mask = np.where(is_h2, px_clean < 25.0, px_clean > pitch_length - 25.0)
            else:
                d_own = np.where(is_h2[:, 0], sx, pitch_length - sx)
                in_opp_box_mask = np.where(is_h2, px_clean > pitch_length - 25.0, px_clean < 25.0)
            n_box = np.sum(np.isfinite(px_clean) & in_opp_box_mask, axis=1)
            wrong_end = valid_s & (d_own > 62.0) & (~(late_window & (n_box >= 12)))
            px_clean[wrong_end, s] = np.nan
            py_clean[wrong_end, s] = np.nan

    # Finite-pair central differences with teleportation guard (> 2.2m single-frame jump -> 0)
    vx_ms = np.zeros((n_frames, n_slots), dtype=np.float64)
    vy_ms = np.zeros((n_frames, n_slots), dtype=np.float64)
    dt2 = 2.0 / float(fps)

    step_ok = np.ones((n_frames, n_slots), dtype=bool)
    if n_frames > 2:
        jump_f = np.hypot(px_clean[2:] - px_clean[:-2], py_clean[2:] - py_clean[:-2])
        step_ok[1:-1] = np.isfinite(jump_f) & (jump_f <= 3.6)
        v_pair = (
            np.isfinite(px_clean[:-2])
            & np.isfinite(px_clean[2:])
            & np.isfinite(py_clean[:-2])
            & np.isfinite(py_clean[2:])
            & step_ok[1:-1]
        )
        vx_ms[1:-1] = np.where(v_pair, np.clip((px_clean[2:] - px_clean[:-2]) / dt2, -12.0, 12.0), 0.0)
        vy_ms[1:-1] = np.where(v_pair, np.clip((py_clean[2:] - py_clean[:-2]) / dt2, -12.0, 12.0), 0.0)

    return px_clean, py_clean, vx_ms, vy_ms


def compute_carry_and_shooting_matrices(
    px_m,
    py_m,
    vx_ms,
    vy_ms,
    team_of_track,
    ball_x_m,
    ball_y_m,
    ball_x_smooth,
    ball_y_smooth,
    vb_x_smooth,
    vb_y_smooth,
    v_ball_smooth,
    cam_x_smooth,
    cam_y_smooth,
    pitch_length=105.3,
    pitch_width=68.0,
    half_split_frame=90652,
    gk_slots=(23, 45),
    chunk_size=4096,
):
    """Compute Matrix 10: Open-Space Knock-On Carrying, TTI Advantage & Shooting-Cone Dynamics.

    Captures three physical decision mechanisms missed by pure nearest-player Voronoi:
      1. Heavy-touch / knock-on carries in open space (`knock_on_h/a`): carrier pushes the ball
         1.4..7.0m ahead into space while sprinting aligned with the ball ray and beating opponents'
         Time-to-Intercept (TTI) -- avoids misclassifying fast solo dribbles as uncertain flight.
      2. Shooting-range risk & take-on freedom (`shot_takeon_open_h/a`, `shot_crowded_risk_h/a`,
         `shot_flight`): counts defenders inside the tapered shot cone to goalposts + 4m bubble;
         open lanes favor solo take-on/carry stability while crowded cones elevate shot/block turns.
      3. Ball-free attacking-third / penalty-box occupancy & camera pitch control (`att_third_occ_diff`,
         `box_occ_h/a`, `pc_cam_log_odds`) for missing-ball prediction without nearest_slot leakage.
    """
    n_frames, n_slots = px_m.shape
    slot_idx = np.arange(n_slots)
    is_gk = np.isin(slot_idx, gk_slots)
    h_all = team_of_track == 0
    a_all = team_of_track == 1
    h_out = h_all & (~is_gk)
    a_out = a_all & (~is_gk)

    gy = 0.5 * pitch_width
    f_arr = np.arange(n_frames)
    is_h2 = f_arr >= half_split_frame
    gx_h = np.where(is_h2, 0.0, pitch_length)
    gx_a = np.where(is_h2, pitch_length, 0.0)

    # Output arrays
    tti_margin_ha = np.zeros(n_frames, dtype=np.float64)
    knock_on_h = np.zeros(n_frames, dtype=bool)
    knock_on_a = np.zeros(n_frames, dtype=bool)
    open_carry_h = np.zeros(n_frames, dtype=np.float64)
    open_carry_a = np.zeros(n_frames, dtype=np.float64)
    space_ahead_h = np.full(n_frames, 12.0, dtype=np.float64)
    space_ahead_a = np.full(n_frames, 12.0, dtype=np.float64)

    d_goal_h = np.hypot(ball_x_smooth - gx_h, ball_y_smooth - gy)
    d_goal_a = np.hypot(ball_x_smooth - gx_a, ball_y_smooth - gy)
    n_def_cone_h = np.zeros(n_frames, dtype=np.float64)
    n_def_cone_a = np.zeros(n_frames, dtype=np.float64)
    shot_takeon_open_h = np.zeros(n_frames, dtype=np.float64)
    shot_takeon_open_a = np.zeros(n_frames, dtype=np.float64)
    shot_crowded_risk_h = np.zeros(n_frames, dtype=np.float64)
    shot_crowded_risk_a = np.zeros(n_frames, dtype=np.float64)
    shot_flight = np.zeros(n_frames, dtype=bool)

    box_occ_h = np.zeros(n_frames, dtype=np.float64)
    box_occ_a = np.zeros(n_frames, dtype=np.float64)
    att_third_occ_diff = np.zeros(n_frames, dtype=np.float64)
    pc_ball_log_odds = np.zeros(n_frames, dtype=np.float64)
    pc_cam_log_odds = np.zeros(n_frames, dtype=np.float64)

    # Predicted ball landing point 0.28s ahead for Time-to-Intercept race
    bpx = np.clip(ball_x_smooth + 0.28 * vb_x_smooth, 0.0, pitch_length)
    bpy = np.clip(ball_y_smooth + 0.28 * vb_y_smooth, 0.0, pitch_width)

    def _sigmoid_arr(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -12.0, 12.0)))

    for s0 in range(0, n_frames, chunk_size):
        s1 = min(n_frames, s0 + chunk_size)
        pad_pre = max(0, s0 - 38)
        pre_len = s0 - pad_pre
        px_pad = px_m[pad_pre:s1]
        py_pad = py_m[pad_pre:s1]
        bx_pad_raw = ball_x_m[pad_pre:s1]
        by_pad_raw = ball_y_m[pad_pre:s1]
        b_vis_pad = np.isfinite(bx_pad_raw) & np.isfinite(by_pad_raw)
        bx_pad = ball_x_smooth[pad_pre:s1]
        by_pad = ball_y_smooth[pad_pre:s1]
        d_pad = np.hypot(bx_pad[:, None] - px_pad, by_pad[:, None] - py_pad)
        touch_22_pad = np.isfinite(px_pad) & b_vis_pad[:, None] & (d_pad <= 2.25)

        # Past-38f sliding cumulative count for each slot & past ball visibility count
        cs_t = np.vstack([np.zeros((1, n_slots), dtype=np.int32), np.cumsum(touch_22_pad, axis=0, dtype=np.int32)])
        cs_vis = np.concatenate(([0], np.cumsum(b_vis_pad.astype(np.int32))))
        loc_idx = np.arange(pre_len, pre_len + (s1 - s0))
        lo_win = np.maximum(0, loc_idx - 38)
        slot_touched_38 = (cs_t[loc_idx + 1] - cs_t[lo_win]) > 0
        had_ball_hist = (cs_vis[loc_idx] - cs_vis[lo_win]) >= 3

        px = px_m[s0:s1]
        py = py_m[s0:s1]
        vx = np.nan_to_num(vx_ms[s0:s1], nan=0.0)
        vy = np.nan_to_num(vy_ms[s0:s1], nan=0.0)

        bx_raw = ball_x_m[s0:s1]
        by_raw = ball_y_m[s0:s1]
        b_vis = np.isfinite(bx_raw) & np.isfinite(by_raw)
        bx = ball_x_smooth[s0:s1]
        by = ball_y_smooth[s0:s1]

        valid = np.isfinite(px) & np.isfinite(py)
        vh_all = valid & h_all[None, :]
        va_all = valid & a_all[None, :]
        vh_out = valid & h_out[None, :]
        va_out = valid & a_out[None, :]

        # --- A. Ball-Free Occupancy & Anisotropic Pitch Control (Camera & Ball) ---
        h2_c = is_h2[s0:s1, None]
        x_att_h = np.where(h2_c, pitch_length - px, px)  # + toward Away goal
        box_occ_h[s0:s1] = np.sum(vh_out & (x_att_h >= pitch_length - 25.0), axis=1)
        box_occ_a[s0:s1] = np.sum(va_out & (x_att_h <= 25.0), axis=1)
        att_third_h = np.sum(vh_out & (x_att_h >= 0.66 * pitch_length), axis=1)
        att_third_a = np.sum(va_out & (x_att_h <= 0.34 * pitch_length), axis=1)
        att_third_occ_diff[s0:s1] = att_third_h - att_third_a

        pc_cam = compute_anisotropic_pitch_control(
            cam_x_smooth[s0:s1], cam_y_smooth[s0:s1], px, py, vx, vy, team_of_track
        )
        pc_cam_log_odds[s0:s1] = np.clip(pc_cam["pitch_control_log_odds"], -5.0, 5.0)
        pc_ball = compute_anisotropic_pitch_control(bx, by, px, py, vx, vy, team_of_track)
        pc_ball_log_odds[s0:s1] = np.clip(pc_ball["pitch_control_log_odds"], -5.0, 5.0)

        # --- B. Time-to-Intercept & Open-Space Knock-On Carrying ---
        # Distances from all players to current ball (use smoothed ball, gate carries on b_vis)
        dx_b = bx[:, None] - px
        dy_b = by[:, None] - py
        dist_b = np.hypot(dx_b, dy_b)
        dh_sort = np.sort(np.where(vh_all, dist_b, 1e6), axis=1)
        da_sort = np.sort(np.where(va_all, dist_b, 1e6), axis=1)
        dh1 = dh_sort[:, 0]
        dh2 = dh_sort[:, 1]
        da1 = da_sort[:, 0]
        da2 = da_sort[:, 1]

        row_loc = np.arange(s1 - s0)
        sh_i = np.argmin(np.where(vh_all, dist_b, 1e6), axis=1)
        sa_i = np.argmin(np.where(va_all, dist_b, 1e6), axis=1)

        # TTI race to predicted ball landing point (bpx, bpy)
        dx_p = bpx[s0:s1, None] - px
        dy_p = bpy[s0:s1, None] - py
        dp = np.hypot(dx_p, dy_p)
        u_px = dx_p / np.maximum(dp, 0.25)
        u_py = dy_p / np.maximum(dp, 0.25)
        v_along_p = np.maximum(0.0, vx * u_px + vy * u_py)
        tti_all = np.where(valid, dp / (3.2 + 0.85 * v_along_p) + 0.08, 20.0)
        tti_h_min = np.min(np.where(vh_all, tti_all, 20.0), axis=1)
        tti_a_min = np.min(np.where(va_all, tti_all, 20.0), axis=1)
        tti_margin_ha[s0:s1] = np.clip((tti_a_min - tti_h_min) / 1.5, -2.0, 2.0)

        # Evaluate knock-on & space-ahead kinematics symmetrically for Home (0) and Away (1)
        vbx = vb_x_smooth[s0:s1]
        vby = vb_y_smooth[s0:s1]
        vb = v_ball_smooth[s0:s1]

        for tm_idx, c_slot, own_mask, own_d1, own_d2, opp_d1, tti_own, tti_opp, opp_mask, att_dir, out_ko, out_oc, out_sa in [
            (0, sh_i, vh_all, dh1, dh2, da1, tti_h_min, tti_a_min, va_all, np.where(is_h2[s0:s1], -1.0, 1.0), knock_on_h, open_carry_h, space_ahead_h),
            (1, sa_i, va_all, da1, da2, dh1, tti_a_min, tti_h_min, vh_all, np.where(is_h2[s0:s1], 1.0, -1.0), knock_on_a, open_carry_a, space_ahead_a),
        ]:
            cvx = vx[row_loc, c_slot]
            cvy = vy[row_loc, c_slot]
            c_spd = np.hypot(cvx, cvy)
            r_cb_x = bx - np.nan_to_num(px[row_loc, c_slot], nan=bx)
            r_cb_y = by - np.nan_to_num(py[row_loc, c_slot], nan=by)
            r_cb_d = np.hypot(r_cb_x, r_cb_y) + 1e-6

            cos_chase = (cvx * r_cb_x + cvy * r_cb_y) / (c_spd * r_cb_d + 1e-6)
            cos_bdir = (cvx * vbx + cvy * vby) / (c_spd * vb + 1e-6)

            # Origin check: carrier slot ITSELF must have come within <=2.25m of seen ball
            # in the last 38 frames (when visible-ball history exists) to count as a knock-on carry;
            # prevents an opponent presser chasing someone else's long pass from faking a dribble!
            c_touched = slot_touched_38[row_loc, c_slot]
            origin_ok = (~had_ball_hist) | c_touched

            # Space ahead along ball motion (or attacking direction when ball is slow)
            dir_x = np.where(vb > 2.5, vbx / np.maximum(vb, 1e-6), att_dir)
            dir_y = np.where(vb > 2.5, vby / np.maximum(vb, 1e-6), 0.0)
            ox_b = px - bx[:, None]
            oy_b = py - by[:, None]
            proj_b = ox_b * dir_x[:, None] + oy_b * dir_y[:, None]
            perp_b = np.abs(-ox_b * dir_y[:, None] + oy_b * dir_x[:, None])
            in_front = opp_mask & (proj_b > 0.0) & (perp_b <= 5.5 + 0.35 * proj_b)
            sa_vals = np.min(np.where(in_front, np.maximum(proj_b, 0.0), 35.0), axis=1)
            out_sa[s0:s1] = sa_vals

            ko_m = (
                b_vis
                & origin_ok
                & (own_d1 >= 1.4)
                & (own_d1 <= 7.0)
                & (opp_d1 >= np.maximum(3.2, own_d1 + 1.1))
                & (own_d2 >= own_d1 + 1.6)
                & (c_spd >= 2.8)
                & (cos_chase > 0.50)
                & (cos_bdir > 0.42)
                & (vb >= 2.5)
                & (vb <= 19.0)
                & ((tti_opp - tti_own) >= 0.08)
            )
            out_ko[s0:s1] = ko_m
            # Continuous open-carry score combines close open dribble + knock-on carry (gated when far without origin)
            touch_or_close = np.where((own_d1 <= 2.4) | origin_ok, 1.0, 0.25)
            oc_score = (
                b_vis.astype(np.float64)
                * touch_or_close
                * np.exp(-0.5 * (np.maximum(own_d1 - 1.5, 0.0) / 3.2) ** 2)
                * _sigmoid_arr((opp_d1 - np.maximum(3.2, own_d1 + 0.8)) / 1.0)
                * _sigmoid_arr((c_spd - 2.6) / 0.9)
                * np.clip(0.5 + 0.5 * np.maximum(cos_bdir, cos_chase), 0.0, 1.0)
                * np.clip(sa_vals / 12.0, 0.25, 1.0)
            )
            out_oc[s0:s1] = np.where(ko_m, np.maximum(oc_score, 0.72), oc_score)

        # --- C. Shooting Range, Goal-Cone Blockers & Shot Flight ---
        for gx_loc, d_gl, opp_mask, opp_d1, out_cone, out_takeon, out_crowd in [
            (gx_h[s0:s1], d_goal_h[s0:s1], va_all, da1, n_def_cone_h, shot_takeon_open_h, shot_crowded_risk_h),
            (gx_a[s0:s1], d_goal_a[s0:s1], vh_all, dh1, n_def_cone_a, shot_takeon_open_a, shot_crowded_risk_a),
        ]:
            dxg = gx_loc - bx
            dyg = gy - by
            dist_g = np.hypot(dxg, dyg) + 1e-6
            ugx = dxg / dist_g
            ugy = dyg / dist_g

            ox = px - bx[:, None]
            oy = py - by[:, None]
            proj_g = ox * ugx[:, None] + oy * ugy[:, None]
            perp_g = np.abs(-ox * ugy[:, None] + oy * ugx[:, None])
            half_w = 1.6 + 3.0 * np.clip(proj_g / dist_g[:, None], 0.0, 1.0)
            in_cone = opp_mask & (proj_g > 0.2) & (proj_g < dist_g[:, None]) & (perp_g <= half_w)
            nc = np.sum(in_cone, axis=1).astype(np.float64)

            in_zone = b_vis & (d_gl <= 28.0) & (np.abs(by - gy) <= 20.0)
            out_cone[s0:s1] = np.where(in_zone, nc, 0.0)
            out_takeon[s0:s1] = np.where(
                in_zone,
                np.exp(-0.5 * (np.maximum(nc - 1.0, 0.0) / 1.1) ** 2) * _sigmoid_arr((opp_d1 - 2.6) / 0.8),
                0.0,
            )
            out_crowd[s0:s1] = np.where(
                in_zone,
                np.clip(0.5 * _sigmoid_arr((nc - 2.2) / 0.8) + 0.5 * _sigmoid_arr((2.4 - opp_d1) / 0.7), 0.0, 1.0),
                0.0,
            )

        # High-velocity shot flight toward either goal inside shooting range
        dxgh, dygh = gx_h[s0:s1] - bx, gy - by
        v_to_h = (vbx * dxgh + vby * dygh) / np.maximum(np.hypot(dxgh, dygh), 1.0)
        dxga, dyga = gx_a[s0:s1] - bx, gy - by
        v_to_a = (vbx * dxga + vby * dyga) / np.maximum(np.hypot(dxga, dyga), 1.0)
        v_to_goal = np.where(d_goal_h[s0:s1] <= d_goal_a[s0:s1], v_to_h, v_to_a)
        shot_flight[s0:s1] = (
            b_vis
            & (np.minimum(d_goal_h[s0:s1], d_goal_a[s0:s1]) <= 28.0)
            & (vb >= 13.5)
            & (v_to_goal >= 9.0)
        )

    return {
        "tti_margin_ha": tti_margin_ha,
        "knock_on_h": knock_on_h,
        "knock_on_a": knock_on_a,
        "open_carry_h": open_carry_h,
        "open_carry_a": open_carry_a,
        "space_ahead_h": space_ahead_h,
        "space_ahead_a": space_ahead_a,
        "d_goal_h": d_goal_h,
        "d_goal_a": d_goal_a,
        "n_def_cone_h": n_def_cone_h,
        "n_def_cone_a": n_def_cone_a,
        "shot_takeon_open_h": shot_takeon_open_h,
        "shot_takeon_open_a": shot_takeon_open_a,
        "shot_crowded_risk_h": shot_crowded_risk_h,
        "shot_crowded_risk_a": shot_crowded_risk_a,
        "shot_flight": shot_flight,
        "box_occ_h": box_occ_h,
        "box_occ_a": box_occ_a,
        "att_third_occ_diff": att_third_occ_diff,
        "pc_ball_log_odds": pc_ball_log_odds,
        "pc_cam_log_odds": pc_cam_log_odds,
    }
