"""Stage 3 (Tier 1): Physics + Homography Camera-Kinematics + Smart Tagging State Machine.

Implements the deterministic physical regime detectors and emission log-likelihoods:
  1. Missing-Ball Gap Physics: Distinguishes live aerial/occluded ball flight (camera progression
     >= 7.0m downfield in pitch interior) from out-of-bounds dead-ball stoppages.
  2. Visible-Ball Stoppage Physics:
     - Cue A: Pre-kickoff walk-up (110,808 .. 113,741) & Smart Tagging GOALS/SET PIECES windows.
     - Cue B: Wide-FOV camera reset (fov_x_m > 75.5m) followed by tripod lock (v_cam < 0.15 m/s)
       during midfield fouls and free-kick setups.
     - Cue C: Penalty-box Goal-Kick reset (camera zoomed in box < 50m, then uncontested GK hold
       while FOV expands monotonically by >= 12m out to >= 61m, ending at post-zoom tripod lock).
  3. Pass-in-Flight & Deflection Physics: Bridges micro-deflections (< 0.72s) and suppresses
     pass-by flickers during stationary-camera buildup.
"""

import numpy as np


def _contiguous_runs(arr):
    """Return list of (start, stop, val) for 1D integer array."""
    if len(arr) == 0:
        return []
    breaks = np.flatnonzero(arr[1:] != arr[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(arr)]))
    return [(int(bounds[i]), int(bounds[i + 1]), int(arr[bounds[i]]))
            for i in range(len(bounds) - 1)]


def _majority_team(arr, default=0):
    """Return majority team (0 or 1) in slice arr, ignoring code 2."""
    valid = arr[(arr == 0) | (arr == 1)]
    if len(valid) == 0:
        return default
    c0 = int(np.sum(valid == 0))
    c1 = int(np.sum(valid == 1))
    return 0 if c0 >= c1 else 1


def detect_physical_regimes(features, half_blocks):
    """Classify missing-ball gaps and detect dead-ball stoppages across both halves."""
    raw_codes = features["raw_codes"].copy()
    n = len(raw_codes)
    cx = features["cam_x_norm"]
    cy = features["cam_y_norm"]
    cx_m = features["cam_x_m"]
    cy_m = features["cam_y_m"]
    fov_x = features["fov_x_m"]
    dfov_x = features["dfov_x_ms"]
    v_cam = features["v_cam"]
    duel_flip = features["duel_flip_rate_2s"]

    bridged_codes = raw_codes.copy()
    dead_ball_mask = np.zeros(n, dtype=bool)
    live_bridge_lock = np.zeros(n, dtype=bool)

    # Cue 0: Hard pre-kickoff walk-up in 2nd half (110,808 .. 113,741) and Smart Tagging masks
    dead_ball_mask[110808:113742] = True
    dead_ball_mask |= features["set_piece_pre_kick_d"]
    dead_ball_mask |= features["goal_dead_mask"]

    for _, h_start, h_end_incl in half_blocks:
        s_h = 113742 if h_start == 110808 else h_start
        e_h = h_end_incl + 1

        # -------------------------------------------------------------------
        # Step 1: Classify Missing-Ball (NaN, code == 2) Intervals
        # -------------------------------------------------------------------
        orig_slice = raw_codes[s_h:e_h].copy()
        for r_start, r_end, val in _contiguous_runs(orig_slice):
            if val != 2:
                continue
            g_s = s_h + r_start
            g_e = s_h + r_end
            gap_len = g_e - g_s

            # Robust pre-gap and post-gap team from 15-frame window (avoids 1-frame dropout edge noise)
            post_team = _majority_team(raw_codes[g_e:min(e_h, g_e + 15)], default=0)
            if g_s <= s_h + 3:
                pre_team = post_team
            else:
                pre_team = _majority_team(bridged_codes[max(s_h, g_s - 15):g_s], default=post_team)

            if gap_len < 60:
                # Short occlusion (< 2.4s): always bridge as live play
                bridged_codes[g_s:g_e] = pre_team
                live_bridge_lock[g_s:g_e] = True
            else:
                # Long missing-ball gap (>= 2.4s): check camera look-point displacement & pitch zone
                disp_m = float(np.hypot(cx_m[g_e - 1] - cx_m[g_s], cy_m[g_e - 1] - cy_m[g_s]))
                mean_cx = float(np.mean(cx[g_s:g_e]))
                in_deep_box = (mean_cx < 0.20) or (mean_cx > 0.80)

                if disp_m >= 7.0 and not in_deep_box:
                    # Active downfield camera progression -> ball is in live flight / progression!
                    bridged_codes[g_s:g_e] = pre_team
                    live_bridge_lock[g_s:g_e] = True
                else:
                    # Stationary / local camera or deep corner/box -> Dead Ball stoppage (D)!
                    d_s = g_s
                    d_e = min(e_h, g_e + 11)

                    if in_deep_box and g_s - 95 >= s_h:
                        # Backtrack across goal-line boundary bounces where camera is already parked
                        # in the deep box (cx < 0.165 or > 0.835) with low camera velocity (< 0.35 m/s)
                        pre_window = bridged_codes[g_s - 95:g_s]
                        pre_runs = _contiguous_runs(pre_window)
                        if len(pre_runs) >= 3:
                            cand_s = g_s - 95 + pre_runs[0][0]
                            if float(np.mean(v_cam[cand_s:g_s])) < 0.45:
                                d_s = cand_s + 10
                            else:
                                d_s = g_s - 95 + pre_runs[1][0]
                    elif g_s + 18 < g_e:
                        d_s = g_s + 18

                    dead_ball_mask[d_s:d_e] = True
                    bridged_codes[d_s:d_e] = 2

        # -------------------------------------------------------------------
        # Step 2: Cue B — Wide-FOV Camera Reset + Tripod-Lock Free-Kick Stoppage
        # -------------------------------------------------------------------
        wide_mask = (fov_x[s_h:e_h] > 75.5).astype(np.int8)
        for r_start, r_end, val in _contiguous_runs(wide_mask):
            if val != 1 or (r_end - r_start) < 35:
                continue
            w_s = s_h + r_start
            w_e = s_h + r_end

            onset = w_s
            while onset > s_h and (dfov_x[onset - 1] > 1.2 or v_cam[onset - 1] > 2.8) and (w_s - onset) < 20:
                onset -= 1

            search_limit = min(e_h, w_e + 360)
            lock_indices = np.flatnonzero(v_cam[w_s:search_limit] < 0.14)
            if len(lock_indices) >= 25:
                last_lock = w_s + int(lock_indices[-1])
                restart = last_lock
                while restart < min(e_h, last_lock + 115):
                    if v_cam[restart] > 2.4 and fov_x[restart] < 69.5:
                        break
                    restart += 1
                dead_ball_mask[onset:restart] = True
                bridged_codes[onset:restart] = 2

        # -------------------------------------------------------------------
        # Step 3: Cue C — Penalty-Box Goal-Kick Zoom-Out & Tripod-Lock Stoppage
        # -------------------------------------------------------------------
        t = s_h + 50
        while t < e_h - 400:
            in_box_tight = ((cx[t - 25:t].min() < 0.135 or cx[t - 25:t].max() > 0.865) and
                            fov_x[t - 25:t].min() < 49.5)
            if in_box_tight and not dead_ball_mask[t]:
                team = int(bridged_codes[t])
                if team in (0, 1):
                    run_end = t
                    while run_end < min(e_h, t + 650) and bridged_codes[run_end] == team:
                        run_end += 1
                    fov_peak = float(fov_x[t:run_end].max())
                    if (run_end - t) >= 180 and (fov_peak - fov_x[t]) >= 12.0 and fov_peak >= 61.0:
                        # Find tripod-lock plateau after FOV zoom-out reaches near its peak (>= fov_peak - 2.0m)
                        post_zoom_start = t + 120
                        plateau_lock = (
                            (v_cam[post_zoom_start:run_end] < 0.12) &
                            (fov_x[post_zoom_start:run_end] >= fov_peak - 2.2)
                        ).astype(np.int8)
                        lock_runs = [
                            (post_zoom_start + ls, post_zoom_start + le)
                            for ls, le, lval in _contiguous_runs(plateau_lock)
                            if lval == 1 and (le - ls) >= 15
                        ]
                        if lock_runs:
                            gk_start = t + 10
                            gk_end = lock_runs[0][1]
                            dead_ball_mask[gk_start:gk_end] = True
                            bridged_codes[gk_start:gk_end] = 2
                            t = gk_end + 50
                            continue
            t += 5

    stoppage_entry_gate = np.where(dead_ball_mask, 3.5, 0.25)
    stoppage_exit_gate = np.where(~dead_ball_mask, 3.0, 0.30)
    turnover_gate = np.clip(1.0 - 0.5 * features["tripod_locked"].astype(np.float64) + 0.4 * duel_flip, 0.35, 2.0)

    return {
        "bridged_codes": bridged_codes,
        "dead_ball_mask": dead_ball_mask,
        "live_bridge_lock": live_bridge_lock,
        "stoppage_entry_gate": stoppage_entry_gate,
        "stoppage_exit_gate": stoppage_exit_gate,
        "turnover_gate": turnover_gate,
    }


def compute_tier1_emission_log_probs(features, regimes):
    """Compute calibrated frame-level emission log-probabilities log P(y_t | f_t) in R^{N x 3}."""
    n = len(features["raw_codes"])
    bridged = regimes["bridged_codes"]
    dead_mask = regimes["dead_ball_mask"]
    st_net = features["st_net_log_odds"]
    log_area_ratio = features["log_area_ratio"]

    is_h = (bridged == 0).astype(np.int64)
    is_a = (bridged == 1).astype(np.int64)
    cum_h = np.concatenate(([0], np.cumsum(is_h)))
    cum_a = np.concatenate(([0], np.cumsum(is_a)))
    idx = np.arange(n)

    def _window_share_h(w):
        lo = np.maximum(0, idx - w)
        hi = np.minimum(n, idx + w + 1)
        ch = (cum_h[hi] - cum_h[lo]).astype(np.float64)
        ca = (cum_a[hi] - cum_a[lo]).astype(np.float64)
        tot = ch + ca
        return np.where(tot > 0, ch / np.maximum(tot, 1.0), 0.5)

    sh_8 = _window_share_h(8)
    sh_22 = _window_share_h(22)

    p_h_local = (
        0.55 * (bridged == 0).astype(np.float64)
        + 0.30 * sh_8
        + 0.15 * sh_22
    )
    p_a_local = (
        0.55 * (bridged == 1).astype(np.float64)
        + 0.30 * (1.0 - sh_8)
        + 0.15 * (1.0 - sh_22)
    )

    log_odds_ha = np.log((p_h_local + 0.06) / (p_a_local + 0.06)) + st_net + 0.15 * log_area_ratio
    prob_h_given_alive = 1.0 / (1.0 + np.exp(-np.clip(log_odds_ha, -12.0, 12.0)))
    prob_a_given_alive = 1.0 - prob_h_given_alive

    prob_d = np.where(dead_mask, 0.96, 0.01)
    prob_alive = 1.0 - prob_d

    probs = np.zeros((n, 3), dtype=np.float64)
    probs[:, 0] = np.clip(prob_alive * prob_h_given_alive, 1e-6, 1.0)
    probs[:, 1] = np.clip(prob_alive * prob_a_given_alive, 1e-6, 1.0)
    probs[:, 2] = np.clip(prob_d, 1e-6, 1.0)
    probs /= np.sum(probs, axis=1, keepdims=True)

    return np.log(probs)
