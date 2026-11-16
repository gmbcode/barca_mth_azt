"""Stage 4: Time-Inhomogeneous Viterbi/HSMM Decoder, Deflection Filter & Boundary Snapper."""

import numpy as np
from src.physics_matrices import build_time_inhomogeneous_transition_matrices


def _runs(arr):
    """Return list of [start, stop, val] for 1D array."""
    if len(arr) == 0:
        return []
    breaks = np.flatnonzero(arr[1:] != arr[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(arr)]))
    return [[int(bounds[i]), int(bounds[i + 1]), int(arr[bounds[i]])]
            for i in range(len(bounds) - 1)]


def viterbi_decode_segment(log_emit, A_t):
    """Run exact time-inhomogeneous Viterbi decoding over a sequence of length T."""
    T, C = log_emit.shape
    dp = np.zeros((T, C), dtype=np.float64)
    back = np.zeros((T, C), dtype=np.int8)

    dp[0] = log_emit[0]
    for t in range(1, T):
        scores = dp[t - 1][:, None] + A_t[t]
        back[t] = np.argmax(scores, axis=0)
        dp[t] = log_emit[t] + scores[back[t], np.arange(C)]

    path = np.zeros(T, dtype=np.int8)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def _filter_live_block_deflections(block_codes, block_bridged, block_v_cam):
    """Suppress pass-in-flight pass-bys, boundary bounces, and brief deflections inside a live block."""
    codes = np.where(block_bridged < 2, block_bridged, block_codes).copy()

    # Pass 0: absorb ultra-short micro-flickers (< 16 frames = 0.64s) flanked by same team
    for max_len in (8, 16):
        changed = True
        while changed:
            changed = False
            runs = _runs(codes)
            if len(runs) < 3:
                break
            for i in range(1, len(runs) - 1):
                s_r, e_r, k_r = runs[i]
                if (e_r - s_r) < max_len and runs[i - 1][2] == runs[i + 1][2]:
                    codes[s_r:e_r] = runs[i - 1][2]
                    changed = True
                    break

    # Pass 0B: High-speed touchline duel right before Dead Ball stoppage
    # If the final ~120 frames before D contain a rapid touchline scramble (v_cam > 2.0 m/s)
    # immediately following a >= 75-frame counter-attack transition, extend the counter-attack to D
    runs = _runs(codes)
    if len(runs) >= 4:
        # Check if last 3 runs are (A, H, A, H) where A started a fast counter-attack
        r_m3, r_m2, r_m1 = runs[-3], runs[-2], runs[-1]
        r_m4 = runs[-4]
        if (r_m4[2] == r_m2[2]) and ((r_m4[1] - r_m4[0]) >= 75) and ((r_m1[1] - r_m3[0]) <= 130):
            if float(np.mean(block_v_cam[r_m4[0]:r_m1[1]])) > 2.0:
                codes[r_m3[0]:r_m1[1]] = r_m4[2]

    # Pass 1: absorb short deflections (< 26 frames = 1.04s) flanked by same team
    changed = True
    while changed:
        changed = False
        runs = _runs(codes)
        if len(runs) < 3:
            break
        for i in range(1, len(runs) - 1):
            s_r, e_r, k_r = runs[i]
            if (e_r - s_r) < 26 and runs[i - 1][2] == runs[i + 1][2]:
                codes[s_r:e_r] = runs[i - 1][2]
                changed = True
                break

    # Pass 1B: absorb short boundary bounces (< 35 frames = 1.40s) at the start or end of a live block
    runs = _runs(codes)
    if len(runs) >= 2:
        if (runs[0][1] - runs[0][0]) < 35 and (runs[1][1] - runs[1][0]) >= 60:
            codes[runs[0][0]:runs[0][1]] = runs[1][2]
        runs = _runs(codes)
        if len(runs) >= 2 and (runs[-1][1] - runs[-1][0]) < 35 and (runs[-2][1] - runs[-2][0]) >= 60:
            codes[runs[-1][0]:runs[-1][1]] = runs[-2][2]

    # Pass 2: absorb pass-by flickers inside sustained possessions
    changed = True
    while changed:
        changed = False
        runs = _runs(codes)
        if len(runs) < 3:
            break
        for i in range(1, len(runs) - 1):
            s_r, e_r, k_r = runs[i]
            dur = e_r - s_r
            prev_dur = runs[i - 1][1] - runs[i - 1][0]
            next_dur = runs[i + 1][1] - runs[i + 1][0]
            if runs[i - 1][2] == runs[i + 1][2] and prev_dur >= 50 and next_dur >= 65:
                mean_v = float(np.mean(block_v_cam[s_r:e_r]))
                tripod_frac = float(np.mean(block_v_cam[s_r:e_r] < 0.14))
                if (dur < 58 and (prev_dur + next_dur) >= 190) or (
                    dur < 115 and (tripod_frac > 0.35 or mean_v < 0.85) and (prev_dur + next_dur) >= 350
                ):
                    codes[s_r:e_r] = runs[i - 1][2]
                    changed = True
                    break

    return codes


def decode_full_match(log_emit, features, regimes, half_blocks):
    """Run Stage 4 Time-Inhomogeneous Viterbi decoding, deflection filtering, and boundary snapping."""
    n = len(log_emit)
    A_t = build_time_inhomogeneous_transition_matrices(
        n_frames=n,
        stoppage_entry_gate=regimes["stoppage_entry_gate"],
        stoppage_exit_gate=regimes["stoppage_exit_gate"],
        live_bridge_lock=regimes["live_bridge_lock"],
        turnover_gate=regimes["turnover_gate"],
        stay_alive_penalty=4.2,
        stay_dead_penalty=6.8,
    )

    final_labels = np.full(n, "X", dtype="<U1")
    final_labels[110808:113742] = "D"

    dead_mask = regimes["dead_ball_mask"]
    bridged = regimes["bridged_codes"]
    v_cam = features["v_cam"]
    code_to_str = np.array(["H", "A", "D"])

    for _, h_start, h_end_incl in half_blocks:
        s_h = 113742 if h_start == 110808 else h_start
        e_h = h_end_incl + 1

        half_codes = viterbi_decode_segment(log_emit[s_h:e_h], A_t[s_h:e_h])
        half_dead = dead_mask[s_h:e_h]
        half_codes[half_dead] = 2

        live_runs = _runs((~half_dead).astype(np.int8))
        for lr_s, lr_e, is_live in live_runs:
            if is_live != 1 or (lr_e - lr_s) == 0:
                continue
            abs_s = s_h + lr_s
            abs_e = s_h + lr_e

            refined_live = _filter_live_block_deflections(
                half_codes[lr_s:lr_e], bridged[abs_s:abs_e], v_cam[abs_s:abs_e]
            )
            half_codes[lr_s:lr_e] = refined_live

        final_labels[s_h:e_h] = code_to_str[half_codes]

    return final_labels
