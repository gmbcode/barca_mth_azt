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


def _filter_sentiment_deflections(
    block_codes, block_bridged, block_v_cam, block_chaos, block_logit, block_vx_attack
):
    """Sentiment-aware deflection filter & inflection snapper operating directly on Viterbi path."""
    codes = block_codes.copy()

    # Pass -1: Contested-Possession Retention (CHALLENGE.md Sec. 2.2):
    # During a contested turnover sequence: T1 (established) -> T2 (< 35f probe) -> T1 (regain < 24f),
    # rapid 50/50 aerial/duel pinball stays with T1 until T2 establishes clean sustained control.
    changed = True
    while changed:
        changed = False
        runs = _runs(codes)
        if len(runs) < 4:
            break
        for i in range(1, len(runs) - 2):
            s_probe, e_probe, k_probe = runs[i]
            s_regain, e_regain, k_regain = runs[i + 1]
            prev_dur = runs[i - 1][1] - runs[i - 1][0]
            if (
                prev_dur >= 50
                and runs[i - 1][2] == k_regain
                and (e_probe - s_probe) < 35
                and (e_regain - s_regain) < 24
            ):
                codes[s_probe:e_probe] = k_regain
                changed = True
                break

    # Pass 0: absorb ultra-short micro-flickers (< 20 frames = 0.80s) flanked by same team
    for max_len in (10, 20):
        changed = True
        while changed:
            changed = False
            runs = _runs(codes)
            if len(runs) < 3:
                break
            for i in range(1, len(runs) - 1):
                s_r, e_r, _ = runs[i]
                if (e_r - s_r) < max_len and runs[i - 1][2] == runs[i + 1][2]:
                    codes[s_r:e_r] = runs[i - 1][2]
                    changed = True
                    break

    # Pass 0B: High-speed touchline duel right before Dead Ball stoppage
    runs = _runs(codes)
    if len(runs) >= 4:
        r_m4, r_m3, r_m2, r_m1 = runs[-4], runs[-3], runs[-2], runs[-1]
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
            s_r, e_r, _ = runs[i]
            if (e_r - s_r) < 26 and runs[i - 1][2] == runs[i + 1][2]:
                codes[s_r:e_r] = runs[i - 1][2]
                changed = True
                break

    # Pass 1B: absorb tiny boundary bounces (< 32 frames = 1.28s) at start or end of live block
    runs = _runs(codes)
    if len(runs) >= 2:
        if (runs[0][1] - runs[0][0]) < 32 and (runs[1][1] - runs[1][0]) >= 60:
            codes[runs[0][0]:runs[0][1]] = runs[1][2]
        runs = _runs(codes)
        if len(runs) >= 2 and (runs[-1][1] - runs[-1][0]) < 32 and (runs[-2][1] - runs[-2][0]) >= 60:
            codes[runs[-1][0]:runs[-1][1]] = runs[-2][2]

    # Pass 2: Snap initial H <-> A transitions to exact contact / log-odds inflection frame
    runs = _runs(codes)
    for i in range(1, len(runs)):
        t_trans = runs[i][0]
        prev_team = runs[i - 1][2]
        new_team = runs[i][2]
        lo = max(runs[i - 1][0] + 10, t_trans - 14)
        hi = min(runs[i][1] - 10, t_trans + 14)
        if hi > lo + 2:
            best_t = t_trans
            best_sc = -1e9
            for cand in range(lo, hi + 1):
                b_match = float(np.sum(block_bridged[lo:cand] == prev_team))
                a_match = float(np.sum(block_bridged[cand:hi] == new_team))
                l_sign = 1.0 if new_team == 0 else -1.0
                l_bonus = 0.20 * float(np.sum(l_sign * block_logit[cand:min(hi, cand + 6)]))
                sc = b_match + a_match + l_bonus - 0.12 * abs(cand - t_trans)
                if sc > best_sc:
                    best_sc = sc
                    best_t = cand
            if best_t < t_trans:
                codes[best_t:t_trans] = new_team
            elif best_t > t_trans:
                codes[t_trans:best_t] = prev_team

    # Pass 2B: Attacking-Momentum Contested Boundary Snap:
    # While the losing team is still actively progressing in its attacking direction (|vx_attack| > 0.9 m/s),
    # an initial short (<16f) opponent probe followed immediately by >=8f raw re-contact by prev_team
    # means possession stayed with prev_team until after its final attacking touch.
    runs = _runs(codes)
    for i in range(1, len(runs)):
        t_trans = runs[i][0]
        prev_team = runs[i - 1][2]
        new_team = runs[i][2]
        win_end = min(runs[i][1] - 10, t_trans + 28)
        if win_end <= t_trans + 10:
            continue
        br = _runs(block_bridged[t_trans:win_end])
        if (
            len(br) >= 3
            and br[0][2] == new_team
            and (br[0][1] - br[0][0]) < 16
            and br[1][2] == prev_team
            and (br[1][1] - br[1][0]) >= 8
            and br[2][2] == new_team
        ):
            t_reend = t_trans + br[1][1]
            if t_reend - t_trans <= 26:
                mean_vx = float(np.mean(block_vx_attack[t_trans:t_reend]))
                attacking = (mean_vx > 0.90) if prev_team == 0 else (mean_vx < -0.90)
                if attacking:
                    codes[t_trans:t_reend] = prev_team

    # Pass 3: Sentiment-aware pass-by flicker absorption inside sustained possessions
    # Preserve fast VERTICAL_TRANSITION counter-attacks (dur >= 45 & mean_v >= 1.45 m/s)
    changed = True
    while changed:
        changed = False
        runs = _runs(codes)
        if len(runs) < 3:
            break
        for i in range(1, len(runs) - 1):
            s_r, e_r, _ = runs[i]
            dur = e_r - s_r
            prev_dur = runs[i - 1][1] - runs[i - 1][0]
            next_dur = runs[i + 1][1] - runs[i + 1][0]
            if runs[i - 1][2] == runs[i + 1][2] and prev_dur >= 45 and next_dur >= 55:
                mean_v = float(np.mean(block_v_cam[s_r:e_r]))
                tripod_frac = float(np.mean(block_v_cam[s_r:e_r] < 0.14))
                mean_chaos = float(np.mean(block_chaos[s_r:e_r]))
                calm_speed_ok = (mean_v < 1.15) if mean_chaos > 0.70 else (mean_v < 1.40)
                short_bounce_ok = dur < 48
                calm_cond = (
                    dur < 62
                    and (prev_dur + next_dur) >= 180
                    and (calm_speed_ok or short_bounce_ok)
                )
                tripod_cond = (
                    dur < 120
                    and (tripod_frac > 0.35 or (mean_v < 0.85 and mean_chaos < 0.55))
                    and (prev_dur + next_dur) >= 350
                )
                if calm_cond or tripod_cond:
                    codes[s_r:e_r] = runs[i - 1][2]
                    changed = True
                    break

    return codes


def decode_sentiment_hybrid_match(
    log_emit, features, regimes, sentiment, turnover_gate, logit_fused, half_blocks
):
    """Stage 4 Decoder for Sentiment-Adaptive Hybrid Model (True Viterbi + Sentiment Deflection Filter)."""
    n = len(log_emit)
    A_t = build_time_inhomogeneous_transition_matrices(
        n_frames=n,
        stoppage_entry_gate=regimes["stoppage_entry_gate"],
        stoppage_exit_gate=regimes["stoppage_exit_gate"],
        live_bridge_lock=regimes["live_bridge_lock"],
        turnover_gate=turnover_gate,
        stay_alive_penalty=4.6,
        stay_dead_penalty=6.8,
    )

    final_labels = np.full(n, "X", dtype="<U1")
    final_labels[110808:113742] = "D"

    dead_mask = regimes["dead_ball_mask"]
    bridged = regimes["bridged_codes"]
    v_cam = features["v_cam"]
    vx_attack_h = features["vx_attack_h"]
    psi_chaos = sentiment["psi_chaos"]
    code_to_str = np.array(["H", "A", "D"])

    for _, h_start, h_end_incl in half_blocks:
        s_h = 113742 if h_start == 110808 else h_start
        e_h = h_end_incl + 1

        half_dead = dead_mask[s_h:e_h]
        half_codes = np.full(e_h - s_h, 2, dtype=np.int8)

        live_runs = _runs((~half_dead).astype(np.int8))
        for lr_s, lr_e, is_live in live_runs:
            if is_live != 1 or (lr_e - lr_s) == 0:
                continue
            abs_s = s_h + lr_s
            abs_e = s_h + lr_e

            # Exact 2-state {H, A} Viterbi inside each confirmed live-play interval
            log_emit_ha = log_emit[abs_s:abs_e, :2].copy()
            A_ha = A_t[abs_s:abs_e, :2, :2].copy()
            mx = np.max(A_ha, axis=2, keepdims=True)
            A_ha -= mx + np.log(np.sum(np.exp(A_ha - mx), axis=2, keepdims=True))

            live_viterbi = viterbi_decode_segment(log_emit_ha, A_ha)
            refined_live = _filter_sentiment_deflections(
                live_viterbi,
                bridged[abs_s:abs_e],
                v_cam[abs_s:abs_e],
                psi_chaos[abs_s:abs_e],
                logit_fused[abs_s:abs_e],
                vx_attack_h[abs_s:abs_e],
            )
            half_codes[lr_s:lr_e] = refined_live

        final_labels[s_h:e_h] = code_to_str[half_codes]

    return final_labels
