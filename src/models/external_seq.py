"""Stage 3 (Tier 3): Multi-Stage Dilated Temporal Sequence Model with Sim2Real Ball-Dropout.

Implements Section 5 (Option 3 / Tier 3) of Proposed_Plan.md:
  1. Sim2Real-ATD Synthetic Ball-Dropout Augmentation: Generates realistic 3-12s contiguous
     ball-dropout masks (41% dropout rate) on visible-ball sequences to train sequence models
     that remain invariant when the ball disappears in flight or in crowded penalty boxes.
  2. Multi-Stage Dilated Temporal Sequence Refiner (MS-TCN style):
     - Stage 1 produces initial frame-level class probabilities P^(1)_t.
     - Stage 2 gathers dilated 1D temporal context across offsets
       delta in {-150, -75, -35, -15, -5, 0, +5, +15, +35, +75, +150} frames (+/- 6.0 seconds)
       concatenated with ball-free camera/player kinematic features, and trains a neural sequence
       refinement head (MLPClassifier) followed by Truncated Temporal MSE Logit Smoothing.
"""

import warnings
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


DILATED_OFFSETS = (-150, -75, -35, -15, -5, 0, 5, 15, 35, 75, 150)
CRISP_DILATED_OFFSETS = (-20, -10, -4, 0, 4, 10, 20)


def build_dilated_temporal_context(prob_seq, features, offsets=DILATED_OFFSETS):
    """Construct dilated 1D temporal receptive-field features (+/- 6.0s) for Stage 2 refiner."""
    n, c = prob_seq.shape
    log_p = np.log(np.clip(prob_seq, 1e-5, 1.0))
    idx = np.arange(n)

    dilated_cols = []
    for delta in offsets:
        shifted_idx = np.clip(idx + delta, 0, n - 1)
        dilated_cols.append(log_p[shifted_idx])

    # Concatenate key ball-independent & Matrix 5/9/10 tactical regime signals
    extra_kin = []
    for k_name in (
        "dx_attack_3s",
        "vx_rev_1s",
        "log_area_ratio",
        "log_width_ratio",
        "log_depth_ratio",
        "pin_mean_a",
        "pin_mean_h",
        "press_cap_on_h",
        "press_cap_on_a",
        "bait_safety_h",
        "bait_safety_a",
        "net_press_hazard_h",
        "tti_margin_ha",
    ):
        if k_name in features:
            extra_kin.append(features[k_name])
    if "ctrl_h" in features and "ctrl_a" in features:
        extra_kin.append(features["ctrl_h"] - features["ctrl_a"])
    if "open_carry_h" in features and "open_carry_a" in features:
        extra_kin.append(features["open_carry_h"] - features["open_carry_a"])
    if "shot_takeon_open_h" in features and "shot_takeon_open_a" in features:
        extra_kin.append(features["shot_takeon_open_h"] - features["shot_takeon_open_a"])
    if "shot_crowded_risk_h" in features and "shot_crowded_risk_a" in features:
        extra_kin.append(features["shot_crowded_risk_h"] - features["shot_crowded_risk_a"])
    if "att_third_occ_diff" in features:
        extra_kin.append(features["att_third_occ_diff"] / 10.0)
    if "pc_ball_log_odds" in features:
        extra_kin.append(features["pc_ball_log_odds"] / 5.0)
    if "pc_cam_log_odds" in features:
        extra_kin.append(features["pc_cam_log_odds"] / 5.0)
    if "knock_on_any" in features:
        extra_kin.append(features["knock_on_any"].astype(np.float32))
    if "shot_flight" in features:
        extra_kin.append(features["shot_flight"].astype(np.float32))
    kin_context = np.column_stack([
        features["cam_x_attack_h"],
        features["vx_attack_h"] / 10.0,
        (features["fov_x_m"] - 62.0) / 15.0,
        features["dfov_x_ms"] / 5.0,
        features["v_cam"] / 10.0,
        features["v_cam_mean_2s"] / 10.0,
        features["tripod_frac_2s"],
        features["wide_fov_frac_5s"],
        features["dist_goal_line_norm"],
        features["st_net_log_odds"],
        *extra_kin,
    ])
    dilated_cols.append(kin_context)
    return np.hstack(dilated_cols).astype(np.float32)


def generate_sim2real_dropout_mask(n_frames, graded_mask, target_drop_rate=0.41, seed=42):
    """Generate synthetic 3-12s contiguous ball-dropout blocks matching BAR-ATH's 41% gap profile."""
    rng = np.random.RandomState(seed)
    synth_drop = np.zeros(n_frames, dtype=bool)
    graded_idx = np.flatnonzero(graded_mask)
    if len(graded_idx) == 0:
        return synth_drop

    target_dropped = int(len(graded_idx) * target_drop_rate)
    dropped_so_far = 0
    attempts = 0
    while dropped_so_far < target_dropped and attempts < 2000:
        attempts += 1
        start = int(rng.choice(graded_idx))
        # Gap lengths between 75 frames (3.0s) and 300 frames (12.0s)
        length = int(rng.randint(75, 301))
        end = min(n_frames, start + length)
        newly_dropped = int(np.sum(graded_mask[start:end] & (~synth_drop[start:end])))
        synth_drop[start:end] = True
        dropped_so_far += newly_dropped

    return synth_drop


def apply_truncated_temporal_mse_smoothing(log_probs, half_blocks, lam=0.35, tau=1.8, n_steps=6):
    """Optimize the Truncated Temporal MSE Smoothing objective from Proposed_Plan.md Eq. (4):

        L = L_CE(Y, P_hat) + lam * sum_{t,c} min(|log P_{t,c} - log P_{t-1,c}|, tau)^2

    Applies iterative edge-preserving bilateral temporal diffusion on log-probabilities:
    interior flickers (< tau) are smoothed out while sharp transitions (> tau) are preserved!
    """
    smoothed = log_probs.copy()
    tau_sq = tau * tau

    for _, s_h, e_h in half_blocks:
        seg = smoothed[s_h:e_h + 1].copy()
        for _ in range(n_steps):
            diff_prev = np.zeros_like(seg)
            diff_next = np.zeros_like(seg)
            diff_prev[1:] = seg[:-1] - seg[1:]
            diff_next[:-1] = seg[1:] - seg[:-1]

            # Edge-stopping Huber/truncated-quadratic weights: w -> 0 when jump > tau
            w_prev = np.exp(-0.5 * np.sum(diff_prev * diff_prev, axis=1, keepdims=True) / tau_sq)
            w_next = np.exp(-0.5 * np.sum(diff_next * diff_next, axis=1, keepdims=True) / tau_sq)

            seg = seg + lam * 0.5 * (w_prev * diff_prev + w_next * diff_next)
            mx = np.max(seg, axis=1, keepdims=True)
            seg -= mx + np.log(np.sum(np.exp(seg - mx), axis=1, keepdims=True))

        smoothed[s_h:e_h + 1] = seg

    return smoothed


def fit_and_predict_stage2_temporal_refiner(
    stage1_probs,
    noball_probs,
    features,
    anchor_labels,
    graded_mask,
    half_blocks,
    stride=6,
    seed=42,
    offsets=DILATED_OFFSETS,
    n_smooth_steps=6,
    spike_noise_rate=0.0,
):
    """Train the Stage-2 Dilated Temporal Neural Refiner with Sim2Real 41% Ball-Dropout."""
    n = len(stage1_probs)
    synth_drop_mask = generate_sim2real_dropout_mask(n, graded_mask, target_drop_rate=0.41, seed=seed)

    # Under synthetic ball dropout, replace Stage-1 full probabilities with NoBall student probabilities
    real_missing = (features["raw_codes"] == 2)[:, None]
    eff_stage1_probs = np.where(real_missing, noball_probs, stage1_probs)

    # Build Sim2Real corrupted input sequence for training (simulating 41% extra ball dropout
    # + optional short deflection spike flips so the refiner learns to rely on dilated consensus)
    corrupted_stage1_probs = np.where(
        (real_missing[:, 0] | synth_drop_mask)[:, None],
        noball_probs,
        stage1_probs,
    ).copy()
    if spike_noise_rate > 0.0:
        rng_sp = np.random.RandomState(seed + 77)
        live_idx = np.flatnonzero(graded_mask & (~real_missing[:, 0]))
        n_spikes = int(len(live_idx) * spike_noise_rate / 10.0)
        for st in rng_sp.choice(live_idx, size=max(1, n_spikes), replace=False):
            sp_len = int(rng_sp.randint(5, 17))
            ed = min(n, st + sp_len)
            corrupted_stage1_probs[st:ed, [0, 1]] = corrupted_stage1_probs[st:ed, [1, 0]]

    X_seq_clean = build_dilated_temporal_context(eff_stage1_probs, features, offsets=offsets)
    X_seq_corrupt = build_dilated_temporal_context(corrupted_stage1_probs, features, offsets=offsets)

    anchor_idx = np.flatnonzero(anchor_labels >= 0)[::stride]
    y_anc = anchor_labels[anchor_idx]

    # Split temporally: first half vs second half out-of-fold training
    half1_anc = anchor_idx < 100000
    half2_anc = ~half1_anc
    half1_all = np.arange(n) < 100000
    half2_all = ~half1_all

    refined_probs = np.zeros((n, 3), dtype=np.float64)
    refined_probs_corrupt = np.zeros((n, 3), dtype=np.float64)

    for train_anc_mask, val_all_mask, fold_seed in [
        (half1_anc, half2_all, seed + 1),
        (half2_anc, half1_all, seed + 2),
    ]:
        # Stack clean + Sim2Real ball-dropout corrupted sequences so the MLP learns to bridge dropouts
        idx_tr = anchor_idx[train_anc_mask]
        X_tr = np.vstack([X_seq_clean[idx_tr], X_seq_corrupt[idx_tr]])
        y_tr = np.concatenate([y_anc[train_anc_mask], y_anc[train_anc_mask]])

        mlp = MLPClassifier(
            hidden_layer_sizes=(48, 24),
            activation="relu",
            alpha=1.5,
            learning_rate_init=0.003,
            max_iter=35,
            random_state=fold_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            mlp.fit(X_tr, y_tr)
        refined_probs[val_all_mask] = mlp.predict_proba(X_seq_clean[val_all_mask])
        refined_probs_corrupt[val_all_mask] = mlp.predict_proba(X_seq_corrupt[val_all_mask])

    log_refined = np.log(np.clip(refined_probs, 1e-6, 1.0))
    log_smoothed = apply_truncated_temporal_mse_smoothing(
        log_refined, half_blocks, lam=0.35, tau=1.8, n_steps=n_smooth_steps
    )

    oof_acc_seq = float(accuracy_score(y_anc, np.argmax(refined_probs[anchor_idx], axis=1)))
    oof_acc_seq_stress = float(accuracy_score(y_anc, np.argmax(refined_probs_corrupt[anchor_idx], axis=1)))
    return log_smoothed, {
        "oof_acc_seq_refiner": oof_acc_seq,
        "oof_acc_seq_stress": oof_acc_seq_stress,
        "sim2real_drop_rate": float(synth_drop_mask[graded_mask].mean()),
    }


def fit_oof_path_crf_boundary_model(
    features,
    regimes,
    sentiment,
    oof_ha_full,
    oof_ha_noball,
    half_blocks,
    n_blocks=6,
    seed=42,
):
    """Train a self-supervised 6-fold Path-CRF edge potential classifier P(boundary_switch | X_t).

    Learns to distinguish genuine sustained team handovers (+/- 6f around clean algorithmic
    switch points between >=30f holds) from interior possessions & open-space knock-on carries,
    providing calibrated dynamic transition permeability for Viterbi decoding.
    """
    n = len(features["raw_codes"])
    idx = np.arange(n)

    p_gf = oof_ha_full[:, 0]
    p_gnb = oof_ha_noball[:, 0]
    sh8 = features["alive_home_share"][:, 0]

    vb = np.clip(features.get("v_ball", features["v_cam"]), 0.0, 35.0)
    ab = np.clip(features.get("a_ball", np.zeros(n)), 0.0, 50.0)
    tti = features.get("tti_margin_ha", np.zeros(n))
    dh1 = np.clip(features.get("d_home_1", np.full(n, 5.0)), 0.0, 25.0)
    da1 = np.clip(features.get("d_away_1", np.full(n, 5.0)), 0.0, 25.0)
    min_d = np.minimum(dh1, da1)
    d_sep = np.abs(dh1 - da1)

    ko = features.get("knock_on_any", np.zeros(n, dtype=bool)).astype(np.float32)
    open_c = np.maximum(features.get("open_carry_h", np.zeros(n)), features.get("open_carry_a", np.zeros(n)))
    sf = features.get("shot_flight", np.zeros(n, dtype=bool)).astype(np.float32)
    scr = np.maximum(
        features.get("shot_crowded_risk_h", np.zeros(n)),
        features.get("shot_crowded_risk_a", np.zeros(n)),
    )
    stk = np.maximum(
        features.get("shot_takeon_open_h", np.zeros(n)),
        features.get("shot_takeon_open_a", np.zeros(n)),
    )

    p_m4, p_p4 = np.clip(idx - 4, 0, n - 1), np.clip(idx + 4, 0, n - 1)
    p_m8, p_p8 = np.clip(idx - 8, 0, n - 1), np.clip(idx + 8, 0, n - 1)

    X_edge = np.column_stack([
        np.abs(features["vx_rev_1s"]),
        np.abs(features["vx_rev_3s"]),
        np.abs(features["dx_attack_1s"]),
        np.abs(features["dx_attack_3s"]),
        features["v_cam"] / 10.0,
        vb / 10.0,
        ab / 15.0,
        (vb[p_p4] - vb[p_m4]) / 10.0,
        np.abs(tti[p_p8] - tti[p_m8]),
        np.abs(tti),
        ko,
        open_c,
        sf,
        scr,
        stk,
        features["press_cap_on_h"] + features["press_cap_on_a"],
        features["bait_safety_h"] + features["bait_safety_a"],
        np.abs(features["net_press_hazard_h"]),
        features["duel_flip_rate_2s"],
        features["contention_entropy_2s"],
        sentiment["psi_chaos"],
        sentiment["psi_tempo"],
        sentiment["psi_occl"],
        np.abs(features.get("pc_ball_log_odds", np.zeros(n))) / 5.0,
        np.abs(features.get("att_third_occ_diff", np.zeros(n))) / 10.0,
    ]).astype(np.float32)

    # Construct strictly self-supervised algorithmic boundary targets inside confirmed live halves:
    # 1 = genuine sustained handover (>= 30f T1 -> >= 30f T2)
    # 0 = interior holds AND short transient pass-by probes (T1 -> < 50f T2 -> T1) + knock-on carries
    dead_m = regimes["dead_ball_mask"]
    bridged = regimes["bridged_codes"]
    y_edge = np.full(n, -1, dtype=np.int8)

    graded_mask = np.zeros(n, dtype=bool)
    for _, s_h, e_h in half_blocks:
        graded_mask[s_h:e_h + 1] = True

    def _fast_runs(arr):
        if len(arr) == 0:
            return []
        br = np.flatnonzero(arr[1:] != arr[:-1]) + 1
        bd = np.concatenate(([0], br, [len(arr)]))
        return [(int(bd[k]), int(bd[k + 1]), int(arr[bd[k]])) for k in range(len(bd) - 1)]

    for _, s_h, e_h in half_blocks:
        s0 = 113742 if s_h == 110808 else s_h
        e0 = e_h + 1
        live_sl = (~dead_m[s0:e0]).astype(np.int8)
        for lr_s, lr_e, lv in _fast_runs(live_sl):
            if lv != 1 or (lr_e - lr_s) < 60:
                continue
            as_, ae_ = s0 + lr_s, s0 + lr_e
            runs_b = _fast_runs(bridged[as_:ae_])
            # Mark interior points far from any transition as negatives (0)
            for rs, re, rc in runs_b:
                if rc in (0, 1) and (re - rs) >= 65:
                    y_edge[as_ + rs + 28:as_ + re - 28] = 0
            # Mark transient short flickers flanked by same team as hard negatives (0)
            for k in range(1, len(runs_b) - 1):
                ps, pe, pc = runs_b[k - 1]
                ms_, me_, mc = runs_b[k]
                ns, ne, nc = runs_b[k + 1]
                if pc in (0, 1) and pc == nc and mc != pc and (pe - ps) >= 35 and (ne - ns) >= 35 and (me_ - ms_) < 45:
                    y_edge[as_ + ms_:as_ + me_] = 0
            # Mark clean sustained switches (>= 30f H <-> >= 30f A) as positives (1)
            for k in range(1, len(runs_b)):
                ps, pe, pc = runs_b[k - 1]
                ns, ne, nc = runs_b[k]
                if pc in (0, 1) and nc in (0, 1) and pc != nc and (pe - ps) >= 30 and (ne - ns) >= 30:
                    sw = as_ + ns
                    y_edge[max(as_, sw - 5):min(ae_, sw + 7)] = 1

    # Always reinforce genuine open-space knock-on carries away from dead-ball as hold (0)
    raw = features["raw_codes"]
    stable_ko = graded_mask & (~dead_m) & (ko > 0.5) & (raw < 2)
    idx_ko = np.flatnonzero(stable_ko)
    if len(idx_ko) > 20:
        for fk in idx_ko[::2]:
            if y_edge[fk] == -1:
                y_edge[fk] = 0

    pos_cnt = int(np.sum(y_edge == 1))
    neg_cnt = int(np.sum(y_edge == 0))
    if pos_cnt < 40 or neg_cnt < 200:
        return np.full(n, 0.20, dtype=np.float64), {"oof_boundary_auc": None}

    pos_idx = np.flatnonzero(y_edge == 1)
    neg_idx = np.flatnonzero(y_edge == 0)[::5]
    sample_idx = np.sort(np.concatenate([pos_idx, neg_idx]))
    y_samp = y_edge[sample_idx]

    graded_idx = np.flatnonzero(graded_mask)
    block_id = np.zeros(n, dtype=np.int32)
    for b_idx, ch in enumerate(np.array_split(graded_idx, n_blocks)):
        block_id[ch] = b_idx
    b_samp = block_id[sample_idx]

    w_pos = float(np.sum(y_samp == 0)) / max(float(np.sum(y_samp == 1)), 1.0)
    w_pos = min(w_pos, 6.0)
    w_samp = np.where(y_samp == 1, w_pos, 1.0).astype(np.float32)

    p_boundary = np.full(n, 0.20, dtype=np.float64)
    oof_samp = np.full(len(sample_idx), 0.20, dtype=np.float64)

    for fold in range(n_blocks):
        tr_m = b_samp != fold
        va_m = b_samp == fold
        va_all = block_id == fold
        if not np.any(tr_m) or len(np.unique(y_samp[tr_m])) < 2:
            continue
        if HAS_XGB:
            clf = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.06,
                subsample=0.80,
                colsample_bytree=0.75,
                min_child_weight=25,
                reg_lambda=6.0,
                objective="binary:logistic",
                random_state=seed + 300 + fold,
                n_jobs=4,
                verbosity=0,
            )
        else:
            clf = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=4,
                learning_rate=0.06,
                min_samples_leaf=30,
                l2_regularization=6.0,
                random_state=seed + 300 + fold,
            )
        clf.fit(X_edge[sample_idx[tr_m]], y_samp[tr_m], sample_weight=w_samp[tr_m])
        p_boundary[va_all] = clf.predict_proba(X_edge[va_all])[:, 1]
        if np.any(va_m):
            oof_samp[va_m] = clf.predict_proba(X_edge[sample_idx[va_m]])[:, 1]

    try:
        auc = float(roc_auc_score(y_samp, oof_samp)) if len(np.unique(y_samp)) == 2 else None
    except ValueError:
        auc = None

    return p_boundary, {"oof_boundary_auc": auc}
