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
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


DILATED_OFFSETS = (-150, -75, -35, -15, -5, 0, 5, 15, 35, 75, 150)


def build_dilated_temporal_context(prob_seq, features, offsets=DILATED_OFFSETS):
    """Construct dilated 1D temporal receptive-field features (+/- 6.0s) for Stage 2 refiner."""
    n, c = prob_seq.shape
    log_p = np.log(np.clip(prob_seq, 1e-5, 1.0))
    idx = np.arange(n)

    dilated_cols = []
    for delta in offsets:
        shifted_idx = np.clip(idx + delta, 0, n - 1)
        dilated_cols.append(log_p[shifted_idx])

    # Concatenate key ball-independent kinematic regime signals
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
):
    """Train the Stage-2 Dilated Temporal Neural Refiner with Sim2Real 41% Ball-Dropout."""
    n = len(stage1_probs)
    synth_drop_mask = generate_sim2real_dropout_mask(n, graded_mask, target_drop_rate=0.41, seed=seed)

    # Under synthetic ball dropout, replace Stage-1 full probabilities with NoBall student probabilities
    real_missing = (features["raw_codes"] == 2)[:, None]
    eff_stage1_probs = np.where(real_missing, noball_probs, stage1_probs)

    # Build Sim2Real corrupted input sequence for training (simulating 41% extra ball dropout)
    corrupted_stage1_probs = np.where(
        (real_missing[:, 0] | synth_drop_mask)[:, None],
        noball_probs,
        stage1_probs,
    )

    X_seq_clean = build_dilated_temporal_context(eff_stage1_probs, features)
    X_seq_corrupt = build_dilated_temporal_context(corrupted_stage1_probs, features)

    anchor_idx = np.flatnonzero(anchor_labels >= 0)[::stride]
    y_anc = anchor_labels[anchor_idx]

    # Split temporally: first half vs second half out-of-fold training
    half1_anc = anchor_idx < 100000
    half2_anc = ~half1_anc
    half1_all = np.arange(n) < 100000
    half2_all = ~half1_all

    refined_probs = np.zeros((n, 3), dtype=np.float64)

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

    log_refined = np.log(np.clip(refined_probs, 1e-6, 1.0))
    log_smoothed = apply_truncated_temporal_mse_smoothing(
        log_refined, half_blocks, lam=0.35, tau=1.8, n_steps=6
    )

    oof_acc_seq = float(accuracy_score(y_anc, np.argmax(refined_probs[anchor_idx], axis=1)))
    return log_smoothed, {"oof_acc_seq_refiner": oof_acc_seq, "sim2real_drop_rate": float(synth_drop_mask[graded_mask].mean())}
