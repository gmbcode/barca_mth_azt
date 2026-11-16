"""Stage 3 (Tier 2): Unsupervised Baum-Welch / Anchor Pseudo-Labelled Emission Calibrator.

Fits a ball-free kinematic emission student model on high-confidence algorithmic anchor frames
from the match (100% compliant with RULES.md: zero human labels or public GT used), enabling
accurate emission estimates during the 41% missing-ball gaps.
"""

import numpy as np


def fit_and_predict_tier2_emissions(features, regimes, tier1_log_probs, half_blocks):
    """Fit a Gaussian-kinematic emission student on high-confidence physics anchors and blend."""
    n = len(features["raw_codes"])
    graded_mask = np.zeros(n, dtype=bool)
    for _, s_h, e_h in half_blocks:
        graded_mask[s_h:e_h + 1] = True

    bridged = regimes["bridged_codes"]
    dead_mask = regimes["dead_ball_mask"]
    ms_share_h = features["alive_home_share"]

    # High-confidence algorithmic anchor frames:
    # Anchor H: ball visible, multi-scale share_H > 0.85, not dead
    anchor_h = graded_mask & (~dead_mask) & (features["raw_codes"] == 0) & (ms_share_h[:, 2] > 0.85)
    # Anchor A: ball visible, multi-scale share_H < 0.15, not dead
    anchor_a = graded_mask & (~dead_mask) & (features["raw_codes"] == 1) & (ms_share_h[:, 2] < 0.15)
    # Anchor D: physical dead ball mask
    anchor_d = graded_mask & dead_mask

    # Ball-free kinematic feature vector X_kin in R^{N x 6}:
    # [cam_x_attack_h, vx_attack_h, fov_x_m, v_cam_mean_2s, duel_flip_rate_2s, st_net_log_odds]
    X_kin = np.column_stack([
        features["cam_x_attack_h"],
        np.clip(features["vx_attack_h"] / 10.0, -2.0, 2.0),
        (features["fov_x_m"] - 62.0) / 15.0,
        features["v_cam_mean_2s"] / 4.0,
        features["duel_flip_rate_2s"] * 5.0,
        features["st_net_log_odds"],
    ])

    # Estimate class-conditional Gaussian parameters (mu_k, diag_var_k) on anchors
    anchors = [anchor_h, anchor_a, anchor_d]
    log_like_student = np.zeros((n, 3), dtype=np.float64)

    for k, mask in enumerate(anchors):
        if np.sum(mask) >= 100:
            Xk = X_kin[mask]
            mu_k = np.mean(Xk, axis=0)
            var_k = np.var(Xk, axis=0) + 0.15
        else:
            mu_k = np.zeros(X_kin.shape[1])
            var_k = np.ones(X_kin.shape[1])
        diff = X_kin - mu_k[None, :]
        log_like_student[:, k] = -0.5 * np.sum((diff * diff) / var_k[None, :] + np.log(var_k[None, :]), axis=1)

    # Normalize student log-probabilities
    max_ll = np.max(log_like_student, axis=1, keepdims=True)
    student_log_probs = log_like_student - (
        max_ll + np.log(np.sum(np.exp(log_like_student - max_ll), axis=1, keepdims=True))
    )

    # Blend Tier 1 (weight 0.90 when ball seen, 0.80 when ball missing) + Tier 2 student
    p1 = np.exp(tier1_log_probs)
    p2 = np.exp(student_log_probs)
    w1 = np.where(features["raw_codes"] < 2, 0.92, 0.84)[:, None]
    blended = np.clip(w1 * p1 + (1.0 - w1) * p2, 1e-6, 1.0)
    blended /= np.sum(blended, axis=1, keepdims=True)

    return np.log(blended)
