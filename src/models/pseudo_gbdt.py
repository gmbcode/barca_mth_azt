"""Stage 3 (Tier 2): Baum-Welch EM HSMM (Model 2A) & Ball-Dropout Pseudo-Labelled GBDT (Model 2B).

Implements Section 5 (Option 2 / Tier 2) of Proposed_Plan.md in full detail:
  - Model 2A: Unsupervised Baum-Welch EM over factored Ball + Kinematic + Smart Tagging heads:
        p(f_t | z_t) = p_kin(f_kin,t | z_t) * p_ST(f_ST,t | z_t) * [p_ball(f_ball,t | z_t)]^{I[ball_seen]}
  - Model 2B: Out-of-Fold Temporal Block Cross-Validated Gradient Boosted Trees (XGBoost / HistGBDT):
      1. GBDT_Full (88 features): Trained on high-confidence algorithmic anchor frames when ball is seen.
      2. GBDT_NoBall (46 ball-free features, Masked-Ball Student): Trained on the exact same anchors
         with all 42 ball features dropped out, predicting (P(H), P(A), P(D)) purely from the 46
         ball-independent Homography Camera-Kinematics, Jacobian, and Collective Player features!
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def _select_algorithmic_anchors(features, regimes, half_blocks):
    """Identify High-Confidence Algorithmic Anchor Frames (100% rule-compliant, zero human GT)."""
    n = len(features["raw_codes"])
    graded_mask = np.zeros(n, dtype=bool)
    for _, s_h, e_h in half_blocks:
        graded_mask[s_h:e_h + 1] = True

    raw = features["raw_codes"]
    bridged = regimes["bridged_codes"]
    dead_mask = regimes["dead_ball_mask"]
    sh_c = features["alive_home_share"]  # (N, 5) over scales (8, 20, 40, 75, 150)

    # Anchor H: ball visible (raw==0), bridged==0, multi-scale share_H > 0.82 at both 1.6s and 3.0s
    anchor_h = (
        graded_mask
        & (~dead_mask)
        & (raw == 0)
        & (bridged == 0)
        & (sh_c[:, 1] > 0.84)
        & (sh_c[:, 2] > 0.80)
    )
    # Anchor A: symmetric for Away
    anchor_a = (
        graded_mask
        & (~dead_mask)
        & (raw == 1)
        & (bridged == 1)
        & (sh_c[:, 1] < 0.16)
        & (sh_c[:, 2] < 0.20)
    )
    # Anchor D: confirmed physical dead-ball stoppage frames
    anchor_d = graded_mask & dead_mask

    anchor_labels = np.full(n, -1, dtype=np.int8)
    anchor_labels[anchor_h] = 0
    anchor_labels[anchor_a] = 1
    anchor_labels[anchor_d] = 2

    return anchor_labels, graded_mask


def fit_baum_welch_em(features, regimes, tier1_log_probs, graded_mask, n_iter=4):
    """Model 2A: Factored Baum-Welch Expectation-Maximization over Ball & Ball-Free Heads."""
    X_full = features["X_full"]
    ball_idx = features["ball_feature_indices"]
    noball_idx = features["noball_feature_indices"]
    ball_seen = (features["raw_codes"] < 2) & (~regimes["dead_ball_mask"])

    X_ball = X_full[:, ball_idx].astype(np.float64)
    X_kin = X_full[:, noball_idx].astype(np.float64)
    n, d_ball = X_ball.shape
    _, d_kin = X_kin.shape

    # Initialize posterior responsibilities gamma_t(k) from Tier 1 physics posteriors
    gamma = np.exp(tier1_log_probs).copy()

    mu_ball = np.zeros((3, d_ball), dtype=np.float64)
    var_ball = np.ones((3, d_ball), dtype=np.float64)
    mu_kin = np.zeros((3, d_kin), dtype=np.float64)
    var_kin = np.ones((3, d_kin), dtype=np.float64)
    pi_k = np.array([0.44, 0.30, 0.26], dtype=np.float64)

    g_idx = np.flatnonzero(graded_mask)
    g_seen_idx = np.flatnonzero(graded_mask & ball_seen)

    for _ in range(n_iter):
        # M-step: update class priors, kinematic Gaussian parameters (all graded frames),
        # and ball Gaussian parameters (only visible-ball graded frames)
        for k in range(3):
            wk_kin = gamma[g_idx, k] + 1e-6
            w_sum_kin = np.sum(wk_kin)
            pi_k[k] = w_sum_kin / len(g_idx)
            mu_kin[k] = np.sum(X_kin[g_idx] * wk_kin[:, None], axis=0) / w_sum_kin
            diff_kin = X_kin[g_idx] - mu_kin[k][None, :]
            var_kin[k] = np.sum((diff_kin * diff_kin) * wk_kin[:, None], axis=0) / w_sum_kin + 0.12

            wk_b = gamma[g_seen_idx, k] + 1e-6
            w_sum_b = np.sum(wk_b)
            mu_ball[k] = np.sum(X_ball[g_seen_idx] * wk_b[:, None], axis=0) / w_sum_b
            diff_b = X_ball[g_seen_idx] - mu_ball[k][None, :]
            var_ball[k] = np.sum((diff_b * diff_b) * wk_b[:, None], axis=0) / w_sum_b + 0.12

        # E-step: evaluate factored log-likelihoods across all frames
        log_ll = np.zeros((n, 3), dtype=np.float64)
        for k in range(3):
            dk = X_kin - mu_kin[k][None, :]
            ll_kin = -0.5 * np.mean((dk * dk) / var_kin[k][None, :] + np.log(var_kin[k][None, :]), axis=1)

            db = X_ball - mu_ball[k][None, :]
            ll_ball = -0.5 * np.mean((db * db) / var_ball[k][None, :] + np.log(var_ball[k][None, :]), axis=1)

            # Factored emission: ball head gated by I[ball_seen_t]
            log_ll[:, k] = np.log(pi_k[k] + 1e-6) + 1.5 * ll_kin + np.where(ball_seen, 2.5 * ll_ball, 0.0)

        # Combine with Tier 1 physical regime anchor prior for stable EM convergence
        joint_ll = 0.65 * tier1_log_probs + 0.35 * log_ll
        mx = np.max(joint_ll, axis=1, keepdims=True)
        log_post = joint_ll - (mx + np.log(np.sum(np.exp(joint_ll - mx), axis=1, keepdims=True)))
        gamma = np.exp(log_post)

    return log_post


def _build_gbdt_model(seed=42, is_noball=False):
    """Instantiate XGBClassifier (or HistGradientBoostingClassifier) with anti-overfit regularization."""
    if HAS_XGB:
        return XGBClassifier(
            n_estimators=140 if not is_noball else 160,
            max_depth=4 if not is_noball else 5,
            learning_rate=0.07,
            subsample=0.80,
            colsample_bytree=0.75,
            min_child_weight=25,
            reg_lambda=3.0,
            reg_alpha=0.5,
            objective="multi:softprob",
            num_class=3,
            random_state=seed,
            n_jobs=4,
            verbosity=0,
        )
    return HistGradientBoostingClassifier(
        max_iter=140,
        max_depth=5,
        learning_rate=0.07,
        min_samples_leaf=40,
        l2_regularization=3.0,
        random_state=seed,
    )


def fit_ball_dropout_gbdt_models(features, anchor_labels, graded_mask, n_blocks=6, stride=4, seed=42):
    """Model 2B: Fit GBDT_Full (88 features) and GBDT_NoBall (46 ball-free features) via 6-fold temporal CV."""
    X_full = features["X_full"]
    noball_idx = features["noball_feature_indices"]
    X_noball = X_full[:, noball_idx]
    n = len(X_full)

    # Assign every frame to one of `n_blocks` contiguous temporal match blocks (~17.6 mins each)
    graded_indices = np.flatnonzero(graded_mask)
    block_id = np.zeros(n, dtype=np.int32)
    chunks = np.array_split(graded_indices, n_blocks)
    for b_idx, ch in enumerate(chunks):
        block_id[ch] = b_idx

    # Subsample high-confidence anchors on a regular stride for fast, balanced training
    anchor_idx = np.flatnonzero(anchor_labels >= 0)[::stride]
    y_anc = anchor_labels[anchor_idx]
    b_anc = block_id[anchor_idx]

    # Inverse-frequency class weights so Dead Ball (D) and Away (A) receive equal representation
    classes, counts = np.unique(y_anc, return_counts=True)
    class_weight_map = {int(c): float(len(y_anc)) / (3.0 * float(cnt)) for c, cnt in zip(classes, counts)}
    w_anc = np.array([class_weight_map[int(y)] for y in y_anc], dtype=np.float32)

    oof_probs_full = np.full((n, 3), 1.0 / 3.0, dtype=np.float64)
    oof_probs_noball = np.full((n, 3), 1.0 / 3.0, dtype=np.float64)

    oof_anc_preds_full = np.zeros((len(anchor_idx), 3), dtype=np.float64)
    oof_anc_preds_noball = np.zeros((len(anchor_idx), 3), dtype=np.float64)

    feature_importances_full = np.zeros(X_full.shape[1], dtype=np.float64)
    feature_importances_noball = np.zeros(X_noball.shape[1], dtype=np.float64)

    for fold in range(n_blocks):
        train_mask = b_anc != fold
        val_anc_mask = b_anc == fold
        val_all_mask = block_id == fold

        X_tr_full = X_full[anchor_idx[train_mask]]
        X_tr_noball = X_noball[anchor_idx[train_mask]]
        y_tr = y_anc[train_mask]
        w_tr = w_anc[train_mask]

        # 1. Fit GBDT_Full (88 features)
        model_full = _build_gbdt_model(seed=seed + fold, is_noball=False)
        model_full.fit(X_tr_full, y_tr, sample_weight=w_tr)
        oof_probs_full[val_all_mask] = model_full.predict_proba(X_full[val_all_mask])
        oof_anc_preds_full[val_anc_mask] = model_full.predict_proba(X_full[anchor_idx[val_anc_mask]])
        if hasattr(model_full, "feature_importances_"):
            feature_importances_full += model_full.feature_importances_ / n_blocks

        # 2. Fit GBDT_NoBall (46 ball-independent features — Masked-Ball Student!)
        model_noball = _build_gbdt_model(seed=seed + 100 + fold, is_noball=True)
        model_noball.fit(X_tr_noball, y_tr, sample_weight=w_tr)
        oof_probs_noball[val_all_mask] = model_noball.predict_proba(X_noball[val_all_mask])
        oof_anc_preds_noball[val_anc_mask] = model_noball.predict_proba(
            X_noball[anchor_idx[val_anc_mask]]
        )
        if hasattr(model_noball, "feature_importances_"):
            feature_importances_noball += model_noball.feature_importances_ / n_blocks

    oof_anc_preds_full /= np.maximum(np.sum(oof_anc_preds_full, axis=1, keepdims=True), 1e-12)
    oof_anc_preds_noball /= np.maximum(np.sum(oof_anc_preds_noball, axis=1, keepdims=True), 1e-12)

    cv_metrics = {
        "n_anchors": int(len(anchor_idx)),
        "anchor_counts": {int(c): int(cnt) for c, cnt in zip(classes, counts)},
        "oof_acc_full": float(accuracy_score(y_anc, np.argmax(oof_anc_preds_full, axis=1))),
        "oof_logloss_full": float(log_loss(y_anc, oof_anc_preds_full)),
        "oof_acc_noball": float(accuracy_score(y_anc, np.argmax(oof_anc_preds_noball, axis=1))),
        "oof_logloss_noball": float(log_loss(y_anc, oof_anc_preds_noball)),
        "feature_importances_full": feature_importances_full,
        "feature_importances_noball": feature_importances_noball,
    }

    return oof_probs_full, oof_probs_noball, cv_metrics
