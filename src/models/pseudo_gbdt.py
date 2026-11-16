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


def fit_sentiment_specialist_gbdt_models(
    features, anchor_labels, graded_mask, regimes=None, n_blocks=6, stride=4, seed=42
):
    """Train binary {H, A} live-team specialist models (GBDT_Alive_Full & GBDT_Alive_NoBall).

    Uses sharp local live anchors (without 3s centered smearing) and drops >=40-frame centered/future
    ball windows so ML models stay razor-sharp at turnover boundaries while capturing +/-0.8s
    causal/anti-causal momentum, duel chaos, and homography kinematics.
    """
    X_full = features["X_full"]
    # Select sharp local/causal ball signals (run_len=3, flips=4..6, ent=7, w8=[8,10,11,12,13], w20=[14,15,17,18,19], past40=23)
    # and half-invariant ball-free kinematics [46,47,50,51,54,56,58,72,76,77] (no match_clock/reproj/flipped-side leakage)
    keep_ball_cols = [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 23]
    inv_noball_cols = [46, 47, 50, 51, 54, 56, 58, 72, 76, 77]
    n_ball_cols = len(keep_ball_cols)
    alive_full_idx = np.array(keep_ball_cols + inv_noball_cols, dtype=np.int32)
    X_alive_full = X_full[:, alive_full_idx].copy()
    X_noball = X_full[:, inv_noball_cols].copy()
    n = len(X_full)

    graded_indices = np.flatnonzero(graded_mask)
    block_id = np.zeros(n, dtype=np.int32)
    for b_idx, ch in enumerate(np.array_split(graded_indices, n_blocks)):
        block_id[ch] = b_idx

    # Build sharp live {H, A} anchors using short w=8 (0.32s) and w=20 (0.80s) consistency
    raw = features["raw_codes"]
    sh_c = features["alive_home_share"]
    dead_m = regimes["dead_ball_mask"] if regimes is not None else (anchor_labels == 2)
    sharp_h = graded_mask & (~dead_m) & (raw == 0) & (sh_c[:, 0] > 0.85) & (sh_c[:, 1] > 0.72)
    sharp_a = graded_mask & (~dead_m) & (raw == 1) & (sh_c[:, 0] < 0.15) & (sh_c[:, 1] < 0.28)
    sharp_y = np.full(n, -1, dtype=np.int8)
    sharp_y[sharp_h] = 0
    sharp_y[sharp_a] = 1

    live_anchor_idx = np.flatnonzero((sharp_y == 0) | (sharp_y == 1))[::stride]
    y_anc = sharp_y[live_anchor_idx]
    b_anc = block_id[live_anchor_idx]

    classes, counts = np.unique(y_anc, return_counts=True)
    w_map = {int(c): float(len(y_anc)) / (2.0 * float(cnt)) for c, cnt in zip(classes, counts)}
    w_anc = np.array([w_map[int(y)] for y in y_anc], dtype=np.float32)

    oof_ha_full = np.full((n, 2), 0.5, dtype=np.float64)
    oof_ha_noball = np.full((n, 2), 0.5, dtype=np.float64)
    oof_anc_full = np.zeros((len(live_anchor_idx), 2), dtype=np.float64)
    oof_anc_noball = np.zeros((len(live_anchor_idx), 2), dtype=np.float64)

    rng = np.random.RandomState(seed)

    def _build_bin_clf(f_seed, is_nb):
        if HAS_XGB:
            return XGBClassifier(
                n_estimators=140 if not is_nb else 100,
                max_depth=4 if not is_nb else 3,
                learning_rate=0.05 if not is_nb else 0.04,
                subsample=0.80,
                colsample_bytree=0.70 if not is_nb else 0.80,
                min_child_weight=30 if not is_nb else 80,
                reg_lambda=5.0 if not is_nb else 10.0,
                reg_alpha=1.0 if not is_nb else 2.0,
                objective="binary:logistic",
                random_state=f_seed,
                n_jobs=4,
                verbosity=0,
            )
        return HistGradientBoostingClassifier(
            max_iter=140 if not is_nb else 100,
            max_depth=4 if not is_nb else 3,
            learning_rate=0.05,
            min_samples_leaf=40 if not is_nb else 80,
            l2_regularization=5.0 if not is_nb else 10.0,
            random_state=f_seed,
        )

    for fold in range(n_blocks):
        tr_m = b_anc != fold
        va_m = b_anc == fold
        va_all = block_id == fold

        X_tr_f = X_alive_full[live_anchor_idx[tr_m]].copy()
        # Sim2Real Ball-Dropout + Confusion Regularization: mask 35% of training rows' ball features
        # and perturb remaining H/A share features so trees rely on multimodal kinematics during chaos/occlusion.
        drop_m = rng.rand(len(X_tr_f)) < 0.35
        X_tr_f[drop_m, :n_ball_cols] = 0.0
        X_tr_f[~drop_m, 5:n_ball_cols] += rng.normal(
            0.0, 0.25, size=(int(np.sum(~drop_m)), n_ball_cols - 5)
        ).astype(np.float32)

        y_tr = y_anc[tr_m]
        w_tr = w_anc[tr_m]

        clf_full = _build_bin_clf(seed + fold, is_nb=False)
        clf_full.fit(X_tr_f, y_tr, sample_weight=w_tr)
        oof_ha_full[va_all] = clf_full.predict_proba(X_alive_full[va_all])
        oof_anc_full[va_m] = clf_full.predict_proba(X_alive_full[live_anchor_idx[va_m]])

        X_tr_nb = X_noball[live_anchor_idx[tr_m]]
        clf_nb = _build_bin_clf(seed + 100 + fold, is_nb=True)
        clf_nb.fit(X_tr_nb, y_tr, sample_weight=w_tr)
        oof_ha_noball[va_all] = clf_nb.predict_proba(X_noball[va_all])
        oof_anc_noball[va_m] = clf_nb.predict_proba(X_noball[live_anchor_idx[va_m]])

    oof_anc_full /= np.maximum(np.sum(oof_anc_full, axis=1, keepdims=True), 1e-12)
    oof_anc_noball /= np.maximum(np.sum(oof_anc_noball, axis=1, keepdims=True), 1e-12)

    metrics = {
        "n_live_anchors": int(len(live_anchor_idx)),
        "oof_ha_acc_full": float(accuracy_score(y_anc, np.argmax(oof_anc_full, axis=1))),
        "oof_ha_logloss_full": float(log_loss(y_anc, oof_anc_full)),
        "oof_ha_acc_noball": float(accuracy_score(y_anc, np.argmax(oof_anc_noball, axis=1))),
        "oof_ha_logloss_noball": float(log_loss(y_anc, oof_anc_noball)),
    }
    return oof_ha_full, oof_ha_noball, metrics
