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


def _hmm_forward_backward_log(joint_ll, stay_prob=0.975):
    """Fast 3-state Forward-Backward HMM smoother in log-space over one contiguous sequence."""
    T, K = joint_ll.shape
    off_prob = (1.0 - stay_prob) / max(K - 1, 1)
    log_A = np.full((K, K), np.log(off_prob), dtype=np.float64)
    np.fill_diagonal(log_A, np.log(stay_prob))

    log_alpha = np.zeros((T, K), dtype=np.float64)
    log_beta = np.zeros((T, K), dtype=np.float64)

    log_alpha[0] = joint_ll[0]
    m0 = np.max(log_alpha[0])
    log_alpha[0] -= m0 + np.log(np.sum(np.exp(log_alpha[0] - m0)))

    for t in range(1, T):
        # log_alpha[t, j] = logsumexp_i(log_alpha[t-1, i] + log_A[i, j]) + joint_ll[t, j]
        scores = log_alpha[t - 1][:, None] + log_A
        ms = np.max(scores, axis=0)
        lse = ms + np.log(np.sum(np.exp(scores - ms), axis=0))
        v = lse + joint_ll[t]
        mv = np.max(v)
        log_alpha[t] = v - (mv + np.log(np.sum(np.exp(v - mv))))

    for t in range(T - 2, -1, -1):
        scores = log_A + (joint_ll[t + 1] + log_beta[t + 1])[None, :]
        ms = np.max(scores, axis=1)
        v = ms + np.log(np.sum(np.exp(scores - ms[:, None]), axis=1))
        mv = np.max(v)
        log_beta[t] = v - (mv + np.log(np.sum(np.exp(v - mv))))

    log_post = log_alpha + log_beta
    mx = np.max(log_post, axis=1, keepdims=True)
    log_post -= mx + np.log(np.sum(np.exp(log_post - mx), axis=1, keepdims=True))
    return log_post


def fit_baum_welch_em(
    features, regimes, tier1_log_probs, graded_mask, n_iter=4, use_invariant_hmm=False, half_blocks=None
):
    """Model 2A: Factored Baum-Welch Expectation-Maximization over Ball & Ball-Free Heads."""
    X_full = features["X_full"]
    ball_idx = features["ball_feature_indices"]
    noball_idx = features["noball_feature_indices"]
    ball_seen = (features["raw_codes"] < 2) & (~regimes["dead_ball_mask"])

    X_ball = X_full[:, ball_idx].astype(np.float64)
    if use_invariant_hmm and "X_noball_inv" in features:
        X_kin = features["X_noball_inv"].astype(np.float64)
    else:
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
        if use_invariant_hmm and half_blocks:
            log_post = np.zeros_like(joint_ll)
            mx0 = np.max(joint_ll, axis=1, keepdims=True)
            log_post[:] = joint_ll - (mx0 + np.log(np.sum(np.exp(joint_ll - mx0), axis=1, keepdims=True)))
            for _, s_h, e_h in half_blocks:
                log_post[s_h:e_h + 1] = _hmm_forward_backward_log(joint_ll[s_h:e_h + 1], stay_prob=0.975)
        else:
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

    Uses clean + borderline live anchors alongside half-invariant trajectory kinematics (`X_noball_inv`)
    and injects both Sim2Real ball-dropout (30%) and w=8 local flicker-inversion noise (20%) during
    training so GBDT_Alive_Full breaks 100% raw memorization and generalizes into noisy duels.
    """
    X_full = features["X_full"]
    # Omit centered sh_8/sh_20 (which Tier 1 already encodes directly) so GBDT_Alive_Full learns
    # non-redundant multi-scale causal/directional momentum [11..13, 17..19, 23] + kinematics X_noball_inv!
    keep_ball_cols = [3, 4, 5, 6, 7, 11, 12, 13, 17, 18, 19, 23]
    n_ball_cols = len(keep_ball_cols)
    if "X_noball_inv" in features:
        X_noball = features["X_noball_inv"].copy()
    else:
        X_noball = X_full[:, [46, 47, 50, 51, 54, 56, 58, 72, 76, 77]].copy()
    X_alive_full = np.hstack([X_full[:, keep_ball_cols], X_noball]).astype(np.float32)
    n = len(X_full)

    graded_indices = np.flatnonzero(graded_mask)
    block_id = np.zeros(n, dtype=np.int32)
    for b_idx, ch in enumerate(np.array_split(graded_indices, n_blocks)):
        block_id[ch] = b_idx

    # Build clean + moderate-contention live {H, A} anchors so training sees realistic transitions
    raw = features["raw_codes"]
    sh_c = features["alive_home_share"]
    dead_m = regimes["dead_ball_mask"] if regimes is not None else (anchor_labels == 2)
    sharp_h = graded_mask & (~dead_m) & (raw == 0) & (sh_c[:, 0] > 0.85) & (sh_c[:, 1] > 0.72)
    sharp_a = graded_mask & (~dead_m) & (raw == 1) & (sh_c[:, 0] < 0.15) & (sh_c[:, 1] < 0.28)
    mod_h = graded_mask & (~dead_m) & (~sharp_h) & (sh_c[:, 1] > 0.70) & (sh_c[:, 2] > 0.66)
    mod_a = graded_mask & (~dead_m) & (~sharp_a) & (sh_c[:, 1] < 0.30) & (sh_c[:, 2] < 0.34)

    sharp_y = np.full(n, -1, dtype=np.int8)
    sharp_y[sharp_h | mod_h] = 0
    sharp_y[sharp_a | mod_a] = 1
    is_sharp = sharp_h | sharp_a

    live_anchor_idx = np.flatnonzero((sharp_y == 0) | (sharp_y == 1))[::stride]
    y_anc = sharp_y[live_anchor_idx]
    b_anc = block_id[live_anchor_idx]
    sharp_anc = is_sharp[live_anchor_idx]

    classes, counts = np.unique(y_anc, return_counts=True)
    w_map = {int(c): float(len(y_anc)) / (2.0 * float(cnt)) for c, cnt in zip(classes, counts)}
    w_anc = np.array(
        [w_map[int(y)] * (1.0 if sh else 0.55) for y, sh in zip(y_anc, sharp_anc)],
        dtype=np.float32,
    )

    oof_ha_full = np.full((n, 2), 0.5, dtype=np.float64)
    oof_ha_noball = np.full((n, 2), 0.5, dtype=np.float64)
    oof_anc_full = np.zeros((len(live_anchor_idx), 2), dtype=np.float64)
    oof_anc_noball = np.zeros((len(live_anchor_idx), 2), dtype=np.float64)
    oof_anc_noisy = np.zeros((len(live_anchor_idx), 2), dtype=np.float64)
    oof_pi_h_frame = np.full(n, 0.55, dtype=np.float64)

    rng = np.random.RandomState(seed)

    def _corrupt_ball_duel_batch(X_in, r_gen, drop_p=0.30, flip_p=0.15):
        """Inject ball dropout + mild w=8 flicker noise without artificial entropy shortcuts."""
        Xo = X_in.copy()
        draw = r_gen.rand(len(Xo))
        drop_m = draw < drop_p
        flip_m = (draw >= drop_p) & (draw < (drop_p + flip_p))
        # Dropped rows: neutral shares across indices [5:past_8, 6:fut_8, 8:past_20, 9:fut_20, 11:past_40]
        Xo[drop_m, :n_ball_cols] = 0.0
        for col_share in (5, 6, 8, 9, 11):
            Xo[drop_m, col_share] = 0.5
        # Short flicker rows: flip only ultra-short w=8 [5,6,7], leaving w=20, w=40 & X_noball_inv intact
        # Do NOT alter col 4 (entropy) so trees never learn a synthetic parity-inversion shortcut!
        Xo[flip_m, 5] = 1.0 - Xo[flip_m, 5]
        Xo[flip_m, 6] = 1.0 - Xo[flip_m, 6]
        Xo[flip_m, 7] = -Xo[flip_m, 7]
        rem_m = ~drop_m
        Xo[rem_m, 5:n_ball_cols] += r_gen.normal(
            0.0, 0.16, size=(int(np.sum(rem_m)), n_ball_cols - 5)
        ).astype(np.float32)
        return Xo

    def _build_bin_clf(f_seed, is_nb):
        if HAS_XGB:
            return XGBClassifier(
                n_estimators=150 if not is_nb else 120,
                max_depth=4 if not is_nb else 3,
                learning_rate=0.05 if not is_nb else 0.04,
                subsample=0.80,
                colsample_bytree=0.70,
                min_child_weight=30 if not is_nb else 80,
                reg_lambda=5.0 if not is_nb else 10.0,
                reg_alpha=1.0 if not is_nb else 2.0,
                objective="binary:logistic",
                random_state=f_seed,
                n_jobs=4,
                verbosity=0,
            )
        return HistGradientBoostingClassifier(
            max_iter=150 if not is_nb else 120,
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

        y_tr = y_anc[tr_m]
        w_tr = w_anc[tr_m]

        # Out-of-fold empirical H possession share prior strictly from training folds
        oof_pi_h_frame[va_all] = float(np.mean(y_tr == 0))

        X_tr_f = _corrupt_ball_duel_batch(X_alive_full[live_anchor_idx[tr_m]], rng, drop_p=0.30, flip_p=0.15)
        clf_full = _build_bin_clf(seed + fold, is_nb=False)
        clf_full.fit(X_tr_f, y_tr, sample_weight=w_tr)
        oof_ha_full[va_all] = clf_full.predict_proba(X_alive_full[va_all])
        X_va_clean = X_alive_full[live_anchor_idx[va_m]]
        oof_anc_full[va_m] = clf_full.predict_proba(X_va_clean)
        # Honest validation under duel stress (w=8 flicker + dropout)
        X_va_noisy = _corrupt_ball_duel_batch(X_va_clean, rng, drop_p=0.25, flip_p=0.20)
        oof_anc_noisy[va_m] = clf_full.predict_proba(X_va_noisy)

        X_tr_nb = X_noball[live_anchor_idx[tr_m]]
        clf_nb = _build_bin_clf(seed + 100 + fold, is_nb=True)
        clf_nb.fit(X_tr_nb, y_tr, sample_weight=w_tr)
        oof_ha_noball[va_all] = clf_nb.predict_proba(X_noball[va_all])
        oof_anc_noball[va_m] = clf_nb.predict_proba(X_noball[live_anchor_idx[va_m]])

    oof_anc_full /= np.maximum(np.sum(oof_anc_full, axis=1, keepdims=True), 1e-12)
    oof_anc_noball /= np.maximum(np.sum(oof_anc_noball, axis=1, keepdims=True), 1e-12)
    oof_anc_noisy /= np.maximum(np.sum(oof_anc_noisy, axis=1, keepdims=True), 1e-12)

    metrics = {
        "n_live_anchors": int(len(live_anchor_idx)),
        "oof_ha_acc_full": float(accuracy_score(y_anc, np.argmax(oof_anc_full, axis=1))),
        "oof_ha_logloss_full": float(log_loss(y_anc, oof_anc_full)),
        "oof_ha_acc_duel_stress": float(accuracy_score(y_anc, np.argmax(oof_anc_noisy, axis=1))),
        "oof_ha_logloss_duel_stress": float(log_loss(y_anc, oof_anc_noisy)),
        "oof_ha_acc_noball": float(accuracy_score(y_anc, np.argmax(oof_anc_noball, axis=1))),
        "oof_ha_logloss_noball": float(log_loss(y_anc, oof_anc_noball)),
        "live_anchor_idx": live_anchor_idx,
        "y_anc": y_anc,
        "b_anc": b_anc,
        "oof_anc_full": oof_anc_full,
        "oof_anc_noball": oof_anc_noball,
        "oof_pi_h_frame": oof_pi_h_frame,
    }
    return oof_ha_full, oof_ha_noball, metrics
