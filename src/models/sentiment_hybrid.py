"""Stage 3 (Tier 4): Match-Sentiment-Adaptive Hybrid Model (Physics + ML Fusion + Tactical Priors).

Computes a continuous Match Sentiment Tensor Psi_t plus Flick Tactical Press/Control and
Player/Zone Retention Signals from homography camera kinematics, contention entropy, duel flips,
and ball occlusion:
  - S_CALM_CONTROL       (0): Low entropy & calm/tripod-locked buildup -> Physics + positional control.
  - S_CONTESTED_CHAOS    (1): High entropy & duel flip rate -> Duel-robust ML dominates.
  - S_VERTICAL_TRANSITION(2): High camera pan speed & acceleration -> Agile counter-attack fusion.
  - S_OCCLUDED_FLIGHT    (3): Missing raw ball in live flight -> Contested handover bridge + NoBall ML.
  - S_STOPPAGE_RESET     (4): Wide-FOV reset / Goal-kick zoom / OOB / Walk-up -> Physics Stoppage.
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss

SENTIMENT_NAMES = (
    "CALM_CONTROL",
    "CONTESTED_CHAOS",
    "VERTICAL_TRANSITION",
    "OCCLUDED_FLIGHT",
    "STOPPAGE_RESET",
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


def compute_match_sentiment_tensor(features, regimes):
    """Compute continuous Match Sentiment axes Psi_t, tactical Flick signals, and regime IDs."""
    ent = features["contention_entropy_2s"]
    flip = features["duel_flip_rate_2s"]
    v_cam = features["v_cam"]
    v_cam_2s = features["v_cam_mean_2s"]
    acc_cam = features["acc_cam"]
    cam_x_h = features["cam_x_attack_h"]
    vx_h = features["vx_attack_h"]
    raw_codes = features["raw_codes"]
    dead_mask = regimes["dead_ball_mask"]
    live_bridge = regimes["live_bridge_lock"]
    contested_occl = regimes.get("contested_occl_mask", np.zeros(len(raw_codes), dtype=bool))

    # 1. Duel Chaos / Contention Sentiment psi_chaos in [0, 1]
    ent_gate = _sigmoid((ent - 0.50) / 0.14)
    flip_boost = 0.55 + 0.45 * np.clip(flip / 0.035, 0.0, 1.0)
    psi_chaos = np.clip(ent_gate * flip_boost, 0.0, 1.0)

    # 2. Vertical Transition / Counter-Attack Tempo Sentiment psi_tempo in [0, 1]
    tempo_signal = 0.60 * v_cam_2s + 0.40 * v_cam + 0.04 * np.clip(acc_cam, 0.0, 15.0)
    psi_tempo = np.clip(_sigmoid((tempo_signal - 1.55) / 0.45), 0.0, 1.0)

    # 3. Calm Positional / Tripod-Locked Siege Sentiment psi_tripod in [0, 1]
    psi_tripod = np.clip(_sigmoid((0.40 - v_cam_2s) / 0.16) * (1.0 - 0.50 * psi_chaos), 0.0, 1.0)

    # 4. Occluded / Missing-Ball Live Flight Sentiment psi_occl in {0, 1}
    psi_occl = ((raw_codes == 2) & (~dead_mask)).astype(np.float64)

    # 5. Stoppage / Reset Sentiment psi_stoppage in {0, 1}
    psi_stoppage = dead_mask.astype(np.float64)

    # 6. Hansi Flick Tactical & Positional Retention Signals:
    # - High-line counter-press trap: Home pushes into mid/attacking half (cam_x > 0.38) while
    #   Away has NOT broken behind the line (vx_attack_h > -0.75 m/s).
    # - Counter-attack line break: Away breaks Flick's high line vertically (vx_attack_h < -1.05 m/s).
    # - Player/Zone retention stability: track-level retention (when ATD present) + low-chaos control.
    psi_high_press = (
        _sigmoid((cam_x_h - 0.38) / 0.14)
        * _sigmoid((vx_h + 0.75) / 0.35)
        * (0.35 + 0.65 * psi_chaos)
    )
    psi_counter_break_a = _sigmoid((-vx_h - 1.05) / 0.35)

    track_ret = features.get("track_retention_prior")
    if features.get("has_full_atd", False) and track_ret is not None:
        player_ret = np.clip((track_ret - 0.45) / 0.25, 0.0, 1.0)
    else:
        player_ret = 0.5
    zone_stability = (
        _sigmoid((0.55 - cam_x_h) / 0.18)
        * (1.0 - 0.60 * psi_chaos)
        * _sigmoid((1.0 - v_cam) / 0.35)
    )
    retention_h_stability = np.clip(0.55 * zone_stability + 0.45 * player_ret, 0.0, 1.0)

    # Categorical dominant sentiment label (0..4) for diagnostics & reporting
    dom = np.zeros(len(raw_codes), dtype=np.int8)  # default: 0 CALM_CONTROL
    dom[psi_tempo >= 0.52] = 2                     # 2 VERTICAL_TRANSITION
    dom[psi_chaos >= 0.48] = 1                     # 1 CONTESTED_CHAOS
    dom[(psi_occl > 0.5) | live_bridge] = 3        # 3 OCCLUDED_FLIGHT
    dom[dead_mask] = 4                             # 4 STOPPAGE_RESET

    return {
        "psi_chaos": psi_chaos,
        "psi_tempo": psi_tempo,
        "psi_tripod": psi_tripod,
        "psi_occl": psi_occl,
        "psi_stoppage": psi_stoppage,
        "contested_occl": contested_occl,
        "psi_high_press": psi_high_press,
        "psi_counter_break_a": psi_counter_break_a,
        "retention_h_stability": retention_h_stability,
        "dominant_sentiment": dom,
    }


def _fit_oof_meta_router_params(
    logit_phys,
    logit_gf,
    logit_gnb,
    logit_seq,
    logit_em,
    sentiment,
    spec_cv_metrics,
    seed=42,
):
    """Optimize sentiment fusion temperatures & Flick prior weight strictly via 6-fold OOF LogLoss.

    Evaluated on clean + stress-corrupted (synthetic ball-dropout & chaotic duel attenuation)
    live anchor samples across the 6 temporal CV blocks so stacking parameters never collapse
    to 100%-memorized GBDT logits and never touch human evaluation labels.
    """
    if (
        spec_cv_metrics is None
        or "live_anchor_idx" not in spec_cv_metrics
        or "y_anc" not in spec_cv_metrics
    ):
        return {
            "w_chaos_boost": 0.36,
            "w_occl_contested_ml": 0.50,
            "beta_flick_prior": 0.22,
            "w_noball_in_ml": 0.22,
            "oof_meta_logloss": None,
        }

    anc_idx = spec_cv_metrics["live_anchor_idx"]
    y_anc = spec_cv_metrics["y_anc"]          # 0=H, 1=A
    b_anc = spec_cv_metrics["b_anc"]
    y_h = (y_anc == 0).astype(np.float64)
    pi_h = spec_cv_metrics["oof_pi_h_frame"][anc_idx]
    base_h_odds = np.log(np.clip(pi_h / np.maximum(1.0 - pi_h, 1e-3), 0.70, 2.40))

    lp_a = logit_phys[anc_idx]
    lgf_a = logit_gf[anc_idx]
    lgnb_a = logit_gnb[anc_idx]
    lseq_a = logit_seq[anc_idx]
    lem_a = logit_em[anc_idx]

    ch_a = sentiment["psi_chaos"][anc_idx]
    tm_a = sentiment["psi_tempo"][anc_idx]
    tr_a = sentiment["psi_tripod"][anc_idx]
    hp_a = sentiment["psi_high_press"][anc_idx]
    cb_a = sentiment["psi_counter_break_a"][anc_idx]

    # Build a stress-augmented training/validation target view on live anchors:
    # 35% of samples simulate missing ball (using GBDT_NoBall + attenuated Phys/EM),
    # 25% simulate chaotic ball confusion (attenuating raw ball dominance + spiking chaos).
    rng = np.random.RandomState(seed + 19)
    n_a = len(anc_idx)
    u = rng.rand(n_a)
    drop_m = u < 0.35
    chaos_m = (u >= 0.35) & (u < 0.60)

    lp_s = np.where(drop_m, 0.25 * lp_a, np.where(chaos_m, 0.55 * lp_a + rng.normal(0.0, 0.45, n_a), lp_a))
    lgf_s = np.where(drop_m, lgnb_a, np.where(chaos_m, 0.55 * lgf_a + rng.normal(0.0, 0.55, n_a), lgf_a))
    ch_s = np.where(chaos_m, np.maximum(ch_a, 0.75), ch_a)
    flick_feat = (
        base_h_odds * (0.45 * (0.4 + 0.6 * tr_a) + 0.30 * hp_a - 0.35 * cb_a)
    )

    unique_folds = np.unique(b_anc)
    learned_folds = []
    oof_probs = np.full(n_a, 0.5, dtype=np.float64)

    def _forward_logit(params, idx_mask):
        w_chaos, w_nb_ml, b_flick = params
        active = np.maximum(ch_s[idx_mask], 0.75 * tm_a[idx_mask])
        w_ml = np.clip(0.24 + w_chaos * active - 0.06 * tr_a[idx_mask], 0.16, 0.68)
        w_gf = max(0.05, 0.72 - w_nb_ml - 0.18)
        l_ml = w_gf * lgf_s[idx_mask] + w_nb_ml * lgnb_a[idx_mask] + 0.14 * lseq_a[idx_mask] + 0.14 * lem_a[idx_mask]
        l_comb = (1.0 - w_ml) * lp_s[idx_mask] + w_ml * l_ml + b_flick * flick_feat[idx_mask]
        return l_comb

    init_p = np.array([0.32, 0.30, 0.16], dtype=np.float64)
    bounds = [(0.16, 0.48), (0.12, 0.45), (0.00, 0.40)]

    for fold in unique_folds:
        tr_m = b_anc != fold
        va_m = b_anc == fold

        def _loss_fn(params):
            lc = _forward_logit(params, tr_m)
            p_h = _sigmoid(lc)
            ll = -np.mean(
                y_h[tr_m] * np.log(np.clip(p_h, 1e-6, 1.0))
                + (1.0 - y_h[tr_m]) * np.log(np.clip(1.0 - p_h, 1e-6, 1.0))
            )
            reg = 0.025 * float(np.sum((params - init_p) ** 2))
            return float(ll + reg)

        res = minimize(_loss_fn, init_p, bounds=bounds, method="L-BFGS-B")
        p_fold = res.x if res.success else init_p
        learned_folds.append(p_fold)
        oof_probs[va_m] = _sigmoid(_forward_logit(p_fold, va_m))

    mean_p = np.mean(np.vstack(learned_folds), axis=0)
    oof_ll = float(log_loss(y_anc, np.column_stack([oof_probs, 1.0 - oof_probs])))
    return {
        "w_chaos_boost": float(mean_p[0]),
        "w_noball_in_ml": float(mean_p[1]),
        "beta_flick_prior": float(mean_p[2]),
        "w_occl_contested_ml": 0.50,
        "oof_meta_logloss": oof_ll,
    }


def fuse_models_by_match_sentiment(
    features,
    regimes,
    sentiment,
    tier1_log_probs,
    em_log_probs,
    oof_ha_full,
    oof_ha_noball,
    seq_log_probs,
    spec_cv_metrics=None,
    p_boundary=None,
    boundary_metrics=None,
):
    """Dynamically fuse Physics and ML predictions via Match Sentiment Psi_t & Flick Tactical Priors.

    Returns:
        hybrid_log_probs: (N, 3) fused log-probabilities over {H, A, D}.
        turnover_gate_ha: (N,) directed H -> A transition permeability multiplier.
        turnover_gate_ah: (N,) directed A -> H counter-press recovery multiplier.
        logit_fused:      (N,) signed H-vs-A log-odds (+ -> H, - -> A).
        routing_summary:  dict of diagnostic routing & OOF meta-stacking weights.
    """
    n = len(features["raw_codes"])
    dead_mask = regimes["dead_ball_mask"]
    psi_chaos = sentiment["psi_chaos"]
    psi_tempo = sentiment["psi_tempo"]
    psi_tripod = sentiment["psi_tripod"]
    psi_occl = sentiment["psi_occl"]
    contested_occl = sentiment["contested_occl"]
    psi_high_press = sentiment["psi_high_press"]
    psi_counter_break_a = sentiment["psi_counter_break_a"]
    ret_h = sentiment["retention_h_stability"]

    # 1. Extract alive H-vs-A log-odds from each model family
    p_t1 = np.exp(tier1_log_probs)
    p_t1_h = np.clip(p_t1[:, 0] / np.maximum(p_t1[:, 0] + p_t1[:, 1], 1e-9), 1e-4, 1.0 - 1e-4)
    logit_phys = np.log(p_t1_h / (1.0 - p_t1_h))

    p_em = np.exp(em_log_probs)
    p_em_h = np.clip(p_em[:, 0] / np.maximum(p_em[:, 0] + p_em[:, 1], 1e-9), 1e-4, 1.0 - 1e-4)
    logit_em = np.log(p_em_h / (1.0 - p_em_h))

    p_seq = np.exp(seq_log_probs)
    p_seq_h = np.clip(p_seq[:, 0] / np.maximum(p_seq[:, 0] + p_seq[:, 1], 1e-9), 1e-4, 1.0 - 1e-4)
    logit_seq = np.log(p_seq_h / (1.0 - p_seq_h))

    p_gf_h = np.clip(oof_ha_full[:, 0], 1e-3, 1.0 - 1e-3)
    logit_gbdt_full = np.clip(np.log(p_gf_h / (1.0 - p_gf_h)) / 2.2, -2.8, 2.8)

    p_gnb_h = np.clip(oof_ha_noball[:, 0], 1e-3, 1.0 - 1e-3)
    logit_gbdt_noball = np.clip(np.log(p_gnb_h / (1.0 - p_gnb_h)) / 1.4, -2.8, 2.8)

    logit_seq_cal = np.clip(logit_seq / 2.2, -2.8, 2.8)
    logit_em_cal = np.clip(logit_em / 3.0, -2.2, 2.2)

    # 2. Fit OOF Sentiment Meta-Parameters cleanly across 6 temporal folds
    meta = _fit_oof_meta_router_params(
        logit_phys=logit_phys,
        logit_gf=logit_gbdt_full,
        logit_gnb=logit_gbdt_noball,
        logit_seq=logit_seq_cal,
        logit_em=logit_em_cal,
        sentiment=sentiment,
        spec_cv_metrics=spec_cv_metrics,
    )
    w_nb_ml = meta["w_noball_in_ml"]
    w_gf_ml = max(0.05, 0.74 - w_nb_ml - 0.12)

    logit_ml_visible = (
        w_gf_ml * logit_gbdt_full
        + w_nb_ml * logit_gbdt_noball
        + 0.16 * logit_seq_cal
        + max(0.04, 1.0 - w_gf_ml - w_nb_ml - 0.16) * logit_em_cal
    )
    logit_ml_occluded = 0.62 * logit_gbdt_noball + 0.24 * logit_seq_cal + 0.14 * logit_em_cal
    logit_ml = np.where(psi_occl > 0.5, logit_ml_occluded, logit_ml_visible)

    # 3. Sentiment Routing Weight w_ml(t) vs w_phys(t):
    # - Calm Control: Physics persistence dominates
    # - Contested Chaos / Vertical Transition: OOF-learned w_chaos_boost elevates ML share
    # - In-Flight Pass: Shift weight toward ML + Matrix 9 structural prior
    # - Occluded Flight (Same-Team / Contested): Matrix 9 NoBall ML bridge
    active_chaos_or_tempo = np.maximum(psi_chaos, 0.75 * psi_tempo)
    w_ml_visible = np.clip(
        0.26 + meta["w_chaos_boost"] * active_chaos_or_tempo - 0.06 * psi_tripod,
        0.18,
        0.64,
    )
    has_full_atd = bool(features.get("has_full_atd", False))
    in_flight = features.get("in_flight", np.zeros(n, dtype=bool))
    if has_full_atd:
        w_ml_visible = np.where(in_flight, np.maximum(w_ml_visible, 0.48), w_ml_visible)
    w_ml_occl = np.where(contested_occl, meta["w_occl_contested_ml"], 0.38 if has_full_atd else 0.32)
    w_ml = np.where(psi_occl > 0.5, w_ml_occl, w_ml_visible)
    w_phys = 1.0 - w_ml

    # 4. Hansi Flick Tactical Control & High-Press / Counter-Break Prior
    if spec_cv_metrics is not None and "oof_pi_h_frame" in spec_cv_metrics:
        pi_h = spec_cv_metrics["oof_pi_h_frame"]
    else:
        pi_h = np.full(n, 0.58, dtype=np.float64)
    base_h_odds = np.log(np.clip(pi_h / np.maximum(1.0 - pi_h, 1e-3), 0.70, 2.40))
    flick_prior = meta["beta_flick_prior"] * base_h_odds * (
        0.45 * (0.4 + 0.6 * psi_tripod)
        + 0.30 * psi_high_press
        - 0.35 * psi_counter_break_a
    )

    logit_fused = w_phys * logit_phys + w_ml * logit_ml + flick_prior

    # 5. Construct 3-Class Emission Log-Probabilities
    p_h_alive = _sigmoid(logit_fused)
    p_a_alive = 1.0 - p_h_alive

    prob_d = np.where(dead_mask, 0.97, 0.008)
    prob_alive = 1.0 - prob_d

    probs = np.zeros((n, 3), dtype=np.float64)
    probs[:, 0] = np.clip(prob_alive * p_h_alive, 1e-6, 1.0)
    probs[:, 1] = np.clip(prob_alive * p_a_alive, 1e-6, 1.0)
    probs[:, 2] = np.clip(prob_d, 1e-6, 1.0)
    probs /= np.sum(probs, axis=1, keepdims=True)
    hybrid_log_probs = np.log(probs)

    # 6. Directed Asymmetric Turnover Gates modulated by Sentiment + Matrix 9/10 + Path-CRF Edge
    base_gate = np.clip(
        0.85 + 0.42 * psi_chaos + 0.28 * psi_tempo - 0.28 * psi_tripod,
        0.62,
        1.48,
    )
    knock_on_any = features.get("knock_on_any", np.zeros(n, dtype=bool))
    shot_flight = features.get("shot_flight", np.zeros(n, dtype=bool))
    if has_full_atd:
        # Hold aerial pass flights AND open-space knock-on dribbles steady; release on true shots!
        base_gate = np.where(shot_flight, base_gate * 1.32, base_gate)
        base_gate = np.where(in_flight & (~shot_flight), base_gate * 0.55, base_gate)
        base_gate = np.where(knock_on_any, base_gate * 0.55, base_gate)
    # In contested missing-ball handovers, slightly ease transitions so NoBall can locate handover
    base_gate = np.where(contested_occl, np.maximum(base_gate, 1.18), base_gate)

    if p_boundary is not None:
        crf_edge_mult = np.clip(0.94 + 0.46 * p_boundary, 0.94, 1.40)
    else:
        crf_edge_mult = 1.0

    if has_full_atd:
        pr_h = features["press_cap_on_h"]
        pr_a = features["press_cap_on_a"]
        bait_h = features["bait_safety_h"]
        bait_a = features["bait_safety_a"]
        pin_a = features["pin_mean_a"]
        pin_h = features["pin_mean_h"]
        stk_h = features.get("shot_takeon_open_h", np.zeros(n))
        stk_a = features.get("shot_takeon_open_a", np.zeros(n))
        scr_h = features.get("shot_crowded_risk_h", np.zeros(n))
        scr_a = features.get("shot_crowded_risk_a", np.zeros(n))
        m9_ha = np.clip(
            1.0 + 0.40 * (pr_h - 0.40) - 0.45 * (bait_h - 0.28) - 0.18 * pin_a + 0.30 * scr_h - 0.22 * stk_h,
            0.60,
            1.65,
        )
        m9_ah = np.clip(
            1.0 + 0.40 * (pr_a - 0.35) - 0.45 * (bait_a - 0.25) - 0.18 * pin_h + 0.30 * scr_a - 0.22 * stk_a,
            0.60,
            1.65,
        )
    else:
        m9_ha = 1.0
        m9_ah = 1.0

    # Flick counter-press & high-retention stability + Path-CRF & Matrix 9/10 directional gating
    break_guard = 1.0 - psi_counter_break_a
    turnover_gate_ha = np.clip(
        base_gate
        * crf_edge_mult
        * m9_ha
        * (1.0 - 0.10 * ret_h * break_guard - 0.06 * psi_high_press)
        * (1.0 + 0.20 * psi_counter_break_a),
        0.42,
        1.75,
    )
    turnover_gate_ah = np.clip(
        base_gate
        * crf_edge_mult
        * m9_ah
        * (1.0 + 0.10 * psi_high_press - 0.08 * psi_counter_break_a),
        0.42,
        1.75,
    )

    dom = sentiment["dominant_sentiment"]
    routing_summary = {
        "mean_w_ml_all": float(np.mean(w_ml[dom != 4])),
        "mean_w_ml_calm": float(np.mean(w_ml[dom == 0])) if np.any(dom == 0) else 0.0,
        "mean_w_ml_chaos": float(np.mean(w_ml[dom == 1])) if np.any(dom == 1) else 0.0,
        "mean_w_ml_tempo": float(np.mean(w_ml[dom == 2])) if np.any(dom == 2) else 0.0,
        "mean_w_ml_occl": float(np.mean(w_ml[dom == 3])) if np.any(dom == 3) else 0.0,
        "mean_w_ml_contested_occl": float(np.mean(w_ml[contested_occl])) if np.any(contested_occl) else 0.0,
        "mean_gate_ha": float(np.mean(turnover_gate_ha[dom != 4])),
        "mean_gate_ah": float(np.mean(turnover_gate_ah[dom != 4])),
        "meta_w_chaos_boost": meta["w_chaos_boost"],
        "meta_w_noball_in_ml": meta["w_noball_in_ml"],
        "meta_beta_flick_prior": meta["beta_flick_prior"],
        "oof_meta_logloss": meta["oof_meta_logloss"],
        "oof_boundary_auc": boundary_metrics.get("oof_boundary_auc") if boundary_metrics else None,
    }

    return hybrid_log_probs, turnover_gate_ha, turnover_gate_ah, logit_fused, routing_summary
