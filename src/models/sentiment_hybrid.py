"""Stage 3 (Tier 4): Match-Sentiment-Adaptive Hybrid Model (Physics + ML Fusion).

Computes a continuous 5-dimensional Match Sentiment Tensor Psi_t at every frame from
homography camera kinematics, duel contention entropy, flip rate, and ball occlusion:
  - S_CALM_CONTROL       (0): Low entropy & calm/tripod-locked buildup -> Physics persistence dominates.
  - S_CONTESTED_CHAOS    (1): High entropy & duel flip rate -> ML (GBDT_Alive + SeqRefiner + EM) dominates.
  - S_VERTICAL_TRANSITION(2): High camera pan speed & acceleration -> Agile ML + Physics counter-attack fusion.
  - S_OCCLUDED_FLIGHT    (3): Missing raw ball in live flight -> GBDT_Alive_NoBall + Homography bridge.
  - S_STOPPAGE_RESET     (4): Wide-FOV reset / Goal-kick zoom / OOB / Walk-up -> Physics Stoppage Matrices.
"""

import numpy as np

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
    """Compute continuous Match Sentiment axes Psi_t and discrete dominant regime IDs."""
    ent = features["contention_entropy_2s"]
    flip = features["duel_flip_rate_2s"]
    v_cam = features["v_cam"]
    v_cam_2s = features["v_cam_mean_2s"]
    acc_cam = features["acc_cam"]
    raw_codes = features["raw_codes"]
    dead_mask = regimes["dead_ball_mask"]
    live_bridge = regimes["live_bridge_lock"]

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
        "dominant_sentiment": dom,
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
):
    """Dynamically fuse Physics and ML predictions conditioned on Match Sentiment Psi_t.

    Combines log-odds across Physics (Tier 1), Specialist ML (GBDT_Alive_Full / NoBall),
    Dilated Temporal Sequence Refiner, and Unsupervised Baum-Welch EM using continuous
    sentiment weights (w_phys(t), w_ml(t)), and generates sentiment-modulated HSMM gates.
    """
    n = len(features["raw_codes"])
    dead_mask = regimes["dead_ball_mask"]
    psi_chaos = sentiment["psi_chaos"]
    psi_tempo = sentiment["psi_tempo"]
    psi_tripod = sentiment["psi_tripod"]
    psi_occl = sentiment["psi_occl"]

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

    # GBDT_Alive_Full & GBDT_Alive_NoBall class 0 is P(H), class 1 is P(A).
    # Temperature-scale ML logits (T=2.2, clipped to [-3.2, +3.2]) so extreme anchor confidence
    # never overrides Viterbi duration persistence on noisy 15-frame flickers.
    p_gf_h = np.clip(oof_ha_full[:, 0], 1e-3, 1.0 - 1e-3)
    logit_gbdt_full = np.clip(np.log(p_gf_h / (1.0 - p_gf_h)) / 2.2, -3.2, 3.2)

    p_gnb_h = np.clip(oof_ha_noball[:, 0], 1e-3, 1.0 - 1e-3)
    logit_gbdt_noball = np.clip(np.log(p_gnb_h / (1.0 - p_gnb_h)) / 1.6, -2.5, 2.5)

    logit_seq_cal = np.clip(logit_seq / 2.0, -3.0, 3.0)
    logit_em_cal = np.clip(logit_em / 1.5, -2.8, 2.8)

    # Composite ML alive log-odds when ball is visible vs missing:
    logit_ml_visible = 0.62 * logit_gbdt_full + 0.18 * logit_seq_cal + 0.20 * logit_em_cal
    logit_ml_occluded = 0.55 * logit_gbdt_noball + 0.20 * logit_seq_cal + 0.25 * logit_em_cal

    logit_ml = np.where(psi_occl > 0.5, logit_ml_occluded, logit_ml_visible)

    # 2. Match Sentiment Routing Gate w_ml(t) vs w_phys(t):
    # - In Calm Buildup (psi_chaos ~ 0, psi_tripod high): Physics dominates (~0.75 vs ML ~0.25).
    # - In Contested Chaos / Fast Transitions: ML weight rises to ~0.58 so ML resolves contested
    #   high-entropy interior duels while preserving sharp boundary discipline.
    # - In Occluded Live Flight: blend Physics Homography bridge (0.68) with ML NoBall (0.32).
    active_chaos_or_tempo = np.maximum(psi_chaos, 0.75 * psi_tempo)
    w_ml_visible = np.clip(0.24 + 0.34 * active_chaos_or_tempo - 0.06 * psi_tripod, 0.18, 0.58)
    w_ml = np.where(psi_occl > 0.5, 0.32, w_ml_visible)
    w_phys = 1.0 - w_ml

    # Fused log-odds via Sentiment-Weighted Log-Opinion Pool
    logit_fused = w_phys * logit_phys + w_ml * logit_ml
    prob_h_given_alive = _sigmoid(logit_fused)
    prob_a_given_alive = 1.0 - prob_h_given_alive

    # 3. Preserve Physics Dead-Ball mastery (D F1 = 0.9830) cleanly separated from alive team call
    prob_d = np.where(dead_mask, 0.97, 0.008)
    prob_alive = 1.0 - prob_d

    probs = np.zeros((n, 3), dtype=np.float64)
    probs[:, 0] = np.clip(prob_alive * prob_h_given_alive, 1e-6, 1.0)
    probs[:, 1] = np.clip(prob_alive * prob_a_given_alive, 1e-6, 1.0)
    probs[:, 2] = np.clip(prob_d, 1e-6, 1.0)
    probs /= np.sum(probs, axis=1, keepdims=True)

    # 4. Sentiment-Modulated HSMM Turnover Gate (bounded to [0.65, 1.52]):
    # Opens turnover barriers during high-chaos/high-tempo transitions so authentic short counter-attacks
    # survive Viterbi; tightens in calm tripod buildup to prevent fragmentation.
    turnover_gate = np.clip(
        0.85 + 0.45 * psi_chaos + 0.30 * psi_tempo - 0.30 * psi_tripod,
        0.65,
        1.52,
    )

    routing_summary = {
        "mean_w_ml_all": float(np.mean(w_ml)),
        "mean_w_ml_calm": float(np.mean(w_ml[sentiment["dominant_sentiment"] == 0])) if np.any(sentiment["dominant_sentiment"] == 0) else 0.0,
        "mean_w_ml_chaos": float(np.mean(w_ml[sentiment["dominant_sentiment"] == 1])) if np.any(sentiment["dominant_sentiment"] == 1) else 0.0,
        "mean_w_ml_tempo": float(np.mean(w_ml[sentiment["dominant_sentiment"] == 2])) if np.any(sentiment["dominant_sentiment"] == 2) else 0.0,
        "mean_w_ml_occl": float(np.mean(w_ml[sentiment["dominant_sentiment"] == 3])) if np.any(sentiment["dominant_sentiment"] == 3) else 0.0,
    }

    return np.log(probs), turnover_gate, logit_fused, routing_summary
