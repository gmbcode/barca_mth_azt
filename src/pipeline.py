"""End-to-End Multi-Tier ML & Physics Ensemble Pipeline (Tier 1 + Tier 2 GBDT/EM + Tier 3 MS-TCN)."""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME  # noqa: E402
from src.preprocessing import load_stage1_data  # noqa: E402
from src.smart_tagging_priors import compute_smart_tagging_priors  # noqa: E402
from src.broadcast_sync import synchronize_broadcast_to_tactical  # noqa: E402
from src.features import extract_physics_feature_dict  # noqa: E402
from src.models.heuristic_fsm import (  # noqa: E402
    compute_tier1_emission_log_probs,
    detect_physical_regimes,
)
from src.models.pseudo_gbdt import (  # noqa: E402
    _select_algorithmic_anchors,
    fit_ball_dropout_gbdt_models,
    fit_baum_welch_em,
    fit_sentiment_specialist_gbdt_models,
)
from src.models.external_seq import (  # noqa: E402
    CRISP_DILATED_OFFSETS,
    fit_and_predict_stage2_temporal_refiner,
    fit_oof_path_crf_boundary_model,
)
from src.models.sentiment_hybrid import (  # noqa: E402
    SENTIMENT_NAMES,
    compute_match_sentiment_tensor,
    fuse_models_by_match_sentiment,
)
from src.decoder import decode_full_match, decode_sentiment_hybrid_match  # noqa: E402


def write_submission_csv(path, labels):
    """Write a frame,label CSV in the exact format required by validate_submission.py."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
    return path


def run_ml_ensemble_pipeline(game=GAME, data_dir=None, out_csv=None, seed=42):
    """Run all 4 stages including Tier 1 Physics, Tier 2 GBDT/EM, and Tier 3 Dilated Sequence ML."""
    np.random.seed(seed)

    # Stage 1: Load Homography, ATD, Kalman-RTS trajectories, Optical Flow & Jacobians
    stage1 = load_stage1_data(game=game, data_dir=data_dir)
    st_priors = compute_smart_tagging_priors(stage1, game=game, data_dir=data_dir)
    bcast_sync = synchronize_broadcast_to_tactical(stage1, game=game, data_dir=data_dir)

    # Stage 2: Extract 88-feature tabular & sequence matrix (42 ball + 46 ball-free features)
    features = extract_physics_feature_dict(stage1, st_priors)

    # Stage 3 — Tier 1: Physical regimes & physics emission log-probabilities
    regimes = detect_physical_regimes(features, stage1["half_blocks"])
    tier1_log_probs = compute_tier1_emission_log_probs(features, regimes)

    # Stage 3 — Tier 2A: Unsupervised Factored Baum-Welch EM HSMM
    anchor_labels, graded_mask = _select_algorithmic_anchors(
        features, regimes, stage1["half_blocks"]
    )
    em_log_probs = fit_baum_welch_em(
        features, regimes, tier1_log_probs, graded_mask, n_iter=4
    )

    # Stage 3 — Tier 2B: Out-of-Fold Temporal CV GBDT_Full (88 feat) & GBDT_NoBall (46 feat)
    oof_probs_full, oof_probs_noball, gbdt_cv_metrics = fit_ball_dropout_gbdt_models(
        features, anchor_labels, graded_mask, n_blocks=6, stride=4, seed=seed
    )

    # Stage 3 — Tier 3: Multi-Stage Dilated Temporal Neural Refiner with Sim2Real 41% Ball-Dropout
    seq_log_probs, seq_metrics = fit_and_predict_stage2_temporal_refiner(
        oof_probs_full,
        oof_probs_noball,
        features,
        anchor_labels,
        graded_mask,
        stage1["half_blocks"],
        stride=6,
        seed=seed,
    )

    # Adaptive Missing-Ball Ensemble Gate across Tier 1 + Tier 2A + Tier 2B + Tier 3:
    # When ball is visible: combine Tier 1 (0.45) + GBDT_Full (0.25) + SeqRefiner (0.15) + EM (0.10) + GBDT_NoBall (0.05)
    # When ball is missing (41% NaN frames): rely on Tier 1 Bridge (0.50) + GBDT_NoBall Student (0.20) + SeqRefiner (0.20) + EM (0.10)
    p_tier1 = np.exp(tier1_log_probs)
    p_em = np.exp(em_log_probs)
    p_seq = np.exp(seq_log_probs)

    ball_seen = (features["raw_codes"] < 2)[:, None]
    ensemble_probs = np.where(
        ball_seen,
        0.45 * p_tier1 + 0.25 * oof_probs_full + 0.15 * p_seq + 0.10 * p_em + 0.05 * oof_probs_noball,
        0.50 * p_tier1 + 0.20 * oof_probs_noball + 0.20 * p_seq + 0.10 * p_em,
    )
    ensemble_probs = np.clip(ensemble_probs, 1e-6, 1.0)
    ensemble_probs /= np.sum(ensemble_probs, axis=1, keepdims=True)
    ensemble_log_probs = np.log(ensemble_probs)

    # Stage 4: Time-Inhomogeneous Viterbi/HSMM decoding, deflection filtering & boundary snapping
    labels = decode_full_match(
        ensemble_log_probs, features, regimes, stage1["half_blocks"]
    )

    if out_csv:
        write_submission_csv(out_csv, labels)

    return {
        "labels": labels,
        "stage1": stage1,
        "st_priors": st_priors,
        "bcast_sync": bcast_sync,
        "features": features,
        "regimes": regimes,
        "log_probs": ensemble_log_probs,
        "oof_probs_full": oof_probs_full,
        "oof_probs_noball": oof_probs_noball,
        "gbdt_cv_metrics": gbdt_cv_metrics,
        "seq_metrics": seq_metrics,
    }


# Keep backward-compatible alias
run_physics_pipeline = run_ml_ensemble_pipeline


def run_sentiment_hybrid_pipeline(game=GAME, data_dir=None, out_csv=None, seed=42):
    """Run Model 3: Match-Sentiment-Adaptive Hybrid (Physics + Specialist ML + True Viterbi)."""
    np.random.seed(seed)

    # Stage 1 & 2: Data, Physics Matrices, Smart Tagging, and 88 Features
    stage1 = load_stage1_data(game=game, data_dir=data_dir)
    st_priors = compute_smart_tagging_priors(stage1, game=game, data_dir=data_dir)
    bcast_sync = synchronize_broadcast_to_tactical(stage1, game=game, data_dir=data_dir)
    features = extract_physics_feature_dict(stage1, st_priors)

    # Stage 3 — Tier 1 Physics Regimes & Emissions (with Contested Missing-Ball Handover Detection)
    regimes = detect_physical_regimes(features, stage1["half_blocks"], split_contested_occl=True)
    tier1_log_probs = compute_tier1_emission_log_probs(features, regimes)

    # Stage 3 — Match Sentiment Tensor Psi_t + Flick Tactical Signals
    sentiment = compute_match_sentiment_tensor(features, regimes)

    # Stage 3 — Tier 2A (Invariant Forward-Backward EM), Tier 2B (Duel-Robust Alive ML), Tier 3 (Crisp Seq)
    anchor_labels, graded_mask = _select_algorithmic_anchors(
        features, regimes, stage1["half_blocks"]
    )
    em_log_probs = fit_baum_welch_em(
        features,
        regimes,
        tier1_log_probs,
        graded_mask,
        n_iter=4,
        use_invariant_hmm=True,
        half_blocks=stage1["half_blocks"],
    )
    oof_ha_full, oof_ha_noball, spec_cv_metrics = fit_sentiment_specialist_gbdt_models(
        features, anchor_labels, graded_mask, regimes=regimes, n_blocks=6, stride=4, seed=seed
    )

    # Build 3-class probabilities for Dilated Sequence Refiner input
    dead_mask = regimes["dead_ball_mask"]
    p_d = np.where(dead_mask, 0.96, 0.01)
    p_alive = 1.0 - p_d
    p_stage1_full = np.column_stack([p_alive * oof_ha_full[:, 0], p_alive * oof_ha_full[:, 1], p_d])
    p_stage1_noball = np.column_stack([p_alive * oof_ha_noball[:, 0], p_alive * oof_ha_noball[:, 1], p_d])

    seq_log_probs, seq_metrics = fit_and_predict_stage2_temporal_refiner(
        p_stage1_full,
        p_stage1_noball,
        features,
        anchor_labels,
        graded_mask,
        stage1["half_blocks"],
        stride=6,
        seed=seed,
        offsets=CRISP_DILATED_OFFSETS,
        n_smooth_steps=3,
        spike_noise_rate=0.18,
    )

    # Path-CRF Self-Supervised Boundary Edge Potential Classifier
    p_boundary, boundary_metrics = fit_oof_path_crf_boundary_model(
        features=features,
        regimes=regimes,
        sentiment=sentiment,
        oof_ha_full=oof_ha_full,
        oof_ha_noball=oof_ha_noball,
        half_blocks=stage1["half_blocks"],
        n_blocks=6,
        seed=seed,
    )

    # Sentiment-Adaptive OOF Meta-Stacked Fusion & Flick Directed Transition Gating
    (
        hybrid_log_probs,
        turnover_gate_ha,
        turnover_gate_ah,
        logit_fused,
        routing_summary,
    ) = fuse_models_by_match_sentiment(
        features=features,
        regimes=regimes,
        sentiment=sentiment,
        tier1_log_probs=tier1_log_probs,
        em_log_probs=em_log_probs,
        oof_ha_full=oof_ha_full,
        oof_ha_noball=oof_ha_noball,
        seq_log_probs=seq_log_probs,
        spec_cv_metrics=spec_cv_metrics,
        p_boundary=p_boundary,
        boundary_metrics=boundary_metrics,
    )
    turnover_gate = 0.5 * (turnover_gate_ha + turnover_gate_ah)

    # Stage 4: Directed Time-Inhomogeneous Viterbi + Evidence-Integrated Sentiment Filter
    labels = decode_sentiment_hybrid_match(
        hybrid_log_probs,
        features,
        regimes,
        sentiment,
        turnover_gate,
        logit_fused,
        stage1["half_blocks"],
        turnover_gate_ha=turnover_gate_ha,
        turnover_gate_ah=turnover_gate_ah,
    )

    if out_csv:
        write_submission_csv(out_csv, labels)

    return {
        "labels": labels,
        "stage1": stage1,
        "st_priors": st_priors,
        "bcast_sync": bcast_sync,
        "features": features,
        "regimes": regimes,
        "sentiment": sentiment,
        "sentiment_names": SENTIMENT_NAMES,
        "log_probs": hybrid_log_probs,
        "logit_fused": logit_fused,
        "turnover_gate": turnover_gate,
        "turnover_gate_ha": turnover_gate_ha,
        "turnover_gate_ah": turnover_gate_ah,
        "oof_ha_full": oof_ha_full,
        "oof_ha_noball": oof_ha_noball,
        "spec_cv_metrics": spec_cv_metrics,
        "seq_metrics": seq_metrics,
        "p_boundary": p_boundary,
        "boundary_metrics": boundary_metrics,
        "routing_summary": routing_summary,
    }
