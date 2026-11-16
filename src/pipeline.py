"""End-to-End Multi-Matrix Physics & Probabilistic Sequence Pipeline."""

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
from src.models.pseudo_gbdt import fit_and_predict_tier2_emissions  # noqa: E402
from src.decoder import decode_full_match  # noqa: E402


def write_submission_csv(path, labels):
    """Write a frame,label CSV in the exact format required by validate_submission.py."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
    return path


def run_physics_pipeline(game=GAME, data_dir=None, out_csv=None, seed=42):
    """Run all 4 stages of the Multi-Matrix Physics Pipeline deterministically."""
    np.random.seed(seed)

    # Stage 1: Load Homography, ATD, Kalman-RTS trajectories, Optical Flow & Jacobians
    stage1 = load_stage1_data(game=game, data_dir=data_dir)

    # Stage 1B & 1C: Smart Tagging conjugate priors and Broadcast cross-camera sync
    st_priors = compute_smart_tagging_priors(stage1, game=game, data_dir=data_dir)
    bcast_sync = synchronize_broadcast_to_tactical(stage1, game=game, data_dir=data_dir)

    # Stage 2: Extract multi-modal physics & matrix features
    features = extract_physics_feature_dict(stage1, st_priors)

    # Stage 3 (Tier 1 & Tier 2): Detect physical regimes & compute calibrated emission log-probs
    regimes = detect_physical_regimes(features, stage1["half_blocks"])
    tier1_log_probs = compute_tier1_emission_log_probs(features, regimes)
    blended_log_probs = fit_and_predict_tier2_emissions(
        features, regimes, tier1_log_probs, stage1["half_blocks"]
    )

    # Stage 4: Time-Inhomogeneous Viterbi/HSMM decoding, deflection filtering & boundary snapping
    labels = decode_full_match(
        blended_log_probs, features, regimes, stage1["half_blocks"]
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
        "log_probs": blended_log_probs,
    }
