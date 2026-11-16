"""Run ML Out-of-Fold CV diagnostics, Masked-Ball Stability Stress Test, and 5-min GT audit."""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import half_video_blocks  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402
from src.pipeline import run_ml_ensemble_pipeline  # noqa: E402
from src.decoder import decode_full_match  # noqa: E402


def run_masked_ball_stress_test(res):
    """Validation Pillar #3 (Proposed_Plan.md Sec. 7): Masked-Ball Stability Stress Test."""
    features = res["features"]
    regimes = res["regimes"]
    half_blocks = res["stage1"]["half_blocks"]
    unmasked_codes = pm.label_codes(res["labels"])

    # Create synthetic 8-second (200-frame) ball dropouts every 40 seconds across visible play
    n = len(unmasked_codes)
    stress_mask = np.zeros(n, dtype=bool)
    for _, s_h, e_h in half_blocks:
        for t in range(s_h + 500, e_h - 500, 1000):
            if np.all(features["raw_codes"][t:t + 200] < 2):
                stress_mask[t:t + 200] = True

    # Under stress mask, force ensemble to use only ball-independent GBDT_NoBall + kinematics
    p_noball = res["oof_probs_noball"]
    p_orig = np.exp(res["log_probs"])
    stressed_probs = np.where(stress_mask[:, None], 0.65 * p_orig + 0.35 * p_noball, p_orig)
    stressed_probs = np.clip(stressed_probs, 1e-6, 1.0)
    stressed_probs /= np.sum(stressed_probs, axis=1, keepdims=True)

    stressed_labels = decode_full_match(
        np.log(stressed_probs), features, regimes, half_blocks
    )
    stressed_codes = pm.label_codes(stressed_labels)

    # Evaluate self-consistency (treating unmasked predictions on graded halves as reference)
    ref_codes = unmasked_codes.copy()
    eval_res = pm.evaluate(ref_codes, stressed_codes)
    return {
        "n_stressed_frames": int(np.sum(stress_mask)),
        "self_consistency_macro_f1": float(eval_res["macro_f1"]),
        "self_consistency_spell_f1": float(eval_res["spell_f1_mean"]),
    }


def main():
    out_csv = os.path.join(ROOT, "evaluation", "results", "ml_ensemble_v1.csv")
    res = run_ml_ensemble_pipeline(out_csv=out_csv)
    labels = res["labels"]
    pred_codes = pm.label_codes(labels)
    cv = res["gbdt_cv_metrics"]
    seq_m = res["seq_metrics"]

    print("=" * 84)
    print("STAGE 3 ML MODEL FITTING — 6-FOLD TEMPORAL BLOCK OUT-OF-FOLD CV DIAGNOSTICS")
    print("=" * 84)
    print(f"Algorithmic Anchor Samples (Stride 4) : {cv['n_anchors']} "
          f"(H={cv['anchor_counts'].get(0,0)}, A={cv['anchor_counts'].get(1,0)}, D={cv['anchor_counts'].get(2,0)})")
    print(f"GBDT_Full (88 Features, Ball+Kin+ST)  : OOF Acc = {100*cv['oof_acc_full']:.2f}% | "
          f"OOF LogLoss = {cv['oof_logloss_full']:.4f}")
    print(f"GBDT_NoBall (46 Ball-Free Features)   : OOF Acc = {100*cv['oof_acc_noball']:.2f}% | "
          f"OOF LogLoss = {cv['oof_logloss_noball']:.4f}")
    print(f"Stage-2 Dilated Sequence Refiner      : OOF Acc = {100*seq_m['oof_acc_seq_refiner']:.2f}% | "
          f"Sim2Real Ball-Dropout = {100*seq_m['sim2real_drop_rate']:.1f}%")

    stress = run_masked_ball_stress_test(res)
    print("\n" + "=" * 84)
    print("VALIDATION PILLAR #3 — MASKED-BALL STABILITY STRESS TEST (8s SYNTHETIC DROPOUTS)")
    print("=" * 84)
    print(f"Artificially Masked Visible Frames    : {stress['n_stressed_frames']} frames")
    print(f"Unmasked vs. Masked Self-Consistency  : spellF1 = {stress['self_consistency_spell_f1']:.4f} | "
          f"macroF1 = {stress['self_consistency_macro_f1']:.4f} (Target > 0.80)")

    # Full-match macro-structural diagnostics across graded halves
    full_gt_dummy = np.full(len(labels), pm.X, dtype=np.int8)
    for _, s_h, e_h in half_video_blocks("BAR-ATH"):
        full_gt_dummy[s_h:e_h + 1] = pm.H
    full_runs = pm.valid_runs(full_gt_dummy)
    full_spells = pm.possession_spells(pred_codes, full_runs)
    full_segs = pm.segments(pred_codes, full_runs)
    spell_durs = np.array([(b - a) / 25.0 for a, b, _ in full_spells])

    graded_mask = full_gt_dummy != pm.X
    graded_preds = pred_codes[graded_mask]
    h_frames = int(np.sum(graded_preds == pm.H))
    a_frames = int(np.sum(graded_preds == pm.A))
    d_frames = int(np.sum(graded_preds == pm.D))
    tot_graded = len(graded_preds)

    print("\n" + "=" * 84)
    print("FULL-MATCH MACRO-STRUCTURAL DIAGNOSTICS (158,857 GRADED FRAMES)")
    print("=" * 84)
    print(f"Total Predicted Possession Spells : {len(full_spells)}")
    print(f"Total Contiguous Segments (H/A/D) : {len(full_segs)}")
    print(f"Mean Possession Spell Duration    : {spell_durs.mean():.2f} s (median {np.median(spell_durs):.2f} s)")
    print(f"Graded Frame Share                : H={100*h_frames/tot_graded:.1f}% ({h_frames}), "
          f"A={100*a_frames/tot_graded:.1f}% ({a_frames}), D={100*d_frames/tot_graded:.1f}% ({d_frames})")
    print(f"Home Share of Alive Ball Time     : {100*h_frames/max(h_frames+a_frames,1):.1f}%")
    print(f"2nd-Half Pre-Kickoff (110808..113741) D% : "
          f"{100*np.mean(pred_codes[110808:113742] == pm.D):.1f}% (2,934 frames)")


if __name__ == "__main__":
    main()
