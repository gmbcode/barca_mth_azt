"""Run Match-Sentiment Hybrid CV, Routing Breakdown, Masked-Ball Stress Test & 3-Model Comparison."""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import half_video_blocks  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402
from src.pipeline import run_sentiment_hybrid_pipeline  # noqa: E402
from src.models.sentiment_hybrid import SENTIMENT_NAMES  # noqa: E402
from src.decoder import decode_sentiment_hybrid_match  # noqa: E402


def run_honest_masked_ball_stress_test(res):
    """Honest Masked-Ball Stability Stress Test using Decode-Time Ball Dropout via GBDT_Alive_NoBall."""
    features = res["features"]
    regimes = res["regimes"]
    sentiment = res["sentiment"]
    half_blocks = res["stage1"]["half_blocks"]
    unmasked_codes = pm.label_codes(res["labels"])

    n = len(unmasked_codes)
    stress_mask = np.zeros(n, dtype=bool)
    for _, s_h, e_h in half_blocks:
        for t in range(s_h + 500, e_h - 500, 1000):
            if np.all(features["raw_codes"][t:t + 200] < 2):
                stress_mask[t:t + 200] = True

    # Under synthetic occlusion stress: mask raw ball detections and replace live emissions inside stress_mask
    # with GBDT_Alive_NoBall kinematics + EM/Bridge priors instead of unmasked raw detections.
    p_nb = np.clip(res["oof_ha_noball"], 1e-3, 1.0 - 1e-3)
    logit_nb = np.clip(np.log(p_nb[:, 0] / p_nb[:, 1]), -2.0, 2.0)

    stressed_lp = res["log_probs"].copy()
    p_d = np.exp(stressed_lp[:, 2])
    p_alive = np.maximum(1.0 - p_d, 1e-4)
    p_h_nb = 1.0 / (1.0 + np.exp(-logit_nb))
    stressed_lp[stress_mask, 0] = np.log(np.clip(p_alive[stress_mask] * p_h_nb[stress_mask], 1e-6, 1.0))
    stressed_lp[stress_mask, 1] = np.log(np.clip(p_alive[stress_mask] * (1.0 - p_h_nb[stress_mask]), 1e-6, 1.0))

    stressed_regimes = dict(regimes)
    stressed_regimes["bridged_codes"] = np.where(stress_mask, 2, regimes["bridged_codes"])

    stressed_labels = decode_sentiment_hybrid_match(
        stressed_lp,
        features,
        stressed_regimes,
        sentiment,
        res["turnover_gate"],
        res["logit_fused"],
        half_blocks,
    )
    stressed_codes = pm.label_codes(stressed_labels)

    eval_res = pm.evaluate(unmasked_codes, stressed_codes)
    return {
        "n_stressed_frames": int(np.sum(stress_mask)),
        "self_consistency_macro_f1": float(eval_res["macro_f1"]),
        "self_consistency_spell_f1": float(eval_res["spell_f1_mean"]),
    }


def main():
    out_csv = os.path.join(ROOT, "evaluation", "results", "sentiment_hybrid_v1.csv")
    res = run_sentiment_hybrid_pipeline(out_csv=out_csv)
    labels = res["labels"]
    pred_codes = pm.label_codes(labels)
    cv = res["spec_cv_metrics"]
    seq_m = res["seq_metrics"]
    rt = res["routing_summary"]
    sent = res["sentiment"]

    print("=" * 88)
    print("STAGE 3 SPECIALIST ML & SENTIMENT HYBRID — 6-FOLD TEMPORAL BLOCK OOF CV DIAGNOSTICS")
    print("=" * 88)
    print(f"Sharp Live Anchor Samples (Stride 4)  : {cv['n_live_anchors']} (H/A binary live anchors)")
    print(f"GBDT_Alive_Full (Multimodal + 35%Drop): OOF Acc = {100*cv['oof_ha_acc_full']:.2f}% | "
          f"LogLoss = {cv['oof_ha_logloss_full']:.4f}")
    print(f"GBDT_Alive_NoBall (Half-Invariant Kin): OOF Acc = {100*cv['oof_ha_acc_noball']:.2f}% | "
          f"LogLoss = {cv['oof_ha_logloss_noball']:.4f}")
    print(f"Crisp Dilated SeqRefiner (offsets+-20): OOF Acc = {100*seq_m['oof_acc_seq_refiner']:.2f}% | "
          f"Sim2Real Drop = {100*seq_m['sim2real_drop_rate']:.1f}%")

    # Match Sentiment Regime Breakdown on Graded Halves
    full_gt_dummy = np.full(len(labels), pm.X, dtype=np.int8)
    for _, s_h, e_h in half_video_blocks("BAR-ATH"):
        full_gt_dummy[s_h:e_h + 1] = pm.H
    graded_mask = full_gt_dummy != pm.X
    tot_graded = int(np.sum(graded_mask))
    dom = sent["dominant_sentiment"][graded_mask]

    print("\n" + "=" * 88)
    print("MATCH SENTIMENT TENSOR (Psi_t) REGIME BREAKDOWN & DYNAMIC ROUTING WEIGHTS")
    print("=" * 88)
    for idx, name in enumerate(SENTIMENT_NAMES):
        cnt = int(np.sum(dom == idx))
        pct = 100.0 * cnt / max(tot_graded, 1)
        m_reg = graded_mask & (sent["dominant_sentiment"] == idx)
        mg = float(np.mean(res["turnover_gate"][m_reg])) if cnt > 0 else 0.0
        print(f"  [{idx}] {name:<20s}: {cnt:6d} frames ({pct:5.1f}%) | Mean Turnover Gate = {mg:.2f}")
    print(f"Mean Dynamic ML Weight w_ml(t) -> All={rt['mean_w_ml_all']:.3f} | "
          f"Calm={rt['mean_w_ml_calm']:.3f} | Chaos={rt['mean_w_ml_chaos']:.3f} | "
          f"Tempo={rt['mean_w_ml_tempo']:.3f} | Occl={rt['mean_w_ml_occl']:.3f}")

    stress = run_honest_masked_ball_stress_test(res)
    print("\n" + "=" * 88)
    print("VALIDATION PILLAR #3 — HONEST MASKED-BALL STABILITY STRESS TEST (8s DROPOUTS)")
    print("=" * 88)
    print(f"Artificially Masked Visible Frames    : {stress['n_stressed_frames']} frames")
    print(f"Unmasked vs. Masked Self-Consistency  : spellF1 = {stress['self_consistency_spell_f1']:.4f} | "
          f"macroF1 = {stress['self_consistency_macro_f1']:.4f}")

    # Full-match macro-structural diagnostics across graded halves
    full_runs = pm.valid_runs(full_gt_dummy)
    full_spells = pm.possession_spells(pred_codes, full_runs)
    full_segs = pm.segments(pred_codes, full_runs)
    spell_durs = np.array([(b - a) / 25.0 for a, b, _ in full_spells])

    graded_preds = pred_codes[graded_mask]
    h_frames = int(np.sum(graded_preds == pm.H))
    a_frames = int(np.sum(graded_preds == pm.A))
    d_frames = int(np.sum(graded_preds == pm.D))

    print("\n" + "=" * 88)
    print("FULL-MATCH MACRO-STRUCTURAL DIAGNOSTICS (158,857 GRADED FRAMES)")
    print("=" * 88)
    print(f"Total Predicted Possession Spells : {len(full_spells)}")
    print(f"Total Contiguous Segments (H/A/D) : {len(full_segs)}")
    print(f"Mean Possession Spell Duration    : {spell_durs.mean():.2f} s (median {np.median(spell_durs):.2f} s)")
    print(f"Graded Frame Share                : H={100*h_frames/tot_graded:.1f}% ({h_frames}), "
          f"A={100*a_frames/tot_graded:.1f}% ({a_frames}), D={100*d_frames/tot_graded:.1f}% ({d_frames})")
    print(f"Home Share of Alive Ball Time     : {100*h_frames/max(h_frames+a_frames,1):.1f}%")
    print(f"2nd-Half Pre-Kickoff (110808..113741) D% : "
          f"{100*np.mean(pred_codes[110808:113742] == pm.D):.1f}% (2,934 frames)")

    # Side-by-side 3-Model Public 5-Min GT Audit
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    if os.path.isfile(gt_path):
        gt_codes = pm.label_codes(pm.read_label_csv(gt_path))
        print("\n" + "=" * 88)
        print("THREE-MODEL BENCHMARK COMPARISON (5-MIN PUBLIC GT)")
        print("=" * 88)
        for model_name, fname in [
            ("1. Physics (`physics` branch)", "physics_matrix_v1.csv"),
            ("2. ML Ensemble (`model-fitting`)", "ml_ensemble_v1.csv"),
            ("3. Sentiment Hybrid (`sentiment-hybrid`)", "sentiment_hybrid_v1.csv"),
        ]:
            fpath = os.path.join(ROOT, "evaluation", "results", fname)
            if not os.path.isfile(fpath):
                continue
            mc = pm.label_codes(pm.read_label_csv(fpath))
            ev = pm.evaluate(gt_codes, mc)
            tr12 = ev["transitions"][12]["f1"]
            tr25 = ev["transitions"][25]["f1"]
            sp5 = ev["spell_f1"][0.5]["f1"]
            print(f"{model_name:<40s} | spellF1={ev['spell_f1_mean']:.4f} (@0.5={sp5:.4f}) | "
                  f"macroF1={ev['macro_f1']:.4f} | tau12={tr12:.4f} | tau25={tr25:.4f} | "
                  f"poss={ev['n_pred_spells']} (dur={ev['mean_spell_duration_s']:.1f}s)")


if __name__ == "__main__":
    main()
