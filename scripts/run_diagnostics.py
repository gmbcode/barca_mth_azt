"""Run full-match structural diagnostics and 5-minute public GT segment-by-segment audit."""

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import half_video_blocks  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402
from src.pipeline import run_physics_pipeline  # noqa: E402


def main():
    out_csv = os.path.join(ROOT, "evaluation", "results", "physics_matrix_v1.csv")
    res = run_physics_pipeline(out_csv=out_csv)
    labels = res["labels"]
    pred_codes = pm.label_codes(labels)

    # 1. Full-match macro-structural diagnostics across graded halves
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

    print("=" * 84)
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

    # 2. Public 5-minute Ground Truth segment-by-segment comparison
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt_codes = pm.label_codes(pm.read_label_csv(gt_path))
    smooth50_codes = pm.label_codes(
        pm.read_label_csv(os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv"))
    )

    pub_runs = pm.valid_runs(gt_codes)
    gt_segs = pm.segments(gt_codes, pub_runs)
    code_map = {0: "H", 1: "A", 2: "D", 3: "X"}

    print("\n" + "=" * 84)
    print("SEGMENT-BY-SEGMENT EXECUTION ON FIRST 5 MINUTES (FRAMES 11,185 .. 18,684)")
    print("=" * 84)
    header = (f"{'Seg':>3}  {'Frame Span':<16} {'Dur(s)':>6}  {'GT':>2}  "
              f"{'Smooth50 Acc':>12}  {'PhysicsMat Acc':>14}  {'PhysicsMat Dominant State'}")
    print(header)
    print("-" * len(header))

    for idx, (s, e, k) in enumerate(gt_segs, 1):
        gt_lbl = code_map[k]
        s50_acc = 100.0 * np.mean(smooth50_codes[s:e] == k)
        pm_acc = 100.0 * np.mean(pred_codes[s:e] == k)
        vals, cnts = np.unique(pred_codes[s:e], return_counts=True)
        dist_str = ", ".join(f"{code_map[v]}:{100*c/(e-s):.0f}%" for v, c in zip(vals, cnts))
        print(f"{idx:>3}  [{s:5d}..{e:5d})  {(e-s)/25:6.2f}  {gt_lbl:>2}  "
              f"{s50_acc:11.1f}%  {pm_acc:13.1f}%   {dist_str}")


if __name__ == "__main__":
    main()
