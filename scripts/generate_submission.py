"""Generate and validate the Match-Sentiment-Adaptive Hybrid (Physics + ML + True Viterbi) submission CSV."""

import argparse
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline import run_sentiment_hybrid_pipeline  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "evaluation", "results", "sentiment_hybrid_v1.csv"),
        help="Output submission CSV path",
    )
    parser.add_argument("--data-dir", default=None, help="Optional data directory override")
    args = parser.parse_args()

    print(
        "Running 4-Stage Match-Sentiment-Adaptive Hybrid Pipeline "
        "(Physics Matrix + Specialist ML + Sentiment Router + True Viterbi)..."
    )
    result = run_sentiment_hybrid_pipeline(data_dir=args.data_dir, out_csv=args.out)
    labels = result["labels"]
    cv = result["spec_cv_metrics"]
    rt = result["routing_summary"]

    df = pm.read_label_csv(args.out)
    err = pm.validate_submission(df, len(labels))
    if err is not None:
        raise RuntimeError(f"Submission validation failed: {err}")

    counts = {lbl: int(np.sum(labels == lbl)) for lbl in ("H", "A", "D", "X")}
    print(
        f"Specialist Live CV -> GBDT_Alive_Full: {100*cv['oof_ha_acc_full']:.2f}% | "
        f"GBDT_Alive_NoBall: {100*cv['oof_ha_acc_noball']:.2f}% | "
        f"Mean w_ml: Calm={rt['mean_w_ml_calm']:.3f}, Chaos={rt['mean_w_ml_chaos']:.3f}, "
        f"Tempo={rt['mean_w_ml_tempo']:.3f}"
    )
    print(f"Wrote valid submission -> {os.path.relpath(args.out, ROOT)}")
    print(f"Frame counts: H={counts['H']}  A={counts['A']}  D={counts['D']}  X={counts['X']}")


if __name__ == "__main__":
    main()
