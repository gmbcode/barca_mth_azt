"""Generate and validate the Multi-Tier ML Ensemble (GBDT_Full + GBDT_NoBall + EM + MS-TCN) submission CSV."""

import argparse
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline import run_ml_ensemble_pipeline  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "evaluation", "results", "ml_ensemble_v1.csv"),
        help="Output submission CSV path",
    )
    parser.add_argument("--data-dir", default=None, help="Optional data directory override")
    args = parser.parse_args()

    print("Running 4-Stage Multi-Tier ML Ensemble Pipeline (XGBoost GBDT_Full + GBDT_NoBall + EM + Dilated Sequence MLP)...")
    result = run_ml_ensemble_pipeline(data_dir=args.data_dir, out_csv=args.out)
    labels = result["labels"]
    cv = result["gbdt_cv_metrics"]

    df = pm.read_label_csv(args.out)
    err = pm.validate_submission(df, len(labels))
    if err is not None:
        raise RuntimeError(f"Submission validation failed: {err}")

    counts = {lbl: int(np.sum(labels == lbl)) for lbl in ("H", "A", "D", "X")}
    print(f"OOF CV Accuracy -> GBDT_Full (88 feat): {100*cv['oof_acc_full']:.2f}% | "
          f"GBDT_NoBall Student (46 feat): {100*cv['oof_acc_noball']:.2f}%")
    print(f"Wrote valid submission -> {os.path.relpath(args.out, ROOT)}")
    print(f"Frame counts: H={counts['H']}  A={counts['A']}  D={counts['D']}  X={counts['X']}")


if __name__ == "__main__":
    main()
