from __future__ import annotations

import argparse

from quality_prediction.metrics.evaluator import PageEvaluator


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate targets over all matching GT/PRED pairs.")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--lambda-ins", type=float, default=1.0)
    ap.add_argument("--log", default="", help="Optional log file path (empty disables logging)")
    args = ap.parse_args()

    evaluator = PageEvaluator(gt_dir=args.gt, pred_dir=args.pred, log_path=args.log)
    evaluator.evaluate_all(lambda_ins=args.lambda_ins)
