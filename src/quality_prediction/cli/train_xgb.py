from __future__ import annotations

import argparse
from pathlib import Path

from quality_prediction.modeling.xgb import (
    TrainingConfig, TrainIOConfig, HPOConfig, WeightingConfig, FeatureSetConfig,
    FeatureSelectionConfig,  # <-- add this
    train_xgb_models
)



def main() -> None:
    ap = argparse.ArgumentParser("Train XGBoost models for target_* columns from dataset CSVs.")

    ap.add_argument("--train-csv", action="append", type=Path, required=True,
                    help="Training CSV path. Can be passed multiple times.")
    ap.add_argument("--eval-csv", action="append", type=Path, default=[],
                    help="Optional external eval CSV path(s).")

    ap.add_argument("--model-dir", type=Path, default=Path("./models"))
    ap.add_argument("--log-dir", type=Path, default=Path("./logs"))
    ap.add_argument("--feature-analysis-dir", type=Path, default=Path("./feature_analysis"))

    ap.add_argument("--metrics-log-csv", type=Path, default=None)
    ap.add_argument("--summary-metrics-csv", type=Path, default=None)

    ap.add_argument("--targets", type=str, default="",
                    help="Comma-separated list of target columns. Default: all columns starting with target_")

    ap.add_argument("--feature-sets", type=str, default="single_htr_line_score_mean,confidence_only,ngram_only,full",
                    help="Comma-separated feature set names.")
    ap.add_argument("--single-feature-name", type=str, default="htr_line_score_mean")

        # -------------------------
    # Feature selection
    # -------------------------
    ap.add_argument("--feature-selection", action="store_true",
                    help="Enable feature selection (corr prune + permutation importance + optional stability).")
    ap.add_argument("--fs-apply-to", type=str, default="full, json_model_only",
                    help="Comma-separated feature sets to apply feature selection to. Default: full")

    ap.add_argument("--fs-corr-prune", action="store_true",
                    help="Enable correlation pruning before training (train-only).")
    ap.add_argument("--fs-corr-threshold", type=float, default=0.98,
                    help="Absolute correlation threshold for pruning. Default: 0.98")

    ap.add_argument("--fs-perm-repeats", type=int, default=5,
                    help="Permutation importance repeats. Default: 5")
    ap.add_argument("--fs-top-k", type=int, default=100,
                    help="Keep top-K features by permutation importance. Default: 100")

    ap.add_argument("--fs-stability-runs", type=int, default=0,
                    help="Stability selection runs (0 disables). Default: 0")
    ap.add_argument("--fs-stability-min-freq", type=float, default=0.6,
                    help="Min frequency to keep a feature in stability selection. Default: 0.6")
    ap.add_argument("--fs-stability-top-k", type=int, default=None,
                    help="Top-K used inside stability runs (defaults to --fs-top-k).")

    ap.add_argument("--fs-retrain-mode", choices=["refit_best_params", "small_hpo"], default="refit_best_params",
                    help="How to retrain after selecting features.")
    ap.add_argument("--fs-retrain-trials", type=int, default=30,
                    help="Trials for retraining if fs-retrain-mode=small_hpo. Default: 30")

    ap.add_argument("--fs-match-select-by", action="store_true",
                    help="If set, permutation importance uses wmae when select-by=wmae and weights enabled.")


    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--early-stopping-rounds", type=int, default=40)
    ap.add_argument("--n-trials-full", type=int, default=100)
    ap.add_argument("--n-trials-baseline", type=int, default=30)
    ap.add_argument("--select-by", choices=["mae", "wmae"], default="mae")

    ap.add_argument("--weights", action="store_true", help="Enable sample weighting (for bow-like targets in [0,1]).")
    ap.add_argument("--weight-alpha", type=float, default=5.0)
    ap.add_argument("--weight-p", type=float, default=2.0)
    ap.add_argument("--weight-clip-max", type=float, default=50.0)

    ap.add_argument("--no-constant-baseline", action="store_true")
    ap.add_argument("--no-feature-importance", action="store_true")

    args = ap.parse_args()

    metrics_log_csv = args.metrics_log_csv or (args.log_dir / "xgb_hpo_metrics.csv")
    summary_metrics_csv = args.summary_metrics_csv or (args.log_dir / "xgb_summary_metrics.csv")

    fs_apply_to = [s.strip() for s in args.fs_apply_to.split(",") if s.strip()]
    feature_selection = FeatureSelectionConfig(
        enabled=args.feature_selection,
        apply_to_feature_sets=tuple(fs_apply_to) if fs_apply_to else ("full",),

        corr_prune_enabled=args.fs_corr_prune,
        corr_threshold=args.fs_corr_threshold,

        perm_enabled=True,  # controlled by enabled anyway
        perm_repeats=args.fs_perm_repeats,
        top_k=args.fs_top_k,

        match_hpo_select_by=args.fs_match_select_by,

        stability_runs=args.fs_stability_runs,
        stability_top_k=args.fs_stability_top_k,
        stability_min_freq=args.fs_stability_min_freq,

        retrain_mode=args.fs_retrain_mode,
        retrain_trials=args.fs_retrain_trials,
        random_state=42,
    )

    io = TrainIOConfig(
        train_csvs=args.train_csv,
        eval_csvs=args.eval_csv,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        feature_analysis_dir=args.feature_analysis_dir,
        metrics_log_csv=metrics_log_csv,
        summary_metrics_csv=summary_metrics_csv,
    )

    hpo = HPOConfig(
        early_stopping_rounds=args.early_stopping_rounds,
        val_size=args.val_size,
        n_trials_full=args.n_trials_full,
        n_trials_baseline=args.n_trials_baseline,
        select_by=args.select_by,
    )

    weighting = WeightingConfig(
        enabled=args.weights,
        alpha=args.weight_alpha,
        p=args.weight_p,
        clip_max=args.weight_clip_max,
    )

    feature_sets = FeatureSetConfig(
        sets=[s.strip() for s in args.feature_sets.split(",") if s.strip()],
        single_feature_name=args.single_feature_name,
    )

    targets = [t.strip() for t in args.targets.split(",") if t.strip()] if args.targets else []

    cfg = TrainingConfig(
        io=io,
        hpo=hpo,
        weighting=weighting,
        feature_sets=feature_sets,
        targets=targets,
        baseline_constant_mean=(not args.no_constant_baseline),
        write_feature_importance=(not args.no_feature_importance),
        feature_selection=feature_selection,
    )

    train_xgb_models(cfg)
