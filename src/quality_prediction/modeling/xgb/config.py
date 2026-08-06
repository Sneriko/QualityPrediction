from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Dict, Any


@dataclass(frozen=True)
class WeightingConfig:
    enabled: bool = False
    # weight = 1 + alpha * (1 - y)^p  (applied to bow-like targets in [0,1])
    alpha: float = 5.0
    p: float = 2.0
    clip_min: float = 1.0
    clip_max: float = 50.0
    report_weighted_mae: bool = True
    apply_only_if_name_contains: str = "bow"  # keep your heuristic


@dataclass(frozen=True)
class HPOConfig:
    base_params: Dict[str, Any] = field(default_factory=lambda: dict(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=10,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        eval_metric="mae",
    ))
    early_stopping_rounds: int = 40
    val_size: float = 0.1

    # random search budgets
    n_trials_full: int = 100
    n_trials_baseline: int = 30

    # selection criterion
    select_by: str = "mae"  # "mae" or "wmae"


@dataclass(frozen=True)
class TrainIOConfig:
    train_csvs: List[Path]
    eval_csvs: List[Path] = field(default_factory=list)

    model_dir: Path = Path("./models")
    log_dir: Path = Path("./logs")
    feature_analysis_dir: Path = Path("./feature_analysis")

    metrics_log_csv: Path = Path("./logs/xgb_hpo_metrics.csv")
    summary_metrics_csv: Path = Path("./logs/xgb_summary_metrics.csv")

    # write per-target prediction CSVs
    write_val_predictions: bool = True

    # include page_id if present
    page_id_col: str = "page_id"


@dataclass(frozen=True)
class FeatureSetConfig:
    """
    Defines which feature sets you want to train.
    Names are resolved by quality_prediction.modeling.xgb.feature_sets.get_feature_set().
    """
    sets: Sequence[str] = ("single_htr_line_score_mean", "confidence_only", "image_only", "dit_only", "ngram_only", "full")
    # used by "single_htr_line_score_mean"
    single_feature_name: str = "htr_line_score_mean"

@dataclass(frozen=True)
class FeatureSelectionConfig:
    enabled: bool = False

    # Apply to these named feature sets (e.g. only "full")
    apply_to_feature_sets: Sequence[str] = ("full",)

    # Step A: correlation pruning on TRAIN only (then apply same columns to val)
    corr_prune_enabled: bool = True
    corr_threshold: float = 0.98

    # Step B: permutation importance on VAL, select top-k
    perm_enabled: bool = True
    perm_repeats: int = 5
    top_k: int = 60  # keep the top-K most important features

    # If True, metric used matches your selection criterion:
    # - if weighting enabled and hpo.select_by == "wmae": uses weighted MAE
    # - else uses MAE
    match_hpo_select_by: bool = True

    # Optional Step C: stability selection (bootstrap TRAIN, recompute top-K, keep stable ones)
    stability_runs: int = 0            # 0 disables
    stability_top_k: Optional[int] = None  # if None, uses top_k
    stability_min_freq: float = 0.6     # keep features appearing in top-K in >=60% runs

    # After selecting features, retrain:
    # - "refit_best_params" = refit with best_params from the pre-selection search (fast)
    # - "small_hpo" = run a small random search again on the reduced set (slower but sometimes better)
    retrain_mode: str = "refit_best_params"  # or "small_hpo"
    retrain_trials: int = 30

    random_state: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    io: TrainIOConfig
    hpo: HPOConfig = HPOConfig()
    weighting: WeightingConfig = WeightingConfig()
    feature_sets: FeatureSetConfig = FeatureSetConfig()

    targets: Sequence[str] = ()
    baseline_constant_mean: bool = True
    write_feature_importance: bool = True

    feature_selection: FeatureSelectionConfig = FeatureSelectionConfig()

