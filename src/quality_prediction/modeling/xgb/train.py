from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .config import TrainingConfig
from .data import load_training_data, get_train_val_indices_for_target
from .feature_sets import get_feature_set
from .weights import make_sample_weights, weighted_mae
from .hpo import random_search_xgb
from .analysis import compute_feature_importance_for_target
from .selection import (
    prune_correlated_features,
    permutation_importance_topk,
    stability_selection_topk,
)

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore


def _write_feature_selection_report(
    out_path: Path,
    *,
    importance_mean: Dict[str, float],
    importance_std: Dict[str, float],
    selected: List[str],
    freq: Optional[Dict[str, float]] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    sel_set = set(selected)
    for feat in sorted(importance_mean.keys()):
        rows.append(
            {
                "feature": feat,
                "perm_importance_mean": float(importance_mean.get(feat, 0.0)),
                "perm_importance_std": float(importance_std.get(feat, 0.0)),
                "stability_freq": (float(freq.get(feat, float("nan"))) if freq is not None else ""),
                "selected": int(feat in sel_set),
            }
        )
    pd.DataFrame(rows).sort_values("perm_importance_mean", ascending=False).to_csv(out_path, index=False)


def _append_csv_row(path: Path, fieldnames: Sequence[str], row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def _use_weighted_metric_for_selection(cfg: TrainingConfig) -> bool:
    fs_cfg = cfg.feature_selection
    if not getattr(fs_cfg, "match_hpo_select_by", True):
        return False
    return bool(
        cfg.weighting.enabled
        and cfg.weighting.report_weighted_mae
        and (cfg.hpo.select_by == "wmae")
    )


def train_xgb_models(cfg: TrainingConfig) -> None:
    io = cfg.io
    io.model_dir.mkdir(parents=True, exist_ok=True)
    io.log_dir.mkdir(parents=True, exist_ok=True)
    io.feature_analysis_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_training_data(io.train_csvs, io.eval_csvs, page_id_col=io.page_id_col)
    df = loaded.df
    X_all = loaded.X_all
    has_page_id = loaded.has_page_id

    targets = list(cfg.targets) if cfg.targets else list(loaded.target_cols)

    # per-trial log schema (matches your original)
    metrics_fields = [
        "target",
        "trial",
        "is_best_for_target",
        "val_mae",
        "val_wmae",
        "best_iteration",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
        "use_sample_weights",
        "weight_alpha",
        "weight_p",
        "weight_clip_max",
    ]

    def per_trial_logger(**row: Any) -> None:
        _append_csv_row(io.metrics_log_csv, metrics_fields, row)

    # summary schema
    summary_fields = [
        "target",
        "model_name",
        "feature_set",
        "n_features",
        "n_train",
        "n_val",
        "val_mae",
        "val_wmae",
        "best_iteration",
        "params_json",
        "notes",
    ]

    fs_cfg = cfg.feature_selection
    use_weighted_metric = _use_weighted_metric_for_selection(cfg)

    for target_col in targets:
        if target_col not in df.columns:
            continue

        y = df[target_col]
        mask = y.notna()
        if int(mask.sum()) < 10:
            continue

        train_idx, val_idx = get_train_val_indices_for_target(
            df=df,
            X=X_all,
            target_col=target_col,
            use_external_eval=loaded.use_external_eval,
            val_size=cfg.hpo.val_size,
        )
        if len(train_idx) < 10 or len(val_idx) < 1:
            continue

        y_train = y.loc[train_idx]
        y_val = y.loc[val_idx]

        w_train = make_sample_weights(y_train, target_col, cfg.weighting)
        w_val = make_sample_weights(y_val, target_col, cfg.weighting)

        # -------------------------
        # Baseline: constant mean
        # -------------------------
        if cfg.baseline_constant_mean:
            global_mean = float(y.loc[mask].mean())
            y_pred_const = np.full(shape=len(val_idx), fill_value=global_mean, dtype=float)
            mae_const = float(mean_absolute_error(y_val, y_pred_const))
            wmae_const = float("nan")
            if cfg.weighting.report_weighted_mae and cfg.weighting.enabled:
                wmae_const = float(weighted_mae(y_val.values, y_pred_const, w_val))

            _append_csv_row(
                io.summary_metrics_csv,
                summary_fields,
                dict(
                    target=target_col,
                    model_name="baseline_constant_mean",
                    feature_set="none",
                    n_features=0,
                    n_train=len(train_idx),
                    n_val=len(val_idx),
                    val_mae=mae_const,
                    val_wmae=(wmae_const if (cfg.weighting.report_weighted_mae and cfg.weighting.enabled) else ""),
                    best_iteration="",
                    params_json=json.dumps({"mean": global_mean}),
                    notes="Predicts global mean over all non-NaN rows.",
                ),
            )

            if io.write_val_predictions:
                out_pred = io.log_dir / f"val_predictions_baseline_constant_mean_{target_col}.csv"
                rows = []
                for idx_row, pred_val in zip(val_idx, y_pred_const):
                    row = {
                        "row_index": int(idx_row),
                        "gt": float(y_val.loc[idx_row]),
                        "pred": float(pred_val),
                        "target": target_col,
                        "model_name": "baseline_constant_mean",
                        "feature_set": "none",
                        "split": df.loc[idx_row, "split"],
                        "mean_used": global_mean,
                    }
                    if has_page_id:
                        row["page_id"] = df.loc[idx_row, io.page_id_col]
                    rows.append(row)
                pd.DataFrame(rows).to_csv(out_pred, index=False)

        # -------------------------
        # Feature-set models
        # -------------------------
        for feat_set in cfg.feature_sets.sets:
            X_feat = get_feature_set(
                feat_set,
                X_all,
                single_feature_name=cfg.feature_sets.single_feature_name,
            )
            X_train0 = X_feat.loc[train_idx]
            X_val0 = X_feat.loc[val_idx]
            if X_train0.shape[1] == 0:
                continue

            is_full = feat_set == "full"
            n_trials = cfg.hpo.n_trials_full if is_full else cfg.hpo.n_trials_baseline
            model_name = "xgb_hpo_full" if is_full else f"xgb_hpo_{feat_set}"

            # only log per-trial details for full model (keeps logs sane)
            trial_logger = per_trial_logger if is_full else None
            target_tag = f"{target_col}__{feat_set}" if is_full else target_col

            # -------------------------------------------------
            # Optional: correlation pruning (train-only)
            # -------------------------------------------------
            do_fs_for_this = bool(
                getattr(fs_cfg, "enabled", False)
                and (feat_set in getattr(fs_cfg, "apply_to_feature_sets", ("full",)))
            )

            X_train = X_train0
            X_val = X_val0

            corr_pruned_cols: Optional[List[str]] = None
            if do_fs_for_this and getattr(fs_cfg, "corr_prune_enabled", True) and X_train.shape[1] > 1:
                corr_pruned_cols = prune_correlated_features(
                    X_train,
                    threshold=float(getattr(fs_cfg, "corr_threshold", 0.98)),
                )
                if len(corr_pruned_cols) >= 1:
                    X_train = X_train[corr_pruned_cols]
                    X_val = X_val[corr_pruned_cols]

            # -------------------------
            # Train base model (possibly corr-pruned)
            # -------------------------
            best_model, best_params, best_mae, best_wmae = random_search_xgb(
                X_train=X_train,
                y_train=y_train,
                w_train=w_train,
                X_val=X_val,
                y_val=y_val,
                w_val=w_val,
                hpo=cfg.hpo,
                weighting=cfg.weighting,
                target_name=target_tag,
                n_trials=n_trials,
                per_trial_logger=trial_logger,
            )

            y_pred = best_model.predict(X_val)
            mae = float(mean_absolute_error(y_val, y_pred))

            wmae = float("nan")
            if cfg.weighting.report_weighted_mae and cfg.weighting.enabled:
                wmae = float(weighted_mae(y_val.values, y_pred, w_val))

            best_iter = getattr(best_model, "best_iteration", None)
            if best_iter is None:
                best_iter = best_params.get("n_estimators", "")

            # save model (one per target + feature_set)
            out_model = io.model_dir / f"{model_name}_{target_col}.joblib"
            joblib.dump(best_model, out_model)

            # write predictions
            if io.write_val_predictions:
                out_pred = io.log_dir / f"val_predictions_{model_name}_{target_col}.csv"
                rows = []
                for idx_row, pred_val in zip(val_idx, y_pred):
                    row = {
                        "row_index": int(idx_row),
                        "gt": float(y_val.loc[idx_row]),
                        "pred": float(pred_val),
                        "target": target_col,
                        "model_name": model_name,
                        "feature_set": feat_set,
                        "split": df.loc[idx_row, "split"],
                    }
                    if has_page_id:
                        row["page_id"] = df.loc[idx_row, io.page_id_col]
                    rows.append(row)
                pd.DataFrame(rows).to_csv(out_pred, index=False)

            # summary row
            notes = ""
            if corr_pruned_cols is not None:
                notes = f"corr_prune=1; corr_threshold={float(getattr(fs_cfg, 'corr_threshold', 0.98))}; n_corr_pruned={int(X_train.shape[1])}"

            _append_csv_row(
                io.summary_metrics_csv,
                summary_fields,
                dict(
                    target=target_col,
                    model_name=model_name,
                    feature_set=feat_set,
                    n_features=int(X_train.shape[1]),
                    n_train=len(train_idx),
                    n_val=len(val_idx),
                    val_mae=mae,
                    val_wmae=(wmae if (cfg.weighting.report_weighted_mae and cfg.weighting.enabled) else ""),
                    best_iteration=int(best_iter) if str(best_iter).isdigit() else best_iter,
                    params_json=json.dumps(best_params),
                    notes=notes,
                ),
            )

            # feature importance (optional; usually only for full)
            if cfg.write_feature_importance and is_full:
                df_mask = df.loc[y.notna()].copy()
                X_mask = X_all.loc[y.notna()]
                _ = compute_feature_importance_for_target(
                    target_col=target_col,
                    model=best_model,
                    X=X_mask,
                    df=df_mask,
                    out_dir=io.feature_analysis_dir,
                    suffix=feat_set,
                )

            # -------------------------------------------------
            # Optional: permutation-importance feature selection
            # Trains an additional model with suffix "_fs"
            # -------------------------------------------------
            if (
                do_fs_for_this
                and getattr(fs_cfg, "perm_enabled", True)
                and X_train.shape[1] > 1
                and XGBRegressor is not None
            ):
                perm_repeats = int(getattr(fs_cfg, "perm_repeats", 5))
                top_k = int(getattr(fs_cfg, "top_k", 60))
                top_k = max(1, min(top_k, int(X_train.shape[1])))

                # For stability selection, we'll use best_params to keep it cheap & comparable.
                def _model_factory():
                    m = XGBRegressor(**best_params)
                    m.set_params(early_stopping_rounds=cfg.hpo.early_stopping_rounds)
                    return m

                freq: Optional[Dict[str, float]] = None
                selected_cols: List[str]

                stability_runs = int(getattr(fs_cfg, "stability_runs", 0) or 0)
                if stability_runs > 0:
                    stab_top_k = getattr(fs_cfg, "stability_top_k", None)
                    stab_k = int(stab_top_k) if stab_top_k is not None else top_k
                    stab_k = max(1, min(stab_k, int(X_train.shape[1])))
                    min_freq = float(getattr(fs_cfg, "stability_min_freq", 0.6))

                    selected_cols, freq = stability_selection_topk(
                        model_factory=_model_factory,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        w_train=(w_train if cfg.weighting.enabled else None),
                        w_val=(w_val if cfg.weighting.enabled else None),
                        runs=stability_runs,
                        repeats=perm_repeats,
                        top_k=stab_k,
                        min_freq=min_freq,
                        random_state=int(getattr(fs_cfg, "random_state", 42)),
                        use_weighted_metric=use_weighted_metric,
                    )

                    # still compute permutation importances once for reporting
                    perm = permutation_importance_topk(
                        best_model,
                        X_val=X_val,
                        y_val=y_val,
                        w_val=(w_val if cfg.weighting.enabled else None),
                        repeats=perm_repeats,
                        top_k=top_k,
                        random_state=int(getattr(fs_cfg, "random_state", 42)),
                        use_weighted_metric=use_weighted_metric,
                    )
                else:
                    perm = permutation_importance_topk(
                        best_model,
                        X_val=X_val,
                        y_val=y_val,
                        w_val=(w_val if cfg.weighting.enabled else None),
                        repeats=perm_repeats,
                        top_k=top_k,
                        random_state=int(getattr(fs_cfg, "random_state", 42)),
                        use_weighted_metric=use_weighted_metric,
                    )
                    selected_cols = perm.selected_features

                if len(selected_cols) == 0:
                    selected_cols = list(X_train.columns[:1])

                # write FS report
                fs_report = io.feature_analysis_dir / f"feature_selection_{target_col}_{feat_set}.csv"
                _write_feature_selection_report(
                    fs_report,
                    importance_mean=perm.importance_mean,
                    importance_std=perm.importance_std,
                    selected=selected_cols,
                    freq=freq,
                )

                X_train_fs = X_train[selected_cols]
                X_val_fs = X_val[selected_cols]

                retrain_mode = str(getattr(fs_cfg, "retrain_mode", "refit_best_params"))
                model_name_fs = f"{model_name}_fs"

                if retrain_mode == "small_hpo":
                    retrain_trials = int(getattr(fs_cfg, "retrain_trials", 30))
                    best_model_fs, best_params_fs, _, _ = random_search_xgb(
                        X_train=X_train_fs,
                        y_train=y_train,
                        w_train=w_train,
                        X_val=X_val_fs,
                        y_val=y_val,
                        w_val=w_val,
                        hpo=cfg.hpo,
                        weighting=cfg.weighting,
                        target_name=f"{target_col}__{feat_set}__fs",
                        n_trials=retrain_trials,
                        per_trial_logger=None,
                    )
                else:
                    best_params_fs = dict(best_params)
                    best_model_fs = XGBRegressor(**best_params_fs)
                    best_model_fs.set_params(early_stopping_rounds=cfg.hpo.early_stopping_rounds)

                    fit_kwargs = dict(
                        X=X_train_fs,
                        y=y_train,
                        eval_set=[(X_val_fs, y_val)],
                        verbose=False,
                    )
                    if cfg.weighting.enabled:
                        fit_kwargs["sample_weight"] = w_train
                    best_model_fs.fit(**fit_kwargs)

                y_pred_fs = best_model_fs.predict(X_val_fs)
                mae_fs = float(mean_absolute_error(y_val, y_pred_fs))

                wmae_fs = float("nan")
                if cfg.weighting.report_weighted_mae and cfg.weighting.enabled:
                    wmae_fs = float(weighted_mae(y_val.values, y_pred_fs, w_val))

                best_iter_fs = getattr(best_model_fs, "best_iteration", None)
                if best_iter_fs is None:
                    best_iter_fs = best_params_fs.get("n_estimators", "")

                out_model_fs = io.model_dir / f"{model_name_fs}_{target_col}.joblib"
                joblib.dump(best_model_fs, out_model_fs)

                if io.write_val_predictions:
                    out_pred_fs = io.log_dir / f"val_predictions_{model_name_fs}_{target_col}.csv"
                    rows = []
                    for idx_row, pred_val in zip(val_idx, y_pred_fs):
                        row = {
                            "row_index": int(idx_row),
                            "gt": float(y_val.loc[idx_row]),
                            "pred": float(pred_val),
                            "target": target_col,
                            "model_name": model_name_fs,
                            "feature_set": feat_set,
                            "split": df.loc[idx_row, "split"],
                            "n_features_selected": int(X_train_fs.shape[1]),
                        }
                        if has_page_id:
                            row["page_id"] = df.loc[idx_row, io.page_id_col]
                        rows.append(row)
                    pd.DataFrame(rows).to_csv(out_pred_fs, index=False)

                fs_notes_parts = [
                    "feature_selection=1",
                    f"top_k={top_k}",
                    f"perm_repeats={perm_repeats}",
                    f"stability_runs={stability_runs}",
                    f"retrain_mode={retrain_mode}",
                ]
                if corr_pruned_cols is not None:
                    fs_notes_parts.append(
                        f"corr_prune=1; corr_threshold={float(getattr(fs_cfg, 'corr_threshold', 0.98))}"
                    )

                _append_csv_row(
                    io.summary_metrics_csv,
                    summary_fields,
                    dict(
                        target=target_col,
                        model_name=model_name_fs,
                        feature_set=feat_set,
                        n_features=int(X_train_fs.shape[1]),
                        n_train=len(train_idx),
                        n_val=len(val_idx),
                        val_mae=mae_fs,
                        val_wmae=(wmae_fs if (cfg.weighting.report_weighted_mae and cfg.weighting.enabled) else ""),
                        best_iteration=int(best_iter_fs) if str(best_iter_fs).isdigit() else best_iter_fs,
                        params_json=json.dumps(best_params_fs),
                        notes="; ".join(fs_notes_parts),
                    ),
                )
