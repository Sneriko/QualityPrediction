from __future__ import annotations

import os
import re
import csv
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

TRAIN_CSV_PATHS: List[str] = [
    "/home/coder/QualityPrediction/data/testsuite/xgboost/training_set/xgboost_eval_test_map_targets.csv"
]

# If you ever move the testsuite CSV here instead, you'll get a true external eval split:
EVAL_CSV_PATHS: List[str] = []

MODEL_DIR = "/home/coder/QualityPrediction/models"
LOG_DIR = "/home/coder/QualityPrediction/data/runs_map_target/logs"
FEAT_ANALYSIS_DIR = "/home/coder/QualityPrediction/data/runs_map_target/feature_analysis"

# Main per-trial log (as you had)
METRICS_LOG_CSV = os.path.join(LOG_DIR, "xgb_hpo_metrics_weighing.csv")

# NEW: one-row-per-model summary (easy to plot)
SUMMARY_METRICS_CSV = os.path.join(LOG_DIR, "baseline_summary_metrics.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FEAT_ANALYSIS_DIR, exist_ok=True)

# Base params used as defaults / center of search
BASE_XGB_PARAMS = dict(
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
)

EARLY_STOPPING_ROUNDS = 40
VAL_SIZE = 0.1
N_TRIALS_HPO = 100

# For baselines you may want fewer trials
N_TRIALS_BASELINES = 30

BOW_TARGETS = [
    "target_bow_f1",
    "target_bow_recall",
    "target_bow_precision",
    "target_map50_line",
    "target_map75_line",
    "target_map50_region",
    "target_map75_region",
]

MAP_TARGETS = [
    "target_map50_line",
    "target_map75_line",
    "target_map50_region",
    "target_map75_region",
]
# target_map50_line,target_map50_region,target_map75_line,target_map75_region
# -------------------------------------------------------------------
# Sample reweighting
# -------------------------------------------------------------------
USE_SAMPLE_WEIGHTS = False

WEIGHT_ALPHA = 5.0
WEIGHT_P = 2.0
WEIGHT_CLIP_MIN = 1.0
WEIGHT_CLIP_MAX = 50.0

REPORT_WEIGHTED_MAE = True


def make_sample_weights(y: pd.Series, target_name: str) -> np.ndarray:
    yv = y.astype(float).values

    if not USE_SAMPLE_WEIGHTS:
        return np.ones_like(yv, dtype=float)

    if "bow" not in target_name:
        return np.ones_like(yv, dtype=float)

    y_min = np.nanmin(yv)
    y_max = np.nanmax(yv)
    if y_min < -0.05 or y_max > 1.05:
        return np.ones_like(yv, dtype=float)

    w = 1.0 + WEIGHT_ALPHA * np.power((1.0 - np.clip(yv, 0.0, 1.0)), WEIGHT_P)
    w = np.clip(w, WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX)
    return w.astype(float)


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.asarray(w, dtype=float)
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        return float(np.mean(np.abs(y_true - y_pred)))
    return float(np.sum(w * np.abs(y_true - y_pred)) / w_sum)


# -------------------------------------------------------------------
# Helpers: logging
# -------------------------------------------------------------------

def append_metrics_row(path: str, row: Dict[str, object]) -> None:
    file_exists = os.path.exists(path)
    fieldnames = [
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
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_summary_row(path: str, row: Dict[str, object]) -> None:
    file_exists = os.path.exists(path)
    fieldnames = [
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
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# -------------------------------------------------------------------
# Hyperparameter search (supports optional weights; used for main + (optionally) baselines)
# -------------------------------------------------------------------

def random_search_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    w_val: np.ndarray,
    base_params: dict,
    n_trials: int,
    early_stopping_rounds: int,
    target_name: str,
    metrics_log_csv: Optional[str] = None,
    select_by: str = "mae",  # "mae" or "wmae"
) -> Tuple[XGBRegressor, Dict[str, float], float, float]:
    param_space = {
        "n_estimators":       [400, 600, 800, 1000, 1200, 1600],
        "max_depth":         [3, 4, 5, 6, 7],
        "learning_rate":     [0.01, 0.02, 0.03, 0.05, 0.1],
        "subsample":         [0.6, 0.8, 1.0],
        "colsample_bytree":  [0.6, 0.8, 1.0],
        "min_child_weight":  [1, 3, 5, 10],
        "gamma":             [0.0, 0.1, 0.3],
        "reg_alpha":         [0.0, 0.1, 0.5],
        "reg_lambda":        [0.5, 1.0, 2.0],
    }

    best_score = float("inf")
    best_params = base_params.copy()
    best_model: Optional[XGBRegressor] = None
    best_trial_idx = -1
    best_mae = np.nan
    best_wmae = np.nan

    for trial in range(1, n_trials + 1):
        params = base_params.copy()
        for key, values in param_space.items():
            params[key] = random.choice(values)

        model = XGBRegressor(**params)
        model.set_params(early_stopping_rounds=early_stopping_rounds)

        fit_kwargs = dict(
            X=X_train,
            y=y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        if USE_SAMPLE_WEIGHTS:
            fit_kwargs["sample_weight"] = w_train

        model.fit(**fit_kwargs)

        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)

        wmae = np.nan
        if REPORT_WEIGHTED_MAE and USE_SAMPLE_WEIGHTS:
            wmae = weighted_mae(y_val.values, y_pred, w_val)

        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None:
            best_iter = params["n_estimators"]

        # optional per-trial log (keep for main model)
        if metrics_log_csv is not None:
            row = {
                "target": target_name,
                "trial": trial,
                "is_best_for_target": 0,
                "val_mae": float(mae),
                "val_wmae": float(wmae) if REPORT_WEIGHTED_MAE and USE_SAMPLE_WEIGHTS else "",
                "best_iteration": int(best_iter),
                "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"],
                "learning_rate": params["learning_rate"],
                "subsample": params["subsample"],
                "colsample_bytree": params["colsample_bytree"],
                "min_child_weight": params.get("min_child_weight", np.nan),
                "gamma": params.get("gamma", np.nan),
                "reg_alpha": params.get("reg_alpha", np.nan),
                "reg_lambda": params.get("reg_lambda", np.nan),
                "use_sample_weights": int(USE_SAMPLE_WEIGHTS),
                "weight_alpha": WEIGHT_ALPHA,
                "weight_p": WEIGHT_P,
                "weight_clip_max": WEIGHT_CLIP_MAX,
            }
            append_metrics_row(metrics_log_csv, row)

        # selection criterion
        score = wmae if (select_by == "wmae" and REPORT_WEIGHTED_MAE and USE_SAMPLE_WEIGHTS) else mae

        if score < best_score:
            best_score = float(score)
            best_params = params
            best_model = model
            best_trial_idx = trial
            best_mae = float(mae)
            best_wmae = float(wmae)

    # mark best trial in the per-trial file (main model only)
    if best_model is not None and metrics_log_csv is not None:
        best_iter = getattr(best_model, "best_iteration", None)
        if best_iter is None:
            best_iter = best_params["n_estimators"]
        best_row = {
            "target": target_name,
            "trial": best_trial_idx,
            "is_best_for_target": 1,
            "val_mae": float(best_mae),
            "val_wmae": float(best_wmae) if REPORT_WEIGHTED_MAE and USE_SAMPLE_WEIGHTS else "",
            "best_iteration": int(best_iter),
            "n_estimators": best_params["n_estimators"],
            "max_depth": best_params["max_depth"],
            "learning_rate": best_params["learning_rate"],
            "subsample": best_params["subsample"],
            "colsample_bytree": best_params["colsample_bytree"],
            "min_child_weight": best_params.get("min_child_weight", np.nan),
            "gamma": best_params.get("gamma", np.nan),
            "reg_alpha": best_params.get("reg_alpha", np.nan),
            "reg_lambda": best_params.get("reg_lambda", np.nan),
            "use_sample_weights": int(USE_SAMPLE_WEIGHTS),
            "weight_alpha": WEIGHT_ALPHA,
            "weight_p": WEIGHT_P,
            "weight_clip_max": WEIGHT_CLIP_MAX,
        }
        append_metrics_row(metrics_log_csv, best_row)

    if best_model is None:
        raise RuntimeError("random_search_xgb failed to produce any model.")

    return best_model, best_params, best_mae, best_wmae


# -------------------------------------------------------------------
# Load data (multiple CSVs for train / eval)
# -------------------------------------------------------------------

def load_concat_csv(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Could not find CSV at {p!r}")
        print(f"Loading CSV: {p}")
        dfs.append(pd.read_csv(p))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


df_train = load_concat_csv(TRAIN_CSV_PATHS)
if df_train.empty:
    raise RuntimeError("No training data loaded. Check TRAIN_CSV_PATHS.")
df_train["split"] = "train"

if EVAL_CSV_PATHS:
    df_eval = load_concat_csv(EVAL_CSV_PATHS)
    if df_eval.empty:
        print("Warning: EVAL_CSV_PATHS specified but eval data is empty; falling back to internal split.")
        use_external_eval = False
        df = df_train.copy()
    else:
        df_eval["split"] = "eval"
        df = pd.concat([df_train, df_eval], ignore_index=True)
        use_external_eval = True
else:
    df = df_train.copy()
    use_external_eval = False

print(f"Total rows: {len(df)}  (train={len(df_train)}, eval={len(df) - len(df_train) if use_external_eval else 0})")

all_target_cols: List[str] = [c for c in df.columns if c.startswith("target_")]
if not all_target_cols:
    raise RuntimeError("No columns starting with 'target_' found in the CSVs.")

has_page_id = "page_id" in df.columns


# -------------------------------------------------------------------
# Feature matrix X: numeric/bool only + cleaning
# -------------------------------------------------------------------

candidate_features = df.drop(columns=all_target_cols)
X_all = candidate_features.select_dtypes(include=["number", "bool"]).copy()
X_all = X_all.dropna(axis=1, how="all")

nunique = X_all.nunique(dropna=True)
zero_var_cols = nunique[nunique <= 1].index.tolist()
if zero_var_cols:
    print("\nDropping zero-variance feature columns:")
    for c in zero_var_cols:
        print("  -", c)
    X_all = X_all.drop(columns=zero_var_cols)

print(f"\nTotal numeric/bool features after cleaning: {X_all.shape[1]}")


# -------------------------------------------------------------------
# Feature group selection (EDIT THESE HEURISTICS TO MATCH YOUR COLUMNS)
# -------------------------------------------------------------------

CONF_REGEXES = [
    r"conf", r"confidence",
    r"score",  # careful: may catch non-confidence scores; tweak if needed
    r"htr_line_score",  # keep your canonical feature inside confidence group
]

# Common ngram naming patterns (EDIT to your dataset)
NGRAM_REGEXES = [
    r"gram", r"chargram", r"cngram",
    r"^tri_", r"^bi_", r"^uni_",
    r"^ng_", r"_ng_", r"_gram",
]

def _matches_any(col: str, regexes: List[str]) -> bool:
    cl = col.lower()
    return any(re.search(rx, cl) is not None for rx in regexes)

def get_feature_set(name: str, X: pd.DataFrame) -> pd.DataFrame:
    cols = list(X.columns)

    if name == "full":
        return X

    if name == "single_htr_line_score_mean":
        if "htr_line_score_mean" not in cols:
            raise KeyError("Feature 'htr_line_score_mean' not found in X.")
        return X[["htr_line_score_mean"]].copy()

    if name == "confidence_only":
        sel = [c for c in cols if _matches_any(c, CONF_REGEXES)]
        if not sel:
            raise RuntimeError("confidence_only selected 0 features. Adjust CONF_REGEXES.")
        return X[sel].copy()

    if name == "ngram_only":
        sel = [c for c in cols if _matches_any(c, NGRAM_REGEXES)]
        if not sel:
            raise RuntimeError("ngram_only selected 0 features. Adjust NGRAM_REGEXES.")
        return X[sel].copy()

    raise ValueError(f"Unknown feature set: {name}")


FEATURE_SETS = [
    # baseline constant is handled separately
    "single_htr_line_score_mean",
    "confidence_only",
    "ngram_only",
    "full",  # your main model (with big HPO)
]


# -------------------------------------------------------------------
# Shared splitting logic (so every baseline uses the exact same val rows per target)
# -------------------------------------------------------------------

def get_train_val_indices_for_target(df: pd.DataFrame, X: pd.DataFrame, target_col: str) -> Tuple[List[int], List[int]]:
    y = df[target_col]
    mask = y.notna()
    idx = list(X.loc[mask].index)

    if use_external_eval and ("eval" in df.loc[mask, "split"].values):
        train_idx = [i for i in idx if df.at[i, "split"] == "train"]
        val_idx = [i for i in idx if df.at[i, "split"] == "eval"]
        if len(train_idx) < 10 or len(val_idx) < 1:
            train_idx, val_idx = train_test_split(idx, test_size=VAL_SIZE, random_state=42)
    else:
        train_idx, val_idx = train_test_split(idx, test_size=VAL_SIZE, random_state=42)

    return train_idx, val_idx


def write_predictions_csv(
    out_csv: str,
    df: pd.DataFrame,
    target_col: str,
    val_idx: List[int],
    y_val: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    feature_set: str,
    extra_cols: Optional[Dict[str, object]] = None,
) -> None:
    rows = []
    for idx_row, pred_val in zip(val_idx, y_pred):
        row = {
            "row_index": int(idx_row),
            "gt": float(y_val.loc[idx_row]),
            "pred": float(pred_val),
            "target": target_col,
            "model_name": model_name,
            "feature_set": feature_set,
            "split": df.loc[idx_row, "split"],
        }
        if has_page_id:
            row["page_id"] = df.loc[idx_row, "page_id"]
        if extra_cols:
            row.update(extra_cols)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    base_cols = ["row_index", "gt", "pred", "target", "model_name", "feature_set", "split"]
    cols = (["page_id"] + base_cols) if has_page_id else base_cols
    df_out = df_out[cols]
    df_out.to_csv(out_csv, index=False)
    print(f"Wrote predictions: {out_csv}")


# -------------------------------------------------------------------
# Training/eval loop for targets + baselines
# -------------------------------------------------------------------

models: Dict[str, Dict[str, XGBRegressor]] = {}  # target -> feature_set -> model

for target_col in BOW_TARGETS:
    print("\n" + "=" * 80)
    print(f"TARGET: {target_col}")
    print("=" * 80)

    y = df[target_col]
    mask = y.notna()
    if mask.sum() < 10:
        print(f"Not enough non-NaN samples for {target_col}. Skipping.")
        continue

    # shared split per target
    train_idx, val_idx = get_train_val_indices_for_target(df, X_all, target_col)

    y_train = y.loc[train_idx]
    y_val = y.loc[val_idx]

    w_train = make_sample_weights(y_train, target_col)
    w_val = make_sample_weights(y_val, target_col)

    print(f"Split sizes: train={len(train_idx)} val={len(val_idx)} (non-NaN total={int(mask.sum())})")
    if USE_SAMPLE_WEIGHTS and "bow" in target_col:
        print(
            f"[weights] train w: min={w_train.min():.2f}, mean={w_train.mean():.2f}, max={w_train.max():.2f} | "
            f"val w: min={w_val.min():.2f}, mean={w_val.mean():.2f}, max={w_val.max():.2f}"
        )

    # -------------------------
    # Baseline A: constant mean
    # -------------------------
    # "average bowf1 for the training+eval"
    # - If you set up EVAL_CSV_PATHS later, df will include both and this becomes train+eval.
    # - Right now you concatenated everything into TRAIN_CSV_PATHS, so "all available rows" = train+eval anyway.
    global_mean = float(y.loc[mask].mean())
    y_pred_const = np.full(shape=len(val_idx), fill_value=global_mean, dtype=float)

    mae_const = float(mean_absolute_error(y_val, y_pred_const))
    wmae_const = float(weighted_mae(y_val.values, y_pred_const, w_val)) if (REPORT_WEIGHTED_MAE and USE_SAMPLE_WEIGHTS) else np.nan

    append_summary_row(SUMMARY_METRICS_CSV, {
        "target": target_col,
        "model_name": "baseline_constant_mean",
        "feature_set": "none",
        "n_features": 0,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "val_mae": mae_const,
        "val_wmae": wmae_const if (REPORT_WEIGHTED_MAE and USE_SAMPLE_WEIGHTS) else "",
        "best_iteration": "",
        "params_json": json.dumps({"mean": global_mean}),
        "notes": "Predicts global mean over all non-NaN rows in concatenated dataset.",
    })

    out_pred_csv = os.path.join(LOG_DIR, f"val_predictions_baseline_constant_mean_{target_col}.csv")
    write_predictions_csv(
        out_csv=out_pred_csv,
        df=df,
        target_col=target_col,
        val_idx=val_idx,
        y_val=y_val,
        y_pred=y_pred_const,
        model_name="baseline_constant_mean",
        feature_set="none",
        extra_cols={"mean_used": global_mean},
    )

    # -------------------------
    # Feature-set models (XGB)
    # -------------------------
    models[target_col] = {}

    for feat_set in FEATURE_SETS:
        X_feat = get_feature_set(feat_set, X_all)

        X_train = X_feat.loc[train_idx]
        X_val = X_feat.loc[val_idx]

        if X_train.shape[1] == 0:
            print(f"[{feat_set}] 0 features -> skipping")
            continue

        # Decide search budget (full gets your full HPO; others smaller)
        if feat_set == "full":
            n_trials = N_TRIALS_HPO
            metrics_log = METRICS_LOG_CSV  # keep per-trial logs for main model
            model_name = "xgb_hpo_full"
        else:
            n_trials = N_TRIALS_BASELINES
            metrics_log = None  # don't spam per-trial log for baselines
            model_name = f"xgb_hpo_{feat_set}"

        print(f"\n[{target_col}] Training {model_name} (features={X_train.shape[1]})")

        best_model, best_params, best_mae, best_wmae = random_search_xgb(
            X_train=X_train,
            y_train=y_train,
            w_train=w_train,
            X_val=X_val,
            y_val=y_val,
            w_val=w_val,
            base_params=BASE_XGB_PARAMS,
            n_trials=n_trials,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            target_name=f"{target_col}__{feat_set}",
            metrics_log_csv=metrics_log,
            select_by="mae",  # you can switch to "wmae" if you want tail-optimized selection
        )

        y_pred = best_model.predict(X_val)
        mae = float(mean_absolute_error(y_val, y_pred))
        wmae = float(weighted_mae(y_val.values, y_pred, w_val)) if (REPORT_WEIGHTED_MAE and USE_SAMPLE_WEIGHTS) else np.nan

        best_iter = getattr(best_model, "best_iteration", None)
        if best_iter is None:
            best_iter = best_params.get("n_estimators", "")

        # Save model
        out_model_path = os.path.join(MODEL_DIR, f"{model_name}_{target_col}.joblib")
        joblib.dump(best_model, out_model_path)

        # Save predictions
        out_pred_csv = os.path.join(LOG_DIR, f"val_predictions_{model_name}_{target_col}.csv")
        write_predictions_csv(
            out_csv=out_pred_csv,
            df=df,
            target_col=target_col,
            val_idx=val_idx,
            y_val=y_val,
            y_pred=y_pred,
            model_name=model_name,
            feature_set=feat_set,
        )

        # Summary row
        append_summary_row(SUMMARY_METRICS_CSV, {
            "target": target_col,
            "model_name": model_name,
            "feature_set": feat_set,
            "n_features": int(X_train.shape[1]),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "val_mae": mae,
            "val_wmae": wmae if (REPORT_WEIGHTED_MAE and USE_SAMPLE_WEIGHTS) else "",
            "best_iteration": int(best_iter) if str(best_iter).isdigit() else best_iter,
            "params_json": json.dumps(best_params),
            "notes": "",
        })

        models[target_col][feat_set] = best_model

    print(f"\nDone target {target_col}. Summary metrics appended to: {SUMMARY_METRICS_CSV}")

print("\nALL DONE.")
print(f"- Summary metrics: {SUMMARY_METRICS_CSV}")
print(f"- Predictions CSVs: {LOG_DIR}")
print(f"- Models saved to: {MODEL_DIR}")
