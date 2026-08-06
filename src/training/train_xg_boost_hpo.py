from __future__ import annotations

import os
import csv
import random
from typing import Dict, List, Tuple

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
    "/home/coder/QualityPrediction/data/eval_from_training/xgboost/training_set/xgboost_eval_from_training_dyn_bins.csv"
]

EVAL_CSV_PATHS: List[str] = [
]

MODEL_DIR = "/home/coder/QualityPrediction/models"
LOG_DIR = "/home/coder/QualityPrediction/data/eval_from_training/predictions/dynamic_bins"
FEAT_ANALYSIS_DIR = "/home/coder/QualityPrediction/data/eval_from_training/dynamic_bins"
METRICS_LOG_CSV = os.path.join(LOG_DIR, "xgb_hpo_metrics_weighing_dyn_bins.csv")

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

BOW_TARGETS = [
    "target_bow_f1",
    "target_bow_recall",
    "target_bow_precision"
]

# -------------------------------------------------------------------
# Sample reweighting (NEW)
# -------------------------------------------------------------------
USE_SAMPLE_WEIGHTS = True

# This emphasizes low target values (good for target_bow_f1 style targets).
# weight = 1 + alpha * (1 - y)^p
WEIGHT_ALPHA = 5.0   # strength (try 5..30)
WEIGHT_P = 2.0        # curvature (try 1..4)
WEIGHT_CLIP_MIN = 1.0
WEIGHT_CLIP_MAX = 50.0

# If True, also compute "weighted MAE" on val (diagnostic)
REPORT_WEIGHTED_MAE = True


def make_sample_weights(y: pd.Series, target_name: str) -> np.ndarray:
    """
    Build sample weights. Default behavior:
      - For BoW F1/precision/recall in [0,1], emphasize low values:
            w = 1 + alpha * (1 - y)^p
    For non-[0,1] targets, we fall back to uniform weights.
    """
    yv = y.astype(float).values

    if not USE_SAMPLE_WEIGHTS:
        return np.ones_like(yv, dtype=float)

    # Heuristic: apply this weighting only to "bow" targets by name
    if "bow" not in target_name:
        return np.ones_like(yv, dtype=float)

    # Guard: if values are not roughly [0,1], don't apply this scheme
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
# Helpers for logging & feature analysis
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


def remap_xgb_feature_names(score_dict: Dict[str, float],
                            feature_names: List[str]) -> Dict[str, float]:
    mapped = {}
    for k, v in score_dict.items():
        if k in feature_names:
            mapped[k] = v
        elif k.startswith("f") and k[1:].isdigit():
            idx = int(k[1:])
            if 0 <= idx < len(feature_names):
                mapped[feature_names[idx]] = v
            else:
                mapped[k] = v
        else:
            mapped[k] = v
    return mapped


def compute_feature_importance_for_target(
    target_col: str,
    model: XGBRegressor,
    X: pd.DataFrame,
    df: pd.DataFrame,
    out_dir: str,
    suffix: str = "",
) -> pd.DataFrame:
    booster = model.get_booster()
    feature_names = list(X.columns)

    raw_gain = booster.get_score(importance_type="gain")
    raw_weight = booster.get_score(importance_type="weight")
    raw_cover = booster.get_score(importance_type="cover")

    gain = remap_xgb_feature_names(raw_gain, feature_names)
    weight = remap_xgb_feature_names(raw_weight, feature_names)
    cover = remap_xgb_feature_names(raw_cover, feature_names)

    def normalize(d):
        total = sum(d.values()) if d else 0.0
        if total <= 0:
            return {k: 0.0 for k in d}
        return {k: v / total for k, v in d.items()}

    gain_norm = normalize(gain)
    weight_norm = normalize(weight)
    cover_norm = normalize(cover)

    y = df[target_col]
    corrs = {}
    for feat in feature_names:
        if df[feat].dtype.kind in "bifc":
            corrs[feat] = df[feat].corr(y)
        else:
            corrs[feat] = np.nan

    rows = []
    for feat in feature_names:
        rows.append(
            dict(
                feature=feat,
                gain=gain.get(feat, 0.0),
                gain_norm=gain_norm.get(feat, 0.0),
                weight=weight.get(feat, 0.0),
                weight_norm=weight_norm.get(feat, 0.0),
                cover=cover.get(feat, 0.0),
                cover_norm=cover_norm.get(feat, 0.0),
                corr_with_target=corrs.get(feat, np.nan),
            )
        )

    df_imp = pd.DataFrame(rows).sort_values("gain_norm", ascending=False)

    suffix_part = f"_{suffix}" if suffix else ""
    out_csv = os.path.join(out_dir, f"feature_importance_{target_col}{suffix_part}.csv")
    df_imp.to_csv(out_csv, index=False)
    print(f"[Feature analysis] Saved importance for {target_col}{suffix_part} to {out_csv}")
    print(df_imp.head(10).to_string(index=False))
    return df_imp


# -------------------------------------------------------------------
# Hyperparameter search (UPDATED: uses sample weights)
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
) -> Tuple[XGBRegressor, Dict[str, float], float]:
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

    best_mae = float("inf")
    best_params = base_params.copy()
    best_model: XGBRegressor | None = None
    best_trial_idx = -1

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
        append_metrics_row(METRICS_LOG_CSV, row)

        msg = (
            f"[{target_name}] trial {trial:02d}/{n_trials}: "
            f"MAE={mae:.4f}"
        )
        if REPORT_WEIGHTED_MAE and USE_SAMPLE_WEIGHTS:
            msg += f", wMAE={wmae:.4f}"
        msg += (
            f", best_iter={best_iter}, "
            f"depth={params['max_depth']}, lr={params['learning_rate']}, "
            f"subs={params['subsample']}, colsub={params['colsample_bytree']}"
        )
        print(msg)

        # You can choose to select by wMAE instead for tail-focus:
        # score = wmae if (REPORT_WEIGHTED_MAE and USE_SAMPLE_WEIGHTS) else mae
        score = mae

        if score < best_mae:
            best_mae = score
            best_params = params
            best_model = model
            best_trial_idx = trial

    if best_model is not None:
        best_iter = getattr(best_model, "best_iteration", None)
        if best_iter is None:
            best_iter = best_params["n_estimators"]

        best_row = {
            "target": target_name,
            "trial": best_trial_idx,
            "is_best_for_target": 1,
            "val_mae": float(best_mae),
            "val_wmae": "",
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
        append_metrics_row(METRICS_LOG_CSV, best_row)

        print(
            f"\n[{target_name}] Best MAE from random search: {best_mae:.4f} "
            f"(trial {best_trial_idx})"
        )
        print("Best params:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

    return best_model, best_params, best_mae


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

print("Found target columns:")
for c in all_target_cols:
    print("  -", c)


# -------------------------------------------------------------------
# Build feature matrix X: numeric/bool only + basic cleaning
# -------------------------------------------------------------------

candidate_features = df.drop(columns=all_target_cols)
X = candidate_features.select_dtypes(include=["number", "bool"]).copy()
X = X.dropna(axis=1, how="all")

nunique = X.nunique(dropna=True)
zero_var_cols = nunique[nunique <= 1].index.tolist()
if zero_var_cols:
    print("\nDropping zero-variance feature columns:")
    for c in zero_var_cols:
        print("  -", c)
    X = X.drop(columns=zero_var_cols)

dropped_cols = sorted(set(candidate_features.columns) - set(X.columns))

print("\nUsing the following feature columns (numeric/bool, cleaned):")
for c in X.columns:
    print("  -", c)

if dropped_cols:
    print("\nDropped non-numeric or useless columns from features:")
    for c in dropped_cols:
        print("  -", c)

has_page_id = "page_id" in df.columns


# -------------------------------------------------------------------
# Train one XGBRegressor per target with HPO
# -------------------------------------------------------------------

models: Dict[str, XGBRegressor] = {}
targets_to_train = BOW_TARGETS

for target_col in targets_to_train:
    print("\n==============================")
    print(f"Training model for target: {target_col}")
    print("==============================")

    y = df[target_col]
    mask = y.notna()
    X_t = X.loc[mask]
    y_t = y.loc[mask]
    splits_t = df.loc[mask, "split"]

    if len(X_t) < 10:
        print(f"Not enough non-NaN samples for {target_col} (only {len(X_t)}). Skipping.")
        continue

    indices = list(X_t.index)

    if use_external_eval and ("eval" in splits_t.values):
        train_idx = [i for i in indices if df.at[i, "split"] == "train"]
        val_idx = [i for i in indices if df.at[i, "split"] == "eval"]

        if len(train_idx) < 10 or len(val_idx) < 1:
            print(f"Not enough data in external train/eval for {target_col}. Falling back to internal split.")
            train_idx, val_idx = train_test_split(indices, test_size=VAL_SIZE, random_state=42)
    else:
        train_idx, val_idx = train_test_split(indices, test_size=VAL_SIZE, random_state=42)

    X_train = X_t.loc[train_idx]
    y_train = y_t.loc[train_idx]

    X_val = X_t.loc[val_idx]
    y_val = y_t.loc[val_idx]

    if len(X_train) < 10 or len(X_val) < 1:
        print(f"Insufficient samples after split for {target_col} (train={len(X_train)}, val={len(X_val)}). Skipping.")
        continue

    # NEW: compute weights for this target
    w_train = make_sample_weights(y_train, target_col)
    w_val = make_sample_weights(y_val, target_col)

    if USE_SAMPLE_WEIGHTS and "bow" in target_col:
        print(
            f"[weights] {target_col}: "
            f"train w: min={w_train.min():.2f}, mean={w_train.mean():.2f}, max={w_train.max():.2f} | "
            f"val w: min={w_val.min():.2f}, mean={w_val.mean():.2f}, max={w_val.max():.2f}"
        )

    best_model, best_params, best_mae = random_search_xgb(
        X_train=X_train,
        y_train=y_train,
        w_train=w_train,
        X_val=X_val,
        y_val=y_val,
        w_val=w_val,
        base_params=BASE_XGB_PARAMS,
        n_trials=N_TRIALS_HPO,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        target_name=target_col,
    )

    if best_model is None:
        print(f"No model obtained for {target_col} (HPO failed?). Skipping saving.")
        continue

    y_pred = best_model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    wmae = None
    if REPORT_WEIGHTED_MAE and USE_SAMPLE_WEIGHTS:
        wmae = weighted_mae(y_val.values, y_pred, w_val)

    best_iter = getattr(best_model, "best_iteration", None)
    if best_iter is None:
        best_iter = best_params["n_estimators"]

    print(f"{target_col}: validation MAE (best) = {mae:.4f}")
    if wmae is not None:
        print(f"{target_col}: validation weighted-MAE = {wmae:.4f}")
    print(f"  Best iteration: {best_iter} / {best_params['n_estimators']}")

    models[target_col] = best_model

    out_model_name = os.path.join(MODEL_DIR, f"xgb_hpo_{target_col}.joblib")
    joblib.dump(best_model, out_model_name)
    print(f"Saved best model for {target_col} to {out_model_name}")

    out_csv_name = os.path.join(LOG_DIR, f"val_predictions_hpo_{target_col}.csv")

    rows = []
    for idx_row, pred_val in zip(X_val.index, y_pred):
        gt_val = y.loc[idx_row]
        row = {
            "row_index": int(idx_row),
            "gt": float(gt_val),
            "pred": float(pred_val),
        }
        if has_page_id:
            row["page_id"] = df.loc[idx_row, "page_id"]
        row["split"] = df.loc[idx_row, "split"]
        rows.append(row)

    df_val_out = pd.DataFrame(rows)

    base_cols = ["row_index", "gt", "pred", "split"]
    cols = (["page_id"] + base_cols) if has_page_id else base_cols
    df_val_out = df_val_out[cols]
    df_val_out.to_csv(out_csv_name, index=False)
    print(f"Wrote validation predictions (best model) to {out_csv_name}")

    print(f"\n[Feature analysis] {target_col} (best HPO model)")
    _ = compute_feature_importance_for_target(
        target_col=target_col,
        model=best_model,
        X=X_t,
        df=df.loc[mask],
        out_dir=FEAT_ANALYSIS_DIR,
        suffix="hpo",
    )
