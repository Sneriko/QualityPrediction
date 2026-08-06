from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

CSV_PATH = "/home/coder/QualityPrediction/data/eval_from_training/xgboost/page_features_with_cer_with_bow.csv"

MODEL_DIR = "/home/coder/QualityPrediction/models"
LOG_DIR = "/home/coder/QualityPrediction/data/eval_from_training/predictions"
FEAT_ANALYSIS_DIR = "/home/coder/QualityPrediction/data/eval_from_training/feature_analysis"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FEAT_ANALYSIS_DIR, exist_ok=True)

# XGBoost parameters tuned a bit for ~1k samples
BASE_XGB_PARAMS = dict(
    n_estimators=800,             # or your previous value
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=8,
    reg_lambda=1.0,
    reg_alpha=0.1,
    random_state=42,
    eval_metric="mae",            # <- move eval_metric here
)

EARLY_STOPPING_ROUNDS = 40
VAL_SIZE = 0.1

# Optional: subsets of targets for HTR / segmentation quality
HTR_QUALITY_TARGETS = [
    "target_perm_cer_htr_only",
    "target_avg_line_cer",
    "target_perm_cer",
]

SEGMENTATION_QUALITY_TARGETS = [
    "target_seg_error",
    "target_ro_error",
    "target_pi_missing_ratio",
    "target_pi_halluc_ratio",
    "target_delta_cer",
]


# -------------------------------------------------------------------
# Helpers for feature analysis
# -------------------------------------------------------------------

def remap_xgb_feature_names(score_dict: Dict[str, float],
                            feature_names: List[str]) -> Dict[str, float]:
    """
    XGBoost may return feature keys as 'f0', 'f1', ... when using the DMatrix
    internal indices. This maps those back to human-readable pandas column names
    if possible.
    """
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
) -> pd.DataFrame:
    """
    For one target/model:
    - Extract XGBoost feature importance (gain/weight/cover)
    - Compute Pearson correlation between each feature and the target
    - Save per-target CSV with all info
    """
    booster = model.get_booster()
    feature_names = list(X.columns)

    # XGBoost importance
    raw_gain = booster.get_score(importance_type="gain")
    raw_weight = booster.get_score(importance_type="weight")
    raw_cover = booster.get_score(importance_type="cover")

    gain = remap_xgb_feature_names(raw_gain, feature_names)
    weight = remap_xgb_feature_names(raw_weight, feature_names)
    cover = remap_xgb_feature_names(raw_cover, feature_names)

    # Normalize by sum (optional but handy for comparability)
    def normalize(d):
        total = sum(d.values()) if d else 0.0
        if total <= 0:
            return {k: 0.0 for k in d}
        return {k: v / total for k, v in d.items()}

    gain_norm = normalize(gain)
    weight_norm = normalize(weight)
    cover_norm = normalize(cover)

    # Correlations (ignore NaNs)
    y = df[target_col]
    corrs = {}
    for feat in feature_names:
        if df[feat].dtype.kind in "bifc":  # numeric/bool/float/complex
            corrs[feat] = df[feat].corr(y)  # pandas handles NaNs by default
        else:
            corrs[feat] = np.nan

    # Build a unified DataFrame (one row per feature)
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

    df_imp = pd.DataFrame(rows)

    # Sort by normalized gain as primary criterion
    df_imp = df_imp.sort_values("gain_norm", ascending=False)

    out_csv = os.path.join(out_dir, f"feature_importance_{target_col}.csv")
    df_imp.to_csv(out_csv, index=False)
    print(f"[Feature analysis] Saved importance for {target_col} to {out_csv}")

    # Print top 10 as a quick summary
    print(df_imp.head(10).to_string(index=False))

    return df_imp


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Could not find CSV at {CSV_PATH!r}")

df = pd.read_csv(CSV_PATH)

# All columns starting with "target_" are considered possible targets
all_target_cols: List[str] = [c for c in df.columns if c.startswith("target_")]
if not all_target_cols:
    raise RuntimeError("No columns starting with 'target_' found in the CSV.")

print("Found target columns:")
for c in all_target_cols:
    print("  -", c)

# -------------------------------------------------------------------
# Build feature matrix X: numeric/bool only + basic cleaning
# -------------------------------------------------------------------

candidate_features = df.drop(columns=all_target_cols)

# Keep only numeric and bool dtypes for XGBoost
X = candidate_features.select_dtypes(include=["number", "bool"]).copy()
initial_cols = list(X.columns)

# Drop all-NaN columns
X = X.dropna(axis=1, how="all")

# Drop zero-variance columns
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
# Train one XGBRegressor per target
# and write validation predictions to text files
# -------------------------------------------------------------------

models: Dict[str, XGBRegressor] = {}

# Choose which targets to train on:
#   all_target_cols
#   HTR_QUALITY_TARGETS
#   SEGMENTATION_QUALITY_TARGETS
targets_to_train = all_target_cols

for target_col in targets_to_train:
    print("\n==============================")
    print(f"Training model for target: {target_col}")
    print("==============================")

    y = df[target_col]

    # Drop rows where this specific target is missing
    mask = y.notna()
    X_t = X.loc[mask]
    y_t = y.loc[mask]

    if len(X_t) < 10:
        print(f"Not enough non-NaN samples for {target_col} (only {len(X_t)}). Skipping.")
        continue

    indices = X_t.index.to_list()
    train_idx, val_idx = train_test_split(
        indices,
        test_size=VAL_SIZE,
        random_state=42,  # fixed for reproducibility
    )

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]

    X_val = X.loc[val_idx]
    y_val = y.loc[val_idx]

    model = XGBRegressor(**BASE_XGB_PARAMS)
    model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Use best iteration (XGBRegressor does this automatically after early stopping)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"{target_col}: validation MAE = {mae:.4f}")
    print(f"  Best iteration: {model.best_iteration} / {BASE_XGB_PARAMS['n_estimators']}")

    models[target_col] = model

    # Save model
    out_model_name = os.path.join(MODEL_DIR, f"xgb_{target_col}.joblib")
    joblib.dump(model, out_model_name)
    print(f"Saved model to {out_model_name}")

    # ---------------------------------------------------------------
    # Write validation predictions to a CSV file
    # ---------------------------------------------------------------
    out_csv_name = os.path.join(LOG_DIR, f"val_predictions_{target_col}.csv")

    rows = []
    for idx_row, pred_val in zip(X_val.index, y_pred):
        gt_val = y.loc[idx_row]

        row = {
            "row_index": idx_row,
            "gt": float(gt_val),
            "pred": float(pred_val),
        }

        if has_page_id:
            row["page_id"] = df.loc[idx_row, "page_id"]

        rows.append(row)

    df_val_out = pd.DataFrame(rows)

    # Order columns nicely
    cols = ["page_id", "row_index", "gt", "pred"] if has_page_id else ["row_index", "gt", "pred"]
    df_val_out = df_val_out[cols]

    df_val_out.to_csv(out_csv_name, index=False)
    print(f"Wrote validation predictions to {out_csv_name}")

    # ---------------------------------------------------------------
    # Feature importance / correlation analysis for this target
    # ---------------------------------------------------------------
    print(f"\n[Feature analysis] {target_col}")
    _ = compute_feature_importance_for_target(
        target_col=target_col,
        model=model,
        X=X,
        df=df,
        out_dir=FEAT_ANALYSIS_DIR,
    )

# -------------------------------------------------------------------
# Optional: quick example prediction on one row
# -------------------------------------------------------------------

if models:
    example_idx = 0
    x_example = X.iloc[[example_idx]]  # single row as DataFrame
    print("\n--- Example predictions for row index", example_idx, "---")
    for target_col, model in models.items():
        pred = model.predict(x_example)[0]
        true_val = df.loc[example_idx, target_col]
        print(f"{target_col}: predicted={pred:.4f}, true={true_val:.4f}")
