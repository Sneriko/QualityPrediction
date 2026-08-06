from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

CSV_PATH = "/home/coder/QualityPrediction/data/eval_from_training/xgboost/page_features_with_cer_v4.csv"

BASE_XGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=8,
    random_state=42,
)

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
# Build feature matrix X: numeric/bool only
# -------------------------------------------------------------------

# Start from everything except targets
candidate_features = df.drop(columns=all_target_cols)

# Keep only numeric and bool dtypes for XGBoost
X = candidate_features.select_dtypes(include=["number", "bool"]).copy()
dropped_cols = sorted(set(candidate_features.columns) - set(X.columns))

print("\nUsing the following feature columns (numeric/bool only):")
for c in X.columns:
    print("  -", c)

if dropped_cols:
    print("\nDropped non-numeric columns from features (IDs, names, etc.):")
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

    # Use indices so we can map back to original rows
    indices = X_t.index.to_list()
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42
    )

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]

    X_val = X.loc[val_idx]
    y_val = y.loc[val_idx]

    model = XGBRegressor(**BASE_XGB_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"{target_col}: validation MAE = {mae:.4f}")

    models[target_col] = model

    # Save model
    out_model_name = f"/home/coder/QualityPrediction/models/xgb_{target_col}.joblib"
    joblib.dump(model, out_model_name)
    print(f"Saved model to {out_model_name}")

    # ---------------------------------------------------------------
    # Write validation predictions to a text file
    # ---------------------------------------------------------------
    out_txt_name = f"/home/coder/QualityPrediction/data/eval_from_training/log/val_predictions_{target_col}.txt"
    with open(out_txt_name, "w", encoding="utf-8") as f:
        # Header
        if has_page_id:
            f.write("page_id\trow_index\tgt\tpred\n")
        else:
            f.write("row_index\tgt\tpred\n")

        # Go through each validation sample
        for idx_row, pred_val in zip(X_val.index, y_pred):
            gt_val = y.loc[idx_row]

            if has_page_id:
                page_id = df.loc[idx_row, "page_id"]
                f.write(f"{page_id}\t{idx_row}\t{gt_val:.8f}\t{pred_val:.8f}\n")
            else:
                f.write(f"{idx_row}\t{gt_val:.8f}\t{pred_val:.8f}\n")

    print(f"Wrote validation predictions to {out_txt_name}")


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
