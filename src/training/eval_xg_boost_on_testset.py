from __future__ import annotations

import os
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import joblib
from xgboost import XGBRegressor

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

# --- PATHS: adjust these three as needed ---
TRAIN_CSV_PATH = (
    "/home/coder/QualityPrediction/data/eval_from_training/xgboost/training_set/page_features_with_cer_with_bow.csv"
)

TEST_CSV_PATH = (
    "/home/coder/QualityPrediction/data/testsuite/xgboost/training_set/xgboost_testset.csv"
)

MODEL_DIR = "/home/coder/QualityPrediction/models"

# Name of the target you want to evaluate.
# This should correspond to a model file named xgb_hpo_<TARGET_COL>.joblib
TARGET_COL = "target_bow_f1"

# Where to store test predictions
LOG_DIR = "/home/coder/QualityPrediction/data/eval_from_training/predictions"
os.makedirs(LOG_DIR, exist_ok=True)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def build_feature_matrix_from_train(
    df_train: pd.DataFrame,
    all_target_cols: List[str],
) -> pd.DataFrame:
    """
    Replicate the feature-building logic from the training script,
    but only on the TRAIN set, so we get the exact feature column list
    that the model was trained on.
    """
    candidate_features = df_train.drop(columns=all_target_cols)

    # Keep only numeric and bool dtypes for XGBoost
    X_train = candidate_features.select_dtypes(include=["number", "bool"]).copy()

    # Drop all-NaN columns
    X_train = X_train.dropna(axis=1, how="all")

    # Drop zero-variance columns
    nunique = X_train.nunique(dropna=True)
    zero_var_cols = nunique[nunique <= 1].index.tolist()
    if zero_var_cols:
        print("\n[Train] Dropping zero-variance feature columns:")
        for c in zero_var_cols:
            print("  -", c)
        X_train = X_train.drop(columns=zero_var_cols)

    print("\n[Train] Final feature columns used for model training:")
    for c in X_train.columns:
        print("  -", c)

    return X_train


def build_feature_matrix_for_test(
    df_test: pd.DataFrame,
    feature_cols: List[str],
    all_target_cols: List[str],
) -> pd.DataFrame:
    """
    Build X_test with EXACTLY the same columns (and order) as in training.
    Any missing columns in the test set are filled with 0.
    Extra columns in the test set are ignored.
    """
    candidate_features_test = df_test.drop(columns=all_target_cols, errors="ignore")

    X_test = candidate_features_test.select_dtypes(include=["number", "bool"]).copy()

    # Ensure all train feature columns exist in test set
    for col in feature_cols:
        if col not in X_test.columns:
            print(f"[Test] Column '{col}' missing in test data. Filling with 0.")
            X_test[col] = 0.0

    # Restrict and reorder to match training feature columns
    X_test = X_test[feature_cols]

    return X_test


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    # ---------------------------
    # Load train + test CSVs
    # ---------------------------
    if not os.path.exists(TRAIN_CSV_PATH):
        raise FileNotFoundError(f"Could not find TRAIN CSV at {TRAIN_CSV_PATH!r}")
    if not os.path.exists(TEST_CSV_PATH):
        raise FileNotFoundError(f"Could not find TEST CSV at {TEST_CSV_PATH!r}")

    df_train = pd.read_csv(TRAIN_CSV_PATH)
    df_test = pd.read_csv(TEST_CSV_PATH)

    # All columns starting with "target_" are treated as possible targets
    all_target_cols: List[str] = [c for c in df_train.columns if c.startswith("target_")]
    if not all_target_cols:
        raise RuntimeError(
            "No columns starting with 'target_' found in the TRAIN CSV."
        )

    print("Found target columns in TRAIN CSV:")
    for c in all_target_cols:
        print("  -", c)

    if TARGET_COL not in all_target_cols:
        print(
            f"\nWARNING: TARGET_COL '{TARGET_COL}' not found among train target columns.\n"
            "Make sure TARGET_COL matches how you trained the model."
        )

    has_page_id = "page_id" in df_test.columns

    # ---------------------------
    # Build feature matrix from TRAIN
    # ---------------------------
    X_train = build_feature_matrix_from_train(df_train, all_target_cols)
    feature_cols = list(X_train.columns)

    # ---------------------------
    # Build feature matrix for TEST with the same columns
    # ---------------------------
    X_test = build_feature_matrix_for_test(df_test, feature_cols, all_target_cols)

    # ---------------------------
    # Load model for TARGET_COL
    # ---------------------------
    model_path = os.path.join(MODEL_DIR, f"xgb_hpo_{TARGET_COL}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path!r}\n"
            f"Expected a model named 'xgb_hpo_{TARGET_COL}.joblib' in MODEL_DIR."
        )

    print(f"\nLoading model from: {model_path}")
    model: XGBRegressor = joblib.load(model_path)

    # ---------------------------
    # Predict on TEST set
    # ---------------------------
    print(f"\nPredicting on test set for target: {TARGET_COL}")
    y_pred = model.predict(X_test)

    # ---------------------------
    # Compute MAE if GT is available
    # ---------------------------
    if TARGET_COL in df_test.columns:
        y_test = df_test[TARGET_COL]
        # Only evaluate on non-NaN entries of the target
        mask = y_test.notna()
        if mask.sum() > 0:
            mae = mean_absolute_error(y_test[mask], np.array(y_pred)[mask])
            print(f"\nTest MAE for {TARGET_COL}: {mae:.6f}")
        else:
            print(
                f"\nTarget column '{TARGET_COL}' exists in TEST CSV but all values are NaN. "
                "Skipping MAE computation."
            )
    else:
        print(
            f"\nTarget column '{TARGET_COL}' not found in TEST CSV. "
            "Skipping MAE computation (predictions only)."
        )

    # ---------------------------
    # Write test predictions to CSV
    # ---------------------------
    out_csv_name = os.path.join(LOG_DIR, f"test_predictions_{TARGET_COL}.csv")

    rows: List[Dict[str, object]] = []
    for idx_row, pred_val in zip(df_test.index, y_pred):
        row: Dict[str, object] = {
            "row_index": int(idx_row),
            "pred": float(pred_val),
        }

        if TARGET_COL in df_test.columns:
            gt_val = df_test.loc[idx_row, TARGET_COL]
            if pd.isna(gt_val):
                row["gt"] = None
            else:
                row["gt"] = float(gt_val)

        if has_page_id:
            row["page_id"] = df_test.loc[idx_row, "page_id"]

        rows.append(row)

    df_out = pd.DataFrame(rows)

    # Order columns nicely
    col_order = []
    if has_page_id:
        col_order.append("page_id")
    col_order.extend(["row_index"])
    if TARGET_COL in df_test.columns:
        col_order.append("gt")
    col_order.append("pred")

    df_out = df_out[col_order]

    df_out.to_csv(out_csv_name, index=False)
    print(f"\nWrote test predictions to {out_csv_name}")


if __name__ == "__main__":
    main()
