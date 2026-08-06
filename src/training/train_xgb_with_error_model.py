from __future__ import annotations

import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor, DMatrix
import joblib


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

TRAIN_CSV_PATHS: List[str] = [
    "/home/coder/QualityPrediction/data/eval_from_training/xgboost/training_set/page_features_with_cer_with_bow.csv",
    "/home/coder/QualityPrediction/data/testsuite/xgboost/training_set/xgboost_testset.csv"
]

EVAL_CSV_PATHS: List[str] = [
]

MODEL_DIR = "/home/coder/QualityPrediction/models/xfold"
LOG_DIR = "/home/coder/QualityPrediction/data/eval_from_training/predictions/error_model_xfold_v2"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TARGETS = ["target_bow_f1"]

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

ERROR_XGB_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=10,
    reg_lambda=0.5,
    reg_alpha=0.0,
    random_state=42,
    eval_metric="mae",
)

EARLY_STOPPING_ROUNDS = 40
K_FOLDS = 5
VAL_SIZE = 0.1

BAD_THRESHOLD = 0.8
ERROR_SCALE = 100.0


# -------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------

def load_concat_csv(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing CSV: {p}")
        print(f"Loading: {p}")
        dfs.append(pd.read_csv(p))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# -------------------------------------------------------------------
# Uncertainty / meta-feature helpers
# -------------------------------------------------------------------

def safe_entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def leaf_features_for_model(model: XGBRegressor, X_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    booster = model.get_booster()
    leaf = booster.predict(DMatrix(X_df), pred_leaf=True)  # (n, n_trees)
    n, t = leaf.shape
    unique_ratio = np.zeros(n, dtype=float)
    ent = np.zeros(n, dtype=float)
    for i in range(n):
        _, cnts = np.unique(leaf[i], return_counts=True)
        unique_ratio[i] = len(cnts) / max(1, t)
        ent[i] = safe_entropy_from_counts(cnts)
    ent = ent / np.log(max(2, t))
    return unique_ratio, ent


def contrib_features_for_model(model: XGBRegressor, X_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    booster = model.get_booster()
    contrib = booster.predict(DMatrix(X_df), pred_contribs=True)  # (n, f+1)
    contrib_feat = contrib[:, :-1]
    abs_c = np.abs(contrib_feat)

    sum_abs = abs_c.sum(axis=1)
    max_abs = abs_c.max(axis=1)

    eps = 1e-12
    p = abs_c / (sum_abs[:, None] + eps)
    ent = -(p * np.log(p + eps)).sum(axis=1)
    ent = ent / np.log(max(2, contrib_feat.shape[1]))
    return sum_abs, max_abs, ent


def compute_ensemble_stats(fold_models: List[XGBRegressor], X_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    preds = np.column_stack([m.predict(X_df) for m in fold_models])  # (n, K)
    return {
        "ens_mean": preds.mean(axis=1),
        "ens_std": preds.std(axis=1),
        "ens_min": preds.min(axis=1),
        "ens_max": preds.max(axis=1),
        "ens_range": preds.max(axis=1) - preds.min(axis=1),
    }


def build_meta_features(
    X_df: pd.DataFrame,
    final_main_model: XGBRegressor,
    fold_models: List[XGBRegressor],
    bad_threshold: float,
) -> pd.DataFrame:
    """
    Meta-features to feed the error model:
      (1) ensemble stats from fold models
      (2) leaf entropy/unique ratio from final main model
      (3) margin to BAD_THRESHOLD from final main prediction
      (4) contrib summary stats from final main model
    Returns only meta-features (recommended).
    """
    ens = compute_ensemble_stats(fold_models, X_df)
    pred_main = final_main_model.predict(X_df)

    margin = np.abs(pred_main - bad_threshold)
    signed_margin = pred_main - bad_threshold

    leaf_unique_ratio, leaf_entropy = leaf_features_for_model(final_main_model, X_df)
    contrib_sum_abs, contrib_max_abs, contrib_entropy = contrib_features_for_model(final_main_model, X_df)

    meta = pd.DataFrame(
        {
            "pred_main": pred_main,
            "ens_mean": ens["ens_mean"],
            "ens_std": ens["ens_std"],
            "ens_min": ens["ens_min"],
            "ens_max": ens["ens_max"],
            "ens_range": ens["ens_range"],
            "margin_to_threshold": margin,
            "signed_margin_to_threshold": signed_margin,
            "leaf_unique_ratio": leaf_unique_ratio,
            "leaf_entropy": leaf_entropy,
            "contrib_sum_abs": contrib_sum_abs,
            "contrib_max_abs": contrib_max_abs,
            "contrib_entropy": contrib_entropy,
        },
        index=X_df.index,
    )
    return meta


# -------------------------------------------------------------------
# Main training logic
# -------------------------------------------------------------------

df_train = load_concat_csv(TRAIN_CSV_PATHS)
if df_train.empty:
    raise RuntimeError("No training data loaded.")
df_train["split"] = "train"

df_eval = load_concat_csv(EVAL_CSV_PATHS) if EVAL_CSV_PATHS else pd.DataFrame()
use_external_eval = not df_eval.empty
if use_external_eval:
    df_eval["split"] = "eval"
    df_all = pd.concat([df_train, df_eval], ignore_index=True)
else:
    df_all = df_train.copy()

print(f"Rows: total={len(df_all)}, train={len(df_train)}, eval={len(df_eval) if use_external_eval else 0}")

# Targets
all_target_cols = [c for c in df_all.columns if c.startswith("target_")]
if not all_target_cols:
    raise RuntimeError("No target_ columns found.")

# Features: numeric/bool only
candidate_features = df_all.drop(columns=all_target_cols)
X_all = candidate_features.select_dtypes(include=["number", "bool"]).copy()
X_all = X_all.dropna(axis=1, how="all")

# Drop zero variance columns based on TRAIN only (more correct)
X_train_tmp = X_all.loc[df_all["split"] == "train"]
nunique = X_train_tmp.nunique(dropna=True)
zero_var_cols = nunique[nunique <= 1].index.tolist()
if zero_var_cols:
    print("Dropping zero-variance features (train-based):")
    for c in zero_var_cols:
        print("  -", c)
    X_all = X_all.drop(columns=zero_var_cols)

has_page_id = "page_id" in df_all.columns

# Indices for splits
train_idx_all = df_all.index[df_all["split"] == "train"]
eval_idx_all = df_all.index[df_all["split"] == "eval"] if use_external_eval else pd.Index([])

# -------------------------------------------------------------------
# Per target pipeline
# -------------------------------------------------------------------

for target_col in TARGETS:
    print("\n==============================")
    print(f"Target: {target_col}")
    print("==============================")

    if target_col not in df_all.columns:
        print(f"Missing {target_col} in dataframe. Skipping.")
        continue

    # ---- TRAIN rows where target exists ----
    y_train_full = df_all.loc[train_idx_all, target_col]
    train_mask = y_train_full.notna()
    train_idx = train_idx_all[train_mask.values]

    if len(train_idx) < K_FOLDS + 5:
        print(f"Not enough train samples for {target_col}: {len(train_idx)}")
        continue

    X_train = X_all.loc[train_idx]
    y_train = df_all.loc[train_idx, target_col].astype(float)

    # ---- EVAL rows where target exists (optional for computing true_error) ----
    if use_external_eval:
        y_eval_full = df_all.loc[eval_idx_all, target_col]
        eval_has_gt = y_eval_full.notna().all()
        eval_idx = eval_idx_all  # keep all eval rows for prediction
        X_eval = X_all.loc[eval_idx]
        y_eval = df_all.loc[eval_idx, target_col].astype(float)  # may contain NaNs
    else:
        eval_idx = pd.Index([])
        X_eval = pd.DataFrame()

    # -------------------------------------------------------------------
    # 1) K-fold training on TRAIN -> OOF preds + store fold models
    # -------------------------------------------------------------------
    print(f"Training {K_FOLDS} fold main models for uncertainty + OOF residuals...")

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X_train), dtype=float)
    fold_models: List[XGBRegressor] = []

    X_train_reset = X_train.reset_index(drop=True)
    y_train_reset = y_train.reset_index(drop=True)

    for fold_i, (tr, va) in enumerate(kf.split(X_train_reset), start=1):
        X_tr, y_tr = X_train_reset.iloc[tr], y_train_reset.iloc[tr]
        X_va, y_va = X_train_reset.iloc[va], y_train_reset.iloc[va]

        m = XGBRegressor(**BASE_XGB_PARAMS)
        m.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)

        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred_va = m.predict(X_va)
        oof_pred[va] = pred_va

        fold_mae = mean_absolute_error(y_va, pred_va)
        print(f"  Fold {fold_i}: MAE={fold_mae:.4f}")

        fold_models.append(m)

    oof_mae = mean_absolute_error(y_train_reset, oof_pred)
    print(f"OOF MAE (train): {oof_mae:.4f}")

    # -------------------------------------------------------------------
    # 2) Train FINAL main model on ALL TRAIN
    # -------------------------------------------------------------------
    print("Training final main model on all TRAIN...")
    # internal holdout for early stopping
    X_tr_main, X_va_main, y_tr_main, y_va_main = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=42
    )

    main_model = XGBRegressor(**BASE_XGB_PARAMS)
    main_model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    main_model.fit(X_tr_main, y_tr_main, eval_set=[(X_va_main, y_va_main)], verbose=False)

    # sanity eval on the internal holdout
    pred_va_main = main_model.predict(X_va_main)
    mae_va_main = mean_absolute_error(y_va_main, pred_va_main)
    print(f"Final main model internal val MAE: {mae_va_main:.4f}")

    # main predictions on external testset (if present)
    if use_external_eval:
        pred_main_test = main_model.predict(X_eval)
        if y_eval.notna().all():
            mae_test = mean_absolute_error(y_eval.values, pred_main_test)
            print(f"Final main model TEST MAE: {mae_test:.4f}")
        else:
            print("Final main model: TEST GT has NaNs; skipping MAE.")

    # -------------------------------------------------------------------
    # 3) Build TRAIN meta-features + train ERROR model on OOF residuals
    # -------------------------------------------------------------------
    print("Building TRAIN meta-features for error model (using final main + fold ensemble)...")
    # IMPORTANT: meta-features for train must align to X_train_reset (same order as oof_pred)
    meta_train = build_meta_features(
        X_df=X_train_reset,
        final_main_model=main_model,
        fold_models=fold_models,
        bad_threshold=BAD_THRESHOLD,
    )

    true_abs_error = np.abs(oof_pred - y_train_reset.values)
    y_err_scaled = true_abs_error * ERROR_SCALE

    print("[DEBUG] true_abs_error stats (OOF):",
          f"min={true_abs_error.min():.6f}",
          f"max={true_abs_error.max():.6f}",
          f"mean={true_abs_error.mean():.6f}",
          f"std={true_abs_error.std():.6f}")

    X_err_tr, X_err_va, y_err_tr, y_err_va = train_test_split(
        meta_train, y_err_scaled, test_size=VAL_SIZE, random_state=42
    )

    error_model = XGBRegressor(**ERROR_XGB_PARAMS)
    error_model.fit(X_err_tr, y_err_tr, verbose=False)

    pred_err_va = error_model.predict(X_err_va) / ERROR_SCALE
    mae_err_va = mean_absolute_error(y_err_va / ERROR_SCALE, pred_err_va)
    print(f"Error model internal val MAE: {mae_err_va:.4f}")

    # -------------------------------------------------------------------
    # 4) Predict ERROR on external testset (and evaluate if GT exists)
    # -------------------------------------------------------------------
    if use_external_eval:
        print("Building TEST meta-features and predicting error on external testset...")
        meta_test = build_meta_features(
            X_df=X_eval,
            final_main_model=main_model,
            fold_models=fold_models,
            bad_threshold=BAD_THRESHOLD,
        )

        pred_err_test = error_model.predict(meta_test) / ERROR_SCALE

        # Evaluate error prediction on testset if GT exists
        if y_eval.notna().all():
            true_err_test = np.abs(pred_main_test - y_eval.values)
            mae_err_test = mean_absolute_error(true_err_test, pred_err_test)
            print(f"Error model TEST MAE (predicting |pred_main-gt|): {mae_err_test:.4f}")
        else:
            print("Error model: TEST GT has NaNs; skipping error MAE.")

    # -------------------------------------------------------------------
    # 5) Save models
    # -------------------------------------------------------------------
    main_path = os.path.join(MODEL_DIR, f"xgb_main_{target_col}.joblib")
    err_path = os.path.join(MODEL_DIR, f"xgb_error_{target_col}.joblib")
    joblib.dump(main_model, main_path)
    joblib.dump(error_model, err_path)
    print(f"Saved main model:  {main_path}")
    print(f"Saved error model: {err_path}")

    # -------------------------------------------------------------------
    # 6) Write outputs: TRAIN (OOF) and TEST (external eval)
    # -------------------------------------------------------------------
    # TRAIN output (OOF)
    train_out = pd.DataFrame(
        {
            "row_index": train_idx.values,
            "gt": y_train.values,
            "pred_oof": oof_pred,
            "true_error_oof": true_abs_error,
        }
    )
    train_meta = meta_train.copy()
    train_meta = train_meta.reset_index(drop=True)
    train_out = pd.concat([train_out.reset_index(drop=True), train_meta], axis=1)

    if has_page_id:
        train_out.insert(0, "page_id", df_all.loc[train_idx, "page_id"].values)

    out_train_csv = os.path.join(LOG_DIR, f"train_oof_with_uncertainty_{target_col}.csv")
    train_out.to_csv(out_train_csv, index=False)
    print(f"Wrote TRAIN OOF+uncertainty CSV: {out_train_csv}")

    # TEST / external eval output
    if use_external_eval:
        test_out = pd.DataFrame(
            {
                "row_index": eval_idx.values,
                "pred_main": pred_main_test,
                "pred_error": pred_err_test,
            }
        )

        if y_eval.notna().all():
            test_out["gt"] = y_eval.values
            test_out["true_error"] = np.abs(pred_main_test - y_eval.values)

        test_meta = meta_test.copy()
        test_out = pd.concat([test_out.reset_index(drop=True), test_meta.reset_index(drop=True)], axis=1)

        if has_page_id:
            test_out.insert(0, "page_id", df_all.loc[eval_idx, "page_id"].values)

        out_test_csv = os.path.join(LOG_DIR, f"test_with_uncertainty_and_error_{target_col}.csv")
        test_out.to_csv(out_test_csv, index=False)
        print(f"Wrote TEST predictions+uncertainty+error CSV: {out_test_csv}")

print("\nAll done.")
