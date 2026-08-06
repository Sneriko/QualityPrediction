from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class LoadedData:
    df: pd.DataFrame
    X_all: pd.DataFrame
    target_cols: List[str]
    has_page_id: bool
    use_external_eval: bool


def load_concat_csv(paths: Sequence[Path]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Could not find CSV at {str(p)!r}")
        dfs.append(pd.read_csv(p))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def build_feature_matrix(df: pd.DataFrame, target_cols: Sequence[str]) -> pd.DataFrame:
    """
    Keeps only numeric/bool features, drops all-NaN and zero-variance columns.
    """
    candidate = df.drop(columns=list(target_cols), errors="ignore")
    X = candidate.select_dtypes(include=["number", "bool"]).copy()
    X = X.dropna(axis=1, how="all")

    nunique = X.nunique(dropna=True)
    zero_var_cols = nunique[nunique <= 1].index.tolist()
    if zero_var_cols:
        X = X.drop(columns=zero_var_cols)

    return X


def load_training_data(
    train_csvs: Sequence[Path],
    eval_csvs: Sequence[Path],
    page_id_col: str = "page_id",
) -> LoadedData:
    df_train = load_concat_csv(train_csvs)
    if df_train.empty:
        raise RuntimeError("No training data loaded. Check train_csvs.")
    df_train = df_train.copy()
    df_train["split"] = "train"

    if eval_csvs:
        df_eval = load_concat_csv(eval_csvs)
        if df_eval.empty:
            df = df_train
            use_external_eval = False
        else:
            df_eval = df_eval.copy()
            df_eval["split"] = "eval"
            df = pd.concat([df_train, df_eval], ignore_index=True)
            use_external_eval = True
    else:
        df = df_train
        use_external_eval = False

    target_cols = [c for c in df.columns if c.startswith("target_")]
    if not target_cols:
        raise RuntimeError("No columns starting with 'target_' found in the input CSV(s).")

    X_all = build_feature_matrix(df, target_cols)
    has_page_id = page_id_col in df.columns

    return LoadedData(
        df=df,
        X_all=X_all,
        target_cols=target_cols,
        has_page_id=has_page_id,
        use_external_eval=use_external_eval,
    )


def get_train_val_indices_for_target(
    df: pd.DataFrame,
    X: pd.DataFrame,
    target_col: str,
    use_external_eval: bool,
    val_size: float,
    random_state: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Ensures every model for the same target uses identical val rows.
    If external eval split exists, uses df['split'] to define val.
    """
    y = df[target_col]
    mask = y.notna()
    idx = list(X.loc[mask].index)

    if not idx:
        return [], []

    if use_external_eval and ("eval" in df.loc[mask, "split"].values):
        train_idx = [i for i in idx if df.at[i, "split"] == "train"]
        val_idx = [i for i in idx if df.at[i, "split"] == "eval"]
        if len(train_idx) < 10 or len(val_idx) < 1:
            train_idx, val_idx = train_test_split(idx, test_size=val_size, random_state=random_state)
    else:
        train_idx, val_idx = train_test_split(idx, test_size=val_size, random_state=random_state)

    return train_idx, val_idx
