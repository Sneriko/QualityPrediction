from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .weights import weighted_mae


def _impute_for_corr(X: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation needs finite values. We impute each column with its median (or 0 if all NaN).
    This is used ONLY for correlation pruning, not for model training.
    """
    Xn = X.copy()
    for c in Xn.columns:
        s = Xn[c]
        if s.dtype.kind not in "bifc":
            Xn[c] = pd.to_numeric(s, errors="coerce")
        med = float(np.nanmedian(Xn[c].values)) if np.isfinite(np.nanmedian(Xn[c].values)) else 0.0
        Xn[c] = Xn[c].astype(float).fillna(med)
    return Xn


def prune_correlated_features(
    X_train: pd.DataFrame,
    threshold: float = 0.98,
) -> List[str]:
    """
    Returns a list of columns to KEEP.
    Deterministic greedy pruning: iterates columns in sorted order and drops later columns
    that are highly correlated with an earlier kept column.
    """
    if X_train.shape[1] <= 1:
        return list(X_train.columns)

    cols = sorted(list(X_train.columns))
    Xc = _impute_for_corr(X_train[cols])
    corr = Xc.corr(method="pearson").abs()

    keep: List[str] = []
    dropped = set()

    for i, ci in enumerate(cols):
        if ci in dropped:
            continue
        keep.append(ci)
        # Drop any cj after ci that correlates too highly with ci
        if i + 1 < len(cols):
            too_high = corr.loc[ci, cols[i + 1 :]] > threshold
            for cj, flag in too_high.items():
                if bool(flag):
                    dropped.add(cj)

    return keep


def _score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    w: Optional[np.ndarray],
    use_weighted: bool,
) -> float:
    if use_weighted and w is not None:
        return float(weighted_mae(y_true, y_pred, w))
    return float(mean_absolute_error(y_true, y_pred))


@dataclass
class PermutationResult:
    importance_mean: Dict[str, float]
    importance_std: Dict[str, float]
    selected_features: List[str]


def permutation_importance_topk(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    w_val: Optional[np.ndarray] = None,
    repeats: int = 5,
    top_k: int = 100,
    random_state: int = 42,
    use_weighted_metric: bool = False,
) -> PermutationResult:
    """
    Model-agnostic permutation importance:
      importance(feature) = mean( score(shuffled) - score(baseline) ) over repeats
    Larger = more important.
    """
    rng = np.random.default_rng(random_state)

    y_true = y_val.astype(float).values
    baseline_pred = model.predict(X_val)
    baseline_score = _score(y_true, baseline_pred, w_val, use_weighted_metric)

    cols = list(X_val.columns)
    imp_means: Dict[str, float] = {}
    imp_stds: Dict[str, float] = {}

    X_work = X_val.copy()

    for c in cols:
        deltas = []
        original = X_work[c].values.copy()

        for _ in range(max(1, int(repeats))):
            shuffled = original.copy()
            rng.shuffle(shuffled)
            X_work[c] = shuffled
            pred = model.predict(X_work)
            s = _score(y_true, pred, w_val, use_weighted_metric)
            deltas.append(s - baseline_score)

        # restore
        X_work[c] = original

        deltas_arr = np.asarray(deltas, dtype=float)
        imp_means[c] = float(np.mean(deltas_arr))
        imp_stds[c] = float(np.std(deltas_arr))

    # Select top-K by mean importance (descending)
    top_k = min(int(top_k), len(cols))
    ranked = sorted(cols, key=lambda k: imp_means.get(k, 0.0), reverse=True)
    selected = ranked[:top_k]

    return PermutationResult(
        importance_mean=imp_means,
        importance_std=imp_stds,
        selected_features=selected,
    )


def stability_selection_topk(
    model_factory,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    w_train: Optional[np.ndarray],
    w_val: Optional[np.ndarray],
    runs: int,
    repeats: int,
    top_k: int,
    min_freq: float,
    random_state: int = 42,
    use_weighted_metric: bool = False,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Bootstrap TRAIN `runs` times, fit model each time, compute permutation importance on VAL,
    take top-K, and keep features that appear frequently.
    Returns (selected_features, freq_dict).
    """
    rng = np.random.default_rng(random_state)
    cols = list(X_train.columns)
    if runs <= 0 or len(cols) == 0:
        return cols, {c: 1.0 for c in cols}

    counts = {c: 0 for c in cols}
    n = X_train.shape[0]

    y_train_arr = y_train.astype(float).values

    for r in range(runs):
        idx = rng.integers(low=0, high=n, size=n)  # bootstrap
        Xb = X_train.iloc[idx]
        yb = y_train_arr[idx]

        model = model_factory()
        fit_kwargs = dict(
            X=Xb,
            y=yb,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        if w_train is not None:
            fit_kwargs["sample_weight"] = w_train[idx]

        model.fit(**fit_kwargs)

        perm = permutation_importance_topk(
            model,
            X_val=X_val,
            y_val=y_val,
            w_val=w_val,
            repeats=repeats,
            top_k=top_k,
            random_state=random_state + 1000 + r,
            use_weighted_metric=use_weighted_metric,
        )

        for f in perm.selected_features:
            counts[f] += 1

    freqs = {c: counts[c] / float(runs) for c in cols}
    selected = [c for c in cols if freqs[c] >= float(min_freq)]
    # if too few survive, fall back to top-K by frequency
    if len(selected) == 0:
        selected = sorted(cols, key=lambda c: freqs[c], reverse=True)[: min(top_k, len(cols))]

    return selected, freqs
