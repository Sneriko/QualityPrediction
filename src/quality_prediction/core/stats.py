from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def safe_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "cv": float("nan")}
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "cv": float("nan")}
    mean = float(arr.mean())
    std = float(arr.std())
    vmin = float(arr.min())
    vmax = float(arr.max())
    cv = float(std / mean) if mean != 0 else float("nan")
    return {"mean": mean, "std": std, "min": vmin, "max": vmax, "cv": cv}


def histogram_features(
    values: List[float],
    bins: int,
    prefix: str,
    value_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    if not values:
        return {f"{prefix}_bin_{i}": 0.0 for i in range(bins)}
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{prefix}_bin_{i}": 0.0 for i in range(bins)}
    counts, _ = np.histogram(arr, bins=bins, range=value_range)
    total = float(counts.sum()) or 1.0
    return {f"{prefix}_bin_{i}": float(c) / total for i, c in enumerate(counts)}


def histogram_features_edges(values: List[float], bin_edges: Sequence[float], prefix: str) -> Dict[str, float]:
    nb = len(bin_edges) - 1
    if not values:
        return {f"{prefix}_bin_{i}": 0.0 for i in range(nb)}
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{prefix}_bin_{i}": 0.0 for i in range(nb)}
    edges = np.array(bin_edges, dtype=float)
    counts, _ = np.histogram(arr, bins=edges)
    total = float(counts.sum()) or 1.0
    return {f"{prefix}_bin_{i}": float(c) / total for i, c in enumerate(counts)}


def entropy(values: List[float]) -> float:
    if not values:
        return float("nan")
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    arr = arr - arr.min()
    s = float(arr.sum())
    if s == 0.0:
        return 0.0
    p = arr / s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def fit_quantile_bins(values: List[float], quantiles: Sequence[float], lo: float = 0.0, hi: float = 1.0) -> List[float]:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [lo, 0.60, 0.75, 0.85, 0.92, 0.96, 0.985, 0.995, hi + 1e-6]
    arr = np.clip(arr, lo, hi)
    qs = np.quantile(arr, np.array(quantiles, dtype=float)).tolist()

    eps = 1e-6
    fixed: List[float] = [float(qs[0])]
    for x in qs[1:]:
        x = float(x)
        if x <= fixed[-1]:
            x = fixed[-1] + eps
        fixed.append(x)

    fixed[0] = lo
    fixed[-1] = hi + 1e-6
    return fixed
