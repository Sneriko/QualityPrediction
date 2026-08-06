# quality_prediction/features/binning.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class ConfidenceBinConfig:
    region_conf_edges: List[float]
    line_conf_edges: List[float]
    htr_line_edges: List[float]
    htr_token_edges: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_conf_edges": self.region_conf_edges,
            "line_conf_edges": self.line_conf_edges,
            "htr_line_edges": self.htr_line_edges,
            "htr_token_edges": self.htr_token_edges,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConfidenceBinConfig":
        return cls(
            region_conf_edges=list(d["region_conf_edges"]),
            line_conf_edges=list(d["line_conf_edges"]),
            htr_line_edges=list(d["htr_line_edges"]),
            htr_token_edges=list(d["htr_token_edges"]),
        )


_DEFAULT_FALLBACK = [0.0, 0.60, 0.75, 0.85, 0.92, 0.96, 0.985, 0.995, 1.0 + 1e-6]


def sanitize_bin_edges(
    edges: Sequence[float] | None,
    lo: float = 0.0,
    hi: float = 1.0,
    eps: float = 1e-6,
) -> List[float]:
    """
    Guarantee a strictly increasing edge array for np.histogram.
    - removes non-finite
    - sorts
    - enforces strict increase with eps
    - forces first=lo and last>=hi+eps
    """
    if not edges:
        base = list(_DEFAULT_FALLBACK)
        base[0] = lo
        base[-1] = hi + eps
        return base

    arr = np.asarray(list(edges), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        base = list(_DEFAULT_FALLBACK)
        base[0] = lo
        base[-1] = hi + eps
        return base

    arr = np.sort(arr)

    fixed = [float(arr[0])]
    for x in arr[1:]:
        x = float(x)
        if x <= fixed[-1]:
            x = fixed[-1] + eps
        fixed.append(x)

    fixed[0] = lo
    if fixed[-1] <= hi:
        fixed[-1] = hi + eps

    # If everything collapsed, fall back
    if len(fixed) < 2 or not np.all(np.diff(np.asarray(fixed)) > 0):
        base = list(_DEFAULT_FALLBACK)
        base[0] = lo
        base[-1] = hi + eps
        return base

    return fixed


def histogram_features_edges(
    values: List[float],
    bin_edges: Sequence[float],
    prefix: str,
) -> Dict[str, float]:
    edges = sanitize_bin_edges(bin_edges, lo=0.0, hi=1.0)
    nb = len(edges) - 1

    if not values:
        return {f"{prefix}_bin_{i}": 0.0 for i in range(nb)}

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{prefix}_bin_{i}": 0.0 for i in range(nb)}

    counts, _ = np.histogram(arr, bins=np.asarray(edges, dtype=float))
    total = float(counts.sum()) or 1.0
    return {f"{prefix}_bin_{i}": float(c) / total for i, c in enumerate(counts)}


def fit_quantile_bins(
    values: List[float],
    quantiles: Sequence[float],
    lo: float = 0.0,
    hi: float = 1.0,
) -> List[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return sanitize_bin_edges(_DEFAULT_FALLBACK, lo=lo, hi=hi)

    arr = np.clip(arr, lo, hi)
    qs = np.quantile(arr, np.asarray(quantiles, dtype=float)).tolist()
    return sanitize_bin_edges(qs, lo=lo, hi=hi)


class ConfidenceBinFitter:
    DEFAULT_Q = [0.0, 0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 0.93, 0.97, 0.99, 1.0]

    def __init__(self, quantiles: Optional[Sequence[float]] = None):
        self.quantiles = list(quantiles) if quantiles is not None else list(self.DEFAULT_Q)

    def fit_from_pages(self, pages: Iterable["PageDocument"]) -> ConfidenceBinConfig:
        region_vals: List[float] = []
        line_vals: List[float] = []
        htr_line_vals: List[float] = []
        htr_token_vals: List[float] = []

        for page in pages:
            for r in page.regions:
                c = r.segmentation_confidence
                if c is not None and np.isfinite(c):
                    region_vals.append(float(c))
            for l in page.all_lines:
                c = l.segmentation_confidence
                if c is not None and np.isfinite(c):
                    line_vals.append(float(c))

            for l in page.all_lines:
                s = l.text_result.best_score
                if s is not None and np.isfinite(s):
                    htr_line_vals.append(float(s))
                for _, ts in l.token_scores:
                    if ts is not None and np.isfinite(ts):
                        htr_token_vals.append(float(ts))

        return ConfidenceBinConfig(
            region_conf_edges=fit_quantile_bins(region_vals, self.quantiles, lo=0.0, hi=1.0),
            line_conf_edges=fit_quantile_bins(line_vals, self.quantiles, lo=0.0, hi=1.0),
            htr_line_edges=fit_quantile_bins(htr_line_vals, self.quantiles, lo=0.0, hi=1.0),
            htr_token_edges=fit_quantile_bins(htr_token_vals, self.quantiles, lo=0.0, hi=1.0),
        )

    def save(self, cfg: ConfidenceBinConfig, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2)

    def load(self, path: str) -> ConfidenceBinConfig:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)

        # sanitize on load too (protect against older broken configs)
        cfg = ConfidenceBinConfig.from_dict(d)
        cfg.region_conf_edges = sanitize_bin_edges(cfg.region_conf_edges)
        cfg.line_conf_edges = sanitize_bin_edges(cfg.line_conf_edges)
        cfg.htr_line_edges = sanitize_bin_edges(cfg.htr_line_edges)
        cfg.htr_token_edges = sanitize_bin_edges(cfg.htr_token_edges)
        return cfg
