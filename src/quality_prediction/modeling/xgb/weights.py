from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import WeightingConfig


def make_sample_weights(y: pd.Series, target_name: str, cfg: WeightingConfig) -> np.ndarray:
    yv = y.astype(float).values

    if not cfg.enabled:
        return np.ones_like(yv, dtype=float)

    if cfg.apply_only_if_name_contains and (cfg.apply_only_if_name_contains not in target_name):
        return np.ones_like(yv, dtype=float)

    y_min = np.nanmin(yv)
    y_max = np.nanmax(yv)
    if y_min < -0.05 or y_max > 1.05:
        return np.ones_like(yv, dtype=float)

    w = 1.0 + cfg.alpha * np.power((1.0 - np.clip(yv, 0.0, 1.0)), cfg.p)
    w = np.clip(w, cfg.clip_min, cfg.clip_max)
    return w.astype(float)


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.asarray(w, dtype=float)
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        return float(np.mean(np.abs(y_true - y_pred)))
    return float(np.sum(w * np.abs(y_true - y_pred)) / w_sum)
