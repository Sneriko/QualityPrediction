from __future__ import annotations

import random
from typing import Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .weights import weighted_mae
from .config import HPOConfig, WeightingConfig

try:
    from xgboost import XGBRegressor
except Exception as e:  # pragma: no cover
    XGBRegressor = None  # type: ignore


def random_search_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    w_val: np.ndarray,
    *,
    hpo: HPOConfig,
    weighting: WeightingConfig,
    target_name: str,
    n_trials: int,
    per_trial_logger: Optional[callable] = None,
) -> Tuple["XGBRegressor", Dict[str, Any], float, float]:
    if XGBRegressor is None:
        raise RuntimeError("xgboost is not available. Please install xgboost.")

    param_space = {
        "n_estimators":      [400, 600, 800, 1000, 1200, 1600],
        "max_depth":         [3, 4, 5, 6, 7],
        "learning_rate":     [0.01, 0.02, 0.03, 0.05, 0.1],
        "subsample":         [0.6, 0.8, 1.0],
        "colsample_bytree":  [0.6, 0.8, 1.0],
        "colsample_bynode":  [0.5, 0.8, 1.0],
        "min_child_weight":  [1, 3, 5, 10],
        "gamma":             [0.0, 0.1, 0.3],
        "reg_alpha":         [0.0, 0.1, 0.5],
        "reg_lambda":        [0.5, 1.0, 2.0],
    }

    best_score = float("inf")
    best_params = dict(hpo.base_params)
    best_model: Optional["XGBRegressor"] = None
    best_trial = -1
    best_mae = float("nan")
    best_wmae = float("nan")

    for trial in range(1, n_trials + 1):
        params = dict(hpo.base_params)
        for k, choices in param_space.items():
            params[k] = random.choice(choices)

        model = XGBRegressor(**params)
        model.set_params(early_stopping_rounds=hpo.early_stopping_rounds)

        fit_kwargs = dict(
            X=X_train,
            y=y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        if weighting.enabled:
            fit_kwargs["sample_weight"] = w_train

        model.fit(**fit_kwargs)

        y_pred = model.predict(X_val)
        mae = float(mean_absolute_error(y_val, y_pred))

        wmae = float("nan")
        if weighting.report_weighted_mae and weighting.enabled:
            wmae = float(weighted_mae(y_val.values, y_pred, w_val))

        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None:
            best_iter = params["n_estimators"]

        if per_trial_logger is not None:
            per_trial_logger(
                target=target_name,
                trial=trial,
                is_best_for_target=0,
                val_mae=mae,
                val_wmae=(wmae if (weighting.report_weighted_mae and weighting.enabled) else ""),
                best_iteration=int(best_iter),
                **{k: params.get(k, "") for k in [
                    "n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
                    "min_child_weight", "gamma", "reg_alpha", "reg_lambda"
                ]},
                use_sample_weights=int(weighting.enabled),
                weight_alpha=weighting.alpha,
                weight_p=weighting.p,
                weight_clip_max=weighting.clip_max,
            )

        # selection criterion
        score = wmae if (hpo.select_by == "wmae" and weighting.report_weighted_mae and weighting.enabled) else mae

        if score < best_score:
            best_score = float(score)
            best_params = params
            best_model = model
            best_trial = trial
            best_mae = mae
            best_wmae = wmae

    if best_model is None:
        raise RuntimeError("Random search produced no model.")

    # mark best
    if per_trial_logger is not None:
        best_iter = getattr(best_model, "best_iteration", None)
        if best_iter is None:
            best_iter = best_params["n_estimators"]
        per_trial_logger(
            target=target_name,
            trial=best_trial,
            is_best_for_target=1,
            val_mae=float(best_mae),
            val_wmae=(float(best_wmae) if (weighting.report_weighted_mae and weighting.enabled) else ""),
            best_iteration=int(best_iter),
            **{k: best_params.get(k, "") for k in [
                "n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
                "min_child_weight", "gamma", "reg_alpha", "reg_lambda"
            ]},
            use_sample_weights=int(weighting.enabled),
            weight_alpha=weighting.alpha,
            weight_p=weighting.p,
            weight_clip_max=weighting.clip_max,
        )

    return best_model, best_params, float(best_mae), float(best_wmae)
