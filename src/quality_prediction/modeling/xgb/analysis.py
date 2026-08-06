from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore


def remap_xgb_feature_names(score_dict: Dict[str, float], feature_names: List[str]) -> Dict[str, float]:
    mapped: Dict[str, float] = {}
    for k, v in score_dict.items():
        if k in feature_names:
            mapped[k] = v
        elif k.startswith("f") and k[1:].isdigit():
            idx = int(k[1:])
            mapped[feature_names[idx]] = v if 0 <= idx < len(feature_names) else v
        else:
            mapped[k] = v
    return mapped


def compute_feature_importance_for_target(
    target_col: str,
    model: "XGBRegressor",
    X: pd.DataFrame,
    df: pd.DataFrame,
    out_dir: Path,
    suffix: str = "",
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    booster = model.get_booster()
    feature_names = list(X.columns)

    raw_gain = booster.get_score(importance_type="gain")
    raw_weight = booster.get_score(importance_type="weight")
    raw_cover = booster.get_score(importance_type="cover")

    gain = remap_xgb_feature_names(raw_gain, feature_names)
    weight = remap_xgb_feature_names(raw_weight, feature_names)
    cover = remap_xgb_feature_names(raw_cover, feature_names)

    def normalize(d: Dict[str, float]) -> Dict[str, float]:
        total = float(sum(d.values())) if d else 0.0
        if total <= 0:
            return {k: 0.0 for k in d}
        return {k: float(v) / total for k, v in d.items()}

    gain_norm = normalize(gain)
    weight_norm = normalize(weight)
    cover_norm = normalize(cover)

    y = df[target_col]
    corrs: Dict[str, float] = {}
    for feat in feature_names:
        if df[feat].dtype.kind in "bifc":
            corrs[feat] = float(df[feat].corr(y))
        else:
            corrs[feat] = float("nan")

    rows = []
    for feat in feature_names:
        rows.append(dict(
            feature=feat,
            gain=gain.get(feat, 0.0),
            gain_norm=gain_norm.get(feat, 0.0),
            weight=weight.get(feat, 0.0),
            weight_norm=weight_norm.get(feat, 0.0),
            cover=cover.get(feat, 0.0),
            cover_norm=cover_norm.get(feat, 0.0),
            corr_with_target=corrs.get(feat, float("nan")),
        ))

    df_imp = pd.DataFrame(rows).sort_values("gain_norm", ascending=False)

    suffix_part = f"_{suffix}" if suffix else ""
    out_csv = out_dir / f"feature_importance_{target_col}{suffix_part}.csv"
    df_imp.to_csv(out_csv, index=False)
    return df_imp
