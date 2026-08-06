# quality_prediction/modeling/xgb/feature_sets.py
from __future__ import annotations

import re
from typing import Sequence

import pandas as pd


CONF_REGEXES_DEFAULT = [
    r"conf", r"confidence",
    r"htr_line_score",
    r"htr_token_score",
    r"region_conf",
    r"line_conf",
    r"_score_hist",
    r"_conf_hist",
]

NGRAM_REGEXES_DEFAULT = [
    r"gram",
    r"char_\d+gram_ratio_present",
    r"page_ppl",
    r"line_ppl_",
]

# Handcrafted image feature name patterns (from your ImageFeatureExtractor)
IMAGE_REGEXES_DEFAULT = [
    r"^img_",                 # img_mean_intensity, img_std_intensity, img_contrast
    r"^blur_score_",          # blur_score_var_laplacian
    r"^noise_",               # noise_std_diff_median
    r"^fg_ratio$",            # fg_ratio
    r"^stroke_width_",        # stroke_width_mean, stroke_width_std
    r"^cc_count$",            # cc_count
    r"^skew_angle_deg$",      # skew_angle_deg
]

# DiT embedding prefix (change if you used a different prefix)
DIT_PREFIX_DEFAULT = "dit_emb"

# NEW: lexical feature patterns (LexiconStore + lexicality)
LEXICAL_REGEXES_DEFAULT = [
    r"^lex_",                 # lex_<dict>_match_ratio/count/etc + lex_combined_*
    r"^lexicality_",          # lexicality_word_ratio, etc.
]

# NEW: ngram-like features from JSON language models (to be excluded from json_model_only)
# (kept separate so we can exclude them without reusing NGRAM regex semantics)
NGRAM_LM_EXCLUDE_DEFAULT = [
    r"^char_\d+gram_ratio_present$",  # char_2gram_ratio_present ...
    r"^page_ppl$",
    r"^line_ppl_",
]


def _matches_any(col: str, regexes: Sequence[str]) -> bool:
    cl = col.lower()
    return any(re.search(rx, cl) is not None for rx in regexes)


def get_feature_set(
    name: str,
    X: pd.DataFrame,
    *,
    single_feature_name: str = "htr_line_score_mean",
    conf_regexes: Sequence[str] = CONF_REGEXES_DEFAULT,
    ngram_regexes: Sequence[str] = NGRAM_REGEXES_DEFAULT,
    image_regexes: Sequence[str] = IMAGE_REGEXES_DEFAULT,
    dit_prefix: str = DIT_PREFIX_DEFAULT,
    lexical_regexes: Sequence[str] = LEXICAL_REGEXES_DEFAULT,
    ngram_lm_exclude: Sequence[str] = NGRAM_LM_EXCLUDE_DEFAULT,
) -> pd.DataFrame:
    cols = list(X.columns)

    if name == "full":
        return X

    if name == "single_htr_line_score_mean":
        if single_feature_name not in cols:
            raise KeyError(f"Feature {single_feature_name!r} not found in X.")
        return X[[single_feature_name]].copy()

    if name == "confidence_only":
        sel = [c for c in cols if _matches_any(c, conf_regexes)]
        if not sel:
            raise RuntimeError("confidence_only selected 0 features. Adjust CONF regexes.")
        return X[sel].copy()

    if name == "ngram_only":
        sel = [c for c in cols if _matches_any(c, ngram_regexes)]
        if not sel:
            raise RuntimeError("ngram_only selected 0 features. Adjust NGRAM regexes.")
        return X[sel].copy()

    # -------------------------
    # image_only
    # -------------------------
    if name == "image_only":
        sel = [c for c in cols if _matches_any(c, image_regexes)]
        if not sel:
            raise RuntimeError("image_only selected 0 features. Adjust IMAGE regexes.")
        return X[sel].copy()

    # -------------------------
    # dit_only
    # -------------------------
    if name == "dit_only":
        dp = dit_prefix.lower()
        sel = [c for c in cols if c.lower().startswith(dp)]
        if not sel:
            raise RuntimeError(f"dit_only selected 0 features. No columns start with {dit_prefix!r}.")
        return X[sel].copy()

    # -------------------------
    # NEW: lexical_only
    # -------------------------
    if name == "lexical_only":
        sel = [c for c in cols if _matches_any(c, lexical_regexes)]
        if not sel:
            raise RuntimeError("lexical_only selected 0 features. Adjust LEXICAL regexes.")
        return X[sel].copy()

    # -------------------------
    # NEW: json_model_only
    #
    # "Only model output in the JSON group":
    #   INCLUDE: features derived from PageDocument JSON model outputs
    #            (segmentation/HTR confidences + their stats/hists, geometry/layout derived from JSON bboxes)
    #   EXCLUDE: image features, DiT embeddings, ngram/perplexity features, lexical matching features
    #
    # This works purely by excluding known non-JSON sources (image/dit)
    # and excluding known "text plausibility" families (ngram + lexical).
    # -------------------------
    if name == "json_model_only":
        dp = dit_prefix.lower()

        sel = []
        for c in cols:
            cl = c.lower()

            # exclude pixel-derived image features
            if _matches_any(c, image_regexes):
                continue

            # exclude DiT embeddings
            if cl.startswith(dp):
                continue

            # exclude lexical matching / lexicality
            if _matches_any(c, lexical_regexes):
                continue

            # exclude ngram + perplexity family
            if _matches_any(c, ngram_regexes) or _matches_any(c, ngram_lm_exclude):
                continue

            # keep the rest (these should be JSON-derived + maybe metadata)
            sel.append(c)

        if not sel:
            raise RuntimeError(
                "json_model_only selected 0 features. "
                "This likely means your dataframe only contains excluded families."
            )
        return X[sel].copy()

    raise ValueError(f"Unknown feature set: {name}")
