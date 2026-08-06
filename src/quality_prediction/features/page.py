from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from quality_prediction.core.stats import (
    entropy,
    histogram_features,
    histogram_features_edges,
    safe_stats,
)
from quality_prediction.features.binning import ConfidenceBinConfig
from quality_prediction.features.image import ImageFeatureExtractor
from quality_prediction.features.ngrams import LMPerplexityScorer, NgramResource
from quality_prediction.features.lexicon import LexiconStore, lexical_features_from_tokens
from quality_prediction.io.htr_json import PageDocument, TextLine
from quality_prediction.features.dit_embeddings import DiTEmbeddingExtractor


def flatten_text(lines: Iterable[TextLine]) -> str:
    return "\n".join(l.full_text for l in lines)


def _finite_floats(values: Iterable[object]) -> List[float]:
    out: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            out.append(fv)
    return out


class PageFeatureExtractor:
    def __init__(
        self,
        image_feature_extractor: Optional[ImageFeatureExtractor] = None,
        ngram_resource: Optional[NgramResource] = None,
        lm_scorer: Optional[LMPerplexityScorer] = None,
        lexicons: Optional["LexiconStore"] = None,
        metadata: Optional[Dict[str, Any]] = None,
        bin_config: Optional[ConfidenceBinConfig] = None,
        dit_extractor: Optional[DiTEmbeddingExtractor] = None,
    ):
        self.image_feature_extractor = image_feature_extractor or ImageFeatureExtractor()
        self.ngram_resource = ngram_resource
        self.lm_scorer = lm_scorer
        self.lexicons = lexicons
        self.metadata = metadata or {}
        self.bin_config = bin_config
        self.dit_extractor = dit_extractor

    def _region_line_segmentation_features(self, page: PageDocument) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        regions = page.regions
        lines = page.all_lines

        feats["num_regions"] = float(len(regions))
        feats["num_lines"] = float(len(lines))

        region_confs = _finite_floats(r.segmentation_confidence for r in regions)
        line_confs = _finite_floats(l.segmentation_confidence for l in lines)

        # Important for make_regions pipeline: region confidences might be missing entirely
        feats["has_region_conf"] = 1.0 if len(region_confs) > 0 else 0.0

        feats.update({f"region_conf_{k}": v for k, v in safe_stats(region_confs).items()})
        feats.update({f"line_conf_{k}": v for k, v in safe_stats(line_confs).items()})

        if self.bin_config is not None:
            feats.update(histogram_features_edges(region_confs, self.bin_config.region_conf_edges, "region_conf_hist"))
            feats.update(histogram_features_edges(line_confs, self.bin_config.line_conf_edges, "line_conf_hist"))
        else:
            feats.update(histogram_features(region_confs, bins=10, prefix="region_conf_hist", value_range=(0.0, 1.0)))
            feats.update(histogram_features(line_confs, bins=10, prefix="line_conf_hist", value_range=(0.0, 1.0)))

        region_areas = [r.bbox.area for r in regions]
        feats.update({f"region_area_{k}": v for k, v in safe_stats([float(x) for x in region_areas]).items()})

        region_aspects = [r.bbox.width / r.bbox.height for r in regions if r.bbox.height > 0]
        feats.update({f"region_aspect_{k}": v for k, v in safe_stats([float(x) for x in region_aspects]).items()})

        overlaps = 0
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                if regions[i].bbox.overlaps(regions[j].bbox):
                    overlaps += 1
        feats["region_overlap_count"] = float(overlaps)

        line_heights = [l.bbox.height for l in lines]
        line_widths = [l.bbox.width for l in lines]
        feats.update({f"line_height_{k}": v for k, v in safe_stats([float(x) for x in line_heights]).items()})
        feats.update({f"line_width_{k}": v for k, v in safe_stats([float(x) for x in line_widths]).items()})

        # Within-region line spacing (becomes empty if regions have 1 line each)
        spacings: List[float] = []
        for r in regions:
            sorted_lines = sorted(r.lines, key=lambda l: l.bbox.center[1])
            for a, b in zip(sorted_lines, sorted_lines[1:]):
                d = b.bbox.center[1] - a.bbox.center[1]
                if d > 0:
                    spacings.append(float(d))
        feats.update({f"line_spacing_{k}": v for k, v in safe_stats(spacings).items()})

        # Page-level line spacing (robust fallback for make_regions)
        page_spacings: List[float] = []
        sorted_page_lines = sorted(lines, key=lambda l: l.bbox.center[1])
        for a, b in zip(sorted_page_lines, sorted_page_lines[1:]):
            d = b.bbox.center[1] - a.bbox.center[1]
            if d > 0:
                page_spacings.append(float(d))
        feats.update({f"page_line_spacing_{k}": v for k, v in safe_stats(page_spacings).items()})

        char_counts = [l.char_count for l in lines]
        if char_counts:
            low = float(np.percentile(char_counts, 20))
            high = float(np.percentile(char_counts, 80))
            feats["frac_short_lines"] = float(sum(1 for c in char_counts if c < low) / len(char_counts))
            feats["frac_long_lines"] = float(sum(1 for c in char_counts if c > high) / len(char_counts))
        else:
            feats["frac_short_lines"] = float("nan")
            feats["frac_long_lines"] = float("nan")

        line_overlaps = 0
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if lines[i].bbox.overlaps(lines[j].bbox):
                    line_overlaps += 1
        feats["line_overlap_count"] = float(line_overlaps)

        feats["region_to_line_ratio"] = float(len(lines)) / float(len(regions)) if regions else float("nan")

        inversions = 0
        for r in regions:
            ys = [l.bbox.center[1] for l in r.lines]
            for a, b in zip(ys, ys[1:]):
                if b < a:
                    inversions += 1
        feats["reading_order_inversions"] = float(inversions)

        feats["region_conf_entropy"] = entropy(region_confs)
        feats["line_conf_entropy"] = entropy(line_confs)
        return feats

    def _regionization_features(self, page: PageDocument) -> Dict[str, float]:
        """
        Features describing how lines are grouped into regions.
        Very useful when regions are programmatically created (make_regions pipeline),
        but also valid for the nested pipeline.
        """
        feats: Dict[str, float] = {}
        regions = page.regions
        if not regions:
            feats["lines_per_region_mean"] = float("nan")
            feats["lines_per_region_std"] = float("nan")
            feats["lines_per_region_min"] = float("nan")
            feats["lines_per_region_max"] = float("nan")
            feats["lines_per_region_cv"] = float("nan")
            feats["frac_single_line_regions"] = float("nan")
            feats["frac_multi_line_regions"] = float("nan")
            feats["max_lines_in_region"] = float("nan")
            feats["region_tightness_area_ratio_mean"] = float("nan")
            feats["region_tightness_area_ratio_std"] = float("nan")
            feats["region_tightness_area_ratio_min"] = float("nan")
            feats["region_tightness_area_ratio_max"] = float("nan")
            feats["region_tightness_area_ratio_cv"] = float("nan")
            feats["region_height_to_lines_height_ratio_mean"] = float("nan")
            feats["region_height_to_lines_height_ratio_std"] = float("nan")
            feats["region_height_to_lines_height_ratio_min"] = float("nan")
            feats["region_height_to_lines_height_ratio_max"] = float("nan")
            feats["region_height_to_lines_height_ratio_cv"] = float("nan")
            feats["region_xmin_std_mean"] = float("nan")
            feats["region_xmax_std_mean"] = float("nan")
            feats["region_centerx_std_mean"] = float("nan")
            return feats

        lpr = [len(r.lines) for r in regions]
        feats.update({f"lines_per_region_{k}": v for k, v in safe_stats([float(x) for x in lpr]).items()})

        n = len(regions)
        single = sum(1 for x in lpr if x == 1)
        multi = sum(1 for x in lpr if x >= 2)
        feats["frac_single_line_regions"] = float(single / max(1, n))
        feats["frac_multi_line_regions"] = float(multi / max(1, n))
        feats["max_lines_in_region"] = float(max(lpr) if lpr else 0.0)

        ratios: List[float] = []
        height_ratios: List[float] = []
        xmin_stds: List[float] = []
        xmax_stds: List[float] = []
        cx_stds: List[float] = []

        for r in regions:
            if not r.lines:
                continue

            region_area = float(r.bbox.area) if r.bbox.area is not None else 0.0
            if region_area <= 0:
                continue

            sum_line_areas = float(sum(l.bbox.area for l in r.lines)) or 0.0
            if sum_line_areas > 0:
                ratios.append(region_area / sum_line_areas)

            region_h = float(r.bbox.height) or 0.0
            sum_line_h = float(sum(l.bbox.height for l in r.lines)) or 0.0
            if region_h > 0 and sum_line_h > 0:
                height_ratios.append(region_h / sum_line_h)

            if len(r.lines) >= 2:
                xmins = [float(l.bbox.xmin) for l in r.lines]
                xmaxs = [float(l.bbox.xmax) for l in r.lines]
                cxs = [float(l.bbox.center[0]) for l in r.lines]
                xmin_stds.append(float(np.std(np.asarray(xmins, dtype=float))))
                xmax_stds.append(float(np.std(np.asarray(xmaxs, dtype=float))))
                cx_stds.append(float(np.std(np.asarray(cxs, dtype=float))))

        feats.update({f"region_tightness_area_ratio_{k}": v for k, v in safe_stats(ratios).items()})
        feats.update({f"region_height_to_lines_height_ratio_{k}": v for k, v in safe_stats(height_ratios).items()})

        # Only meaningful when regions have >=2 lines; otherwise mostly empty -> NaN
        feats["region_xmin_std_mean"] = float(np.mean(xmin_stds)) if xmin_stds else float("nan")
        feats["region_xmax_std_mean"] = float(np.mean(xmax_stds)) if xmax_stds else float("nan")
        feats["region_centerx_std_mean"] = float(np.mean(cx_stds)) if cx_stds else float("nan")

        return feats

    def _layout_features(self, page: PageDocument) -> Dict[str, float]:
        page_bbox = page.page_bbox
        if page_bbox is None:
            return {
                "text_density": float("nan"),
                "num_columns_est": float("nan"),
                "layout_fragmentation_score": float("nan"),
            }

        lines = page.all_lines
        page_area = page_bbox.area or 1.0
        text_area = sum(l.bbox.area for l in lines)
        text_density = float(text_area / page_area)

        x_centers = np.array([l.bbox.center[0] for l in lines], dtype=float) if lines else np.array([], dtype=float)
        if x_centers.size == 0:
            num_cols = float("nan")
        else:
            hist, _ = np.histogram(x_centers, bins=10)
            thr = 0.2 * hist.max() if hist.size else 0.0
            cols = int((hist > thr).sum())
            num_cols = float(max(1, min(cols, 4)))

        small_region_count = sum(1 for r in page.regions if r.bbox.area < 0.01 * page_area)
        frag = float(small_region_count / max(1, len(page.regions)))

        return {"text_density": text_density, "num_columns_est": num_cols, "layout_fragmentation_score": frag}

    def _htr_confidence_features(self, page: PageDocument) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        lines = page.all_lines

        # Filter non-finite / None to avoid float(None) crashes
        line_scores = _finite_floats(l.text_result.best_score for l in lines)
        feats.update({f"htr_line_score_{k}": v for k, v in safe_stats(line_scores).items()})

        if self.bin_config is not None:
            feats.update(histogram_features_edges(line_scores, self.bin_config.htr_line_edges, "htr_line_score_hist"))
        else:
            feats.update(histogram_features(line_scores, bins=10, prefix="htr_line_score_hist", value_range=(0.0, 1.0)))

        # token_confidences should already be floats, but filter anyway
        token_scores = _finite_floats(s for l in lines for s in getattr(l, "token_confidences", []))
        feats.update({f"htr_token_score_{k}": v for k, v in safe_stats(token_scores).items()})

        if self.bin_config is not None:
            feats.update(histogram_features_edges(token_scores, self.bin_config.htr_token_edges, "htr_token_score_hist"))
        else:
            feats.update(histogram_features(token_scores, bins=10, prefix="htr_token_score_hist", value_range=(0.0, 1.0)))

        feats["htr_token_score_entropy"] = entropy(token_scores)

        char_counts = [l.char_count for l in lines]
        word_counts = [l.word_count for l in lines]
        feats.update({f"line_char_count_{k}": v for k, v in safe_stats([float(x) for x in char_counts]).items()})
        feats.update({f"line_word_count_{k}": v for k, v in safe_stats([float(x) for x in word_counts]).items()})
        feats["total_chars"] = float(sum(char_counts))
        feats["total_words"] = float(sum(word_counts))
        return feats

    def _token_category_features(self, page: PageDocument) -> Dict[str, float]:
        text = flatten_text(page.all_lines)
        chars = list(text)
        if not chars:
            return {
                "char_digit_ratio": float("nan"),
                "char_punct_ratio": float("nan"),
                "char_upper_ratio": float("nan"),
                "char_unknown_ratio": float("nan"),
                "frac_no_alpha_lines": float("nan"),
                "long_repetition_run_count": float("nan"),
            }

        total = len(chars)
        digits = sum(c.isdigit() for c in chars)
        punct = sum((not c.isalnum()) and (not c.isspace()) for c in chars)
        upper = sum(c.isupper() for c in chars)
        unknown = sum(1 for c in chars if (not (32 <= ord(c) <= 126)) and (not (0x00C0 <= ord(c) <= 0x017F)))

        lines = page.all_lines
        no_alpha_lines = sum(1 for l in lines if not any(ch.isalpha() for ch in l.full_text))

        long_runs = 0
        for l in lines:
            t = l.full_text
            if not t:
                continue
            run = 1
            for a, b in zip(t, t[1:]):
                if a == b:
                    run += 1
                else:
                    if run >= 5:
                        long_runs += 1
                    run = 1
            if run >= 5:
                long_runs += 1

        return {
            "char_digit_ratio": float(digits / total),
            "char_punct_ratio": float(punct / total),
            "char_upper_ratio": float(upper / total),
            "char_unknown_ratio": float(unknown / total),
            "frac_no_alpha_lines": float(no_alpha_lines / len(lines)) if lines else float("nan"),
            "long_repetition_run_count": float(long_runs),
        }

    def _ngram_features(self, page: PageDocument) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        if self.ngram_resource is None:
            for n in range(2, 8):
                feats[f"char_{n}gram_ratio_present"] = float("nan")
            return feats
        text = flatten_text(page.all_lines)
        for n in range(2, 8):
            feats[f"char_{n}gram_ratio_present"] = float(self.ngram_resource.ratio_present(text, n))
        return feats

    def _lm_perplexity_features(self, page: PageDocument) -> Dict[str, float]:
        if self.lm_scorer is None:
            return {
                "page_ppl": float("nan"),
                "line_ppl_mean": float("nan"),
                "line_ppl_std": float("nan"),
                "line_ppl_var": float("nan"),
            }

        texts = [l.full_text for l in page.all_lines]
        ppl_vals = [float(self.lm_scorer.line_ppl(t)) for t in texts if t.strip()]
        ppl_vals = [p for p in ppl_vals if not math.isnan(p)]

        if ppl_vals:
            arr = np.array(ppl_vals, dtype=float)
            line_mean = float(arr.mean())
            line_std = float(arr.std())
            line_var = float(arr.var())
        else:
            line_mean = line_std = line_var = float("nan")

        page_text = flatten_text(page.all_lines)
        return {
            "page_ppl": float(self.lm_scorer.page_ppl(page_text)),
            "line_ppl_mean": line_mean,
            "line_ppl_std": line_std,
            "line_ppl_var": line_var,
        }

    def _lexicality_features(self, page: PageDocument) -> Dict[str, float]:
        if self.lexicons is None:
            return {
                "lexicality_word_ratio": float("nan"),
                "lex_combined_match_ratio": float("nan"),
                "lex_combined_oov_ratio": float("nan"),
            }

        # Always derive tokens from line text (works for all PageDocument variants)
        text = flatten_text(page.all_lines).replace("\n", " ")
        tokens = [t for t in text.split() if t]

        feats = lexical_features_from_tokens(tokens, self.lexicons)
        feats["lexicality_word_ratio"] = feats.get("lex_combined_match_ratio", float("nan"))
        return feats


    def _spatial_text_interaction_features(self, page: PageDocument) -> Dict[str, float]:
        if self.lm_scorer is None:
            return {"corr_line_conf_vs_ppl": float("nan")}

        pairs: List[Tuple[float, float]] = []
        for l in page.all_lines:
            if not l.full_text.strip():
                continue
            conf = l.text_result.best_score
            if conf is None:
                continue
            conf = float(conf)
            if not np.isfinite(conf):
                continue

            ppl = float(self.lm_scorer.line_ppl(l.full_text))
            if math.isnan(ppl):
                continue
            pairs.append((conf, ppl))

        if len(pairs) < 2:
            return {"corr_line_conf_vs_ppl": float("nan")}

        arr_conf = np.array([c for c, _ in pairs], dtype=float)
        arr_ppl = np.array([p for _, p in pairs], dtype=float)
        if arr_conf.std() == 0 or arr_ppl.std() == 0:
            return {"corr_line_conf_vs_ppl": float("nan")}
        return {"corr_line_conf_vs_ppl": float(np.corrcoef(arr_conf, arr_ppl)[0, 1])}

    def _metadata_features(self) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        if self.metadata.get("century") is not None:
            feats["century"] = float(self.metadata["century"])
        if self.metadata.get("script_type") is not None:
            feats["script_type_code"] = float(hash(self.metadata["script_type"]) % 1000)
        return feats

    def extract_features(self, page: PageDocument) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        feats.update(self.image_feature_extractor.extract_all(page.image_path))

        if self.dit_extractor is not None:
            feats.update(self.dit_extractor.extract_all(page.image_path))

        feats.update(self._region_line_segmentation_features(page))
        feats.update(self._regionization_features(page))  # NEW: robust for make_regions, useful for nested too
        feats.update(self._layout_features(page))
        feats.update(self._htr_confidence_features(page))
        feats.update(self._token_category_features(page))
        feats.update(self._ngram_features(page))
        feats.update(self._lm_perplexity_features(page))
        feats.update(self._lexicality_features(page))
        feats.update(self._spatial_text_interaction_features(page))
        feats.update(self._metadata_features())
        return feats
