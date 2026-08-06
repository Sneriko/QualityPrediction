from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set, Sequence

from collections import Counter, defaultdict

import numpy as np
from PIL import Image
import cv2
import re

from ngram.ngrammodel import NgramModel  # <- from your previous ngram_model.py


# ------------------------------
# Confidence binning (NEW)
# ------------------------------

@dataclass
class ConfidenceBinConfig:
    """
    Frozen bin edges to be used consistently for train/val/test.
    """
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


def histogram_features_edges(
    values: List[float],
    bin_edges: Sequence[float],
    prefix: str,
) -> Dict[str, float]:
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


def fit_quantile_bins(
    values: List[float],
    quantiles: Sequence[float],
    lo: float = 0.0,
    hi: float = 1.0,
) -> List[float]:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        # robust fallback (still tail-focused)
        return [lo, 0.60, 0.75, 0.85, 0.92, 0.96, 0.985, 0.995, hi + 1e-6]

    arr = np.clip(arr, lo, hi)
    qs = np.quantile(arr, np.array(quantiles, dtype=float)).tolist()

    # Enforce strictly increasing edges (histogram stability)
    eps = 1e-6
    fixed: List[float] = [float(qs[0])]
    for x in qs[1:]:
        x = float(x)
        if x <= fixed[-1]:
            x = fixed[-1] + eps
        fixed.append(x)

    fixed[0] = lo
    fixed[-1] = hi + 1e-6  # include 1.0 safely
    return fixed


class ConfidenceBinFitter:
    """
    Fits bin edges from the raw confidence values in PageDocument JSONs.
    """

    # Dense near 1.0 to resolve the “everything is 0.95–1.0” pileup.
    DEFAULT_Q = [0.0, 0.01, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 0.93, 0.97, 0.99, 1.0]

    def __init__(self, quantiles: Optional[Sequence[float]] = None):
        self.quantiles = list(quantiles) if quantiles is not None else list(self.DEFAULT_Q)

    def fit_from_pages(self, pages: Iterable["PageDocument"]) -> ConfidenceBinConfig:
        region_vals: List[float] = []
        line_vals: List[float] = []
        htr_line_vals: List[float] = []
        htr_token_vals: List[float] = []

        for page in pages:
            # YOLO region + line segmentation confidences
            for r in page.regions:
                if r.segmentation_confidence is not None and np.isfinite(r.segmentation_confidence):
                    region_vals.append(float(r.segmentation_confidence))
            for l in page.all_lines:
                if l.segmentation_confidence is not None and np.isfinite(l.segmentation_confidence):
                    line_vals.append(float(l.segmentation_confidence))

            # TrOCR line + token confidences
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
            return ConfidenceBinConfig.from_dict(json.load(f))


# ------------------------------
# Data model mirroring JSON
# ------------------------------

@dataclass
class BBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.xmin + self.width / 2.0, self.ymin + self.height / 2.0)

    def overlaps(self, other: "BBox") -> bool:
        return not (
            self.xmax <= other.xmin
            or self.xmin >= other.xmax
            or self.ymax <= other.ymin
            or self.ymin >= other.ymax
        )

    def intersection_area(self, other: "BBox") -> int:
        x1 = max(self.xmin, other.xmin)
        y1 = max(self.ymin, other.ymin)
        x2 = min(self.xmax, other.xmax)
        y2 = min(self.ymax, other.ymax)
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)


@dataclass
class TextResult:
    texts: List[str]
    scores: List[float]

    @property
    def best_text(self) -> str:
        return self.texts[0] if self.texts else ""

    @property
    def best_score(self) -> float:
        return self.scores[0] if self.scores else float("nan")


@dataclass
class Word:
    label: str
    text_result: TextResult
    segmentation_label: str
    segmentation_confidence: Optional[float]
    bbox: BBox
    polygon: str

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "Word":
        tr = obj["text_result"]
        text_result = TextResult(
            texts=tr.get("texts", []),
            scores=tr.get("scores", []),
        )
        bbox = BBox(**obj["bbox"])
        return cls(
            label=obj["label"],
            text_result=text_result,
            segmentation_label=obj.get("segmentation_label", "word"),
            segmentation_confidence=obj.get("segmentation_confidence"),
            bbox=bbox,
            polygon=obj.get("polygon", ""),
        )


@dataclass
class TextLine:
    label: str
    text_result: TextResult
    token_scores: List[Tuple[str, float]]
    words: List[Word]
    segmentation_label: str
    segmentation_confidence: Optional[float]
    bbox: BBox
    polygon: str

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "TextLine":
        tr = obj["text_result"]
        text_result = TextResult(
            texts=tr.get("texts", []),
            scores=tr.get("scores", []),
        )
        token_scores = [(t, float(s)) for t, s in obj.get("token_scores", [])]
        word_objs = [Word.from_json(w) for w in obj.get("contains", [])]
        bbox = BBox(**obj["bbox"])
        return cls(
            label=obj["label"],
            text_result=text_result,
            token_scores=token_scores,
            words=word_objs,
            segmentation_label=obj.get("segmentation_label", "textline"),
            segmentation_confidence=obj.get("segmentation_confidence"),
            bbox=bbox,
            polygon=obj.get("polygon", ""),
        )

    @property
    def full_text(self) -> str:
        return self.text_result.best_text

    @property
    def token_confidences(self) -> List[float]:
        return [s for _, s in self.token_scores]

    @property
    def char_count(self) -> int:
        return len(self.full_text)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


@dataclass
class TextRegion:
    label: str
    lines: List[TextLine]
    segmentation_label: str
    segmentation_confidence: Optional[float]
    bbox: BBox
    polygon: str

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "TextRegion":
        raw_lines = obj.get("contains", [])
        line_objs: List[TextLine] = []

        for tl in raw_lines:
            if tl.get("segmentation_label", "textline") != "textline":
                continue

            tr = tl.get("text_result")
            if tr is None:
                continue

            texts = tr.get("texts", [])
            if not texts or not any(t.strip() for t in texts):
                continue

            line_objs.append(TextLine.from_json(tl))

        bbox = BBox(**obj["bbox"])
        return cls(
            label=obj["label"],
            lines=line_objs,
            segmentation_label=obj.get("segmentation_label", "textregion"),
            segmentation_confidence=obj.get("segmentation_confidence"),
            bbox=bbox,
            polygon=obj.get("polygon", ""),
        )


@dataclass
class PageDocument:
    file_name: str
    image_path: str
    image_name: str
    label: str
    regions: List[TextRegion]

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "PageDocument":
        raw_items = obj.get("contains", [])
        regions: List[TextRegion] = []

        def is_textline_like(item: Dict[str, Any]) -> bool:
            tr = item.get("text_result")
            if not isinstance(tr, dict):
                return False

            children = item.get("contains") or []
            has_child_textlines = any(ch.get("segmentation_label") == "textline" for ch in children)
            return not has_child_textlines

        for item in raw_items:
            seg_label = item.get("segmentation_label", "")

            if seg_label == "textline" or is_textline_like(item):
                try:
                    line_obj = TextLine.from_json(item)
                except KeyError:
                    continue

                bbox = line_obj.bbox
                region = TextRegion(
                    label=item.get("label", "implicit_region"),
                    lines=[line_obj],
                    segmentation_label="textregion",
                    segmentation_confidence=item.get("segmentation_confidence"),
                    bbox=bbox,
                    polygon=item.get("polygon", ""),
                )
                regions.append(region)
                continue

            if seg_label == "textregion":
                regions.append(TextRegion.from_json(item))
                continue

            regions.append(TextRegion.from_json(item))

        return cls(
            file_name=obj["file_name"],
            image_path=obj["image_path"],
            image_name=obj["image_name"],
            label=obj["label"],
            regions=regions,
        )

    @property
    def all_lines(self) -> List[TextLine]:
        lines: List[TextLine] = []
        for r in self.regions:
            lines.extend(r.lines)
        return lines

    @property
    def all_words(self) -> List[Word]:
        words: List[Word] = []
        for l in self.all_lines:
            words.extend(l.words)
        return words

    @property
    def page_bbox(self) -> Optional[BBox]:
        if not self.regions:
            return None
        xmin = min(r.bbox.xmin for r in self.regions)
        ymin = min(r.bbox.ymin for r in self.regions)
        xmax = max(r.bbox.xmax for r in self.regions)
        ymax = max(r.bbox.ymax for r in self.regions)
        return BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


# ----------------------------------
# Pluggable external resources
# ----------------------------------

class NgramResource:
    def __init__(self, ngram_sets: Dict[int, Set[str]]):
        self.ngram_sets = ngram_sets

    def ratio_present(self, text: str, n: int) -> float:
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        allowed = self.ngram_sets.get(n)
        if not allowed:
            return float("nan")
        grams = [text[i: i + n] for i in range(len(text) - n + 1)]
        if not grams:
            return float("nan")
        present = sum(1 for g in grams if g in allowed)
        return present / len(grams)


class LMPerplexityScorer:
    def page_ppl(self, text: str) -> float:
        raise NotImplementedError

    def line_ppl(self, line: str) -> float:
        raise NotImplementedError


class NgramLMPerplexityScorer(LMPerplexityScorer):
    def __init__(self, model: NgramModel, smoothing_k: float = 1.0):
        self.model = model
        self.smoothing_k = smoothing_k
        self.n = model.n
        self.level = model.level
        self._token_pattern = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

    def _tokenize(self, text: str) -> List[str]:
        if self.level == "char":
            return list(text)
        return self._token_pattern.findall(text)

    def _sequence_logprob_and_length(self, text: str) -> Tuple[float, int]:
        tokens = self._tokenize(text)
        tokens = [t for t in tokens if t.strip()]
        if not tokens:
            return 0.0, 0

        if self.n == 1:
            logp = 0.0
            for tok in tokens:
                p = self.model.prob((), tok, k=self.smoothing_k)
                if p <= 0.0:
                    p = 1e-12
                logp += math.log(p)
            return logp, len(tokens)

        if len(tokens) < self.n:
            return 0.0, 0

        logp = 0.0
        count = 0
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i: i + self.n - 1])
            tok = tokens[i + self.n - 1]
            p = self.model.prob(context, tok, k=self.smoothing_k)
            if p <= 0.0:
                p = 1e-12
            logp += math.log(p)
            count += 1

        return logp, count

    def _perplexity_from_logprob(self, logp: float, length: int) -> float:
        if length == 0:
            return float("nan")
        H = -logp / float(length)
        return float(math.exp(H))

    def page_ppl(self, text: str) -> float:
        text = text.strip()
        if not text:
            return float("nan")
        logp, length = self._sequence_logprob_and_length(text)
        return self._perplexity_from_logprob(logp, length)

    def line_ppl(self, line: str) -> float:
        line = line.strip()
        if not line:
            return float("nan")
        logp, length = self._sequence_logprob_and_length(line)
        return self._perplexity_from_logprob(logp, length)


class Lexicon:
    def __init__(self, words: Set[str]):
        self.words = words

    def contains(self, w: str) -> bool:
        return w in self.words


# ------------------------------
# Utility helpers
# ------------------------------

def safe_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "cv": float("nan")}
    arr = np.array(values, dtype=float)
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
    counts, _ = np.histogram(arr, bins=bins, range=value_range)
    total = float(counts.sum()) or 1.0
    return {f"{prefix}_bin_{i}": float(c) / total for i, c in enumerate(counts)}


def entropy(values: List[float]) -> float:
    if not values:
        return float("nan")
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    arr = arr - arr.min()
    s = arr.sum()
    if s == 0:
        return 0.0
    p = arr / s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def flatten_text(lines: Iterable[TextLine]) -> str:
    return "\n".join(l.full_text for l in lines)


# ------------------------------
# Image feature extraction
# ------------------------------

class ImageFeatureExtractor:
    def __init__(self, cv2_module=None):
        self.cv2 = cv2_module if cv2_module is not None else cv2

    def load_image(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("L")
        return np.array(img)

    def basic_stats(self, img: np.ndarray) -> Dict[str, float]:
        arr = img.astype(np.float32) / 255.0
        mean = float(arr.mean())
        std = float(arr.std())
        return {"img_mean_intensity": mean, "img_std_intensity": std, "img_contrast": std}

    def blur_score(self, img: np.ndarray) -> float:
        if self.cv2 is None:
            return float("nan")
        lap = self.cv2.Laplacian(img, ddepth=self.cv2.CV_64F)
        return float(lap.var())

    def noise_score(self, img: np.ndarray) -> float:
        if self.cv2 is None:
            return float("nan")
        blurred = self.cv2.medianBlur(img, 3)
        diff = img.astype(np.float32) - blurred.astype(np.float32)
        return float(diff.std())

    def binarisation_features(self, img: np.ndarray) -> Dict[str, float]:
        if self.cv2 is None:
            return {"fg_ratio": float("nan")}
        _, th = self.cv2.threshold(img, 0, 255, self.cv2.THRESH_BINARY + self.cv2.THRESH_OTSU)
        fg = th == 0
        return {"fg_ratio": float(fg.mean())}

    def stroke_width_stats(self, img: np.ndarray) -> Dict[str, float]:
        if self.cv2 is None:
            return {"stroke_width_mean": float("nan"), "stroke_width_std": float("nan")}
        _, th = self.cv2.threshold(img, 0, 255, self.cv2.THRESH_BINARY + self.cv2.THRESH_OTSU)
        fg = (255 - th).astype(np.uint8)
        dist = self.cv2.distanceTransform(fg, self.cv2.DIST_L2, 3)
        vals = dist[fg > 0]
        if vals.size == 0:
            return {"stroke_width_mean": float("nan"), "stroke_width_std": float("nan")}
        return {"stroke_width_mean": float(vals.mean()), "stroke_width_std": float(vals.std())}

    def connected_components(self, img: np.ndarray) -> Dict[str, float]:
        if self.cv2 is None:
            return {"cc_count": float("nan")}
        _, th = self.cv2.threshold(img, 0, 255, self.cv2.THRESH_BINARY + self.cv2.THRESH_OTSU)
        fg = (255 - th).astype(np.uint8)
        num_labels, _ = self.cv2.connectedComponents(fg)
        return {"cc_count": float(num_labels)}

    def skew_angle(self, img: np.ndarray) -> float:
        if self.cv2 is None:
            return float("nan")
        edges = self.cv2.Canny(img, 50, 150, apertureSize=3)
        lines = self.cv2.HoughLines(edges, 1, np.pi / 180.0, 200)
        if lines is None:
            return float("nan")
        angles = []
        for rho, theta in lines[:, 0]:
            angle_deg = (theta * 180.0 / np.pi) - 90.0
            if -45 <= angle_deg <= 45:
                angles.append(angle_deg)
        if not angles:
            return float("nan")
        return float(np.median(np.array(angles)))

    def extract_all(self, image_path: str) -> Dict[str, float]:
        img = self.load_image(image_path)
        feats: Dict[str, float] = {}
        feats.update(self.basic_stats(img))
        feats["blur_score_var_laplacian"] = self.blur_score(img)
        feats["noise_std_diff_median"] = self.noise_score(img)
        feats.update(self.binarisation_features(img))
        feats.update(self.stroke_width_stats(img))
        feats.update(self.connected_components(img))
        feats["skew_angle_deg"] = self.skew_angle(img)
        return feats


# ------------------------------
# Page-level feature extraction
# ------------------------------

class PageFeatureExtractor:
    def __init__(
        self,
        image_feature_extractor: Optional[ImageFeatureExtractor] = None,
        ngram_resource: Optional[NgramResource] = None,
        lm_scorer: Optional[LMPerplexityScorer] = None,
        lexicon: Optional[Lexicon] = None,
        metadata: Optional[Dict[str, Any]] = None,
        bin_config: Optional[ConfidenceBinConfig] = None,  # NEW
    ):
        self.image_feature_extractor = image_feature_extractor if image_feature_extractor is not None else ImageFeatureExtractor()
        self.ngram_resource = ngram_resource
        self.lm_scorer = lm_scorer
        self.lexicon = lexicon
        self.metadata = metadata or {}
        self.bin_config = bin_config  # NEW

    # ---- Region & line segmentation features ----

    def _region_line_segmentation_features(self, page: PageDocument) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        regions = page.regions
        lines = page.all_lines

        feats["num_regions"] = float(len(regions))
        feats["num_lines"] = float(len(lines))

        region_confs = [r.segmentation_confidence for r in regions if r.segmentation_confidence is not None]
        line_confs = [l.segmentation_confidence for l in lines if l.segmentation_confidence is not None]

        feats.update({f"region_conf_{k}": v for k, v in safe_stats(region_confs).items()})
        feats.update({f"line_conf_{k}": v for k, v in safe_stats(line_confs).items()})

        # --- HISTOGRAMS (UPDATED) ---
        if self.bin_config is not None:
            feats.update(histogram_features_edges(region_confs, self.bin_config.region_conf_edges, "region_conf_hist"))
            feats.update(histogram_features_edges(line_confs, self.bin_config.line_conf_edges, "line_conf_hist"))
        else:
            feats.update(histogram_features(region_confs, bins=10, prefix="region_conf_hist", value_range=(0.0, 1.0)))
            feats.update(histogram_features(line_confs, bins=10, prefix="line_conf_hist", value_range=(0.0, 1.0)))

        # Region geometry
        region_areas = [r.bbox.area for r in regions]
        feats.update({f"region_area_{k}": v for k, v in safe_stats(region_areas).items()})

        region_aspects = [r.bbox.width / r.bbox.height for r in regions if r.bbox.height > 0]
        feats.update({f"region_aspect_{k}": v for k, v in safe_stats(region_aspects).items()})

        overlaps = 0
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                if regions[i].bbox.overlaps(regions[j].bbox):
                    overlaps += 1
        feats["region_overlap_count"] = float(overlaps)

        # Line geometry
        line_heights = [l.bbox.height for l in lines]
        line_widths = [l.bbox.width for l in lines]
        feats.update({f"line_height_{k}": v for k, v in safe_stats(line_heights).items()})
        feats.update({f"line_width_{k}": v for k, v in safe_stats(line_widths).items()})

        # Line spacing
        spacings: List[float] = []
        for r in regions:
            sorted_lines = sorted(r.lines, key=lambda l: l.bbox.center[1])
            for a, b in zip(sorted_lines, sorted_lines[1:]):
                spas = b.bbox.center[1] - a.bbox.center[1]
                if spas > 0:
                    spacings.append(spas)
        feats.update({f"line_spacing_{k}": v for k, v in safe_stats(spacings).items()})

        # Short/long line fractions by char count
        char_counts = [l.char_count for l in lines]
        if char_counts:
            low_thresh = np.percentile(char_counts, 20)
            high_thresh = np.percentile(char_counts, 80)
            short = sum(1 for c in char_counts if c < low_thresh)
            long = sum(1 for c in char_counts if c > high_thresh)
            n = len(char_counts)
            feats["frac_short_lines"] = short / n
            feats["frac_long_lines"] = long / n
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
            y_centers = [l.bbox.center[1] for l in r.lines]
            for a, b in zip(y_centers, y_centers[1:]):
                if b < a:
                    inversions += 1
        feats["reading_order_inversions"] = float(inversions)

        feats["region_conf_entropy"] = entropy(region_confs)
        feats["line_conf_entropy"] = entropy(line_confs)

        return feats

    # ---- Layout complexity features ----

    def _layout_features(self, page: PageDocument) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        lines = page.all_lines
        page_bbox = page.page_bbox
        if page_bbox is None:
            return {
                "text_density": float("nan"),
                "num_columns_est": float("nan"),
                "layout_fragmentation_score": float("nan"),
            }

        page_area = page_bbox.area or 1
        text_area = sum(l.bbox.area for l in lines)
        feats["text_density"] = text_area / page_area

        x_centers = np.array([l.bbox.center[0] for l in lines]) if lines else np.array([])
        if x_centers.size == 0:
            feats["num_columns_est"] = float("nan")
        else:
            hist, _ = np.histogram(x_centers, bins=10)
            threshold = 0.2 * hist.max() if hist.size else 0.0
            cols = int((hist > threshold).sum())
            feats["num_columns_est"] = float(max(1, min(cols, 4)))

        small_region_count = sum(1 for r in page.regions if r.bbox.area < 0.01 * page_area)
        feats["layout_fragmentation_score"] = float(small_region_count / max(1, len(page.regions)))
        return feats

    # ---- HTR confidence & text-shape features ----

    def _htr_confidence_features(self, page: PageDocument) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        lines = page.all_lines

        # Line-level HTR scores (best score)
        line_scores = [l.text_result.best_score for l in lines]
        feats.update({f"htr_line_score_{k}": v for k, v in safe_stats(line_scores).items()})

        # --- HISTOGRAMS (UPDATED: data-driven bins if provided) ---
        if self.bin_config is not None:
            feats.update(histogram_features_edges(line_scores, self.bin_config.htr_line_edges, "htr_line_score_hist"))
        else:
            # fallback to your old hand-made bins (kept)
            line_bin_edges = np.array(
                [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99, 1.0000001],
                dtype=float,
            )
            if line_scores:
                arr = np.array(line_scores, dtype=float)
                counts, _ = np.histogram(arr, bins=line_bin_edges)
                total = float(counts.sum()) or 1.0
                for i, c in enumerate(counts):
                    feats[f"htr_line_score_hist_bin_{i}"] = float(c) / total
            else:
                for i in range(len(line_bin_edges) - 1):
                    feats[f"htr_line_score_hist_bin_{i}"] = 0.0

        # Token-level HTR scores
        token_scores = [s for l in lines for s in l.token_confidences]
        feats.update({f"htr_token_score_{k}": v for k, v in safe_stats(token_scores).items()})

        if self.bin_config is not None:
            feats.update(histogram_features_edges(token_scores, self.bin_config.htr_token_edges, "htr_token_score_hist"))
        else:
            bin_edges = np.array(
                [0.0, 0.5, 0.7, 0.85, 0.87, 0.89, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.001],
                dtype=float,
            )
            if token_scores:
                arr = np.array(token_scores, dtype=float)
                counts, _ = np.histogram(arr, bins=bin_edges)
                total = float(counts.sum()) or 1.0
                for i, c in enumerate(counts):
                    feats[f"htr_token_score_hist_bin_{i}"] = float(c) / total
            else:
                for i in range(len(bin_edges) - 1):
                    feats[f"htr_token_score_hist_bin_{i}"] = 0.0

        feats["htr_token_score_entropy"] = entropy(token_scores)

        # Text length & shape
        char_counts = [l.char_count for l in lines]
        word_counts = [l.word_count for l in lines]

        feats.update({f"line_char_count_{k}": v for k, v in safe_stats(char_counts).items()})
        feats.update({f"line_word_count_{k}": v for k, v in safe_stats(word_counts).items()})

        feats["total_chars"] = float(sum(char_counts))
        feats["total_words"] = float(sum(word_counts))
        return feats

    # ---- Token-category & structural anomaly features ----

    def _token_category_features(self, page: PageDocument) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        text = flatten_text(page.all_lines)
        chars = list(text)
        if not chars:
            feats.update({
                "char_digit_ratio": float("nan"),
                "char_punct_ratio": float("nan"),
                "char_upper_ratio": float("nan"),
                "char_unknown_ratio": float("nan"),
            })
            return feats

        total = len(chars)
        digits = sum(c.isdigit() for c in chars)
        punct = sum(not c.isalnum() and not c.isspace() for c in chars)
        upper = sum(c.isupper() for c in chars)

        unknown = 0
        for c in chars:
            code = ord(c)
            if (not (32 <= code <= 126)) and (not (0x00C0 <= code <= 0x017F)):
                unknown += 1

        feats["char_digit_ratio"] = digits / total
        feats["char_punct_ratio"] = punct / total
        feats["char_upper_ratio"] = upper / total
        feats["char_unknown_ratio"] = unknown / total

        lines = page.all_lines
        no_alpha_lines = sum(1 for l in lines if not any(ch.isalpha() for ch in l.full_text))
        feats["frac_no_alpha_lines"] = (no_alpha_lines / len(lines) if lines else float("nan"))

        long_runs = 0
        for l in lines:
            t = l.full_text
            if not t:
                continue
            run_len = 1
            for a, b in zip(t, t[1:]):
                if a == b:
                    run_len += 1
                else:
                    if run_len >= 5:
                        long_runs += 1
                    run_len = 1
            if run_len >= 5:
                long_runs += 1
        feats["long_repetition_run_count"] = float(long_runs)
        return feats

    # ---- N-gram LM-based features ----

    def _ngram_features(self, page: PageDocument) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        if self.ngram_resource is None:
            for n in range(2, 8):
                feats[f"char_{n}gram_ratio_present"] = float("nan")
            return feats

        text = flatten_text(page.all_lines)
        for n in range(2, 8):
            feats[f"char_{n}gram_ratio_present"] = self.ngram_resource.ratio_present(text, n)
        return feats

    # ---- Word/transformer LM-based perplexity features ----

    def _lm_perplexity_features(self, page: PageDocument) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        if self.lm_scorer is None:
            feats["page_ppl"] = float("nan")
            feats["line_ppl_mean"] = float("nan")
            feats["line_ppl_std"] = float("nan")
            feats["line_ppl_var"] = float("nan")
            return feats

        lines = page.all_lines
        texts = [l.full_text for l in lines]

        raw_ppl_vals = [self.lm_scorer.line_ppl(t) for t in texts if t.strip()]
        ppl_vals = [p for p in raw_ppl_vals if not math.isnan(p)]

        if ppl_vals:
            arr = np.array(ppl_vals, dtype=float)
            feats["line_ppl_mean"] = float(arr.mean())
            feats["line_ppl_std"] = float(arr.std())
            feats["line_ppl_var"] = float(arr.var())
        else:
            feats["line_ppl_mean"] = float("nan")
            feats["line_ppl_std"] = float("nan")
            feats["line_ppl_var"] = float("nan")

        full_text = flatten_text(lines)
        feats["page_ppl"] = float(self.lm_scorer.page_ppl(full_text))
        return feats

    # ---- Lexicality features ----

    def _lexicality_features(self, page: PageDocument) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        text = flatten_text(page.all_lines)
        words = [w for w in text.replace("\n", " ").split() if w]
        if not words:
            feats["lexicality_word_ratio"] = float("nan")
            return feats

        if self.lexicon is None:
            feats["lexicality_word_ratio"] = float("nan")
            return feats

        in_dict = sum(self.lexicon.contains(w.lower()) for w in words)
        feats["lexicality_word_ratio"] = in_dict / len(words)
        return feats

    # ---- Spatial-text interaction features ----

    def _spatial_text_interaction_features(self, page: PageDocument) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        if self.lm_scorer is None:
            feats["corr_line_conf_vs_ppl"] = float("nan")
            return feats

        lines = page.all_lines
        pairs = []
        for l in lines:
            if not l.full_text.strip():
                continue
            conf = l.text_result.best_score
            ppl = self.lm_scorer.line_ppl(l.full_text)
            if math.isnan(conf) or math.isnan(ppl):
                continue
            pairs.append((conf, ppl))

        if len(pairs) < 2:
            feats["corr_line_conf_vs_ppl"] = float("nan")
            return feats

        arr_conf = np.array([c for c, _ in pairs], dtype=float)
        arr_ppl = np.array([p for _, p in pairs], dtype=float)
        if arr_conf.std() == 0 or arr_ppl.std() == 0:
            feats["corr_line_conf_vs_ppl"] = float("nan")
        else:
            feats["corr_line_conf_vs_ppl"] = float(np.corrcoef(arr_conf, arr_ppl)[0, 1])
        return feats

    # ---- Metadata features ----

    def _metadata_features(self) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        century = self.metadata.get("century")
        if century is not None:
            feats["century"] = float(century)
        script_type = self.metadata.get("script_type")
        if script_type is not None:
            feats["script_type_code"] = float(hash(script_type) % 1000)
        return feats

    # ---- Main entry point ----

    def extract_features(self, page: PageDocument) -> Dict[str, float]:
        features: Dict[str, float] = {}
        features.update(self.image_feature_extractor.extract_all(page.image_path))
        features.update(self._region_line_segmentation_features(page))
        features.update(self._layout_features(page))
        features.update(self._htr_confidence_features(page))
        features.update(self._token_category_features(page))
        features.update(self._ngram_features(page))
        features.update(self._lm_perplexity_features(page))
        features.update(self._lexicality_features(page))
        features.update(self._spatial_text_interaction_features(page))
        features.update(self._metadata_features())
        return features


# ------------------------------
# Convenience loader
# ------------------------------

def load_page_document_from_file(path: str) -> PageDocument:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return PageDocument.from_json(data)
