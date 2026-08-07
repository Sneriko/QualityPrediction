from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from quality_prediction.core.geometry import BBox
from quality_prediction.core.types import PageContent, TextLine as SimpleTextLine


def _segmentation_label(obj: Dict[str, Any]) -> str:
    """Return a normalized label from either supported HTRflow JSON schema."""
    annotations = obj.get("annotations") or {}
    return str(obj.get("segmentation_label") or annotations.get("segmentation_label") or "").lower()


def _segmentation_confidence(obj: Dict[str, Any]) -> Optional[float]:
    annotations = obj.get("annotations") or {}
    value = obj.get("segmentation_confidence", annotations.get("segmentation_confidence"))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _bbox(obj: Dict[str, Any]) -> BBox:
    raw = obj.get("bbox")
    if isinstance(raw, dict):
        return BBox(float(raw["xmin"]), float(raw["ymin"]), float(raw["xmax"]), float(raw["ymax"]))
    polygon = obj.get("polygon", "") or ""
    return BBox.from_page_coords(polygon)


def _text_result(obj: Dict[str, Any]) -> TextResult:
    raw = obj.get("text_result")
    if isinstance(raw, dict):
        return TextResult(raw.get("texts", []) or [], raw.get("scores", []) or [])
    transcriptions = obj.get("transcription") or []
    texts = [str(t.get("text", "")) for t in transcriptions if isinstance(t, dict)]
    scores = [t.get("confidence") for t in transcriptions if isinstance(t, dict) and t.get("confidence") is not None]
    return TextResult(texts, [float(s) for s in scores])


def _children(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [x for x in (obj.get("contains") or obj.get("regions") or []) if isinstance(x, dict)]


# ------------------------------
# Rich model for features
# ------------------------------

@dataclass
class TextResult:
    texts: List[str]
    scores: List[float]

    @property
    def best_text(self) -> str:
        return self.texts[0] if self.texts else ""

    @property
    def best_score(self) -> float:
        return float(self.scores[0]) if self.scores else float("nan")


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
        tr = obj.get("text_result", {}) or {}
        bbox_d = obj["bbox"]
        return cls(
            label=obj.get("label", ""),
            text_result=TextResult(tr.get("texts", []) or [], tr.get("scores", []) or []),
            segmentation_label=obj.get("segmentation_label", "word"),
            segmentation_confidence=obj.get("segmentation_confidence"),
            bbox=BBox(float(bbox_d["xmin"]), float(bbox_d["ymin"]), float(bbox_d["xmax"]), float(bbox_d["ymax"])),
            polygon=obj.get("polygon", "") or "",
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
        tr = _text_result(obj)
        legacy_tr = next((x for x in (obj.get("transcription") or []) if isinstance(x, dict)), {})
        return cls(
            label=obj.get("label", ""),
            text_result=tr,
            token_scores=[(t, float(s)) for t, s in (obj.get("token_scores") or legacy_tr.get("token_scores") or [])],
            words=[Word.from_json(w) for w in (obj.get("contains") or []) if isinstance(w, dict)],
            segmentation_label=_segmentation_label(obj) or "textline",
            segmentation_confidence=_segmentation_confidence(obj),
            bbox=_bbox(obj),
            polygon=obj.get("polygon", "") or "",
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
        raw_lines = _children(obj)
        lines: List[TextLine] = []
        for tl in raw_lines:
            if not isinstance(tl, dict):
                continue
            if _segmentation_label(tl) not in ("", "textline"):
                continue
            texts = _text_result(tl).texts
            if not texts or not any(str(t).strip() for t in texts):
                continue
            try:
                lines.append(TextLine.from_json(tl))
            except Exception:
                continue

        return cls(
            label=obj.get("label", ""),
            lines=lines,
            segmentation_label=_segmentation_label(obj) or "textregion",
            segmentation_confidence=_segmentation_confidence(obj),
            bbox=_bbox(obj),
            polygon=obj.get("polygon", "") or "",
        )


@dataclass
class PageDocument:
    file_name: str
    image_path: str
    image_name: str
    label: str
    regions: List[TextRegion]
    lines: List[TextLine] = field(default_factory=list)

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "PageDocument":
        raw_items = _children(obj)
        regions: List[TextRegion] = []
        lines: List[TextLine] = []

        def is_textline_like(item: Dict[str, Any]) -> bool:
            if not _text_result(item).texts:
                return False
            children = _children(item)
            has_child_textlines = any(_segmentation_label(ch) == "textline" for ch in children)
            return not has_child_textlines

        for item in raw_items:
            if not isinstance(item, dict):
                continue
            seg_label = _segmentation_label(item)

            # Flat style: top-level textlines. Keep these as page lines rather
            # than inventing regions that were not produced by the model.
            if seg_label == "textline" or is_textline_like(item):
                try:
                    lines.append(TextLine.from_json(item))
                except Exception:
                    continue
                continue

            # Normal region
            try:
                regions.append(TextRegion.from_json(item))
            except Exception:
                continue

        return cls(
            file_name=obj.get("file_name", obj.get("image_name", "")),
            image_path=obj.get("image_path", obj.get("_image_path", "")),
            image_name=obj.get("image_name", ""),
            label=obj.get("label", ""),
            regions=regions,
            lines=lines,
        )

    @property
    def all_lines(self) -> List[TextLine]:
        out: List[TextLine] = list(self.lines)
        for r in self.regions:
            out.extend(r.lines)
        return out

    @property
    def page_bbox(self) -> Optional[BBox]:
        objects = [*self.regions, *self.lines]
        if not objects:
            return None
        xmin = min(obj.bbox.xmin for obj in objects)
        ymin = min(obj.bbox.ymin for obj in objects)
        xmax = max(obj.bbox.xmax for obj in objects)
        ymax = max(obj.bbox.ymax for obj in objects)
        return BBox(xmin, ymin, xmax, ymax)


def load_page_document(path: str) -> PageDocument:
    with open(path, "r", encoding="utf-8") as f:
        return PageDocument.from_json(json.load(f))


# ------------------------------
# Lightweight parsers for metrics
# ------------------------------

class HtrJsonParser:
    """Extract predicted line texts + bboxes, preserving JSON order."""

    def parse(self, json_path: str) -> PageContent:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PageContent(lines=self._extract_lines(data))

    def _extract_lines(self, data: dict) -> List[SimpleTextLine]:
        page = PageDocument.from_json(data)
        return [SimpleTextLine(text=line.full_text, bbox=line.bbox) for line in page.all_lines]

    def _parse_one_line(self, tl: dict) -> Optional[SimpleTextLine]:
        tr = tl.get("text_result", {}) or {}
        texts = tr.get("texts") or []
        text = texts[0] if texts else ""

        bbox = None
        bbox_d = tl.get("bbox")
        if isinstance(bbox_d, dict):
            try:
                bbox = BBox(float(bbox_d["xmin"]), float(bbox_d["ymin"]), float(bbox_d["xmax"]), float(bbox_d["ymax"]))
            except Exception:
                bbox = None

        return SimpleTextLine(text=str(text), bbox=bbox)


@dataclass
class Detection:
    bbox: BBox
    score: float
    polygon: str = ""


class HtrJsonDetectionsParser:
    """Extract region + line detections for mAP targets (supports flat + nested JSON)."""

    def parse(self, json_path: str) -> Tuple[List[Detection], List[Detection]]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        region_dets: List[Detection] = []
        line_dets: List[Detection] = []

        def visit(item: Dict[str, Any]) -> None:
            label = _segmentation_label(item)
            if label in ("textline", "textregion"):
                det = self._det_from(item)
                if det:
                    (line_dets if label == "textline" else region_dets).append(det)
            for child in _children(item):
                visit(child)

        for item in _children(data):
            visit(item)

        return region_dets, line_dets

    def _det_from(self, obj: dict) -> Optional[Detection]:
        try:
            bbox = _bbox(obj)
        except Exception:
            return None
        score = _segmentation_confidence(obj)
        try:
            s = float(score) if score is not None and np.isfinite(score) else 1.0
        except Exception:
            s = 1.0

        poly = obj.get("polygon", "") or ""
        return Detection(bbox=bbox, score=s, polygon=poly)
