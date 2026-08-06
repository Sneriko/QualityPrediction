from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from quality_prediction.core.geometry import BBox
from quality_prediction.core.types import PageContent, TextLine as SimpleTextLine


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
        tr = obj.get("text_result", {}) or {}
        bbox_d = obj["bbox"]
        return cls(
            label=obj.get("label", ""),
            text_result=TextResult(tr.get("texts", []) or [], tr.get("scores", []) or []),
            token_scores=[(t, float(s)) for t, s in (obj.get("token_scores") or [])],
            words=[Word.from_json(w) for w in (obj.get("contains") or []) if isinstance(w, dict)],
            segmentation_label=obj.get("segmentation_label", "textline"),
            segmentation_confidence=obj.get("segmentation_confidence"),
            bbox=BBox(float(bbox_d["xmin"]), float(bbox_d["ymin"]), float(bbox_d["xmax"]), float(bbox_d["ymax"])),
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
        bbox_d = obj["bbox"]
        raw_lines = obj.get("contains", []) or []
        lines: List[TextLine] = []
        for tl in raw_lines:
            if not isinstance(tl, dict):
                continue
            if tl.get("segmentation_label", "textline") != "textline":
                continue
            tr = tl.get("text_result")
            if not isinstance(tr, dict):
                continue
            texts = tr.get("texts", []) or []
            if not texts or not any(str(t).strip() for t in texts):
                continue
            try:
                lines.append(TextLine.from_json(tl))
            except Exception:
                continue

        return cls(
            label=obj.get("label", ""),
            lines=lines,
            segmentation_label=obj.get("segmentation_label", "textregion"),
            segmentation_confidence=obj.get("segmentation_confidence"),
            bbox=BBox(float(bbox_d["xmin"]), float(bbox_d["ymin"]), float(bbox_d["xmax"]), float(bbox_d["ymax"])),
            polygon=obj.get("polygon", "") or "",
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
        raw_items = obj.get("contains", []) or []
        regions: List[TextRegion] = []

        def is_textline_like(item: Dict[str, Any]) -> bool:
            tr = item.get("text_result")
            if not isinstance(tr, dict):
                return False
            children = item.get("contains") or []
            has_child_textlines = any(isinstance(ch, dict) and ch.get("segmentation_label") == "textline" for ch in children)
            return not has_child_textlines

        for item in raw_items:
            if not isinstance(item, dict):
                continue
            seg_label = item.get("segmentation_label", "")

            # Flat style: top-level textlines
            if seg_label == "textline" or is_textline_like(item):
                try:
                    line_obj = TextLine.from_json(item)
                except Exception:
                    continue
                # wrap into an implicit region so downstream code stays stable
                regions.append(
                    TextRegion(
                        label=item.get("label", "implicit_region"),
                        lines=[line_obj],
                        segmentation_label="textregion",
                        segmentation_confidence=item.get("segmentation_confidence"),
                        bbox=line_obj.bbox,
                        polygon=item.get("polygon", "") or "",
                    )
                )
                continue

            # Normal region
            try:
                regions.append(TextRegion.from_json(item))
            except Exception:
                continue

        return cls(
            file_name=obj.get("file_name", ""),
            image_path=obj.get("image_path", ""),
            image_name=obj.get("image_name", ""),
            label=obj.get("label", ""),
            regions=regions,
        )

    @property
    def all_lines(self) -> List[TextLine]:
        out: List[TextLine] = []
        for r in self.regions:
            out.extend(r.lines)
        return out

    @property
    def page_bbox(self) -> Optional[BBox]:
        if not self.regions:
            return None
        xmin = min(r.bbox.xmin for r in self.regions)
        ymin = min(r.bbox.ymin for r in self.regions)
        xmax = max(r.bbox.xmax for r in self.regions)
        ymax = max(r.bbox.ymax for r in self.regions)
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
        out: List[SimpleTextLine] = []
        items = data.get("contains", []) or []

        for item in items:
            if not isinstance(item, dict):
                continue

            if item.get("segmentation_label") == "textline":
                maybe = self._parse_one_line(item)
                if maybe:
                    out.append(maybe)
                continue

            # region-like: parse children
            for tl in (item.get("contains") or []):
                if not isinstance(tl, dict):
                    continue
                if tl.get("segmentation_label") != "textline":
                    continue
                maybe = self._parse_one_line(tl)
                if maybe:
                    out.append(maybe)

        return out

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

        items = data.get("contains", []) or []
        for item in items:
            if not isinstance(item, dict):
                continue
            seg_label = item.get("segmentation_label", "")

            # Region detection: anything non-textline with a bbox
            bbox_d = item.get("bbox")
            if isinstance(bbox_d, dict) and seg_label != "textline":
                det = self._det_from(item, bbox_d)
                if det:
                    region_dets.append(det)

            # Flat textline detection
            if seg_label == "textline":
                lb = item.get("bbox")
                if isinstance(lb, dict):
                    det = self._det_from(item, lb)
                    if det:
                        line_dets.append(det)
                continue

            # Nested lines
            for tl in (item.get("contains") or []):
                if not isinstance(tl, dict):
                    continue
                if tl.get("segmentation_label") != "textline":
                    continue
                lb = tl.get("bbox")
                if not isinstance(lb, dict):
                    continue
                det = self._det_from(tl, lb)
                if det:
                    line_dets.append(det)

        return region_dets, line_dets

    def _det_from(self, obj: dict, bbox_d: dict) -> Optional[Detection]:
        try:
            bbox = BBox(float(bbox_d["xmin"]), float(bbox_d["ymin"]), float(bbox_d["xmax"]), float(bbox_d["ymax"]))
        except Exception:
            return None
        score = obj.get("segmentation_confidence")
        try:
            s = float(score) if score is not None and np.isfinite(score) else 1.0
        except Exception:
            s = 1.0

        poly = obj.get("polygon", "") or ""
        return Detection(bbox=bbox, score=s, polygon=poly)
