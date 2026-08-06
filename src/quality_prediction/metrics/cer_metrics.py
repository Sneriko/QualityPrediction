from __future__ import annotations

import re
from collections import Counter
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterable, Dict

import numpy as np
from scipy.optimize import linear_sum_assignment  # pip install scipy
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
from dataclasses import dataclass

# ---------------------------
# Basic data structures
# ---------------------------

@dataclass
class BBox:
    """Simple bounding box: (x_min, y_min, x_max, y_max) in image coordinates."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @staticmethod
    def from_page_coords(points_str: str) -> "BBox":
        """
        PAGE XML stores coords as 'x1,y1 x2,y2 x3,y3 x4,y4'.
        We'll take min/max over all points.
        """
        pts = []
        for pair in points_str.strip().split():
            x_str, y_str = pair.split(',')
            pts.append((float(x_str), float(y_str)))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return BBox(min(xs), min(ys), max(xs), max(ys))

    def sort_key(self) -> Tuple[float, float]:
        """Key for reading order approximation: top-to-bottom, then left-to-right."""
        return (self.y_min, self.x_min)

@dataclass
class Detection:
    bbox: BBox
    score: float


def iou(a: BBox, b: BBox) -> float:
    inter_x1 = max(a.x_min, b.x_min)
    inter_y1 = max(a.y_min, b.y_min)
    inter_x2 = min(a.x_max, b.x_max)
    inter_y2 = min(a.y_max, b.y_max)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, (a.x_max - a.x_min)) * max(0.0, (a.y_max - a.y_min))
    area_b = max(0.0, (b.x_max - b.x_min)) * max(0.0, (b.y_max - b.y_min))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def ap_from_detections(gt_boxes: List[BBox], preds: List[Detection], iou_thr: float) -> float:
    """
    Single-class AP (per page).
    Uses standard precision-recall integration with monotonic interpolation.
    """
    if len(gt_boxes) == 0 and len(preds) == 0:
        return 1.0  # trivially perfect
    if len(gt_boxes) == 0 and len(preds) > 0:
        return 0.0
    if len(gt_boxes) > 0 and len(preds) == 0:
        return 0.0

    preds_sorted = sorted(preds, key=lambda d: d.score, reverse=True)
    gt_used = [False] * len(gt_boxes)

    tp = np.zeros(len(preds_sorted), dtype=float)
    fp = np.zeros(len(preds_sorted), dtype=float)

    for i, det in enumerate(preds_sorted):
        best_j = -1
        best_iou = 0.0
        for j, gt in enumerate(gt_boxes):
            if gt_used[j]:
                continue
            v = iou(det.bbox, gt)
            if v > best_iou:
                best_iou = v
                best_j = j

        if best_j >= 0 and best_iou >= iou_thr:
            tp[i] = 1.0
            gt_used[best_j] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    rec = tp_cum / max(1.0, float(len(gt_boxes)))
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # AP via precision envelope + trapezoidal integration
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])

    # Integrate where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap



@dataclass
class TextLine:
    text: str
    bbox: Optional[BBox] = None


@dataclass
class PageContent:
    """Container for all lines on a page."""
    lines: List[TextLine]

    def sorted_by_reading_order(self) -> "PageContent":
        """
        Optional helper if you ever want to sort by geometry.
        Not used in the CER metrics here.
        """
        if self.lines and all(line.bbox is not None for line in self.lines):
            sorted_lines = sorted(self.lines, key=lambda l: l.bbox.sort_key())
        else:
            sorted_lines = list(self.lines)
        return PageContent(lines=sorted_lines)

    def texts(self) -> List[str]:
        return [l.text for l in self.lines]


# ---------------------------
# PAGE XML loader (ground truth)
# ---------------------------

class PageXmlParser:
    """
    Loads GT text lines from PAGE XML or ALTO XML, keeping the XML order
    as the ground-truth reading order.
    """
    def parse_region_bboxes(self, xml_path: str) -> List[BBox]:
        try:
            tree = ET.parse(xml_path)
        except ParseError as e:
            raise ParseError(f"Failed to parse XML '{xml_path}': {e}")

        root = tree.getroot()
        tag_lower = root.tag.lower()

        if "alto" in tag_lower:
            # ALTO: use TextBlock bbox approximations if present
            boxes: List[BBox] = []
            for tb in root.findall(".//{*}TextBlock"):
                # Some ALTO files store HPOS/VPOS/WIDTH/HEIGHT on TextBlock
                try:
                    x = float(tb.get("HPOS"))
                    y = float(tb.get("VPOS"))
                    w = float(tb.get("WIDTH"))
                    h = float(tb.get("HEIGHT"))
                    boxes.append(BBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h))
                except (TypeError, ValueError):
                    continue
            return boxes

        # PAGE: TextRegion coords
        boxes: List[BBox] = []
        for tr in root.findall(".//{*}TextRegion"):
            coords_el = tr.find(".//{*}Coords")
            if coords_el is None:
                continue
            points_str = coords_el.get("points")
            if not points_str:
                continue
            boxes.append(BBox.from_page_coords(points_str))
        return boxes


    def parse(self, xml_path: str) -> PageContent:
        try:
            tree = ET.parse(xml_path)
        except ParseError as e:
            raise ParseError(f"Failed to parse XML '{xml_path}': {e}")

        root = tree.getroot()
        tag_lower = root.tag.lower()

        # Heuristic: detect ALTO vs PAGE
        if "alto" in tag_lower:
            return self._parse_alto(root)
        else:
            # default: treat as PAGE XML
            return self._parse_page_xml(root)

    # -------- PAGE XML --------

    def _parse_page_xml(self, root: ET.Element) -> PageContent:
        """
        PAGE XML:
          <TextLine>
            <TextEquiv conf="...">
              <Unicode> ... </Unicode>
            </TextEquiv>
            ...
          </TextLine>
        """
        text_lines: List[TextLine] = []

        for tl in root.findall(".//{*}TextLine"):
            text = self._extract_page_textline_text(tl)

            coords_el = tl.find(".//{*}Coords")
            bbox = None
            if coords_el is not None:
                points_str = coords_el.get("points")
                if points_str:
                    bbox = BBox.from_page_coords(points_str)

            text_lines.append(TextLine(text=text, bbox=bbox))

        return PageContent(lines=text_lines)

    def _extract_page_textline_text(self, tl: ET.Element) -> str:
        """
        Extract full line text from a PAGE <TextLine>.

        Strategy:
        1) Prefer TextEquiv directly under TextLine (line text).
        2) If none, concatenate Word-level Unicode texts.
        3) Fallback: any Unicode we can find.
        """
        # --- 1) Prefer TextEquiv directly under TextLine ---
        best_text = None
        best_conf = -1.0

        # Only direct children, NOT descendants
        for te in tl.findall("./{*}TextEquiv"):
            unicode_el = te.find("./{*}Unicode")
            if unicode_el is None or unicode_el.text is None:
                continue
            text = unicode_el.text
            conf_attr = te.get("conf")
            try:
                conf = float(conf_attr) if conf_attr is not None else 1.0
            except ValueError:
                conf = 1.0

            if conf > best_conf:
                best_conf = conf
                best_text = text

        if best_text is not None:
            return best_text.strip()

        # --- 2) Fallback: join Word-level Unicode texts ---
        word_texts = []
        for w in tl.findall(".//{*}Word"):
            te = w.findall("./{*}TextEquiv")
            if not te:
                continue
            # if multiple per word, pick first with text
            unicode_el = te[0].find("./{*}Unicode")
            if unicode_el is not None and unicode_el.text:
                word_texts.append(unicode_el.text)

        if word_texts:
            return " ".join(word_texts).strip()

        # --- 3) Last resort: any Unicode under tl ---
        unicode_el = tl.find(".//{*}Unicode")
        if unicode_el is not None and unicode_el.text is not None:
            return unicode_el.text.strip()

        return ""

    

    # -------- ALTO XML --------

    def _parse_alto(self, root: ET.Element) -> PageContent:
        """
        ALTO XML:
          <TextBlock>
            <TextLine>
              <String CONTENT="word" HPOS="..." VPOS="..." WIDTH="..." HEIGHT="..."/>
              ...
            </TextLine>
          </TextBlock>
        """
        text_lines: List[TextLine] = []

        for tl in root.findall(".//{*}TextLine"):
            string_elems = tl.findall(".//{*}String")
            if not string_elems:
                continue

            # Join CONTENT attributes into one line of text
            words = []
            for se in string_elems:
                content = se.get("CONTENT")
                if content:
                    words.append(content)
            line_text = " ".join(words).strip()
            if not line_text:
                continue

            # Approximate bbox from first and last String
            bbox = None
            try:
                first = string_elems[0]
                last = string_elems[-1]
                x1 = float(first.get("HPOS"))
                y1 = float(first.get("VPOS"))
                h = float(first.get("HEIGHT"))
                x2 = float(last.get("HPOS")) + float(last.get("WIDTH"))
                y2 = y1 + h
                bbox = BBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)
            except (TypeError, ValueError):
                bbox = None

            text_lines.append(TextLine(text=line_text, bbox=bbox))

        return PageContent(lines=text_lines)


# ---------------------------
# HTR JSON loader (predictions)
# ---------------------------

class HtrJsonDetectionsParser:
    """
    Extracts predicted region and line detections (bbox + score) from your HTR JSON.
    Scores used:
      - region: region.segmentation_confidence (fallback 1.0)
      - line: textline.segmentation_confidence (fallback 1.0)
    """

    def parse(self, json_path: str) -> Tuple[List[Detection], List[Detection]]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        region_dets: List[Detection] = []
        line_dets: List[Detection] = []

        regions = data.get("contains", []) or []
        for r in regions:
            # region bbox
            rb = r.get("bbox")
            if isinstance(rb, dict):
                try:
                    bbox_r = BBox(
                        x_min=float(rb["xmin"]),
                        y_min=float(rb["ymin"]),
                        x_max=float(rb["xmax"]),
                        y_max=float(rb["ymax"]),
                    )
                    score_r = r.get("segmentation_confidence")
                    score_r = float(score_r) if score_r is not None else 1.0
                    region_dets.append(Detection(bbox=bbox_r, score=score_r))
                except (KeyError, TypeError, ValueError):
                    pass

            # contained lines
            for tl in (r.get("contains") or []):
                if tl.get("segmentation_label") != "textline":
                    continue
                lb = tl.get("bbox")
                if not isinstance(lb, dict):
                    continue
                try:
                    bbox_l = BBox(
                        x_min=float(lb["xmin"]),
                        y_min=float(lb["ymin"]),
                        x_max=float(lb["xmax"]),
                        y_max=float(lb["ymax"]),
                    )
                    score_l = tl.get("segmentation_confidence")
                    score_l = float(score_l) if score_l is not None else 1.0
                    line_dets.append(Detection(bbox=bbox_l, score=score_l))
                except (KeyError, TypeError, ValueError):
                    continue

        return region_dets, line_dets


class HtrJsonParser:
    """
    Loads predicted text lines from HTR pipeline JSON.

    We keep the order as they appear in JSON (pipeline reading order).
    """

    def parse(self, json_path: str) -> PageContent:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        lines = self._extract_lines_from_json(data)
        return PageContent(lines=lines)

    def _extract_lines_from_json(self, data: dict) -> List[TextLine]:
        lines: List[TextLine] = []

        regions = data.get("contains", [])
        for region in regions:
            region_contains = region.get("contains", [])
            for tl in region_contains:
                if tl.get("segmentation_label") != "textline":
                    continue

                # Line text
                text_result = tl.get("text_result", {})
                texts = text_result.get("texts") or []
                text = texts[0] if texts else ""

                # Line bbox
                bbox_dict = tl.get("bbox")
                bbox = None
                if isinstance(bbox_dict, dict):
                    try:
                        bbox = BBox(
                            x_min=float(bbox_dict["xmin"]),
                            y_min=float(bbox_dict["ymin"]),
                            x_max=float(bbox_dict["xmax"]),
                            y_max=float(bbox_dict["ymax"]),
                        )
                    except KeyError:
                        bbox = None

                lines.append(TextLine(text=text, bbox=bbox))

        return lines


# ---------------------------
# CER utilities
# ---------------------------

class CerCalculator:
    """Utility class for CER and related page metrics."""

    @staticmethod
    def levenshtein(a: str, b: str) -> int:
        """Plain Levenshtein edit distance (char-level)."""
        if a == b:
            return 0
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la

        # DP table
        dp = [[0] * (lb + 1) for _ in range(la + 1)]
        for i in range(la + 1):
            dp[i][0] = i
        for j in range(lb + 1):
            dp[0][j] = j

        for i in range(1, la + 1):
            ca = a[i - 1]
            for j in range(1, lb + 1):
                cb = b[j - 1]
                cost = 0 if ca == cb else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,          # deletion
                    dp[i][j - 1] + 1,          # insertion
                    dp[i - 1][j - 1] + cost,   # substitution,
                )
        return dp[la][lb]

    @staticmethod
    def page_cer_permutation_invariant_htr_only(
        gt_lines: List[str],
        pred_lines: List[str],
        lambda_ins: float = 1.0,
    ) -> Tuple[float, float, float, float]:
        """
        Permutation-invariant page CER via Hungarian matching,
        PLUS an HTR-only CER that only looks at matched GT–pred line pairs.

        Returns:
            perm_cer              (includes deletions + insertions)
            missing_gt_ratio      (unmatched_gt_lines / max(1, len(gt_lines)))
            hallucinated_ratio    (unmatched_pred_lines / max(1, len(pred_lines)))
            perm_cer_htr_only     (CER on matched lines only: no penalty for missing/hallucinated lines)
        """
        m = len(gt_lines)
        n = len(pred_lines)
        K = max(m, n)

        if m == 0 and n == 0:
            # Nothing to compare
            return 0.0, 0.0, 0.0, float("nan")

        gt_lens = [len(s) for s in gt_lines]
        pred_lens = [len(s) for s in pred_lines]
        total_gt_chars = sum(gt_lens) if gt_lens else 1  # avoid div-by-zero

        BIG = 10_000_000
        cost = np.full((K, K), BIG, dtype=float)

        # 1) Real GT–pred pairs: cost = edit distance
        for i in range(m):
            for j in range(n):
                cost[i, j] = CerCalculator.levenshtein(gt_lines[i], pred_lines[j])

        # 2) Unmatched GT lines -> deletion, CER = 1 for that line
        for i in range(m):
            for j in range(n, K):  # dummy columns
                cost[i, j] = gt_lens[i]

        # 3) Unmatched pred lines -> insertion / hallucination
        for j in range(n):
            for i in range(m, K):  # dummy rows
                cost[i, j] = lambda_ins * pred_lens[j]

        # 4) Dummy–dummy pairs: cost 0
        for i in range(m, K):
            for j in range(n, K):
                cost[i, j] = 0.0

        row_ind, col_ind = linear_sum_assignment(cost)
        total_edit_cost = float(cost[row_ind, col_ind].sum())

        # Track which GT/pred lines were matched to real partners vs dummies
        matched_gt = [False] * m
        matched_pred = [False] * n

        for r, c in zip(row_ind, col_ind):
            if r < m and c < n:
                matched_gt[r] = True
                matched_pred[c] = True

        missing_gt = matched_gt.count(False)
        hallucinated_pred = matched_pred.count(False)

        missing_gt_ratio = missing_gt / max(1, m)
        hallucinated_ratio = hallucinated_pred / max(1, n)

        # Standard permCER (includes deletions + insertions)
        perm_cer = total_edit_cost / total_gt_chars

        # ---- HTR-only CER on matched lines ----
        htr_pairs = [(r, c) for r, c in zip(row_ind, col_ind) if r < m and c < n]

        if not htr_pairs:
            perm_cer_htr_only = float("nan")
        else:
            htr_total_cost = 0.0
            htr_total_gt_chars = 0
            for r, c in htr_pairs:
                htr_total_cost += float(cost[r, c])
                htr_total_gt_chars += gt_lens[r]

            if htr_total_gt_chars == 0:
                perm_cer_htr_only = float("nan")
            else:
                perm_cer_htr_only = htr_total_cost / float(htr_total_gt_chars)

        return perm_cer, missing_gt_ratio, hallucinated_ratio, perm_cer_htr_only

    @staticmethod
    def page_cer_permutation_invariant(
        gt_lines: List[str],
        pred_lines: List[str],
        lambda_ins: float = 1.0,
    ) -> Tuple[float, float, float]:
        """
        Permutation-invariant page CER via Hungarian matching.
        """
        m = len(gt_lines)
        n = len(pred_lines)
        K = max(m, n)

        if m == 0 and n == 0:
            return 0.0, 0.0, 0.0

        gt_lens = [len(s) for s in gt_lines]
        pred_lens = [len(s) for s in pred_lines]
        total_gt_chars = sum(gt_lens) if gt_lens else 1  # avoid div-by-zero

        BIG = 10_000_000
        cost = np.full((K, K), BIG, dtype=float)

        # 1) Real GT–pred pairs
        for i in range(m):
            for j in range(n):
                cost[i, j] = CerCalculator.levenshtein(gt_lines[i], pred_lines[j])

        # 2) Unmatched GT lines -> deletion, CER = 1 for that line
        for i in range(m):
            for j in range(n, K):  # dummy columns
                cost[i, j] = gt_lens[i]

        # 3) Unmatched pred lines -> insertion / hallucination
        for j in range(n):
            for i in range(m, K):  # dummy rows
                cost[i, j] = lambda_ins * pred_lens[j]

        # 4) Dummy–dummy pairs: cost 0
        for i in range(m, K):
            for j in range(n, K):
                cost[i, j] = 0.0

        row_ind, col_ind = linear_sum_assignment(cost)
        total_edit_cost = float(cost[row_ind, col_ind].sum())

        matched_gt = [False] * m
        matched_pred = [False] * n

        for r, c in zip(row_ind, col_ind):
            if r < m and c < n:
                matched_gt[r] = True
                matched_pred[c] = True

        missing_gt = matched_gt.count(False)
        hallucinated_pred = matched_pred.count(False)

        missing_gt_ratio = missing_gt / max(1, m)
        hallucinated_ratio = hallucinated_pred / max(1, n)

        perm_cer = total_edit_cost / total_gt_chars
        return perm_cer, missing_gt_ratio, hallucinated_ratio

    @staticmethod
    def line_cer(gt: str, pred: str) -> float:
        """
        Line-level CER between a single GT and predicted line.
        """
        gt_str = gt
        pred_str = pred

        if len(gt_str) == 0:
            if len(pred_str) == 0:
                return 0.0
            else:
                return 1.0

        dist = CerCalculator.levenshtein(gt_str, pred_str)
        return dist / max(1, len(gt_str))

    @staticmethod
    def page_cer_linewise_average_geom(
        gt_page: "PageContent",
        pred_page: "PageContent",
    ) -> Tuple[float, float, float]:
        """
        Reading-order-sensitive average line-level CER,
        after sorting BOTH GT and PRED by the same geometric reading order
        (y_min, then x_min).

        This reuses page_cer_linewise_average on the sorted text lists.
        """
        gt_sorted = gt_page.sorted_by_reading_order()
        pred_sorted = pred_page.sorted_by_reading_order()

        gt_lines = gt_sorted.texts()
        pred_lines = pred_sorted.texts()

        return CerCalculator.page_cer_linewise_average(gt_lines, pred_lines)

    @staticmethod
    def page_cer_linewise_average(
        gt_lines: List[str],
        pred_lines: List[str],
    ) -> Tuple[float, float, float]:
        """
        Reading-order-sensitive average line-level CER.
        """
        m = len(gt_lines)
        n = len(pred_lines)

        if m == 0 and n == 0:
            return 0.0, 0.0, 0.0

        line_cers: List[float] = []
        for i in range(m):
            gt = gt_lines[i]
            if i < n:
                pred = pred_lines[i]
                cer_i = CerCalculator.line_cer(gt, pred)
            else:
                cer_i = 1.0
            line_cers.append(cer_i)

        avg_line_cer = sum(line_cers) / max(1, len(line_cers))

        missing_gt = max(0, m - n)
        hallucinated_pred = max(0, n - m)

        missing_gt_ratio = missing_gt / max(1, m)
        hallucinated_ratio = hallucinated_pred / max(1, n)

        return avg_line_cer, missing_gt_ratio, hallucinated_ratio

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Simple Unicode-aware tokenizer: lowercase, take word characters.
        You can later swap this for a Swedish-specific tokenizer/stemmer.
        """
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    @staticmethod
    def page_bow_metrics(
        gt_lines: List[str],
        pred_lines: List[str],
    ) -> Tuple[float, float, float]:
        """
        Page-level bag-of-words precision / recall / F1.

        - Concatenates all GT lines and all PRED lines.
        - Tokenizes into words (lowercased, word characters).
        - Builds multisets (Counters).
        - matched = sum_t min(count_gt[t], count_pred[t])

        Returns:
            bow_precision, bow_recall, bow_f1
        """
        gt_text = " ".join(gt_lines)
        pred_text = " ".join(pred_lines)

        gt_tokens = CerCalculator._tokenize(gt_text)
        pred_tokens = CerCalculator._tokenize(pred_text)

        if not gt_tokens and not pred_tokens:
            return 1.0, 1.0, 1.0  # trivial perfect

        gt_counts = Counter(gt_tokens)
        pred_counts = Counter(pred_tokens)

        total_gt = sum(gt_counts.values())
        total_pred = sum(pred_counts.values())

        if total_gt == 0 and total_pred > 0:
            # no GT words, but predictions exist -> all hallucinations
            return 0.0, 1.0, 0.0
        if total_pred == 0 and total_gt > 0:
            # no predictions at all
            return 0.0, 0.0, 0.0

        matched = 0
        for tok in set(gt_counts.keys()) | set(pred_counts.keys()):
            matched += min(gt_counts.get(tok, 0), pred_counts.get(tok, 0))

        bow_recall = matched / max(1, total_gt)
        bow_precision = matched / max(1, total_pred)

        if bow_precision + bow_recall == 0:
            bow_f1 = 0.0
        else:
            bow_f1 = 2 * bow_precision * bow_recall / (bow_precision + bow_recall)

        return bow_precision, bow_recall, bow_f1


# ---------------------------
# Orchestrator: evaluation runner
# ---------------------------

class PageEvaluator:
    """
    Orchestrates:
      - loading GT PAGE XML
      - loading HTR JSON
      - computing:
        * permutation-invariant page CER (strict)
        * permutation-invariant page CER (split-tolerant)
        * HTR-only permCER
        * geometric and reading-order average line-level CER
        * BoW precision/recall/F1
        * various derived errors
      - logging all targets per page to a text file
    """

    def __init__(
        self,
        gt_dir: str,
        pred_dir: str,
        xml_suffix: str = ".xml",
        json_suffix: str = ".json",
        log_path: str = "/home/coder/QualityPrediction/data/testsuite/log/page_cer_targets.txt",
    ):
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.xml_suffix = xml_suffix
        self.json_suffix = json_suffix
        self.log_path = log_path

        self.page_parser = PageXmlParser()
        self.pred_parser = HtrJsonParser()
        self.det_parser = HtrJsonDetectionsParser()


        # If the log file doesn't exist yet, write a header line
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("# basename\ttargets...\n")

    def _index_files_by_basename(self, root_dir: str, suffix: str) -> dict:
        """
        Walk root_dir recursively and build a mapping:
            basename_without_suffix -> full_path

        If the same basename appears multiple times, the first one wins
        and the others are ignored with a warning.
        """
        index = {}
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if not fname.endswith(suffix):
                    continue
                base = fname[: -len(suffix)]
                full_path = os.path.join(dirpath, fname)
                if base in index:
                    print(
                        f"Warning: duplicate basename '{base}' in '{root_dir}'. "
                        f"Keeping '{index[base]}' and ignoring '{full_path}'."
                    )
                    continue
                index[base] = full_path
        return index

    def iter_page_pairs(self) -> Iterable[Tuple[str, str, str]]:
        """
        Yield triples: (page_id, gt_xml_path, pred_json_path)
        for all pages where both GT and prediction exist, matched by basename,
        regardless of directory depth.
        """
        gt_index = self._index_files_by_basename(self.gt_dir, self.xml_suffix)
        pred_index = self._index_files_by_basename(self.pred_dir, self.json_suffix)

        common_bases = sorted(set(gt_index.keys()) & set(pred_index.keys()))
        if not common_bases:
            print("No matching basenames between GT and PRED dirs.")
        else:
            print(f"Found {len(common_bases)} matching page basenames.")

        for base in common_bases:
            yield base, gt_index[base], pred_index[base]

    def _log_targets(self, basename: str, targets: Dict[str, float]) -> None:
        """
        Append one nicely formatted line to the log file:
        basename<TAB>key1=val1<TAB>key2=val2...
        """
        parts = [basename]
        # Sort keys for stable column order
        for k in sorted(targets.keys()):
            parts.append(f"{k}={targets[k]}")
        line = "\t".join(parts)

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def compute_page_metrics(
        self,
        gt_path: str,
        pred_path: str,
        lambda_ins_strict: float = 1.0,
        lambda_ins_split_tol: float = 0.0,
    ) -> dict:
        """
        Compute all CER-based targets for a single page pair and
        return them as a flat dict suitable to be merged with features.

        Includes:
          - target_perm_cer_strict
          - target_perm_cer_split_tol (no penalty for extra predicted lines)
          - target_perm_cer_split_penalty = strict - split_tol
          - target_perm_cer_htr_only (from strict assignment)
          - target_geom_order_avg_line_cer
          - target_avg_line_cer
          - target_seg_error  = geomAvgLineCER - permCER_strict
          - target_ro_error   = avgLineCER - geomAvgLineCER
          - target_delta_cer  = avgLineCER - permCER_strict
          - target_pi_missing_ratio
          - target_pi_halluc_ratio
          - target_avg_missing_ratio
          - target_avg_halluc_ratio
          - target_gt_num_lines
          - target_pred_num_lines
          - target_bow_precision
          - target_bow_recall
          - target_bow_f1
        """
        gt_page = self.page_parser.parse(gt_path)      # PAGE-XML order
        pred_page = self.pred_parser.parse(pred_path)  # JSON order
        # ---- Detection mAP targets (NEW) ----
        gt_line_boxes = [l.bbox for l in gt_page.lines if l.bbox is not None]
        gt_region_boxes = self.page_parser.parse_region_bboxes(gt_path)

        pred_region_dets, pred_line_dets = self.det_parser.parse(pred_path)

        map50_line = ap_from_detections(gt_line_boxes, pred_line_dets, iou_thr=0.50)
        map75_line = ap_from_detections(gt_line_boxes, pred_line_dets, iou_thr=0.75)
        map50_region = ap_from_detections(gt_region_boxes, pred_region_dets, iou_thr=0.50)
        map75_region = ap_from_detections(gt_region_boxes, pred_region_dets, iou_thr=0.75)


        gt_lines = gt_page.texts()
        pred_lines = pred_page.texts()

        # Strict permutation-invariant CER (penalizes splits/hallucinations)
        perm_cer_strict, pi_missing, pi_halluc, perm_cer_htr_only = (
            CerCalculator.page_cer_permutation_invariant_htr_only(
                gt_lines,
                pred_lines,
                lambda_ins=lambda_ins_strict,
            )
        )

        # Split-tolerant permutation-invariant CER (no penalty for extra preds)
        perm_cer_split_tol, _, _, _ = (
            CerCalculator.page_cer_permutation_invariant_htr_only(
                gt_lines,
                pred_lines,
                lambda_ins=lambda_ins_split_tol,
            )
        )

        perm_cer_split_penalty = perm_cer_strict - perm_cer_split_tol

        # Reading-order average line-level CER
        avg_line_cer, avg_missing, avg_halluc = CerCalculator.page_cer_linewise_average(
            gt_lines,
            pred_lines,
        )

        # Geometric-order average line-level CER:
        geom_avg_line_cer, _, _ = CerCalculator.page_cer_linewise_average_geom(
            gt_page,
            pred_page,
        )

        # Derived error terms (based on strict permCER)
        seg_error = geom_avg_line_cer - perm_cer_strict   # segmentation / alignment cost
        ro_error = avg_line_cer - geom_avg_line_cer       # extra cost from pipeline reading order
        delta_cer = avg_line_cer - perm_cer_strict        # total gap between best-case and pipeline order

        # --- Bag-of-words metrics (order and segmentation invariant) ---
        bow_prec, bow_rec, bow_f1 = CerCalculator.page_bow_metrics(
            gt_lines,
            pred_lines,
        )

        targets = {
            "target_perm_cer_strict": perm_cer_strict,
            "target_perm_cer_split_tol": perm_cer_split_tol,
            "target_perm_cer_split_penalty": perm_cer_split_penalty,
            "target_perm_cer_htr_only": perm_cer_htr_only,
            "target_geom_order_avg_line_cer": geom_avg_line_cer,
            "target_avg_line_cer": avg_line_cer,
            "target_seg_error": seg_error,
            "target_ro_error": ro_error,
            "target_delta_cer": delta_cer,
            "target_pi_missing_ratio": pi_missing,
            "target_pi_halluc_ratio": pi_halluc,
            "target_avg_missing_ratio": avg_missing,
            "target_avg_halluc_ratio": avg_halluc,
            "target_gt_num_lines": len(gt_lines),
            "target_pred_num_lines": len(pred_lines),
            "target_bow_precision": bow_prec,
            "target_bow_recall": bow_rec,
            "target_bow_f1": bow_f1,
            "target_map50_line": map50_line,
            "target_map75_line": map75_line,
            "target_map50_region": map50_region,
            "target_map75_region": map75_region,
        }

        # log nicely formatted line to text file
        basename = os.path.splitext(os.path.basename(gt_path))[0]
        self._log_targets(basename, targets)

        return targets

    def evaluate_all(self, lambda_ins: float = 1.0) -> None:
        """
        Run evaluation over all page pairs and print stats.

        - permCER_strict: permutation-invariant, penalizing extra pred lines
        - permCER_split_tol: permutation-invariant, tolerant to splits (lambda_ins=0)
        - permCER_split_penalty: difference (strict - split_tol)
        - geomAvgLineCER: average line CER after shared geometric ordering
        - avgLineCER: reading-order-sensitive average line-level CER
        - seg_error: geomAvgLineCER - permCER_strict
        - ro_error:  avgLineCER - geomAvgLineCER
        - BoW precision/recall/F1
        """
        perm_cers_strict = []
        perm_cers_split_tol = []
        perm_cers_split_penalty = []
        geom_avg_line_cers = []
        avg_line_cers = []
        seg_errors = []
        ro_errors = []
        delta_cers = []

        pi_missing_vals = []
        pi_halluc_vals = []
        avg_missing_vals = []
        avg_halluc_vals = []

        bow_prec_vals = []
        bow_rec_vals = []
        bow_f1_vals = []

        for page_id, gt_path, pred_path in self.iter_page_pairs():
            try:
                t = self.compute_page_metrics(
                    gt_path=gt_path,
                    pred_path=pred_path,
                    lambda_ins_strict=lambda_ins,
                    lambda_ins_split_tol=0.0,
                )
            except ParseError as e:
                print(f"[WARN] Skipping page '{page_id}' due to XML parse error: {e}")
                continue

            perm_strict = t["target_perm_cer_strict"]
            perm_split_tol = t["target_perm_cer_split_tol"]
            perm_split_penalty = t["target_perm_cer_split_penalty"]
            geom_avg_line_cer = t["target_geom_order_avg_line_cer"]
            avg_line_cer = t["target_avg_line_cer"]
            seg_error = t["target_seg_error"]
            ro_error = t["target_ro_error"]
            delta_cer = t["target_delta_cer"]

            pi_missing = t["target_pi_missing_ratio"]
            pi_halluc = t["target_pi_halluc_ratio"]
            avg_missing = t["target_avg_missing_ratio"]
            avg_halluc = t["target_avg_halluc_ratio"]

            bow_prec = t["target_bow_precision"]
            bow_rec = t["target_bow_recall"]
            bow_f1 = t["target_bow_f1"]

            perm_cers_strict.append(perm_strict)
            perm_cers_split_tol.append(perm_split_tol)
            perm_cers_split_penalty.append(perm_split_penalty)
            geom_avg_line_cers.append(geom_avg_line_cer)
            avg_line_cers.append(avg_line_cer)
            seg_errors.append(seg_error)
            ro_errors.append(ro_error)
            delta_cers.append(delta_cer)

            pi_missing_vals.append(pi_missing)
            pi_halluc_vals.append(pi_halluc)
            avg_missing_vals.append(avg_missing)
            avg_halluc_vals.append(avg_halluc)

            bow_prec_vals.append(bow_prec)
            bow_rec_vals.append(bow_rec)
            bow_f1_vals.append(bow_f1)

            print(
                f"{page_id} | "
                f"permCER_strict={perm_strict:.3f} | "
                f"permCER_splitTol={perm_split_tol:.3f} | "
                f"splitPenalty={perm_split_penalty:.3f} | "
                f"geomAvgLineCER={geom_avg_line_cer:.3f} | "
                f"avgLineCER={avg_line_cer:.3f} | "
                f"seg_error(geo-perm)={seg_error:.3f} | "
                f"ro_error(avg-geo)={ro_error:.3f} | "
                f"deltaCER(avg-perm)={delta_cer:.3f} | "
                f"PI_missing={pi_missing:.3f} PI_halluc={pi_halluc:.3f} | "
                f"AVG_missing={avg_missing:.3f} AVG_halluc={avg_halluc:.3f} | "
                f"BoW_P={bow_prec:.3f} BoW_R={bow_rec:.3f} BoW_F1={bow_f1:.3f}"
            )

        if perm_cers_strict:
            print("\n=== Aggregate stats ===")
            print(f"Pages evaluated:                  {len(perm_cers_strict)}")
            print(f"Mean permCER_strict:              {np.mean(perm_cers_strict):.3f}")
            print(f"Mean permCER_splitTol:            {np.mean(perm_cers_split_tol):.3f}")
            print(f"Mean permCER_splitPenalty:        {np.mean(perm_cers_split_penalty):.3f}")
            print(f"Mean geomAvgLineCER:              {np.mean(geom_avg_line_cers):.3f}")
            print(f"Mean avgLineCER:                  {np.mean(avg_line_cers):.3f}")
            print(f"Mean seg_error(geo-perm):         {np.mean(seg_errors):.3f}")
            print(f"Mean ro_error(avg-geo):           {np.mean(ro_errors):.3f}")
            print(f"Mean deltaCER(avg-perm):          {np.mean(delta_cers):.3f}")
            print(f"Mean PI_missing:                  {np.mean(pi_missing_vals):.3f}")
            print(f"Mean PI_halluc:                   {np.mean(pi_halluc_vals):.3f}")
            print(f"Mean AVG_missing:                 {np.mean(avg_missing_vals):.3f}")
            print(f"Mean AVG_halluc:                  {np.mean(avg_halluc_vals):.3f}")
            print(f"Mean BoW_precision:               {np.mean(bow_prec_vals):.3f}")
            print(f"Mean BoW_recall:                  {np.mean(bow_rec_vals):.3f}")
            print(f"Mean BoW_F1:                      {np.mean(bow_f1_vals):.3f}")
        else:
            print("No matching GT/PRED page pairs found.")


if __name__ == "__main__":
    # Adjust these to your actual folders
    gt_dir = "/home/coder/QualityPrediction/data/outputs_htrflow/page"
    pred_dir = "/home/coder/QualityPrediction/data/outputs_htrflow/json"

    evaluator = PageEvaluator(gt_dir=gt_dir, pred_dir=pred_dir)
    evaluator.evaluate_all(lambda_ins=1.0)
