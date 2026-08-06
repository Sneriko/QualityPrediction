from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from quality_prediction.core.geometry import BBox, iou as bbox_iou


Point = Tuple[float, float]
Polygon = List[Point]


def parse_points(poly: str) -> Optional[Polygon]:
    """
    Parses PAGE-style points: "x,y x,y x,y ..."
    Also matches your JSON polygon format.
    """
    if not poly:
        return None
    s = poly.strip()
    if not s:
        return None

    pts: List[Point] = []
    for token in s.replace(";", " ").split():
        if "," not in token:
            continue
        xs, ys = token.split(",", 1)
        try:
            x = float(xs)
            y = float(ys)
        except Exception:
            return None
        pts.append((x, y))
    return pts if len(pts) >= 3 else None


def poly_iou(a: Polygon, b: Polygon) -> float:
    """
    Exact polygon IoU via shapely if available, else returns NaN (caller will bbox-fallback).
    """
    try:
        from shapely.geometry import Polygon as ShpPoly  # type: ignore
        pa = ShpPoly(a)
        pb = ShpPoly(b)
        if not pa.is_valid or not pb.is_valid:
            return 0.0
        inter = pa.intersection(pb).area
        union = pa.union(pb).area
        if union <= 0:
            return 0.0
        return float(inter / union)
    except Exception:
        return float("nan")


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class SegObj:
    bbox: BBox
    polygon: str = ""  # points string; if empty/unparseable -> bbox fallback


def seg_iou(a: SegObj, b: SegObj) -> float:
    pa = parse_points(a.polygon) if a.polygon else None
    pb = parse_points(b.polygon) if b.polygon else None
    if pa is not None and pb is not None:
        v = poly_iou(pa, pb)
        if np.isfinite(v):
            return float(v)
    return float(bbox_iou(a.bbox, b.bbox))


def greedy_match(gt: Sequence[SegObj], pred: Sequence[SegObj]) -> List[Tuple[int, int, float]]:
    """
    Greedy 1-1 matching by IoU descending. Returns (gt_idx, pred_idx, iou).
    """
    if not gt or not pred:
        return []

    cand: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            cand.append((seg_iou(g, p), gi, pi))

    cand.sort(key=lambda t: t[0], reverse=True)

    gt_used = [False] * len(gt)
    pr_used = [False] * len(pred)
    out: List[Tuple[int, int, float]] = []

    for v, gi, pi in cand:
        if gt_used[gi] or pr_used[pi]:
            continue
        gt_used[gi] = True
        pr_used[pi] = True
        out.append((gi, pi, float(v)))
    return out


def pr_f1_at_iou(
    gt: Sequence[SegObj],
    pred: Sequence[SegObj],
    thr: float,
    *,
    soft: bool = False,
    soft_k: float = 12.0,
) -> Tuple[float, float, float]:
    """
    Hard:
      TP = #matches with IoU >= thr
      P = TP / #pred, R = TP / #gt
    Soft:
      TP_mass = sum(sigmoid(k*(IoU-thr))) over matched pairs
      P = TP_mass / #pred, R = TP_mass / #gt
    """
    m, n = len(gt), len(pred)

    if m == 0 and n == 0:
        return 1.0, 1.0, 1.0
    if m == 0 and n > 0:
        return 0.0, 1.0, 0.0
    if m > 0 and n == 0:
        return 1.0, 0.0, 0.0

    matches = greedy_match(gt, pred)

    if soft:
        tp = sum(sigmoid(soft_k * (v - thr)) for _, _, v in matches)
    else:
        tp = sum(1.0 for _, _, v in matches if v >= thr)

    prec = float(tp / max(1, n))
    rec = float(tp / max(1, m))
    f1 = 0.0 if (prec + rec) == 0 else float(2 * prec * rec / (prec + rec))
    return prec, rec, f1
