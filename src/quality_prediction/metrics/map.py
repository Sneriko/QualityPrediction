from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from quality_prediction.core.geometry import BBox, iou


@dataclass
class Detection:
    bbox: BBox
    score: float


def ap_from_detections(gt_boxes: List[BBox], preds: List[Detection], iou_thr: float) -> float:
    if len(gt_boxes) == 0 and len(preds) == 0:
        return 1.0
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

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
