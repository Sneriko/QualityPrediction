from __future__ import annotations

import os
from typing import Dict, Iterable, Tuple

import numpy as np
from xml.etree.ElementTree import ParseError

from quality_prediction.io.pagexml import PageXmlParser
from quality_prediction.io.htr_json import HtrJsonParser, HtrJsonDetectionsParser
from quality_prediction.metrics.cer import CerCalculator
from quality_prediction.metrics.map import Detection, ap_from_detections
from quality_prediction.metrics.seg_pr import SegObj, pr_f1_at_iou


TARGET_NAMES = (
    "target_perm_cer_strict",
    "target_perm_cer_split_tol",
    "target_perm_cer_split_penalty",
    "target_perm_cer_htr_only",
    "target_geom_order_avg_line_cer",
    "target_avg_line_cer",
    "target_seg_error",
    "target_ro_error",
    "target_delta_cer",
    "target_pi_missing_ratio",
    "target_pi_halluc_ratio",
    "target_avg_missing_ratio",
    "target_avg_halluc_ratio",
    "target_gt_num_lines",
    "target_pred_num_lines",
    "target_bow_precision",
    "target_bow_recall",
    "target_bow_f1",
    "target_map50_line",
    "target_map75_line",
    "target_map50_region",
    "target_map75_region",
    "target_iou50_line_precision",
    "target_iou50_line_recall",
    "target_iou50_line_f1",
    "target_iou75_line_precision",
    "target_iou75_line_recall",
    "target_iou75_line_f1",
    "target_soft_iou50_line_precision",
    "target_soft_iou50_line_recall",
    "target_soft_iou50_line_f1",
    "target_soft_iou75_line_precision",
    "target_soft_iou75_line_recall",
    "target_soft_iou75_line_f1",
)



class PageEvaluator:
    def __init__(self, gt_dir: str, pred_dir: str, xml_suffix: str = ".xml", json_suffix: str = ".json", log_path: str = ""):
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.xml_suffix = xml_suffix
        self.json_suffix = json_suffix
        self.log_path = log_path

        self.page_parser = PageXmlParser()
        self.pred_parser = HtrJsonParser()
        self.det_parser = HtrJsonDetectionsParser()

        if self.log_path and (not os.path.exists(self.log_path)):
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("# basename\ttargets...\n")

    def _index_files_by_basename(self, root_dir: str, suffix: str) -> dict:
        index = {}
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if not fname.endswith(suffix):
                    continue
                base = fname[: -len(suffix)]
                full_path = os.path.join(dirpath, fname)
                if base in index:
                    continue
                index[base] = full_path
        return index

    def iter_page_pairs(self) -> Iterable[Tuple[str, str, str]]:
        gt_index = self._index_files_by_basename(self.gt_dir, self.xml_suffix)
        pred_index = self._index_files_by_basename(self.pred_dir, self.json_suffix)
        for base in sorted(set(gt_index) & set(pred_index)):
            yield base, gt_index[base], pred_index[base]

    def _log_targets(self, basename: str, targets: Dict[str, float]) -> None:
        if not self.log_path:
            return
        parts = [basename] + [f"{k}={targets[k]}" for k in sorted(targets)]
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("\t".join(parts) + "\n")

    def compute_page_metrics(self, gt_path: str, pred_path: str, lambda_ins_strict: float = 1.0, lambda_ins_split_tol: float = 0.0) -> dict:
        gt_page = self.page_parser.parse(gt_path)
        pred_page = self.pred_parser.parse(pred_path)

        gt_line_geoms = self.page_parser.parse_line_geoms(gt_path)
        gt_line_objs = [
            SegObj(bbox=lg.bbox, polygon=lg.polygon)
            for lg in gt_line_geoms
            if lg.bbox is not None
        ]
        gt_line_boxes = [o.bbox for o in gt_line_objs]  # keep for existing mAP code
        gt_region_boxes = self.page_parser.parse_region_bboxes(gt_path)

        pred_region_raw, pred_line_raw = self.det_parser.parse(pred_path)

        pred_line_objs = [
            SegObj(bbox=d.bbox, polygon=getattr(d, "polygon", "") or "")
            for d in pred_line_raw
        ]
        pred_region = [Detection(d.bbox, d.score) for d in pred_region_raw]
        pred_line = [Detection(d.bbox, d.score) for d in pred_line_raw]

        p50, r50, f50 = pr_f1_at_iou(gt_line_objs, pred_line_objs, 0.50, soft=False)
        p75, r75, f75 = pr_f1_at_iou(gt_line_objs, pred_line_objs, 0.75, soft=False)

        sp50, sr50, sf50 = pr_f1_at_iou(gt_line_objs, pred_line_objs, 0.50, soft=True, soft_k=12.0)
        sp75, sr75, sf75 = pr_f1_at_iou(gt_line_objs, pred_line_objs, 0.75, soft=True, soft_k=15.0)

        map50_line = ap_from_detections(gt_line_boxes, pred_line, 0.50)
        map75_line = ap_from_detections(gt_line_boxes, pred_line, 0.75)
        map50_region = ap_from_detections(gt_region_boxes, pred_region, 0.50)
        map75_region = ap_from_detections(gt_region_boxes, pred_region, 0.75)

        gt_lines = gt_page.texts()
        pred_lines = pred_page.texts()

        perm_strict, pi_missing, pi_halluc, perm_htr_only = CerCalculator.page_cer_permutation_invariant_htr_only(
            gt_lines, pred_lines, lambda_ins=lambda_ins_strict
        )
        perm_split, _, _, _ = CerCalculator.page_cer_permutation_invariant_htr_only(gt_lines, pred_lines, lambda_ins=lambda_ins_split_tol)
        split_penalty = perm_strict - perm_split

        avg_line, avg_missing, avg_halluc = CerCalculator.page_cer_linewise_average(gt_lines, pred_lines)
        geom_avg, _, _ = CerCalculator.page_cer_linewise_average_geom(gt_page, pred_page)

        seg_error = geom_avg - perm_strict
        ro_error = avg_line - geom_avg
        delta = avg_line - perm_strict

        bow_p, bow_r, bow_f1 = CerCalculator.page_bow_metrics(gt_lines, pred_lines)

        targets = {
            "target_perm_cer_strict": perm_strict,
            "target_perm_cer_split_tol": perm_split,
            "target_perm_cer_split_penalty": split_penalty,
            "target_perm_cer_htr_only": perm_htr_only,
            "target_geom_order_avg_line_cer": geom_avg,
            "target_avg_line_cer": avg_line,
            "target_seg_error": seg_error,
            "target_ro_error": ro_error,
            "target_delta_cer": delta,
            "target_pi_missing_ratio": pi_missing,
            "target_pi_halluc_ratio": pi_halluc,
            "target_avg_missing_ratio": avg_missing,
            "target_avg_halluc_ratio": avg_halluc,
            "target_gt_num_lines": float(len(gt_lines)),
            "target_pred_num_lines": float(len(pred_lines)),
            "target_bow_precision": bow_p,
            "target_bow_recall": bow_r,
            "target_bow_f1": bow_f1,
            "target_map50_line": map50_line,
            "target_map75_line": map75_line,
            "target_map50_region": map50_region,
            "target_map75_region": map75_region,
            "target_iou50_line_precision": p50,
            "target_iou50_line_recall": r50,
            "target_iou50_line_f1": f50,
            "target_iou75_line_precision": p75,
            "target_iou75_line_recall": r75,
            "target_iou75_line_f1": f75,

            "target_soft_iou50_line_precision": sp50,
            "target_soft_iou50_line_recall": sr50,
            "target_soft_iou50_line_f1": sf50,
            "target_soft_iou75_line_precision": sp75,
            "target_soft_iou75_line_recall": sr75,
            "target_soft_iou75_line_f1": sf75,

        }

        # Keep the public target catalogue and computed metrics in sync.
        assert tuple(targets) == TARGET_NAMES

        self._log_targets(os.path.splitext(os.path.basename(gt_path))[0], targets)
        return targets

    def evaluate_all(self, lambda_ins: float = 1.0) -> None:
        vals = []
        for _, gt, pred in self.iter_page_pairs():
            try:
                vals.append(self.compute_page_metrics(gt, pred, lambda_ins_strict=lambda_ins, lambda_ins_split_tol=0.0))
            except ParseError:
                continue

        if not vals:
            print("No matching GT/PRED page pairs found.")
            return

        def mean(key: str) -> float:
            return float(np.mean([v[key] for v in vals]))

        print("Pages evaluated:", len(vals))
        print("Mean permCER_strict:", mean("target_perm_cer_strict"))
        print("Mean permCER_splitTol:", mean("target_perm_cer_split_tol"))
        print("Mean splitPenalty:", mean("target_perm_cer_split_penalty"))
        print("Mean geomAvgLineCER:", mean("target_geom_order_avg_line_cer"))
        print("Mean avgLineCER:", mean("target_avg_line_cer"))
        print("Mean seg_error:", mean("target_seg_error"))
        print("Mean ro_error:", mean("target_ro_error"))
        print("Mean deltaCER:", mean("target_delta_cer"))
        print("Mean BoW_F1:", mean("target_bow_f1"))
