from __future__ import annotations

import csv
import json
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from xml.etree.ElementTree import ParseError

from quality_prediction.config.settings import MetadataDefaults, ResourcePaths
from quality_prediction.features.binning import ConfidenceBinConfig, ConfidenceBinFitter
from quality_prediction.features.factory import make_page_feature_extractor
from quality_prediction.io.htr_json import load_page_document
from quality_prediction.metrics.evaluator import PageEvaluator


@dataclass(frozen=True)
class DatasetSpec:
    tag: str
    gt_dir: Path
    pred_dir: Path


def _iter_pages_for_binfit(pred_paths: Iterable[Path]):
    for p in pred_paths:
        try:
            yield load_page_document(str(p))
        except json.JSONDecodeError:
            continue
        except Exception:
            continue


def fit_or_load_bins_global(all_pred_paths: List[Path], bin_config_path: Path, force_refit: bool = False) -> ConfidenceBinConfig:
    fitter = ConfidenceBinFitter()
    bin_config_path.parent.mkdir(parents=True, exist_ok=True)

    if bin_config_path.exists() and not force_refit:
        return fitter.load(str(bin_config_path))

    cfg = fitter.fit_from_pages(_iter_pages_for_binfit(all_pred_paths))
    fitter.save(cfg, str(bin_config_path))
    return cfg


def build_dataset_multi(
    datasets: List[DatasetSpec],
    out_csv: Path,
    resources: ResourcePaths,
    metadata: MetadataDefaults,
    lambda_ins: float = 1.0,
    force_refit_bins: bool = False,
) -> None:
    per_ds_pairs = []
    all_pred_paths: List[Path] = []

    for ds in datasets:
        evaluator = PageEvaluator(gt_dir=str(ds.gt_dir), pred_dir=str(ds.pred_dir))
        pairs = list(evaluator.iter_page_pairs())
        per_ds_pairs.append((ds, evaluator, pairs))
        all_pred_paths.extend([Path(pred_path) for _, _, pred_path in pairs])

    if not all_pred_paths:
        print("No matching GT/PRED page pairs found across datasets; nothing written.")
        return

    bin_cfg = fit_or_load_bins_global(all_pred_paths, resources.global_bin_config, force_refit=force_refit_bins)
    extractor = make_page_feature_extractor(resources, metadata, bin_cfg)

    rows: List[dict] = []
    for ds, evaluator, pairs in per_ds_pairs:
        for source_page_id, gt_path, pred_path in tqdm(
            pairs,
            desc=f"build {ds.tag}",
            unit="page",
            total=len(pairs),
        ):
            page_id = f"{ds.tag}__{source_page_id}"

            try:
                targets = evaluator.compute_page_metrics(
                    gt_path=gt_path,
                    pred_path=pred_path,
                    lambda_ins_strict=lambda_ins,
                    lambda_ins_split_tol=0.0,
                )
            except ParseError as e:
                print(f"[WARN] skip {page_id} XML parse error: {e}")
                continue
            except Exception as e:
                print(f"[WARN] skip {page_id} target error: {e}")
                continue

            try:
                page_doc = load_page_document(pred_path)
            except json.JSONDecodeError as e:
                print(f"[WARN] skip {page_id} JSON parse error: {e}")
                continue
            except Exception as e:
                print(f"[WARN] skip {page_id} JSON load error: {e}")
                continue

            try:
                feats = extractor.extract_features(page_doc)
            except Exception as e:
                print(f"[WARN] skip {page_id} feature error: {e}")
                continue

            row = {"page_id": page_id, "source_page_id": source_page_id, "dataset_tag": ds.tag}
            row.update(feats)
            row.update(targets)
            rows.append(row)

    if not rows:
        print("No rows produced; nothing written.")
        return

    fieldnames = sorted({k for r in rows for k in r.keys()})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")
    print(f"Used global bin config: {resources.global_bin_config}")
