from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Iterable, List, Tuple

from xml.etree.ElementTree import ParseError

from ngram.ngrammodel import NgramModel

from features.features import (
    load_page_document_from_file,
    PageFeatureExtractor,
    ImageFeatureExtractor,
    NgramResource,
    Lexicon,
    NgramLMPerplexityScorer,
    ConfidenceBinFitter,          # NEW (from your features code)
    ConfidenceBinConfig,          # NEW
)

from metrics.cer_metrics import PageEvaluator


# ---------------------------
# Wiring up the feature extractor
# ---------------------------

def create_feature_extractor(bin_config: ConfidenceBinConfig) -> PageFeatureExtractor:
    """
    Construct a PageFeatureExtractor with:
      - image features
      - char n-gram LM perplexity
      - ratio_present n-gram sets
      - FROZEN confidence bin config (global)
    """
    ngram_model_path = Path("/home/coder/QualityPrediction/models/char5.pkl")
    ngram_model = NgramModel.load(ngram_model_path)

    lm_scorer = NgramLMPerplexityScorer(ngram_model, smoothing_k=1.0)

    ngram_sets_path = Path("/home/coder/QualityPrediction/data/ngramdata/ngram_sets/ngram_sets.pkl")
    with ngram_sets_path.open("rb") as f:
        ngram_sets = pickle.load(f)

    ngram_resource = NgramResource(ngram_sets=ngram_sets)

    lexicon = None
    img_extractor = ImageFeatureExtractor()

    return PageFeatureExtractor(
        image_feature_extractor=img_extractor,
        ngram_resource=ngram_resource,
        lm_scorer=lm_scorer,
        lexicon=lexicon,
        metadata={"century": 17, "script_type": "kurrent"},
        bin_config=bin_config,   # IMPORTANT
    )


# ---------------------------
# Bin fitting (global)
# ---------------------------

def _iter_pages_for_binfit(pred_paths: Iterable[str]):
    """
    Generator yielding PageDocument objects for bin fitting.
    Skips bad JSONs but keeps going.
    """
    for p in pred_paths:
        try:
            yield load_page_document_from_file(p)
        except json.JSONDecodeError:
            continue
        except Exception:
            continue


def fit_or_load_bins_global(
    all_pred_paths: List[str],
    bin_config_path: str,
    force_refit: bool = False,
) -> ConfidenceBinConfig:
    """
    Fit bins ONCE from the union of all datasets (recommended),
    or load from disk if present.
    """
    fitter = ConfidenceBinFitter()
    cfg_path = Path(bin_config_path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg_path.exists() and not force_refit:
        print(f"[bins] Loading existing bin config: {cfg_path}")
        return fitter.load(str(cfg_path))

    print(f"[bins] Fitting global bins from {len(all_pred_paths)} prediction JSONs ...")
    pages = _iter_pages_for_binfit(all_pred_paths)
    cfg = fitter.fit_from_pages(pages)
    fitter.save(cfg, str(cfg_path))
    print(f"[bins] Saved bin config to: {cfg_path}")
    return cfg


# ---------------------------
# Dataset builder (multi-source + global bins)
# ---------------------------

def build_dataset_multi(
    datasets: List[Tuple[str, str]],
    out_csv: str,
    lambda_ins: float = 1.0,
    bin_config_path: str = "/home/coder/QualityPrediction/models/confidence_bins_global.json",
    force_refit_bins: bool = False,
) -> None:
    """
    Build one merged CSV from multiple (gt_dir, pred_dir) dataset pairs.

    Workflow:
      1) PRE-SCAN all datasets -> materialize all matching (page_id, gt_path, pred_path)
      2) Fit confidence bins ONCE on *all pred JSONs* (or load frozen config)
      3) Process all datasets using the same frozen bins
    """
    # -------- 1) PRE-SCAN --------
    per_ds_pairs = []  # list of (dataset_tag, evaluator, pairs)
    all_pred_paths: List[str] = []

    for ds_idx, (gt_dir, pred_dir) in enumerate(datasets):
        dataset_tag = f"ds{ds_idx}"
        evaluator = PageEvaluator(gt_dir=gt_dir, pred_dir=pred_dir)

        pairs = list(evaluator.iter_page_pairs())  # triggers "Found N matching..."
        print(f"[pre-scan] {dataset_tag}: {len(pairs)} pairs | GT={gt_dir} | PRED={pred_dir}")

        per_ds_pairs.append((dataset_tag, evaluator, pairs))
        all_pred_paths.extend([pred_path for _, _, pred_path in pairs])

    if not all_pred_paths:
        print("No matching GT/PRED page pairs found across all datasets; nothing written.")
        return

    # -------- 2) FIT / LOAD GLOBAL BINS --------
    bin_cfg = fit_or_load_bins_global(
        all_pred_paths=all_pred_paths,
        bin_config_path=bin_config_path,
        force_refit=force_refit_bins,
    )

    # -------- 3) PROCESS --------
    extractor = create_feature_extractor(bin_cfg)
    rows: List[dict] = []

    for dataset_tag, evaluator, pairs in per_ds_pairs:
        print(f"\n=== Processing {dataset_tag} ({len(pairs)} pages) ===")

        for source_page_id, gt_path, pred_path in pairs:
            page_id = f"{dataset_tag}__{source_page_id}"

            # --- targets ---
            try:
                targets = evaluator.compute_page_metrics(
                    gt_path=gt_path,
                    pred_path=pred_path,
                    lambda_ins_strict=lambda_ins,
                    lambda_ins_split_tol=0.0,
                )
            except ParseError as e:
                print(f"[WARN] Skipping page '{page_id}' due to XML parse error: {e}")
                continue
            except Exception as e:
                print(f"[WARN] Skipping page '{page_id}' due to unexpected error in compute_page_metrics: {e}")
                continue

            # --- features ---
            try:
                page_doc = load_page_document_from_file(pred_path)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping page '{page_id}' due to JSON parse error: {e}")
                continue
            except Exception as e:
                print(f"[WARN] Skipping page '{page_id}' due to unexpected error loading JSON: {e}")
                continue

            try:
                feats = extractor.extract_features(page_doc)
            except Exception as e:
                print(f"[WARN] Skipping page '{page_id}' due to error in feature extraction: {e}")
                continue

            row = {
                "page_id": page_id,
                "source_page_id": source_page_id,
                "dataset_tag": dataset_tag,
            }
            row.update(feats)
            row.update(targets)
            rows.append(row)

    if not rows:
        print("No rows produced (everything skipped); nothing written.")
        return

    # -------- write CSV --------
    fieldnames = sorted({k for r in rows for k in r.keys()})
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nWrote {len(rows)} rows to {out_path}")
    print(f"[bins] Used global bin config: {bin_config_path}")


if __name__ == "__main__":
    DATASETS = [
        (
            "/home/coder/QualityPrediction/data/testsuite/images_page",
            "/home/coder/QualityPrediction/data/testsuite/htrflow_output_json",
        ),
        (
            "/home/coder/QualityPrediction/data/eval_from_training/page_no_duplicate_basenames",
            "/home/coder/QualityPrediction/data/eval_from_training/htrflow_out_json_no_duplicate_filenames/images_no_duplicate_basenames",
        ),
    ]

    out_csv = "/home/coder/QualityPrediction/data/testsuite/xgboost/training_set/xgboost_eval_test_map_targets.csv"

    build_dataset_multi(
        datasets=DATASETS,
        out_csv=out_csv,
        lambda_ins=1.0,
        bin_config_path="/home/coder/QualityPrediction/models/confidence_bins_global.json",
        force_refit_bins=False,   # set True if you want to refit bins deliberately
    )
