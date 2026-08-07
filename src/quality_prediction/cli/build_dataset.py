from __future__ import annotations

import argparse
from pathlib import Path

from quality_prediction.config.settings import MetadataDefaults, ResourcePaths
from quality_prediction.dataset.builder import DatasetSpec, build_dataset_multi
from quality_prediction.features.page import PageFeatureExtractor


def _csv_values(values):
    if not values:
        return None
    return [item.strip() for value in values for item in value.split(",") if item.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build merged CSV dataset for quality prediction.")
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--bin-config", type=Path, help="Optional fitted confidence-bin JSON.")
    ap.add_argument("--char-lm", type=Path, help="Required only for lm/interaction features.")
    ap.add_argument("--ngram-sets", type=Path, help="Required only for ngram features.")
    ap.add_argument("--lambda-ins", type=float, default=1.0)
    ap.add_argument("--force-refit-bins", action="store_true")
    ap.add_argument("--century", type=int, default=None)
    ap.add_argument("--script-type", type=str, default=None)
    ap.add_argument("--dataset", action="append", nargs=3, metavar=("TAG", "GT_DIR", "PRED_DIR"), required=True)
    ap.add_argument("--use-dit", action="store_true")
    ap.add_argument("--dit-model", type=str, default="microsoft/dit-base")
    ap.add_argument("--dit-pool", choices=["cls", "mean"], default="cls")
    ap.add_argument("--dit-fp16", action="store_true")
    ap.add_argument("--dit-pca", type=Path, default=None)
    ap.add_argument("--dit-prefix", type=str, default="dit_emb")
    ap.add_argument(
        "--feature", action="append", metavar="GROUP[,GROUP...]",
        help=f"Feature groups to include (repeatable): {', '.join(PageFeatureExtractor.FEATURE_GROUPS)}. Default: all.",
    )
    ap.add_argument(
        "--target", action="append", metavar="NAME[,NAME...]",
        help="Target columns to include (repeatable, e.g. target_perm_cer_strict,target_map50_line). Default: all.",
    )


    args = ap.parse_args()
    feature_groups = _csv_values(args.feature)
    targets = _csv_values(args.target)
    if feature_groups:
        unknown = set(feature_groups).difference(PageFeatureExtractor.FEATURE_GROUPS)
        if unknown:
            ap.error(f"unknown feature group(s): {', '.join(sorted(unknown))}")
    if feature_groups and any(x in feature_groups for x in ("lm", "interaction")) and args.char_lm is None:
        ap.error("--char-lm is required when selecting lm or interaction features")
    if feature_groups and "ngram" in feature_groups and args.ngram_sets is None:
        ap.error("--ngram-sets is required when selecting ngram features")

    datasets = [DatasetSpec(tag=t, gt_dir=Path(g), pred_dir=Path(p)) for t, g, p in args.dataset]
    resources = ResourcePaths(
        char_ngram_model=args.char_lm,
        ngram_sets=args.ngram_sets,
        global_bin_config=args.bin_config,
        use_dit=args.use_dit,
        dit_model_name=args.dit_model,
        dit_pool=args.dit_pool,
        dit_fp16=args.dit_fp16,
        dit_pca_path=args.dit_pca,
        dit_prefix=args.dit_prefix,
    )
    metadata = MetadataDefaults(century=args.century, script_type=args.script_type)

    build_dataset_multi(
        datasets=datasets,
        out_csv=args.out_csv,
        resources=resources,
        metadata=metadata,
        lambda_ins=args.lambda_ins,
        force_refit_bins=args.force_refit_bins,
        feature_groups=feature_groups,
        targets=targets,
    )
