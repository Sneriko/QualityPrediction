from __future__ import annotations

import argparse
from pathlib import Path

from quality_prediction.config.settings import MetadataDefaults, ResourcePaths
from quality_prediction.dataset.builder import DatasetSpec, build_dataset_multi


def main() -> None:
    ap = argparse.ArgumentParser(description="Build merged CSV dataset for quality prediction.")
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--bin-config", type=Path, required=True)
    ap.add_argument("--char-lm", type=Path, required=True)
    ap.add_argument("--ngram-sets", type=Path, required=True)
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


    args = ap.parse_args()

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
    )
