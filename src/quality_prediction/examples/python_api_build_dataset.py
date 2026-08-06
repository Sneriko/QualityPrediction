from pathlib import Path

from quality_prediction.config import ResourcePaths, MetadataDefaults
from quality_prediction.dataset import DatasetSpec, build_dataset_multi


def main() -> None:
    datasets = [
        DatasetSpec(
            tag="ds1",
            gt_dir=Path("/home/coder/QualityPrediction/data/eval_from_training/gt_no_duplicate_filenames"),
            pred_dir=Path(
                "/home/coder/QualityPrediction/data/eval_from_training/htrflow_output_json_max_length_160/images_no_duplicate_basenames"
            ),
        ),
    ]

    resources = ResourcePaths(
        char_ngram_model=Path("/home/coder/QualityPrediction/models/char5.pkl"),
        ngram_sets=Path("/home/coder/QualityPrediction/data/ngramdata/ngram_sets/ngram_sets.pkl"),
        global_bin_config=Path("/home/coder/QualityPrediction/data/eval_from_training/xgboost/training_set/confidence_bins.json"),
        use_dit=True,
        dit_model_name="microsoft/dit-base",
        dit_pca_path=Path("/home/coder/QualityPrediction/models/pca_dit/dit_pca_128.pkl"),
        lexicon_manifest_json=Path("/home/coder/QualityPrediction/data/lexicons/lexicons_manifest.json"),
    )


    metadata = MetadataDefaults(century=17, script_type="handwriting")

    out_csv = Path(
        "/home/coder/QualityPrediction/data/eval_from_training/xgboost/training_set/ngram_gt_swe_only_eval_from_training_1660_standard_pipeline_dit_emb_pca_v1.csv"
    )

    build_dataset_multi(
        datasets=datasets,
        out_csv=out_csv,
        resources=resources,
        metadata=metadata,
        lambda_ins=1.0,
        force_refit_bins=False,  # set True if you want to deliberately refit bin edges
    )


if __name__ == "__main__":
    main()
