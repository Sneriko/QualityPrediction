# Training XGBoost on the Swedish Lion 26 datasets

## Recommendation in brief

Train **two independent model families**. The line and region+line CSVs describe
different inference pipelines and do not have the same feature schema or target
distribution, so they should not be concatenated. Use identical target and
feature-set experiments where possible so that their validation metrics remain
comparable.

The files contain 1,727 line-pipeline pages (131 columns) and 1,679
region+line-pipeline pages (152 columns). All region+line pages occur in the
line file, while the line file has 48 additional pages. In particular,
`target_map50_line` has very different distributions in the existing CSVs: its
mean is 0.9616 for line and 0.0540 for region+line. Investigation found that
nested line polygons in the region+line HTRflow output are relative to their
region crops, while the evaluator previously treated them as page coordinates.
The parser now converts them to page coordinates. **Rebuild the region+line CSV
before training segmentation targets**; its existing mAP/IoU columns are stale.

## Targets and experiments

Start with these targets:

* `target_bow_f1` for a single overall transcription-quality score;
* `target_bow_precision` and `target_bow_recall` when distinguishing extra text
  from missing text matters;
* `target_perm_cer_htr_only` as an error-rate counterpart (note that it is not
  restricted to `[0, 1]` and has 8/9 missing labels in the two files);
* `target_map50_line` for segmentation quality, but investigate the anomalously
  low region+line labels first.

For every target, retain the constant baseline and compare:

1. `single_htr_line_score_mean` (calibration sanity check),
2. `confidence_only` (small operational baseline),
3. `full` (all available numeric prediction-time signals).

Do not request `ngram_only`, `image_only`, `dit_only`, or `lexical_only` for
these particular CSVs: their corresponding columns are absent, and the trainer
will reject an empty feature set. `json_model_only` and `full` currently select
the same columns in these files, so running both only duplicates work. The
current `full` set is appropriate; identifiers and `dataset_tag` are
non-numeric and are excluded automatically.

## First-pass commands

Run from the repository root after installing the package and its modeling
dependencies:

```bash
uv sync --extra modeling
```

### Exact all-target, all-feature runs

The following is the minimal recipe for one independently optimized model per
`target_*` column, using every usable feature in each CSV. Omitting `--targets`
makes the trainer discover all five target columns in the respective file.
`--feature-sets full` selects every numeric or Boolean non-target column and
automatically drops identifiers/text fields, all-missing columns, and constant
columns. Do not pass both CSVs to one command: that would concatenate two
different pipeline representations and schemas into one training table.

```bash
uv run qp-train-xgb \
  --train-csv data/eval_swedish_lion_26/line_sl26_refit_bins_dataset.csv \
  --feature-sets full \
  --model-dir models/sl26/line_all_features \
  --log-dir models/sl26/line_all_features/logs \
  --feature-analysis-dir models/sl26/line_all_features/feature_analysis \
  --val-size 0.20 \
  --n-trials-full 100 \
  --early-stopping-rounds 40

uv run qp-train-xgb \
  --train-csv data/eval_swedish_lion_26/region_line_sl26_refit_bins_dataset.csv \
  --feature-sets full \
  --model-dir models/sl26/region_line_all_features \
  --log-dir models/sl26/region_line_all_features/logs \
  --feature-analysis-dir models/sl26/region_line_all_features/feature_analysis \
  --val-size 0.20 \
  --n-trials-full 100 \
  --early-stopping-rounds 40
```

Each of the 100 trials samples tree count/depth, learning rate, row and column
subsampling, child weight, gamma, and L1/L2 regularization, and early stopping
uses the validation split. Increase `--n-trials-full` for a larger random
search. The model directory receives one Joblib model per target; the log
directory receives validation predictions, trial metrics, and summary metrics;
and the feature-analysis directory receives importance reports.

### Recommended baseline and feature-selection comparison

These more extensive commands use the built-in deterministic 80/20 development
split and modest search budgets. Separate output trees prevent one pipeline's
identically named model files from overwriting the other's.

```bash
COMMON_TARGETS=target_bow_f1,target_bow_precision,target_bow_recall,target_perm_cer_htr_only,target_map50_line
COMMON_SETS=single_htr_line_score_mean,confidence_only,full

qp-train-xgb \
  --train-csv data/eval_swedish_lion_26/line_sl26_refit_bins_dataset.csv \
  --model-dir models/sl26/line \
  --log-dir models/sl26/line/logs \
  --feature-analysis-dir models/sl26/line/feature_analysis \
  --targets "$COMMON_TARGETS" \
  --feature-sets "$COMMON_SETS" \
  --val-size 0.20 \
  --n-trials-full 100 \
  --n-trials-baseline 30 \
  --early-stopping-rounds 40 \
  --feature-selection \
  --fs-apply-to full \
  --fs-corr-prune \
  --fs-top-k 60

qp-train-xgb \
  --train-csv data/eval_swedish_lion_26/region_line_sl26_refit_bins_dataset.csv \
  --model-dir models/sl26/region_line \
  --log-dir models/sl26/region_line/logs \
  --feature-analysis-dir models/sl26/region_line/feature_analysis \
  --targets "$COMMON_TARGETS" \
  --feature-sets "$COMMON_SETS" \
  --val-size 0.20 \
  --n-trials-full 100 \
  --n-trials-baseline 30 \
  --early-stopping-rounds 40 \
  --feature-selection \
  --fs-apply-to full \
  --fs-corr-prune \
  --fs-top-k 60
```

Feature selection is applied only to `full`, preserving the three interpretable
baselines. The correlation pruning and permutation selection are learned from
the training/validation split rather than from the other pipeline's data.

## Weighting and model selection

The BoW labels are concentrated near high quality, especially in the line
dataset. If detecting rare bad pages is the operational goal, run a **second**
BoW-only experiment with `--weights --select-by wmae --fs-match-select-by` and
compare both ordinary MAE and weighted MAE. Keep it in a separate output
directory. Do not use this weighting recipe for CER or mAP: the implementation
intentionally applies it only to target names containing `bow`.

```bash
qp-train-xgb \
  --train-csv data/eval_swedish_lion_26/line_sl26_refit_bins_dataset.csv \
  --model-dir models/sl26/line_weighted_bow \
  --log-dir models/sl26/line_weighted_bow/logs \
  --feature-analysis-dir models/sl26/line_weighted_bow/feature_analysis \
  --targets target_bow_f1,target_bow_precision,target_bow_recall \
  --feature-sets confidence_only,full \
  --val-size 0.20 --n-trials-full 100 --n-trials-baseline 30 \
  --weights --select-by wmae \
  --feature-selection --fs-apply-to full --fs-corr-prune \
  --fs-top-k 60 --fs-match-select-by
```

Repeat the weighted command with the region+line CSV and a
`models/sl26/region_line_weighted_bow` output tree if low-quality region+line
pages are equally important.

## Validation and final training

The internal split is useful for iteration, but a single random page split is
not a final performance estimate. Prefer a held-out CSV from different volumes,
archives, or time periods and pass it with `--eval-csv`; do not split adjacent
pages from one volume across train and evaluation. Apply the **same held-out
page IDs** to both pipelines so their errors can be compared on paired pages.

When no independent corpus exists, create group-aware folds outside the current
CLI (group by volume/manuscript parsed from `source_page_id`), repeat HPO across
several folds, and report median MAE plus a bad-page metric such as MAE/recall
on the lowest-quality target decile. Never use the other pipeline's rows for
validation after training on a CSV containing the same `source_page_id`: those
are paired representations of the same pages, not independent observations.

Choose a model only if it improves on both the constant and confidence-only
baselines. Inspect the prediction CSVs and feature-selection reports in each
run directory. After fixing the design and hyperparameters, refit on all
development pages and evaluate exactly once on the untouched external set.
