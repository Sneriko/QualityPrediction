
---

## `examples/build_dataset_example.sh`
This mirrors your current `__main__` dataset paths.

```bash
#!/usr/bin/env bash
set -euo pipefail

# --- Resources ---
CHAR_LM="/home/coder/QualityPrediction/models/char5.pkl"
NGRAM_SETS="/home/coder/QualityPrediction/data/ngramdata/ngram_sets/ngram_sets.pkl"
BIN_CONFIG="/home/coder/QualityPrediction/models/confidence_bins_global.json"

# --- Output ---
OUT_CSV="/home/coder/QualityPrediction/data/testsuite/xgboost/training_set/xgboost_eval_test_map_targets.csv"

# --- Datasets ---
GT0="/home/coder/QualityPrediction/data/testsuite/images_page"
PRED0="/home/coder/QualityPrediction/data/testsuite/htrflow_output_json"

GT1="/home/coder/QualityPrediction/data/eval_from_training/page_no_duplicate_basenames"
PRED1="/home/coder/QualityPrediction/data/eval_from_training/htrflow_out_json_no_duplicate_filenames/images_no_duplicate_basenames"

qp-build-dataset \
  --out-csv "$OUT_CSV" \
  --bin-config "$BIN_CONFIG" \
  --char-lm "$CHAR_LM" \
  --ngram-sets "$NGRAM_SETS" \
  --lambda-ins 1.0 \
  --century 17 \
  --script-type kurrent \
  --dataset ds0 "$GT0" "$PRED0" \
  --dataset ds1 "$GT1" "$PRED1"
