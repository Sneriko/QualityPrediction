# quality-prediction

Make a quality prediction dataset from images (paths specified in output jsons), htrflow output jsons and pagexmls ground truth

## Install (editable)
pip install -e .

## Build dataset
qp-build-dataset \
  --out-csv /path/out.csv \
  --bin-config /path/confidence_bins_global.json \
  --char-lm /path/char5.pkl \
  --ngram-sets /path/ngram_sets.pkl \
  --dataset ds0 /path/gt_xml_dir /path/pred_json_dir \
  --dataset ds1 /path/gt2 /path/pred2

## Evaluate
qp-evaluate --gt /path/gt_xml_dir --pred /path/pred_json_dir

## Training XGBoost models

Train one or more XGBoost regressors for columns named `target_*` in your dataset CSV.

### Train with internal train/val split
```bash
qp-train-xgb \
  --train-csv /path/to/xgboost_eval_test_map_targets.csv \
  --model-dir /home/coder/QualityPrediction/models \
  --log-dir /home/coder/QualityPrediction/data/runs_map_target/logs \
  --feature-analysis-dir /home/coder/QualityPrediction/data/runs_map_target/feature_analysis \
  --n-trials-full 100 \
  --n-trials-baseline 30 \
  --feature-sets single_htr_line_score_mean,confidence_only,ngram_only,full

