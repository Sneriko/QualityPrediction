qp-train-xgb \
  --train-csv /home/coder/QualityPrediction/data/eval_from_training/xgboost/training_set/ngram_gt_swe_only_eval_from_training_1660_standard_pipeline_dit_emb_pca_v1.csv \
  --model-dir /home/coder/QualityPrediction/models/ngram_gt_swe_only_lex_eval_split_max_length_160_standard_pipeline_dit_emb_pca_fs \
  --log-dir /home/coder/QualityPrediction/models/ngram_gt_swe_only_lex_eval_split_max_length_160_standard_pipeline_dit_emb_pca_fs/logs \
  --feature-analysis-dir /home/coder/QualityPrediction/models/ngram_gt_swe_only_lex_eval_split_max_length_160_standard_pipeline_dit_emb_pca_fs/feature_analysis \
  --n-trials-full 100 \
  --n-trials-baseline 30 \
  --feature-selection \
  --feature-sets single_htr_line_score_mean,confidence_only,json_model_only,image_only,dit_only,ngram_only,lexical_only,full \
  --targets target_bow_f1,target_bow_precision,target_bow_recall,target_iou50_line_f1,target_iou50_line_precision,target_iou50_line_recall,target_map50_line,target_map50_region \
