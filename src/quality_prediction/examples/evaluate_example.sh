#!/usr/bin/env bash
set -euo pipefail

GT="/home/coder/QualityPrediction/data/outputs_htrflow/page"
PRED="/home/coder/QualityPrediction/data/outputs_htrflow/json"
LOG="/home/coder/QualityPrediction/data/testsuite/log/page_cer_targets.txt"

qp-evaluate \
  --gt "$GT" \
  --pred "$PRED" \
  --lambda-ins 1.0 \
  --log "$LOG"
