#!/usr/bin/env bash
set -euo pipefail

REPO="gt-transcriptions"
REF="main"
DST="/home/coder/QualityPrediction/data/gt-transcription"
FAIL_LOG="$DST/download_failures.log"

export LAKECTL_BASE_URI="lakefs://$REPO/$REF"
mkdir -p "$DST"
: > "$FAIL_LOG"

lakectl fs ls -r / \
  | awk '{print $NF}' \
  | grep -Ei '\.xml$' \
  | while IFS= read -r p; do
      [[ "$p" != /* ]] && p="/$p"
      rel="${p#/}"

      mkdir -p "$DST/$(dirname "$rel")"
      echo "Downloading: $p" >&2

      if ! lakectl fs download "$p" "$DST/$rel" --pre-sign=false; then
        echo "FAILED: $p" | tee -a "$FAIL_LOG" >&2
        continue
      fi
    done

echo "Done. Failures (if any) are in: $FAIL_LOG" >&2