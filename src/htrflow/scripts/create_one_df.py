from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd

# Import the per-file builder from your existing script
# Make sure this file sits next to `htr_json_to_df_with_gt_matching.py` OR that directory is on PYTHONPATH
try:
    from create_cer_pred_train_set import htr_json_to_df_with_gt
except Exception as e:  # pragma: no cover
    print("ERROR: Could not import htr_json_to_df_with_gt_matching.htr_json_to_df_with_gt\n" \
          "Make sure this script is in the same folder as 'htr_json_to_df_with_gt_matching.py'\n" \
          f"Original import error: {e}", file=sys.stderr)
    raise


def find_jsons(root: Path, pattern: str) -> List[Path]:
    """Recursively find JSON files under root matching a glob pattern (e.g., '**/*.json')."""
    if "**" in pattern:
        return sorted(root.glob(pattern))
    # Always recurse into subdirectories, even if user gives '*.json'
    return sorted(root.rglob(pattern))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch: walk a directory, parse all HTR JSONs, and build ONE combined DataFrame matched to an HF parquet dataset."
    )
    p.add_argument("json_root", type=str, help="Directory containing HTR output JSONs (searched recursively)")
    p.add_argument("hf_source", type=str, help="HF dataset id OR local dir/file of parquet shards with GT")
    p.add_argument("--pattern", default="**/*.json", help="Glob for JSONs (default '**/*.json')")

    # Matching options (forwarded to the single-file function)
    p.add_argument("--gt-text-col", default="text", help="GT column with full-line transcription (e.g., 'transcription')")
    p.add_argument("--hf-split", default=None, help="Split to use when loading a hub dataset (e.g., 'test')")
    p.add_argument("--keep-diacritics", action="store_true", help="Keep diacritics when normalizing (default)")
    p.add_argument("--strip-diacritics", dest="keep_diacritics", action="store_false", help="Strip diacritics when normalizing")
    p.add_argument("--normalize-gt", dest="normalize_gt", action="store_true", help="Normalize GT text (default)")
    p.add_argument("--no-normalize-gt", dest="normalize_gt", action="store_false", help="Do not normalize GT text")
    p.add_argument("--scorer", default="ratio", choices=["ratio", "token_sort_ratio", "token_set_ratio", "WRatio"], help="Similarity scorer")
    p.add_argument("--min-match-score", type=float, default=0.0, help="Minimum score [0..100] to accept a match")

    # Output controls
    p.add_argument("--out", type=str, default=None, help="Output path (.csv or .parquet). Default: '<json_root_basename>_combined_lines_with_gt.csv'")
    p.add_argument("--parquet-snappy", action="store_true", help="If saving parquet, use snappy compression (default)" )

    p.set_defaults(keep_diacritics=True, normalize_gt=True)

    args = p.parse_args()

    root = Path(args.json_root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    json_files = find_jsons(root, args.pattern)
    if not json_files:
        print(f"No JSONs found under {root} with pattern {args.pattern}")
        return

    print(f"Found {len(json_files)} JSON files. Building combined DataFrame…")

    frames: List[pd.DataFrame] = []
    failures: List[str] = []

    for i, jpath in enumerate(json_files, 1):
        jpath = jpath.resolve()
        try:
            df = htr_json_to_df_with_gt(
                json_path=jpath,
                hf_source=args.hf_source,
                gt_text_col=args.gt_text_col,
                hf_split=args.hf_split,
                keep_diacritics=args.keep_diacritics,
                #normalize_gt=args.normalize_gt,
                #scorer=args.scorer,
                #min_match_score=args.min_match_score,
            )
            # Add file provenance
            df.insert(0, "json_file", str(jpath))
            frames.append(df)
        except Exception as e:  # keep going
            failures.append(f"{jpath}: {e}")
            print(f"[{i}/{len(json_files)}] FAILED: {jpath} -> {e}")
        else:
            print(f"[{i}/{len(json_files)}] OK: {jpath} -> {len(df)} lines")

    if not frames:
        print("No dataframes produced.")
        if failures:
            print("Failures:\n" + "\n".join(failures))
        return

    combined = pd.concat(frames, ignore_index=True)

    # Decide output path
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = root.with_name(f"{root.name}_combined_lines_with_gt.csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".parquet":
        compression = "snappy" if args.parquet_snappy else None
        combined.to_parquet(out_path, index=False, compression=compression)
    else:
        combined.to_csv(out_path, index=False)

    print(f"Wrote {len(combined)} rows to {out_path}")

    if failures:
        print("\nThe following files failed to process:")
        for msg in failures:
            print(" -", msg)


if __name__ == "__main__":
    main()
