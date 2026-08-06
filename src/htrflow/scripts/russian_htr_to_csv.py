from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import pandas as pd

from htr_json_to_df_pred_only import htr_json_to_df_pred_only


def find_jsons(root: Path, pattern: str) -> List[Path]:
    # Always recurse
    return sorted(root.rglob(pattern))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Recursively load all HTR JSONs in a directory and combine all textlines into one DataFrame (prediction-only)."
    )
    p.add_argument("json_root", type=str, help="Directory containing HTR JSONs")
    p.add_argument("--pattern", default="*.json", help="Glob pattern for JSON files (default: '*.json')")
    p.add_argument("--keep-diacritics", action="store_true", help="Keep diacritics when normalizing (default)")
    p.add_argument("--strip-diacritics", dest="keep_diacritics", action="store_false", help="Strip diacritics when normalizing")

    # outputs
    p.add_argument("--out", type=str, default=None, help="Output file (.csv or .parquet). Default: '<json_root_basename>_pred_only.csv'")
    p.add_argument("--parquet-snappy", action="store_true", help="If saving parquet, use snappy compression")

    p.set_defaults(keep_diacritics=True)

    args = p.parse_args()

    root = Path(args.json_root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    files = find_jsons(root, args.pattern)
    if not files:
        print(f"No JSON files found under {root} with pattern {args.pattern}")
        return

    frames: List[pd.DataFrame] = []
    for i, fp in enumerate(files, 1):
        try:
            df = htr_json_to_df_pred_only(fp, keep_diacritics=args.keep_diacritics)
            df.insert(0, "json_file", str(fp.resolve()))
            frames.append(df)
            print(f"[{i}/{len(files)}] OK: {fp} -> {len(df)} lines")
        except Exception as e:
            print(f"[{i}/{len(files)}] FAIL: {fp} -> {e}")

    if not frames:
        print("No dataframes produced.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Decide output path
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = root.with_name(f"{root.name}_pred_only.csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".parquet":
        compression = "snappy" if args.parquet_snappy else None
        combined.to_parquet(out_path, index=False, compression=compression)
    else:
        combined.to_csv(out_path, index=False)

    print(f"Wrote {len(combined)} rows to {out_path}")


if __name__ == "__main__":
    main()
