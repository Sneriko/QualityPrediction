#!/usr/bin/env python3
import argparse
import os
import glob
import shutil

import pandas as pd


IMAGE_EXTENSIONS = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".gif")


def find_image_files(image_base_dir: str, page_id: str):
    """
    Find image files in image_base_dir whose basename starts with page_id
    and has one of the allowed IMAGE_EXTENSIONS.
    """
    matches = []
    for ext in IMAGE_EXTENSIONS:
        pattern = os.path.join(image_base_dir, f"{page_id}{ext}")
        matches.extend(glob.glob(pattern))
    return matches


def copy_group_rows(
    df_group: pd.DataFrame,
    image_base_dir: str,
    target_dir: str,
    page_id_col: str,
) -> None:
    os.makedirs(target_dir, exist_ok=True)
    print(f"  -> Copying {len(df_group)} rows into: {target_dir}")

    copied_count = 0
    missing_count = 0

    for _, row in df_group.iterrows():
        page_id = str(row[page_id_col])
        pred_rank = int(row["pred_rank"])
        gt_rank = int(row["gt_rank"])

        image_paths = find_image_files(image_base_dir, page_id)

        if not image_paths:
            print(
                f"  [WARN] No image found for page_id='{page_id}' "
                f"(searched for {page_id} with typical image extensions)"
            )
            missing_count += 1
            continue

        for src_path in image_paths:
            filename = os.path.basename(src_path)
            new_filename = f"{pred_rank}_{gt_rank}_{filename}"
            dst_path = os.path.join(target_dir, new_filename)

            shutil.copy2(src_path, dst_path)
            copied_count += 1

    print(
        f"  Group done: copied {copied_count} files, "
        f"{missing_count} page_ids with no image matches."
    )


def process_single_csv(
    csv_path: str,
    image_base_dir: str,
    output_base_dir: str,
    top_k: int = 30,
    page_id_col: str = "page_id",
    pred_col: str = "pred",
    target_col: str = "target",
) -> None:
    csv_name = os.path.basename(csv_path)
    csv_stem = os.path.splitext(csv_name)[0]

    print(f"\nProcessing CSV: {csv_name}")

    df = pd.read_csv(csv_path)

    if pred_col not in df.columns:
        raise ValueError(f"Column '{pred_col}' not found in {csv_name}")

    if page_id_col not in df.columns:
        raise ValueError(f"Column '{page_id_col}' not found in {csv_name}")

    if target_col not in df.columns:
        raise ValueError(
            f"Column '{target_col}' (ground truth / target) not found in {csv_name}"
        )

    # ---- Compute ranks over the FULL CSV ----
    # GT rank: 1 = highest GT (e.g. worst CER).
    # If you prefer 1 = lowest GT, change ascending=False to ascending=True.
    df["gt_rank"] = df[target_col].rank(method="min", ascending=False).astype(int)

    # Pred rank: 1 = highest prediction.
    df = df.sort_values(pred_col, ascending=False).reset_index(drop=True)
    df["pred_rank"] = range(1, len(df) + 1)

    # Top-K highest predictions (already sorted descending by pred)
    df_top = df.head(top_k)

    # Bottom-K lowest predictions (take last K, then sort ascending just for nicer viewing)
    df_bottom = df.tail(top_k).sort_values(pred_col, ascending=True)

    # Create base folder for this CSV
    csv_output_dir = os.path.join(output_base_dir, csv_stem)
    top_dir = os.path.join(csv_output_dir, "top_predictions")
    bottom_dir = os.path.join(csv_output_dir, "bottom_predictions")

    # Copy top predictions
    copy_group_rows(
        df_group=df_top,
        image_base_dir=image_base_dir,
        target_dir=top_dir,
        page_id_col=page_id_col,
    )

    # Copy bottom predictions
    copy_group_rows(
        df_group=df_bottom,
        image_base_dir=image_base_dir,
        target_dir=bottom_dir,
        page_id_col=page_id_col,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy images for top-K and bottom-K predictions from multiple CSV files.\n"
            "Images are renamed with '<pred_rank>_<gt_rank>_<original_filename>'."
        )
    )
    parser.add_argument(
        "--csv-dir",
        required=True,
        help="Directory containing the prediction CSV files.",
    )
    parser.add_argument(
        "--csv-pattern",
        default="*.csv",
        help="Glob pattern for CSV files inside csv-dir (default: '*.csv').",
    )
    parser.add_argument(
        "--image-base-dir",
        required=True,
        help="Base directory where the page images are stored.",
    )
    parser.add_argument(
        "--output-base-dir",
        required=True,
        help=(
            "Base directory where target folders will be created. "
            "Each CSV gets its own subfolder named after the CSV basename, "
            "with 'top_predictions' and 'bottom_predictions' inside."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of rows for top/bottom predictions per CSV (default: 30).",
    )
    parser.add_argument(
        "--page-id-col",
        default="page_id",
        help="Name of the column containing page IDs (default: 'page_id').",
    )
    parser.add_argument(
        "--pred-col",
        default="pred",
        help="Name of the column containing prediction scores (default: 'pred').",
    )
    parser.add_argument(
        "--target-col",
        default="gt",
        help=(
            "Name of the column containing the ground-truth / target values "
            "used to compute gt_rank (default: 'target')."
        ),
    )

    args = parser.parse_args()

    csv_glob = os.path.join(args.csv_dir, args.csv_pattern)
    csv_files = sorted(glob.glob(csv_glob))

    if not csv_files:
        print(f"No CSV files found matching pattern: {csv_glob}")
        return

    print(f"Found {len(csv_files)} CSV file(s).")

    for csv_path in csv_files:
        process_single_csv(
            csv_path=csv_path,
            image_base_dir=args.image_base_dir,
            output_base_dir=args.output_base_dir,
            top_k=args.top_k,
            page_id_col=args.page_id_col,
            pred_col=args.pred_col,
            target_col=args.target_col,
        )


if __name__ == "__main__":
    main()
