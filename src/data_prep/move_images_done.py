#!/usr/bin/env python3
import shutil
from pathlib import Path

# --- CONFIGURATION ---

# Folder containing the JSON files
JSON_DIR = Path("/home/coder/QualityPrediction/data/eval_from_training/htrflow_out_json_no_duplicate_filenames/images_no_duplicate_basenames")

# Folder containing the image files
IMAGE_DIR = Path("/home/coder/QualityPrediction/data/eval_from_training/images_no_duplicate_basenames")

# Destination folder for matched images (can be same as IMAGE_DIR if you just want to "organize" in-place)
DEST_DIR = Path("/home/coder/QualityPrediction/data/eval_from_training/images_scratch")

# Image extensions to consider (lowercase, without dot)
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "tif", "tiff", "bmp", "gif", "webp"}


# --- SCRIPT LOGIC ---

def main():
    if not JSON_DIR.is_dir():
        raise SystemExit(f"JSON directory does not exist: {JSON_DIR}")
    if not IMAGE_DIR.is_dir():
        raise SystemExit(f"Image directory does not exist: {IMAGE_DIR}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # Collect basenames (stems) of all JSON files
    json_stems = {p.stem for p in JSON_DIR.glob("*.json")}
    print(f"Found {len(json_stems)} JSON files.")

    moved_count = 0
    missing_count = 0

    # For each JSON basename, look for an image with the same basename
    for stem in sorted(json_stems):
        # For each possible image extension, check if that file exists
        matched = False
        for ext in IMAGE_EXTENSIONS:
            img_path = IMAGE_DIR / f"{stem}.{ext}"
            if img_path.exists():
                dest_path = DEST_DIR / img_path.name
                print(f"Moving: {img_path} -> {dest_path}")
                shutil.move(str(img_path), str(dest_path))
                moved_count += 1
                matched = True
                # If you only expect one image per basename, break here
                break

        if not matched:
            print(f"No image found for basename: {stem}")
            missing_count += 1

    print(f"\nDone. Moved {moved_count} images. No image found for {missing_count} json files.")


if __name__ == "__main__":
    main()
