```bash
#!/usr/bin/env bash

set -uo pipefail

# Usage:
#   ./collect_testset.sh /path/to/dataset_root /path/to/target_directory
#
# This script:
#   - finds every page/test_0050 file
#   - copies the listed PAGE-XML files
#   - copies images with matching stems from one directory above page/
#   - guarantees unique, reasonably short output filenames
#   - skips duplicate source entries
#   - logs errors and continues processing

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 DATASET_ROOT TARGET_DIRECTORY" >&2
    exit 1
fi

dataset_root=$(realpath "$1")
target_dir=$(realpath -m "$2")

xml_target="$target_dir/page"
image_target="$target_dir/images"
error_log="$target_dir/collection_errors.log"

mkdir -p "$xml_target" "$image_target"

# Clear the previous error log.
: > "$error_log"

copied_xml=0
copied_images=0
missing_xml=0
missing_images=0
skipped_duplicates=0
copy_errors=0

declare -A seen_xml_sources
declare -A used_output_names

image_extensions=(
    jpg jpeg png tif tiff jp2 webp
    JPG JPEG PNG TIF TIFF JP2 WEBP
)

log_error() {
    local message=$1
    echo "  ERROR: $message" >&2
    printf '%s\n' "$message" >> "$error_log"
}

sanitize_name() {
    # Convert potentially unsafe characters to underscores.
    # LC_ALL=C makes byte-oriented truncation predictable.
    LC_ALL=C sed \
        -e 's|[^A-Za-z0-9._-]|_|g' \
        -e 's|_\{2,\}|_|g' \
        -e 's|^_*||' \
        -e 's|_*$||'
}

short_hash() {
    printf '%s' "$1" | sha256sum | cut -c1-12
}

make_unique_output_stem() {
    local source_path=$1
    local dataset_dir=$2
    local original_stem=$3

    local dataset_name
    local safe_dataset_name
    local safe_stem
    local source_hash
    local candidate
    local counter

    dataset_name=$(basename "$dataset_dir")
    safe_dataset_name=$(printf '%s' "$dataset_name" | sanitize_name)
    safe_stem=$(printf '%s' "$original_stem" | sanitize_name)
    source_hash=$(short_hash "$source_path")

    # Keep the human-readable portions short enough that the complete
    # filename remains well below the usual 255-byte filename limit.
    safe_dataset_name=$(printf '%.60s' "$safe_dataset_name")
    safe_stem=$(printf '%.100s' "$safe_stem")

    [[ -n "$safe_dataset_name" ]] || safe_dataset_name="dataset"
    [[ -n "$safe_stem" ]] || safe_stem="page"

    candidate="${safe_dataset_name}__${safe_stem}__${source_hash}"
    counter=2

    # The hash should already guarantee uniqueness. The loop also protects
    # against extremely unlikely hash collisions and existing output files.
    while [[ -n "${used_output_names[$candidate]+x}" ]] ||
          [[ -e "$xml_target/${candidate}.xml" ]] ||
          compgen -G "$image_target/${candidate}.*" >/dev/null; do
        candidate="${safe_dataset_name}__${safe_stem}__${source_hash}__${counter}"
        ((counter += 1))
    done

    printf '%s' "$candidate"
}

while IFS= read -r -d '' test_file; do
    page_dir=$(dirname "$test_file")
    dataset_dir=$(dirname "$page_dir")
    relative_dataset_path=${dataset_dir#"$dataset_root"/}

    echo "Processing: $relative_dataset_path"

    while IFS= read -r filename || [[ -n "$filename" ]]; do
        # Remove Windows carriage returns.
        filename=${filename//$'\r'/}

        # Remove surrounding whitespace.
        filename="${filename#"${filename%%[![:space:]]*}"}"
        filename="${filename%"${filename##*[![:space:]]}"}"

        # Ignore blank lines and comments.
        [[ -z "$filename" || "$filename" == \#* ]] && continue

        xml_name=$(basename "$filename")

        if [[ "${xml_name,,}" != *.xml ]]; then
            xml_name="${xml_name}.xml"
        fi

        xml_source="$page_dir/$xml_name"
        stem="${xml_name%.*}"

        if [[ ! -f "$xml_source" ]]; then
            log_error "XML not found: $xml_source"
            ((missing_xml += 1))
            continue
        fi

        # realpath can fail, so handle that without terminating the script.
        if ! canonical_xml_source=$(realpath "$xml_source" 2>/dev/null); then
            log_error "Could not resolve path: $xml_source"
            ((copy_errors += 1))
            continue
        fi

        # Skip repeated references to the exact same source XML.
        if [[ -n "${seen_xml_sources[$canonical_xml_source]+x}" ]]; then
            echo "  Skipping duplicate entry: $xml_source"
            ((skipped_duplicates += 1))
            continue
        fi

        seen_xml_sources["$canonical_xml_source"]=1

        output_stem=$(
            make_unique_output_stem \
                "$canonical_xml_source" \
                "$dataset_dir" \
                "$stem"
        )

        xml_destination="$xml_target/${output_stem}.xml"

        if cp -- "$xml_source" "$xml_destination"; then
            used_output_names["$output_stem"]=1
            ((copied_xml += 1))
        else
            log_error "Failed to copy XML: $xml_source -> $xml_destination"
            ((copy_errors += 1))
            continue
        fi

        image_found=false

        for extension in "${image_extensions[@]}"; do
            image_source="$dataset_dir/${stem}.${extension}"

            if [[ -f "$image_source" ]]; then
                image_destination="$image_target/${output_stem}.${extension}"

                if cp -- "$image_source" "$image_destination"; then
                    ((copied_images += 1))
                else
                    log_error \
                        "Failed to copy image: $image_source -> $image_destination"
                    ((copy_errors += 1))
                fi

                image_found=true
                break
            fi
        done

        if [[ "$image_found" == false ]]; then
            log_error "Image not found for XML: $xml_source"
            ((missing_images += 1))
        fi

    done < "$test_file"

done < <(
    find "$dataset_root" \
        -type f \
        -path '*/page/test_0050' \
        -print0
)

echo
echo "Finished."
echo "XML files copied:          $copied_xml"
echo "Images copied:             $copied_images"
echo "Duplicate entries skipped: $skipped_duplicates"
echo "Missing XML files:         $missing_xml"
echo "Missing images:            $missing_images"
echo "Copy/path errors:           $copy_errors"
echo "Target directory:          $target_dir"
echo "Error log:                 $error_log"

if (( copy_errors > 0 || missing_xml > 0 || missing_images > 0 )); then
    exit 2
fi

exit 0

