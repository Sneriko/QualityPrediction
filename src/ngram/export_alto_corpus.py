# export_alto_corpus.py
from pathlib import Path
from alto_parser import iter_alto_pages


def export_all_pages_to_single_file(root_dir: Path, out_path: Path) -> None:
    """
    Walk all ALTO XML files under root_dir and write their text
    to a single UTF-8 text file.

    Format:
        - One page after another.
        - Lines preserved as in the ALTO reconstruction.
        - A blank line between pages.
        - Optional header line with volume/page info (commented with '#').
    """
    with out_path.open("w", encoding="utf-8") as f_out:
        for volume_id, xml_path, text in iter_alto_pages(root_dir):
            page_id = xml_path.stem  # e.g. "30002022_00004"
            # Optional header so you can trace back where text came from
            f_out.write(f"# VOLUME={volume_id} PAGE={page_id}\n")

            # Write page text (already contains newlines between lines)
            f_out.write(text.strip() + "\n\n")  # blank line between pages


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export ALTO PAGE-XML to a single plain-text corpus file."
    )
    parser.add_argument("root_dir", type=str,
                        help="Root directory with volume folders containing ALTO XML.")
    parser.add_argument("output", type=str,
                        help="Output corpus file, e.g. /data/corpus/all_pages.txt")

    args = parser.parse_args()
    export_all_pages_to_single_file(Path(args.root_dir), Path(args.output))
