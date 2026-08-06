#!/usr/bin/env python3
# export_alto_corpus.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Tuple, List

from alto_parser import iter_alto_pages


def _load_excludes(exclude_file: Path, root_dir: Path) -> Set[Path]:
    """
    Exclude file format:
      - one path per line
      - paths are relative to root_dir
      - blank lines and lines starting with '#' are ignored
    Returns a set of normalized *relative* Paths.
    """
    excludes: Set[Path] = set()

    for line in exclude_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        p = Path(line)

        # If an absolute path is provided, convert to relative to root_dir when possible.
        if p.is_absolute():
            try:
                p = p.relative_to(root_dir)
            except ValueError:
                # Absolute path outside root_dir; keep as-is (likely won't match).
                pass

        excludes.add(p)

    return excludes


def _is_excluded(xml_path: Path, root_dir: Path, excludes: Set[Path]) -> bool:
    """
    Decide if xml_path should be excluded based on an exclude set of relative paths.

    Handles cases where actual ALTO files live under an 'alto/' subfolder but the
    exclude list may or may not include that 'alto/' segment.

    Matches:
      1) exact relative path (xml_path relative to root_dir)
      2) relative path with first 'alto' segment removed
      3) if rel path is <vol>/alto/<rest>, also match <vol>/<rest>
    """
    try:
        rel = xml_path.relative_to(root_dir)
    except ValueError:
        return False

    if rel in excludes:
        return True

    parts = rel.parts

    # Variant A: remove the first 'alto' segment if present anywhere
    if "alto" in parts:
        i = parts.index("alto")
        rel_without_alto = Path(*parts[:i], *parts[i + 1 :])
        if rel_without_alto in excludes:
            return True

    # Variant B: specifically handle <vol>/alto/<rest> -> <vol>/<rest>
    if len(parts) >= 2 and parts[1] == "alto":
        rel_with_alto_removed = Path(parts[0], *parts[2:])
        if rel_with_alto_removed in excludes:
            return True

    return False


def export_all_pages_to_single_file(
    root_dir: Path,
    out_path: Path,
    exclude_list: Optional[Path] = None,
    sort_pages: bool = True,
) -> None:
    """
    Walk all ALTO XML files under root_dir and write their text to a single UTF-8 file.

    - Excludes any files listed in exclude_list (one relative path per line).
    - Sorts pages deterministically by relative XML path if sort_pages=True.

    Output format:
      - One page after another
      - Lines preserved as reconstructed by ALTO parser
      - Blank line between pages
      - Header line with volume/page info (commented with '#')
    """
    root_dir = root_dir.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    excludes: Set[Path] = set()
    if exclude_list:
        exclude_list = exclude_list.resolve()
        excludes = _load_excludes(exclude_list, root_dir)

    # Collect then sort for deterministic order (simple + reproducible)
    pages: List[Tuple[str, Path, str]] = list(iter_alto_pages(root_dir))
    if sort_pages:
        pages.sort(key=lambda t: str(t[1].relative_to(root_dir)).lower())

    with out_path.open("w", encoding="utf-8") as f_out:
        for volume_id, xml_path, text in pages:
            if excludes and _is_excluded(xml_path, root_dir, excludes):
                continue

            page_id = xml_path.stem  # e.g. "30002022_00004"
            f_out.write(f"# VOLUME={volume_id} PAGE={page_id}\n")
            f_out.write(text.strip() + "\n\n")  # blank line between pages


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export ALTO XML pages to a single plain-text corpus file."
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="ALTO root directory (volumes beneath; ALTO XMLs are under 'alto/' subfolders).",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output corpus file, e.g. /data/corpus/all_pages.txt",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Text file with one path per line (relative to root_dir) to exclude.",
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Do not sort pages before exporting (default is sorted).",
    )

    args = parser.parse_args()

    export_all_pages_to_single_file(
        root_dir=Path(args.root_dir),
        out_path=Path(args.output),
        exclude_list=Path(args.exclude) if args.exclude else None,
        sort_pages=not args.no_sort,
    )


if __name__ == "__main__":
    main()