#!/usr/bin/env python3
# train_ngrams_from_alto.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set

from alto_parser import iter_alto_pages
from ngram_model import NgramModel


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


def train_global_ngram_model(
    root_dir: Path,
    n: int = 5,
    level: str = "char",
    exclude_list: Optional[Path] = None,
) -> NgramModel:
    """
    Train a single global n-gram model on all pages, optionally excluding pages.

    exclude_list:
      text file with one path per line, relative to root_dir
    """
    root_dir = root_dir.resolve()
    excludes: Set[Path] = set()
    if exclude_list:
        excludes = _load_excludes(exclude_list.resolve(), root_dir)

    model = NgramModel(n=n, level=level)  # type: ignore[arg-type]

    for volume_id, xml_path, text in iter_alto_pages(root_dir):
        if excludes and _is_excluded(xml_path, root_dir, excludes):
            continue
        model.update(text)

    return model


def train_volume_specific_models(
    root_dir: Path,
    n: int = 5,
    level: str = "char",
    exclude_list: Optional[Path] = None,
) -> Dict[str, NgramModel]:
    """
    OPTIONAL: Train one n-gram model per volume, optionally excluding pages.

    Returns:
        dict mapping volume_id -> NgramModel
    """
    root_dir = root_dir.resolve()
    excludes: Set[Path] = set()
    if exclude_list:
        excludes = _load_excludes(exclude_list.resolve(), root_dir)

    models: Dict[str, NgramModel] = {}

    for volume_id, xml_path, text in iter_alto_pages(root_dir):
        if excludes and _is_excluded(xml_path, root_dir, excludes):
            continue

        if volume_id not in models:
            models[volume_id] = NgramModel(n=n, level=level)  # type: ignore[arg-type]
        models[volume_id].update(text)

    return models


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train n-gram model(s) from ALTO PAGE-XML pages."
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory containing volume folders with ALTO XML files.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Text file with one path per line (relative to root_dir) to exclude.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Order of the n-gram model (default: 5).",
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["char", "word"],
        default="char",
        help="Model level: 'char' or 'word' (default: char).",
    )
    parser.add_argument(
        "--per-volume",
        action="store_true",
        help="If set, trains one model per volume instead of a single global model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path. For global model: path to a .pkl file. "
        "For per-volume: directory where each volume model is saved.",
    )

    args = parser.parse_args()
    root_dir = Path(args.root_dir)
    exclude_list = Path(args.exclude) if args.exclude else None

    if args.per_volume:
        models = train_volume_specific_models(
            root_dir, n=args.n, level=args.level, exclude_list=exclude_list
        )

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        for vol_id, model in models.items():
            out_path = out_dir / f"ngram_{args.level}_n{args.n}_{vol_id}.pkl"
            print(f"Saving model for volume {vol_id} to {out_path}")
            model.save(out_path)
    else:
        model = train_global_ngram_model(
            root_dir, n=args.n, level=args.level, exclude_list=exclude_list
        )
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving global model to {out_path}")
        model.save(out_path)


if __name__ == "__main__":
    main()