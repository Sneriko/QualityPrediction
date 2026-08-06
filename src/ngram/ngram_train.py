# train_ngrams_from_alto.py
from pathlib import Path
from typing import Dict

from alto_parser import iter_alto_pages
from ngram_model import NgramModel


def train_global_ngram_model(
    root_dir: Path,
    n: int = 5,
    level: str = "char",
) -> NgramModel:
    """
    Train a single global n-gram model on all pages.
    
    Parameters
    ----------
    root_dir : Path
        Root directory containing volume folders with ALTO XML files.
    n : int
        N-gram order.
    level : {"char", "word"}
        Character or word-level model.
    """
    model = NgramModel(n=n, level=level)  # type: ignore[arg-type]

    for volume_id, xml_path, text in iter_alto_pages(root_dir):
        # You can log progress here if you like
        # print(f"Updating with {xml_path} (volume={volume_id})")
        model.update(text)

    return model


def train_volume_specific_models(
    root_dir: Path,
    n: int = 5,
    level: str = "char",
) -> Dict[str, NgramModel]:
    """
    OPTIONAL: Train one n-gram model per volume.
    
    Returns:
        dict mapping volume_id -> NgramModel
    """
    models: Dict[str, NgramModel] = {}

    for volume_id, xml_path, text in iter_alto_pages(root_dir):
        if volume_id not in models:
            models[volume_id] = NgramModel(n=n, level=level)  # type: ignore[arg-type]
        models[volume_id].update(text)

    return models


if __name__ == "__main__":
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

    if args.per_volume:
        models = train_volume_specific_models(root_dir, n=args.n, level=args.level)

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        for vol_id, model in models.items():
            out_path = out_dir / f"ngram_{args.level}_n{args.n}_{vol_id}.pkl"
            print(f"Saving model for volume {vol_id} to {out_path}")
            model.save(out_path)
    else:
        model = train_global_ngram_model(root_dir, n=args.n, level=args.level)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving global model to {out_path}")
        model.save(out_path)
