# train_ngrams_from_corpus.py
from pathlib import Path
from ngrammodel import NgramModel


def train_ngram_from_corpus(
    corpus_path: Path,
    n: int = 5,
    level: str = "char",
) -> NgramModel:
    """
    Train an n-gram model directly from a pre-exported corpus file.

    Expects the corpus to be:
        - UTF-8 text
        - Optional comment/header lines starting with '#'
        - Blank lines allowed (ignored)
        - All other lines are text to train on
    """
    model = NgramModel(n=n, level=level)  # type: ignore[arg-type]

    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            # Skip comments and blank lines
            if not line.strip() or line.startswith("#"):
                continue
            model.update(line)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train an n-gram model from a plain-text corpus file."
    )
    parser.add_argument(
        "corpus",
        type=str,
        help="Path to corpus file (e.g. all_pages.txt).",
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
        "--output",
        type=str,
        required=True,
        help="Output path for the pickled model, e.g. /models/char5_global.pkl",
    )

    args = parser.parse_args()
    corpus_path = Path(args.corpus)
    out_path = Path(args.output)

    model = train_ngram_from_corpus(corpus_path, n=args.n, level=args.level)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {out_path}")
    model.save(out_path)
