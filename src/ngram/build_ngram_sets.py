# build_ngram_sets.py
from pathlib import Path
from collections import Counter, defaultdict
import pickle

def normalise_text(s: str) -> str:
    s = s.replace("\n", " ")
    s = " ".join(s.split())
    return s

def build_ngram_sets(
    corpus_path: Path,
    n_min: int = 2,
    n_max: int = 7,
    min_count: int = 5,
    top_k_per_n: int | None = None,
) -> dict[int, set[str]]:
    """
    Scan corpus and build sets of frequent character n-grams.

    Parameters
    ----------
    corpus_path : Path
    n_min, n_max : int
        Inclusive range of n.
    min_count : int
        Only keep n-grams with frequency >= min_count.
    top_k_per_n : int or None
        If not None, keep only the top-K most frequent n-grams for each n.

    Returns
    -------
    dict[int, set[str]]
        Mapping n -> set of allowed n-grams.
    """
    counters: dict[int, Counter[str]] = {
        n: Counter() for n in range(n_min, n_max + 1)
    }

    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            text = normalise_text(line)
            if not text:
                continue
            for n in range(n_min, n_max + 1):
                if len(text) < n:
                    continue
                for i in range(len(text) - n + 1):
                    g = text[i : i + n]
                    counters[n][g] += 1

    ngram_sets: dict[int, set[str]] = {}
    for n, counter in counters.items():
        # apply min_count
        items = [(g, c) for g, c in counter.items() if c >= min_count]
        # sort by frequency
        items.sort(key=lambda x: x[1], reverse=True)
        if top_k_per_n is not None:
            items = items[:top_k_per_n]
        ngram_sets[n] = {g for g, _ in items}
        print(
            f"n={n}: kept {len(ngram_sets[n])} n-grams "
            f"(min_count={min_count}, top_k={top_k_per_n})"
        )

    return ngram_sets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build allowed char n-gram sets from corpus."
    )
    parser.add_argument("corpus", type=str, help="Path to corpus file (e.g. all_pages.txt)")
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output pickle path for ngram_sets dict.",
    )
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=500000)

    args = parser.parse_args()
    corpus_path = Path(args.corpus)
    out_path = Path(args.out)

    ngram_sets = build_ngram_sets(
        corpus_path,
        n_min=2,
        n_max=7,
        min_count=args.min_count,
        top_k_per_n=args.top_k,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(ngram_sets, f)

    print(f"Saved ngram_sets to {out_path}")
