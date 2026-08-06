from __future__ import annotations
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

# Optional: rapid and robust string similarity
try:
    from rapidfuzz import fuzz
    _USE_RAPIDFUZZ = True
except Exception:
    import difflib
    _USE_RAPIDFUZZ = False

# Hugging Face datasets (Parquet-backed)
try:
    from datasets import load_dataset
except Exception as e:
    raise ImportError("This script expects `datasets` + `pyarrow`. Please `pip install datasets pyarrow rapidfuzz`.\nOriginal import error: %s" % e)


# -------------------------------
# JSON parsing helpers
# -------------------------------

def _flatten_lines(node: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if isinstance(node, dict):
        if node.get("segmentation_label") == "textline":
            yield node
        for child in node.get("contains", []) or []:
            yield from _flatten_lines(child)
    elif isinstance(node, list):
        for item in node:
            yield from _flatten_lines(item)


def _first_or_join_text(text_result: Dict[str, Any]) -> Tuple[str, float]:
    texts = text_result.get("texts") or [""]
    scores = text_result.get("scores") or []
    text = texts[0] if texts else ""
    if scores:
        if isinstance(scores, list):
            conf = float(sum(scores) / len(scores))
        else:
            conf = float(scores)
    else:
        conf = float("nan")
    return text, conf


def _parse_token_scores(token_scores: Any) -> Tuple[List[str], List[float], Optional[float]]:
    tokens, scores = [], []
    if isinstance(token_scores, list):
        for pair in token_scores:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                tok, sc = pair
                tokens.append(str(tok))
                try:
                    scores.append(float(sc))
                except Exception:
                    scores.append(float("nan"))
    mean = (sum(scores) / len(scores)) if scores else None
    return tokens, scores, mean


_NORM_WHITESPACE = re.compile(r"\s+", re.UNICODE)

def normalize_text(s: str, *, keep_diacritics: bool = True) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00AD", "")  # soft hyphen
    s = s.replace("¬", "")
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\-]+", " ", s)  # hyphens/dashes -> space
    s = s.lower()
    if not keep_diacritics:
        s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
    s = _NORM_WHITESPACE.sub(" ", s).strip()
    return s


# -------------------------------
# Full-line similarity (no token-set/partial scoring)
# -------------------------------

def _ratio_score(a: str, b: str) -> float:
    if _USE_RAPIDFUZZ:
        return float(fuzz.ratio(a, b))  # 0..100
    else:
        return difflib.SequenceMatcher(None, a, b).ratio() * 100.0


def _best_full_line_match(query_norm: str, candidates_norm: List[str]) -> Tuple[int, float]:
    # Exact equality wins
    try:
        idx = candidates_norm.index(query_norm)
        return idx, 100.0
    except ValueError:
        pass

    best_i, best_s = -1, 0.0
    for i, cand in enumerate(candidates_norm):
        base = _ratio_score(query_norm, cand)
        # symmetric length penalty to discourage tiny substrings
        if query_norm and cand:
            lr = min(len(cand), len(query_norm)) / max(len(cand), len(query_norm))
            score = base * (0.5 + 0.5 * lr)
        else:
            score = 0.0
        if score > best_s:
            best_i, best_s = i, score
    return best_i, best_s


# -------------------------------
# Ground truth loading (Hugging Face Parquet)
# -------------------------------

def _load_hf_parquet_texts(
    source: Union[str, Path],
    *,
    split: Optional[str] = None,
    text_col: str = "transcription",
) -> Tuple[List[str], List[str]]:
    p = Path(str(source))

    if p.exists():
        if p.is_file() and p.suffix == ".parquet":
            dsd = load_dataset("parquet", data_files={"train": str(p)})
            ds = dsd["train"]
        else:
            files = sorted(str(f) for f in p.rglob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No parquet files found under {p}")
            dsd = load_dataset("parquet", data_files={"train": files})
            ds = dsd["train"]
    else:
        used_split = split or "train"
        ds = load_dataset(str(source), split=used_split)

    if text_col not in ds.column_names:
        raise KeyError(f"Column '{text_col}' not found. Available: {ds.column_names}")

    gt_texts = [str(x) if x is not None else "" for x in ds[text_col]]
    gt_texts_norm = [normalize_text(t) for t in gt_texts]
    return gt_texts, gt_texts_norm


# -------------------------------
# Main builder
# -------------------------------

def htr_json_to_df_with_gt(
    json_path: Union[str, Path],
    hf_source: Union[str, Path],
    *,
    gt_text_col: str = "transcription",
    hf_split: Optional[str] = None,
    keep_diacritics: bool = True,
    min_match_score: float = 0.0,
    min_len_ratio: float = 0.7,
) -> pd.DataFrame:
    """Create a per-textline DataFrame by parsing an HTR JSON and matching to *entire* GT lines.

    Matching is:
      1) exact normalized equality (best), else
      2) Levenshtein ratio with a symmetric length penalty.

    No token-set or partial scoring is used.
    """
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    gt_texts, gt_texts_norm = _load_hf_parquet_texts(
        hf_source, split=hf_split, text_col=gt_text_col
    )

    rows: List[Dict[str, Any]] = []

    image_name = data.get("image_name") or data.get("file_name")

    region_counter = -1
    for region in data.get("contains", []) or []:
        region_counter += 1
        line_counter = -1
        for line in _flatten_lines(region):
            line_counter += 1
            label = line.get("label")
            seg_conf = line.get("segmentation_confidence")
            bbox = line.get("bbox") or {}

            text_result = line.get("text_result") or {}
            htr_text, htr_conf = _first_or_join_text(text_result)

            tokens, token_scores, token_conf_mean = _parse_token_scores(line.get("token_scores"))

            q = normalize_text(htr_text, keep_diacritics=keep_diacritics)
            best_idx, best_score = _best_full_line_match(q, gt_texts_norm)

            # Enforce a minimum length ratio so single-word hits don't win
            if best_idx >= 0:
                lr = min(len(gt_texts_norm[best_idx]), len(q)) / max(1, max(len(gt_texts_norm[best_idx]), len(q)))
            else:
                lr = 0.0

            best_gt_text = gt_texts[best_idx] if best_idx >= 0 else None
            if best_idx < 0 or best_score < min_match_score or lr < min_len_ratio:
                best_idx = None
                best_gt_text = None

            rows.append({
                "image_name": image_name,
                "region_idx": region_counter,
                "line_idx": line_counter,
                "label": label,
                "htr_text": htr_text,
                "htr_conf": htr_conf,
                "seg_conf": seg_conf,
                "tokens": tokens,
                "token_scores": token_scores,
                "token_conf_mean": token_conf_mean,
                "best_gt_text": best_gt_text,
                "best_gt_score": best_score,
                "best_gt_index": best_idx,
                "bbox_xmin": bbox.get("xmin"),
                "bbox_ymin": bbox.get("ymin"),
                "bbox_xmax": bbox.get("xmax"),
                "bbox_ymax": bbox.get("ymax"),
            })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build a per-line DataFrame from HTR JSON and *entire-line* GT matching (HF parquet).")
    p.add_argument("json_path", type=str, help="Path to the HTR JSON file")
    p.add_argument("hf_source", type=str, help="HF dataset id OR local dir/file of parquet shards")
    p.add_argument("--gt-text-col", default="transcription", help="Column in GT containing ground-truth full line")
    p.add_argument("--hf-split", default=None, help="Split to use when loading a hub dataset (e.g., 'test')")
    p.add_argument("--keep-diacritics", action="store_true", help="Keep diacritics when normalizing (default)")
    p.add_argument("--strip-diacritics", dest="keep_diacritics", action="store_false", help="Strip diacritics when normalizing")
    p.add_argument("--min-match-score", type=float, default=0.0, help="Minimum score [0..100] to accept a match")
    p.add_argument("--min-len-ratio", type=float, default=0.7, help="Reject matches if min(|gt|,|pred|)/max(|gt|,|pred|) < this")
    p.set_defaults(keep_diacritics=True)

    args = p.parse_args()
    df = htr_json_to_df_with_gt(
        args.json_path,
        args.hf_source,
        gt_text_col=args.gt_text_col,
        hf_split=args.hf_split,
        keep_diacritics=args.keep_diacritics,
        min_match_score=args.min_match_score,
        min_len_ratio=args.min_len_ratio,
    )
    out_csv = Path(args.json_path).with_suffix(".full_line_GT.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}")
