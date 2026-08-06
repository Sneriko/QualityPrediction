from __future__ import annotations
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

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
# Main builder (pred-only)
# -------------------------------

def htr_json_to_df_pred_only(
    json_path: str | Path,
    *,
    keep_diacritics: bool = True,
) -> pd.DataFrame:
    """Parse a single HTR JSON file into a DataFrame."""
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

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

            htr_text_norm = normalize_text(htr_text, keep_diacritics=keep_diacritics)

            rows.append({
                "json_file": str(json_path),
                "image_name": image_name,
                "region_idx": region_counter,
                "line_idx": line_counter,
                "label": label,
                "htr_text": htr_text,
                "htr_text_norm": htr_text_norm,
                "htr_conf": htr_conf,
                "seg_conf": seg_conf,
                "tokens": tokens,
                "token_scores": token_scores,
                "token_conf_mean": token_conf_mean,
                "bbox_xmin": bbox.get("xmin"),
                "bbox_ymin": bbox.get("ymin"),
                "bbox_xmax": bbox.get("xmax"),
                "bbox_ymax": bbox.get("ymax"),
            })

    return pd.DataFrame(rows)


def collect_jsons_to_df(
    root_dir: str | Path,
    pattern: str = "*.json",
    keep_diacritics: bool = True,
) -> pd.DataFrame:
    """Recursively parse all JSONs under a directory into one combined DataFrame. Adds a 'json_file' column in the per-file stage, so no duplicate insertions here."""
    root = Path(root_dir)
    files = sorted(root.rglob(pattern))
    frames: List[pd.DataFrame] = []
    for f in files:
        try:
            df = htr_json_to_df_pred_only(f, keep_diacritics=keep_diacritics)
            frames.append(df)
        except Exception as e:
            print(f"FAILED {f}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Collect all HTR JSONs in a directory into one DataFrame (prediction-only, no GT).")
    p.add_argument("json_dir", type=str, help="Directory containing JSON files (searched recursively)")
    p.add_argument("--pattern", default="**/*.json", help="Glob pattern for JSON files (default '**/*.json')")
    p.add_argument("--keep-diacritics", action="store_true", help="Keep diacritics when normalizing (default)")
    p.add_argument("--strip-diacritics", dest="keep_diacritics", action="store_false", help="Strip diacritics when normalizing")
    p.add_argument("--out", type=str, default=None, help="Output file path (.csv or .parquet). Default: '<dir>_pred_only.csv'")
    p.set_defaults(keep_diacritics=True)

    args = p.parse_args()

    df = collect_jsons_to_df(args.json_dir, pattern=args.pattern, keep_diacritics=args.keep_diacritics)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(args.json_dir).with_name(f"{Path(args.json_dir).name}_pred_only.csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")
