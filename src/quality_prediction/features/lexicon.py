# quality_prediction/features/lexicons.py
from __future__ import annotations

import csv
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# ------------------------------
# Normalization (shared)
# ------------------------------

_PUNCT_EDGE = re.compile(r"^[^\wåäöÅÄÖ]+|[^\wåäöÅÄÖ]+$")


@dataclass(frozen=True)
class LexiconNormConfig:
    map_w_to_v: bool = False
    strip_edge_punct: bool = True


def normalize_token(s: str, cfg: LexiconNormConfig = LexiconNormConfig()) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold().strip()
    if cfg.strip_edge_punct:
        s = _PUNCT_EDGE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    if cfg.map_w_to_v:
        s = s.replace("w", "v")
    return s


# ------------------------------
# Lexicon objects
# ------------------------------

@dataclass(frozen=True)
class Lexicon:
    name: str
    words: frozenset[str]
    norm: LexiconNormConfig = LexiconNormConfig()

    def contains(self, w: str) -> bool:
        return normalize_token(w, self.norm) in self.words

    @classmethod
    def from_words(cls, name: str, words: Iterable[str], norm: LexiconNormConfig = LexiconNormConfig()) -> "Lexicon":
        nw = {normalize_token(w, norm) for w in words}
        nw.discard("")
        return cls(name=name, words=frozenset(nw), norm=norm)


@dataclass
class LexiconStore:
    """
    Holds many dictionaries; can compute per-dict and combined matches.
    """
    lexicons: Dict[str, Lexicon]
    norm: LexiconNormConfig = LexiconNormConfig()

    @property
    def combined(self) -> Lexicon:
        all_words: Set[str] = set()
        for lex in self.lexicons.values():
            all_words.update(lex.words)
        return Lexicon(name="combined", words=frozenset(all_words), norm=self.norm)

    def names(self) -> List[str]:
        return sorted(self.lexicons.keys())


# ------------------------------
# Loaders
# ------------------------------

def load_dmlex_xml_words(path: str | Path) -> Set[str]:
    """
    Extract <feat att="writtenForm" val="..."/> from DMLex XML.
    """
    path = Path(path)
    root = ET.parse(path).getroot()

    out: Set[str] = set()
    # find all feats where att="writtenForm"
    for feat in root.iter():
        if feat.tag.endswith("feat"):
            att = feat.attrib.get("att")
            if att == "writtenForm":
                v = feat.attrib.get("val", "")
                if v:
                    out.add(v)
    return out


def load_table_words(
    path: str | Path,
    *,
    word_col: str = "entry",
    delimiter: str = "\t",
    comment_prefix: str = "#",
) -> Set[str]:
    """
    Load TSV/CSV like:
    #dictionary entry dmlex standardized ...
    a55-dln a a..nn.d.1 ...
    """
    path = Path(path)
    out: Set[str] = set()

    with path.open("r", encoding="utf-8", newline="") as f:
        # sniff header line (skip comments/empties)
        header: Optional[List[str]] = None
        rows: List[List[str]] = []

        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith(comment_prefix):
                # header may be in comment line: "#dictionary entry ..."
                cand = line.lstrip(comment_prefix).strip()
                if cand and header is None:
                    header = cand.split(delimiter)
                continue
            rows.append(line.split(delimiter))

        if header is None:
            # fallback: assume first row is header
            if not rows:
                return out
            header = rows[0]
            rows = rows[1:]

        try:
            idx = header.index(word_col)
        except ValueError:
            raise ValueError(f"Column '{word_col}' not found in {path}. Header={header}")

        for r in rows:
            if idx < len(r):
                w = r[idx].strip()
                if w:
                    out.add(w)

    return out


def load_saol_matches_words(
    path: str | Path,
    *,
    lemgram_idx: int = 1,
    dash_token: str = "–",
) -> Set[str]:
    """
    Load your 'newspaper run against SAOL dict' output.
    Keep only matched rows: lemgram != "–".
    Tolerant whitespace parsing.
    """
    path = Path(path)
    out: Set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) <= lemgram_idx:
                continue
            form = parts[0]
            lemgram = parts[lemgram_idx]
            if lemgram == dash_token:
                continue
            if form:
                out.add(form)
    return out


# ------------------------------
# Feature computation
# ------------------------------

def lexical_features_from_tokens(
    tokens: Iterable[str],
    store: LexiconStore,
) -> Dict[str, float]:
    toks = [normalize_token(t, store.norm) for t in tokens]
    toks = [t for t in toks if t]
    n = len(toks)

    # Ensure stable set of output columns
    feats: Dict[str, float] = {}
    for name in store.names():
        feats[f"lex_{name}_match_ratio"] = 0.0
        feats[f"lex_{name}_match_count"] = 0.0
        feats[f"lex_{name}_unique_match_count"] = 0.0
    feats["lex_combined_match_ratio"] = 0.0
    feats["lex_combined_oov_ratio"] = 0.0

    if n == 0:
        return feats

    combined = store.combined.words
    combined_matches = 0

    for name, lex in store.lexicons.items():
        m = [t for t in toks if t in lex.words]
        feats[f"lex_{name}_match_count"] = float(len(m))
        feats[f"lex_{name}_unique_match_count"] = float(len(set(m)))
        feats[f"lex_{name}_match_ratio"] = float(len(m)) / float(n)

    for t in toks:
        if t in combined:
            combined_matches += 1
    feats["lex_combined_match_ratio"] = float(combined_matches) / float(n)
    feats["lex_combined_oov_ratio"] = 1.0 - feats["lex_combined_match_ratio"]

    return feats
