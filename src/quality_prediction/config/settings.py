from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ResourcePaths:
    """External resources used by feature extraction."""
    char_ngram_model: Path
    ngram_sets: Path
    global_bin_config: Path
    lexicon_manifest_json: Path

    # --- NEW: DiT embedding options (all optional) ---
    use_dit: bool = False
    dit_model_name: str = "microsoft/dit-base"
    dit_pool: str = "cls"  # "cls" or "mean"
    dit_fp16: bool = False
    dit_pca_path: Optional[Path] = None
    dit_prefix: str = "dit_emb"


@dataclass(frozen=True)
class MetadataDefaults:
    """Weak metadata features you can attach to all pages."""
    century: Optional[int] = None
    script_type: Optional[str] = None
    