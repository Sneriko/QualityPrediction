from __future__ import annotations

import pickle
import json

from ngram.ngrammodel import NgramModel  # your existing module

from quality_prediction.config.settings import MetadataDefaults, ResourcePaths
from quality_prediction.features.binning import ConfidenceBinConfig
from quality_prediction.features.image import ImageFeatureExtractor
from quality_prediction.features.ngrams import NgramLMPerplexityScorer, NgramResource
from quality_prediction.features.page import PageFeatureExtractor
from quality_prediction.features.dit_embeddings import DiTEmbeddingExtractor, PCAModel
from quality_prediction.features.lexicon import (
    Lexicon,
    LexiconNormConfig,
    LexiconStore,
    load_dmlex_xml_words,
    load_table_words,
    load_saol_matches_words,
)

def load_lexicon_store_from_manifest(manifest_path: str) -> LexiconStore:
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    norm_cfg = LexiconNormConfig(**(m.get("norm") or {}))

    lexicons = {}
    for spec in m.get("lexicons", []):
        name = spec["name"]
        typ = spec["type"]
        path = spec["path"]

        if typ == "dmlex_xml":
            words = load_dmlex_xml_words(path)
        elif typ == "table":
            words = load_table_words(
                path,
                word_col=spec.get("word_col", "entry"),
                delimiter=spec.get("delimiter", "\t"),
            )
        elif typ == "saol_matches":
            words = load_saol_matches_words(path)
        else:
            raise ValueError(f"Unknown lexicon type: {typ}")

        lexicons[name] = Lexicon.from_words(name=name, words=words, norm=norm_cfg)

    return LexiconStore(lexicons=lexicons, norm=norm_cfg)


def make_page_feature_extractor(resources: ResourcePaths, metadata: MetadataDefaults, bin_config: ConfidenceBinConfig) -> PageFeatureExtractor:
    ngram_model = NgramModel.load(resources.char_ngram_model)
    lm_scorer = NgramLMPerplexityScorer(ngram_model, smoothing_k=1.0)

    with resources.ngram_sets.open("rb") as f:
        ngram_sets = pickle.load(f)


    # ---- OPTIONAL DiT ----
    dit_extractor = None
    if getattr(resources, "use_dit", False):
        pca = PCAModel.load(str(resources.dit_pca_path)) if resources.dit_pca_path else None
        dit_extractor = DiTEmbeddingExtractor(
            model_name=resources.dit_model_name,
            pool=resources.dit_pool,
            fp16=resources.dit_fp16,
            pca=pca,
            prefix=resources.dit_prefix,
        )

    lexicons = None
    if getattr(resources, "lexicon_manifest_json", None):
        lexicons = load_lexicon_store_from_manifest(str(resources.lexicon_manifest_json))

    return PageFeatureExtractor(
        image_feature_extractor=ImageFeatureExtractor(),
        ngram_resource=NgramResource(ngram_sets=ngram_sets),
        lm_scorer=lm_scorer,
        lexicons=lexicons,  # <-- changed
        metadata={"century": metadata.century, "script_type": metadata.script_type},
        bin_config=bin_config,
        dit_extractor=dit_extractor,
    )

