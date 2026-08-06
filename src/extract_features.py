# example_ngram_usage.py (or in a notebook / script)
import pickle

from pathlib import Path
from ngram.ngrammodel import NgramModel
from features.features import (
    load_page_document_from_file,
    PageFeatureExtractor,
    ImageFeatureExtractor,
    NgramResource,
    Lexicon,
    NgramLMPerplexityScorer,
)

# 1. Load the trained n-gram model (global char 5-gram, for example)
ngram_model_path = Path("/home/coder/QualityPrediction/models/char5.pkl")  # adjust to your path
ngram_model = NgramModel.load(ngram_model_path)

# 2. Wrap it in our LMPerplexityScorer implementation
lm_scorer = NgramLMPerplexityScorer(ngram_model, smoothing_k=1.0)

# 3. (Optional) load your char n-gram sets for ratio_present features
#    Suppose you've precomputed these and saved as e.g. JSON or pickle.
#    Here I'll just show the structure:
#       ngram_sets = {2: set_of_bigrams, 3: set_of_trigrams, ..., 7: set_of_7grams}
ngram_sets_path = Path("/home/coder/QualityPrediction/data/ngramdata/ngram_sets/ngram_sets.pkl")  # adjust path
with ngram_sets_path.open("rb") as f:
    ngram_sets = pickle.load(f)  # dict[int, set[str]]

ngram_resource = NgramResource(ngram_sets=ngram_sets)

# 4. (Optional) lexicon for historical Swedish
# lexicon_words = set()  # fill with your vocab
# lexicon = Lexicon(words=lexicon_words)

# 5. Image feature extractor
img_extractor = ImageFeatureExtractor()

# 6. Load the HTR JSON page (your example file)
json_path = "/home/coder/QualityPrediction/data/outputs_htrflow/json/1654_(R0001308)/R0001308_00006.json"
page = load_page_document_from_file(json_path)

# 7. Build the feature extractor with all pieces plugged in
extractor = PageFeatureExtractor(
    image_feature_extractor=img_extractor,
    ngram_resource=ngram_resource,
    lm_scorer=lm_scorer,
    metadata={"century": 17, "script_type": "kurrent"},
)

# 8. Extract all features (including n-gram ratios and n-gram-perplexity features)
features = extractor.extract_features(page)

# 'features' is now a flat dict[str, float] including:
#   - img_* features
#   - region_conf_*, line_conf_*, histograms
#   - htr_* score stats & histograms
#   - token_category_* (digit/punct/upper/unknown ratios, repetition counts, etc.)
#   - char_{2..7}gram_ratio_present from NgramResource
#   - page_ppl, line_ppl_mean/std/var from NgramLMPerplexityScorer
#   - lexicality_word_ratio (if lexicon given)
#   - corr_line_conf_vs_ppl (if LM is set, as above)


for key, value in features.items():
    print(f"{key}: {value}")
