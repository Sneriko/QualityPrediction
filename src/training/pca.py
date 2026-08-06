from pathlib import Path
from quality_prediction.features.dit_embeddings import DiTEmbeddingExtractor, fit_pca_for_dit, PCAModel

img_dir = Path("/home/coder/QualityPrediction/data/eval_from_training/images_no_duplicate_basenames")  # ONLY train fold images

exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
train_image_paths = [str(p) for p in sorted(img_dir.rglob("*")) if p.suffix.lower() in exts]

extractor = DiTEmbeddingExtractor(
    model_name="microsoft/dit-base",
    pool="cls",
    fp16=True,
    pca=None,
)

pca = fit_pca_for_dit(
    extractor=extractor,
    image_paths=train_image_paths,
    n_components=128,
    out_pca_path="/home/coder/QualityPrediction/models/pca_dit/dit_pca_128.pkl",
)

# Later (feature extraction time):
pca_loaded = PCAModel.load("/home/coder/QualityPrediction/models/pca_dit/dit_pca_128.pkl")
extractor_pca = DiTEmbeddingExtractor(
    model_name="microsoft/dit-base",
    pool="cls",
    fp16=True,
    pca=pca_loaded,
)
