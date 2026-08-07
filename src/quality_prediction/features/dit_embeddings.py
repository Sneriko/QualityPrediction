from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List

import numpy as np
from PIL import Image

# Optional heavy deps
try:
    import torch
    from transformers import AutoImageProcessor, AutoModel
except Exception:  # pragma: no cover
    torch = None
    AutoImageProcessor = None
    AutoModel = None


@dataclass
class PCAModel:
    """
    Tiny PCA implementation (no sklearn dependency).
    Store mean + components; transform does: (x-mean) @ components.T
    components shape: (n_components, d)
    """
    mean: np.ndarray
    components: np.ndarray  # (k, d)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean) @ self.components.T

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "PCAModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, PCAModel):
            raise TypeError(f"Pickle at {path} is not a PCAModel.")
        return obj

    @classmethod
    def fit(cls, X: np.ndarray, n_components: int) -> "PCAModel":
        """
        Fit PCA with SVD on centered data.
        X: (n, d)
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n, d).")
        mean = X.mean(axis=0, keepdims=False)
        Xc = X - mean

        # SVD: Xc = U S Vt; principal axes are rows of Vt
        # Use full_matrices=False for efficiency
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        comps = Vt[:n_components].astype(np.float32)  # (k, d)
        return cls(mean=mean.astype(np.float32), components=comps)


class DiTEmbeddingExtractor:
    """
    Extract DiT embeddings (CLS or mean pooled) as numeric features.

    - If pca is provided, embeddings are projected to PCA space.
    - Returns dict feature names -> float, e.g. dit_emb_000, dit_emb_001, ...
    """

    def __init__(
        self,
        model_name: str = "microsoft/dit-base",
        pool: str = "cls",  # "cls" or "mean"
        device: Optional[str] = None,  # "cuda" / "cpu" / None(auto)
        fp16: bool = False,
        pca: Optional[PCAModel] = None,
        prefix: str = "dit_emb",
    ):
        self.model_name = model_name
        self.pool = pool
        self.fp16 = fp16
        self.pca = pca
        self.prefix = prefix

        if torch is None or AutoModel is None or AutoImageProcessor is None:
            raise ImportError(
                "DiTEmbeddingExtractor requires torch + transformers. "
                "Install: pip install torch transformers"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_str = device

        self._processor = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device_str)

        # fp16 only makes sense on CUDA
        if self.fp16 and self.device_str == "cuda":
            self._model.half()

    @staticmethod
    def _load_rgb(image_path: str) -> Image.Image:
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def embed_image(self, image_path: str) -> np.ndarray:
        # The dependency is checked by __init__; avoiding a decorator keeps this
        # optional module importable in environments without PyTorch.
        with torch.inference_mode():
            return self._embed_image(image_path)

    def _embed_image(self, image_path: str) -> np.ndarray:
        self._lazy_load()
        assert self._processor is not None
        assert self._model is not None

        img = self._load_rgb(image_path)
        inputs = self._processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device_str)

        if self.fp16 and self.device_str == "cuda":
            pixel_values = pixel_values.half()

        outputs = self._model(pixel_values=pixel_values)
        hs = outputs.last_hidden_state  # (1, seq, hidden)

        if self.pool == "cls":
            vec = hs[:, 0, :]  # (1, hidden)
        elif self.pool == "mean":
            vec = hs[:, 1:, :].mean(dim=1)  # (1, hidden)
        else:
            raise ValueError("pool must be 'cls' or 'mean'")

        emb = vec.float().cpu().numpy()[0].astype(np.float32)  # (hidden,)

        if self.pca is not None:
            emb = self.pca.transform(emb[None, :])[0].astype(np.float32)

        return emb

    def extract_all(self, image_path: str) -> Dict[str, float]:
        emb = self.embed_image(image_path)
        # stable, sortable names: dit_emb_000, dit_emb_001, ...
        return {f"{self.prefix}_{i:03d}": float(v) for i, v in enumerate(emb)}


def fit_pca_for_dit(
    extractor: DiTEmbeddingExtractor,
    image_paths: Sequence[str],
    n_components: int,
    out_pca_path: str,
) -> PCAModel:
    """
    Fit PCA on *TRAIN ONLY* embeddings, save it, and return it.
    """
    embs: List[np.ndarray] = []
    for p in image_paths:
        embs.append(extractor.embed_image(p))
    X = np.stack(embs, axis=0)  # (n, d)
    pca = PCAModel.fit(X, n_components=n_components)
    pca.save(out_pca_path)
    return pca
