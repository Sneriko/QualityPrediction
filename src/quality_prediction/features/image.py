from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


class ImageFeatureExtractor:
    def __init__(self, cv2_module=None):
        self.cv2 = cv2_module if cv2_module is not None else cv2

    def load_image(self, path: str) -> np.ndarray:
        return np.array(Image.open(path).convert("L"))

    def basic_stats(self, img: np.ndarray) -> Dict[str, float]:
        arr = img.astype(np.float32) / 255.0
        mean = float(arr.mean())
        std = float(arr.std())
        return {"img_mean_intensity": mean, "img_std_intensity": std, "img_contrast": std}

    def blur_score(self, img: np.ndarray) -> float:
        if self.cv2 is None:
            return float("nan")
        lap = self.cv2.Laplacian(img, ddepth=self.cv2.CV_64F)
        return float(lap.var())

    def noise_score(self, img: np.ndarray) -> float:
        if self.cv2 is None:
            return float("nan")
        blurred = self.cv2.medianBlur(img, 3)
        diff = img.astype(np.float32) - blurred.astype(np.float32)
        return float(diff.std())

    def binarisation_features(self, img: np.ndarray) -> Dict[str, float]:
        if self.cv2 is None:
            return {"fg_ratio": float("nan")}
        _, th = self.cv2.threshold(img, 0, 255, self.cv2.THRESH_BINARY + self.cv2.THRESH_OTSU)
        fg = th == 0
        return {"fg_ratio": float(fg.mean())}

    def stroke_width_stats(self, img: np.ndarray) -> Dict[str, float]:
        if self.cv2 is None:
            return {"stroke_width_mean": float("nan"), "stroke_width_std": float("nan")}
        _, th = self.cv2.threshold(img, 0, 255, self.cv2.THRESH_BINARY + self.cv2.THRESH_OTSU)
        fg = (255 - th).astype(np.uint8)
        dist = self.cv2.distanceTransform(fg, self.cv2.DIST_L2, 3)
        vals = dist[fg > 0]
        if vals.size == 0:
            return {"stroke_width_mean": float("nan"), "stroke_width_std": float("nan")}
        return {"stroke_width_mean": float(vals.mean()), "stroke_width_std": float(vals.std())}

    def connected_components(self, img: np.ndarray) -> Dict[str, float]:
        if self.cv2 is None:
            return {"cc_count": float("nan")}
        _, th = self.cv2.threshold(img, 0, 255, self.cv2.THRESH_BINARY + self.cv2.THRESH_OTSU)
        fg = (255 - th).astype(np.uint8)
        num_labels, _ = self.cv2.connectedComponents(fg)
        return {"cc_count": float(num_labels)}

    def skew_angle(self, img: np.ndarray) -> float:
        if self.cv2 is None:
            return float("nan")
        edges = self.cv2.Canny(img, 50, 150, apertureSize=3)
        lines = self.cv2.HoughLines(edges, 1, np.pi / 180.0, 200)
        if lines is None:
            return float("nan")
        angles = []
        for rho, theta in lines[:, 0]:
            angle_deg = (theta * 180.0 / np.pi) - 90.0
            if -45 <= angle_deg <= 45:
                angles.append(angle_deg)
        return float(np.median(np.array(angles))) if angles else float("nan")

    def extract_all(self, image_path: str) -> Dict[str, float]:
        img = self.load_image(image_path)
        feats: Dict[str, float] = {}
        feats.update(self.basic_stats(img))
        feats["blur_score_var_laplacian"] = self.blur_score(img)
        feats["noise_std_diff_median"] = self.noise_score(img)
        feats.update(self.binarisation_features(img))
        feats.update(self.stroke_width_stats(img))
        feats.update(self.connected_components(img))
        feats["skew_angle_deg"] = self.skew_angle(img)
        return feats
