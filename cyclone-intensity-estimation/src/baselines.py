"""Simple non-deep baselines, so the CNN has something honest to be compared with."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.preprocessing import load_image, preprocess_array


class MeanBaseline:
    """Always predicts the mean training wind speed."""

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            raise ValueError("Cannot fit on empty data")
        self.mean_ = float(y.mean())
        return self

    def predict(self, n: int) -> np.ndarray:
        return np.full(int(n), self.mean_, dtype=float)


def extract_features(df: pd.DataFrame, size: int = 16, normalization: str = "fixed",
                     max_pixel_value: float = 255.0) -> np.ndarray:
    """Tiny flattened images (size x size) used as features for the ridge baseline."""
    feats = [preprocess_array(load_image(p), size, normalization, max_pixel_value).ravel()
             for p in df["abs_path"]]
    return np.stack(feats).astype(np.float32)


class RidgeBaseline:
    """Ridge regression on down-sampled pixels (a classical machine-learning reference)."""

    def __init__(self, size: int = 16, alpha: float = 10.0, normalization="fixed", max_pixel_value=255.0):
        self.size, self.alpha = size, alpha
        self.normalization, self.max_pixel_value = normalization, max_pixel_value
        self.model = Ridge(alpha=alpha)

    def fit(self, df: pd.DataFrame, max_samples: int = 5000, seed: int = 42):
        if len(df) == 0:
            raise ValueError("Cannot fit on empty data")
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=seed)
        X = extract_features(df, self.size, self.normalization, self.max_pixel_value)
        self.model.fit(X, df["wind_speed_kt"].to_numpy(dtype=float))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = extract_features(df, self.size, self.normalization, self.max_pixel_value)
        return self.model.predict(X)
