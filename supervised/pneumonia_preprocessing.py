"""Shared preprocessing for PneumoniaMNIST (pickle-safe for joblib)."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import FunctionTransformer


def flatten_normalize(X: np.ndarray) -> np.ndarray:
    """Flatten N×28×28 (or N×784) to N×784 and scale pixels from [0,255] to [0,1]."""
    X = np.asarray(X, dtype=np.float32)
    X = X.reshape(len(X), -1) / 255.0
    return X


def build_preprocess() -> FunctionTransformer:
    return FunctionTransformer(flatten_normalize, validate=False)
