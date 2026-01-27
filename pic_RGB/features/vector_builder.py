from __future__ import annotations

"""Helpers to build input vectors from ROIResult collections."""

import numpy as np
from typing import Iterable, List

from pic_RGB.extraction.roi_processor import ROIResult


def build_feature_vectors(results: Iterable[ROIResult]) -> np.ndarray:
    """Return stacked 4-channel corrected feature vectors."""
    vectors: List[np.ndarray] = [r.get_feature_vector() for r in results]
    if len(vectors) == 0:
        return np.zeros((0, 4), dtype=float)
    return np.vstack(vectors)


def build_rgb_vectors(results: Iterable[ROIResult]) -> np.ndarray:
    """Return stacked RGB display vectors."""
    vectors: List[np.ndarray] = [r.get_rgb_vector() for r in results]
    if len(vectors) == 0:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(vectors)
