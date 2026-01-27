from __future__ import annotations

"""
Reusable K-means segmentation wrapper.

Thin wrapper around the internal AdvancedKmeansSegmentation so downstream code
can depend on a stable API.
"""

from typing import Dict, Tuple

import numpy as np

from .advanced import AdvancedKmeansSegmentation


class KMeansAlgorithm:
    """Thin wrapper so downstream code can depend on a stable interface."""

    def __init__(self):
        self.segmenter = AdvancedKmeansSegmentation()

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Normalize/prepare input for clustering."""
        return self.segmenter.preprocess_image(image)

    def segment(self, gray_image: np.ndarray) -> Tuple[np.ndarray, Dict, Dict]:
        """
        Run nested K-means clustering.

        Returns:
            labels: int mask (0=background, 1=bottle, 2=liquid)
            kmeans_info: metadata from the model
            region_stats: pixel statistics
        """
        return self.segmenter.segment_kmeans_nested(gray_image)
