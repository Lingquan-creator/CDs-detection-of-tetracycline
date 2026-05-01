"""
Exponential Fitting Concentration Inversion Package.

This package provides tools for fitting exponential models to ROI RGB statistics
and inverting them to predict chemical concentrations.

Main modules:
    - train: Fit exponential models and generate predictions
    - predict: Use saved parameters to predict concentrations
    - config: Configuration and default paths
    - physics_features: Utilities for concentration extraction
"""

__version__ = "1.0.0"
__author__ = "CDs Project"

from .config import ExpConfig
from .physics_features import extract_concentration_from_name

__all__ = [
    "ExpConfig",
    "extract_concentration_from_name",
]
