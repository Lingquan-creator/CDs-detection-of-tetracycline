"""
Top-level configuration for the integrated K-means + ROI workflow.
"""
from pathlib import Path

# Default input/output paths (reuse existing ROI defaults)
DATA_DIR = Path(r"h:\000\CDs\pic_RGB\data_pic")
OUTPUT_DIR = Path(r"h:\000\CDs\pic_RGB\roi_results")

# Selection cache path for persisted ROI/BG choices
SELECTION_CACHE_PATH = OUTPUT_DIR / "roi_selection_cache.json"

# Toggle GUI by default
DEFAULT_INTERACTIVE = True

# Whether to overwrite existing selection entries when saving
OVERWRITE_SELECTION = True

# File extension to process
IMAGE_EXTENSION = "*.dng"
