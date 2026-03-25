"""
ROI extractor configuration for GUI-driven manual selection.
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# ============================================
# Paths
# ============================================
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("PIC_RGB_DATA_DIR", str(PROJECT_DIR / "dataset2"))).expanduser().resolve()
OUTPUT_DIR = Path(
    os.environ.get("PIC_RGB_OUTPUT_DIR", str(PROJECT_DIR / "roi_results2"))
).expanduser().resolve()

# ============================================
# ROI parameters (dict form)
# ============================================
DEFAULT_PARAMS: Dict[str, Any] = {
    "fixed_roi_width": 160,
    "fixed_roi_height": 400,
    "roi_angle_deg": 0.0,
    "bg_ring_inner_x": 80,
    "bg_ring_inner_y": 80,
    "bg_ring_outer_x": 240,
    "bg_ring_outer_y": 240,
    "move_step": 2,
    "size_step": 5,
    "angle_step_deg": 0.5,
}

CURRENT_CONFIG: Dict[str, Any] = DEFAULT_PARAMS


# ============================================
# Structured settings dataclass
# ============================================
@dataclass
class ROISettings:
    """Structured ROI configuration."""

    fixed_roi_width: int = 160
    fixed_roi_height: int = 400
    roi_angle_deg: float = 0.0
    bg_ring_inner_x: int = 80
    bg_ring_inner_y: int = 80
    bg_ring_outer_x: int = 240
    bg_ring_outer_y: int = 240
    move_step: int = 5
    size_step: int = 5
    angle_step_deg: float = 0.5

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ROISettings":
        """Initialize settings from a dict; ignore unknown keys."""
        settings = cls()
        for key, value in config.items():
            if hasattr(settings, key) and value is not None:
                setattr(settings, key, value)
        return settings

    def with_overrides(self, overrides: Optional[Dict[str, Any]] = None) -> "ROISettings":
        """Return a new instance with override values applied."""
        merged = asdict(self)
        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    merged[key] = value
        return ROISettings.from_dict(merged)


# ============================================
# Output control
# ============================================
SAVE_VISUALIZATION = True
VISUALIZATION_DPI = 150
VISUALIZATION_FIGSIZE = (15, 10)

# ============================================
# Selection persistence
# ============================================
# When True, reuse cached ROI/BG selections for images with the same filename.
# Sources: existing `roi_results/roi_results.json` (previous runs) + a lightweight cache file
# for single-image runs. With reuse enabled and a match found, GUI is skipped.
REUSE_LAST_SELECTION = False
# JSON file to store per-image ROI/BG selections (used as a lightweight cache for single runs).
SELECTION_CACHE_PATH = OUTPUT_DIR / "roi_selection_cache.json"

# ============================================
# GUI shortcuts (for quick reference / help screens)
# ============================================
GUI_SHORTCUTS = [
    "Mouse drag: draw ROI (keeps current angle)",
    "Tab: toggle ROI / BG ring",
    "W/A/S/D: move active region center",
    "Arrows (BG active): adjust outer ring X/Y",
    "E/R: rotate +/- (step=angle_step_deg); B: align BG center to ROI",
    "Enter: confirm | Q/ESC: cancel",
]

# ============================================
# Misc
# ============================================
VERBOSE = True
IMAGE_EXTENSION = "*.dng"


def get_config_dict() -> Dict[str, Any]:
    """Return a shallow copy of the active config."""
    return dict(CURRENT_CONFIG)


def get_settings(overrides: Optional[Dict[str, Any]] = None) -> ROISettings:
    """Return structured settings with optional overrides."""
    base = ROISettings.from_dict(CURRENT_CONFIG)
    return base.with_overrides(overrides)


def print_config():
    """Print active config."""
    settings = get_settings()
    print("=" * 50)
    print("Current ROI extractor config")
    print("=" * 50)
    print(f"Data path:   {DATA_DIR}")
    print(f"Output path: {OUTPUT_DIR}")
    print(f"\nROI params:")
    print(f"  Fixed ROI size: {settings.fixed_roi_width} x {settings.fixed_roi_height} px")
    print(f"  ROI angle: {settings.roi_angle_deg} deg")
    print(f"  BG ring inner: ({settings.bg_ring_inner_x}, {settings.bg_ring_inner_y}) px")
    print(f"  BG ring outer: ({settings.bg_ring_outer_x}, {settings.bg_ring_outer_y}) px")
    print(f"  Move step: {settings.move_step}px | Size step: {settings.size_step}px | Angle step: {settings.angle_step_deg} deg")
    print(f"\nSelection cache: reuse={REUSE_LAST_SELECTION} -> {SELECTION_CACHE_PATH}")
    print("\nGUI shortcuts:")
    for line in GUI_SHORTCUTS:
        print(f"  - {line}")
    print(f"\nOutput:")
    print(f"  Save visualization: {SAVE_VISUALIZATION}")
    print(f"  DPI: {VISUALIZATION_DPI}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
