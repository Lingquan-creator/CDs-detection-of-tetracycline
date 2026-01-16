"""
ROI extractor configuration (manual GUI workflow).
All defaults consolidated; selections can be persisted to reuse per image.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# ============================================
# Paths
# ============================================
DATA_DIR = Path(r"h:\000\CDs\pic_RGB\data_pic")       # Input DNG directory
OUTPUT_DIR = Path(r"h:\000\CDs\pic_RGB\roi_results")  # Output directory
SELECTION_STORE = OUTPUT_DIR / "roi_selections.json"

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
    "angle_step_deg": 1.0,
    "use_saved_selection": True,
    "selection_store": str(SELECTION_STORE),
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
    move_step: int = 2
    size_step: int = 5
    angle_step_deg: float = 1.0
    use_saved_selection: bool = True
    selection_store: str = str(SELECTION_STORE)

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
    print(f"  Use saved selection: {settings.use_saved_selection}")
    print(f"  Selection store: {settings.selection_store}")
    print(f"\nOutput:")
    print(f"  Save visualization: {SAVE_VISUALIZATION}")
    print(f"  DPI: {VISUALIZATION_DPI}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
