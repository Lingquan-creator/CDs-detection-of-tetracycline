"""
Unified configuration for the ROI + K-means workflow.
Keeps a single source of truth for paths, ROI defaults, and output switches.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# ============================================
# Paths
# ============================================
DATA_DIR = Path(r"H:\000\CDs\pic_RGB\data_set_2")  # Input DNG directory
OUTPUT_DIR = Path(r"H:\000\CDs\pic_RGB\data_set_2\roi_results")  # Output directory
IMAGE_EXTENSION = "*.dng"

# ============================================
# Output switches
# ============================================
EXPORT_JSON = True  # Always keep JSON output
EXPORT_CSV = True   # Detailed CSV statistics
EXPORT_PNG = False  # Visualization PNGs

# ============================================
# Workflow toggles
# ============================================
DEFAULT_INTERACTIVE = True
OVERWRITE_SELECTION = True
REUSE_LAST_SELECTION = False

# Selection cache path for persisted ROI/BG choices
SELECTION_CACHE_PATH = OUTPUT_DIR / "roi_selection_cache.json"

# ============================================
# Visualization
# ============================================
VISUALIZATION_DPI = 150
VISUALIZATION_FIGSIZE = (15, 10)

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

CURRENT_CONFIG: Dict[str, Any] = DEFAULT_PARAMS.copy()


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
# GUI shortcuts (for quick reference / help screens)
# ============================================
GUI_SHORTCUTS = [
    "Mouse drag: ROI active -> draw ROI; BG active -> move BG center",
    "Tab: toggle ROI / BG ring",
    "W/A/S/D: move active region center",
    "Arrows: active=BG adjust outer_x/outer_y; active=ROI adjust width/height",
    "Q/E: rotate active region (step=angle_step_deg); B: align BG center to ROI",
    "Enter: confirm | Q/ESC: cancel",
]

# ============================================
# Helpers
# ============================================
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
    print("\nOutput:")
    print(f"  JSON: always on")
    print(f"  CSV enabled: {EXPORT_CSV}")
    print(f"  PNG enabled: {EXPORT_PNG} (dpi={VISUALIZATION_DPI}, size={VISUALIZATION_FIGSIZE})")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
