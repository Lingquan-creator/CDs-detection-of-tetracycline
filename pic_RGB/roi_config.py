"""ROI configuration for GUI-based ROI selection."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# ============================================
# Paths
# ============================================
DATA_DIR = Path(r"h:\000\CDs\pic_RGB\data_pic")
OUTPUT_DIR = Path(r"h:\000\CDs\pic_RGB\roi_results")

# ============================================
# ROI parameters (dict form)
# ============================================
DEFAULT_PARAMS: Dict[str, Any] = {
    "bg_ring_inner": 80,
    "bg_ring_outer": 240,
}

# GUI preview controls
GUI_PARAMS: Dict[str, Any] = {
    "preview_max_dim": 1000,
    "preview_percentile": 99.5,
}

# Active configuration
CURRENT_CONFIG: Dict[str, Any] = {**DEFAULT_PARAMS, **GUI_PARAMS}


# ============================================
# Structured settings dataclass
# ============================================
@dataclass
class ROISettings:
    """Structured ROI configuration."""
    bg_ring_inner: int = 80
    bg_ring_outer: int = 240
    preview_max_dim: int = 1000
    preview_percentile: float = 99.5

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
    print("Current ROI extractor config (GUI selection)")
    print("=" * 50)
    print(f"Data path:   {DATA_DIR}")
    print(f"Output path: {OUTPUT_DIR}")
    print(f"Background ring: inner={settings.bg_ring_inner}, outer={settings.bg_ring_outer}")
    print(f"Preview: max_dim={settings.preview_max_dim}, percentile={settings.preview_percentile}")
    print(f"Visualization: save={SAVE_VISUALIZATION}, dpi={VISUALIZATION_DPI}")
    print("=" * 50)

    print(f"Output path: {OUTPUT_DIR}")
    print("\nROI:")
    print(f"  Default ROI size hint: {settings.fixed_roi_width} x {settings.fixed_roi_height} px")
    print(f"  ROI angle (applied after selection): {settings.roi_angle_deg} deg")
    print(f"  BG ring inner: {settings.bg_ring_inner} px")
    print(f"  BG ring outer: {settings.bg_ring_outer} px")
    print("\nPreview:")
    print(f"  Downscale max dimension: {settings.preview_max_dim}px")
    print(f"  Contrast percentile: {settings.preview_percentile}")
    print("\nOutput:")
    print(f"  Save visualization: {SAVE_VISUALIZATION}")
    print(f"  DPI: {VISUALIZATION_DPI}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
