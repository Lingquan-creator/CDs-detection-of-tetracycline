"""
ROI extractor configuration.

- Default mode（默认模式）: 自动识别 ROI，仅允许调整背景环厚度等人工输入项。
- Advanced mode（高级模式）: 手动/高级模式，可自行设置全部参数。
- 只保留一套默认参数（DEFAULT_PARAMS），两种模式基于同一默认值。
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# ============================================
# Paths
# ============================================
DATA_DIR = Path(r"h:\000\CDs\pic_RGB\data_pic")       # Input DNG directory
OUTPUT_DIR = Path(r"h:\000\CDs\pic_RGB\roi_results")  # Output directory

# ============================================
# ROI parameters (dict form)
# ============================================
# One set of defaults, shared by both modes
DEFAULT_PARAMS: Dict[str, Any] = {
    "fixed_roi_width": 120,
    "fixed_roi_height": 400,
    "roi_angle_deg": 0.0,
    "bg_ring_inner": 80,
    "bg_ring_outer": 240,
    "min_area": 1000,
    "morph_kernel_size": 5,
    "otsu_percentile": 0.85,
    "prefer_center": True,
    "center_weight": 0.3,
    "center_region_ratio": 0.7,
    "edge_exclusion_ratio": 0.15,
    "auto_roi_margin": 0.8,
    "auto_color_threshold": 0.6,
    "auto_brightness_percentile": 95,
    "auto_remove_cap": True,
    "cap_width_ratio": 1.3,
    "cap_detection_height": 0.3,
}

# Default mode (自动 / “默认模式”): only expose bg ring etc. for user input
DEFAULT_MODE_CONFIG: Dict[str, Any] = {
    **DEFAULT_PARAMS,
    "roi_mode": "default",
    # 收紧中心筛选，尽量忽略边缘光斑
    "center_weight": 0.6,
    "center_region_ratio": 0.6,
    "edge_exclusion_ratio": 0.25,
    # 收紧自动 ROI 尺寸，避免覆盖到瓶盖/瓶颈等低亮区域
    "auto_roi_margin": 0.65,
}

# Advanced mode (“高级模式”): manual control over all parameters
ADVANCED_MODE_CONFIG: Dict[str, Any] = {
    **DEFAULT_PARAMS,
    "roi_mode": "advanced",
}

# Active configuration (switch to ADVANCED_MODE_CONFIG if needed)
CURRENT_CONFIG: Dict[str, Any] = DEFAULT_MODE_CONFIG


# ============================================
# Structured settings dataclass
# ============================================
@dataclass
class ROISettings:
    """Structured ROI configuration."""

    roi_mode: str = "default"
    fixed_roi_width: int = 120
    fixed_roi_height: int = 400
    roi_angle_deg: float = 0.0
    bg_ring_inner: int = 80
    bg_ring_outer: int = 240
    min_area: int = 1000
    morph_kernel_size: int = 5
    otsu_percentile: float = 0.85
    prefer_center: bool = True
    center_weight: float = 0.3
    center_region_ratio: float = 0.7
    edge_exclusion_ratio: float = 0.15
    auto_roi_margin: float = 0.8
    auto_color_threshold: float = 0.6
    auto_brightness_percentile: float = 95
    auto_remove_cap: bool = True
    cap_width_ratio: float = 1.3
    cap_detection_height: float = 0.3

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
    print(f"  Mode: {settings.roi_mode}")
    print(f"  Fixed ROI size: {settings.fixed_roi_width} x {settings.fixed_roi_height} px")
    print(f"  ROI angle: {settings.roi_angle_deg} deg")
    print(f"  BG ring inner: {settings.bg_ring_inner} px")
    print(f"  BG ring outer: {settings.bg_ring_outer} px")
    print(f"\nSegmentation:")
    print(f"  Min area: {settings.min_area} px^2")
    print(f"  Morph kernel: {settings.morph_kernel_size} px")
    print(f"\nOutput:")
    print(f"  Save visualization: {SAVE_VISUALIZATION}")
    print(f"  DPI: {VISUALIZATION_DPI}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
