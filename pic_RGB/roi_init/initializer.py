from __future__ import annotations

"""
Compute initial ROI and background ring from K-means segmentation results.
"""

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from pic_RGB.gui.roi_gui import BackgroundSelection, ROISelection
from pic_RGB.kmeans import KMeansAlgorithm
from pic_RGB.config import settings as config


def build_preview_from_raw(raw_channels: np.ndarray) -> np.ndarray:
    """Convert 4-channel RAW split into an 8-bit RGB preview for clustering/GUI."""
    rgb = np.stack(
        [
            raw_channels[:, :, 0],
            (raw_channels[:, :, 1] + raw_channels[:, :, 2]) / 2.0,
            raw_channels[:, :, 3],
        ],
        axis=-1,
    ).astype(np.float32)
    rgb = np.clip(rgb - np.percentile(rgb, 1), 0, None)
    scale = float(np.percentile(rgb, 99.5) or 1.0)
    rgb = np.clip(rgb / scale, 0, 1)
    return (rgb * 255).astype(np.uint8)


def _fit_rotated_rect(mask: np.ndarray) -> Optional[Tuple[Tuple[float, float], float, float, float]]:
    """Fit a minimum-area rotated rectangle; returns (center, width, height, angle_deg)."""
    pts = cv2.findNonZero(mask.astype(np.uint8))
    if pts is None or len(pts) < 5:
        return None
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    if w < h:
        w, h = h, w
        angle += 90.0
    # Normalize to [-90, 90) to keep downstream rotation logic stable
    angle = ((angle + 90.0) % 180.0) - 90.0
    # If the normalized angle is still leaning past 45°, swap axes and rotate back
    if angle >= 45.0:
        w, h = h, w
        angle -= 90.0
    elif angle < -45.0:
        w, h = h, w
        angle += 90.0
    return (float(cx), float(cy)), float(w), float(h), float(angle)


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[Tuple[float, float], float, float]]:
    """Axis-aligned fallback bbox for empty/degenerate fits."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    width = float(max_x - min_x + 1)
    height = float(max_y - min_y + 1)
    center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    return center, width, height


def _load_saved_selection(filename: str, output_dir: Path) -> Optional[Tuple[ROISelection, BackgroundSelection]]:
    """
    Try to reuse existing ROI/BG from cached JSON (prefer roi_results.json, fallback results.json).
    """
    for candidate in ("roi_results.json", "results.json"):
        results_path = output_dir / candidate
        if not results_path.exists():
            continue
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue

        for entry in data:
            if isinstance(entry, dict) and entry.get("filename") == filename:
                try:
                    roi_sel = ROISelection(
                        center=(float(entry["center_x"]), float(entry["center_y"])),
                        width=float(entry["width"]),
                        height=float(entry["height"]),
                        angle_deg=float(entry.get("angle_deg", 0.0)),
                    )
                    bg_sel = BackgroundSelection(
                        center=(
                            float(entry.get("bg_center_x", roi_sel.center[0])),
                            float(entry.get("bg_center_y", roi_sel.center[1])),
                        ),
                        inner_x=float(entry.get("bg_inner_x", 0)),
                        inner_y=float(entry.get("bg_inner_y", 0)),
                        outer_x=float(entry.get("bg_outer_x", 0)),
                        outer_y=float(entry.get("bg_outer_y", 0)),
                        angle_deg=float(entry.get("bg_angle_deg", roi_sel.angle_deg)),
                    )
                    return roi_sel, bg_sel
                except Exception:
                    continue
    return None


def initialize_selections(
    raw_channels: np.ndarray,
    settings,
    algorithm: Optional[KMeansAlgorithm] = None,
    filename: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[ROISelection, BackgroundSelection, Dict]:
    """
    Derive initial ROI/BG selections.

    Priority:
      1) Reuse saved ROI/BG from `output_dir/roi_results.json` (by filename match).
      2) Fall back to K-means based automatic guess.
    """
    out_dir = output_dir or config.OUTPUT_DIR

    if filename:
        saved = _load_saved_selection(filename, out_dir)
        if saved:
            roi_sel, bg_sel = saved
            return roi_sel, bg_sel, {"source": "saved_json"}

    algo = algorithm or KMeansAlgorithm()
    preview = build_preview_from_raw(raw_channels)
    gray = algo.preprocess(preview)
    labels, _, _ = algo.segment(gray)

    bottle_rect = _fit_rotated_rect(labels > 0)
    liquid_rect = _fit_rotated_rect(labels == 2)
    bottle_bbox = _bbox_from_mask(labels > 0)
    liquid_bbox = _bbox_from_mask(labels == 2)

    H, W = gray.shape[:2]
    fallback_center = (W / 2.0, H / 2.0)

    if liquid_rect and liquid_rect[1] >= 5 and liquid_rect[2] >= 5:
        roi_center, roi_w, roi_h, roi_angle = liquid_rect
    elif liquid_bbox and liquid_bbox[1] >= 5 and liquid_bbox[2] >= 5:
        roi_center, roi_w, roi_h = liquid_bbox
        roi_angle = float(settings.roi_angle_deg)
    else:
        roi_center = fallback_center
        roi_w = float(settings.fixed_roi_width)
        roi_h = float(settings.fixed_roi_height)
        roi_angle = float(settings.roi_angle_deg)

    if bottle_rect and bottle_rect[1] >= 10 and bottle_rect[2] >= 10:
        bottle_center, bottle_w, bottle_h, bottle_angle = bottle_rect
    elif bottle_bbox and bottle_bbox[1] >= 10 and bottle_bbox[2] >= 10:
        bottle_center, bottle_w, bottle_h = bottle_bbox
        bottle_angle = roi_angle
    else:
        bottle_center = roi_center
        bottle_w, bottle_h = roi_w, roi_h
        bottle_angle = roi_angle

    inner_width = bottle_w * 1.2
    inner_height = bottle_h * 1.2
    outer_scale = math.sqrt(1.3)  # keep aspect while making area 1.3x
    outer_width = inner_width * outer_scale
    outer_height = inner_height * outer_scale

    inner_gap_x = max(0.0, (inner_width - roi_w) / 2.0)
    inner_gap_y = max(0.0, (inner_height - roi_h) / 2.0)
    outer_gap_x = max(inner_gap_x + 1e-3, (outer_width - roi_w) / 2.0)
    outer_gap_y = max(inner_gap_y + 1e-3, (outer_height - roi_h) / 2.0)

    roi_sel = ROISelection(
        center=roi_center,
        width=float(roi_w),
        height=float(roi_h),
        angle_deg=float(roi_angle),
    )
    bg_sel = BackgroundSelection(
        center=bottle_center,
        inner_x=float(inner_gap_x),
        inner_y=float(inner_gap_y),
        outer_x=float(outer_gap_x),
        outer_y=float(outer_gap_y),
        angle_deg=float(bottle_angle),
    )

    debug_info = {
        "roi_rect": (roi_center, roi_w, roi_h, roi_angle),
        "bottle_rect": (bottle_center, bottle_w, bottle_h, bottle_angle),
        "inner_size": (inner_width, inner_height),
        "outer_size": (outer_width, outer_height),
    }
    return roi_sel, bg_sel, debug_info
