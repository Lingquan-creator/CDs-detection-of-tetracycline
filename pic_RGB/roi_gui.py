from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from roi_config import ROISettings


@dataclass
class ROISelection:
    center: Tuple[float, float]
    width: float
    height: float
    angle_deg: float = 0.0


class ROISelector:
    """Simple OpenCV GUI for drawing a rectangular ROI."""

    def __init__(self, settings: ROISettings):
        self.settings = settings
        self.move_step = 5  # pixels in preview space per key press

    def _build_preview(self, raw_channels: np.ndarray) -> Tuple[np.ndarray, float]:
        """Convert 4-channel RAW planes into an 8-bit BGR preview and a scale factor."""
        rgb = np.stack(
            [
                raw_channels[:, :, 0],
                (raw_channels[:, :, 1] + raw_channels[:, :, 2]) / 2,
                raw_channels[:, :, 3],
            ],
            axis=-1,
        ).astype(np.float32)

        percentile = float(np.percentile(rgb, self.settings.preview_percentile))
        if percentile <= 0:
            percentile = float(np.max(rgb))
        if percentile <= 0:
            percentile = 1.0

        rgb_norm = np.clip(rgb / percentile, 0, 1)
        preview_rgb = (rgb_norm * 255).astype(np.uint8)
        preview_bgr = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)

        height, width = preview_bgr.shape[:2]
        scale = 1.0
        longest = max(height, width)
        if longest > self.settings.preview_max_dim:
            scale = self.settings.preview_max_dim / float(longest)
            new_size = (int(width * scale), int(height * scale))
            preview_bgr = cv2.resize(preview_bgr, new_size, interpolation=cv2.INTER_AREA)

        return preview_bgr, scale

    def _draw_rotated_rect(self, canvas: np.ndarray, center: Tuple[float, float], size: Tuple[float, float], angle: float, color=(0, 255, 0)) -> None:
        rect = ((float(center[0]), float(center[1])), (float(size[0]), float(size[1])), float(angle))
        box = cv2.boxPoints(rect)
        box = np.int32(np.round(box))
        cv2.polylines(canvas, [box], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    def _adjust_selection(self, preview_bgr: np.ndarray, bbox, filepath: Path) -> Tuple[float, float, float, float, float]:
        """Allow tweaking center/size/angle after initial drag selection."""
        x, y, w, h = bbox
        cx, cy = x + w / 2.0, y + h / 2.0
        base_w, base_h = float(w), float(h)

        window_title = f"Adjust ROI - {filepath.name}"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, min(1200, preview_bgr.shape[1]), min(900, preview_bgr.shape[0]))

        cv2.createTrackbar("Angle (-90~90)", window_title, 90, 180, lambda v: None)   # stored as +90 offset
        cv2.createTrackbar("Width %", window_title, 100, 300, lambda v: None)         # 10%~300%
        cv2.createTrackbar("Height %", window_title, 100, 300, lambda v: None)

        while True:
            angle = cv2.getTrackbarPos("Angle (-90~90)", window_title) - 90
            scale_w = max(10, cv2.getTrackbarPos("Width %", window_title)) / 100.0
            scale_h = max(10, cv2.getTrackbarPos("Height %", window_title)) / 100.0

            adj_w = base_w * scale_w
            adj_h = base_h * scale_h

            canvas = preview_bgr.copy()
            self._draw_rotated_rect(canvas, (cx, cy), (adj_w, adj_h), angle, color=(0, 255, 0))
            cv2.drawMarker(canvas, (int(round(cx)), int(round(cy))), (0, 180, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

            instructions = [
                "Adjust ROI: Arrow/WASD move, +/- resize base, trackbars tweak size/angle",
                "Enter/Space: confirm    Esc/C: cancel",
            ]
            for i, text in enumerate(instructions):
                cv2.putText(
                    canvas,
                    text,
                    (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_title, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key in (13, 32):  # Enter or Space
                break
            if key in (27, ord("c"), ord("C")):
                cv2.destroyWindow(window_title)
                raise RuntimeError(f"ROI selection canceled for {filepath.name}")

            # Move controls
            if key in (ord("a"), 81):  # left / arrow left
                cx -= self.move_step
            elif key in (ord("d"), 83):  # right
                cx += self.move_step
            elif key in (ord("w"), 82):  # up
                cy -= self.move_step
            elif key in (ord("s"), 84):  # down
                cy += self.move_step

            # Uniform size nudge
            if key in (ord("+"), ord("=")):
                base_w *= 1.02
                base_h *= 1.02
            elif key in (ord("-"), ord("_")):
                base_w *= 0.98
                base_h *= 0.98
                base_w = max(base_w, 5)
                base_h = max(base_h, 5)

            cx = float(np.clip(cx, 0, preview_bgr.shape[1] - 1))
            cy = float(np.clip(cy, 0, preview_bgr.shape[0] - 1))

        cv2.destroyWindow(window_title)
        return cx, cy, adj_w, adj_h, float(angle)

    def select_roi(self, raw_channels: np.ndarray, filepath: Path) -> ROISelection:
        """Show a GUI window for ROI selection; returns selection in original pixel units."""
        preview_bgr, scale = self._build_preview(raw_channels)
        display = preview_bgr.copy()

        instructions = [
            f"File: {filepath.name}",
            "Drag to draw rectangle",
            "Enter/Space=confirm, C/Esc=cancel",
        ]
        for i, text in enumerate(instructions):
            cv2.putText(
                display,
                text,
                (10, 25 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (50, 220, 50) if i == 0 else (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

        window_title = f"Select ROI - {filepath.name}"
        bbox = cv2.selectROI(window_title, display, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window_title)

        x, y, w, h = bbox
        if w == 0 or h == 0:
            raise RuntimeError(f"ROI selection canceled for {filepath.name}")

        cx, cy, adj_w, adj_h, angle = self._adjust_selection(preview_bgr, bbox, filepath)

        inv_scale = 1.0 / scale
        center = (cx * inv_scale, cy * inv_scale)
        width = adj_w * inv_scale
        height = adj_h * inv_scale

        return ROISelection(center=center, width=width, height=height, angle_deg=angle)
