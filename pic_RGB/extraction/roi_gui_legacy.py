from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from ..config.roi_config import ROISettings
from ..processing.roi_manual import ManualROITools


@dataclass
class ROISelection:
    """Represents the ROI rectangle."""

    center: Tuple[float, float]
    width: float
    height: float
    angle_deg: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": [float(self.center[0]), float(self.center[1])],
            "width": float(self.width),
            "height": float(self.height),
            "angle_deg": float(self.angle_deg),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ROISelection":
        center = data.get("center")
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            raise ValueError("Invalid ROI center in persisted data")
        return cls(
            center=(float(center[0]), float(center[1])),
            width=float(data["width"]),
            height=float(data["height"]),
            angle_deg=float(data.get("angle_deg", 0.0)),
        )


@dataclass
class BackgroundSelection:
    """Represents the background ring region."""

    center: Tuple[float, float]
    inner_x: float
    inner_y: float
    outer_x: float
    outer_y: float
    angle_deg: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": [float(self.center[0]), float(self.center[1])],
            "inner_x": float(self.inner_x),
            "inner_y": float(self.inner_y),
            "outer_x": float(self.outer_x),
            "outer_y": float(self.outer_y),
            "angle_deg": float(self.angle_deg),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackgroundSelection":
        center = data.get("center")
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            raise ValueError("Invalid BG center in persisted data")
        return cls(
            center=(float(center[0]), float(center[1])),
            inner_x=float(data["inner_x"]),
            inner_y=float(data["inner_y"]),
            outer_x=float(data["outer_x"]),
            outer_y=float(data["outer_y"]),
            angle_deg=float(data.get("angle_deg", 0.0)),
        )


def _build_preview_image(raw_channels: np.ndarray) -> np.ndarray:
    """Convert 4-channel RAW array into an 8-bit RGB preview for GUI."""
    rgb = np.stack(
        [
            raw_channels[:, :, 0],
            (raw_channels[:, :, 1] + raw_channels[:, :, 2]) / 2,
            raw_channels[:, :, 3],
        ],
        axis=-1,
    ).astype(np.float32)

    rgb = np.clip(rgb - np.percentile(rgb, 1), 0, None)
    scale = np.percentile(rgb, 99.5)
    scale = scale if scale > 0 else 1.0
    rgb = np.clip(rgb / scale, 0, 1)
    return (rgb * 255).astype(np.uint8)


class ROISelectorGUI:
    """Interactive GUI for selecting ROI and background ring."""

    def __init__(
        self,
        raw_channels: np.ndarray,
        settings: ROISettings,
        initial_roi: Optional[ROISelection] = None,
        initial_bg: Optional[BackgroundSelection] = None,
        window_name: str = "ROI Selector",
    ):
        self.settings = settings
        self.window_name = window_name
        self.preview = _build_preview_image(raw_channels)
        self.manual = ManualROITools(settings)

        h, w = self.preview.shape[:2]
        default_roi = ROISelection(
            center=(w / 2, h / 2),
            width=float(settings.fixed_roi_width),
            height=float(settings.fixed_roi_height),
            angle_deg=float(settings.roi_angle_deg),
        )
        default_bg = BackgroundSelection(
            center=default_roi.center,
            inner_x=float(settings.bg_ring_inner_x),
            inner_y=float(settings.bg_ring_inner_y),
            outer_x=float(settings.bg_ring_outer_x),
            outer_y=float(settings.bg_ring_outer_y),
            angle_deg=float(settings.roi_angle_deg),
        )

        self.roi = initial_roi or default_roi
        self.bg = initial_bg or default_bg
        self.active_region = "roi"

        self.dragging = False
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_end: Optional[Tuple[int, int]] = None

        self.move_step = int(settings.move_step)
        self.size_step = int(settings.size_step)
        self.angle_step = float(settings.angle_step_deg)
        self.angle_scale = 10  # Trackbar precision: 0.1°

        self.max_width = w
        self.max_height = h
        self.max_ring = int(max(w, h) * 0.75)
        self.trackbar_ready = False

    def run(self) -> Tuple[ROISelection, BackgroundSelection]:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, min(1200, self.max_width), min(900, self.max_height))
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        self._create_trackbars()
        self.trackbar_ready = True

        while True:
            display = self._render()
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(30)
            if key == -1:
                continue

            if key in (27, ord("q")):
                cv2.destroyWindow(self.window_name)
                raise RuntimeError("ROI selection cancelled by user")
            if key in (13, ord("\r"), ord("\n")):
                cv2.destroyWindow(self.window_name)
                return self.roi, self.bg
            if key in (9, ord("\t")):
                self._toggle_active_region()
            elif key in (ord("w"), ord("W")):
                self._nudge_position(0, -self.move_step)
            elif key in (ord("s"), ord("S")):
                self._nudge_position(0, self.move_step)
            elif key in (ord("a"), ord("A")):
                self._nudge_position(-self.move_step, 0)
            elif key in (ord("d"), ord("D")):
                self._nudge_position(self.move_step, 0)
            elif key in (ord("e"), ord("E")):
                self._adjust_angle(self.angle_step)
            elif key in (ord("r"), ord("R")):
                self._adjust_angle(-self.angle_step)
            elif key in (ord("b"), ord("B")):
                self.bg = BackgroundSelection(
                    center=self.roi.center,
                    inner_x=self.bg.inner_x,
                    inner_y=self.bg.inner_y,
                    outer_x=self.bg.outer_x,
                    outer_y=self.bg.outer_y,
                    angle_deg=self.roi.angle_deg,
                )
                self._sync_trackbars()
            else:
                moved = self._handle_arrows(key)
                if moved:
                    continue

        cv2.destroyWindow(self.window_name)
        return self.roi, self.bg

    def _handle_arrows(self, key: int) -> bool:
        arrow_map = {
            81: "left",   # Linux
            82: "up",
            83: "right",
            84: "down",
            2424832: "left",   # Windows
            2490368: "up",
            2555904: "right",
            2621440: "down",
        }
        if key not in arrow_map:
            return False
        direction = arrow_map[key]
        if self.active_region == "bg":
            if direction == "up":
                self._adjust_ring_outer(y_delta=-self.size_step)
            elif direction == "down":
                self._adjust_ring_outer(y_delta=self.size_step)
            elif direction == "left":
                self._adjust_ring_outer(x_delta=-self.size_step)
            elif direction == "right":
                self._adjust_ring_outer(x_delta=self.size_step)
            return True
        return False

    def _toggle_active_region(self):
        self.active_region = "bg" if self.active_region == "roi" else "roi"

    def _nudge_position(self, dx: float, dy: float):
        cx, cy = (self.roi.center if self.active_region == "roi" else self.bg.center)
        new_center = (
            np.clip(cx + dx, 0, self.max_width - 1),
            np.clip(cy + dy, 0, self.max_height - 1),
        )
        if self.active_region == "roi":
            self.roi = ROISelection(
                center=new_center,
                width=self.roi.width,
                height=self.roi.height,
                angle_deg=self.roi.angle_deg,
            )
        else:
            self.bg = BackgroundSelection(
                center=new_center,
                inner_x=self.bg.inner_x,
                inner_y=self.bg.inner_y,
                outer_x=self.bg.outer_x,
                outer_y=self.bg.outer_y,
                angle_deg=self.bg.angle_deg,
            )
        self._sync_trackbars()

    def _adjust_roi_width(self, delta: float):
        new_width = np.clip(self.roi.width + delta, 10, self.max_width)
        self.roi = ROISelection(
            center=self.roi.center,
            width=float(new_width),
            height=self.roi.height,
            angle_deg=self.roi.angle_deg,
        )
        self._sync_trackbars()

    def _adjust_roi_height(self, delta: float):
        new_height = np.clip(self.roi.height + delta, 10, self.max_height)
        self.roi = ROISelection(
            center=self.roi.center,
            width=self.roi.width,
            height=float(new_height),
            angle_deg=self.roi.angle_deg,
        )
        self._sync_trackbars()

    def _adjust_angle(self, delta: float):
        new_angle = (self.roi.angle_deg + delta) % 180
        self.roi = ROISelection(
            center=self.roi.center,
            width=self.roi.width,
            height=self.roi.height,
            angle_deg=float(new_angle),
        )
        self.bg = BackgroundSelection(
            center=self.bg.center,
            inner_x=self.bg.inner_x,
            inner_y=self.bg.inner_y,
            outer_x=self.bg.outer_x,
            outer_y=self.bg.outer_y,
            angle_deg=float(new_angle),
        )
        self._sync_trackbars()

    def _adjust_ring_inner(self, delta: float):
        new_inner_x = np.clip(self.bg.inner_x + delta, 0, self.bg.outer_x - 1)
        new_inner_y = np.clip(self.bg.inner_y + delta, 0, self.bg.outer_y - 1)
        self.bg = BackgroundSelection(
            center=self.bg.center,
            inner_x=float(new_inner_x),
            inner_y=float(new_inner_y),
            outer_x=self.bg.outer_x,
            outer_y=self.bg.outer_y,
            angle_deg=self.bg.angle_deg,
        )
        self._sync_trackbars()

    def _adjust_ring_outer(self, x_delta: float = 0.0, y_delta: float = 0.0):
        new_outer_x = np.clip(self.bg.outer_x + x_delta, self.bg.inner_x + 1, self.max_ring)
        new_outer_y = np.clip(self.bg.outer_y + y_delta, self.bg.inner_y + 1, self.max_ring)
        self.bg = BackgroundSelection(
            center=self.bg.center,
            inner_x=self.bg.inner_x,
            inner_y=self.bg.inner_y,
            outer_x=float(new_outer_x),
            outer_y=float(new_outer_y),
            angle_deg=self.bg.angle_deg,
        )
        self._sync_trackbars()

    def _create_trackbars(self):
        cv2.createTrackbar(
            "ROI center X",
            self.window_name,
            int(self.roi.center[0]),
            self.max_width - 1,
            lambda v: self._track_roi_center_x(v),
        )
        cv2.createTrackbar(
            "ROI center Y",
            self.window_name,
            int(self.roi.center[1]),
            self.max_height - 1,
            lambda v: self._track_roi_center_y(v),
        )
        cv2.createTrackbar(
            "ROI width",
            self.window_name,
            int(self.roi.width),
            self.max_width,
            lambda v: self._track_roi_width(v),
        )
        cv2.createTrackbar(
            "ROI height",
            self.window_name,
            int(self.roi.height),
            self.max_height,
            lambda v: self._track_roi_height(v),
        )
        cv2.createTrackbar(
            "ROI angle (0.1 deg)",
            self.window_name,
            int(round(self.roi.angle_deg * self.angle_scale) % (180 * self.angle_scale)),
            180 * self.angle_scale - 1,
            lambda v: self._track_angle(v),
        )
        cv2.createTrackbar(
            "BG center X",
            self.window_name,
            int(self.bg.center[0]),
            self.max_width - 1,
            lambda v: self._track_bg_center_x(v),
        )
        cv2.createTrackbar(
            "BG center Y",
            self.window_name,
            int(self.bg.center[1]),
            self.max_height - 1,
            lambda v: self._track_bg_center_y(v),
        )
        cv2.createTrackbar(
            "BG inner X",
            self.window_name,
            int(self.bg.inner_x),
            self.max_ring,
            lambda v: self._track_ring_inner_x(v),
        )
        cv2.createTrackbar(
            "BG inner Y",
            self.window_name,
            int(self.bg.inner_y),
            self.max_ring,
            lambda v: self._track_ring_inner_y(v),
        )
        cv2.createTrackbar(
            "BG outer X",
            self.window_name,
            int(self.bg.outer_x),
            self.max_ring,
            lambda v: self._track_ring_outer_x(v),
        )
        cv2.createTrackbar(
            "BG outer Y",
            self.window_name,
            int(self.bg.outer_y),
            self.max_ring,
            lambda v: self._track_ring_outer_y(v),
        )

    def _track_roi_width(self, value: int):
        self._adjust_roi_width(value - self.roi.width)

    def _track_roi_height(self, value: int):
        self._adjust_roi_height(value - self.roi.height)

    def _track_angle(self, value: int):
        angle = (value / self.angle_scale) % 180
        self.roi = ROISelection(
            center=self.roi.center,
            width=self.roi.width,
            height=self.roi.height,
            angle_deg=float(angle),
        )
        self.bg = BackgroundSelection(
            center=self.bg.center,
            inner_x=self.bg.inner_x,
            inner_y=self.bg.inner_y,
            outer_x=self.bg.outer_x,
            outer_y=self.bg.outer_y,
            angle_deg=float(angle),
        )

    def _track_roi_center_x(self, value: int):
        clipped = np.clip(value, 0, self.max_width - 1)
        self.roi = ROISelection(
            center=(float(clipped), self.roi.center[1]),
            width=self.roi.width,
            height=self.roi.height,
            angle_deg=self.roi.angle_deg,
        )

    def _track_roi_center_y(self, value: int):
        clipped = np.clip(value, 0, self.max_height - 1)
        self.roi = ROISelection(
            center=(self.roi.center[0], float(clipped)),
            width=self.roi.width,
            height=self.roi.height,
            angle_deg=self.roi.angle_deg,
        )

    def _track_bg_center_x(self, value: int):
        clipped = np.clip(value, 0, self.max_width - 1)
        self.bg = BackgroundSelection(
            center=(float(clipped), self.bg.center[1]),
            inner_x=self.bg.inner_x,
            inner_y=self.bg.inner_y,
            outer_x=self.bg.outer_x,
            outer_y=self.bg.outer_y,
            angle_deg=self.bg.angle_deg,
        )

    def _track_bg_center_y(self, value: int):
        clipped = np.clip(value, 0, self.max_height - 1)
        self.bg = BackgroundSelection(
            center=(self.bg.center[0], float(clipped)),
            inner_x=self.bg.inner_x,
            inner_y=self.bg.inner_y,
            outer_x=self.bg.outer_x,
            outer_y=self.bg.outer_y,
            angle_deg=self.bg.angle_deg,
        )

    def _track_ring_inner_x(self, value: int):
        self.bg = BackgroundSelection(
            center=self.bg.center,
            inner_x=float(np.clip(value, 0, self.bg.outer_x - 1)),
            inner_y=self.bg.inner_y,
            outer_x=self.bg.outer_x,
            outer_y=self.bg.outer_y,
            angle_deg=self.bg.angle_deg,
        )

    def _track_ring_inner_y(self, value: int):
        self.bg = BackgroundSelection(
            center=self.bg.center,
            inner_x=self.bg.inner_x,
            inner_y=float(np.clip(value, 0, self.bg.outer_y - 1)),
            outer_x=self.bg.outer_x,
            outer_y=self.bg.outer_y,
            angle_deg=self.bg.angle_deg,
        )

    def _track_ring_outer_x(self, value: int):
        clipped = np.clip(value, self.bg.inner_x + 1, self.max_ring)
        self.bg = BackgroundSelection(
            center=self.bg.center,
            inner_x=self.bg.inner_x,
            inner_y=self.bg.inner_y,
            outer_x=float(clipped),
            outer_y=self.bg.outer_y,
            angle_deg=self.bg.angle_deg,
        )

    def _track_ring_outer_y(self, value: int):
        clipped = np.clip(value, self.bg.inner_y + 1, self.max_ring)
        self.bg = BackgroundSelection(
            center=self.bg.center,
            inner_x=self.bg.inner_x,
            inner_y=self.bg.inner_y,
            outer_x=self.bg.outer_x,
            outer_y=float(clipped),
            angle_deg=self.bg.angle_deg,
        )

    def _sync_trackbars(self):
        if not self.trackbar_ready:
            return
        cv2.setTrackbarPos("ROI center X", self.window_name, int(self.roi.center[0]))
        cv2.setTrackbarPos("ROI center Y", self.window_name, int(self.roi.center[1]))
        cv2.setTrackbarPos("ROI width", self.window_name, int(self.roi.width))
        cv2.setTrackbarPos("ROI height", self.window_name, int(self.roi.height))
        cv2.setTrackbarPos(
            "ROI angle (0.1 deg)",
            self.window_name,
            int(round(self.roi.angle_deg * self.angle_scale) % (180 * self.angle_scale)),
        )
        cv2.setTrackbarPos("BG center X", self.window_name, int(self.bg.center[0]))
        cv2.setTrackbarPos("BG center Y", self.window_name, int(self.bg.center[1]))
        cv2.setTrackbarPos("BG inner X", self.window_name, int(self.bg.inner_x))
        cv2.setTrackbarPos("BG inner Y", self.window_name, int(self.bg.inner_y))
        cv2.setTrackbarPos("BG outer X", self.window_name, int(self.bg.outer_x))
        cv2.setTrackbarPos("BG outer Y", self.window_name, int(self.bg.outer_y))

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
            self.drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.drag_end = (x, y)
            self._finalize_drag()
            self.dragging = False
            self.drag_start = None
            self.drag_end = None

    def _finalize_drag(self):
        if not self.drag_start or not self.drag_end:
            return
        x0, y0 = self.drag_start
        x1, y1 = self.drag_end
        width = max(10, abs(x1 - x0))
        height = max(10, abs(y1 - y0))
        center = ((x0 + x1) / 2, (y0 + y1) / 2)
        self.roi = ROISelection(
            center=center,
            width=float(np.clip(width, 10, self.max_width)),
            height=float(np.clip(height, 10, self.max_height)),
            angle_deg=self.roi.angle_deg,
        )
        self.bg = BackgroundSelection(
            center=self.bg.center if self.active_region == "bg" else center,
            inner_x=self.bg.inner_x,
            inner_y=self.bg.inner_y,
            outer_x=self.bg.outer_x,
            outer_y=self.bg.outer_y,
            angle_deg=self.bg.angle_deg,
        )
        self._sync_trackbars()

    def _render(self) -> np.ndarray:
        canvas = self.preview.copy()
        roi_box = self.manual.get_rotated_box(
            self.roi.center, self.roi.width, self.roi.height, self.roi.angle_deg
        )
        bg_outer_box = self.manual.get_rotated_box(
            self.bg.center,
            self.roi.width + 2 * self.bg.outer_x,
            self.roi.height + 2 * self.bg.outer_y,
            self.bg.angle_deg,
        )
        bg_inner_box = self.manual.get_rotated_box(
            self.bg.center,
            self.roi.width + 2 * self.bg.inner_x,
            self.roi.height + 2 * self.bg.inner_y,
            self.bg.angle_deg,
        )

        cv2.polylines(canvas, [bg_outer_box.astype(int)], True, (255, 165, 0), 2)
        cv2.polylines(canvas, [bg_inner_box.astype(int)], True, (255, 165, 0), 2)
        cv2.polylines(canvas, [roi_box.astype(int)], True, (0, 0, 255), 2)

        roi_center_int = tuple(int(v) for v in self.roi.center)
        bg_center_int = tuple(int(v) for v in self.bg.center)
        cv2.drawMarker(canvas, roi_center_int, (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.drawMarker(canvas, bg_center_int, (255, 165, 0), cv2.MARKER_TILTED_CROSS, 12, 2)

        if self.dragging and self.drag_start and self.drag_end:
            x0, y0 = self.drag_start
            x1, y1 = self.drag_end
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 1)

        info_lines = [
            f"Active: {'ROI' if self.active_region == 'roi' else 'BG ring'} (Tab to switch)",
            "Mouse drag: draw ROI rectangle (angle preserved)",
            "W/A/S/D: move active center | Arrows (BG active): adjust outer ring X/Y",
            f"E/R: rotate ±{self.angle_step:.2f}° (trackbar 0.1°); B: align BG center to ROI",
            "Enter: confirm  |  Q/ESC: cancel",
        ]
        y = 20
        for line in info_lines:
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2)
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
            y += 18

        return canvas


def select_roi_with_gui(
    raw_channels: np.ndarray,
    settings: ROISettings,
    initial_roi: Optional[ROISelection] = None,
    initial_bg: Optional[BackgroundSelection] = None,
) -> Tuple[ROISelection, BackgroundSelection]:
    """Helper to launch the selector and return user choices."""
    selector = ROISelectorGUI(
        raw_channels=raw_channels,
        settings=settings,
        initial_roi=initial_roi,
        initial_bg=initial_bg,
    )
    return selector.run()
