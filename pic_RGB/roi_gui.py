from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QEvent, QEventLoop, QTimer
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QPolygonF,
    QMouseEvent,
    QKeyEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QSizePolicy,
    QScrollArea,
    QGridLayout,
    QStyle,
    QStyleOptionSpinBox,
)

from roi_config import ROISettings
from roi_manual import ManualROITools


DEFAULT_SPIN_REPEAT_INTERVAL_MS = 100

try:
    SPINBOX_COMPLEX_CONTROL = QStyle.ComplexControl.CC_SpinBox
    SPINBOX_UP_CONTROL = QStyle.SubControl.SC_SpinBoxUp
    SPINBOX_DOWN_CONTROL = QStyle.SubControl.SC_SpinBoxDown
except AttributeError:
    SPINBOX_COMPLEX_CONTROL = QStyle.CC_SpinBox
    SPINBOX_UP_CONTROL = QStyle.SC_SpinBoxUp
    SPINBOX_DOWN_CONTROL = QStyle.SC_SpinBoxDown


def _mouse_event_pos(event: QMouseEvent):
    if hasattr(event, "position"):
        return event.position().toPoint()
    return event.pos()


class RepeatStepSpinBox(QSpinBox):
    """Integer spinbox with 1-unit clicks and 10 Hz press-and-hold stepping."""

    def __init__(self, repeat_interval_ms: int = DEFAULT_SPIN_REPEAT_INTERVAL_MS, parent=None):
        super().__init__(parent)
        self._held_step_direction = 0
        self._repeat_timer = QTimer(self)
        self._repeat_timer.setInterval(max(1, int(repeat_interval_ms)))
        self._repeat_timer.timeout.connect(self._repeat_held_step)
        self.setAccelerated(False)

    def _step_button_direction(self, event: QMouseEvent) -> int:
        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        style = self.style()
        pos = _mouse_event_pos(event)
        up_rect = style.subControlRect(
            SPINBOX_COMPLEX_CONTROL,
            option,
            SPINBOX_UP_CONTROL,
            self,
        )
        down_rect = style.subControlRect(
            SPINBOX_COMPLEX_CONTROL,
            option,
            SPINBOX_DOWN_CONTROL,
            self,
        )
        if up_rect.contains(pos):
            return 1
        if down_rect.contains(pos):
            return -1
        return 0

    def _repeat_held_step(self):
        if self._held_step_direction:
            self.stepBy(self._held_step_direction)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            direction = self._step_button_direction(event)
            if direction:
                self.setFocus(Qt.MouseFocusReason)
                self._held_step_direction = direction
                self.stepBy(direction)
                self._repeat_timer.start()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._held_step_direction:
            self._repeat_timer.stop()
            self._held_step_direction = 0
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self._repeat_timer.stop()
        self._held_step_direction = 0
        super().leaveEvent(event)


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
    """Represents the independent background inner and outer rectangles."""

    center: Tuple[float, float]
    inner_x: float
    inner_y: float
    outer_x: float
    outer_y: float
    angle_deg: float
    inner_center: Optional[Tuple[float, float]] = None
    outer_center: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        if self.inner_center is None:
            self.inner_center = self.center
        if self.outer_center is None:
            self.outer_center = self.center
        self.center = self.outer_center

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": [float(self.outer_center[0]), float(self.outer_center[1])],
            "inner_center": [float(self.inner_center[0]), float(self.inner_center[1])],
            "outer_center": [float(self.outer_center[0]), float(self.outer_center[1])],
            "inner_x": float(self.inner_x),
            "inner_y": float(self.inner_y),
            "outer_x": float(self.outer_x),
            "outer_y": float(self.outer_y),
            "angle_deg": float(self.angle_deg),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], roi: Optional[ROISelection] = None) -> "BackgroundSelection":
        center = data.get("center")
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            raise ValueError("Invalid BG center in persisted data")
        inner_center = data.get("inner_center")
        outer_center = data.get("outer_center")
        has_independent_boxes = (
            isinstance(inner_center, (list, tuple))
            and len(inner_center) == 2
            and isinstance(outer_center, (list, tuple))
            and len(outer_center) == 2
        )
        inner_x = float(data["inner_x"])
        inner_y = float(data["inner_y"])
        outer_x = float(data["outer_x"])
        outer_y = float(data["outer_y"])
        if roi is not None and not has_independent_boxes:
            inner_x = float(roi.width) + 2.0 * inner_x
            inner_y = float(roi.height) + 2.0 * inner_y
            outer_x = float(roi.width) + 2.0 * outer_x
            outer_y = float(roi.height) + 2.0 * outer_y
        inner_center_tuple = (
            (float(inner_center[0]), float(inner_center[1]))
            if has_independent_boxes
            else (float(center[0]), float(center[1]))
        )
        outer_center_tuple = (
            (float(outer_center[0]), float(outer_center[1]))
            if has_independent_boxes
            else (float(center[0]), float(center[1]))
        )
        return cls(
            center=outer_center_tuple,
            inner_center=inner_center_tuple,
            outer_center=outer_center_tuple,
            inner_x=inner_x,
            inner_y=inner_y,
            outer_x=outer_x,
            outer_y=outer_y,
            angle_deg=float(data.get("angle_deg", 0.0)),
        )


def _build_preview_image(raw_channels: np.ndarray) -> np.ndarray:
    """Convert 4-channel RAW array into an 8-bit RGB preview for GUI."""
    H, W = raw_channels.shape[:2]
    sample_step = max(1, int(np.ceil(np.sqrt((H * W) / 1_000_000))))
    sample = raw_channels[::sample_step, ::sample_step, :]
    green_sample = (sample[:, :, 1] + sample[:, :, 2]) * 0.5
    level_sample = np.concatenate(
        (
            sample[:, :, 0].ravel(),
            green_sample.ravel(),
            sample[:, :, 3].ravel(),
        )
    )

    low = float(np.percentile(level_sample, 1))
    high = float(np.percentile(level_sample, 99.5))
    scale = high - low
    if scale <= 0:
        scale = 1.0

    preview = np.empty((H, W, 3), dtype=np.uint8)
    work = np.empty((H, W), dtype=np.float32)

    def write_channel(channel: np.ndarray, target_index: int) -> None:
        nonlocal work
        np.subtract(channel, low, out=work)
        np.clip(work, 0, scale, out=work)
        work *= 255.0 / scale
        preview[:, :, target_index] = work.astype(np.uint8)

    write_channel(raw_channels[:, :, 0], 0)
    np.add(raw_channels[:, :, 1], raw_channels[:, :, 2], out=work)
    work *= 0.5
    write_channel(work, 1)
    write_channel(raw_channels[:, :, 3], 2)

    return np.ascontiguousarray(preview)


class ImageWidget(QWidget):
    """Widget to display the image and handle mouse interaction for ROI selection."""

    roi_updated = Signal(object)  # Emits updated data (partial or full)
    
    def __init__(self, raw_image: np.ndarray, manual_tools: ManualROITools, parent=None):
        super().__init__(parent)
        self.manual_tools = manual_tools
        self.raw_image = raw_image
        h, w = raw_image.shape[:2]
        
        # Convert numpy array to QImage (Assuming RGB format from _build_preview_image)
        # Note: raw_image is contiguous? QImage needs it.
        if not raw_image.flags['C_CONTIGUOUS']:
            raw_image = np.ascontiguousarray(raw_image)
        self.qimage = QImage(
            raw_image.data, w, h, 3 * w, QImage.Format_RGB888
        )
        self.pixmap = QPixmap.fromImage(self.qimage)
        
        self.roi: Optional[ROISelection] = None
        self.bg: Optional[BackgroundSelection] = None
        self.active_region = "roi"  # "roi", "bg_inner", or "bg_outer"

        # Zoom/Pan state
        self.scale_factor = 1.0
        self.transform_offset = QPointF(0, 0)
        
        # Interaction
        self.dragging = False
        self.drag_start: Optional[QPointF] = None
        self.drag_current: Optional[QPointF] = None

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_data(self, roi: ROISelection, bg: BackgroundSelection):
        self.roi = roi
        self.bg = bg
        self.update()

    def set_active_region(self, region: str):
        self.active_region = region
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw Image centered and scaled
        w_widget = self.width()
        h_widget = self.height()
        
        # Determine scale to fit if 1.0 (or implement zoom later)
        # For now, let's just scale to fit widget
        if self.pixmap.isNull():
            return

        img_w = self.pixmap.width()
        img_h = self.pixmap.height()
        
        scale_w = w_widget / img_w
        scale_h = h_widget / img_h
        self.scale_factor = min(scale_w, scale_h)
        
        # Centering offset
        drawn_w = img_w * self.scale_factor
        drawn_h = img_h * self.scale_factor
        offset_x = (w_widget - drawn_w) / 2
        offset_y = (h_widget - drawn_h) / 2
        
        self.transform_offset = QPointF(offset_x, offset_y)
        
        painter.translate(self.transform_offset)
        painter.scale(self.scale_factor, self.scale_factor)
        
        painter.drawPixmap(0, 0, self.pixmap)
        
        # Draw Overlays
        if self.roi and self.bg:
            self._draw_overlays(painter)

        # Draw drag rectangle
        if self.dragging and self.drag_start and self.drag_current:
            pen = QPen(Qt.green)
            pen.setWidthF(1.0 / self.scale_factor) # Keep constant width on screen
            painter.setPen(pen)
            rect = QRectF(self.drag_start, self.drag_current).normalized()
            painter.drawRect(rect)

    def _draw_overlays(self, painter: QPainter):
        roi_pen = QPen(Qt.red)
        roi_pen.setWidthF((3.0 if self.active_region == "roi" else 2.0) / self.scale_factor)

        bg_outer_pen = QPen(QColor(255, 140, 0))
        bg_outer_pen.setWidthF((3.0 if self.active_region == "bg_outer" else 2.0) / self.scale_factor)

        bg_inner_pen = QPen(QColor(255, 215, 0))
        bg_inner_pen.setWidthF((3.0 if self.active_region == "bg_inner" else 2.0) / self.scale_factor)
        bg_inner_pen.setStyle(Qt.DashLine)

        # Draw ROI
        roi_points = self.manual_tools.get_rotated_box(
            self.roi.center, self.roi.width, self.roi.height, self.roi.angle_deg
        )
        painter.setPen(roi_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPolygon(QPolygonF([QPointF(*p) for p in roi_points]))
        
        # Draw ROI Center
        painter.drawPoint(QPointF(*self.roi.center))

        # Draw BG Outer
        bg_outer_points = self.manual_tools.get_rotated_box(
            self.bg.outer_center,
            self.bg.outer_x,
            self.bg.outer_y,
            self.bg.angle_deg,
        )
        painter.setPen(bg_outer_pen)
        painter.drawPolygon(QPolygonF([QPointF(*p) for p in bg_outer_points]))

        # Draw BG Inner
        bg_inner_points = self.manual_tools.get_rotated_box(
            self.bg.inner_center,
            self.bg.inner_x,
            self.bg.inner_y,
            self.bg.angle_deg,
        )
        painter.setPen(bg_inner_pen)
        painter.drawPolygon(QPolygonF([QPointF(*p) for p in bg_inner_points]))
        
        painter.drawPoint(QPointF(*self.bg.inner_center))
        painter.drawPoint(QPointF(*self.bg.outer_center))

    def _map_to_image(self, pos: QPointF) -> QPointF:
        """Map widget coordinates to image coordinates."""
        return (pos - self.transform_offset) / self.scale_factor

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            img_pos = self._map_to_image(event.position())
            self.drag_start = img_pos
            self.drag_current = img_pos
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.dragging:
            img_pos = self._map_to_image(event.position())
            self.drag_current = img_pos
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            img_pos = self._map_to_image(event.position())
            self._finalize_drag(self.drag_start, img_pos)
            self.drag_start = None
            self.drag_current = None
            self.update()

    def _finalize_drag(self, start: QPointF, end: QPointF):
        width = abs(end.x() - start.x())
        height = abs(end.y() - start.y())
        cx = (start.x() + end.x()) / 2
        cy = (start.y() + end.y()) / 2
        
        if width < 5 or height < 5:
            return  # Ignoring accidental clicks

        if self.active_region == "roi":
            self.roi_updated.emit(
                {
                    "roi": ROISelection(
                        center=(cx, cy),
                        width=width,
                        height=height,
                        angle_deg=self.roi.angle_deg,
                    )
                }
            )
            return

        if self.active_region == "bg_inner":
            self.roi_updated.emit(
                {
                    "bg": BackgroundSelection(
                        center=self.bg.outer_center,
                        inner_center=(cx, cy),
                        outer_center=self.bg.outer_center,
                        inner_x=width,
                        inner_y=height,
                        outer_x=self.bg.outer_x,
                        outer_y=self.bg.outer_y,
                        angle_deg=self.bg.angle_deg,
                    )
                }
            )
            return

        if self.active_region == "bg_outer":
            self.roi_updated.emit(
                {
                    "bg": BackgroundSelection(
                        center=(cx, cy),
                        inner_center=self.bg.inner_center,
                        outer_center=(cx, cy),
                        inner_x=self.bg.inner_x,
                        inner_y=self.bg.inner_y,
                        outer_x=width,
                        outer_y=height,
                        angle_deg=self.bg.angle_deg,
                    )
                }
            )


class ControlsWidget(QWidget):
    """Side panel for controls."""
    
    values_changed = Signal(object) # Emit full state or partial
    active_region_changed = Signal(str)
    
    def __init__(self, settings: ROISettings, parent=None):
        super().__init__(parent)
        self._updating_controls = False
        self.spin_repeat_interval_ms = int(settings.spin_repeat_interval_ms)
        self.layout = QVBoxLayout(self)

        self.mode_group = QGroupBox("Mouse Drag Target")
        mode_layout = QVBoxLayout()
        self.radio_roi = QRadioButton("ROI")
        self.radio_bg_inner = QRadioButton("BG inner")
        self.radio_bg_outer = QRadioButton("BG outer")
        self.radio_roi.setChecked(True)
        self.mode_buttons = QButtonGroup(self)
        self.mode_buttons.setExclusive(True)
        self.mode_buttons.addButton(self.radio_roi)
        self.mode_buttons.addButton(self.radio_bg_inner)
        self.mode_buttons.addButton(self.radio_bg_outer)
        mode_layout.addWidget(self.radio_roi)
        mode_layout.addWidget(self.radio_bg_inner)
        mode_layout.addWidget(self.radio_bg_outer)
        self.mode_group.setLayout(mode_layout)
        self.layout.addWidget(self.mode_group)
        
        # --- ROI Group ---
        self.roi_group = QGroupBox("ROI Settings")
        roi_layout = QGridLayout()
        
        self.spin_roi_cx = self._create_spinbox(0, 10000, "Center X", settings.move_step)
        self.spin_roi_cy = self._create_spinbox(0, 10000, "Center Y", settings.move_step)
        self.spin_roi_w = self._create_spinbox(1, 5000, "Width", settings.size_step)
        self.spin_roi_h = self._create_spinbox(1, 5000, "Height", settings.size_step)
        self.spin_roi_a = self._create_double_spinbox(0, 360, "Angle")

        roi_layout.addWidget(QLabel("Center X:"), 0, 0)
        roi_layout.addWidget(self.spin_roi_cx, 0, 1)
        roi_layout.addWidget(QLabel("Center Y:"), 1, 0)
        roi_layout.addWidget(self.spin_roi_cy, 1, 1)
        roi_layout.addWidget(QLabel("Width:"), 2, 0)
        roi_layout.addWidget(self.spin_roi_w, 2, 1)
        roi_layout.addWidget(QLabel("Height:"), 3, 0)
        roi_layout.addWidget(self.spin_roi_h, 3, 1)
        roi_layout.addWidget(QLabel("Angle:"), 4, 0)
        roi_layout.addWidget(self.spin_roi_a, 4, 1)
        
        self.roi_group.setLayout(roi_layout)
        self.layout.addWidget(self.roi_group)

        # --- BG Group ---
        self.bg_group = QGroupBox("Background Settings")
        bg_layout = QGridLayout()
        
        self.spin_bg_inner_cx = self._create_spinbox(0, 10000, "BG inner center X", settings.move_step)
        self.spin_bg_inner_cy = self._create_spinbox(0, 10000, "BG inner center Y", settings.move_step)
        self.spin_bg_in_x = self._create_spinbox(1, 5000, "BG inner width", settings.size_step)
        self.spin_bg_in_y = self._create_spinbox(1, 5000, "BG inner height", settings.size_step)
        self.spin_bg_outer_cx = self._create_spinbox(0, 10000, "BG outer center X", settings.move_step)
        self.spin_bg_outer_cy = self._create_spinbox(0, 10000, "BG outer center Y", settings.move_step)
        self.spin_bg_out_x = self._create_spinbox(1, 5000, "BG outer width", settings.size_step)
        self.spin_bg_out_y = self._create_spinbox(1, 5000, "BG outer height", settings.size_step)

        bg_layout.addWidget(QLabel("Inner Cx:"), 0, 0)
        bg_layout.addWidget(self.spin_bg_inner_cx, 0, 1)
        bg_layout.addWidget(QLabel("Inner Cy:"), 1, 0)
        bg_layout.addWidget(self.spin_bg_inner_cy, 1, 1)
        bg_layout.addWidget(QLabel("Inner W:"), 2, 0)
        bg_layout.addWidget(self.spin_bg_in_x, 2, 1)
        bg_layout.addWidget(QLabel("Inner H:"), 3, 0)
        bg_layout.addWidget(self.spin_bg_in_y, 3, 1)
        bg_layout.addWidget(QLabel("Outer Cx:"), 4, 0)
        bg_layout.addWidget(self.spin_bg_outer_cx, 4, 1)
        bg_layout.addWidget(QLabel("Outer Cy:"), 5, 0)
        bg_layout.addWidget(self.spin_bg_outer_cy, 5, 1)
        bg_layout.addWidget(QLabel("Outer W:"), 6, 0)
        bg_layout.addWidget(self.spin_bg_out_x, 6, 1)
        bg_layout.addWidget(QLabel("Outer H:"), 7, 0)
        bg_layout.addWidget(self.spin_bg_out_y, 7, 1)
        
        self.btn_align_bg = QPushButton("Align BG Boxes to ROI Center")
        bg_layout.addWidget(self.btn_align_bg, 8, 0, 1, 2)
        
        self.bg_group.setLayout(bg_layout)
        self.layout.addWidget(self.bg_group)

        self.btn_confirm = QPushButton("Confirm Selection (Enter)")
        self.btn_confirm.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.layout.addWidget(self.btn_confirm)

        self.layout.addStretch()

        # Connections
        for spin in [self.spin_roi_cx, self.spin_roi_cy, self.spin_roi_w, self.spin_roi_h]:
            spin.valueChanged.connect(self._on_roi_changed)
        self.spin_roi_a.valueChanged.connect(self._on_roi_changed)
        
        for spin in [
            self.spin_bg_inner_cx,
            self.spin_bg_inner_cy,
            self.spin_bg_in_x,
            self.spin_bg_in_y,
            self.spin_bg_outer_cx,
            self.spin_bg_outer_cy,
            self.spin_bg_out_x,
            self.spin_bg_out_y,
        ]:
            spin.valueChanged.connect(self._on_bg_changed)
            
        self.btn_align_bg.clicked.connect(self._on_align_bg)
        self.radio_roi.toggled.connect(self._on_mode_changed)
        self.radio_bg_inner.toggled.connect(self._on_mode_changed)
        self.radio_bg_outer.toggled.connect(self._on_mode_changed)

    def _create_spinbox(self, min_val, max_val, tooltip, single_step=1):
        sb = RepeatStepSpinBox(repeat_interval_ms=self.spin_repeat_interval_ms)
        sb.setRange(min_val, max_val)
        sb.setSingleStep(max(1, int(single_step)))
        sb.setToolTip(tooltip)
        return sb

    def _create_double_spinbox(self, min_val, max_val, tooltip):
        sb = QDoubleSpinBox()
        sb.setRange(min_val, max_val)
        sb.setSingleStep(0.1)
        sb.setToolTip(tooltip)
        return sb

    def update_from_selection(self, roi: ROISelection, bg: BackgroundSelection):
        self._updating_controls = True
        self.blockSignals(True)
        self.spin_roi_cx.setValue(int(roi.center[0]))
        self.spin_roi_cy.setValue(int(roi.center[1]))
        self.spin_roi_w.setValue(int(roi.width))
        self.spin_roi_h.setValue(int(roi.height))
        self.spin_roi_a.setValue(roi.angle_deg)
        
        self.spin_bg_inner_cx.setValue(int(bg.inner_center[0]))
        self.spin_bg_inner_cy.setValue(int(bg.inner_center[1]))
        self.spin_bg_in_x.setValue(int(bg.inner_x))
        self.spin_bg_in_y.setValue(int(bg.inner_y))
        self.spin_bg_outer_cx.setValue(int(bg.outer_center[0]))
        self.spin_bg_outer_cy.setValue(int(bg.outer_center[1]))
        self.spin_bg_out_x.setValue(int(bg.outer_x))
        self.spin_bg_out_y.setValue(int(bg.outer_y))
        self.blockSignals(False)
        self._updating_controls = False

    def set_active_region(self, region: str):
        self._updating_controls = True
        if region == "roi":
            self.radio_roi.setChecked(True)
        elif region == "bg_inner":
            self.radio_bg_inner.setChecked(True)
        elif region == "bg_outer":
            self.radio_bg_outer.setChecked(True)
        self._updating_controls = False

    def _on_roi_changed(self):
        if self._updating_controls:
            return
        roi = ROISelection(
            center=(self.spin_roi_cx.value(), self.spin_roi_cy.value()),
            width=self.spin_roi_w.value(),
            height=self.spin_roi_h.value(),
            angle_deg=self.spin_roi_a.value()
        )
        self.values_changed.emit({"roi": roi})

    def _on_bg_changed(self):
        if self._updating_controls:
            return
        bg = BackgroundSelection(
            center=(self.spin_bg_outer_cx.value(), self.spin_bg_outer_cy.value()),
            inner_center=(self.spin_bg_inner_cx.value(), self.spin_bg_inner_cy.value()),
            outer_center=(self.spin_bg_outer_cx.value(), self.spin_bg_outer_cy.value()),
            inner_x=self.spin_bg_in_x.value(),
            inner_y=self.spin_bg_in_y.value(),
            outer_x=self.spin_bg_out_x.value(),
            outer_y=self.spin_bg_out_y.value(),
            angle_deg=self.spin_roi_a.value() 
        )
        self.values_changed.emit({"bg": bg})

    def _on_align_bg(self):
        self.spin_bg_inner_cx.setValue(self.spin_roi_cx.value())
        self.spin_bg_inner_cy.setValue(self.spin_roi_cy.value())
        self.spin_bg_outer_cx.setValue(self.spin_roi_cx.value())
        self.spin_bg_outer_cy.setValue(self.spin_roi_cy.value())

    def _on_mode_changed(self):
        if self._updating_controls:
            return
        if self.radio_roi.isChecked():
            self.active_region_changed.emit("roi")
        elif self.radio_bg_inner.isChecked():
            self.active_region_changed.emit("bg_inner")
        elif self.radio_bg_outer.isChecked():
            self.active_region_changed.emit("bg_outer")


class ROISelectorWindow(QMainWindow):
    finished = Signal()

    def __init__(
        self,
        raw_channels: np.ndarray,
        settings: ROISettings,
        initial_roi: Optional[ROISelection] = None,
        initial_bg: Optional[BackgroundSelection] = None,
    ):
        super().__init__()
        self.setWindowTitle("ROI Selector (PySide6)")
        
        self.preview_img = _build_preview_image(raw_channels)
        self.settings = settings
        
        # Initial State
        h, w = self.preview_img.shape[:2]
        default_roi = ROISelection(
            center=(w / 2, h / 2),
            width=float(settings.fixed_roi_width),
            height=float(settings.fixed_roi_height),
            angle_deg=float(settings.roi_angle_deg),
        )
        default_bg = BackgroundSelection(
            center=default_roi.center,
            inner_center=default_roi.center,
            outer_center=default_roi.center,
            inner_x=float(default_roi.width + 2 * settings.bg_ring_inner_x),
            inner_y=float(default_roi.height + 2 * settings.bg_ring_inner_y),
            outer_x=float(default_roi.width + 2 * settings.bg_ring_outer_x),
            outer_y=float(default_roi.height + 2 * settings.bg_ring_outer_y),
            angle_deg=float(settings.roi_angle_deg),
        )
        
        self.roi = initial_roi or default_roi
        self.bg = initial_bg or default_bg
        
        self.result: Optional[Tuple[ROISelection, BackgroundSelection]] = None

        # Components
        self.manual_tools = ManualROITools(settings)
        self.canvas = ImageWidget(self.preview_img, self.manual_tools)
        self.controls = ControlsWidget(settings)
        
        # Layouts
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.addWidget(self.canvas, stretch=1)
        
        # Dock or Side Panel
        scroll = QScrollArea()
        scroll.setWidget(self.controls)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(300)
        main_layout.addWidget(scroll, stretch=0)
        
        self.setCentralWidget(central_widget)
        self.resize(1200, 800)

        # Initial Updates
        self.canvas.set_data(self.roi, self.bg)
        self.controls.update_from_selection(self.roi, self.bg)

        # Connect Signals
        self.canvas.roi_updated.connect(self._on_canvas_interaction)
        self.controls.values_changed.connect(self._on_controls_interaction)
        self.controls.active_region_changed.connect(self._set_active_region)
        self.controls.btn_confirm.clicked.connect(self._confirm_selection)
        
        self.active_region_mode = "roi"
        self._set_active_region(self.active_region_mode)
        
        # Install Event Filter to hijack navigation keys from SpinBoxes
        self.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            key = event.key()
            # Intercept W/A/S/D/Q/E/B/Tab even if focus is on a child widget (like SpinBox)
            # We do NOT intercept Enter/Esc or Arrows here, letting child widgets handle them if they want
            # (e.g. SpinBox uses arrows), OR keyPressEvent fallback will handle them.
            intercept_keys = [
                Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, 
                Qt.Key_Q, Qt.Key_E, Qt.Key_B, Qt.Key_Tab
            ]
            if key in intercept_keys:
                # Dispatch manually to our handler logic
                self._handle_global_key(event)
                return True # Event handled, do not propagate to child
        return super().eventFilter(obj, event)

    def _on_canvas_interaction(self, data):
        if "roi" in data:
            self.roi = data["roi"]
        if "bg" in data:
            self.bg = data["bg"]
        self.bg.angle_deg = self.roi.angle_deg

        self.canvas.set_data(self.roi, self.bg)
        self.controls.update_from_selection(self.roi, self.bg)

    def _on_controls_interaction(self, data):
        if "roi" in data:
            new_roi = data["roi"]
            self.roi = new_roi
            self.bg = BackgroundSelection(
                center=self.bg.outer_center,
                inner_center=self.bg.inner_center,
                outer_center=self.bg.outer_center,
                inner_x=self.bg.inner_x,
                inner_y=self.bg.inner_y,
                outer_x=self.bg.outer_x,
                outer_y=self.bg.outer_y,
                angle_deg=self.roi.angle_deg
            )
            
        if "bg" in data:
            self.bg = data["bg"]
            
        self.canvas.set_data(self.roi, self.bg)
        self.controls.update_from_selection(self.roi, self.bg)

    def _set_active_region(self, region: str):
        if region not in {"roi", "bg_inner", "bg_outer"}:
            return
        self.active_region_mode = region
        self.canvas.set_active_region(region)
        self.controls.set_active_region(region)
        if self.statusBar():
            self.statusBar().showMessage(f"Active Region: {region.upper()}", 2000)

    def _confirm_selection(self):
        self.result = (self.roi, self.bg)
        self.close()

    def closeEvent(self, event):
        self.finished.emit()
        super().closeEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        # Fallback for keys not handled by eventFilter or child widgets
        self._handle_global_key(event)
        
    def _handle_global_key(self, event: QKeyEvent):
        key = event.key()
        modifiers = event.modifiers()
        
        if key == Qt.Key_Enter or key == Qt.Key_Return:
            self._confirm_selection()
            return
        elif key == Qt.Key_Escape:
            self.close()
            return
        elif key == Qt.Key_Tab:
            order = ["roi", "bg_inner", "bg_outer"]
            next_idx = (order.index(self.active_region_mode) + 1) % len(order)
            self._set_active_region(order[next_idx])
            return

        # --- Movement (WASD) ---
        # Default speed = move_step. Shift = 1 pixel (fine control).
        move_step = 1.0 if (modifiers & Qt.ShiftModifier) else float(self.settings.move_step)
        
        dx, dy = 0.0, 0.0
        if key == Qt.Key_W: dy = -move_step
        elif key == Qt.Key_S: dy = move_step
        elif key == Qt.Key_A: dx = -move_step
        elif key == Qt.Key_D: dx = move_step
        
        if dx != 0 or dy != 0:
            if self.active_region_mode == "roi":
                new_center = (self.roi.center[0] + dx, self.roi.center[1] + dy)
                self.roi = ROISelection(
                    center=new_center,
                    width=self.roi.width,
                    height=self.roi.height,
                    angle_deg=self.roi.angle_deg
                )
            elif self.active_region_mode == "bg_inner":
                new_inner_center = (self.bg.inner_center[0] + dx, self.bg.inner_center[1] + dy)
                self.bg = BackgroundSelection(
                    center=self.bg.outer_center,
                    inner_center=new_inner_center,
                    outer_center=self.bg.outer_center,
                    inner_x=self.bg.inner_x,
                    inner_y=self.bg.inner_y,
                    outer_x=self.bg.outer_x,
                    outer_y=self.bg.outer_y,
                    angle_deg=self.bg.angle_deg,
                )
            else:
                new_outer_center = (self.bg.outer_center[0] + dx, self.bg.outer_center[1] + dy)
                self.bg = BackgroundSelection(
                    center=new_outer_center,
                    inner_center=self.bg.inner_center,
                    outer_center=new_outer_center,
                    inner_x=self.bg.inner_x,
                    inner_y=self.bg.inner_y,
                    outer_x=self.bg.outer_x,
                    outer_y=self.bg.outer_y,
                    angle_deg=self.bg.angle_deg
                )
            self.canvas.set_data(self.roi, self.bg)
            self.controls.update_from_selection(self.roi, self.bg)
            return

        # --- Rotation (E/Q) ---
        # Default step = angle_step_deg.
        angle_step = float(self.settings.angle_step_deg)
        d_angle = 0.0
        if key == Qt.Key_E: d_angle = angle_step
        elif key == Qt.Key_Q: d_angle = -angle_step
        
        if d_angle != 0:
            new_angle = (self.roi.angle_deg + d_angle) % 360
            self.roi.angle_deg = new_angle
            self.bg.angle_deg = new_angle
            self.canvas.set_data(self.roi, self.bg)
            self.controls.update_from_selection(self.roi, self.bg)
            return

        size_step = 1.0 if (modifiers & Qt.ShiftModifier) else float(self.settings.size_step)
        
        d_w, d_h = 0.0, 0.0
        if key == Qt.Key_Up:
            d_h = -size_step
        elif key == Qt.Key_Down:
            d_h = size_step
        elif key == Qt.Key_Left:
            d_w = -size_step
        elif key == Qt.Key_Right:
            d_w = size_step

        if d_w != 0 or d_h != 0:
            if self.active_region_mode == "roi":
                self.roi.width = max(10, self.roi.width + d_w)
                self.roi.height = max(10, self.roi.height + d_h)
            elif self.active_region_mode == "bg_inner":
                self.bg.inner_x = max(5, self.bg.inner_x + d_w)
                self.bg.inner_y = max(5, self.bg.inner_y + d_h)
            elif self.active_region_mode == "bg_outer":
                self.bg.outer_x = max(5, self.bg.outer_x + d_w)
                self.bg.outer_y = max(5, self.bg.outer_y + d_h)
            self.canvas.set_data(self.roi, self.bg)
            self.controls.update_from_selection(self.roi, self.bg)
            return

        # --- Misc (B) ---
        if key == Qt.Key_B:
            self.bg = BackgroundSelection(
                center=self.roi.center,
                inner_center=self.roi.center,
                outer_center=self.roi.center,
                inner_x=self.bg.inner_x,
                inner_y=self.bg.inner_y,
                outer_x=self.bg.outer_x,
                outer_y=self.bg.outer_y,
                angle_deg=self.bg.angle_deg,
            )
            self.canvas.set_data(self.roi, self.bg)
            self.controls.update_from_selection(self.roi, self.bg)


def select_roi_with_gui(
    raw_channels: np.ndarray,
    settings: ROISettings,
    initial_roi: Optional[ROISelection] = None,
    initial_bg: Optional[BackgroundSelection] = None,
) -> Tuple[ROISelection, BackgroundSelection]:
    """Helper to launch the selector and return user choices."""
    
    # Check if QApplication already exists
    app = QApplication.instance()
    own_app = False
    if not app:
        app = QApplication(sys.argv)
        own_app = True
        
    window = ROISelectorWindow(
        raw_channels=raw_channels,
        settings=settings,
        initial_roi=initial_roi,
        initial_bg=initial_bg,
    )
    window.show()
    
    # Run the event loop
    if own_app:
        app.exec()
    else:
        loop = QEventLoop()
        window.finished.connect(loop.quit)
        loop.exec()
    
    if window.result:
        return window.result
    else:
        # If user closed window without confirming, maybe raise error or return defaults?
        raise RuntimeError("ROI selection cancelled by user")

if __name__ == "__main__":
    print("This module is intended to be imported.")
