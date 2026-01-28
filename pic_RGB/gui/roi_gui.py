from __future__ import annotations

import sys
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QEvent, QEventLoop
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
    QSizePolicy,
    QScrollArea,
    QGridLayout,
)

from ..config import settings as config
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
    # Assuming channels are R, G1, G2, B or similar Bayer split.
    # Stack them to form an RGB image.
    rgb = np.stack(
        [
            raw_channels[:, :, 0],
            (raw_channels[:, :, 1] + raw_channels[:, :, 2]) / 2,
            raw_channels[:, :, 3],
        ],
        axis=-1,
    ).astype(np.float32)

    # Simple auto-level
    rgb = np.clip(rgb - np.percentile(rgb, 1), 0, None)
    scale = np.percentile(rgb, 99.5)
    scale = scale if scale > 0 else 1.0
    rgb = np.clip(rgb / scale, 0, 1)
    
    # Return as uint8, range [0, 255]
    return (rgb * 255).astype(np.uint8)


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
        self.active_region = "roi"  # "roi" or "bg"

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
        # ROI Pen
        roi_pen = QPen(Qt.red)
        roi_pen.setWidthF(2.0 / self.scale_factor)
        
        # BG Pen
        bg_pen = QPen(QColor(255, 165, 0)) # Orange
        bg_pen.setWidthF(2.0 / self.scale_factor)

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
            self.bg.center,
            self.roi.width + 2 * self.bg.outer_x,
            self.roi.height + 2 * self.bg.outer_y,
            self.bg.angle_deg,
        )
        painter.setPen(bg_pen)
        painter.drawPolygon(QPolygonF([QPointF(*p) for p in bg_outer_points]))

        # Draw BG Inner
        bg_inner_points = self.manual_tools.get_rotated_box(
            self.bg.center,
            self.roi.width + 2 * self.bg.inner_x,
            self.roi.height + 2 * self.bg.inner_y,
            self.bg.angle_deg,
        )
        painter.setPen(bg_pen)
        painter.drawPolygon(QPolygonF([QPointF(*p) for p in bg_inner_points]))
        
        # Draw BG Center
        painter.drawPoint(QPointF(*self.bg.center))

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
        # Calculate new ROI from drag
        width = abs(end.x() - start.x())
        height = abs(end.y() - start.y())
        cx = (start.x() + end.x()) / 2
        cy = (start.y() + end.y()) / 2
        
        if width < 5 or height < 5:
            return  # Ignoring accidental clicks
        
        # Update ROI
        new_roi = ROISelection(
            center=(cx, cy),
            width=width,
            height=height,
            angle_deg=self.roi.angle_deg
        )
        
        payload = {}
        if self.active_region == "roi":
            payload["roi"] = new_roi
        else:
            payload["bg_center"] = (cx, cy)

        if payload:
            self.roi_updated.emit(payload)


class ControlsWidget(QWidget):
    """Side panel for controls."""
    
    values_changed = Signal(object) # Emit full state or partial
    
    def __init__(self, settings: config.ROISettings, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # --- ROI Group ---
        self.roi_group = QGroupBox("ROI Settings")
        roi_layout = QGridLayout()
        
        self.spin_roi_cx = self._create_spinbox(0, 10000, "Center X")
        self.spin_roi_cy = self._create_spinbox(0, 10000, "Center Y")
        self.spin_roi_w = self._create_spinbox(1, 5000, "Width")
        self.spin_roi_h = self._create_spinbox(1, 5000, "Height")
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
        
        self.spin_bg_cx = self._create_spinbox(0, 10000, "BG Center X")
        self.spin_bg_cy = self._create_spinbox(0, 10000, "BG Center Y")
        
        self.spin_bg_in_x = self._create_spinbox(0, 1000, "Inner X Offset")
        self.spin_bg_in_y = self._create_spinbox(0, 1000, "Inner Y Offset")
        self.spin_bg_out_x = self._create_spinbox(0, 1000, "Outer X Offset")
        self.spin_bg_out_y = self._create_spinbox(0, 1000, "Outer Y Offset")
        self.spin_bg_angle = self._create_double_spinbox(0, 360, "BG Angle")

        bg_layout.addWidget(QLabel("Center X:"), 0, 0)
        bg_layout.addWidget(self.spin_bg_cx, 0, 1)
        bg_layout.addWidget(QLabel("Center Y:"), 1, 0)
        bg_layout.addWidget(self.spin_bg_cy, 1, 1)
        bg_layout.addWidget(QLabel("Inner X:"), 2, 0)
        bg_layout.addWidget(self.spin_bg_in_x, 2, 1)
        bg_layout.addWidget(QLabel("Inner Y:"), 3, 0)
        bg_layout.addWidget(self.spin_bg_in_y, 3, 1)
        bg_layout.addWidget(QLabel("Outer X:"), 4, 0)
        bg_layout.addWidget(self.spin_bg_out_x, 4, 1)
        bg_layout.addWidget(QLabel("Outer Y:"), 5, 0)
        bg_layout.addWidget(self.spin_bg_out_y, 5, 1)
        bg_layout.addWidget(QLabel("Angle:"), 6, 0)
        bg_layout.addWidget(self.spin_bg_angle, 6, 1)
        
        self.btn_align_bg = QPushButton("Align BG to ROI Center")
        bg_layout.addWidget(self.btn_align_bg, 7, 0, 1, 2)
        
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
        
        for spin in [self.spin_bg_cx, self.spin_bg_cy, self.spin_bg_in_x, self.spin_bg_in_y, self.spin_bg_out_x, self.spin_bg_out_y]:
            spin.valueChanged.connect(self._on_bg_changed)
        self.spin_bg_angle.valueChanged.connect(self._on_bg_changed)
            
        self.btn_align_bg.clicked.connect(self._on_align_bg)

    def _create_spinbox(self, min_val, max_val, tooltip):
        sb = QSpinBox()
        sb.setRange(min_val, max_val)
        sb.setToolTip(tooltip)
        return sb

    def _create_double_spinbox(self, min_val, max_val, tooltip):
        sb = QDoubleSpinBox()
        sb.setRange(min_val, max_val)
        sb.setSingleStep(0.1)
        sb.setToolTip(tooltip)
        return sb

    def update_from_selection(self, roi: ROISelection, bg: BackgroundSelection):
        self.blockSignals(True)
        self.spin_roi_cx.setValue(int(roi.center[0]))
        self.spin_roi_cy.setValue(int(roi.center[1]))
        self.spin_roi_w.setValue(int(roi.width))
        self.spin_roi_h.setValue(int(roi.height))
        self.spin_roi_a.setValue(roi.angle_deg)
        
        self.spin_bg_cx.setValue(int(bg.center[0]))
        self.spin_bg_cy.setValue(int(bg.center[1]))
        self.spin_bg_in_x.setValue(int(bg.inner_x))
        self.spin_bg_in_y.setValue(int(bg.inner_y))
        self.spin_bg_out_x.setValue(int(bg.outer_x))
        self.spin_bg_out_y.setValue(int(bg.outer_y))
        self.spin_bg_angle.setValue(bg.angle_deg)
        self.blockSignals(False)

    def _on_roi_changed(self):
        roi = ROISelection(
            center=(self.spin_roi_cx.value(), self.spin_roi_cy.value()),
            width=self.spin_roi_w.value(),
            height=self.spin_roi_h.value(),
            angle_deg=self.spin_roi_a.value()
        )
        self.values_changed.emit({"roi": roi})

    def _on_bg_changed(self):
        bg = BackgroundSelection(
            center=(self.spin_bg_cx.value(), self.spin_bg_cy.value()),
            inner_x=self.spin_bg_in_x.value(),
            inner_y=self.spin_bg_in_y.value(),
            outer_x=self.spin_bg_out_x.value(),
            outer_y=self.spin_bg_out_y.value(),
            angle_deg=self.spin_bg_angle.value(),
        )
        self.values_changed.emit({"bg": bg})

    def _on_align_bg(self):
        self.spin_bg_cx.setValue(self.spin_roi_cx.value())
        self.spin_bg_cy.setValue(self.spin_roi_cy.value())


class ROISelectorWindow(QMainWindow):
    finished = Signal()

    def __init__(
        self,
        raw_channels: np.ndarray,
        settings: config.ROISettings,
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
            inner_x=float(settings.bg_ring_inner_x),
            inner_y=float(settings.bg_ring_inner_y),
            outer_x=float(settings.bg_ring_outer_x),
            outer_y=float(settings.bg_ring_outer_y),
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
        self.controls.btn_confirm.clicked.connect(self._confirm_selection)
        
        self.active_region_mode = "roi" # Used for keyboard shortcuts
        self.canvas.set_active_region(self.active_region_mode)
        
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
        # Data contains new roi, and potentially new bg center
        if "roi" in data:
            self.roi = data["roi"]
        if "bg_center" in data:
            self.bg = BackgroundSelection(
                center=data["bg_center"],
                inner_x=self.bg.inner_x,
                inner_y=self.bg.inner_y,
                outer_x=self.bg.outer_x,
                outer_y=self.bg.outer_y,
                angle_deg=self.bg.angle_deg,
            )

        self.canvas.set_data(self.roi, self.bg)
        self.controls.update_from_selection(self.roi, self.bg)

    def _on_controls_interaction(self, data):
        # When user changes values in spinboxes, keep center fixed when only size changes.
        # Only update center if the user explicitly changed center coordinates.
        
        if "roi" in data:
            new_roi = data["roi"]
            # Check if only size/angle changed (center unchanged) -> preserve center
            if (abs(new_roi.center[0] - self.roi.center[0]) < 0.01 and
                abs(new_roi.center[1] - self.roi.center[1]) < 0.01):
                # User didn't change center, so keep the original center
                self.roi = ROISelection(
                    center=self.roi.center,
                    width=new_roi.width,
                    height=new_roi.height,
                    angle_deg=new_roi.angle_deg
                )
            else:
                # User changed center, so use the new center
                self.roi = new_roi
            
        if "bg" in data:
            new_bg = data["bg"]
            # Check if only size/angle changed (center unchanged) -> preserve center
            if (abs(new_bg.center[0] - self.bg.center[0]) < 0.01 and
                abs(new_bg.center[1] - self.bg.center[1]) < 0.01):
                # User didn't change center, so keep the original center
                self.bg = BackgroundSelection(
                    center=self.bg.center,
                    inner_x=new_bg.inner_x,
                    inner_y=new_bg.inner_y,
                    outer_x=new_bg.outer_x,
                    outer_y=new_bg.outer_y,
                    angle_deg=new_bg.angle_deg
                )
            else:
                # User changed center, so use the new center
                self.bg = new_bg
            
        self.canvas.set_data(self.roi, self.bg)
        # Update spinboxes to ensure they reflect current state
        self.controls.update_from_selection(self.roi, self.bg)

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
            self.active_region_mode = "bg" if self.active_region_mode == "roi" else "roi"
            self.canvas.set_active_region(self.active_region_mode)
            # Ensure statusBar() is available (QMainWindow has it)
            if self.statusBar():
                self.statusBar().showMessage(f"Active Region: {self.active_region_mode.upper()}", 2000)
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
            else:
                new_center = (self.bg.center[0] + dx, self.bg.center[1] + dy)
                self.bg = BackgroundSelection(
                    center=new_center,
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
            if self.active_region_mode == "roi":
                self.roi.angle_deg = (self.roi.angle_deg + d_angle) % 360
            else:
                self.bg.angle_deg = (self.bg.angle_deg + d_angle) % 360
            self.canvas.set_data(self.roi, self.bg)
            self.controls.update_from_selection(self.roi, self.bg)
            return

        # --- Resizing (Arrow Keys) ---
        # If BG active -> Adjust Background Ring
        # If ROI active -> Adjust ROI size (Optional extension over legacy)
        size_step = 1.0 if (modifiers & Qt.ShiftModifier) else float(self.settings.size_step)
        
        # --- Resizing (Arrow Keys) ---
        # If BG active -> Adjust Background Ring
        # If ROI active -> Adjust ROI size (Optional extension over legacy)
        size_step = 1.0 if (modifiers & Qt.ShiftModifier) else float(self.settings.size_step)
        
        if self.active_region_mode == "bg":
            d_outer_x, d_outer_y = 0.0, 0.0
            if key == Qt.Key_Up: d_outer_y = -size_step
            elif key == Qt.Key_Down: d_outer_y = size_step
            elif key == Qt.Key_Left: d_outer_x = -size_step
            elif key == Qt.Key_Right: d_outer_x = size_step
            
            if d_outer_x != 0 or d_outer_y != 0:
                new_outer_x = max(self.bg.inner_x + 1, self.bg.outer_x + d_outer_x)
                new_outer_y = max(self.bg.inner_y + 1, self.bg.outer_y + d_outer_y)
                
                # Logic to shift center to keep Bottom/Left fixed
                angle_rad = math.radians(self.bg.angle_deg)
                
                # For BG Ring Outer offset, d_outer adds to BOTH sides.
                # Total Width change is 2 * d_outer_x.
                # Center shift to keep left fixed = + (2*d_outer_x)/2 = + d_outer_x (along local X)
                # Height change is 2 * d_outer_y.
                # Center shift to keep bottom fixed = - (2*d_outer_y)/2 = - d_outer_y (along local Y)
                
                # Local Shift vector (dx_local, dy_local)
                dx_local = d_outer_x
                dy_local = -d_outer_y
                
                # Rotate to Global
                shift_x = dx_local * math.cos(angle_rad) - dy_local * math.sin(angle_rad)
                shift_y = dx_local * math.sin(angle_rad) + dy_local * math.cos(angle_rad)
                
                new_cx = self.bg.center[0] + shift_x
                new_cy = self.bg.center[1] + shift_y
                
                self.bg.outer_x = new_outer_x
                self.bg.outer_y = new_outer_y
                self.bg.center = (new_cx, new_cy)
                
                self.canvas.set_data(self.roi, self.bg)
                self.controls.update_from_selection(self.roi, self.bg)
        
        elif self.active_region_mode == "roi":
            # Extension: Allow resizing ROI with arrows
            d_w, d_h = 0.0, 0.0
            if key == Qt.Key_Up: d_h = -size_step
            elif key == Qt.Key_Down: d_h = size_step
            elif key == Qt.Key_Left: d_w = -size_step
            elif key == Qt.Key_Right: d_w = size_step
            
            if d_w != 0 or d_h != 0:
                 new_w = max(10, self.roi.width + d_w)
                 new_h = max(10, self.roi.height + d_h)
                 
                 # Logic to shift center to keep Bottom/Left fixed
                 angle_rad = math.radians(self.roi.angle_deg)
                 
                 # Width change d_w. Keep Left Edge fixed.
                 # Center moves Right by d_w / 2 (Local X)
                 # Height change d_h. Keep Bottom Edge (max Y) fixed.
                 # Center moves Up by d_h / 2 (Local Y is negative)
                 
                 dx_local = d_w / 2.0
                 dy_local = -d_h / 2.0
                 
                 # Rotate to Global (Standard Rotation Matrix for CCW visual)
                 # x' = x cos - y sin
                 # y' = x sin + y cos
                 shift_x = dx_local * math.cos(angle_rad) - dy_local * math.sin(angle_rad)
                 shift_y = dx_local * math.sin(angle_rad) + dy_local * math.cos(angle_rad)
                 
                 new_cx = self.roi.center[0] + shift_x
                 new_cy = self.roi.center[1] + shift_y
                 
                 self.roi.width = new_w
                 self.roi.height = new_h
                 self.roi.center = (new_cx, new_cy)
                 
                 self.canvas.set_data(self.roi, self.bg)
                 self.controls.update_from_selection(self.roi, self.bg)

        # --- Misc (B) ---
        if key == Qt.Key_B:
            # Check for B key to align BG center
            self.bg.center = self.roi.center
            self.canvas.set_data(self.roi, self.bg)
            self.controls.update_from_selection(self.roi, self.bg)


def select_roi_with_gui(
    raw_channels: np.ndarray,
    settings: config.ROISettings,
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
