from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
import rawpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


RAW_CHANNELS = ("R", "G1", "G2", "B")
EPS = 1e-6
CLIP_MIN = 0.1
DEFAULT_EXPOSURE_TIME = 1.0


def resource_path(relative: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative


@dataclass
class RectSelection:
    center: Tuple[float, float]
    width: float
    height: float
    angle_deg: float = 0.0


@dataclass
class SelectionState:
    roi: RectSelection
    bg_inner: RectSelection
    bg_outer: RectSelection


def rotated_box(selection: RectSelection) -> np.ndarray:
    rect = (
        (float(selection.center[0]), float(selection.center[1])),
        (float(selection.width), float(selection.height)),
        float(selection.angle_deg),
    )
    return cv2.boxPoints(rect)


def create_box_mask(shape: Tuple[int, int], selection: RectSelection) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(np.round(rotated_box(selection))), 1)
    return mask


def create_bg_mask(shape: Tuple[int, int], state: SelectionState) -> np.ndarray:
    roi_mask = create_box_mask(shape, state.roi)
    outer = create_box_mask(shape, state.bg_outer)
    inner = create_box_mask(shape, state.bg_inner)
    return (outer.astype(bool) & ~inner.astype(bool) & ~roi_mask.astype(bool)).astype(np.uint8)


def build_preview(raw_channels: np.ndarray) -> np.ndarray:
    rgb = np.stack(
        [
            raw_channels[:, :, 0],
            (raw_channels[:, :, 1] + raw_channels[:, :, 2]) / 2.0,
            raw_channels[:, :, 3],
        ],
        axis=-1,
    ).astype(np.float32)
    rgb = np.clip(rgb - np.percentile(rgb, 1), 0, None)
    scale = float(np.percentile(rgb, 99.5))
    if scale <= 0:
        scale = 1.0
    return np.ascontiguousarray((np.clip(rgb / scale, 0, 1) * 255).astype(np.uint8))


def load_raw_image(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    with rawpy.imread(str(path)) as raw:
        metadata: Dict[str, Any] = {
            "filename": path.name,
            "exposure_time": None,
            "iso": None,
            "black_level": None,
            "white_level": None,
        }
        raw_meta = getattr(raw, "metadata", None)
        if raw_meta is not None:
            shutter = getattr(raw_meta, "shutter", None)
            iso = getattr(raw_meta, "iso_speed", None)
            if isinstance(shutter, (int, float)) and shutter > 0:
                metadata["exposure_time"] = float(shutter)
            if isinstance(iso, (int, float)) and iso > 0:
                metadata["iso"] = int(iso)
        try:
            metadata["white_level"] = getattr(raw, "white_level", None)
        except Exception:
            pass

        black_levels = [0.0, 0.0, 0.0, 0.0]
        try:
            bl = getattr(raw, "black_level_per_channel", None)
            if bl is not None:
                if isinstance(bl, (list, tuple, np.ndarray)) and len(bl) >= 4:
                    black_levels = [float(x) for x in bl[:4]]
                elif isinstance(bl, (int, float)):
                    black_levels = [float(bl)] * 4
                metadata["black_level"] = black_levels
        except Exception:
            pass

        raw_data = raw.raw_image.copy().astype(np.float32)
        r = np.maximum(raw_data[0::2, 0::2] - black_levels[0], 0)
        g1 = np.maximum(raw_data[0::2, 1::2] - black_levels[1], 0)
        g2 = np.maximum(raw_data[1::2, 0::2] - black_levels[2], 0)
        b = np.maximum(raw_data[1::2, 1::2] - black_levels[3], 0)
        return np.stack([r, g1, g2, b], axis=-1), metadata


def channel_stats(raw_channels: np.ndarray, mask: np.ndarray, prefix: str) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    valid = mask > 0
    if int(np.sum(valid)) == 0:
        raise ValueError(f"{prefix} mask is empty. Please adjust ROI/BG boxes.")
    for idx, name in enumerate(RAW_CHANNELS):
        values = raw_channels[:, :, idx][valid]
        stats[f"{prefix}_{name}_mean"] = float(np.mean(values))
        stats[f"{prefix}_{name}_var"] = float(np.var(values))
    return stats


def build_roi_record(filename: str, raw_channels: np.ndarray, state: SelectionState, metadata: Dict[str, Any]) -> Dict[str, Any]:
    shape = raw_channels.shape[:2]
    roi_mask = create_box_mask(shape, state.roi)
    bg_mask = create_bg_mask(shape, state)
    record: Dict[str, Any] = {
        "filename": filename,
        "exposure_time": metadata.get("exposure_time") or DEFAULT_EXPOSURE_TIME,
        "roi_center_x": state.roi.center[0],
        "roi_center_y": state.roi.center[1],
        "roi_width": state.roi.width,
        "roi_height": state.roi.height,
        "roi_angle_deg": state.roi.angle_deg,
        "bg_inner_center_x": state.bg_inner.center[0],
        "bg_inner_center_y": state.bg_inner.center[1],
        "bg_inner_width": state.bg_inner.width,
        "bg_inner_height": state.bg_inner.height,
        "bg_outer_center_x": state.bg_outer.center[0],
        "bg_outer_center_y": state.bg_outer.center[1],
        "bg_outer_width": state.bg_outer.width,
        "bg_outer_height": state.bg_outer.height,
        "bg_angle_deg": state.bg_outer.angle_deg,
    }
    record.update(channel_stats(raw_channels, roi_mask, "roi"))
    record.update(channel_stats(raw_channels, bg_mask, "bg"))
    return record


def corrected_signal(roi: float, bg: float, mode: str) -> float:
    if mode == "divide":
        return max(roi / (bg + EPS), CLIP_MIN)
    if mode == "relative":
        return max((roi - bg) / (bg + EPS), CLIP_MIN)
    return max(roi - bg, CLIP_MIN)


def build_feature_dict(record: Dict[str, Any], *, g_mode: str, use_bg: bool, bg_correction: str) -> Dict[str, float]:
    exposure = max(float(record.get("exposure_time") or DEFAULT_EXPOSURE_TIME), EPS)
    if use_bg:
        r = corrected_signal(float(record["roi_R_mean"]), float(record["bg_R_mean"]), bg_correction)
        g1 = corrected_signal(float(record["roi_G1_mean"]), float(record["bg_G1_mean"]), bg_correction)
        g2 = corrected_signal(float(record["roi_G2_mean"]), float(record["bg_G2_mean"]), bg_correction)
        b = corrected_signal(float(record["roi_B_mean"]), float(record["bg_B_mean"]), bg_correction)
        prefix = "_sig"
    else:
        r = max(float(record["roi_R_mean"]), CLIP_MIN)
        g1 = max(float(record["roi_G1_mean"]), CLIP_MIN)
        g2 = max(float(record["roi_G2_mean"]), CLIP_MIN)
        b = max(float(record["roi_B_mean"]), CLIP_MIN)
        prefix = ""

    g = (g1 + g2) / 2.0 if g_mode == "avg" else g1
    var_g = (
        (float(record.get("roi_G1_var", 0.0)) + float(record.get("roi_G2_var", 0.0))) / 2.0
        if g_mode == "avg"
        else float(record.get("roi_G1_var", 0.0))
    )

    features = {
        f"ln_R{prefix}": math.log(r + EPS),
        f"ln_G{prefix}": math.log(g + EPS),
        f"ln_B{prefix}": math.log(b + EPS),
        f"ln_R{prefix}_per_exp": math.log(r / exposure + EPS),
        f"ln_G{prefix}_per_exp": math.log(g / exposure + EPS),
        f"ln_B{prefix}_per_exp": math.log(b / exposure + EPS),
        "Ratio_RG": r / (g + EPS),
        "Delta_RG": (g - r) / (g + r + EPS),
        "Ratio_BG": b / (g + EPS),
        "Delta_GB": (b - g) / (b + g + EPS),
        "Var_G": var_g,
        "exposure_time": exposure,
        "ln_exposure": math.log(exposure + EPS),
        "R_sig": r,
        "G1_sig": g1,
        "G2_sig": g2,
        "G_sig": g,
        "B_sig": b,
    }
    return features


def inverse_exp_model(value: float, a: float, b: float, c0: float) -> float:
    if a <= 0 or b == 0:
        return math.nan
    ratio = np.clip((value - c0) / a, 1e-12, 1.0)
    return max(float(math.log(float(ratio)) / b), 0.0)


class AmortizedPosteriorNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, min_std: float) -> None:
        super().__init__()
        self.min_std = float(min_std)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.std_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        mu = self.mu_head(hidden).squeeze(-1)
        std = self.min_std + F.softplus(self.std_head(hidden).squeeze(-1))
        return mu, std


class ModelStore:
    def __init__(self) -> None:
        registry_path = resource_path("model_registry.json")
        with registry_path.open("r", encoding="utf-8") as f:
            self.registry = json.load(f)
        self._cache: Dict[Tuple[str, str], Any] = {}

    def phone_names(self) -> list[str]:
        return list(self.registry.keys())

    def display_name(self, phone_key: str) -> str:
        return str(self.registry[phone_key].get("display_name", phone_key))

    def path_for(self, phone_key: str, key: str) -> Path:
        return resource_path(str(self.registry[phone_key][key]))

    def predict_regression(self, phone_key: str, record: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = (phone_key, "regression")
        if cache_key not in self._cache:
            self._cache[cache_key] = joblib.load(self.path_for(phone_key, "regression_model"))
        bundle = self._cache[cache_key]
        model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
        g_mode = str(bundle.get("g_mode", "avg")) if isinstance(bundle, dict) else "avg"
        use_bg = bool(bundle.get("use_bg", True)) if isinstance(bundle, dict) else True
        bg_correction = str(bundle.get("bg_correction", "subtract")) if isinstance(bundle, dict) else "subtract"
        target_transform = str(bundle.get("target_transform", "identity")) if isinstance(bundle, dict) else "identity"
        target_eps = float(bundle.get("target_eps", EPS)) if isinstance(bundle, dict) else EPS
        feature_names = tuple(bundle.get("feature_names", ())) if isinstance(bundle, dict) else ()
        calibrator = bundle.get("isotonic") if isinstance(bundle, dict) else None

        features = build_feature_dict(record, g_mode=g_mode, use_bg=use_bg, bg_correction=bg_correction)
        if not feature_names:
            raise ValueError("Regression model bundle does not contain feature_names.")
        missing = [name for name in feature_names if name not in features]
        if missing:
            raise ValueError(f"Cannot build regression features: {missing}")

        x = np.asarray([[features[name] for name in feature_names]], dtype=float)
        pred_target = np.asarray(model.predict(x), dtype=float)
        if target_transform == "log":
            pred_raw = np.exp(pred_target) - target_eps
        else:
            pred_raw = pred_target
        pred_raw = np.maximum(pred_raw, 0.0)
        if calibrator is not None:
            pred = np.maximum(np.asarray(calibrator.predict(pred_raw), dtype=float), 0.0)
        else:
            pred = pred_raw
        return {
            "prediction": float(pred[0]),
            "model": "Regression",
            "feature_names": ", ".join(feature_names),
        }

    def predict_exp(self, phone_key: str, record: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = (phone_key, "exp")
        if cache_key not in self._cache:
            df = pd.read_csv(self.path_for(phone_key, "exp_params"))
            score_col = "inv_r2" if "inv_r2" in df.columns else "r2"
            if score_col in df.columns:
                df = df.copy()
                df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
                selected = df[np.isfinite(df[score_col])].sort_values(score_col, ascending=False).iloc[0]
            else:
                selected = df.iloc[0]
            self._cache[cache_key] = (df, selected)

        _, selected = self._cache[cache_key]
        channel = str(selected["channel"])
        a = float(selected["a"])
        b = float(selected["b"])
        c0 = float(selected["c"])
        features = build_feature_dict(record, g_mode="avg", use_bg=True, bg_correction="subtract")
        if channel in features:
            value = float(features[channel])
        elif channel.endswith("_per_exp") and channel[:-8] in features:
            exposure = max(float(record.get("exposure_time") or DEFAULT_EXPOSURE_TIME), EPS)
            value = float(features[channel[:-8]]) / exposure
        else:
            raise ValueError(f"Cannot build EXP channel '{channel}' from current ROI result.")
        return {
            "prediction": inverse_exp_model(value, a, b, c0),
            "model": "EXP",
            "channel": channel,
            "value": value,
        }

    def predict_gs(self, phone_key: str, record: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = (phone_key, "gs")
        if cache_key not in self._cache:
            try:
                bundle = torch.load(self.path_for(phone_key, "gs_model"), map_location="cpu", weights_only=False)
            except TypeError:
                bundle = torch.load(self.path_for(phone_key, "gs_model"), map_location="cpu")
            model = AmortizedPosteriorNet(
                input_dim=int(bundle["input_dim"]),
                hidden_dim=int(bundle["hidden_dim"]),
                dropout=float(bundle["dropout"]),
                min_std=float(bundle["min_std"]),
            )
            model.load_state_dict(bundle["state_dict"])
            model.eval()
            self._cache[cache_key] = (bundle, model)

        bundle, model = self._cache[cache_key]
        feature_names = tuple(str(name) for name in bundle["feature_names"])
        features = build_feature_dict(
            record,
            g_mode=str(bundle.get("g_mode", "avg")),
            use_bg=bool(bundle.get("use_bg", True)),
            bg_correction="subtract",
        )
        missing = [name for name in feature_names if name not in features]
        if missing:
            raise ValueError(f"Cannot build GS features: {missing}")
        x = np.asarray([[features[name] for name in feature_names]], dtype=np.float64)
        mean = np.asarray(bundle["feature_mean"], dtype=np.float64)
        std = np.asarray(bundle["feature_std"], dtype=np.float64)
        x_std = (x - mean) / std
        with torch.no_grad():
            mu_std, std_std = model(torch.as_tensor(x_std, dtype=torch.float32))
        mu_log = float(bundle["target_log_mean"]) + float(bundle["target_log_std"]) * float(mu_std.item())
        std_log = max(float(bundle["target_log_std"]) * float(std_std.item()), 1e-6)
        max_std_log = bundle.get("max_std_log")
        if max_std_log is not None and float(max_std_log) > 0:
            std_log = min(std_log, float(max_std_log))
        target_eps = float(bundle.get("target_eps", EPS))
        median = max(math.exp(mu_log) - target_eps, 0.0)
        mean_pred = max(math.exp(mu_log + 0.5 * std_log**2) - target_eps, 0.0)
        return {
            "prediction": median,
            "model": "GS posterior median",
            "posterior_mean": mean_pred,
            "posterior_log_std": std_log,
            "feature_names": ", ".join(feature_names),
        }

    def predict(self, phone_key: str, model_key: str, record: Dict[str, Any]) -> Dict[str, Any]:
        if model_key == "regression":
            return self.predict_regression(phone_key, record)
        if model_key == "gs":
            return self.predict_gs(phone_key, record)
        if model_key == "exp":
            return self.predict_exp(phone_key, record)
        raise ValueError(f"Unknown model type: {model_key}")


class ImageCanvas(QWidget):
    state_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(720, 520)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview: Optional[np.ndarray] = None
        self.pixmap: Optional[QPixmap] = None
        self.state: Optional[SelectionState] = None
        self.active_target = "roi"
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self.drag_start: Optional[QPointF] = None
        self.drag_current: Optional[QPointF] = None

    def set_image(self, preview: np.ndarray) -> None:
        self.preview = np.ascontiguousarray(preview)
        h, w = preview.shape[:2]
        qimage = QImage(self.preview.data, w, h, 3 * w, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimage)
        roi = RectSelection((w / 2, h / 2), max(w * 0.08, 40), max(h * 0.20, 80), 0.0)
        inner = RectSelection(roi.center, roi.width + 80, roi.height + 80, 0.0)
        outer = RectSelection(roi.center, roi.width + 240, roi.height + 240, 0.0)
        self.state = SelectionState(roi=roi, bg_inner=inner, bg_outer=outer)
        self.update()
        self.state_changed.emit()

    def set_active_target(self, target: str) -> None:
        self.active_target = target
        self.update()

    def update_state_from_controls(self, state: SelectionState) -> None:
        self.state = state
        self.update()

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(28, 31, 36))
        if self.pixmap is None:
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(self.rect(), Qt.AlignCenter, "Open a DNG image")
            return

        img_w = self.pixmap.width()
        img_h = self.pixmap.height()
        self.scale = min(self.width() / img_w, self.height() / img_h)
        drawn_w = img_w * self.scale
        drawn_h = img_h * self.scale
        self.offset = QPointF((self.width() - drawn_w) / 2, (self.height() - drawn_h) / 2)

        painter.translate(self.offset)
        painter.scale(self.scale, self.scale)
        painter.drawPixmap(0, 0, self.pixmap)
        if self.state is not None:
            self._draw_selection(painter)
        if self.drag_start and self.drag_current:
            pen = QPen(QColor(40, 220, 120))
            pen.setWidthF(2.0 / self.scale)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(self.drag_start, self.drag_current).normalized())

    def _draw_selection(self, painter: QPainter) -> None:
        assert self.state is not None
        specs = [
            ("roi", self.state.roi, QColor(230, 60, 60), Qt.SolidLine),
            ("bg_inner", self.state.bg_inner, QColor(255, 215, 0), Qt.DashLine),
            ("bg_outer", self.state.bg_outer, QColor(255, 140, 0), Qt.SolidLine),
        ]
        for name, selection, color, style in specs:
            pen = QPen(color)
            pen.setStyle(style)
            pen.setWidthF((3.0 if name == self.active_target else 1.8) / self.scale)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            points = [QPointF(float(x), float(y)) for x, y in rotated_box(selection)]
            painter.drawPolygon(QPolygonF(points))

    def _map_to_image(self, pos: QPointF) -> QPointF:
        return (pos - self.offset) / self.scale

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.pixmap is None or self.state is None or event.button() != Qt.LeftButton:
            return
        self.drag_start = self._map_to_image(event.position())
        self.drag_current = self.drag_start
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.drag_start is None:
            return
        self.drag_current = self._map_to_image(event.position())
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.drag_start is None or self.state is None or event.button() != Qt.LeftButton:
            return
        end = self._map_to_image(event.position())
        start = self.drag_start
        self.drag_start = None
        self.drag_current = None
        rect = QRectF(start, end).normalized()
        if rect.width() < 5 or rect.height() < 5:
            self.update()
            return
        new_rect = RectSelection(
            center=(rect.center().x(), rect.center().y()),
            width=rect.width(),
            height=rect.height(),
            angle_deg=self.state.roi.angle_deg,
        )
        if self.active_target == "roi":
            self.state.roi = new_rect
        elif self.active_target == "bg_inner":
            self.state.bg_inner = new_rect
        else:
            self.state.bg_outer = new_rect
        self.update()
        self.state_changed.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Concentration Predictor")
        self.resize(1180, 760)
        self.model_store = ModelStore()
        self.raw_channels: Optional[np.ndarray] = None
        self.metadata: Dict[str, Any] = {}
        self.image_path: Optional[Path] = None
        self._updating_controls = False

        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        self.canvas = ImageCanvas()
        main_layout.addWidget(self.canvas, stretch=1)

        side = QWidget()
        side.setMinimumWidth(360)
        side_layout = QVBoxLayout(side)
        main_layout.addWidget(side)

        self.open_button = QPushButton("Open DNG")
        self.open_button.clicked.connect(self.open_image)
        side_layout.addWidget(self.open_button)

        model_group = QGroupBox("Prediction")
        model_layout = QGridLayout(model_group)
        self.phone_combo = QComboBox()
        for key in self.model_store.phone_names():
            self.phone_combo.addItem(self.model_store.display_name(key), key)
        self.model_combo = QComboBox()
        self.model_combo.addItem("Regression", "regression")
        self.model_combo.addItem("GS posterior", "gs")
        self.model_combo.addItem("EXP", "exp")
        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict)
        model_layout.addWidget(QLabel("Phone:"), 0, 0)
        model_layout.addWidget(self.phone_combo, 0, 1)
        model_layout.addWidget(QLabel("Model:"), 1, 0)
        model_layout.addWidget(self.model_combo, 1, 1)
        model_layout.addWidget(self.predict_button, 2, 0, 1, 2)
        side_layout.addWidget(model_group)

        self._build_selection_controls(side_layout)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(180)
        side_layout.addWidget(self.result_text)
        side_layout.addStretch()

        self.canvas.state_changed.connect(self.sync_controls_from_canvas)

    def _build_selection_controls(self, parent_layout: QVBoxLayout) -> None:
        mode_group = QGroupBox("Mouse Drag Target")
        mode_layout = QVBoxLayout(mode_group)
        self.radio_roi = QRadioButton("ROI")
        self.radio_inner = QRadioButton("BG inner")
        self.radio_outer = QRadioButton("BG outer")
        self.radio_roi.setChecked(True)
        group = QButtonGroup(self)
        group.setExclusive(True)
        for button in (self.radio_roi, self.radio_inner, self.radio_outer):
            group.addButton(button)
            mode_layout.addWidget(button)
        self.radio_roi.toggled.connect(self._mode_changed)
        self.radio_inner.toggled.connect(self._mode_changed)
        self.radio_outer.toggled.connect(self._mode_changed)
        parent_layout.addWidget(mode_group)

        rect_group = QGroupBox("Boxes")
        layout = QGridLayout(rect_group)
        self.spins: Dict[str, QSpinBox | QDoubleSpinBox] = {}

        def add_spin(row: int, key: str, label: str, maximum: int = 10000) -> None:
            spin = QSpinBox()
            spin.setRange(0, maximum)
            spin.setSingleStep(5)
            spin.valueChanged.connect(self.sync_canvas_from_controls)
            self.spins[key] = spin
            layout.addWidget(QLabel(label), row, 0)
            layout.addWidget(spin, row, 1)

        def add_angle(row: int) -> None:
            spin = QDoubleSpinBox()
            spin.setRange(-180.0, 180.0)
            spin.setSingleStep(0.5)
            spin.valueChanged.connect(self.sync_canvas_from_controls)
            self.spins["angle"] = spin
            layout.addWidget(QLabel("Angle"), row, 0)
            layout.addWidget(spin, row, 1)

        labels = [
            ("roi_cx", "ROI cx"), ("roi_cy", "ROI cy"), ("roi_w", "ROI w"), ("roi_h", "ROI h"),
            ("inner_cx", "Inner cx"), ("inner_cy", "Inner cy"), ("inner_w", "Inner w"), ("inner_h", "Inner h"),
            ("outer_cx", "Outer cx"), ("outer_cy", "Outer cy"), ("outer_w", "Outer w"), ("outer_h", "Outer h"),
        ]
        for row, (key, label) in enumerate(labels):
            add_spin(row, key, label)
        add_angle(len(labels))
        self.align_bg_button = QPushButton("Align BG centers to ROI")
        self.align_bg_button.clicked.connect(self.align_bg_to_roi)
        layout.addWidget(self.align_bg_button, len(labels) + 1, 0, 1, 2)
        parent_layout.addWidget(rect_group)

    def _mode_changed(self) -> None:
        if self.radio_roi.isChecked():
            self.canvas.set_active_target("roi")
        elif self.radio_inner.isChecked():
            self.canvas.set_active_target("bg_inner")
        else:
            self.canvas.set_active_target("bg_outer")

    def open_image(self) -> None:
        path_text, _ = QFileDialog.getOpenFileName(
            self,
            "Open RAW image",
            str(Path.home()),
            "RAW images (*.dng *.DNG);;All files (*)",
        )
        if not path_text:
            return
        path = Path(path_text)
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            raw_channels, metadata = load_raw_image(path)
            preview = build_preview(raw_channels)
        except Exception as exc:
            QMessageBox.critical(self, "Open image failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.raw_channels = raw_channels
        self.metadata = metadata
        self.image_path = path
        self.canvas.set_image(preview)
        self.result_text.setPlainText(f"Loaded: {path}\nImage shape: {raw_channels.shape[1]} x {raw_channels.shape[0]}")

    def sync_controls_from_canvas(self) -> None:
        if self.canvas.state is None:
            return
        self._updating_controls = True
        s = self.canvas.state
        values = {
            "roi_cx": s.roi.center[0], "roi_cy": s.roi.center[1], "roi_w": s.roi.width, "roi_h": s.roi.height,
            "inner_cx": s.bg_inner.center[0], "inner_cy": s.bg_inner.center[1], "inner_w": s.bg_inner.width, "inner_h": s.bg_inner.height,
            "outer_cx": s.bg_outer.center[0], "outer_cy": s.bg_outer.center[1], "outer_w": s.bg_outer.width, "outer_h": s.bg_outer.height,
        }
        for key, value in values.items():
            self.spins[key].setValue(int(round(float(value))))
        self.spins["angle"].setValue(float(s.roi.angle_deg))
        self._updating_controls = False

    def sync_canvas_from_controls(self) -> None:
        if self._updating_controls or self.canvas.state is None:
            return
        angle = float(self.spins["angle"].value())
        state = SelectionState(
            roi=RectSelection(
                (float(self.spins["roi_cx"].value()), float(self.spins["roi_cy"].value())),
                float(self.spins["roi_w"].value()),
                float(self.spins["roi_h"].value()),
                angle,
            ),
            bg_inner=RectSelection(
                (float(self.spins["inner_cx"].value()), float(self.spins["inner_cy"].value())),
                float(self.spins["inner_w"].value()),
                float(self.spins["inner_h"].value()),
                angle,
            ),
            bg_outer=RectSelection(
                (float(self.spins["outer_cx"].value()), float(self.spins["outer_cy"].value())),
                float(self.spins["outer_w"].value()),
                float(self.spins["outer_h"].value()),
                angle,
            ),
        )
        self.canvas.update_state_from_controls(state)

    def align_bg_to_roi(self) -> None:
        self.spins["inner_cx"].setValue(self.spins["roi_cx"].value())
        self.spins["inner_cy"].setValue(self.spins["roi_cy"].value())
        self.spins["outer_cx"].setValue(self.spins["roi_cx"].value())
        self.spins["outer_cy"].setValue(self.spins["roi_cy"].value())

    def predict(self) -> None:
        if self.raw_channels is None or self.canvas.state is None or self.image_path is None:
            QMessageBox.warning(self, "No image", "Please open a DNG image and select ROI/BG first.")
            return
        phone_key = str(self.phone_combo.currentData())
        model_key = str(self.model_combo.currentData())
        try:
            record = build_roi_record(self.image_path.name, self.raw_channels, self.canvas.state, self.metadata)
            result = self.model_store.predict(phone_key, model_key, record)
        except Exception as exc:
            QMessageBox.critical(self, "Prediction failed", str(exc))
            return

        lines = [
            f"Image: {self.image_path.name}",
            f"Phone: {self.model_store.display_name(phone_key)}",
            f"Model: {result.get('model', model_key)}",
            f"Predicted concentration: {float(result['prediction']):.6g} uM",
            "",
            "ROI/BG means:",
            f"ROI R/G1/G2/B: {record['roi_R_mean']:.3f}, {record['roi_G1_mean']:.3f}, {record['roi_G2_mean']:.3f}, {record['roi_B_mean']:.3f}",
            f"BG  R/G1/G2/B: {record['bg_R_mean']:.3f}, {record['bg_G1_mean']:.3f}, {record['bg_G2_mean']:.3f}, {record['bg_B_mean']:.3f}",
        ]
        extra = {k: v for k, v in result.items() if k not in {"prediction", "model"}}
        if extra:
            lines.append("")
            lines.append("Model details:")
            for key, value in extra.items():
                lines.append(f"{key}: {value}")
        self.result_text.setPlainText("\n".join(lines))


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
