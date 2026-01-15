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
    bg_ring_inner_h: float = 0.0  # 水平内边距
    bg_ring_outer_h: float = 0.0  # 水平外边距
    bg_ring_inner_v: float = 0.0  # 垂直内边距
    bg_ring_outer_v: float = 0.0  # 垂直外边距
    bg_center: Tuple[float, float] = None
    
    # 向后兼容属性：如果只设置了单一值，使用水平值
    @property
    def bg_ring_inner(self) -> float:
        return self.bg_ring_inner_h
    
    @property
    def bg_ring_outer(self) -> float:
        return self.bg_ring_outer_h


class ROISelector:
    """Simple OpenCV GUI for drawing a rectangular ROI."""

    def __init__(self, settings: ROISettings):
        self.settings = settings
        self.move_step = 1  # 移动步长（1像素）

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

    def _adjust_selection(self, preview_bgr: np.ndarray, bbox, filepath: Path, scale: float) -> Tuple[float, float, float, float, float, float, float, float, float, float, float]:
        """Allow tweaking ROI center/size/angle and independent BG ring center/size after initial drag selection.
        Returns: (cx_roi, cy_roi, width, height, angle, ring_inner_h, ring_outer_h, ring_inner_v, ring_outer_v, cx_bg, cy_bg)
        """
        x, y, w, h = bbox
        cx_roi, cy_roi = x + w / 2.0, y + h / 2.0
        cx_bg, cy_bg = cx_roi, cy_roi
        base_w, base_h = float(w), float(h)
        ring_inner_h = float(self.settings.bg_ring_inner * scale)  # 水平内边距
        ring_outer_h = float(self.settings.bg_ring_outer * scale)  # 水平外边距
        ring_inner_v = float(self.settings.bg_ring_inner * scale)  # 垂直内边距
        ring_outer_v = float(self.settings.bg_ring_outer * scale)  # 垂直外边距
        offset_limit = max(preview_bgr.shape)
        
        # 键盘控制模式：0=ROI, 1=BG
        control_mode = 0  # 默认控制ROI

        # 创建主显示窗口和左侧控制窗口
        window_title = f"Adjust ROI - {filepath.name}"
        control_window = "Controls"
        
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.namedWindow(control_window, cv2.WINDOW_NORMAL)
        
        cv2.resizeWindow(window_title, min(1200, preview_bgr.shape[1]), min(900, preview_bgr.shape[0]))
        cv2.resizeWindow(control_window, 320, min(900, preview_bgr.shape[0]))
        
        # 移动控制窗口到左侧
        cv2.moveWindow(control_window, 50, 50)
        cv2.moveWindow(window_title, 400, 50)

        # 在控制窗口创建所有 trackbars
        max_ring = max(preview_bgr.shape) * 2
        cv2.createTrackbar("Angle (-90~90)", control_window, 90, 180, lambda v: None)
        cv2.createTrackbar("Width %", control_window, 100, 300, lambda v: None)
        cv2.createTrackbar("Height %", control_window, 100, 300, lambda v: None)
        cv2.createTrackbar("BG inner H", control_window, int(ring_inner_h), max_ring, lambda v: None)
        cv2.createTrackbar("BG outer H", control_window, int(ring_outer_h), max_ring, lambda v: None)
        cv2.createTrackbar("BG inner V", control_window, int(ring_inner_v), max_ring, lambda v: None)
        cv2.createTrackbar("BG outer V", control_window, int(ring_outer_v), max_ring, lambda v: None)
        cv2.createTrackbar("ROI X", control_window, offset_limit, offset_limit * 2, lambda v: None)
        cv2.createTrackbar("ROI Y", control_window, offset_limit, offset_limit * 2, lambda v: None)
        cv2.createTrackbar("BG X", control_window, offset_limit, offset_limit * 2, lambda v: None)
        cv2.createTrackbar("BG Y", control_window, offset_limit, offset_limit * 2, lambda v: None)
        
        # 在控制窗口显示说明（使用更大更清晰的文本）
        control_panel = np.ones((900, 320, 3), dtype=np.uint8) * 50  # 稍亮的背景
        y = 320
        instructions = [
            "=== KEYBOARD ===",
            "",
            "Arrow / WASD:",
            "  Move (1px)",
            "",
            "Tab:",
            "  Switch ROI/BG",
            "",
            "Enter:",
            "  Confirm",
            "",
            "Esc / C:",
            "  Cancel",
        ]
        for i, text in enumerate(instructions):
            if text == "":
                continue
            if "===" in text:
                color = (100, 255, 255)  # 黄色标题
                thickness = 2
                font_scale = 0.6
            elif text.startswith("  "):
                color = (200, 200, 200)  # 浅灰色说明
                thickness = 1
                font_scale = 0.5
            else:
                color = (255, 255, 255)  # 白色键名
                thickness = 2
                font_scale = 0.55
            cv2.putText(control_panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            y += 28 if "===" in text else 24
        cv2.imshow(control_window, control_panel)

        while True:
            angle = cv2.getTrackbarPos("Angle (-90~90)", control_window) - 90
            scale_w = max(10, cv2.getTrackbarPos("Width %", control_window)) / 100.0
            scale_h = max(10, cv2.getTrackbarPos("Height %", control_window)) / 100.0

            adj_w = base_w * scale_w
            adj_h = base_h * scale_h

            # centers via sliders (offset around initial center)
            roi_dx = cv2.getTrackbarPos("ROI X", control_window) - offset_limit
            roi_dy = cv2.getTrackbarPos("ROI Y", control_window) - offset_limit
            bg_dx = cv2.getTrackbarPos("BG X", control_window) - offset_limit
            bg_dy = cv2.getTrackbarPos("BG Y", control_window) - offset_limit
            cx_roi = float(np.clip(x + w / 2.0 + roi_dx, 0, preview_bgr.shape[1] - 1))
            cy_roi = float(np.clip(y + h / 2.0 + roi_dy, 0, preview_bgr.shape[0] - 1))
            cx_bg = float(np.clip(x + w / 2.0 + bg_dx, 0, preview_bgr.shape[1] - 1))
            cy_bg = float(np.clip(y + h / 2.0 + bg_dy, 0, preview_bgr.shape[0] - 1))

            canvas = preview_bgr.copy()
            self._draw_rotated_rect(canvas, (cx_roi, cy_roi), (adj_w, adj_h), angle, color=(0, 255, 0))
            cv2.drawMarker(canvas, (int(round(cx_roi)), int(round(cy_roi))), (0, 180, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

            # Draw background ring preview using current inner/outer (独立的水平和垂直边距)
            inner_h = cv2.getTrackbarPos("BG inner H", control_window)
            outer_h = cv2.getTrackbarPos("BG outer H", control_window)
            inner_v = cv2.getTrackbarPos("BG inner V", control_window)
            outer_v = cv2.getTrackbarPos("BG outer V", control_window)
            inner_h = max(0, min(inner_h, max_ring))
            outer_h = max(inner_h + 1, min(outer_h, max_ring))
            inner_v = max(0, min(inner_v, max_ring))
            outer_v = max(inner_v + 1, min(outer_v, max_ring))
            ring_inner_h = float(inner_h)
            ring_outer_h = float(outer_h)
            ring_inner_v = float(inner_v)
            ring_outer_v = float(outer_v)

            ring_outer_size = (adj_w + 2 * ring_outer_h, adj_h + 2 * ring_outer_v)
            ring_inner_size = (adj_w + 2 * ring_inner_h, adj_h + 2 * ring_inner_v)
            ring_canvas = canvas.copy()
            self._draw_rotated_rect(ring_canvas, (cx_bg, cy_bg), ring_outer_size, angle, color=(255, 200, 0))
            self._draw_rotated_rect(ring_canvas, (cx_bg, cy_bg), ring_inner_size, angle, color=(255, 255, 0))
            cv2.drawMarker(ring_canvas, (int(round(cx_bg)), int(round(cy_bg))), (255, 140, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)
            canvas = ring_canvas

            # 显示图像和当前模式提示
            display = canvas.copy()
            mode_text = f"Mode: [{'ROI' if control_mode == 0 else 'BG'}]"
            cv2.putText(display, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow(window_title, display)
            key = cv2.waitKeyEx(30)

            if key == -1:  # No key pressed
                continue
            if key in (13, 32):  # Enter or Space
                break
            if key in (27, ord("c"), ord("C")):
                cv2.destroyWindow(window_title)
                raise RuntimeError(f"ROI selection canceled for {filepath.name}")

            # Tab键切换控制模式
            if key == 9:  # Tab
                control_mode = 1 - control_mode
                continue
            
            # 方向键移动（1像素步长）
            step = self.move_step
            target_dx, target_dy = (roi_dx, roi_dy) if control_mode == 0 else (bg_dx, bg_dy)
            x_trackbar = "ROI X" if control_mode == 0 else "BG X"
            y_trackbar = "ROI Y" if control_mode == 0 else "BG Y"
            
            moved = False
            if key in (2424832, 65362, ord('w'), ord('W')):  # Up
                target_dy -= step
                moved = True
            elif key in (2555904, 65364, ord('s'), ord('S')):  # Down
                target_dy += step
                moved = True
            elif key in (2490368, 65361, ord('a'), ord('A')):  # Left
                target_dx -= step
                moved = True
            elif key in (2621440, 65363, ord('d'), ord('D')):  # Right
                target_dx += step
                moved = True
            
            if moved:
                if control_mode == 0:
                    roi_dx, roi_dy = target_dx, target_dy
                else:
                    bg_dx, bg_dy = target_dx, target_dy
                cv2.setTrackbarPos(x_trackbar, control_window, int(offset_limit + target_dx))
                cv2.setTrackbarPos(y_trackbar, control_window, int(offset_limit + target_dy))

            cx_roi = float(np.clip(cx_roi, 0, preview_bgr.shape[1] - 1))
            cy_roi = float(np.clip(cy_roi, 0, preview_bgr.shape[0] - 1))
            cx_bg = float(np.clip(cx_bg, 0, preview_bgr.shape[1] - 1))
            cy_bg = float(np.clip(cy_bg, 0, preview_bgr.shape[0] - 1))

        cv2.destroyWindow(window_title)
        cv2.destroyWindow(control_window)
        return cx_roi, cy_roi, adj_w, adj_h, float(angle), ring_inner_h, ring_outer_h, ring_inner_v, ring_outer_v, cx_bg, cy_bg

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

        cx, cy, adj_w, adj_h, angle, ring_inner_h, ring_outer_h, ring_inner_v, ring_outer_v, cx_bg, cy_bg = self._adjust_selection(preview_bgr, bbox, filepath, scale)

        inv_scale = 1.0 / scale
        center = (cx * inv_scale, cy * inv_scale)
        width = adj_w * inv_scale
        height = adj_h * inv_scale

        return ROISelection(
            center=center,
            width=width,
            height=height,
            angle_deg=angle,
            bg_ring_inner_h=ring_inner_h * inv_scale,
            bg_ring_outer_h=ring_outer_h * inv_scale,
            bg_ring_inner_v=ring_inner_v * inv_scale,
            bg_ring_outer_v=ring_outer_v * inv_scale,
            bg_center=(cx_bg * inv_scale, cy_bg * inv_scale),
        )
