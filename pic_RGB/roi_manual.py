from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from roi_config import ROISettings


class ManualROITools:
    """固定尺寸 ROI 及背景环的生成工具。"""

    def __init__(self, settings: ROISettings):
        self.settings = settings

    def get_rotated_box(
        self,
        center: Tuple[float, float],
        width: Optional[float] = None,
        height: Optional[float] = None,
        angle: Optional[float] = None,
    ) -> np.ndarray:
        """返回旋转矩形的四个顶点。"""
        w = width if width is not None else self.settings.fixed_roi_width
        h = height if height is not None else self.settings.fixed_roi_height
        a = angle if angle is not None else self.settings.roi_angle_deg
        rect = (
            (float(center[0]), float(center[1])),
            (float(w), float(h)),
            float(a),
        )
        return cv2.boxPoints(rect)

    def create_fixed_roi(
        self,
        center: Tuple[float, float],
        shape: Tuple[int, int],
        width: Optional[float] = None,
        height: Optional[float] = None,
        angle: Optional[float] = None,
    ) -> np.ndarray:
        """创建固定尺寸、可旋转的矩形 ROI mask。"""
        H, W = shape
        mask = np.zeros((H, W), dtype=np.uint8)
        box = self.get_rotated_box(center, width, height, angle)
        box_int = np.int32(np.round(box))
        cv2.fillConvexPoly(mask, box_int, 1)
        return mask

    def create_background_roi(
        self,
        center: Tuple[float, float],
        shape: Tuple[int, int],
        roi_width: Optional[float] = None,
        roi_height: Optional[float] = None,
        roi_angle: Optional[float] = None,
        inner_offset: Optional[float] = None,
        outer_offset: Optional[float] = None,
        inner_offset_x: Optional[float] = None,
        inner_offset_y: Optional[float] = None,
        outer_offset_x: Optional[float] = None,
        outer_offset_y: Optional[float] = None,
        inner_center: Optional[Tuple[float, float]] = None,
        outer_center: Optional[Tuple[float, float]] = None,
        inner_width: Optional[float] = None,
        inner_height: Optional[float] = None,
        outer_width: Optional[float] = None,
        outer_height: Optional[float] = None,
        exclude_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """创建空心矩形背景 ROI，支持独立内外框中心与尺寸。"""
        H, W = shape

        w = roi_width if roi_width is not None else self.settings.fixed_roi_width
        h = roi_height if roi_height is not None else self.settings.fixed_roi_height
        a = roi_angle if roi_angle is not None else self.settings.roi_angle_deg

        if inner_width is None or inner_height is None:
            inner_gap_x = (
                inner_offset_x
                if inner_offset_x is not None
                else (inner_offset if inner_offset is not None else self.settings.bg_ring_inner_x)
            )
            inner_gap_y = (
                inner_offset_y
                if inner_offset_y is not None
                else (inner_offset if inner_offset is not None else self.settings.bg_ring_inner_y)
            )
            inner_width = w + 2 * inner_gap_x
            inner_height = h + 2 * inner_gap_y

        if outer_width is None or outer_height is None:
            outer_gap_x = (
                outer_offset_x
                if outer_offset_x is not None
                else (outer_offset if outer_offset is not None else self.settings.bg_ring_outer_x)
            )
            outer_gap_y = (
                outer_offset_y
                if outer_offset_y is not None
                else (outer_offset if outer_offset is not None else self.settings.bg_ring_outer_y)
            )
            outer_width = w + 2 * outer_gap_x
            outer_height = h + 2 * outer_gap_y

        inner_cx, inner_cy = inner_center if inner_center is not None else center
        outer_cx, outer_cy = outer_center if outer_center is not None else center

        outer_mask = np.zeros((H, W), dtype=np.uint8)
        outer_rect = ((outer_cx, outer_cy), (outer_width, outer_height), a)
        outer_box = cv2.boxPoints(outer_rect)
        cv2.fillConvexPoly(outer_mask, np.int32(outer_box), 1)

        inner_mask = np.zeros((H, W), dtype=np.uint8)
        inner_rect = ((inner_cx, inner_cy), (inner_width, inner_height), a)
        inner_box = cv2.boxPoints(inner_rect)
        cv2.fillConvexPoly(inner_mask, np.int32(inner_box), 1)

        ring_mask = outer_mask & (~inner_mask)

        if exclude_mask is not None:
            ring_mask = ring_mask & (~exclude_mask.astype(bool))

        return ring_mask.astype(np.uint8)
