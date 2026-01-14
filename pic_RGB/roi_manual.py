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
        exclude_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """创建空心矩形背景 ROI（与 ROI 同心、同角度）。"""
        H, W = shape
        cx, cy = center

        w = roi_width if roi_width is not None else self.settings.fixed_roi_width
        h = roi_height if roi_height is not None else self.settings.fixed_roi_height
        a = roi_angle if roi_angle is not None else self.settings.roi_angle_deg

        outer_width = w + 2 * self.settings.bg_ring_outer
        outer_height = h + 2 * self.settings.bg_ring_outer
        inner_width = w + 2 * self.settings.bg_ring_inner
        inner_height = h + 2 * self.settings.bg_ring_inner

        outer_mask = np.zeros((H, W), dtype=np.uint8)
        outer_rect = ((cx, cy), (outer_width, outer_height), a)
        outer_box = cv2.boxPoints(outer_rect)
        cv2.fillConvexPoly(outer_mask, np.int32(outer_box), 1)

        inner_mask = np.zeros((H, W), dtype=np.uint8)
        inner_rect = ((cx, cy), (inner_width, inner_height), a)
        inner_box = cv2.boxPoints(inner_rect)
        cv2.fillConvexPoly(inner_mask, np.int32(inner_box), 1)

        ring_mask = outer_mask & (~inner_mask)

        if exclude_mask is not None:
            ring_mask = ring_mask & (~exclude_mask.astype(bool))

        return ring_mask.astype(np.uint8)
