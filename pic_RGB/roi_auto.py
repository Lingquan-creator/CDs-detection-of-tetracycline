from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from roi_config import ROISettings


class AutoROIFinder:
    """自动模式下的荧光区域分割与 ROI 参数估计。"""

    def __init__(self, settings: ROISettings):
        self.settings = settings

    def create_intensity_map(self, raw_channels: np.ndarray) -> np.ndarray:
        """生成用于分割的强度图 Y = max(R, G1, G2, B)。"""
        Y = np.max(raw_channels, axis=-1)
        Y = cv2.medianBlur(Y.astype(np.uint16), 3).astype(np.float32)

        # 预先在强度图阶段压制外围光斑：将边缘带的值压到低位，减弱后续分割的干扰
        if self.settings.edge_exclusion_ratio > 0:
            H, W = Y.shape
            margin_h = int(H * self.settings.edge_exclusion_ratio)
            margin_w = int(W * self.settings.edge_exclusion_ratio)
            if margin_h > 0 or margin_w > 0:
                low_val = float(np.percentile(Y, 5))
                if margin_h > 0:
                    Y[:margin_h, :] = low_val
                    Y[-margin_h:, :] = low_val
                if margin_w > 0:
                    Y[:, :margin_w] = low_val
                    Y[:, -margin_w:] = low_val
        return Y

    def segment_fluorescence(self, Y: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """分割荧光区域，返回 mask 与简单的几何属性。"""
        H, W = Y.shape
        img_center_x, img_center_y = W / 2, H / 2
        max_dist = np.sqrt(img_center_x**2 + img_center_y**2)
        strict_center = self.settings.roi_mode in ["default", "auto", "advanced", "manual"]

        edge_mask = np.ones((H, W), dtype=np.uint8)
        margin_h = int(H * self.settings.edge_exclusion_ratio)
        margin_w = int(W * self.settings.edge_exclusion_ratio)
        edge_mask[:margin_h, :] = 0
        edge_mask[-margin_h:, :] = 0
        edge_mask[:, :margin_w] = 0
        edge_mask[:, -margin_w:] = 0

        Y_norm = cv2.normalize(Y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(Y_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_and(binary, binary, mask=edge_mask)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.settings.morph_kernel_size, self.settings.morph_kernel_size),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        if num_labels <= 1:
            raise ValueError("未找到荧光区域")

        areas = stats[1:, cv2.CC_STAT_AREA]
        center_region_h = H * self.settings.center_region_ratio / 2
        center_region_w = W * self.settings.center_region_ratio / 2

        if self.settings.prefer_center and len(areas) > 1:
            candidates = []
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < self.settings.min_area:
                    continue

                cx, cy = centroids[i]
                dist_to_center = np.sqrt((cx - img_center_x) ** 2 + (cy - img_center_y) ** 2)
                dist_norm = dist_to_center / max_dist

                in_center = (
                    abs(cx - img_center_x) <= center_region_w
                    and abs(cy - img_center_y) <= center_region_h
                )

                area_score = stats[i, cv2.CC_STAT_AREA] / areas.max()
                center_score = 1 - dist_norm
                total_score = (1 - self.settings.center_weight) * area_score + self.settings.center_weight * center_score

                if strict_center and dist_norm > 0.6:
                    total_score = -1  # 过远的外围高亮斑点直接过滤

                candidates.append((total_score, in_center, i))

            best_idx = None
            best_center = [c for c in candidates if c[1] and c[0] > 0]
            if best_center:
                best_idx = max(best_center, key=lambda x: x[0])[2]
            else:
                positive = [c for c in candidates if c[0] > 0]
                if positive:
                    best_idx = max(positive, key=lambda x: x[0])[2]

            if best_idx is None:
                best_idx = np.argmax(areas) + 1
        else:
            best_idx = np.argmax(areas) + 1

        if stats[best_idx, cv2.CC_STAT_AREA] < self.settings.min_area:
            raise ValueError(
                f"最佳连通域面积 {stats[best_idx, cv2.CC_STAT_AREA]} 小于阈值 {self.settings.min_area}"
            )

        mask = (labels == best_idx).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            rect = cv2.minAreaRect(contours[0])
            (cx, cy), (w, h), angle = rect
        else:
            cx, cy = centroids[best_idx]
            angle = 0.0

        props = {
            "center_x": float(cx),
            "center_y": float(cy),
            "angle_deg": float(angle),
            "bbox": (
                stats[best_idx, cv2.CC_STAT_LEFT],
                stats[best_idx, cv2.CC_STAT_TOP],
                stats[best_idx, cv2.CC_STAT_LEFT] + stats[best_idx, cv2.CC_STAT_WIDTH],
                stats[best_idx, cv2.CC_STAT_TOP] + stats[best_idx, cv2.CC_STAT_HEIGHT],
            ),
            "area": int(stats[best_idx, cv2.CC_STAT_AREA]),
        }
        return mask, props

    def detect_and_remove_cap(
        self, fluor_mask: np.ndarray, center: Tuple[float, float]
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        检测上部瓶盖并将整个瓶盖区域从 mask 中移除，同时返回新的中心。
        """
        if not self.settings.auto_remove_cap:
            return fluor_mask, center

        coords = np.column_stack(np.where(fluor_mask > 0))
        if len(coords) == 0:
            return fluor_mask, center

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        height = y_max - y_min

        if height < 50:
            return fluor_mask, center

        width_profile = []
        y_positions = []
        step = max(1, height // 20)
        for y in range(y_min, y_max, step):
            row_mask = fluor_mask[y, :]
            row_coords = np.where(row_mask > 0)[0]
            if len(row_coords) > 0:
                width_profile.append(row_coords.max() - row_coords.min())
                y_positions.append(y)

        if len(width_profile) < 5:
            return fluor_mask, center

        width_profile = np.array(width_profile)
        y_positions = np.array(y_positions)
        n = len(width_profile)
        top_idx = int(n * self.settings.cap_detection_height)
        mid_start = int(n * 0.3)
        mid_end = int(n * 0.7)

        if top_idx < 2 or mid_end - mid_start < 2:
            return fluor_mask, center

        top_width = np.mean(width_profile[:top_idx])
        mid_width = np.mean(width_profile[mid_start:mid_end])
        has_cap = top_width > mid_width * self.settings.cap_width_ratio

        if not has_cap:
            return fluor_mask, center

        cleaned = fluor_mask.copy()

        cap_boundary_y = None
        for i in range(1, min(top_idx + 5, len(width_profile))):
            if width_profile[i] < mid_width * 1.1:
                cap_boundary_y = int(y_positions[i])
                break

        if cap_boundary_y is None:
            cap_boundary_y = int(y_min + height * 0.25)

        cleaned[: cap_boundary_y + 1, :] = 0

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )
        if num_labels <= 1:
            return fluor_mask, center

        body_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cleaned = (labels == body_idx).astype(np.uint8)

        body_center = tuple(centroids[body_idx]) if centroids.shape[0] > body_idx else center
        return cleaned, (float(body_center[0]), float(body_center[1]))

    
    def auto_detect_roi(
        self,
        Y: np.ndarray,
        raw_channels: np.ndarray,
        fluor_mask: np.ndarray,
        center: Tuple[float, float],
    ) -> Tuple[float, float, float, Tuple[float, float], np.ndarray]:
        """
        ?????????????????????/??/???
        - ?????????minAreaRect????????
        - auto_roi_margin ??????????????
        - ???? minAreaRect ?????????????????????
        - ????????????????????
        """
        cleaned_mask, adjusted_center = self.detect_and_remove_cap(fluor_mask, center)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return 100.0, 100.0, 0.0, adjusted_center, cleaned_mask

        rect = cv2.minAreaRect(contours[0])
        (cx_rect, cy_rect), (w, h), angle = rect

        # ?? OpenCV ?????????????? 90? ??????
        angle = float(angle)

        # ??????????????
        width = float(w * self.settings.auto_roi_margin)
        height = float(h * self.settings.auto_roi_margin)

        # ?????????/??
        H_img, W_img = Y.shape
        min_size = 30
        max_size = min(H_img, W_img) * 0.8
        width = float(np.clip(width, min_size, max_size))
        height = float(np.clip(height, min_size, max_size))

        # ROI ?????????????????????????????
        final_center = adjusted_center if adjusted_center else (cx_rect, cy_rect)

        # ????????: ?????????????????? -> ????????, ???????
        original_height = h
        if original_height > 0 and height < original_height:
            reduction = original_height - height
            rad = np.deg2rad(angle)
            height_dir = np.array([-np.sin(rad), np.cos(rad)])  # ????????
            if height_dir[1] < 0:
                height_dir = -height_dir  # ?????????
            shift = (reduction / 2.0) * height_dir
            final_center = (final_center[0] + shift[0], final_center[1] + shift[1])

        return width, height, angle, final_center, cleaned_mask
