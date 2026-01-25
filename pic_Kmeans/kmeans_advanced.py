"""
高级K-means图像分割：包含形态学操作、连通组件分析和详细统计
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rawpy
from scipy import ndimage
from sklearn.cluster import KMeans

from config import (
    ANALYZE_COLOR_CHANNELS,
    ANALYZE_CONNECTED_COMPONENTS,
    APPLY_MORPHOLOGY,
    DATA_DIR,
    FIGURE_SIZE,
    MIN_COMPONENT_AREA,
    MORPHOLOGY_KERNEL_SIZE,
    N_CLUSTERS,
    N_INIT,
    NO_AUTO_BRIGHT,
    OUTPUT_BPS,
    OUTPUT_DIR,
    RANDOM_STATE,
    SAVE_CSV_SUMMARY,
    SAVE_JSON_RESULTS,
    SAVE_MASKS,
    SAVE_VISUALIZATION,
    USE_CAMERA_WB,
    VISUALIZATION_DPI,
    USE_POSITION_WEIGHT,
    POSITION_WEIGHT_FACTOR,
    MIN_LIQUID_AREA_RATIO,
    MAX_LIQUID_AREA_RATIO,
    MAX_EDGE_DISTANCE_RATIO,
    FILL_LIQUID_HOLES,
    HOLE_FILL_KERNEL_SIZE,
    N_CLUSTERS_LEVEL1,
    N_CLUSTERS_LEVEL2,
)


class AdvancedKmeansSegmentation:
    """高级K-means分割器，包含更多分析功能"""
    
    def __init__(self):
        """初始化分割器"""
        self.n_clusters = N_CLUSTERS
        self.random_state = RANDOM_STATE
        self.n_init = N_INIT
        
    def load_dng_image(self, filepath: Path) -> Tuple[np.ndarray, Dict]:
        """
        加载DNG图像文件并提取元数据
        
        Args:
            filepath: DNG文件路径
            
        Returns:
            rgb_image: RGB图像
            metadata: 图像元数据
        """
        with rawpy.imread(str(filepath)) as raw:
            # 转换为RGB图像
            rgb = raw.postprocess(
                use_camera_wb=USE_CAMERA_WB,
                output_bps=OUTPUT_BPS,
                no_auto_bright=NO_AUTO_BRIGHT
            )
            
            # 提取元数据
            metadata = {
                'width': rgb.shape[1],
                'height': rgb.shape[0],
                'channels': rgb.shape[2] if len(rgb.shape) > 2 else 1
            }
            
        return rgb, metadata
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: 输入RGB图像
            
        Returns:
            预处理后的灰度图像
        """
        # 转换为灰度图（使用加权平均，给绿色通道更高权重）
        if len(image.shape) == 3:
            # RGB -> 灰度，考虑人眼对绿光更敏感
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # 归一化
        gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return gray_norm
    
    def _enhance_for_dim_liquid(self, image: np.ndarray) -> np.ndarray:
        """Boost dim liquids so the whole bottle is visible instead of only a highlight line."""
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        gamma = 0.8
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        enhanced = cv2.LUT(enhanced, lut)
        
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        return enhanced
    
    def _run_level1_clustering(self, image: np.ndarray, use_position_weight: bool = False):
        """First-stage clustering with optional position prior to keep dim bottles intact."""
        h, w = image.shape
        
        if use_position_weight:
            y_coords, x_coords = np.indices(image.shape)
            distances = np.sqrt((y_coords - h / 2) ** 2 + (x_coords - w / 2) ** 2)
            normalized_distances = (distances / distances.max()).reshape(-1, 1)
            intensities = image.reshape(-1, 1) / 255.0
            features = np.column_stack([
                intensities,
                normalized_distances * POSITION_WEIGHT_FACTOR
            ])
        else:
            features = image.reshape(-1, 1)
        
        kmeans = KMeans(
            n_clusters=N_CLUSTERS_LEVEL1,
            random_state=self.random_state,
            n_init=self.n_init
        )
        labels_flat = kmeans.fit_predict(features)
        labels_image = labels_flat.reshape(image.shape)
        
        centers = kmeans.cluster_centers_
        intensity_centers = centers[:, 0] * 255 if centers.shape[1] > 1 else centers.flatten()
        
        bright_region_label = int(np.argmax(intensity_centers))
        background_label = int(np.argmin(intensity_centers))
        bright_region_mask = labels_image == bright_region_label
        
        return labels_image, intensity_centers, bright_region_label, background_label, bright_region_mask, kmeans
    
    def _expand_small_liquid_mask(self, liquid_mask: np.ndarray, support_mask: np.ndarray, total_pixels: int) -> np.ndarray:
        """Grow a too-small liquid mask so dim liquids are not reduced to a thin line."""
        min_pixels = int(MIN_LIQUID_AREA_RATIO * total_pixels)
        max_pixels = int(MAX_LIQUID_AREA_RATIO * total_pixels)
        current_pixels = int(np.sum(liquid_mask > 0))
        if current_pixels >= min_pixels:
            return liquid_mask
        
        support_pixels = int(np.sum(support_mask)) if support_mask is not None else 0
        constrain_to_support = support_mask is not None and support_pixels > min_pixels
        expanded = liquid_mask.copy()
        if current_pixels == 0 and support_pixels > 0:
            expanded = (support_mask.astype(np.uint8)) * 255
        
        kernel_size = max(3, MORPHOLOGY_KERNEL_SIZE)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        iterations = 0
        while np.sum(expanded > 0) < min_pixels and iterations < 6:
            expanded = cv2.dilate(expanded, kernel, iterations=1)
            iterations += 1
            if constrain_to_support:
                expanded = cv2.bitwise_and(expanded, (support_mask.astype(np.uint8) * 255))
            if np.sum(expanded > 0) >= max_pixels:
                break
        
        return expanded
    
    def segment_kmeans_nested(self, image: np.ndarray) -> Tuple[np.ndarray, Dict, Dict]:
        """
        使用嵌套K-means进行图像分割，结合位置、形状和亮度信息
        第一层：将图像分为背景和高亮区域
        第二层：在高亮区域内再分为瓶体和高亮液体
        
        Args:
            image: 输入灰度图像
            
        Returns:
            final_labels: 最终的三分类标签图像 (0=背景, 1=瓶体, 2=高亮液体)
            kmeans_info: K-means模型信息字典
            region_stats: 区域统计信息
        """
        h, w = image.shape
        center_y, center_x = h / 2, w / 2
        total_pixels = h * w
        
        # ============ 第一层K-means: 背景 vs 高亮区域 ============
        print(f"  Stage 1: Clustering into {N_CLUSTERS_LEVEL1} regions...")
        labels_level1_image, centers_level1, bright_region_label, background_label, bright_region_mask, kmeans_level1 = \
            self._run_level1_clustering(image)
        
        print(f"    Level 1 centers: Background={float(centers_level1[background_label]):.1f}, Bright={float(centers_level1[bright_region_label]):.1f}")
        
        bright_region_pixels = np.sum(bright_region_mask)
        bright_area_ratio = bright_region_pixels / total_pixels
        print(f"    Bright region: {bright_region_pixels} pixels ({bright_area_ratio:.1%})")
        
        image_for_level2 = image
        if bright_area_ratio < MIN_LIQUID_AREA_RATIO:
            print("    Bright region too small, boosting contrast and re-clustering with position prior")
            enhanced_image = self._enhance_for_dim_liquid(image)
            labels_level1_image, centers_level1, bright_region_label, background_label, bright_region_mask, kmeans_level1 = \
                self._run_level1_clustering(enhanced_image, use_position_weight=True)
            bright_region_pixels = np.sum(bright_region_mask)
            bright_area_ratio = bright_region_pixels / total_pixels
            image_for_level2 = enhanced_image
            print(f"    After enhancement: Background={float(centers_level1[background_label]):.1f}, Bright={float(centers_level1[bright_region_label]):.1f}")
            print(f"    Bright region: {bright_region_pixels} pixels ({bright_area_ratio:.1%})")
        
        # ============ 第二层K-means: 瓶体 vs 高亮液体 (加入位置权重) ============
        print(f"  Stage 2: Clustering bright region into {N_CLUSTERS_LEVEL2} sub-regions...")
        
        # 提取高亮区域的像素值和位置
        bright_indices = np.where(bright_region_mask)
        bright_region_values = image_for_level2[bright_indices]
        
        if len(bright_region_values) < 10:
            print("    Warning: Bright region too small, skipping level 2 clustering")
            final_liquid_mask = np.zeros_like(image, dtype=np.uint8)
            final_labels = np.zeros_like(image, dtype=np.int32)
            final_labels[bright_region_mask] = 1
            
            kmeans_info = {
                'level1': kmeans_level1,
                'level2': None,
                'level1_centers': centers_level1.tolist(),
                'level2_centers': None,
                'n_clusters_level1': N_CLUSTERS_LEVEL1,
                'n_clusters_level2': N_CLUSTERS_LEVEL2
            }
        else:
            # 计算高亮区域中每个像素到图像中心的距离
            y_coords = bright_indices[0]
            x_coords = bright_indices[1]
            distances_to_center = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
            
            # 归一化距离到 [0, 1]
            max_distance = np.sqrt(center_y**2 + center_x**2)
            normalized_distances = distances_to_center / max_distance
            
            # 构造特征矩阵：[亮度, 位置权重]
            if USE_POSITION_WEIGHT:
                # 亮度归一化到 [0, 1]
                normalized_intensity = bright_region_values / 255.0
                # 结合亮度和位置，位置越靠近中心，权重越大
                features = np.column_stack([
                    normalized_intensity,
                    normalized_distances * POSITION_WEIGHT_FACTOR
                ])
                print(f"    Using position-weighted features (weight factor: {POSITION_WEIGHT_FACTOR})")
            else:
                features = bright_region_values.reshape(-1, 1)
            
            # 第二次聚类: k=N_CLUSTERS_LEVEL2
            kmeans_level2 = KMeans(
                n_clusters=N_CLUSTERS_LEVEL2,
                random_state=self.random_state,
                n_init=self.n_init
            )
            labels_level2 = kmeans_level2.fit_predict(features)
            
            # 识别瓶体和高亮液体
            # 通过比较平均亮度来识别
            cluster0_intensity = np.mean(bright_region_values[labels_level2 == 0])
            cluster1_intensity = np.mean(bright_region_values[labels_level2 == 1])
            
            if cluster0_intensity > cluster1_intensity:
                liquid_label_temp = 0
                bottle_label_temp = 1
            else:
                liquid_label_temp = 1
                bottle_label_temp = 0
            
            print(f"    Level 2 intensity: Bottle={min(cluster0_intensity, cluster1_intensity):.1f}, Liquid={max(cluster0_intensity, cluster1_intensity):.1f}")
            
            # ============ 空间后处理：基于位置和形状的优化 ============
            # 创建初始标签
            temp_labels = np.zeros_like(image, dtype=np.int32)
            temp_labels[bright_region_mask] = labels_level2 + 1  # 1=bottle or liquid, 2=liquid or bottle
            
            # 找出初始标记为液体的区域
            initial_liquid_mask = np.zeros_like(image, dtype=np.uint8)
            initial_liquid_mask[bright_region_mask] = (labels_level2 == liquid_label_temp).astype(np.uint8)
            initial_liquid_mask *= 255
            
            # 分析连通组件
            num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(
                initial_liquid_mask, connectivity=8
            )
            
            # 选择最接近图像中心的连通组件作为液体
            final_liquid_mask = np.zeros_like(image, dtype=np.uint8)
            
            valid_components = []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                cx, cy = centroids[i]
                
                # 计算组件中心到图像中心的距离
                dist_to_center = np.sqrt((cy - center_y)**2 + (cx - center_x)**2)
                dist_ratio = dist_to_center / max_distance
                
                # 计算组件离边缘的最小距离
                min_edge_dist = min(cx, cy, w - cx, h - cy)
                edge_dist_ratio = min_edge_dist / min(w, h)
                
                area_ratio = area / total_pixels
                
                # 筛选条件：
                # 1. 面积大于最小阈值
                # 2. 不要离边缘太近
                # 3. 不要离中心太远
                if (area_ratio >= MIN_LIQUID_AREA_RATIO and 
                    area_ratio <= MAX_LIQUID_AREA_RATIO and
                    edge_dist_ratio >= MAX_EDGE_DISTANCE_RATIO and
                    dist_ratio <= 0.5):  # 不要离中心太远
                    
                    valid_components.append({
                        'label': i,
                        'area': area,
                        'dist_to_center': dist_to_center,
                        'area_ratio': area_ratio
                    })
            
            if valid_components:
                # 选择最接近中心且面积最大的组件
                valid_components.sort(key=lambda x: (-x['area'], x['dist_to_center']))
                selected_component = valid_components[0]
                
                # 标记为液体
                final_liquid_mask[labels_cc == selected_component['label']] = 255
                
                print(f"    Selected liquid component: area={selected_component['area_ratio']:.2%}, dist={selected_component['dist_to_center']:.1f}px")
            else:
                # 如果没有符合条件的组件，使用原始的亮度分类
                print("    Warning: No valid liquid component found, using intensity-based classification")
                final_liquid_mask = initial_liquid_mask
            
            # ============ 填充液体内部的空洞（阴影） ============
            if FILL_LIQUID_HOLES and np.sum(final_liquid_mask > 0) > 0:
                # 使用形态学闭运算填充内部空洞
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, 
                    (HOLE_FILL_KERNEL_SIZE, HOLE_FILL_KERNEL_SIZE)
                )
                filled_mask = cv2.morphologyEx(final_liquid_mask, cv2.MORPH_CLOSE, kernel)
                
                # 使用floodFill填充内部空洞
                filled_mask_copy = filled_mask.copy()
                h_mask, w_mask = filled_mask.shape
                mask_for_flood = np.zeros((h_mask + 2, w_mask + 2), dtype=np.uint8)
                cv2.floodFill(filled_mask_copy, mask_for_flood, (0, 0), 255)
                filled_mask_inv = cv2.bitwise_not(filled_mask_copy)
                final_liquid_mask = filled_mask | filled_mask_inv
                
                print(f"    Filled holes in liquid region")
            
            final_liquid_mask = self._expand_small_liquid_mask(
                final_liquid_mask, bright_region_mask, total_pixels
            )
            
            # 构建最终标签
            final_labels = np.zeros_like(image, dtype=np.int32)
            final_labels[final_liquid_mask > 0] = 2  # 液体
            
            # 其余高亮区域为瓶体
            bottle_mask = bright_region_mask & (final_liquid_mask == 0)
            final_labels[bottle_mask] = 1  # 瓶体
            
            # 统计第二层结果
            bottle_pixels = np.sum(final_labels == 1)
            liquid_pixels = np.sum(final_labels == 2)
            print(f"    Final - Bottle: {bottle_pixels} pixels ({bottle_pixels/total_pixels:.1%})")
            print(f"    Final - Liquid: {liquid_pixels} pixels ({liquid_pixels/total_pixels:.1%})")
            
            kmeans_info = {
                'level1': kmeans_level1,
                'level2': kmeans_level2,
                'level1_centers': centers_level1.tolist(),
                'level2_centers': [cluster0_intensity, cluster1_intensity],
                'use_position_weight': USE_POSITION_WEIGHT,
                'position_weight_factor': POSITION_WEIGHT_FACTOR if USE_POSITION_WEIGHT else 0,
                'n_clusters_level1': N_CLUSTERS_LEVEL1,
                'n_clusters_level2': N_CLUSTERS_LEVEL2
            }
        
        # 区域统计
        region_stats = {
            'background_pixels': int(np.sum(final_labels == 0)),
            'bottle_pixels': int(np.sum(final_labels == 1)),
            'liquid_pixels': int(np.sum(final_labels == 2)),
            'total_pixels': int(total_pixels)
        }
        
        return final_labels, kmeans_info, region_stats
    
    def refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        使用形态学操作优化掩码
        
        Args:
            mask: 输入掩码
            
        Returns:
            优化后的掩码
        """
        if not APPLY_MORPHOLOGY:
            return mask
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE)
        )
        
        # 先闭运算（填充小孔），再开运算（去除小噪点）
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_refined = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
        
        return mask_refined
    
    def analyze_connected_components(self, mask: np.ndarray) -> Dict:
        """
        分析掩码中的连通组件
        
        Args:
            mask: 二值掩码
            
        Returns:
            连通组件统计信息
        """
        if not ANALYZE_CONNECTED_COMPONENTS:
            return {}
        
        # 寻找连通组件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        # 过滤掉背景（标签0）和小组件
        valid_components = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= MIN_COMPONENT_AREA:
                valid_components.append({
                    'label': i,
                    'area': int(area),
                    'left': int(stats[i, cv2.CC_STAT_LEFT]),
                    'top': int(stats[i, cv2.CC_STAT_TOP]),
                    'width': int(stats[i, cv2.CC_STAT_WIDTH]),
                    'height': int(stats[i, cv2.CC_STAT_HEIGHT]),
                    'centroid_x': float(centroids[i][0]),
                    'centroid_y': float(centroids[i][1])
                })
        
        return {
            'num_components': len(valid_components),
            'components': valid_components,
            'largest_component_area': max([c['area'] for c in valid_components]) if valid_components else 0
        }
    
    def calculate_detailed_statistics(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """
        计算详细的统计信息
        
        Args:
            image: 原始RGB图像
            mask: 区域掩码
            
        Returns:
            详细统计信息
        """
        masked_pixels = image[mask > 0]
        
        if len(masked_pixels) == 0:
            return self._empty_stats()
        
        stats = {
            'area': int(np.sum(mask > 0)),
            'mean_intensity': float(np.mean(masked_pixels)),
            'std_intensity': float(np.std(masked_pixels)),
            'min_intensity': float(np.min(masked_pixels)),
            'max_intensity': float(np.max(masked_pixels)),
            'median_intensity': float(np.median(masked_pixels))
        }
        
        # 分析各颜色通道
        if ANALYZE_COLOR_CHANNELS and len(image.shape) == 3:
            for i, channel in enumerate(['R', 'G', 'B']):
                channel_data = masked_pixels[:, i]
                stats[f'{channel}_mean'] = float(np.mean(channel_data))
                stats[f'{channel}_std'] = float(np.std(channel_data))
                stats[f'{channel}_min'] = float(np.min(channel_data))
                stats[f'{channel}_max'] = float(np.max(channel_data))
        
        return stats
    
    def _empty_stats(self) -> Dict:
        """返回空的统计信息"""
        stats = {
            'area': 0,
            'mean_intensity': 0.0,
            'std_intensity': 0.0,
            'min_intensity': 0.0,
            'max_intensity': 0.0,
            'median_intensity': 0.0
        }
        
        if ANALYZE_COLOR_CHANNELS:
            for channel in ['R', 'G', 'B']:
                stats[f'{channel}_mean'] = 0.0
                stats[f'{channel}_std'] = 0.0
                stats[f'{channel}_min'] = 0.0
                stats[f'{channel}_max'] = 0.0
        
        return stats
    
    def process_image(self, filepath: Path) -> Tuple[Dict, Dict]:
        """
        处理单个图像
        
        Args:
            filepath: 图像文件路径
            
        Returns:
            result: 处理结果
            vis_data: 可视化数据
        """
        print(f"处理: {filepath.name}")
        
        # 加载图像
        rgb_image, metadata = self.load_dng_image(filepath)
        
        # 预处理
        gray_image = self.preprocess_image(rgb_image)
        
        # 嵌套K-means分割
        final_labels, kmeans_info, region_stats = self.segment_kmeans_nested(gray_image)
        
        # 创建三个区域的掩码
        background_mask_raw = (final_labels == 0).astype(np.uint8) * 255
        bottle_mask_raw = (final_labels == 1).astype(np.uint8) * 255
        bright_mask_raw = (final_labels == 2).astype(np.uint8) * 255
        
        # 优化掩码
        background_mask = self.refine_mask(background_mask_raw)
        bottle_mask = self.refine_mask(bottle_mask_raw)
        bright_mask = self.refine_mask(bright_mask_raw)
        
        # 统计信息
        background_stats = self.calculate_detailed_statistics(rgb_image, background_mask)
        bottle_stats = self.calculate_detailed_statistics(rgb_image, bottle_mask)
        bright_stats = self.calculate_detailed_statistics(rgb_image, bright_mask)
        
        # 连通组件分析
        bright_components = self.analyze_connected_components(bright_mask)
        bottle_components = self.analyze_connected_components(bottle_mask)
        
        # 计算比例
        total_pixels = gray_image.shape[0] * gray_image.shape[1]
        
        # 组织结果
        result = {
            'filename': filepath.name,
            'metadata': metadata,
            'kmeans_method': 'nested',
            'level1_centers': kmeans_info['level1_centers'],
            'level2_centers': kmeans_info['level2_centers'],
            'region_stats': region_stats,
            'background_stats': background_stats,
            'bottle_stats': bottle_stats,
            'bright_region_stats': bright_stats,
            'bright_components': bright_components,
            'bottle_components': bottle_components,
            'background_ratio': background_stats['area'] / total_pixels,
            'bottle_ratio': bottle_stats['area'] / total_pixels,
            'bright_ratio': bright_stats['area'] / total_pixels,
            'intensity_contrast_bright_bg': bright_stats['mean_intensity'] - background_stats['mean_intensity'],
            'intensity_contrast_bright_bottle': bright_stats['mean_intensity'] - bottle_stats['mean_intensity']
        }
        
        # 可视化数据
        vis_data = {
            'rgb_image': rgb_image,
            'gray_image': gray_image,
            'labels': final_labels,
            'background_mask': background_mask,
            'bottle_mask': bottle_mask,
            'bright_mask': bright_mask,
            'bright_mask_raw': bright_mask_raw
        }
        
        return result, vis_data
    
    def save_visualization(self, filepath: Path, result: Dict, vis_data: Dict, output_dir: Path):
        """保存可视化结果"""
        if not SAVE_VISUALIZATION:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE)
        fig.suptitle(f'Nested K-means Segmentation: {filepath.name}', fontsize=16, fontweight='bold')
        
        # 原始图像
        axes[0, 0].imshow(vis_data['rgb_image'])
        axes[0, 0].set_title('Original RGB Image')
        axes[0, 0].axis('off')
        
        # 灰度图
        axes[0, 1].imshow(vis_data['gray_image'], cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')
        
        # 聚类标签（彩色显示3个区域）
        axes[0, 2].imshow(vis_data['labels'], cmap='tab10', vmin=0, vmax=2)
        axes[0, 2].set_title('Nested K-means Labels\n(0=BG, 1=Bottle, 2=Liquid)')
        axes[0, 2].axis('off')
        
        # 背景掩码
        axes[1, 0].imshow(vis_data['background_mask'], cmap='gray')
        axes[1, 0].set_title(f"Background ({result['background_ratio']:.1%})")
        axes[1, 0].axis('off')
        
        # 瓶体掩码
        axes[1, 1].imshow(vis_data['bottle_mask'], cmap='Blues')
        title = f"Bottle ({result['bottle_ratio']:.1%})"
        if result['bottle_components'].get('num_components'):
            title += f"\n{result['bottle_components']['num_components']} regions"
        axes[1, 1].set_title(title)
        axes[1, 1].axis('off')
        
        # 高亮液体掩码
        axes[1, 2].imshow(vis_data['bright_mask'], cmap='hot')
        title = f"Bright Liquid ({result['bright_ratio']:.1%})"
        if result['bright_components'].get('num_components'):
            title += f"\n{result['bright_components']['num_components']} regions"
        axes[1, 2].set_title(title)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        output_path = output_dir / f"{filepath.stem}_nested_kmeans.png"
        plt.savefig(output_path, dpi=VISUALIZATION_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"  Visualization saved: {output_path.name}")
    
    def save_masks(self, filepath: Path, vis_data: Dict, output_dir: Path):
        """保存掩码（可选）"""
        if not SAVE_MASKS:
            return
        
        # 保存三个区域的掩码
        bg_path = output_dir / f"{filepath.stem}_background.png"
        bottle_path = output_dir / f"{filepath.stem}_bottle.png"
        bright_path = output_dir / f"{filepath.stem}_bright.png"
        
        cv2.imwrite(str(bg_path), vis_data['background_mask'])
        cv2.imwrite(str(bottle_path), vis_data['bottle_mask'])
        cv2.imwrite(str(bright_path), vis_data['bright_mask'])
        
        print(f"  Masks saved")


def save_results(all_results: List[Dict], output_dir: Path):
    """保存所有结果"""
    
    # 保存JSON
    if SAVE_JSON_RESULTS:
        json_path = output_dir / "kmeans_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nJSON结果已保存: {json_path}")
    
    # 保存CSV汇总
    if SAVE_CSV_SUMMARY and all_results:
        csv_path = output_dir / "kmeans_summary.csv"
        
        # 提取关键字段
        fieldnames = [
            'filename',
            'background_ratio',
            'bottle_ratio',
            'bright_ratio',
            'background_mean_intensity',
            'bottle_mean_intensity',
            'bright_mean_intensity',
            'intensity_contrast_bright_bg',
            'intensity_contrast_bright_bottle',
            'bright_num_components',
            'bottle_num_components'
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in all_results:
                row = {
                    'filename': result['filename'],
                    'background_ratio': f"{result['background_ratio']:.4f}",
                    'bottle_ratio': f"{result['bottle_ratio']:.4f}",
                    'bright_ratio': f"{result['bright_ratio']:.4f}",
                    'background_mean_intensity': f"{result['background_stats']['mean_intensity']:.2f}",
                    'bottle_mean_intensity': f"{result['bottle_stats']['mean_intensity']:.2f}",
                    'bright_mean_intensity': f"{result['bright_region_stats']['mean_intensity']:.2f}",
                    'intensity_contrast_bright_bg': f"{result['intensity_contrast_bright_bg']:.2f}",
                    'intensity_contrast_bright_bottle': f"{result['intensity_contrast_bright_bottle']:.2f}",
                    'bright_num_components': result['bright_components'].get('num_components', 0),
                    'bottle_num_components': result['bottle_components'].get('num_components', 0)
                }
                writer.writerow(row)
        
        print(f"CSV汇总已保存: {csv_path}")


def main():
    """主函数"""
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 初始化分割器
    segmenter = AdvancedKmeansSegmentation()
    
    # 获取所有DNG文件
    dng_files = sorted(DATA_DIR.glob("*.dng"))
    
    if not dng_files:
        print(f"在 {DATA_DIR} 中未找到DNG文件")
        return
    
    print(f"找到 {len(dng_files)} 个DNG文件")
    print("=" * 70)
    
    # 处理所有图像
    all_results = []
    
    for filepath in dng_files:
        try:
            result, vis_data = segmenter.process_image(filepath)
            all_results.append(result)
            
            # 保存可视化
            segmenter.save_visualization(filepath, result, vis_data, OUTPUT_DIR)
            if SAVE_MASKS:
                segmenter.save_masks(filepath, vis_data, OUTPUT_DIR)
            
            # 打印关键信息
            print(f"  BG: {result['background_ratio']:.1%}, Bottle: {result['bottle_ratio']:.1%}, Liquid: {result['bright_ratio']:.1%}")
            print(f"  Intensities - BG: {result['background_stats']['mean_intensity']:.0f}, Bottle: {result['bottle_stats']['mean_intensity']:.0f}, Liquid: {result['bright_region_stats']['mean_intensity']:.0f}")
            print("-" * 70)
            
        except Exception as e:
            print(f"  处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存汇总结果
    if all_results:
        save_results(all_results, OUTPUT_DIR)
        print(f"\n处理完成！共处理 {len(all_results)}/{len(dng_files)} 个图像")
    else:
        print("\n没有成功处理任何图像")


if __name__ == "__main__":
    main()
