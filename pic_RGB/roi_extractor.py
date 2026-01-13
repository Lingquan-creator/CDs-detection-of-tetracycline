"""
ROI自动识别和读取模块
用于从荧光图像中自动提取发黄白色光的高亮区域，并生成输入特征向量
"""

import numpy as np
import rawpy
from pathlib import Path
import cv2
import json
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib

# 导入配置参数
try:
    from roi_config import (
        DATA_DIR,
        OUTPUT_DIR,
        CURRENT_CONFIG,
        SAVE_VISUALIZATION,
        VISUALIZATION_DPI,
        VISUALIZATION_FIGSIZE,
        print_config,
    )
    DEFAULT_ROI_CONFIG = dict(CURRENT_CONFIG)
except ImportError:
    # 如果配置文件不存在，使用默认值
    print("警告: 未找到roi_config.py，使用默认配置")
    DATA_DIR = Path(r'h:\000\CDs\pic_RGB\data_pic')
    OUTPUT_DIR = Path(r'h:\000\CDs\pic_RGB\roi_results')
    DEFAULT_ROI_CONFIG = {
        "fixed_roi_width": 120,
        "fixed_roi_height": 400,
        "roi_angle_deg": 0.0,
        "bg_ring_inner": 80,
        "bg_ring_outer": 240,
        "min_area": 1000,
        "morph_kernel_size": 5,
        "otsu_percentile": 0.85,
    }
    SAVE_VISUALIZATION = True
    VISUALIZATION_DPI = 150
    VISUALIZATION_FIGSIZE = (15, 10)

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


@dataclass
class ROIResult:
    """ROI提取结果"""
    filename: str
    # ROI位置信息
    center_x: float
    center_y: float
    width: float
    height: float
    angle_deg: float
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    area: int
    
    # 通道统计值（背景扣除前）
    R_mean: float
    G1_mean: float
    G2_mean: float
    B_mean: float
    
    # 背景统计值
    R_bg: float
    G1_bg: float
    G2_bg: float
    B_bg: float
    
    # 背景扣除后的值
    R_corrected: float
    G1_corrected: float
    G2_corrected: float
    B_corrected: float
    
    # RGB合成值（用于显示和分析）
    R_display: float
    G_display: float
    B_display: float
    
    # 元信息
    exposure_time: Optional[float] = None
    iso: Optional[int] = None
    
    def to_dict(self):
        """转换为字典（处理numpy类型以便JSON序列化）"""
        data = asdict(self)
        # 转换numpy类型为Python原生类型
        for key, value in data.items():
            if isinstance(value, (np.integer, np.floating)):
                data[key] = value.item()
            elif isinstance(value, np.ndarray):
                data[key] = value.tolist()
        return data
    
    def get_feature_vector(self) -> np.ndarray:
        """获取输入特征向量（4通道背景校正后的值）"""
        return np.array([
            self.R_corrected,
            self.G1_corrected,
            self.G2_corrected,
            self.B_corrected
        ])
    
    def get_rgb_vector(self) -> np.ndarray:
        """获取RGB向量（用于显示）"""
        return np.array([
            self.R_display,
            self.G_display,
            self.B_display
        ])


class ROIExtractor:
    """ROI自动识别和特征提取器"""
    
    def __init__(self, 
                 roi_params: Optional[Dict[str, float]] = None,
                 fixed_roi_width: Optional[int] = None,
                 fixed_roi_height: Optional[int] = None,
                 roi_angle_deg: Optional[float] = None,
                 bg_ring_inner: Optional[int] = None,
                 bg_ring_outer: Optional[int] = None,
                 otsu_percentile: Optional[float] = None,
                 morph_kernel_size: Optional[int] = None,
                 min_area: Optional[int] = None):
        """
        参数:
            roi_params: 以字典形式传入ROI识别参数，键与roi_config.CURRENT_CONFIG一致
            fixed_roi_width: 固定ROI矩形宽度（像素），None则使用配置文件
            fixed_roi_height: 固定ROI矩形高度（像素），None则使用配置文件
            roi_angle_deg: ROI旋转角度（度，逆时针为正），None则使用配置文件
            bg_ring_inner: 背景环带内半径，None则使用配置文件
            bg_ring_outer: 背景环带外半径，None则使用配置文件
            otsu_percentile: 阈值分割的百分位数（如果不用Otsu），None则使用配置文件
            morph_kernel_size: 形态学操作的核大小，None则使用配置文件
            min_area: 最小连通域面积，None则使用配置文件
        """
        # 以配置字典为基准，支持整体或单项覆盖
        cfg = dict(DEFAULT_ROI_CONFIG)
        if roi_params:
            cfg.update({k: v for k, v in roi_params.items() if v is not None})
        
        overrides = {
            "fixed_roi_width": fixed_roi_width,
            "fixed_roi_height": fixed_roi_height,
            "roi_angle_deg": roi_angle_deg,
            "bg_ring_inner": bg_ring_inner,
            "bg_ring_outer": bg_ring_outer,
            "otsu_percentile": otsu_percentile,
            "morph_kernel_size": morph_kernel_size,
            "min_area": min_area,
        }
        for key, value in overrides.items():
            if value is not None:
                cfg[key] = value
        
        self.roi_config = cfg
        self.fixed_roi_width = cfg["fixed_roi_width"]
        self.fixed_roi_height = cfg["fixed_roi_height"]
        self.roi_angle_deg = cfg["roi_angle_deg"]
        self.bg_ring_inner = cfg["bg_ring_inner"]
        self.bg_ring_outer = cfg["bg_ring_outer"]
        self.otsu_percentile = cfg["otsu_percentile"]
        self.morph_kernel_size = cfg["morph_kernel_size"]
        self.min_area = cfg["min_area"]
        self.prefer_center = cfg.get("prefer_center", True)
        self.center_weight = cfg.get("center_weight", 0.3)
        self.center_region_ratio = cfg.get("center_region_ratio", 0.7)
        self.edge_exclusion_ratio = cfg.get("edge_exclusion_ratio", 0.15)
        self.roi_mode = cfg.get("roi_mode", "manual")
        self.auto_roi_margin = cfg.get("auto_roi_margin", 0.8)
        self.auto_color_threshold = cfg.get("auto_color_threshold", 0.6)
        self.auto_brightness_percentile = cfg.get("auto_brightness_percentile", 95)
        self.auto_remove_cap = cfg.get("auto_remove_cap", True)
        self.cap_width_ratio = cfg.get("cap_width_ratio", 1.3)
        self.cap_detection_height = cfg.get("cap_detection_height", 0.3)
        
    def load_raw_image(self, filepath: Path) -> Tuple[np.ndarray, Dict]:
        """
        加载RAW图像并线性化
        
        返回:
            raw_channels: (H, W, 4) 数组，顺序为 [R, G1, G2, B]
            metadata: 包含曝光、ISO等信息的字典
        """
        with rawpy.imread(str(filepath)) as raw:
            # 获取元信息（简化版，避免访问不存在的属性）
            metadata = {
                'exposure_time': None,  # 需要通过EXIF单独读取
                'iso': None,  # 需要通过EXIF单独读取
                'black_level': None,
                'white_level': None,
            }
            
            # 尝试获取白电平
            try:
                if hasattr(raw, 'white_level'):
                    metadata['white_level'] = raw.white_level
                elif hasattr(raw, 'camera_white_level_per_channel'):
                    metadata['white_level'] = raw.camera_white_level_per_channel
            except:
                pass
            
            # 获取RAW数据（线性，不做demosaic）
            raw_data = raw.raw_image.copy().astype(np.float32)
            
            # 黑电平扣除 - 支持多种属性形式
            black_levels = [0, 0, 0, 0]
            try:
                if hasattr(raw, 'black_level_per_channel') and raw.black_level_per_channel is not None:
                    bl = raw.black_level_per_channel
                    if isinstance(bl, (list, tuple, np.ndarray)) and len(bl) >= 4:
                        black_levels = list(bl)[:4]
                    elif isinstance(bl, (int, float)):
                        black_levels = [bl] * 4
                    metadata['black_level'] = black_levels
                elif hasattr(raw, 'camera_black_level_per_channel') and raw.camera_black_level_per_channel is not None:
                    bl = raw.camera_black_level_per_channel
                    if isinstance(bl, (list, tuple, np.ndarray)) and len(bl) >= 4:
                        black_levels = list(bl)[:4]
                    elif isinstance(bl, (int, float)):
                        black_levels = [bl] * 4
                    metadata['black_level'] = black_levels
            except Exception as e:
                print(f"  警告: 无法获取黑电平，使用默认值0: {e}")
            
            # 按Bayer模式拆分通道
            H, W = raw_data.shape
            
            # 标准RGGB模式处理
            R = raw_data[0::2, 0::2] - black_levels[0]
            G1 = raw_data[0::2, 1::2] - black_levels[1]
            G2 = raw_data[1::2, 0::2] - black_levels[2]
            B = raw_data[1::2, 1::2] - black_levels[3]
            
            # 裁剪负值
            R = np.maximum(R, 0)
            G1 = np.maximum(G1, 0)
            G2 = np.maximum(G2, 0)
            B = np.maximum(B, 0)
            
            # 堆叠为4通道图像
            raw_channels = np.stack([R, G1, G2, B], axis=-1)
            
            return raw_channels, metadata
    
    def create_intensity_map(self, raw_channels: np.ndarray) -> np.ndarray:
        """
        创建强度图用于分割
        使用 Y = max(R, G1, G2, B) 方法，对颜色变化更鲁棒
        """
        Y = np.max(raw_channels, axis=-1)
        
        # 轻微去噪（抑制热像素）
        Y = cv2.medianBlur(Y.astype(np.uint16), 3).astype(np.float32)
        
        return Y
    
    def segment_fluorescence(self, Y: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        分割荧光区域，优先选择中央区域，排除边缘
        
        返回:
            mask: 二值mask
            props: 包含质心、外接框、旋转角度等属性的字典
        """
        H, W = Y.shape
        img_center_x, img_center_y = W / 2, H / 2
        
        # 创建边缘排除mask（排除图像边缘区域）
        edge_mask = np.ones((H, W), dtype=np.uint8)
        margin_h = int(H * self.edge_exclusion_ratio)
        margin_w = int(W * self.edge_exclusion_ratio)
        edge_mask[:margin_h, :] = 0  # 上边缘
        edge_mask[-margin_h:, :] = 0  # 下边缘
        edge_mask[:, :margin_w] = 0  # 左边缘
        edge_mask[:, -margin_w:] = 0  # 右边缘
        
        # 归一化到0-255用于阈值分割
        Y_norm = cv2.normalize(Y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Otsu阈值分割
        _, binary = cv2.threshold(Y_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 应用边缘排除mask
        binary = cv2.bitwise_and(binary, binary, mask=edge_mask)
        
        # 形态学操作：开运算去噪，闭运算填洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (self.morph_kernel_size, self.morph_kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # 找到合适的连通域（优先中央+面积）
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            
            # 计算中心区域范围
            center_region_h = H * self.center_region_ratio / 2
            center_region_w = W * self.center_region_ratio / 2
            
            # 如果启用中央优先，计算综合得分
            if self.prefer_center and len(areas) > 1:
                scores = []
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] < self.min_area:
                        scores.append(-1)  # 过滤掉太小的
                        continue
                    
                    # 计算到图像中心的距离
                    cx, cy = centroids[i]
                    dist_to_center = np.sqrt((cx - img_center_x)**2 + (cy - img_center_y)**2)
                    max_dist = np.sqrt(img_center_x**2 + img_center_y**2)
                    
                    # 检查是否在中心区域内
                    in_center_region = (
                        abs(cx - img_center_x) <= center_region_w and 
                        abs(cy - img_center_y) <= center_region_h
                    )
                    
                    # 如果不在中心区域，大幅降低得分
                    if not in_center_region:
                        scores.append(-1)
                        continue
                    
                    # 综合得分：面积归一化 + 中心距离归一化（距离越近得分越高）
                    area_score = stats[i, cv2.CC_STAT_AREA] / areas.max()
                    center_score = 1 - (dist_to_center / max_dist)
                    
                    # 加权组合
                    total_score = (1 - self.center_weight) * area_score + self.center_weight * center_score
                    scores.append(total_score)
                
                scores = np.array(scores)
                if scores.max() > 0:
                    best_idx = np.argmax(scores) + 1
                else:
                    best_idx = np.argmax(areas) + 1
            else:
                # 不启用中央优先，直接选最大
                best_idx = np.argmax(areas) + 1
            
            # 过滤小区域
            if stats[best_idx, cv2.CC_STAT_AREA] < self.min_area:
                raise ValueError(f"最佳连通域面积 {stats[best_idx, cv2.CC_STAT_AREA]} 小于阈值 {self.min_area}")
            
            # 创建选中连通域的mask
            mask = (labels == best_idx).astype(np.uint8)
            
            # 使用最小外接矩形检测旋转角度
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                # 获取最小外接旋转矩形
                rect = cv2.minAreaRect(contours[0])
                (cx, cy), (w, h), angle = rect
                
                # OpenCV的角度定义：[-90, 0)，需要调整
                # 让长边作为主轴，角度范围[-90, 90]
                if w < h:
                    angle = angle - 90
            else:
                cx, cy = centroids[best_idx]
                angle = 0.0
            
            # 提取属性
            props = {
                'center_x': float(cx),
                'center_y': float(cy),
                'angle_deg': float(angle),
                'bbox': (stats[best_idx, cv2.CC_STAT_LEFT],
                        stats[best_idx, cv2.CC_STAT_TOP],
                        stats[best_idx, cv2.CC_STAT_LEFT] + stats[best_idx, cv2.CC_STAT_WIDTH],
                        stats[best_idx, cv2.CC_STAT_TOP] + stats[best_idx, cv2.CC_STAT_HEIGHT]),
                'area': int(stats[best_idx, cv2.CC_STAT_AREA])
            }
        else:
            raise ValueError("未找到荧光区域")
        
        return mask, props
    
    def detect_and_remove_cap(self, fluor_mask: np.ndarray, center: Tuple[float, float]) -> Tuple[float, float]:
        """
        检测并移除瓶盖，返回调整后的中心位置
        
        原理：
        1. 分析荧光区域沿垂直方向的宽度变化
        2. 如果上部明显比中部宽，认为是瓶盖
        3. 找到瓶盖与瓶身的分界线，调整ROI中心向下偏移
        
        参数:
            fluor_mask: 荧光区域mask
            center: 原始中心位置
        
        返回:
            (new_cx, new_cy) 调整后的中心位置
        """
        if not self.auto_remove_cap:
            return center
        
        H, W = fluor_mask.shape
        cx, cy = center
        
        # 获取荧光区域的外接矩形
        coords = np.column_stack(np.where(fluor_mask > 0))
        if len(coords) == 0:
            return center
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        height = y_max - y_min
        
        if height < 50:  # 太小不处理
            return center
        
        # 分析沿垂直方向的宽度变化
        width_profile = []
        y_positions = []
        
        for y in range(y_min, y_max, max(1, height // 20)):  # 采林20个点
            row_mask = fluor_mask[y, :]
            row_coords = np.where(row_mask > 0)[0]
            if len(row_coords) > 0:
                width = row_coords.max() - row_coords.min()
                width_profile.append(width)
                y_positions.append(y)
        
        if len(width_profile) < 5:
            return center
        
        width_profile = np.array(width_profile)
        y_positions = np.array(y_positions)
        
        # 计算上部、中部、下部的平均宽度
        n = len(width_profile)
        top_idx = int(n * self.cap_detection_height)
        mid_start = int(n * 0.3)
        mid_end = int(n * 0.7)
        
        if top_idx < 2 or mid_end - mid_start < 2:
            return center
        
        top_width = np.mean(width_profile[:top_idx])
        mid_width = np.mean(width_profile[mid_start:mid_end])
        
        # 判断是否有瓶盖：上部明显比中部宽
        has_cap = top_width > mid_width * self.cap_width_ratio
        
        if has_cap:
            # 找到宽度变化的分界点（瓶盖与瓶身的分界）
            # 从上往下搜索，找到宽度突然减小的位置
            for i in range(1, min(top_idx + 5, len(width_profile))):
                if width_profile[i] < mid_width * 1.1:  # 接近中部宽度
                    cap_boundary_y = y_positions[i]
                    
                    # 计算瓶身区域的新中心（从瓶盖分界到底部）
                    body_height = y_max - cap_boundary_y
                    new_cy = cap_boundary_y + body_height / 2
                    
                    print(f"  检测到瓶盖: 上部宽度={top_width:.1f}, 中部宽度={mid_width:.1f}, 比值={top_width/mid_width:.2f}")
                    print(f"  瓶盖分界: y={cap_boundary_y}, 调整中心: {cy:.1f} -> {new_cy:.1f} (下移{new_cy-cy:.1f}px)")
                    
                    return (cx, new_cy)
        
        return center
    
    def auto_detect_roi(self, Y: np.ndarray, raw_channels: np.ndarray, 
                        fluor_mask: np.ndarray, center: Tuple[float, float]) -> Tuple[float, float, float, Tuple[float, float]]:
        """
        自动检测ROI尺寸和角度，基于形状、亮度和颜色特征
        
        参数:
            Y: 强度图
            raw_channels: 原始4通道数据
            fluor_mask: 荧光区域mask
            center: 荧光区域中心
        
        返回:
            (width, height, angle_deg, adjusted_center) ROI的宽度、高度、旋转角度和调整后的中心
        """
        # 0. 检测并移除瓶盖，获取调整后的中心
        adjusted_center = self.detect_and_remove_cap(fluor_mask, center)
        cx, cy = adjusted_center
        
        # 1. 获取最小外接矩形作为基准
        contours, _ = cv2.findContours(fluor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            # 无法检测，返回默认值
            return 100.0, 100.0, 0.0
        
        rect = cv2.minAreaRect(contours[0])
        (cx, cy), (w, h), angle = rect
        
        # 2. 调整尺寸：收缩一定比例避免边缘噪声
        width = w * self.auto_roi_margin
        height = h * self.auto_roi_margin
        
        # 3. 形状优化：分析荧光区域的形状特征
        # 计算长宽比，判断是否为细长条状
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        
        # 如果长宽比过大（细长条），可能需要调整
        if aspect_ratio > 3:
            # 保持长边，适当增加短边
            if w > h:
                height = min(height * 1.5, width * 0.5)
            else:
                width = min(width * 1.5, height * 0.5)
        
        # 4. 亮度分析：确保ROI覆盖高亮核心区域
        # 创建临时ROI mask（使用调整后的中心）
        H_img, W_img = Y.shape
        temp_mask = np.zeros((H_img, W_img), dtype=np.uint8)
        temp_rect = ((cx, cy), (width, height), angle)
        temp_box = cv2.boxPoints(temp_rect)
        cv2.fillConvexPoly(temp_mask, np.int32(temp_box), 1)
        
        # 计算ROI内的平均亮度
        roi_brightness = Y[temp_mask > 0].mean() if np.any(temp_mask) else 0
        overall_brightness = Y[fluor_mask > 0].mean() if np.any(fluor_mask) else 0
        
        # 如果ROI亮度明显低于整体，说明可能太小，适当放大
        if roi_brightness < overall_brightness * 0.8:
            width *= 1.2
            height *= 1.2
        
        # 5. 颜色分析：判断是否为黄白色荧光
        R = raw_channels[:, :, 0]
        G = (raw_channels[:, :, 1] + raw_channels[:, :, 2]) / 2
        B = raw_channels[:, :, 3]
        
        # 在荧光区域内计算颜色特征
        if np.any(fluor_mask):
            R_mean = R[fluor_mask > 0].mean()
            G_mean = G[fluor_mask > 0].mean()
            B_mean = B[fluor_mask > 0].mean()
            
            # 黄白色判定：R+G 应该明显大于 B
            color_ratio = (R_mean + G_mean) / (B_mean + 1e-6)
            
            # 如果颜色偏蓝，可能不是目标荧光，缩小ROI
            if color_ratio < self.auto_color_threshold:
                width *= 0.9
                height *= 0.9
        
        # 6. 尺寸限制：避免过大或过小
        min_size = 30
        max_size = min(H_img, W_img) * 0.8
        width = np.clip(width, min_size, max_size)
        height = np.clip(height, min_size, max_size)
        
        # 7. 角度规范化：[-90, 90]
        if w < h:
            angle = angle - 90
        
        print(f"  自动检测ROI: {width:.1f}x{height:.1f}px, 角度={angle:.1f}°")
        print(f"    形状比={aspect_ratio:.2f}, 亮度比={roi_brightness/overall_brightness:.2f}, 颜色比={color_ratio:.2f}")
        
        return float(width), float(height), float(angle), adjusted_center
    
    def _get_rotated_box(self, center: Tuple[float, float], 
                         width: Optional[float] = None, 
                         height: Optional[float] = None,
                         angle: Optional[float] = None) -> np.ndarray:
        """获取旋转矩形四个顶点（float），用于绘制或填充"""
        w = width if width is not None else self.fixed_roi_width
        h = height if height is not None else self.fixed_roi_height
        a = angle if angle is not None else self.roi_angle_deg
        rect = (
            (float(center[0]), float(center[1])),
            (float(w), float(h)),
            float(a)
        )
        return cv2.boxPoints(rect)  # (4, 2)

    def create_fixed_roi(self, center: Tuple[float, float], shape: Tuple[int, int],
                        width: Optional[float] = None,
                        height: Optional[float] = None,
                        angle: Optional[float] = None) -> np.ndarray:
        """
        创建固定尺寸且可旋转的矩形 ROI mask
        
        参数:
            center: (x, y) 中心坐标
            shape: (H, W) 图像尺寸
            width: ROI宽度（None则使用配置值）
            height: ROI高度（None则使用配置值）
            angle: 旋转角度（None则使用配置值）
        """
        H, W = shape
        mask = np.zeros((H, W), dtype=np.uint8)
        
        box = self._get_rotated_box(center, width, height, angle)
        box_int = np.int32(np.round(box))
        cv2.fillConvexPoly(mask, box_int, 1)
        
        return mask
    
    def create_background_roi(self, center: Tuple[float, float], 
                             shape: Tuple[int, int],
                             roi_width: Optional[float] = None,
                             roi_height: Optional[float] = None,
                             roi_angle: Optional[float] = None,
                             exclude_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        创建空心矩形背景ROI，长宽比与ROI一致
        
        参数:
            center: (x, y) 中心坐标
            shape: (H, W) 图像尺寸
            roi_width: ROI宽度（None则使用配置值）
            roi_height: ROI高度（None则使用配置值）
            roi_angle: ROI角度（None则使用配置值）
            exclude_mask: 需要排除的区域（如荧光粗mask）
        """
        H, W = shape
        cx, cy = center
        
        # 使用实际ROI参数或配置值
        w = roi_width if roi_width is not None else self.fixed_roi_width
        h = roi_height if roi_height is not None else self.fixed_roi_height
        a = roi_angle if roi_angle is not None else self.roi_angle_deg
        
        # 计算长宽比
        aspect_ratio = h / w if w > 0 else 1.0
        
        # 创建外矩形（扩大bg_ring_outer）
        outer_width = w + 2 * self.bg_ring_outer
        outer_height = h + 2 * self.bg_ring_outer
        
        # 创建内矩形（扩大bg_ring_inner）
        inner_width = w + 2 * self.bg_ring_inner
        inner_height = h + 2 * self.bg_ring_inner
        
        # 创建外矩形mask
        outer_mask = np.zeros((H, W), dtype=np.uint8)
        outer_rect = ((cx, cy), (outer_width, outer_height), a)
        outer_box = cv2.boxPoints(outer_rect)
        cv2.fillConvexPoly(outer_mask, np.int32(outer_box), 1)
        
        # 创建内矩形mask
        inner_mask = np.zeros((H, W), dtype=np.uint8)
        inner_rect = ((cx, cy), (inner_width, inner_height), a)
        inner_box = cv2.boxPoints(inner_rect)
        cv2.fillConvexPoly(inner_mask, np.int32(inner_box), 1)
        
        # 空心矩形 = 外矩形 - 内矩形
        ring_mask = outer_mask & (~inner_mask)
        
        # 排除荧光区域
        if exclude_mask is not None:
            ring_mask = ring_mask & (~exclude_mask.astype(bool))
        
        return ring_mask.astype(np.uint8)
    
    def compute_roi_statistics(self, raw_channels: np.ndarray, 
                               roi_mask: np.ndarray) -> Dict[str, float]:
        """
        计算ROI内各通道的统计值
        """
        stats = {}
        channel_names = ['R', 'G1', 'G2', 'B']
        
        for i, name in enumerate(channel_names):
            channel_data = raw_channels[:, :, i]
            roi_values = channel_data[roi_mask > 0]
            
            if len(roi_values) > 0:
                stats[f'{name}_mean'] = float(np.mean(roi_values))
                stats[f'{name}_std'] = float(np.std(roi_values))
                stats[f'{name}_median'] = float(np.median(roi_values))
                stats[f'{name}_p10'] = float(np.percentile(roi_values, 10))
                stats[f'{name}_p90'] = float(np.percentile(roi_values, 90))
            else:
                stats[f'{name}_mean'] = 0.0
                stats[f'{name}_std'] = 0.0
                stats[f'{name}_median'] = 0.0
                stats[f'{name}_p10'] = 0.0
                stats[f'{name}_p90'] = 0.0
        
        return stats
    
    def process_image(self, filepath: Path, 
                     save_visualization: bool = True,
                     output_dir: Optional[Path] = None) -> ROIResult:
        """
        处理单张图像，提取ROI和特征向量
        
        参数:
            filepath: 图像文件路径
            save_visualization: 是否保存可视化结果
            output_dir: 输出目录
        """
        print(f"\n处理图像: {filepath.name}")
        
        # 1. 加载RAW图像
        raw_channels, metadata = self.load_raw_image(filepath)
        H, W = raw_channels.shape[:2]
        print(f"  图像尺寸: {W}x{H}")
        
        # 2. 创建强度图
        Y = self.create_intensity_map(raw_channels)
        
        # 3. 分割荧光区域
        fluor_mask, props = self.segment_fluorescence(Y)
        center = (props['center_x'], props['center_y'])
        detected_angle = props.get('angle_deg', 0.0)
        # 计算中心偏移
        center_offset = np.sqrt((center[0] - W/2)**2 + (center[1] - H/2)**2)
        center_offset_pct = center_offset / np.sqrt((W/2)**2 + (H/2)**2) * 100
        print(f"  荧光区域中心: ({center[0]:.1f}, {center[1]:.1f}), 偏移中心: {center_offset_pct:.1f}%, 角度: {detected_angle:.1f}°, 面积: {props['area']}")
        
        # 4. 根据模式创建ROI
        if self.roi_mode == "auto":
            # 自动模式：基于图像特征自动确定ROI尺寸和角度
            roi_width, roi_height, roi_angle, adjusted_center = self.auto_detect_roi(Y, raw_channels, fluor_mask, center)
            # 使用调整后的中心创建ROI
            roi_mask = self.create_fixed_roi(adjusted_center, (H, W), roi_width, roi_height, roi_angle)
            # 更新center为center调整后的值
            center = adjusted_center
        else:
            # 手动模式：使用配置的固定尺寸
            roi_width = self.fixed_roi_width
            roi_height = self.fixed_roi_height
            roi_angle = detected_angle  # 使用检测到的角度
            roi_mask = self.create_fixed_roi(center, (H, W))
        
        # 5. 创建背景ROI（传递实际ROI参数）
        if self.roi_mode == "auto":
            bg_mask = self.create_background_roi(center, (H, W), roi_width, roi_height, roi_angle, exclude_mask=fluor_mask)
        else:
            bg_mask = self.create_background_roi(center, (H, W), self.fixed_roi_width, self.fixed_roi_height, roi_angle, exclude_mask=fluor_mask)
        
        # 6. 计算统计值
        roi_stats = self.compute_roi_statistics(raw_channels, roi_mask)
        bg_stats = self.compute_roi_statistics(raw_channels, bg_mask)
        
        # 7. 背景扣除
        R_corrected = roi_stats['R_mean'] - bg_stats['R_mean']
        G1_corrected = roi_stats['G1_mean'] - bg_stats['G1_mean']
        G2_corrected = roi_stats['G2_mean'] - bg_stats['G2_mean']
        B_corrected = roi_stats['B_mean'] - bg_stats['B_mean']
        
        # 8. 生成RGB显示值（G1和G2平均）
        G_display = (G1_corrected + G2_corrected) / 2
        
        # 9. 构造结果
        result = ROIResult(
            filename=filepath.name,
            center_x=center[0],
            center_y=center[1],
            width=float(roi_width),
            height=float(roi_height),
            angle_deg=float(roi_angle),
            bbox=props['bbox'],
            area=props['area'],
            R_mean=roi_stats['R_mean'],
            G1_mean=roi_stats['G1_mean'],
            G2_mean=roi_stats['G2_mean'],
            B_mean=roi_stats['B_mean'],
            R_bg=bg_stats['R_mean'],
            G1_bg=bg_stats['G1_mean'],
            G2_bg=bg_stats['G2_mean'],
            B_bg=bg_stats['B_mean'],
            R_corrected=max(0, R_corrected),
            G1_corrected=max(0, G1_corrected),
            G2_corrected=max(0, G2_corrected),
            B_corrected=max(0, B_corrected),
            R_display=max(0, R_corrected),
            G_display=max(0, G_display),
            B_display=max(0, B_corrected),
            exposure_time=metadata.get('exposure_time'),
            iso=metadata.get('iso')
        )
        
        print(f"  特征向量: R={R_corrected:.1f}, G1={G1_corrected:.1f}, G2={G2_corrected:.1f}, B={B_corrected:.1f}")
        
        # 10. 保存可视化
        if save_visualization and output_dir is not None:
            self._save_visualization(filepath, raw_channels, Y, roi_mask, bg_mask, 
                                    fluor_mask, result, output_dir)
        
        return result
    
    def _save_visualization(self, filepath: Path, raw_channels: np.ndarray, 
                           Y: np.ndarray, roi_mask: np.ndarray, bg_mask: np.ndarray,
                           fluor_mask: np.ndarray, result: ROIResult, output_dir: Path):
        """保存可视化结果"""
        fig, axes = plt.subplots(2, 3, figsize=VISUALIZATION_FIGSIZE)
        
        # 1. 强度图
        ax = axes[0, 0]
        im = ax.imshow(Y, cmap='gray')
        ax.set_title('强度图 Y = max(R,G1,G2,B)')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        
        # 2. 荧光粗分割
        ax = axes[0, 1]
        ax.imshow(fluor_mask, cmap='gray')
        ax.plot(result.center_x, result.center_y, 'r+', markersize=20, markeredgewidth=2)
        ax.set_title(f'荧光区域分割\n中心: ({result.center_x:.1f}, {result.center_y:.1f})')
        ax.axis('off')
        
        # 3. 固定ROI
        ax = axes[0, 2]
        ax.imshow(Y, cmap='gray')
        ax.imshow(roi_mask, cmap='Reds', alpha=0.3)
        box = self._get_rotated_box((result.center_x, result.center_y), 
                                    result.width, result.height, result.angle_deg)
        poly = plt.Polygon(box, closed=True, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(poly)
        ax.plot(result.center_x, result.center_y, 'r+', markersize=15, markeredgewidth=2)
        mode_label = "自动" if self.roi_mode == "auto" else "固定"
        ax.set_title(f'{mode_label}ROI ({result.width:.0f}x{result.height:.0f}px, {result.angle_deg:.1f}°)')
        ax.axis('off')
        
        # 4. 背景ROI
        ax = axes[1, 0]
        ax.imshow(Y, cmap='gray')
        ax.imshow(bg_mask, cmap='Blues', alpha=0.3)
        ax.set_title('背景环形ROI')
        ax.axis('off')
        
        # 5. RGB伪彩色（背景校正后）
        ax = axes[1, 1]
        rgb_image = np.stack([
            raw_channels[:, :, 0],  # R
            (raw_channels[:, :, 1] + raw_channels[:, :, 2]) / 2,  # G
            raw_channels[:, :, 3]   # B
        ], axis=-1)
        rgb_image = np.clip(rgb_image / np.percentile(rgb_image, 99.5), 0, 1)
        ax.imshow(rgb_image)
        ax.imshow(roi_mask, cmap='Reds', alpha=0.3)
        ax.set_title('RGB伪彩色 + ROI')
        ax.axis('off')
        
        # 6. 通道柱状图
        ax = axes[1, 2]
        channels = ['R', 'G1', 'G2', 'B']
        corrected_values = [result.R_corrected, result.G1_corrected, 
                           result.G2_corrected, result.B_corrected]
        colors = ['red', 'green', 'green', 'blue']
        bars = ax.bar(channels, corrected_values, color=colors, alpha=0.7)
        ax.set_title('背景校正后的通道值')
        ax.set_ylabel('强度 (ADU)')
        ax.grid(axis='y', alpha=0.3)
        
        # 在柱子上标注数值
        for bar, val in zip(bars, corrected_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(f'ROI提取结果: {filepath.name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 保存
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{filepath.stem}_roi.png"
        plt.savefig(output_path, dpi=VISUALIZATION_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"  可视化已保存: {output_path}")
    
    def batch_process(self, image_dir: Path, 
                     output_dir: Optional[Path] = None,
                     save_visualization: bool = True) -> List[ROIResult]:
        """
        批量处理文件夹中的所有图像
        
        参数:
            image_dir: 图像文件夹路径
            output_dir: 输出目录
            save_visualization: 是否保存可视化
        """
        if output_dir is None:
            output_dir = image_dir.parent / 'roi_results'
        
        # 查找所有DNG文件
        image_files = sorted(image_dir.glob('*.dng'))
        
        if len(image_files) == 0:
            raise ValueError(f"在 {image_dir} 中未找到DNG文件")
        
        print(f"找到 {len(image_files)} 个DNG文件")
        
        results = []
        failed_files = []
        
        for filepath in image_files:
            try:
                result = self.process_image(filepath, save_visualization, output_dir)
                results.append(result)
            except Exception as e:
                print(f"  错误: {str(e)}")
                failed_files.append(filepath.name)
        
        print(f"\n处理完成: 成功 {len(results)}/{len(image_files)}")
        if failed_files:
            print(f"失败的文件: {failed_files}")
        
        # 保存结果到JSON
        if results:
            self._save_results(results, output_dir)
        
        return results
    
    def _save_results(self, results: List[ROIResult], output_dir: Path):
        """保存结果到JSON和CSV"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为JSON
        json_path = output_dir / 'roi_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        print(f"\nJSON结果已保存: {json_path}")
        
        # 保存为CSV
        csv_path = output_dir / 'roi_features.csv'
        with open(csv_path, 'w', encoding='utf-8') as f:
            # 写入表头
            f.write('filename,center_x,center_y,width,height,angle_deg,area,')
            f.write('R_mean,G1_mean,G2_mean,B_mean,')
            f.write('R_bg,G1_bg,G2_bg,B_bg,')
            f.write('R_corrected,G1_corrected,G2_corrected,B_corrected,')
            f.write('R_display,G_display,B_display,')
            f.write('exposure_time,iso\n')
            
            # 写入数据
            for r in results:
                f.write(f'{r.filename},{r.center_x:.2f},{r.center_y:.2f},{r.width:.2f},{r.height:.2f},{r.angle_deg:.2f},{r.area},')
                f.write(f'{r.R_mean:.2f},{r.G1_mean:.2f},{r.G2_mean:.2f},{r.B_mean:.2f},')
                f.write(f'{r.R_bg:.2f},{r.G1_bg:.2f},{r.G2_bg:.2f},{r.B_bg:.2f},')
                f.write(f'{r.R_corrected:.2f},{r.G1_corrected:.2f},{r.G2_corrected:.2f},{r.B_corrected:.2f},')
                f.write(f'{r.R_display:.2f},{r.G_display:.2f},{r.B_display:.2f},')
                f.write(f'{r.exposure_time},{r.iso}\n')
        
        print(f"CSV特征已保存: {csv_path}")
        
        # 保存特征矩阵（用于后续建模）
        features_path = output_dir / 'feature_matrix.npy'
        feature_matrix = np.array([r.get_feature_vector() for r in results])
        np.save(features_path, feature_matrix)
        print(f"特征矩阵已保存: {features_path} (shape: {feature_matrix.shape})")


def main():
    """主函数：示例用法"""
    # 从配置文件读取路径和参数
    data_dir = DATA_DIR
    output_dir = OUTPUT_DIR
    
    # 打印当前配置
    if 'print_config' in globals():
        print_config()
    else:
        print(f"\n数据目录: {data_dir}")
        print(f"输出目录: {output_dir}")
    
    # 创建提取器（使用配置文件中的参数）
    extractor = ROIExtractor()
    
    print(f"\n开始批量处理...")
    
    # 批量处理
    results = extractor.batch_process(
        image_dir=data_dir,
        output_dir=output_dir,
        save_visualization=SAVE_VISUALIZATION
    )
    
    # 输出统计信息
    if results:
        print("\n=== 统计摘要 ===")
        feature_matrix = np.array([r.get_feature_vector() for r in results])
        print(f"特征向量维度: {feature_matrix.shape}")
        print(f"R通道范围: [{feature_matrix[:, 0].min():.1f}, {feature_matrix[:, 0].max():.1f}]")
        print(f"G1通道范围: [{feature_matrix[:, 1].min():.1f}, {feature_matrix[:, 1].max():.1f}]")
        print(f"G2通道范围: [{feature_matrix[:, 2].min():.1f}, {feature_matrix[:, 2].max():.1f}]")
        print(f"B通道范围: [{feature_matrix[:, 3].min():.1f}, {feature_matrix[:, 3].max():.1f}]")


if __name__ == '__main__':
    main()
