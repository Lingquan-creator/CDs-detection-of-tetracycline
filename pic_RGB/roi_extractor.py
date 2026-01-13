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
    radius: float
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
                 fixed_roi_radius: int = 50,
                 bg_ring_inner: int = 80,
                 bg_ring_outer: int = 120,
                 otsu_percentile: float = 0.85,
                 morph_kernel_size: int = 5,
                 min_area: int = 1000):
        """
        参数:
            fixed_roi_radius: 固定ROI圆形半径（像素）
            bg_ring_inner: 背景环带内半径
            bg_ring_outer: 背景环带外半径
            otsu_percentile: 阈值分割的百分位数（如果不用Otsu）
            morph_kernel_size: 形态学操作的核大小
            min_area: 最小连通域面积
        """
        self.fixed_roi_radius = fixed_roi_radius
        self.bg_ring_inner = bg_ring_inner
        self.bg_ring_outer = bg_ring_outer
        self.otsu_percentile = otsu_percentile
        self.morph_kernel_size = morph_kernel_size
        self.min_area = min_area
        
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
        分割荧光区域
        
        返回:
            mask: 二值mask
            props: 包含质心、外接框等属性的字典
        """
        # 归一化到0-255用于阈值分割
        Y_norm = cv2.normalize(Y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Otsu阈值分割
        _, binary = cv2.threshold(Y_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学操作：开运算去噪，闭运算填洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (self.morph_kernel_size, self.morph_kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # 找到最大连通域（排除背景label=0）
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = np.argmax(areas) + 1
            
            # 过滤小区域
            if stats[largest_idx, cv2.CC_STAT_AREA] < self.min_area:
                raise ValueError(f"最大连通域面积 {stats[largest_idx, cv2.CC_STAT_AREA]} 小于阈值 {self.min_area}")
            
            # 创建最大连通域的mask
            mask = (labels == largest_idx).astype(np.uint8)
            
            # 提取属性
            props = {
                'center_x': centroids[largest_idx][0],
                'center_y': centroids[largest_idx][1],
                'bbox': (stats[largest_idx, cv2.CC_STAT_LEFT],
                        stats[largest_idx, cv2.CC_STAT_TOP],
                        stats[largest_idx, cv2.CC_STAT_LEFT] + stats[largest_idx, cv2.CC_STAT_WIDTH],
                        stats[largest_idx, cv2.CC_STAT_TOP] + stats[largest_idx, cv2.CC_STAT_HEIGHT]),
                'area': stats[largest_idx, cv2.CC_STAT_AREA]
            }
        else:
            raise ValueError("未找到荧光区域")
        
        return mask, props
    
    def create_fixed_roi(self, center: Tuple[float, float], shape: Tuple[int, int]) -> np.ndarray:
        """
        创建固定半径的圆形ROI
        
        参数:
            center: (x, y) 中心坐标
            shape: (H, W) 图像尺寸
        """
        H, W = shape
        y, x = np.ogrid[:H, :W]
        cx, cy = center
        
        mask = ((x - cx)**2 + (y - cy)**2) <= self.fixed_roi_radius**2
        return mask.astype(np.uint8)
    
    def create_background_roi(self, center: Tuple[float, float], 
                             shape: Tuple[int, int],
                             exclude_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        创建环形背景ROI
        
        参数:
            center: (x, y) 中心坐标
            shape: (H, W) 图像尺寸
            exclude_mask: 需要排除的区域（如荧光粗mask）
        """
        H, W = shape
        y, x = np.ogrid[:H, :W]
        cx, cy = center
        
        dist_sq = (x - cx)**2 + (y - cy)**2
        ring_mask = (dist_sq > self.bg_ring_inner**2) & (dist_sq <= self.bg_ring_outer**2)
        
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
        print(f"  荧光区域中心: ({center[0]:.1f}, {center[1]:.1f}), 面积: {props['area']}")
        
        # 4. 创建固定ROI
        roi_mask = self.create_fixed_roi(center, (H, W))
        
        # 5. 创建背景ROI
        bg_mask = self.create_background_roi(center, (H, W), exclude_mask=fluor_mask)
        
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
            radius=float(self.fixed_roi_radius),
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
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
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
        circle = plt.Circle((result.center_x, result.center_y), self.fixed_roi_radius, 
                           color='red', fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.set_title(f'固定ROI (半径={self.fixed_roi_radius}px)')
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
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
            f.write('filename,center_x,center_y,radius,area,')
            f.write('R_mean,G1_mean,G2_mean,B_mean,')
            f.write('R_bg,G1_bg,G2_bg,B_bg,')
            f.write('R_corrected,G1_corrected,G2_corrected,B_corrected,')
            f.write('R_display,G_display,B_display,')
            f.write('exposure_time,iso\n')
            
            # 写入数据
            for r in results:
                f.write(f'{r.filename},{r.center_x:.2f},{r.center_y:.2f},{r.radius:.2f},{r.area},')
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
    # 设置路径
    data_dir = Path(r'h:\000\CDs\pic_RGB\data_pic')
    output_dir = Path(r'h:\000\CDs\pic_RGB\roi_results')
    
    # 创建提取器
    extractor = ROIExtractor(
        fixed_roi_radius=50,      # 固定ROI半径
        bg_ring_inner=80,         # 背景环内径
        bg_ring_outer=120,        # 背景环外径
        min_area=1000             # 最小荧光区域面积
    )
    
    # 批量处理
    results = extractor.batch_process(
        image_dir=data_dir,
        output_dir=output_dir,
        save_visualization=True
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
