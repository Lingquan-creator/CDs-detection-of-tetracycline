"""
(已弃用 - 请使用 kmeans_advanced.py)

此文件仅作为参考保留，实际使用请运行 kmeans_advanced.py
基于嵌套K-means的分割方法更加准确。
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rawpy
from sklearn.cluster import KMeans


class KmeansSegmentation:
    """使用K-means方法将图像分割为背景和高亮区域"""
    
    def __init__(self, n_clusters: int = 2):
        """
        初始化K-means分割器
        
        Args:
            n_clusters: 聚类数量（默认2：背景和高亮区域）
        """
        self.n_clusters = n_clusters
        
    def load_dng_image(self, filepath: Path) -> np.ndarray:
        """
        加载DNG图像文件
        
        Args:
            filepath: DNG文件路径
            
        Returns:
            处理后的RGB图像数组
        """
        with rawpy.imread(str(filepath)) as raw:
            # 转换为RGB图像
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=16,
                no_auto_bright=True
            )
        return rgb
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像：转换为灰度图并增强对比度
        
        Args:
            image: 输入RGB图像
            
        Returns:
            预处理后的灰度图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # 归一化到0-255范围
        gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return gray_norm
    
    def segment_kmeans(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, KMeans]:
        """
        使用K-means进行图像分割
        
        Args:
            image: 输入灰度图像
            
        Returns:
            labels: 每个像素的聚类标签
            segmented: 分割后的图像
            kmeans: 训练好的KMeans模型
        """
        # 将图像重塑为一维特征向量
        pixels = image.reshape((-1, 1))
        
        # 应用K-means聚类
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # 重塑标签为图像形状
        labels_image = labels.reshape(image.shape)
        
        # 创建分割图像（使用聚类中心的值）
        segmented = kmeans.cluster_centers_[labels].reshape(image.shape)
        
        return labels_image, segmented.astype(np.uint8), kmeans
    
    def identify_bright_region(self, labels: np.ndarray, centers: np.ndarray) -> int:
        """
        识别哪个聚类是高亮区域（基于聚类中心的亮度值）
        
        Args:
            labels: 聚类标签图像
            centers: 聚类中心
            
        Returns:
            高亮区域的标签索引
        """
        # 高亮区域应该有更高的亮度值
        bright_label = np.argmax(centers)
        return bright_label
    
    def create_masks(self, labels: np.ndarray, bright_label: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建背景和高亮区域的掩码
        
        Args:
            labels: 聚类标签图像
            bright_label: 高亮区域的标签
            
        Returns:
            background_mask: 背景掩码
            bright_mask: 高亮区域掩码
        """
        bright_mask = (labels == bright_label).astype(np.uint8) * 255
        background_mask = (labels != bright_label).astype(np.uint8) * 255
        
        return background_mask, bright_mask
    
    def calculate_statistics(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """
        计算指定区域的统计信息
        
        Args:
            image: 原始RGB图像
            mask: 区域掩码
            
        Returns:
            统计信息字典
        """
        # 提取掩码区域的像素
        masked_pixels = image[mask > 0]
        
        if len(masked_pixels) == 0:
            return {
                'area': 0,
                'mean_intensity': 0,
                'std_intensity': 0,
                'R_mean': 0,
                'G_mean': 0,
                'B_mean': 0
            }
        
        # 计算统计信息
        stats = {
            'area': np.sum(mask > 0),
            'mean_intensity': np.mean(masked_pixels),
            'std_intensity': np.std(masked_pixels),
            'R_mean': np.mean(masked_pixels[:, 0]),
            'G_mean': np.mean(masked_pixels[:, 1]),
            'B_mean': np.mean(masked_pixels[:, 2])
        }
        
        return stats
    
    def process_image(self, filepath: Path) -> Dict:
        """
        处理单个图像文件
        
        Args:
            filepath: 图像文件路径
            
        Returns:
            处理结果字典
        """
        print(f"处理图像: {filepath.name}")
        
        # 加载图像
        rgb_image = self.load_dng_image(filepath)
        
        # 预处理
        gray_image = self.preprocess_image(rgb_image)
        
        # K-means分割
        labels, segmented, kmeans = self.segment_kmeans(gray_image)
        
        # 识别高亮区域
        bright_label = self.identify_bright_region(labels, kmeans.cluster_centers_)
        
        # 创建掩码
        background_mask, bright_mask = self.create_masks(labels, bright_label)
        
        # 计算统计信息
        background_stats = self.calculate_statistics(rgb_image, background_mask)
        bright_stats = self.calculate_statistics(rgb_image, bright_mask)
        
        # 组织结果
        result = {
            'filename': filepath.name,
            'cluster_centers': kmeans.cluster_centers_.flatten().tolist(),
            'bright_label': int(bright_label),
            'background_stats': background_stats,
            'bright_region_stats': bright_stats,
            'background_ratio': background_stats['area'] / (gray_image.shape[0] * gray_image.shape[1]),
            'bright_ratio': bright_stats['area'] / (gray_image.shape[0] * gray_image.shape[1])
        }
        
        # 保存可视化结果
        visualization_data = {
            'rgb_image': rgb_image,
            'gray_image': gray_image,
            'labels': labels,
            'segmented': segmented,
            'background_mask': background_mask,
            'bright_mask': bright_mask
        }
        
        return result, visualization_data
    
    def save_visualization(self, filepath: Path, result: Dict, vis_data: Dict, output_dir: Path):
        """
        保存可视化结果
        
        Args:
            filepath: 原始文件路径
            result: 处理结果
            vis_data: 可视化数据
            output_dir: 输出目录
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'K-means分割结果: {filepath.name}', fontsize=16, fontweight='bold')
        
        # 原始RGB图像
        axes[0, 0].imshow(vis_data['rgb_image'])
        axes[0, 0].set_title('原始RGB图像')
        axes[0, 0].axis('off')
        
        # 灰度图
        axes[0, 1].imshow(vis_data['gray_image'], cmap='gray')
        axes[0, 1].set_title('预处理灰度图')
        axes[0, 1].axis('off')
        
        # 聚类标签
        axes[0, 2].imshow(vis_data['labels'], cmap='viridis')
        axes[0, 2].set_title(f'K-means聚类标签 (k={self.n_clusters})')
        axes[0, 2].axis('off')
        
        # 分割结果
        axes[1, 0].imshow(vis_data['segmented'], cmap='gray')
        axes[1, 0].set_title('分割结果')
        axes[1, 0].axis('off')
        
        # 背景掩码
        axes[1, 1].imshow(vis_data['background_mask'], cmap='gray')
        axes[1, 1].set_title(f"背景区域 ({result['background_ratio']:.1%})")
        axes[1, 1].axis('off')
        
        # 高亮区域掩码
        axes[1, 2].imshow(vis_data['bright_mask'], cmap='gray')
        axes[1, 2].set_title(f"高亮区域 ({result['bright_ratio']:.1%})")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        output_path = output_dir / f"{filepath.stem}_kmeans_segmentation.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  可视化结果已保存: {output_path.name}")
    
    def save_masks(self, filepath: Path, vis_data: Dict, output_dir: Path):
        """
        保存掩码图像
        
        Args:
            filepath: 原始文件路径
            vis_data: 可视化数据
            output_dir: 输出目录
        """
        # 保存背景掩码
        bg_path = output_dir / f"{filepath.stem}_background_mask.png"
        cv2.imwrite(str(bg_path), vis_data['background_mask'])
        
        # 保存高亮区域掩码
        bright_path = output_dir / f"{filepath.stem}_bright_mask.png"
        cv2.imwrite(str(bright_path), vis_data['bright_mask'])
        
        print(f"  掩码已保存: {bg_path.name}, {bright_path.name}")


def main():
    """主函数"""
    # 设置路径
    data_dir = Path(r"h:\000\CDs\pic_Kmeans\data_pic")
    output_dir = Path(r"h:\000\CDs\pic_Kmeans\roi_results")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化分割器
    segmenter = KmeansSegmentation(n_clusters=2)
    
    # 获取所有DNG文件
    dng_files = sorted(data_dir.glob("*.dng"))
    
    if not dng_files:
        print(f"在 {data_dir} 中未找到DNG文件")
        return
    
    print(f"找到 {len(dng_files)} 个DNG文件")
    print("=" * 60)
    
    # 处理所有图像
    all_results = []
    
    for filepath in dng_files:
        try:
            # 处理图像
            result, vis_data = segmenter.process_image(filepath)
            all_results.append(result)
            
            # 保存可视化
            segmenter.save_visualization(filepath, result, vis_data, output_dir)
            
            # 保存掩码
            segmenter.save_masks(filepath, vis_data, output_dir)
            
            print(f"  背景区域面积: {result['background_stats']['area']} 像素 ({result['background_ratio']:.1%})")
            print(f"  高亮区域面积: {result['bright_region_stats']['area']} 像素 ({result['bright_ratio']:.1%})")
            print(f"  高亮区域平均亮度: {result['bright_region_stats']['mean_intensity']:.1f}")
            print("-" * 60)
            
        except Exception as e:
            print(f"  处理失败: {e}")
            continue
    
    # 保存汇总结果
    if all_results:
        results_path = output_dir / "kmeans_segmentation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n汇总结果已保存到: {results_path}")
        print(f"处理完成！共处理 {len(all_results)} 个图像")
    else:
        print("\n没有成功处理任何图像")


if __name__ == "__main__":
    main()
