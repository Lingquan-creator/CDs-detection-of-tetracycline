"""
快速测试脚本：处理单个图像进行K-means分割测试
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rawpy
from sklearn.cluster import KMeans
import cv2


def quick_test():
    """快速测试单个图像的K-means分割"""
    
    # 设置路径
    data_dir = Path(r"h:\000\CDs\pic_Kmeans\data_pic")
    output_dir = Path(r"h:\000\CDs\pic_Kmeans\roi_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 选择第一个DNG文件进行测试
    dng_files = sorted(data_dir.glob("*.dng"))
    
    if not dng_files:
        print(f"错误：在 {data_dir} 中未找到DNG文件")
        return
    
    test_file = dng_files[0]
    print(f"测试文件: {test_file.name}")
    print("=" * 60)
    
    # 1. 加载图像
    print("1. 加载DNG图像...")
    with rawpy.imread(str(test_file)) as raw:
        rgb_image = raw.postprocess(
            use_camera_wb=True,
            output_bps=16,
            no_auto_bright=True
        )
    print(f"   图像尺寸: {rgb_image.shape}")
    
    # 2. 转换为灰度图
    print("2. 转换为灰度图...")
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray_norm = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    print(f"   灰度值范围: {gray_norm.min()} - {gray_norm.max()}")
    
    # 3. K-means聚类
    print("3. 执行K-means聚类...")
    pixels = gray_norm.reshape((-1, 1))
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    labels_image = labels.reshape(gray_norm.shape)
    
    print(f"   聚类中心: {kmeans.cluster_centers_.flatten()}")
    
    # 4. 识别高亮区域
    print("4. 识别高亮区域...")
    bright_label = np.argmax(kmeans.cluster_centers_)
    print(f"   高亮区域标签: {bright_label}")
    
    # 5. 创建掩码
    print("5. 创建掩码...")
    bright_mask = (labels_image == bright_label).astype(np.uint8) * 255
    background_mask = (labels_image != bright_label).astype(np.uint8) * 255
    
    # 6. 统计信息
    print("6. 计算统计信息...")
    total_pixels = gray_norm.shape[0] * gray_norm.shape[1]
    bright_pixels = np.sum(bright_mask > 0)
    background_pixels = np.sum(background_mask > 0)
    
    bright_ratio = bright_pixels / total_pixels
    background_ratio = background_pixels / total_pixels
    
    print(f"   背景区域: {background_pixels} 像素 ({background_ratio:.1%})")
    print(f"   高亮区域: {bright_pixels} 像素 ({bright_ratio:.1%})")
    
    # 提取各区域的RGB统计
    bright_rgb = rgb_image[bright_mask > 0]
    bg_rgb = rgb_image[background_mask > 0]
    
    if len(bright_rgb) > 0:
        print(f"   高亮区域平均RGB: R={np.mean(bright_rgb[:,0]):.1f}, "
              f"G={np.mean(bright_rgb[:,1]):.1f}, B={np.mean(bright_rgb[:,2]):.1f}")
    
    if len(bg_rgb) > 0:
        print(f"   背景区域平均RGB: R={np.mean(bg_rgb[:,0]):.1f}, "
              f"G={np.mean(bg_rgb[:,1]):.1f}, B={np.mean(bg_rgb[:,2]):.1f}")
    
    # 7. 可视化
    print("7. 生成可视化...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'K-means分割测试: {test_file.name}', fontsize=14, fontweight='bold')
    
    # 原始图像
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title('原始RGB图像')
    axes[0, 0].axis('off')
    
    # 灰度图
    axes[0, 1].imshow(gray_norm, cmap='gray')
    axes[0, 1].set_title('灰度图')
    axes[0, 1].axis('off')
    
    # 聚类标签
    axes[0, 2].imshow(labels_image, cmap='viridis')
    axes[0, 2].set_title('K-means聚类标签')
    axes[0, 2].axis('off')
    
    # 背景掩码
    axes[1, 0].imshow(background_mask, cmap='gray')
    axes[1, 0].set_title(f'背景区域 ({background_ratio:.1%})')
    axes[1, 0].axis('off')
    
    # 高亮掩码
    axes[1, 1].imshow(bright_mask, cmap='hot')
    axes[1, 1].set_title(f'高亮区域 ({bright_ratio:.1%})')
    axes[1, 1].axis('off')
    
    # 叠加显示
    overlay = rgb_image.copy()
    overlay[bright_mask > 0] = [255, 255, 0]  # 高亮区域标黄
    blended = cv2.addWeighted(rgb_image.astype(np.uint8), 0.6, 
                              overlay.astype(np.uint8), 0.4, 0)
    axes[1, 2].imshow(blended)
    axes[1, 2].set_title('高亮区域叠加')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / f"{test_file.stem}_test.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   可视化已保存: {output_path}")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print(f"如果结果正常，运行以下命令处理所有图像：")
    print("  python kmeans_advanced.py")


if __name__ == "__main__":
    quick_test()
