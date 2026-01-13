"""
ROI提取测试脚本
快速测试ROI自动识别功能
"""

from pathlib import Path
from roi_extractor import ROIExtractor
import numpy as np


def test_single_image():
    """测试单张图像处理"""
    print("=== 测试单张图像 ===\n")
    
    data_dir = Path(r'h:\000\CDs\pic_RGB\data_pic')
    output_dir = Path(r'h:\000\CDs\pic_RGB\roi_results_test')
    
    # 获取第一张图像
    image_files = sorted(data_dir.glob('*.dng'))
    if len(image_files) == 0:
        print("错误: 未找到DNG文件")
        return
    
    test_image = image_files[0]
    print(f"测试图像: {test_image.name}\n")
    
    # 创建提取器
    extractor = ROIExtractor(
        fixed_roi_radius=50,
        bg_ring_inner=80,
        bg_ring_outer=120,
        min_area=1000
    )
    
    # 处理图像
    try:
        result = extractor.process_image(
            test_image, 
            save_visualization=True, 
            output_dir=output_dir
        )
        
        print("\n=== 提取结果 ===")
        print(f"文件名: {result.filename}")
        print(f"ROI中心: ({result.center_x:.1f}, {result.center_y:.1f})")
        print(f"ROI半径: {result.radius:.1f} px")
        print(f"荧光区域面积: {result.area} px²")
        print(f"\n原始通道值:")
        print(f"  R={result.R_mean:.1f}, G1={result.G1_mean:.1f}, G2={result.G2_mean:.1f}, B={result.B_mean:.1f}")
        print(f"\n背景值:")
        print(f"  R={result.R_bg:.1f}, G1={result.G1_bg:.1f}, G2={result.G2_bg:.1f}, B={result.B_bg:.1f}")
        print(f"\n背景校正后:")
        print(f"  R={result.R_corrected:.1f}, G1={result.G1_corrected:.1f}, G2={result.G2_corrected:.1f}, B={result.B_corrected:.1f}")
        print(f"\n特征向量 (4D): {result.get_feature_vector()}")
        print(f"RGB向量 (3D): {result.get_rgb_vector()}")
        
        if result.exposure_time:
            print(f"\n曝光时间: {result.exposure_time} s")
        if result.iso:
            print(f"ISO: {result.iso}")
        
        print(f"\n可视化结果已保存到: {output_dir}")
        
    except Exception as e:
        print(f"处理失败: {str(e)}")
        import traceback
        traceback.print_exc()


def test_batch_processing():
    """测试批量处理"""
    print("\n\n=== 测试批量处理 ===\n")
    
    data_dir = Path(r'h:\000\CDs\pic_RGB\data_pic')
    output_dir = Path(r'h:\000\CDs\pic_RGB\roi_results')
    
    # 创建提取器
    extractor = ROIExtractor(
        fixed_roi_radius=50,
        bg_ring_inner=80,
        bg_ring_outer=120,
        min_area=1000
    )
    
    # 批量处理
    try:
        results = extractor.batch_process(
            image_dir=data_dir,
            output_dir=output_dir,
            save_visualization=True
        )
        
        if len(results) > 0:
            print("\n=== 统计摘要 ===")
            
            # 提取特征矩阵
            feature_matrix = np.array([r.get_feature_vector() for r in results])
            rgb_matrix = np.array([r.get_rgb_vector() for r in results])
            
            print(f"\n处理成功: {len(results)} 张图像")
            print(f"特征矩阵形状: {feature_matrix.shape}")
            
            print(f"\n4通道特征范围:")
            print(f"  R:  [{feature_matrix[:, 0].min():.1f}, {feature_matrix[:, 0].max():.1f}], 均值={feature_matrix[:, 0].mean():.1f}")
            print(f"  G1: [{feature_matrix[:, 1].min():.1f}, {feature_matrix[:, 1].max():.1f}], 均值={feature_matrix[:, 1].mean():.1f}")
            print(f"  G2: [{feature_matrix[:, 2].min():.1f}, {feature_matrix[:, 2].max():.1f}], 均值={feature_matrix[:, 2].mean():.1f}")
            print(f"  B:  [{feature_matrix[:, 3].min():.1f}, {feature_matrix[:, 3].max():.1f}], 均值={feature_matrix[:, 3].mean():.1f}")
            
            print(f"\nRGB显示值范围:")
            print(f"  R: [{rgb_matrix[:, 0].min():.1f}, {rgb_matrix[:, 0].max():.1f}], 均值={rgb_matrix[:, 0].mean():.1f}")
            print(f"  G: [{rgb_matrix[:, 1].min():.1f}, {rgb_matrix[:, 1].max():.1f}], 均值={rgb_matrix[:, 1].mean():.1f}")
            print(f"  B: [{rgb_matrix[:, 2].min():.1f}, {rgb_matrix[:, 2].max():.1f}], 均值={rgb_matrix[:, 2].mean():.1f}")
            
            # ROI位置统计
            centers_x = [r.center_x for r in results]
            centers_y = [r.center_y for r in results]
            areas = [r.area for r in results]
            
            print(f"\nROI位置统计:")
            print(f"  中心X: [{min(centers_x):.1f}, {max(centers_x):.1f}], 标准差={np.std(centers_x):.1f}")
            print(f"  中心Y: [{min(centers_y):.1f}, {max(centers_y):.1f}], 标准差={np.std(centers_y):.1f}")
            print(f"  面积: [{min(areas)}, {max(areas)}], 均值={np.mean(areas):.0f}")
            
            print(f"\n结果已保存到: {output_dir}")
            print(f"  - roi_results.json (完整结果)")
            print(f"  - roi_features.csv (特征表)")
            print(f"  - feature_matrix.npy (特征矩阵)")
            print(f"  - *_roi.png (可视化图像)")
        
    except Exception as e:
        print(f"批量处理失败: {str(e)}")
        import traceback
        traceback.print_exc()


def test_parameter_adjustment():
    """测试不同参数设置"""
    print("\n\n=== 测试参数调整 ===\n")
    
    data_dir = Path(r'h:\000\CDs\pic_RGB\data_pic')
    
    # 获取第一张图像
    image_files = sorted(data_dir.glob('*.dng'))
    if len(image_files) == 0:
        print("错误: 未找到DNG文件")
        return
    
    test_image = image_files[0]
    
    # 测试不同的ROI半径
    radii = [30, 50, 70]
    
    print("测试不同ROI半径的影响:\n")
    
    for radius in radii:
        output_dir = Path(rf'h:\000\CDs\pic_RGB\roi_results_r{radius}')
        
        extractor = ROIExtractor(
            fixed_roi_radius=radius,
            bg_ring_inner=radius + 30,
            bg_ring_outer=radius + 60
        )
        
        try:
            result = extractor.process_image(
                test_image, 
                save_visualization=True, 
                output_dir=output_dir
            )
            
            feature = result.get_feature_vector()
            print(f"半径={radius}px: 特征向量 = [{feature[0]:.0f}, {feature[1]:.0f}, {feature[2]:.0f}, {feature[3]:.0f}]")
            
        except Exception as e:
            print(f"半径={radius}px: 处理失败 - {str(e)}")


if __name__ == '__main__':
    # 选择要运行的测试
    print("ROI提取测试脚本\n")
    print("1. 测试单张图像")
    print("2. 测试批量处理")
    print("3. 测试参数调整")
    print("4. 运行所有测试")
    
    choice = input("\n请选择 (1-4): ").strip()
    
    if choice == '1':
        test_single_image()
    elif choice == '2':
        test_batch_processing()
    elif choice == '3':
        test_parameter_adjustment()
    elif choice == '4':
        test_single_image()
        test_batch_processing()
        test_parameter_adjustment()
    else:
        print("无效选择，运行默认测试...")
        test_single_image()
