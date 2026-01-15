"""
ROI 提取测试脚本（GUI 手动选区）。
"""

import numpy as np
from pathlib import Path

from roi_config import CURRENT_CONFIG
from roi_gui import BackgroundSelection, ROISelection
from roi_processor import ROIProcessor


def test_single_image():
    """测试单张图像处理（会弹出GUI进行ROI/背景环选择）。"""
    print("=== 测试单张图像（GUI 选区） ===\n")

    data_dir = Path(r"h:\000\CDs\pic_RGB\data_pic")
    output_dir = Path(r"h:\000\CDs\pic_RGB\roi_results_test")

    image_files = sorted(data_dir.glob("*.dng"))
    if len(image_files) == 0:
        print("错误: 未找到DNG文件")
        return

    test_image = image_files[0]
    print(f"将打开 GUI 选择 ROI: {test_image.name}\n")

    extractor = ROIProcessor(roi_params=CURRENT_CONFIG)

    try:
        result = extractor.process_image(
            test_image,
            save_visualization=True,
            output_dir=output_dir,
            interactive=True,
        )

        print("\n=== 提取结果 ===")
        print(
            f"ROI: center=({result.center_x:.1f}, {result.center_y:.1f}), size={result.width:.1f}x{result.height:.1f} px, angle={result.angle_deg:.1f}°"
        )
        print(
            f"BG : center=({result.bg_center_x:.1f}, {result.bg_center_y:.1f}), inner=({result.bg_inner_x:.1f},{result.bg_inner_y:.1f})px, outer=({result.bg_outer_x:.1f},{result.bg_outer_y:.1f})px"
        )
        print(f"ROI像素数: {result.area} px")
        print("\n原始通道值")
        print(
            f"  R={result.R_mean:.1f}, G1={result.G1_mean:.1f}, G2={result.G2_mean:.1f}, B={result.B_mean:.1f}"
        )
        print("\n背景值")
        print(
            f"  R={result.R_bg:.1f}, G1={result.G1_bg:.1f}, G2={result.G2_bg:.1f}, B={result.B_bg:.1f}"
        )
        print("\n背景校正后")
        print(
            f"  R={result.R_corrected:.1f}, G1={result.G1_corrected:.1f}, G2={result.G2_corrected:.1f}, B={result.B_corrected:.1f}"
        )
        print(f"\n特征向量 (4D): {result.get_feature_vector()}")
        print(f"RGB向量 (3D): {result.get_rgb_vector()}")

        if result.exposure_time:
            print(f"\n曝光时间: {result.exposure_time} s")
        if result.iso:
            print(f"ISO: {result.iso}")

        print(f"\n可视化结果已保存到 {output_dir}")

    except Exception as exc:
        print(f"处理失败: {exc}")
        import traceback

        traceback.print_exc()


def test_batch_processing():
    """测试批量处理，GUI 只在第一张弹出一次。"""
    print("\n\n=== 测试批量处理（首次需手动选区） ===\n")

    data_dir = Path(r"h:\000\CDs\pic_RGB\data_pic")
    output_dir = Path(r"h:\000\CDs\pic_RGB\roi_results")

    extractor = ROIProcessor(roi_params=CURRENT_CONFIG)

    try:
        results = extractor.batch_process(
            image_dir=data_dir,
            output_dir=output_dir,
            save_visualization=True,
            interactive=True,
        )

        if len(results) > 0:
            print("\n=== 统计摘要 ===")

            feature_matrix = np.array([r.get_feature_vector() for r in results])
            rgb_matrix = np.array([r.get_rgb_vector() for r in results])

            print(f"\n处理成功: {len(results)} 张图像")
            print(f"特征矩阵形状: {feature_matrix.shape}")

            print("\n4通道特征范围:")
            print(
                f"  R:  [{feature_matrix[:, 0].min():.1f}, {feature_matrix[:, 0].max():.1f}], 均值{feature_matrix[:, 0].mean():.1f}"
            )
            print(
                f"  G1: [{feature_matrix[:, 1].min():.1f}, {feature_matrix[:, 1].max():.1f}], 均值{feature_matrix[:, 1].mean():.1f}"
            )
            print(
                f"  G2: [{feature_matrix[:, 2].min():.1f}, {feature_matrix[:, 2].max():.1f}], 均值{feature_matrix[:, 2].mean():.1f}"
            )
            print(
                f"  B:  [{feature_matrix[:, 3].min():.1f}, {feature_matrix[:, 3].max():.1f}], 均值{feature_matrix[:, 3].mean():.1f}"
            )

            print("\nRGB显示值范围:")
            print(
                f"  R: [{rgb_matrix[:, 0].min():.1f}, {rgb_matrix[:, 0].max():.1f}], 均值{rgb_matrix[:, 0].mean():.1f}"
            )
            print(
                f"  G: [{rgb_matrix[:, 1].min():.1f}, {rgb_matrix[:, 1].max():.1f}], 均值{rgb_matrix[:, 1].mean():.1f}"
            )
            print(
                f"  B: [{rgb_matrix[:, 2].min():.1f}, {rgb_matrix[:, 2].max():.1f}], 均值{rgb_matrix[:, 2].mean():.1f}"
            )

            centers_x = [r.center_x for r in results]
            centers_y = [r.center_y for r in results]
            areas = [r.area for r in results]

            print("\nROI位置统计:")
            print(
                f"  中心X: [{min(centers_x):.1f}, {max(centers_x):.1f}], 标准差{np.std(centers_x):.1f}"
            )
            print(
                f"  中心Y: [{min(centers_y):.1f}, {max(centers_y):.1f}], 标准差{np.std(centers_y):.1f}"
            )
            print(f"  面积: [{min(areas)}, {max(areas)}], 均值{np.mean(areas):.0f}")

            print(f"\n结果已保存到: {output_dir}")
            print("  - roi_results.json (完整结果)")
            print("  - roi_features.csv (特征表)")
            print("  - feature_matrix.npy (特征矩阵)")
            print("  - *_roi.png (可视化图)")

    except Exception as exc:
        print(f"批量处理失败: {exc}")
        import traceback

        traceback.print_exc()


def test_parameter_adjustment():
    """测试不同尺寸/背景环参数的快速对比（不弹出GUI）。"""
    print("\n\n=== 测试参数调整（使用固定中心、关闭GUI） ===\n")

    data_dir = Path(r"h:\000\CDs\pic_RGB\data_pic")

    image_files = sorted(data_dir.glob("*.dng"))
    if len(image_files) == 0:
        print("错误: 未找到DNG文件")
        return

    test_image = image_files[0]

    sizes = [(60, 60), (100, 100), (140, 140)]

    for w, h in sizes:
        output_dir = Path(rf"h:\000\CDs\pic_RGB\roi_results_{w}x{h}")

        roi_params = {
            **CURRENT_CONFIG,
            "fixed_roi_width": w,
            "fixed_roi_height": h,
            "bg_ring_inner_x": max(w, h) // 2 + 20,
            "bg_ring_inner_y": max(w, h) // 2 + 20,
            "bg_ring_outer_x": max(w, h) // 2 + 50,
            "bg_ring_outer_y": max(w, h) // 2 + 50,
        }

        extractor = ROIProcessor(roi_params=roi_params)

        # 默认以图像中心为起点，不弹GUI
        raw_channels, _ = extractor.load_raw_image(test_image)
        H, W = raw_channels.shape[:2]
        roi_sel = ROISelection(
            center=(W / 2, H / 2),
            width=w,
            height=h,
            angle_deg=extractor.settings.roi_angle_deg,
        )
        bg_sel = BackgroundSelection(
            center=roi_sel.center,
            inner_x=roi_params["bg_ring_inner_x"],
            inner_y=roi_params["bg_ring_inner_y"],
            outer_x=roi_params["bg_ring_outer_x"],
            outer_y=roi_params["bg_ring_outer_y"],
            angle_deg=roi_sel.angle_deg,
        )

        try:
            result = extractor.process_image(
                test_image,
                save_visualization=True,
                output_dir=output_dir,
                roi_selection=roi_sel,
                bg_selection=bg_sel,
                interactive=False,
            )

            feature = result.get_feature_vector()
            print(
                f"尺寸={w}x{h}px: 特征向量 = [{feature[0]:.0f}, {feature[1]:.0f}, {feature[2]:.0f}, {feature[3]:.0f}]"
            )

        except Exception as exc:
            print(f"尺寸={w}x{h}px: 处理失败 - {exc}")


if __name__ == "__main__":
    print("ROI提取测试脚本\n")
    print("1. 测试单张图像（GUI）")
    print("2. 测试批量处理（首次 GUI）")
    print("3. 测试参数调整（无 GUI）")
    print("4. 运行所有测试")

    choice = input("\n请选择 (1-4): ").strip()

    if choice == "1":
        test_single_image()
    elif choice == "2":
        test_batch_processing()
    elif choice == "3":
        test_parameter_adjustment()
    elif choice == "4":
        test_single_image()
        test_batch_processing()
        test_parameter_adjustment()
    else:
        print("无效选择，运行默认单张测试...")
        test_single_image()
