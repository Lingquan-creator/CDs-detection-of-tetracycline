"""测试详细统计生成（GUI 选区一次后复用）。"""

from pathlib import Path

from roi_gui import select_roi_with_gui
from roi_processor import ROIProcessor


def test_detailed_statistics():
    """生成详细统计CSV，首次弹出GUI选择ROI/背景环，其余复用。"""
    print("=" * 60)
    print("测试详细统计功能")
    print("=" * 60)

    processor = ROIProcessor()

    data_dir = Path("h:/000/CDs/pic_RGB/data_pic")
    output_dir = Path("h:/000/CDs/pic_RGB/roi_results")

    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        return

    dng_files = list(data_dir.glob("*.dng"))
    print(f"\n找到 {len(dng_files)} 个DNG文件")

    if len(dng_files) == 0:
        print("没有DNG文件可供测试")
        return

    test_count = min(3, len(dng_files))
    print(f"测试处理前 {test_count} 个文件...\n")

    # 先在第一张图上用 GUI 选区
    raw_channels, _ = processor.load_raw_image(dng_files[0])
    roi_sel, bg_sel = select_roi_with_gui(raw_channels, processor.settings)

    results = []
    for i, filepath in enumerate(dng_files[:test_count]):
        print(f"\n[{i+1}/{test_count}] 处理: {filepath.name}")
        try:
            result = processor.process_image(
                filepath,
                save_visualization=False,
                output_dir=output_dir,
                roi_selection=roi_sel,
                bg_selection=bg_sel,
                interactive=False,
            )
            results.append(result)
            print("✔ 成功")
        except Exception as exc:
            print(f"✖ 失败: {exc}")

    if results:
        print(f"\n{'=' * 60}")
        print("保存结果...")
        processor._save_results(results, output_dir)
        processor._save_detailed_statistics(results, output_dir)

        print(f"\n{'=' * 60}")
        print("测试完成。")
        print(f"输出目录: {output_dir}")
        print("生成的文件：")
        print("  - roi_results.json")
        print("  - roi_features.csv")
        print("  - roi_detailed_statistics.csv")
        print("  - feature_matrix.npy")
    else:
        print("\n没有成功处理的文件")


if __name__ == "__main__":
    test_detailed_statistics()
