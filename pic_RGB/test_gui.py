"""测试GUI ROI处理流程"""
from pathlib import Path
from roi_processor import ROIProcessor

def test_gui_roi():
    """测试GUI ROI选择和处理"""
    print("=" * 60)
    print("GUI ROI处理测试")
    print("=" * 60)
    
    processor = ROIProcessor()
    
    data_dir = Path("h:/000/CDs/pic_RGB/data_pic")
    output_dir = Path("h:/000/CDs/pic_RGB/roi_results")
    
    if not data_dir.exists():
        print(f"错误: 数据目录不存在 {data_dir}")
        return
    
    dng_files = list(data_dir.glob("*.dng"))
    print(f"\n找到 {len(dng_files)} 个DNG文件")
    
    if len(dng_files) == 0:
        print("没有DNG文件")
        return
    
    # 测试处理前3个文件
    test_count = min(3, len(dng_files))
    print(f"测试处理前 {test_count} 个文件...\n")
    print("GUI操作:")
    print("  1. 拖拽选择ROI矩形")
    print("  2. 左侧控制窗口调整参数")
    print("  3. 键盘: 方向键/WASD移动, Tab切换ROI/BG, Enter确认\n")
    
    results = []
    for i, filepath in enumerate(dng_files[:test_count]):
        print(f"\n[{i+1}/{test_count}] {filepath.name}")
        try:
            result = processor.process_image(filepath, save_visualization=True, output_dir=output_dir)
            results.append(result)
            print("✓ 成功")
        except Exception as exc:
            print(f"✗ 失败: {exc}")
    
    if results:
        print(f"\n{'=' * 60}")
        print("保存结果...")
        processor._save_results(results, output_dir)
        processor._save_detailed_statistics(results, output_dir)
        
        print(f"\n{'=' * 60}")
        print("完成！")
        print(f"\n输出目录: {output_dir}")
        print("生成文件:")
        print("  - roi_results.json")
        print("  - roi_features.csv")
        print("  - roi_detailed_statistics.csv")
        print("  - feature_matrix.npy")
        print("  - *_roi_analysis.png")
    else:
        print("\n无成功处理的文件")

if __name__ == "__main__":
    test_gui_roi()
