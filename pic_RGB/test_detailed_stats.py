"""测试详细统计功能"""
from pathlib import Path
from roi_processor import ROIProcessor
from roi_config import get_settings

def test_detailed_statistics():
    """测试生成详细统计CSV"""
    print("=" * 60)
    print("测试详细统计功能")
    print("=" * 60)
    
    # 创建处理器
    processor = ROIProcessor()
    
    # 获取测试数据目录
    data_dir = Path("h:/000/CDs/pic_RGB/data_pic")
    output_dir = Path("h:/000/CDs/pic_RGB/roi_results")
    
    if not data_dir.exists():
        print(f"错误: 数据目录不存在 {data_dir}")
        return
    
    # 查找DNG文件
    dng_files = list(data_dir.glob("*.dng"))
    print(f"\n找到 {len(dng_files)} 个DNG文件")
    
    if len(dng_files) == 0:
        print("没有DNG文件可供测试")
        return
    
    # 处理前几个文件进行测试
    test_count = min(3, len(dng_files))
    print(f"测试处理前 {test_count} 个文件...\n")
    
    results = []
    for i, filepath in enumerate(dng_files[:test_count]):
        print(f"\n[{i+1}/{test_count}] 处理: {filepath.name}")
        try:
            result = processor.process_image(
                filepath, 
                save_visualization=False,  # 不保存可视化以加快速度
                output_dir=output_dir
            )
            results.append(result)
            print(f"✓ 成功")
        except Exception as exc:
            print(f"✗ 失败: {exc}")
    
    if results:
        print(f"\n{'=' * 60}")
        print("保存结果...")
        processor._save_results(results, output_dir)
        processor._save_detailed_statistics(results, output_dir)
        
        print(f"\n{'=' * 60}")
        print("测试完成！")
        print(f"输出目录: {output_dir}")
        print(f"生成的文件:")
        print(f"  - roi_results.json")
        print(f"  - roi_features.csv")
        print(f"  - roi_detailed_statistics.csv  <-- 新增的详细统计文件")
        print(f"  - feature_matrix.npy")
    else:
        print("\n没有成功处理的文件")

if __name__ == "__main__":
    test_detailed_statistics()
