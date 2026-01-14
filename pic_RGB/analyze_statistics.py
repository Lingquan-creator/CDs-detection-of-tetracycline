"""
查看和分析详细统计CSV的示例脚本
"""
import pandas as pd
from pathlib import Path

def analyze_detailed_statistics():
    """分析roi_detailed_statistics.csv"""
    
    csv_path = Path("h:/000/CDs/pic_RGB/roi_results/roi_detailed_statistics.csv")
    
    if not csv_path.exists():
        print(f"文件不存在: {csv_path}")
        print("请先运行 roi_processor.py 或 test_detailed_stats.py 生成数据")
        return
    
    # 读取CSV
    df = pd.read_csv(csv_path)
    
    print("=" * 70)
    print("详细统计数据分析")
    print("=" * 70)
    
    print(f"\n总共处理了 {len(df)} 张图像\n")
    
    # 显示列名
    print("可用的列:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print("\n" + "=" * 70)
    print("1. ROI区域通道均值")
    print("=" * 70)
    mean_cols = ['filename', 'roi_R_mean', 'roi_G1_mean', 'roi_G2_mean', 'roi_B_mean']
    if all(col in df.columns for col in mean_cols):
        print(df[mean_cols].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("2. 通道比值（抗亮度漂移）")
    print("=" * 70)
    ratio_cols = ['filename', 'roi_R_G_ratio', 'roi_B_G_ratio', 'roi_R_B_ratio', 'roi_G1_G2_ratio']
    if all(col in df.columns for col in ratio_cols):
        print(df[ratio_cols].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("3. 饱和比例（质量检查）")
    print("=" * 70)
    sat_cols = ['filename', 'roi_R_saturation_ratio', 'roi_G1_saturation_ratio', 
                'roi_G2_saturation_ratio', 'roi_B_saturation_ratio']
    if all(col in df.columns for col in sat_cols):
        print(df[sat_cols].to_string(index=False))
        
        # 检查是否有过曝图像
        max_sat = df[['roi_R_saturation_ratio', 'roi_G1_saturation_ratio', 
                      'roi_G2_saturation_ratio', 'roi_B_saturation_ratio']].max(axis=1)
        over_saturated = df[max_sat > 0.05]
        
        print(f"\n警告: {len(over_saturated)} 张图像可能过曝 (饱和比例 > 5%)")
        if len(over_saturated) > 0:
            print("过曝图像:")
            print(over_saturated['filename'].to_list())
    
    print("\n" + "=" * 70)
    print("4. 分位数统计（R通道示例）")
    print("=" * 70)
    percentile_cols = ['filename', 'roi_R_p10', 'roi_R_p50', 'roi_R_p90']
    if all(col in df.columns for col in percentile_cols):
        print(df[percentile_cols].to_string(index=False))
        # 计算动态范围
        df['R_dynamic_range'] = df['roi_R_p90'] - df['roi_R_p10']
        print(f"\nR通道动态范围 (p90-p10):")
        print(df[['filename', 'R_dynamic_range']].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("5. 方差分析（信号稳定性）")
    print("=" * 70)
    var_cols = ['filename', 'roi_R_var', 'roi_G1_var', 'roi_G2_var', 'roi_B_var']
    if all(col in df.columns for col in var_cols):
        print(df[var_cols].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("6. 背景水平")
    print("=" * 70)
    bg_cols = ['filename', 'bg_R_mean', 'bg_G1_mean', 'bg_G2_mean', 'bg_B_mean']
    if all(col in df.columns for col in bg_cols):
        print(df[bg_cols].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("统计摘要")
    print("=" * 70)
    
    # ROI均值统计
    if 'roi_R_mean' in df.columns:
        print("\nROI通道均值范围:")
        for ch in ['R', 'G1', 'G2', 'B']:
            col = f'roi_{ch}_mean'
            if col in df.columns:
                print(f"  {ch:3s}: [{df[col].min():8.2f}, {df[col].max():8.2f}], "
                      f"平均={df[col].mean():8.2f}, 标准差={df[col].std():8.2f}")
    
    # 通道比值统计
    if 'roi_R_G_ratio' in df.columns:
        print("\n通道比值范围:")
        for ratio in ['R_G_ratio', 'B_G_ratio', 'R_B_ratio']:
            col = f'roi_{ratio}'
            if col in df.columns:
                print(f"  {ratio:12s}: [{df[col].min():6.3f}, {df[col].max():6.3f}], "
                      f"平均={df[col].mean():6.3f}")
    
    # G1/G2一致性检查
    if 'roi_G1_G2_ratio' in df.columns:
        g_ratio = df['roi_G1_G2_ratio']
        print(f"\nG1/G2比值一致性:")
        print(f"  范围: [{g_ratio.min():.3f}, {g_ratio.max():.3f}]")
        print(f"  平均: {g_ratio.mean():.3f} (应接近1.0)")
        outliers = df[(g_ratio < 0.8) | (g_ratio > 1.2)]
        if len(outliers) > 0:
            print(f"  警告: {len(outliers)} 张图像的G1/G2比值异常")


if __name__ == "__main__":
    analyze_detailed_statistics()
