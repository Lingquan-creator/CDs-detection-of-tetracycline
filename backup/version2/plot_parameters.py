"""
量子点光谱拟合结果可视化

绘制拟合参数随浓度变化的趋势图。
Visualize fitting parameters vs concentration.

使用方法 / Usage:
    python plot_parameters.py
"""

import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# 配置 / Configuration
# ============================================================================

INPUT_CSV = "fit_results_batch.csv"  # 输入CSV文件 / Input CSV file
OUTPUT_DIR = "parameter_plots"       # 输出图片目录 / Output directory for plots
DPI = 150                            # 图片分辨率 / Image resolution
FIGSIZE = (10, 6)                    # 图片尺寸 / Figure size

# 要绘制的参数列表 / Parameters to plot
PARAMETERS = [
    'A_prime',
    'k',
    'lambda0_nm',
    'sigma_E_eV',
    'peak_position_nm',
    'E0_eV',
    'max_intensity',
    'integrated_intensity',
]

# 参数显示名称和单位 / Parameter display names and units
PARAM_INFO = {
    'A_prime': {
        'label': "A' (Amplitude)",
        'unit': 'a.u.',
        'ylabel': "A' (a.u.)",
    },
    'k': {
        'label': 'k (Position-Intensity Slope)',
        'unit': '',
        'ylabel': 'k',
    },
    'lambda0_nm': {
        'label': 'λ₀ (Center Wavelength)',
        'unit': 'nm',
        'ylabel': 'λ₀ (nm)',
    },
    'sigma_E_eV': {
        'label': 'σₑ (Energy Broadening)',
        'unit': 'eV',
        'ylabel': 'σₑ (eV)',
    },
    'peak_position_nm': {
        'label': 'λₚₑₐₖ (Peak Position)',
        'unit': 'nm',
        'ylabel': 'λₚₑₐₖ (nm)',
    },
    'E0_eV': {
        'label': 'E₀ (Center Energy)',
        'unit': 'eV',
        'ylabel': 'E₀ (eV)',
    },
    'max_intensity': {
        'label': 'Iₘₐₓ (Maximum Intensity)',
        'unit': 'a.u.',
        'ylabel': 'Iₘₐₓ (a.u.)',
    },
    'integrated_intensity': {
        'label': 'Integrated Intensity',
        'unit': 'a.u.',
        'ylabel': 'Integrated Intensity (a.u.)',
    },
}

# 峰的颜色和标记 / Peak colors and markers
PEAK_STYLES = [
    {'color': '#1f77b4', 'marker': 'o', 'label': 'Peak 1'},
    {'color': '#ff7f0e', 'marker': 's', 'label': 'Peak 2'},
    {'color': '#2ca02c', 'marker': '^', 'label': 'Peak 3'},
    {'color': '#d62728', 'marker': 'D', 'label': 'Peak 4'},
    {'color': '#9467bd', 'marker': 'v', 'label': 'Peak 5'},
    {'color': '#8c564b', 'marker': 'p', 'label': 'Peak 6'},
]


# ============================================================================
# 数据处理 / Data Processing
# ============================================================================

def extract_concentration(concentration_str: str) -> float:
    """
    从浓度字符串中提取数值 / Extract numeric value from concentration string
    
    Examples:
        "0.05μM" -> 0.05
        "Blank" -> 0.0
        "12μM" -> 12.0
    """
    if pd.isna(concentration_str) or str(concentration_str).lower() == 'blank':
        return 0.0
    
    # 提取数字（包括小数点）/ Extract numbers (including decimals)
    match = re.search(r'(\d+\.?\d*)', str(concentration_str))
    if match:
        return float(match.group(1))
    return 0.0


def load_and_process_data(csv_path: Path) -> Tuple[pd.DataFrame, int]:
    """
    加载并处理CSV数据 / Load and process CSV data
    
    Returns:
        (DataFrame, number_of_peaks)
    """
    df = pd.read_csv(csv_path)
    
    # 提取浓度数值 / Extract numeric concentration
    df['Concentration_numeric'] = df['Concentration'].apply(extract_concentration)
    
    # 按浓度排序 / Sort by concentration
    df = df.sort_values('Concentration_numeric').reset_index(drop=True)
    
    # 检测峰数量 / Detect number of peaks
    n_peaks = 0
    for col in df.columns:
        if col.startswith('Peak') and col.endswith('_A_prime'):
            peak_num = int(col.split('_')[0].replace('Peak', ''))
            n_peaks = max(n_peaks, peak_num)
    
    print(f"✓ 数据加载成功 / Data loaded successfully")
    print(f"  样品数量 / Number of samples: {len(df)}")
    print(f"  峰数量 / Number of peaks: {n_peaks}")
    print(f"  浓度范围 / Concentration range: {df['Concentration_numeric'].min():.2f} - {df['Concentration_numeric'].max():.2f}")
    
    return df, n_peaks


# ============================================================================
# 绘图函数 / Plotting Functions
# ============================================================================

def plot_parameter(
    df: pd.DataFrame,
    parameter: str,
    n_peaks: int,
    output_dir: Path
) -> None:
    """
    绘制单个参数随浓度变化的图 / Plot single parameter vs concentration
    
    Args:
        df: 数据框 / DataFrame
        parameter: 参数名称（不含Peak前缀）/ Parameter name (without Peak prefix)
        n_peaks: 峰数量 / Number of peaks
        output_dir: 输出目录 / Output directory
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    concentrations = df['Concentration_numeric'].values
    
    # 绘制每个峰的数据 / Plot data for each peak
    for peak_idx in range(1, n_peaks + 1):
        col_name = f'Peak{peak_idx}_{parameter}'
        
        if col_name not in df.columns:
            continue
        
        values = pd.to_numeric(df[col_name], errors='coerce').values
        
        # 过滤无效数据 / Filter invalid data
        valid_mask = np.isfinite(values)
        if valid_mask.sum() == 0:
            continue
        
        style = PEAK_STYLES[(peak_idx - 1) % len(PEAK_STYLES)]
        
        ax.plot(
            concentrations[valid_mask],
            values[valid_mask],
            marker=style['marker'],
            color=style['color'],
            linewidth=2,
            markersize=8,
            label=style['label'],
            alpha=0.8
        )
    
    # 设置标签和标题 / Set labels and title
    param_info = PARAM_INFO.get(parameter, {})
    ylabel = param_info.get('ylabel', parameter)
    title = param_info.get('label', parameter)
    
    ax.set_xlabel('Concentration (μM)', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(f'{title} vs Concentration', fontsize=14, fontweight='bold', pad=15)
    
    # 图例 / Legend
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    
    # 网格 / Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 设置坐标轴范围 / Set axis limits
    ax.set_xlim(left=-0.5)
    
    # 紧凑布局 / Tight layout
    plt.tight_layout()
    
    # 保存图片 / Save figure
    output_file = output_dir / f'{parameter}_vs_concentration.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {output_file.name}")


def plot_all_parameters(df: pd.DataFrame, n_peaks: int, output_dir: Path) -> None:
    """
    绘制所有参数的图 / Plot all parameters
    
    Args:
        df: 数据框 / DataFrame
        n_peaks: 峰数量 / Number of peaks
        output_dir: 输出目录 / Output directory
    """
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n📊 开始绘图 / Starting plotting...")
    print(f"   输出目录 / Output directory: {output_dir}\n")
    
    for param in PARAMETERS:
        try:
            plot_parameter(df, param, n_peaks, output_dir)
        except Exception as e:
            print(f"  ✗ 绘制 {param} 失败 / Failed to plot {param}: {e}")
    
    print(f"\n✅ 所有图片已保存 / All plots saved to: {output_dir}")


def create_summary_plot(df: pd.DataFrame, n_peaks: int, output_dir: Path) -> None:
    """
    创建汇总图：多个子图展示关键参数 / Create summary plot with multiple subplots
    
    Args:
        df: 数据框 / DataFrame
        n_peaks: 峰数量 / Number of peaks
        output_dir: 输出目录 / Output directory
    """
    # 选择4个关键参数 / Select 4 key parameters
    key_params = ['lambda0_nm', 'sigma_E_eV', 'max_intensity', 'integrated_intensity']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    concentrations = df['Concentration_numeric'].values
    
    for idx, param in enumerate(key_params):
        ax = axes[idx]
        
        # 绘制每个峰 / Plot each peak
        for peak_idx in range(1, n_peaks + 1):
            col_name = f'Peak{peak_idx}_{param}'
            
            if col_name not in df.columns:
                continue
            
            values = pd.to_numeric(df[col_name], errors='coerce').values
            valid_mask = np.isfinite(values)
            
            if valid_mask.sum() == 0:
                continue
            
            style = PEAK_STYLES[(peak_idx - 1) % len(PEAK_STYLES)]
            
            ax.plot(
                concentrations[valid_mask],
                values[valid_mask],
                marker=style['marker'],
                color=style['color'],
                linewidth=2,
                markersize=6,
                label=style['label'],
                alpha=0.8
            )
        
        # 设置标签 / Set labels
        param_info = PARAM_INFO.get(param, {})
        ylabel = param_info.get('ylabel', param)
        title = param_info.get('label', param)
        
        ax.set_xlabel('Concentration (μM)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=-0.5)
    
    plt.suptitle('Fitting Parameters vs Concentration - Summary', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # 保存 / Save
    output_file = output_dir / 'summary_plot.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 汇总图已保存 / Summary plot saved: {output_file}")


# ============================================================================
# 主程序 / Main Program
# ============================================================================

def main() -> None:
    """主程序入口 / Main program entry"""
    csv_path = Path(INPUT_CSV)
    
    if not csv_path.exists():
        print(f"❌ 错误 / Error: 文件不存在 / File not found: {csv_path}")
        return
    
    print("="*70)
    print("量子点光谱拟合结果可视化 / Quantum Dot Fitting Results Visualization")
    print("="*70)
    
    try:
        # 加载数据 / Load data
        df, n_peaks = load_and_process_data(csv_path)
        
        # 创建输出目录 / Create output directory
        output_dir = Path(OUTPUT_DIR)
        
        # 绘制所有参数 / Plot all parameters
        plot_all_parameters(df, n_peaks, output_dir)
        
        # 创建汇总图 / Create summary plot
        create_summary_plot(df, n_peaks, output_dir)
        
        print("\n" + "="*70)
        print("🎉 可视化完成！/ Visualization completed!")
        print(f"   图片数量 / Number of plots: {len(PARAMETERS) + 1}")
        print(f"   输出目录 / Output directory: {output_dir.absolute()}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 程序错误 / Program error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
