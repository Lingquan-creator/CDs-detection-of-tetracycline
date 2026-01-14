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
from scipy.optimize import curve_fit


# ============================================================================
# 配置 / Configuration
# ============================================================================

INPUT_CSV = "results_spec.csv"  # 输入CSV文件 / Input CSV file
OUTPUT_DIR = "parameter_plots"       # 输出图片目录 / Output directory for plots
DPI = 150                            # 图片分辨率 / Image resolution
FIGSIZE = (10, 6)                    # 图片尺寸 / Figure size
LINEAR_FIT_CSV = "results_parameter.csv"  # 线性拟合结果输出 / Linear fit results output

# 拟合时忽略的浓度点配置 / Concentration points to ignore during fitting
IGNORE_CONCENTRATIONS = [0, 0.2, 0.4]  # 要忽略的特定浓度值列表（0对应空白） / List of specific concentration values to ignore (0 for blank)
                                        # 例如 / Example: [0, 0.05, 0.1, 5.0]
                                        # 留空则不忽略任何浓度 / Leave empty to not ignore any concentrations

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

# 参数最佳拟合模型配置 / Best fit model configuration for each parameter
# 只使用指数函数和线性函数进行拟合
BEST_FIT_MODELS = {
    'A_prime': 'exponential',              # 振幅：指数增长
    'k': 'linear',                         # 斜率：线性关系
    'lambda0_nm': 'linear',                # 中心波长：线性
    'sigma_E_eV': 'linear',                # 能量展宽：线性
    'peak_position_nm': 'linear',          # 峰位置：线性
    'E0_eV': 'linear',                     # 中心能量：线性
    'max_intensity': 'exponential',        # 最大强度：指数增长
    'integrated_intensity': 'exponential', # 积分强度：指数增长
}


# ============================================================================
# 数据处理 / Data Processing
# ============================================================================

def linear_fit(x: np.ndarray, y: np.ndarray) -> dict | None:
    """
    对数据做一元线性拟合并返回指标 / Perform simple linear fit and metrics
    y = a*x + b
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    
    x_valid = x[mask]
    y_valid = y[mask]
    
    # 最小二乘直线 y = m*x + b
    slope, intercept = np.polyfit(x_valid, y_valid, 1)
    y_pred = slope * x_valid + intercept
    residuals = y_valid - y_pred
    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_valid - y_valid.mean())**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rmse = np.sqrt(np.mean(residuals**2))
    
    return {
        'type': 'linear',
        'formula': 'y = a*x + b',
        'params': {'a': slope, 'b': intercept},
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'rmse': rmse,
        'n_points': mask.sum(),
        'x_min': x_valid.min(),
        'x_max': x_valid.max(),
    }


def quadratic_fit(x: np.ndarray, y: np.ndarray) -> dict | None:
    """
    二次多项式拟合 / Quadratic polynomial fit
    y = a*x^2 + b*x + c
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    
    x_valid = x[mask]
    y_valid = y[mask]
    
    try:
        coeffs = np.polyfit(x_valid, y_valid, 2)
        a, b, c = coeffs
        y_pred = a * x_valid**2 + b * x_valid + c
        residuals = y_valid - y_pred
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_valid - y_valid.mean())**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        rmse = np.sqrt(np.mean(residuals**2))
        
        return {
            'type': 'quadratic',
            'formula': 'y = a*x² + b*x + c',
            'params': {'a': a, 'b': b, 'c': c},
            'r_squared': r_squared,
            'rmse': rmse,
            'n_points': mask.sum(),
            'x_min': x_valid.min(),
            'x_max': x_valid.max(),
        }
    except:
        return None


def exponential_fit(x: np.ndarray, y: np.ndarray) -> dict | None:
    """
    指数拟合 / Exponential fit
    y = a * exp(b*x)
    """
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if mask.sum() < 2:
        return None
    
    x_valid = x[mask]
    y_valid = y[mask]
    
    try:
        # 对数变换: ln(y) = ln(a) + b*x
        log_y = np.log(y_valid)
        b, log_a = np.polyfit(x_valid, log_y, 1)
        a = np.exp(log_a)
        
        y_pred = a * np.exp(b * x_valid)
        residuals = y_valid - y_pred
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_valid - y_valid.mean())**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        rmse = np.sqrt(np.mean(residuals**2))
        
        return {
            'type': 'exponential',
            'formula': 'y = a * exp(b*x)',
            'params': {'a': a, 'b': b},
            'r_squared': r_squared,
            'rmse': rmse,
            'n_points': mask.sum(),
            'x_min': x_valid.min(),
            'x_max': x_valid.max(),
        }
    except:
        return None


def logarithmic_fit(x: np.ndarray, y: np.ndarray) -> dict | None:
    """
    对数拟合 / Logarithmic fit
    y = a * ln(x) + b
    """
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if mask.sum() < 2:
        return None
    
    x_valid = x[mask]
    y_valid = y[mask]
    
    try:
        log_x = np.log(x_valid)
        a, b = np.polyfit(log_x, y_valid, 1)
        
        y_pred = a * np.log(x_valid) + b
        residuals = y_valid - y_pred
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_valid - y_valid.mean())**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        rmse = np.sqrt(np.mean(residuals**2))
        
        return {
            'type': 'logarithmic',
            'formula': 'y = a * ln(x) + b',
            'params': {'a': a, 'b': b},
            'r_squared': r_squared,
            'rmse': rmse,
            'n_points': mask.sum(),
            'x_min': x_valid.min(),
            'x_max': x_valid.max(),
        }
    except:
        return None


def power_fit(x: np.ndarray, y: np.ndarray) -> dict | None:
    """
    幂律拟合 / Power law fit
    y = a * x^b
    """
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < 2:
        return None
    
    x_valid = x[mask]
    y_valid = y[mask]
    
    try:
        # 对数变换: ln(y) = ln(a) + b*ln(x)
        log_x = np.log(x_valid)
        log_y = np.log(y_valid)
        b, log_a = np.polyfit(log_x, log_y, 1)
        a = np.exp(log_a)
        
        y_pred = a * x_valid**b
        residuals = y_valid - y_pred
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_valid - y_valid.mean())**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        rmse = np.sqrt(np.mean(residuals**2))
        
        return {
            'type': 'power',
            'formula': 'y = a * x^b',
            'params': {'a': a, 'b': b},
            'r_squared': r_squared,
            'rmse': rmse,
            'n_points': mask.sum(),
            'x_min': x_valid.min(),
            'x_max': x_valid.max(),
        }
    except:
        return None


def hyperbolic_fit(x: np.ndarray, y: np.ndarray) -> dict | None:
    """
    双曲线拟合 / Hyperbolic fit
    y = a/(x + b) + c
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    
    x_valid = x[mask]
    y_valid = y[mask]
    
    try:
        def hyperbolic(x, a, b, c):
            return a / (x + b) + c
        
        # 初始猜测
        p0 = [1.0, 1.0, y_valid.mean()]
        popt, _ = curve_fit(hyperbolic, x_valid, y_valid, p0=p0, maxfev=5000)
        a, b, c = popt
        
        y_pred = hyperbolic(x_valid, a, b, c)
        residuals = y_valid - y_pred
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_valid - y_valid.mean())**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        rmse = np.sqrt(np.mean(residuals**2))
        
        return {
            'type': 'hyperbolic',
            'formula': 'y = a/(x + b) + c',
            'params': {'a': a, 'b': b, 'c': c},
            'r_squared': r_squared,
            'rmse': rmse,
            'n_points': mask.sum(),
            'x_min': x_valid.min(),
            'x_max': x_valid.max(),
        }
    except:
        return None


def find_best_fit(x: np.ndarray, y: np.ndarray) -> dict | None:
    """
    尝试多种拟合函数，返回最佳拟合结果 / Try multiple fitting functions and return the best
    只使用线性和指数拟合
    """
    fit_functions = [
        linear_fit,
        exponential_fit,
    ]
    
    results = []
    for fit_func in fit_functions:
        result = fit_func(x, y)
        if result is not None:
            results.append(result)
    
    if not results:
        return None
    
    # 选择 R² 最高的拟合
    best = max(results, key=lambda r: r['r_squared'])
    best['all_fits'] = results  # 保存所有拟合结果用于比较
    
    return best


def get_best_fit_for_parameter(x: np.ndarray, y: np.ndarray, parameter: str, peak_idx: int) -> dict | None:
    """
    根据参数类型获取最佳拟合 / Get best fit based on parameter type
    
    Args:
        x: 自变量数据 / Independent variable
        y: 因变量数据 / Dependent variable
        parameter: 参数名称 / Parameter name
        peak_idx: 峰索引 / Peak index
    
    Returns:
        拟合结果字典 / Fit result dictionary
    """
    # 获取该参数的预设拟合模型
    model_config = BEST_FIT_MODELS.get(parameter)
    
    if model_config is None:
        # 如果没有预设，尝试所有模型并选择最佳
        return find_best_fit(x, y)
    
    # 如果是字典类型（针对不同峰有不同模型）
    if isinstance(model_config, dict):
        model_type = model_config.get(peak_idx, model_config.get('default'))
    else:
        model_type = model_config
    
    # 根据模型类型调用对应的拟合函数
    fit_functions = {
        'linear': linear_fit,
        'quadratic': quadratic_fit,
        'exponential': exponential_fit,
        'logarithmic': logarithmic_fit,
        'power': power_fit,
        'hyperbolic': hyperbolic_fit,
    }
    
    fit_func = fit_functions.get(model_type)
    if fit_func is None:
        # 如果模型类型无效，尝试所有模型
        return find_best_fit(x, y)
    
    result = fit_func(x, y)
    
    # 如果预设模型拟合失败，尝试所有模型
    if result is None:
        print(f"    警告: {model_type} 拟合失败，尝试其他模型...")
        return find_best_fit(x, y)
    
    return result


def evaluate_fit_function(x: np.ndarray, fit_result: dict) -> np.ndarray:
    """
    根据拟合结果计算拟合曲线 / Calculate fitted curve from fit result
    
    Args:
        x: 自变量数组 / Independent variable array
        fit_result: 拟合结果字典 / Fit result dictionary
    
    Returns:
        拟合曲线的 y 值 / Fitted y values
    """
    fit_type = fit_result['type']
    params = fit_result['params']
    
    if fit_type == 'linear':
        return params['a'] * x + params['b']
    elif fit_type == 'quadratic':
        return params['a'] * x**2 + params['b'] * x + params['c']
    elif fit_type == 'exponential':
        return params['a'] * np.exp(params['b'] * x)
    elif fit_type == 'logarithmic':
        return params['a'] * np.log(np.maximum(x, 1e-10)) + params['b']
    elif fit_type == 'power':
        return params['a'] * np.power(np.maximum(x, 1e-10), params['b'])
    elif fit_type == 'hyperbolic':
        return params['a'] / (x + params['b']) + params['c']
    else:
        return x * 0  # fallback


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
    # 提取数字（包括小数点和科学计数法）/ Extract numbers (including decimals and scientific notation)
    match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', str(concentration_str))
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

def apply_concentration_filter(concentrations: np.ndarray, values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    应用浓度过滤，排除指定的浓度点 / Apply concentration filter to exclude specified points
    
    Args:
        concentrations: 浓度数组 / Concentration array
        values: 参数值数组 / Parameter value array
        valid_mask: 有效数据掩码 / Valid data mask
    
    Returns:
        过滤后的掩码 / Filtered mask
    """
    fit_mask = valid_mask.copy()
    
    # 忽略特定浓度点 / Ignore specific concentrations
    if IGNORE_CONCENTRATIONS:
        for ignore_conc in IGNORE_CONCENTRATIONS:
            # 使用小的容差来比较浮点数 / Use small tolerance for float comparison
            fit_mask = fit_mask & (np.abs(concentrations - ignore_conc) > 1e-6)
    
    return fit_mask


def plot_parameter(
    df: pd.DataFrame,
    parameter: str,
    n_peaks: int,
    output_dir: Path,
    fit_records: List[dict]
) -> None:
    """
    绘制单个参数随浓度变化的图 / Plot single parameter vs concentration
    
    Args:
        df: 数据框 / DataFrame
        parameter: 参数名称（不含Peak前缀）/ Parameter name (without Peak prefix)
        n_peaks: 峰数量 / Number of peaks
        output_dir: 输出目录 / Output directory
        fit_records: 拟合结果记录 / Fit records
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
        
        # 应用浓度过滤 / Apply concentration filter
        fit_mask = apply_concentration_filter(concentrations, values, valid_mask)
        
        # 检查绘图数据点数量
        if fit_mask.sum() == 0:
            print(f"  ⚠️ Peak{peak_idx} {parameter}: 所有数据点都被忽略")
            continue
        
        n_ignored = valid_mask.sum() - fit_mask.sum()
        if n_ignored > 0:
            ignored_concentrations = concentrations[valid_mask & ~fit_mask]
            print(f"  ℹ️ Peak{peak_idx} {parameter}: 已忽略 {n_ignored} 个浓度点 {list(ignored_concentrations)}")
        
        style = PEAK_STYLES[(peak_idx - 1) % len(PEAK_STYLES)]
        
        # 使用 fit_mask 绘制散点，被忽略的点不显示
        ax.plot(
            concentrations[fit_mask],
            values[fit_mask],
            marker=style['marker'],
            color=style['color'],
            linewidth=2,
            markersize=8,
            label=f'Peak {peak_idx}',
            alpha=0.8
        )
        
        if fit_mask.sum() < 2:
            print(f"  ⚠️ Peak{peak_idx} {parameter}: 拟合数据点不足 (< 2)")
            continue
        
        # 使用预设的最佳拟合模型 / Use preset best fit model
        fit_result = get_best_fit_for_parameter(
            concentrations[fit_mask], 
            values[fit_mask], 
            parameter, 
            peak_idx
        )
        
        if fit_result:
            # 使用实际绘制的数据点范围（fit_mask）绘制拟合曲线
            x_min_plot = concentrations[fit_mask].min()
            x_max_plot = concentrations[fit_mask].max()
            x_fit = np.linspace(x_min_plot, x_max_plot, 100)
            y_fit = evaluate_fit_function(x_fit, fit_result)
            
            fit_label = f"Peak {peak_idx} fit: {fit_result['type']} (R²={fit_result['r_squared']:.4f})"
            ax.plot(
                x_fit,
                y_fit,
                color=style['color'],
                linestyle='--',
                linewidth=1.5,
                alpha=0.9,
                label=fit_label
            )
            
            # 记录拟合结果
            record = {
                'parameter': parameter,
                'peak': peak_idx,
                'fit_type': fit_result['type'],
                'formula': fit_result['formula'],
                'r_squared': fit_result['r_squared'],
                'rmse': fit_result['rmse'],
                'n_points': fit_result['n_points'],
                'x_min': fit_result['x_min'],
                'x_max': fit_result['x_max'],
            }
            # 添加参数
            for key, val in fit_result['params'].items():
                record[f'param_{key}'] = val
            
            fit_records.append(record)
            
            # 输出拟合结果
            print(f"  Peak{peak_idx} {parameter}: {fit_result['type']:12s} - R²={fit_result['r_squared']:.6f}, RMSE={fit_result['rmse']:.6f}")
    
    # 设置标签和标题 / Set labels and title
    param_info = PARAM_INFO.get(parameter, {})
    ylabel = param_info.get('ylabel', parameter)
    title = param_info.get('label', parameter)
    
    ax.set_xlabel('Concentration (μM)', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(f'{title} vs Concentration', fontsize=14, fontweight='bold', pad=15)
    
    # 图例 / Legend
    ax.legend(loc='best', framealpha=0.9, fontsize=9)
    
    # 设置坐标轴范围 / Set axis limits (data-driven)
    x_min = concentrations.min()
    x_margin = max(0.5, x_min * 0.05)  # 5% margin or at least 0.5
    ax.set_xlim(left=max(0, x_min - x_margin))
    
    # 添加网格线 / Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 紧凑布局 / Tight layout
    plt.tight_layout()
    
    # 保存图片 / Save figure
    output_file = output_dir / f'{parameter}_vs_concentration.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {output_file.name}")


def plot_all_parameters(df: pd.DataFrame, n_peaks: int, output_dir: Path, fit_records: List[dict]) -> None:
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
            plot_parameter(df, param, n_peaks, output_dir, fit_records)
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
            
            # 应用浓度过滤，被忽略的点不显示
            fit_mask = apply_concentration_filter(concentrations, values, valid_mask)
            
            if fit_mask.sum() == 0:
                continue
            
            style = PEAK_STYLES[(peak_idx - 1) % len(PEAK_STYLES)]
            
            ax.plot(
                concentrations[fit_mask],
                values[fit_mask],
                marker=style['marker'],
                color=style['color'],
                linewidth=2,
                markersize=6,
                label=f'Peak {peak_idx}',
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
        
        # 设置坐标轴范围 / Set axis limits (data-driven)
        x_min = concentrations.min()
        x_margin = max(0.5, x_min * 0.05)
        ax.set_xlim(left=max(0, x_min - x_margin))
        
        # 添加网格线 / Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.suptitle('Fitting Parameters vs Concentration - Summary', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # 保存 / Save
    output_file = output_dir / 'summary_plot.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 汇总图已保存 / Summary plot saved: {output_file}")


def save_linear_fit_results(fit_records: List[dict], output_path: Path) -> None:
    """保存拟合结果到CSV / Save fit metrics to CSV"""
    if not fit_records:
        print("\n⚠️ 无可用的拟合结果 / No fit results to save.")
        return
    
    df_fit = pd.DataFrame(fit_records)
    df_fit.sort_values(['parameter', 'peak'], inplace=True)
    df_fit.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✅ 拟合结果已保存 / Fit results saved: {output_path}")
    
    # 打印最佳拟合统计 / Print best fit statistics
    print(f"\n📊 拟合类型统计 / Fit type statistics:")
    fit_type_counts = df_fit['fit_type'].value_counts()
    for fit_type, count in fit_type_counts.items():
        print(f"   {fit_type:12s}: {count:2d} 次")
    
    # 打印平均 R²
    avg_r2 = df_fit['r_squared'].mean()
    print(f"\n   平均 R² / Average R²: {avg_r2:.4f}")


# ============================================================================
# 主程序 / Main Program
# ============================================================================

def main() -> None:
    """主程序入口 / Main program entry"""
    # 获取脚本所在目录 / Get script directory
    script_dir = Path(__file__).parent
    csv_path = script_dir / INPUT_CSV
    
    if not csv_path.exists():
        print(f"❌ 错误 / Error: 文件不存在 / File not found: {csv_path}")
        return
    
    print("="*70)
    print("量子点光谱拟合结果可视化 / Quantum Dot Fitting Results Visualization")
    print("="*70)
    
    # 显示浓度过滤配置 / Display concentration filter configuration
    print("\n📌 浓度过滤配置 / Concentration Filter Configuration:")
    if IGNORE_CONCENTRATIONS:
        print(f"   忽略的浓度点 / Ignored concentration points: {IGNORE_CONCENTRATIONS} μM")
    else:
        print(f"   忽略的浓度点 / Ignored concentration points: 无 / None")
    print()
    
    try:
        # 加载数据 / Load data
        df, n_peaks = load_and_process_data(csv_path)
        
        # 创建输出目录 / Create output directory
        output_dir = script_dir / OUTPUT_DIR
        
        # 用于收集线性拟合结果 / Collect linear fit metrics
        fit_records: List[dict] = []
        
        # 绘制所有参数 / Plot all parameters
        plot_all_parameters(df, n_peaks, output_dir, fit_records)
        
        # 创建汇总图 / Create summary plot
        create_summary_plot(df, n_peaks, output_dir)

        # 保存线性拟合结果 / Save linear fit metrics
        save_linear_fit_results(fit_records, script_dir / LINEAR_FIT_CSV)
        
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
