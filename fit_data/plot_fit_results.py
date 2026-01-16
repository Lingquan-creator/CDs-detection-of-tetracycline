"""
为每个浓度生成拟合结果图
根据 results_spec.csv 中的拟合参数绘制拟合曲线和原始数据
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def wavelength_to_energy(wavelength_nm):
    """将波长(nm)转换为能量(eV)"""
    # 处理标量和数组两种情况
    if np.isscalar(wavelength_nm):
        if wavelength_nm == 0:
            return 0
        return 1239.84193 / wavelength_nm
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(wavelength_nm != 0, 1239.84193 / wavelength_nm, 0)


def multi_peak_model(wavelength_nm, B, *peak_params):
    """
    多峰模型函数
    
    参数:
    - wavelength_nm: 波长数组
    - B: 背景
    - peak_params: 展平的峰参数 [A1, k1, lambda0_1, sigma_E_1, A2, k2, lambda0_2, sigma_E_2, ...]
    """
    n_peaks = len(peak_params) // 4
    intensity = np.full_like(wavelength_nm, B, dtype=float)
    
    E = wavelength_to_energy(wavelength_nm)
    
    for i in range(n_peaks):
        A_prime = peak_params[4*i]
        k_m = peak_params[4*i + 1]
        lambda0_m = peak_params[4*i + 2]
        sigma_E_m = peak_params[4*i + 3]
        
        E0_m = wavelength_to_energy(lambda0_m)
        A_m = A_prime * np.exp(k_m * lambda0_m)
        
        # 高斯峰
        gaussian_term = np.exp(-0.5 * ((E - E0_m) / sigma_E_m) ** 2)
        intensity += A_m * np.exp(-k_m * wavelength_nm) * gaussian_term
    
    return intensity


def plot_single_concentration(concentration, wavelength, intensity, params, output_dir):
    """
    为单个浓度绘制拟合图
    
    参数:
    - concentration: 浓度标签
    - wavelength: 波长数据
    - intensity: 强度数据
    - params: 拟合参数 (Series from results_spec.csv)
    - output_dir: 输出目录
    """
    # 提取拟合参数
    B = params['Background_B']
    
    # 提取峰参数
    peak_params = []
    peak_idx = 1
    while f'Peak{peak_idx}_A_prime' in params.index:
        A_prime = params[f'Peak{peak_idx}_A_prime']
        k = params[f'Peak{peak_idx}_k']
        lambda0 = params[f'Peak{peak_idx}_lambda0_nm']
        sigma_E = params[f'Peak{peak_idx}_sigma_E_eV']
        peak_params.extend([A_prime, k, lambda0, sigma_E])
        peak_idx += 1
    
    n_peaks = (peak_idx - 1)
    
    # 生成拟合曲线
    fitted_intensity = multi_peak_model(wavelength, B, *peak_params)
    
    # 生成单峰曲线
    individual_peaks = []
    for i in range(n_peaks):
        single_peak_params = [0] * len(peak_params)
        single_peak_params[4*i:4*i+4] = peak_params[4*i:4*i+4]
        peak_curve = multi_peak_model(wavelength, 0, *single_peak_params)
        individual_peaks.append(peak_curve)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制原始数据
    ax.plot(wavelength, intensity, 'ko', markersize=3, alpha=0.5, label='实验数据')
    
    # 绘制总拟合曲线
    ax.plot(wavelength, fitted_intensity, 'r-', linewidth=2, label='总拟合')
    
    # 绘制各个峰
    colors = ['b', 'g', 'm', 'c', 'y']
    for i, peak_curve in enumerate(individual_peaks):
        lambda0 = peak_params[4*i + 2]
        ax.plot(wavelength, peak_curve + B, '--', 
                color=colors[i % len(colors)], 
                linewidth=1.5,
                label=f'峰{i+1} (λ₀={lambda0:.1f} nm)',
                alpha=0.7)
    
    # 绘制背景线
    ax.axhline(y=B, color='gray', linestyle=':', linewidth=1, label=f'背景 (B={B:.0f})')
    
    # 设置标签和标题
    ax.set_xlabel('波长 (nm)', fontsize=12)
    ax.set_ylabel('强度 (a.u.)', fontsize=12)
    ax.set_title(f'浓度: {concentration}\n$R^2$ = {params["R_squared"]:.5f}, RMSE = {params["RMSE"]:.1f}', 
                 fontsize=13, fontweight='bold')
    
    # 添加图例
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    filename = f"fit_{concentration.replace('μM', 'uM')}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存: {filename}")


def main():
    """主函数"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    
    # 文件路径（使用绝对路径）
    data_file = script_dir / "data.csv"
    results_file = script_dir / "results" / "results_spec.csv"
    output_dir = script_dir / "fit_plots"
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    # 读取原始数据
    print("读取数据文件...")
    data_df = pd.read_csv(data_file, index_col=0)
    wavelength = data_df.index.values.astype(float)
    
    # 读取拟合结果
    print("读取拟合结果...")
    results_df = pd.read_csv(results_file, index_col=0)
    
    # 为每个浓度绘图
    print(f"\n开始生成拟合图，共 {len(results_df)} 个浓度...\n")
    
    for concentration in results_df.index:
        try:
            # 获取该浓度的强度数据
            if concentration not in data_df.columns:
                print(f"警告: 数据文件中找不到浓度 {concentration}，跳过")
                continue
            
            intensity = data_df[concentration].values
            
            # 获取拟合参数
            params = results_df.loc[concentration]
            
            # 绘制并保存图片
            plot_single_concentration(concentration, wavelength, intensity, params, output_dir)
            
        except Exception as e:
            import traceback
            print(f"处理浓度 {concentration} 时出错:")
            traceback.print_exc()
            continue
    
    print(f"\n所有图片已保存到目录: {output_dir.absolute()}")
    print(f"共生成 {len(list(output_dir.glob('*.png')))} 张图片")


if __name__ == "__main__":
    main()
