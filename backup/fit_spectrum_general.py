"""
量子点光谱拟合 - 主程序

输入输出处理和批量拟合管理。
Multi-peak spectrum fitting - Main program with I/O and batch processing.

使用方法 / Usage:
    python fit_spectrum_general.py

配置参数请修改 fit_config.py
Configure parameters in fit_config.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

# 导入配置和核心算法 / Import configuration and core algorithms
import fit_config as cfg
from spectrum_fitter import (
    ParamInfo,
    compute_metrics,
    compute_single_peak,
    fit_spectrum,
    peak_position,
)


# ============================================================================
# 数据加载 / Data Loading
# ============================================================================

def load_batch_data(path: Path, skip_rows: int, x_col: int) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """
    加载CSV文件，读取波长和多列强度数据 / Load CSV with wavelength and multiple intensity columns
    
    Returns:
        (波长数组, 强度矩阵, 列标题) / (wavelength array, intensity matrix, column headers)
    """
    # 读取表头 / Read header
    with open(path, 'r', encoding='utf-8') as f:
        for _ in range(skip_rows - 1):
            f.readline()
        header_line = f.readline().strip()
    
    headers = [h.strip() for h in header_line.split(',')]
    
    # 加载数据 / Load data
    data = np.genfromtxt(path, delimiter=",", skip_header=skip_rows)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    
    x_idx = x_col - 1
    lambda_exp = data[:, x_idx]
    
    # 获取所有强度列（除波长列外） / Get all intensity columns (except wavelength)
    y_cols = [i for i in range(data.shape[1]) if i != x_idx]
    intensity_matrix = data[:, y_cols]
    
    # 获取对应的表头 / Get corresponding headers
    intensity_headers = [headers[i] for i in y_cols if i < len(headers)]
    
    # 确保有列名 / Ensure we have column names
    while len(intensity_headers) < intensity_matrix.shape[1]:
        intensity_headers.append(f"Column_{len(intensity_headers)+1}")
    
    return lambda_exp, intensity_matrix, intensity_headers


# ============================================================================
# 结果输出 / Results Output
# ============================================================================

def save_batch_results_csv(output_path: Path, results: List[dict]) -> None:
    """将批量拟合结果保存为CSV / Save batch fitting results to CSV"""
    if not results:
        print("No results to save.")
        return
    
    # 准备CSV行 / Prepare CSV rows
    rows = []
    for result in results:
        row = {
            'Concentration': result['concentration'],
            'R_squared': f"{result['R_squared']:.8f}",
            'RMSE': f"{result['RMSE']:.6e}",
            'Relative_RMSE_%': f"{result['Relative_RMSE_%']:.4f}",
            'Background_B': f"{result['Background_B']:.6e}",
        }
        
        # 添加峰参数 / Add peak parameters
        for i, peak in enumerate(result['peaks'], 1):
            row[f'Peak{i}_A_prime'] = f"{peak['A_prime']:.6e}"
            row[f'Peak{i}_k'] = f"{peak['k']:.6f}"
            row[f'Peak{i}_lambda0_nm'] = f"{peak['lambda0_nm']:.4f}"
            row[f'Peak{i}_sigma_E_eV'] = f"{peak['sigma_E_eV']:.6f}"
            row[f'Peak{i}_peak_position_nm'] = f"{peak['peak_position_nm']:.4f}"
            row[f'Peak{i}_E0_eV'] = f"{peak['E0_eV']:.6f}"
            row[f'Peak{i}_integrated_intensity'] = f"{peak['integrated_intensity']:.6e}"
        
        rows.append(row)
    
    # 写入CSV / Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    print(f"\n✓ Batch results saved to: {output_path}")


def save_summary_text(output_path: Path, results: List[dict]) -> None:
    """保存简要文本摘要 / Save brief text summary"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("光谱拟合批量结果汇总 / Batch Spectrum Fitting Results Summary\n")
        f.write("="*70 + "\n\n")
        
        for result in results:
            f.write(f"样品 / Sample: {result['concentration']}\n")
            f.write(f"  R² = {result['R_squared']:.6f}\n")
            f.write(f"  RMSE = {result['RMSE']:.4e} (相对RMSE = {result['Relative_RMSE_%']:.2f}%)\n")
            f.write(f"  背景 / Background B = {result['Background_B']:.4e}\n")
            
            for i, peak in enumerate(result['peaks'], 1):
                f.write(f"  峰 {i} / Peak {i}:\n")
                f.write(f"    λ = {peak['peak_position_nm']:.2f} nm, "
                       f"σ_E = {peak['sigma_E_eV']:.4f} eV, "
                       f"积分强度 / Integral = {peak['integrated_intensity']:.4e}\n")
            f.write("\n")
    
    print(f"✓ Summary saved to: {output_path}")


def plot_fitting_comparison(
    output_path: Path,
    lambda_exp: np.ndarray,
    intensity_matrix: np.ndarray,
    results: List[dict],
    headers: List[str]
) -> None:
    """
    绘制所有样品的拟合对比图 / Plot fitting comparison for all samples
    """
    n_samples = len(results)
    if n_samples == 0:
        return
    
    # 创建子图 / Create subplots
    n_cols = min(3, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (result, ax) in enumerate(zip(results, axes)):
        concentration = result['concentration']
        I_exp = intensity_matrix[:, idx]
        
        # 过滤有效数据 / Filter valid data
        mask = np.isfinite(lambda_exp) & np.isfinite(I_exp)
        lambda_valid = lambda_exp[mask]
        I_valid = I_exp[mask]
        
        # 重建拟合曲线 / Reconstruct fitted curve
        params = []
        for peak in result['peaks']:
            params.extend([
                peak['A_prime'],
                peak['k'],
                peak['lambda0_nm'],
                peak['sigma_E_eV']
            ])
        params.append(result['Background_B'])
        params = np.array(params)
        
        # 计算拟合曲线 / Calculate fitted curve
        from spectrum_fitter import multi_peak_model
        I_fit = multi_peak_model(params, lambda_valid, len(result['peaks']), cfg.C_EV_NM)
        
        # 绘图 / Plot
        ax.plot(lambda_valid, I_valid, 'k.', markersize=2, alpha=0.5, label='Data')
        ax.plot(lambda_valid, I_fit, 'r-', linewidth=2, label='Fit')
        
        # 绘制各个峰 / Plot individual peaks
        colors = plt.cm.tab10(np.linspace(0, 1, len(result['peaks'])))
        for i, (peak, color) in enumerate(zip(result['peaks'], colors), 1):
            I_peak = compute_single_peak(
                lambda_valid,
                peak['A_prime'],
                peak['k'],
                peak['lambda0_nm'],
                peak['sigma_E_eV'],
                cfg.C_EV_NM
            )
            ax.plot(lambda_valid, I_peak, '--', color=color, linewidth=1,
                   label=f"Peak {i} ({peak['peak_position_nm']:.1f} nm)")
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity')
        ax.set_title(f"{concentration}\nR² = {result['R_squared']:.4f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图 / Hide extra subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison plot saved to: {output_path}")


# ============================================================================
# 主程序 / Main Program
# ============================================================================

def main() -> None:
    """主程序入口 / Main program entry"""
    data_path = Path(cfg.DATA_FILE)
    
    if not data_path.exists():
        print(f"❌ Error: Data file not found: {data_path}")
        return
    
    print("="*70)
    print("量子点光谱批量拟合 / Quantum Dot Spectrum Batch Fitting")
    print("="*70)
    
    try:
        # 加载数据 / Load data
        lambda_exp, intensity_matrix, headers = load_batch_data(
            data_path, cfg.SKIP_ROWS, cfg.X_COLUMN
        )
        
        print(f"\n📂 数据文件 / Data file: {data_path}")
        print(f"📊 波长范围 / Wavelength range: {lambda_exp.min():.1f} ~ {lambda_exp.max():.1f} nm")
        print(f"📈 样品数量 / Number of samples: {intensity_matrix.shape[1]}")
        print(f"📋 样品列表 / Sample list: {', '.join(headers)}\n")
        
        batch_results = []
        
        # 处理每一列 / Process each column
        for col_idx in range(intensity_matrix.shape[1]):
            I_exp = intensity_matrix[:, col_idx]
            concentration = headers[col_idx]
            
            # 过滤有效数据 / Filter valid data
            mask = np.isfinite(lambda_exp) & np.isfinite(I_exp)
            lambda_valid = lambda_exp[mask]
            I_valid = I_exp[mask]
            
            if lambda_valid.size == 0:
                print(f"⚠ 跳过 / Skipping '{concentration}': 无有效数据 / No valid data")
                continue
            
            print(f"\n{'='*70}")
            print(f"🔬 处理 / Processing: {concentration}")
            print(f"{'='*70}")
            print(f"有效数据点 / Valid points: {lambda_valid.size}")
            print(f"强度范围 / Intensity range: {I_valid.min():.2e} ~ {I_valid.max():.2e}")
            
            try:
                # 执行拟合 / Perform fitting
                params_fit, I_fit, residuals, param_info = fit_spectrum(
                    lambda_valid, I_valid,
                    cfg.N_PEAKS, cfg.PEAK_PARAMS, cfg.C_EV_NM,
                    cfg.MAX_FUNCTION_EVALS, cfg.DISPLAY_ITER,
                    cfg.B_INITIAL, cfg.A_MIN, cfg.A_MAX,
                    cfg.K_MIN, cfg.K_MAX,
                    cfg.SIGMA_E_MIN, cfg.SIGMA_E_MAX,
                    cfg.B_MIN, cfg.B_MAX_RATIO
                )
                
                # 计算指标 / Calculate metrics
                R_squared, RMSE = compute_metrics(I_valid, I_fit, residuals)
                rel_RMSE = 100 * RMSE / I_valid.mean()
                
                # 提取峰信息 / Extract peak information
                peaks_info = []
                for i in range(cfg.N_PEAKS):
                    idx = i * 4
                    Ap = params_fit[idx]
                    km = params_fit[idx + 1]
                    lambda0 = params_fit[idx + 2]
                    sigma_E = params_fit[idx + 3]
                    peak_pos = peak_position(Ap, km, lambda0)
                    E0 = cfg.C_EV_NM / peak_pos - 2 * (sigma_E**2) / cfg.C_EV_NM * peak_pos
                    I_peak = compute_single_peak(lambda_valid, Ap, km, lambda0, sigma_E, cfg.C_EV_NM)
                    peak_integral = np.trapezoid(I_peak, lambda_valid)
                    
                    peaks_info.append({
                        'A_prime': Ap,
                        'k': km,
                        'lambda0_nm': lambda0,
                        'sigma_E_eV': sigma_E,
                        'peak_position_nm': peak_pos,
                        'E0_eV': E0,
                        'integrated_intensity': peak_integral,
                    })
                
                B_fit = params_fit[-1]
                
                # 存储结果 / Store results
                batch_results.append({
                    'concentration': concentration,
                    'R_squared': R_squared,
                    'RMSE': RMSE,
                    'Relative_RMSE_%': rel_RMSE,
                    'Background_B': B_fit,
                    'peaks': peaks_info,
                })
                
                # 打印摘要 / Print summary
                print(f"\n✅ 拟合完成 / Fitting completed:")
                print(f"   R² = {R_squared:.6f}")
                print(f"   RMSE = {RMSE:.4e} (相对 / Relative = {rel_RMSE:.2f}%)")
                for i, peak in enumerate(peaks_info, 1):
                    print(f"   峰 {i} / Peak {i}: λ = {peak['peak_position_nm']:.2f} nm, "
                          f"σ_E = {peak['sigma_E_eV']:.4f} eV")
                
            except Exception as e:
                print(f"❌ 拟合失败 / Fitting failed for '{concentration}': {e}")
                continue
        
        # 保存结果 / Save results
        print(f"\n{'='*70}")
        print(f"💾 保存结果 / Saving results...")
        print(f"{'='*70}")
        
        if batch_results:
            if cfg.SAVE_CSV:
                save_batch_results_csv(Path(cfg.CSV_OUTPUT_FILE), batch_results)
            
            if cfg.SAVE_RESULTS:
                save_summary_text(Path(cfg.RESULTS_FILE), batch_results)
            
            if cfg.SAVE_FIGURE:
                plot_fitting_comparison(
                    Path(cfg.FIGURE_FILE),
                    lambda_exp,
                    intensity_matrix,
                    batch_results,
                    headers
                )
        
        print(f"\n{'='*70}")
        print(f"🎉 批量处理完成 / Batch processing completed!")
        print(f"   成功 / Successful: {len(batch_results)}/{intensity_matrix.shape[1]}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ 程序错误 / Program error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
