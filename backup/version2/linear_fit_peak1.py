"""
线性拟合 - Peak1 参数分析
对拟合结果中的 Peak1 参数进行线性拟合：
- A (Peak1_A_prime)
- max_intensity (Peak1_max_intensity)
- integrated_intensity (Peak1_integrated_intensity)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

def extract_concentration_value(conc_str):
    """
    从浓度字符串中提取数值
    例如: '0.2μM' -> 0.2, '240μM' -> 240, 'Blank' -> 0
    """
    if conc_str == 'Blank':
        return 0.0
    
    # 移除 'μM' 并转换为浮点数
    return float(conc_str.replace('μM', ''))

def perform_linear_fit(x, y, param_name):
    """
    执行线性拟合并返回结果
    
    参数:
        x: 浓度值数组
        y: 待拟合参数值数组
        param_name: 参数名称
    
    返回:
        dict: 包含拟合结果的字典
    """
    # 线性拟合
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # 计算拟合值
    y_fit = slope * x + intercept
    
    # 计算残差
    residuals = y - y_fit
    
    # 计算 RMSE
    rmse = np.sqrt(np.mean(residuals**2))
    
    return {
        'param_name': param_name,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'rmse': rmse,
        'y_fit': y_fit
    }

def plot_linear_fit(x, y, fit_result, param_name, save_path=None):
    """
    绘制线性拟合图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：数据点和拟合线
    ax1.scatter(x, y, s=50, alpha=0.6, label='实验数据', color='blue')
    ax1.plot(x, fit_result['y_fit'], 'r-', linewidth=2, label='线性拟合')
    
    ax1.set_xlabel('浓度 (μM)', fontsize=12)
    ax1.set_ylabel(param_name, fontsize=12)
    ax1.set_title(f'{param_name} vs 浓度', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 添加拟合方程和统计信息
    equation_text = (
        f'y = {fit_result["slope"]:.3e}x + {fit_result["intercept"]:.3e}\n'
        f'R² = {fit_result["r_squared"]:.6f}\n'
        f'p-value = {fit_result["p_value"]:.3e}\n'
        f'RMSE = {fit_result["rmse"]:.3e}'
    )
    ax1.text(0.05, 0.95, equation_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 右图：残差图
    residuals = y - fit_result['y_fit']
    ax2.scatter(x, residuals, s=50, alpha=0.6, color='green')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('浓度 (μM)', fontsize=12)
    ax2.set_ylabel('残差', fontsize=12)
    ax2.set_title('残差分布', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'图表已保存至: {save_path}')
    
    plt.show()

def main():
    """
    主函数
    """
    # 读取拟合结果
    print('正在读取拟合结果...')
    df = pd.read_csv('fit_results_batch.csv')
    
    # 提取浓度数值
    df['concentration_value'] = df['Concentration'].apply(extract_concentration_value)
    
    # 定义要拟合的 Peak1 参数
    parameters = {
        'Peak1_A_prime': 'A (Peak1)',
        'Peak1_max_intensity': '最大强度 (Peak1)',
        'Peak1_integrated_intensity': '积分强度 (Peak1)'
    }
    
    # 存储所有拟合结果
    all_results = []
    
    print('\n' + '='*60)
    print('Peak1 参数线性拟合结果')
    print('='*60)
    
    # 对每个参数进行线性拟合
    for param_col, param_display_name in parameters.items():
        print(f'\n【{param_display_name}】')
        print('-'*60)
        
        # 提取数据
        x = df['concentration_value'].values
        y = df[param_col].values
        
        # 执行线性拟合
        fit_result = perform_linear_fit(x, y, param_display_name)
        all_results.append(fit_result)
        
        # 打印结果
        print(f'拟合方程: y = {fit_result["slope"]:.6e} * x + {fit_result["intercept"]:.6e}')
        print(f'相关系数 R: {fit_result["r_value"]:.6f}')
        print(f'决定系数 R²: {fit_result["r_squared"]:.6f}')
        print(f'p值: {fit_result["p_value"]:.6e}')
        print(f'标准误差: {fit_result["std_err"]:.6e}')
        print(f'RMSE: {fit_result["rmse"]:.6e}')
        
        # 绘制拟合图
        save_path = f'parameter_plots/linear_fit_{param_col}.png'
        plot_linear_fit(x, y, fit_result, param_display_name, save_path)
    
    # 保存拟合结果到 CSV
    results_df = pd.DataFrame([
        {
            '参数': r['param_name'],
            '斜率': r['slope'],
            '截距': r['intercept'],
            '相关系数R': r['r_value'],
            '决定系数R²': r['r_squared'],
            'p值': r['p_value'],
            '标准误差': r['std_err'],
            'RMSE': r['rmse']
        }
        for r in all_results
    ])
    
    output_file = 'linear_fit_results_peak1.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f'\n拟合结果已保存至: {output_file}')
    
    print('\n' + '='*60)
    print('分析完成！')
    print('='*60)

if __name__ == '__main__':
    main()
