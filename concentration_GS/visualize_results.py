"""
可视化 concentration_GS/train.py 输出结果

功能：
1. 预测值 vs 真实值散点图（训练集和LOO）
2. 残差分布图
3. 后验不确定性可视化（误差棒和置信区间）
4. 个别样本的后验概率分布
5. 性能指标汇总表
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self, output_dir: str = "concentration_GS/amortized_output"):
        self.output_dir = Path(output_dir)
        self.train_summary = None
        self.loo_summary = None
        self.train_probs = None
        self.loo_probs = None
        
    def load_data(self):
        """加载所有CSV文件"""
        print("正在加载结果文件...")
        
        # 加载汇总数据
        train_summary_path = self.output_dir / "train_posterior_summary.csv"
        loo_summary_path = self.output_dir / "loo_posterior_summary.csv"
        
        if train_summary_path.exists():
            self.train_summary = pd.read_csv(train_summary_path)
            print(f"  ✓ 训练集汇总: {len(self.train_summary)} 样本")
        
        if loo_summary_path.exists():
            self.loo_summary = pd.read_csv(loo_summary_path)
            print(f"  ✓ LOO汇总: {len(self.loo_summary)} 样本")
        
        # 加载概率分布数据
        train_probs_path = self.output_dir / "train_posterior_probabilities.csv"
        loo_probs_path = self.output_dir / "loo_posterior_probabilities.csv"
        
        if train_probs_path.exists():
            self.train_probs = pd.read_csv(train_probs_path)
            print(f"  ✓ 训练集概率分布")
        
        if loo_probs_path.exists():
            self.loo_probs = pd.read_csv(loo_probs_path)
            print(f"  ✓ LOO概率分布")
        
        print("数据加载完成！\n")
    
    def plot_prediction_scatter(self, save_path: str = None):
        """绘制预测值 vs 真实值散点图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 训练集
        if self.train_summary is not None:
            ax = axes[0]
            self._plot_single_scatter(ax, self.train_summary, "训练集")
        
        # LOO交叉验证
        if self.loo_summary is not None:
            ax = axes[1]
            self._plot_single_scatter(ax, self.loo_summary, "LOO交叉验证")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存散点图: {save_path}")
        plt.close()
    
    def _plot_single_scatter(self, ax, df, title):
        """绘制单个散点图"""
        # 计算性能指标
        y_true = df['conc_true'].values
        y_pred = df['posterior_mean'].values
        
        residuals = y_pred - y_true
        r2 = 1 - np.sum(residuals**2) / np.sum((y_true - y_true.mean())**2)
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        
        # 绘制散点和误差棒
        ax.errorbar(
            y_true, y_pred,
            yerr=df['posterior_std'].values,
            fmt='o', markersize=6, capsize=4,
            color='#2E86AB', ecolor='lightgray', alpha=0.7,
            label='预测值 ± 标准差'
        )
        
        # 绘制理想对角线
        all_vals = np.concatenate([y_true, y_pred])
        lims = [all_vals.min() * 0.95, all_vals.max() * 1.05]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5, label='理想预测')
        
        # 设置坐标轴
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('真实浓度 (μM)', fontsize=12)
        ax.set_ylabel('预测浓度 (μM)', fontsize=12)
        ax.set_title(f'{title}\nR²={r2:.4f}, RMSE={rmse:.2f} μM, MAE={mae:.2f} μM',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def plot_residuals(self, save_path: str = None):
        """绘制残差分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 训练集残差
        if self.train_summary is not None:
            self._plot_residual_analysis(axes[:, 0], self.train_summary, "训练集")
        
        # LOO残差
        if self.loo_summary is not None:
            self._plot_residual_analysis(axes[:, 1], self.loo_summary, "LOO交叉验证")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存残差图: {save_path}")
        plt.close()
    
    def _plot_residual_analysis(self, axes, df, title):
        """绘制残差分析"""
        y_true = df['conc_true'].values
        residuals = df['residual_mean'].values
        
        # 残差 vs 真实值
        ax = axes[0]
        ax.scatter(y_true, residuals, alpha=0.6, s=60, color='#A23B72')
        ax.axhline(0, color='k', linestyle='--', linewidth=1.5)
        ax.set_xlabel('真实浓度 (μM)', fontsize=11)
        ax.set_ylabel('残差 (μM)', fontsize=11)
        ax.set_title(f'{title} - 残差分布', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 残差直方图
        ax = axes[1]
        ax.hist(residuals, bins=15, color='#F18F01', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='k', linestyle='--', linewidth=1.5)
        ax.set_xlabel('残差 (μM)', fontsize=11)
        ax.set_ylabel('频数', fontsize=11)
        ax.set_title(f'{title} - 残差直方图\nMean={residuals.mean():.2f}, Std={residuals.std():.2f}',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_uncertainty(self, save_path: str = None):
        """绘制不确定性分析图（误差棒和置信区间）"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 训练集
        if self.train_summary is not None:
            self._plot_single_uncertainty(axes[0], self.train_summary, "训练集")
        
        # LOO
        if self.loo_summary is not None:
            self._plot_single_uncertainty(axes[1], self.loo_summary, "LOO交叉验证")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存不确定性图: {save_path}")
        plt.close()
    
    def _plot_single_uncertainty(self, ax, df, title):
        """绘制单个不确定性图"""
        x = np.arange(len(df))
        y_true = df['conc_true'].values
        y_pred = df['posterior_mean'].values
        ci_low = df['ci_90_low'].values
        ci_high = df['ci_90_high'].values
        
        # 绘制真实值
        ax.scatter(x, y_true, color='black', s=80, marker='o', 
                   label='真实值', zorder=3, alpha=0.8)
        
        # 绘制预测值和90%置信区间
        ax.errorbar(x, y_pred, 
                    yerr=[y_pred - ci_low, ci_high - y_pred],
                    fmt='s', markersize=6, capsize=5,
                    color='#C1121F', ecolor='#780000', alpha=0.7,
                    label='预测值 ± 90% CI', zorder=2)
        
        ax.set_xlabel('样本索引', fontsize=12)
        ax.set_ylabel('浓度 (μM)', fontsize=12)
        ax.set_title(f'{title} - 预测不确定性', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x[::2])  # 每隔一个显示标签
    
    def plot_posterior_distributions(self, sample_indices: list = None, 
                                     save_path: str = None):
        """绘制个别样本的后验概率分布"""
        if self.loo_probs is None:
            print("未找到LOO概率分布数据")
            return
        
        # 如果未指定样本，选择几个代表性样本
        if sample_indices is None:
            unique_files = self.loo_probs['filename'].unique()
            # 选择低、中、高浓度各两个样本
            sample_indices = [0, 1, len(unique_files)//2-1, len(unique_files)//2,
                            len(unique_files)-2, len(unique_files)-1]
        
        n_samples = len(sample_indices)
        n_cols = 3
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_samples > 1 else [axes]
        
        unique_files = self.loo_probs['filename'].unique()
        
        for idx, (ax, sample_idx) in enumerate(zip(axes, sample_indices)):
            if sample_idx >= len(unique_files):
                ax.axis('off')
                continue
                
            filename = unique_files[sample_idx]
            sample_data = self.loo_probs[self.loo_probs['filename'] == filename]
            
            # 获取真实浓度
            true_conc = sample_data['ref_concentration'].iloc[0]
            
            # 绘制概率分布（柱状图）
            bin_centers = sample_data['ref_concentration'].values
            probabilities = sample_data['bin_probability'].values
            
            # 只绘制概率 > 0.01的区间
            mask = probabilities > 0.01
            
            ax.bar(bin_centers[mask], probabilities[mask], 
                   width=np.diff(bin_centers)[0] * 0.8 if len(bin_centers) > 1 else 1,
                   color='#06A77D', alpha=0.7, edgecolor='black')
            
            # 标记真实值
            ax.axvline(true_conc, color='red', linestyle='--', 
                      linewidth=2, label=f'真实值={true_conc:.1f} μM')
            
            # 获取后验均值和众数
            if self.loo_summary is not None:
                summary_row = self.loo_summary[self.loo_summary['filename'] == filename]
                if not summary_row.empty:
                    post_mean = summary_row['posterior_mean'].iloc[0]
                    ax.axvline(post_mean, color='blue', linestyle='-', 
                              linewidth=2, alpha=0.7, label=f'后验均值={post_mean:.1f} μM')
            
            ax.set_xlabel('浓度 (μM)', fontsize=10)
            ax.set_ylabel('概率密度', fontsize=10)
            ax.set_title(f'样本: {filename}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 隐藏多余的子图
        for idx in range(n_samples, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 保存后验分布图: {save_path}")
        plt.close()
    
    def print_metrics_summary(self):
        """打印性能指标汇总表"""
        print("\n" + "="*60)
        print("性能指标汇总".center(60))
        print("="*60 + "\n")
        
        for name, df in [("训练集", self.train_summary), 
                         ("LOO交叉验证", self.loo_summary)]:
            if df is None:
                continue
            
            y_true = df['conc_true'].values
            y_pred = df['posterior_mean'].values
            residuals = y_pred - y_true
            
            r2 = 1 - np.sum(residuals**2) / np.sum((y_true - y_true.mean())**2)
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            mape = np.mean(np.abs(residuals / y_true)) * 100
            
            # 平均不确定性
            avg_std = df['posterior_std'].mean()
            avg_ci_width = (df['ci_90_high'] - df['ci_90_low']).mean()
            
            print(f"{name}:")
            print(f"  样本数量:         {len(df)}")
            print(f"  R² 系数:          {r2:.4f}")
            print(f"  RMSE:             {rmse:.2f} μM")
            print(f"  MAE:              {mae:.2f} μM")
            print(f"  MAPE:             {mape:.2f} %")
            print(f"  平均后验标准差:   {avg_std:.2f} μM")
            print(f"  平均90% CI宽度:   {avg_ci_width:.2f} μM")
            print(f"  平均NLL:          {df['nll'].mean():.4f}")
            print()
        
        print("="*60 + "\n")
    
    def create_comprehensive_report(self, save_dir: str = None):
        """生成完整的可视化报告"""
        if save_dir is None:
            save_dir = self.output_dir
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n正在生成可视化报告...")
        print("-" * 60)
        
        # 1. 散点图
        self.plot_prediction_scatter(save_dir / "1_prediction_scatter.png")
        
        # 2. 残差图
        self.plot_residuals(save_dir / "2_residuals_analysis.png")
        
        # 3. 不确定性图
        self.plot_uncertainty(save_dir / "3_uncertainty_analysis.png")
        
        # 4. 后验分布图
        self.plot_posterior_distributions(save_path=save_dir / "4_posterior_distributions.png")
        
        # 5. 性能指标
        self.print_metrics_summary()
        
        print("-" * 60)
        print(f"✓ 所有可视化图表已保存至: {save_dir}")
        print("✓ 报告生成完成！\n")


def main():
    """主函数"""
    # 创建可视化器
    viz = ResultVisualizer()
    
    # 加载数据
    viz.load_data()
    
    # 生成完整报告
    viz.create_comprehensive_report()


if __name__ == "__main__":
    main()
