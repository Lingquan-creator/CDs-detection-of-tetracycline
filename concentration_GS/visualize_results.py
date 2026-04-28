"""Visualize outputs produced by concentration_GS/train.py."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def format_plot_number(value, _pos=None):
    if not np.isfinite(value):
        return ""
    text = f"{value:.6g}"
    return "0" if text in ("", "-0") else text


def format_figure_numbers(fig):
    formatter = FuncFormatter(format_plot_number)
    for ax in fig.axes:
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)


plt.rcParams.update({"font.family": ["DejaVu Sans"], "axes.unicode_minus": False})


class ResultVisualizer:
    def __init__(self, output_dir: str = "concentration_GS/amortized_output"):
        self.output_dir = Path(output_dir)
        self.train_summary = None
        self.loo_summary = None
        self.train_probs = None
        self.loo_probs = None
        
    def load_data(self):
        print("Loading result files...")
        
        train_summary_path = self.output_dir / "train_posterior_summary.csv"
        loo_summary_path = self.output_dir / "loo_posterior_summary.csv"
        
        if train_summary_path.exists():
            self.train_summary = pd.read_csv(train_summary_path)
            print(f"  Training summary: {len(self.train_summary)} samples")
        
        if loo_summary_path.exists():
            self.loo_summary = pd.read_csv(loo_summary_path)
            print(f"  LOO summary: {len(self.loo_summary)} samples")
        
        train_probs_path = self.output_dir / "train_posterior_probabilities.csv"
        loo_probs_path = self.output_dir / "loo_posterior_probabilities.csv"
        
        if train_probs_path.exists():
            self.train_probs = pd.read_csv(train_probs_path)
            print("  Training probability distribution loaded")
        
        if loo_probs_path.exists():
            self.loo_probs = pd.read_csv(loo_probs_path)
            print("  LOO probability distribution loaded")
        
        print("Data loading completed.\n")
    
    def plot_prediction_scatter(self, save_path: str = None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        if self.train_summary is not None:
            ax = axes[0]
            self._plot_single_scatter(ax, self.train_summary, "Training Set")
        
        if self.loo_summary is not None:
            ax = axes[1]
            self._plot_single_scatter(ax, self.loo_summary, "LOO Cross-Validation")
        
        format_figure_numbers(fig)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved scatter plot -> {save_path}")
        plt.close()
    
    def _plot_single_scatter(self, ax, df, title):
        y_true = df['conc_true'].values
        y_pred = df['posterior_mean'].values
        
        residuals = y_pred - y_true
        r2 = 1 - np.sum(residuals**2) / np.sum((y_true - y_true.mean())**2)
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        
        ax.errorbar(
            y_true, y_pred,
            yerr=df['posterior_std'].values,
            fmt='o', markersize=6, capsize=4,
            color='#2E86AB', ecolor='lightgray', alpha=0.7,
            label='Prediction +/- std'
        )
        
        all_vals = np.concatenate([y_true, y_pred])
        lims = [all_vals.min() * 0.95, all_vals.max() * 1.05]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5, label='Ideal prediction')
        
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('True concentration (uM)', fontsize=12)
        ax.set_ylabel('Predicted concentration (uM)', fontsize=12)
        ax.set_title(f'{title}\nR2={r2:.6g}, RMSE={rmse:.6g} uM, MAE={mae:.6g} uM',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def plot_residuals(self, save_path: str = None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        if self.train_summary is not None:
            self._plot_residual_analysis(axes[:, 0], self.train_summary, "Training Set")
        
        if self.loo_summary is not None:
            self._plot_residual_analysis(axes[:, 1], self.loo_summary, "LOO Cross-Validation")
        
        format_figure_numbers(fig)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved residual plot -> {save_path}")
        plt.close()
    
    def _plot_residual_analysis(self, axes, df, title):
        y_true = df['conc_true'].values
        residuals = df['residual_mean'].values
        
        ax = axes[0]
        ax.scatter(y_true, residuals, alpha=0.6, s=60, color='#A23B72')
        ax.axhline(0, color='k', linestyle='--', linewidth=1.5)
        ax.set_xlabel('True concentration (uM)', fontsize=11)
        ax.set_ylabel('Residual (uM)', fontsize=11)
        ax.set_title(f'{title} - Residuals', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        ax.hist(residuals, bins=15, color='#F18F01', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='k', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Residual (uM)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{title} - Residual Histogram\nMean={residuals.mean():.6g}, Std={residuals.std():.6g}',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_uncertainty(self, save_path: str = None):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        if self.train_summary is not None:
            self._plot_single_uncertainty(axes[0], self.train_summary, "Training Set")
        
        if self.loo_summary is not None:
            self._plot_single_uncertainty(axes[1], self.loo_summary, "LOO Cross-Validation")
        
        format_figure_numbers(fig)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved uncertainty plot -> {save_path}")
        plt.close()
    
    def _plot_single_uncertainty(self, ax, df, title):
        x = np.arange(len(df))
        y_true = df['conc_true'].values
        y_pred = df['posterior_mean'].values
        ci_low = df['ci_90_low'].values
        ci_high = df['ci_90_high'].values
        
        ax.scatter(x, y_true, color='black', s=80, marker='o', 
                   label='True value', zorder=3, alpha=0.8)
        
        ax.errorbar(x, y_pred, 
                    yerr=[y_pred - ci_low, ci_high - y_pred],
                    fmt='s', markersize=6, capsize=5,
                    color='#C1121F', ecolor='#780000', alpha=0.7,
                    label='Prediction +/- 90% CI', zorder=2)
        
        ax.set_xlabel('Sample index', fontsize=12)
        ax.set_ylabel('Concentration (uM)', fontsize=12)
        ax.set_title(f'{title} - Prediction Uncertainty', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x[::2])
    
    def plot_posterior_distributions(self, sample_indices: list = None, 
                                     save_path: str = None):
        if self.loo_probs is None:
            print("LOO probability distribution data not found")
            return
        
        if sample_indices is None:
            unique_files = self.loo_probs['filename'].unique()
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
            
            true_conc = sample_data['ref_concentration'].iloc[0]
            
            bin_centers = sample_data['ref_concentration'].values
            probabilities = sample_data['bin_probability'].values
            
            mask = probabilities > 0.01
            
            ax.bar(bin_centers[mask], probabilities[mask], 
                   width=np.diff(bin_centers)[0] * 0.8 if len(bin_centers) > 1 else 1,
                   color='#06A77D', alpha=0.7, edgecolor='black')
            
            ax.axvline(true_conc, color='red', linestyle='--', 
                      linewidth=2, label=f'True={true_conc:.6g} uM')
            
            if self.loo_summary is not None:
                summary_row = self.loo_summary[self.loo_summary['filename'] == filename]
                if not summary_row.empty:
                    post_mean = summary_row['posterior_mean'].iloc[0]
                    ax.axvline(post_mean, color='blue', linestyle='-', 
                              linewidth=2, alpha=0.7, label=f'Posterior mean={post_mean:.6g} uM')
            
            ax.set_xlabel('Concentration (uM)', fontsize=10)
            ax.set_ylabel('Probability density', fontsize=10)
            ax.set_title(f'Sample: {filename}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, axis='y')
        
        for idx in range(n_samples, len(axes)):
            axes[idx].axis('off')
        
        format_figure_numbers(fig)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved posterior distribution plot -> {save_path}")
        plt.close()
    
    def print_metrics_summary(self):
        print("\n" + "="*60)
        print("Performance Metrics Summary".center(60))
        print("="*60 + "\n")
        
        for name, df in [("Training Set", self.train_summary), 
                         ("LOO Cross-Validation", self.loo_summary)]:
            if df is None:
                continue
            
            y_true = df['conc_true'].values
            y_pred = df['posterior_mean'].values
            residuals = y_pred - y_true
            
            r2 = 1 - np.sum(residuals**2) / np.sum((y_true - y_true.mean())**2)
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            mape = np.mean(np.abs(residuals / y_true)) * 100
            
            avg_std = df['posterior_std'].mean()
            avg_ci_width = (df['ci_90_high'] - df['ci_90_low']).mean()
            
            print(f"{name}:")
            print(f"  Sample count:       {len(df)}")
            print(f"  R2:                 {r2:.6g}")
            print(f"  RMSE:               {rmse:.6g} uM")
            print(f"  MAE:                {mae:.6g} uM")
            print(f"  MAPE:               {mape:.6g} %")
            print(f"  Mean posterior std: {avg_std:.6g} uM")
            print(f"  Mean 90% CI width:  {avg_ci_width:.6g} uM")
            print(f"  Mean NLL:           {df['nll'].mean():.6g}")
            print()
        
        print("="*60 + "\n")
    
    def create_comprehensive_report(self, save_dir: str = None):
        if save_dir is None:
            save_dir = self.output_dir
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating visualization report...")
        print("-" * 60)
        
        self.plot_prediction_scatter(save_dir / "1_prediction_scatter.png")
        
        self.plot_residuals(save_dir / "2_residuals_analysis.png")
        
        self.plot_uncertainty(save_dir / "3_uncertainty_analysis.png")
        
        self.plot_posterior_distributions(save_path=save_dir / "4_posterior_distributions.png")
        
        self.print_metrics_summary()
        
        print("-" * 60)
        print(f"All visualization figures saved to: {save_dir}")
        print("Report generation completed.\n")


def main():
    viz = ResultVisualizer()
    
    viz.load_data()
    
    viz.create_comprehensive_report()


if __name__ == "__main__":
    main()
