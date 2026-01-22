import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def set_mpl_style():
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun", "STSong"],
            "axes.unicode_minus": False,
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    set_mpl_style()
    
    # 1. Load Ridge Predictions (LOO CV)
    ridge_cv_path = os.path.join(current_dir, 'ridge_predictions_LOO_CV.csv')
    if not os.path.exists(ridge_cv_path):
        print(f"File not found: {ridge_cv_path}")
        return
    df_ridge = pd.read_csv(ridge_cv_path)
    
    # 2. Load G1 Intensity Data
    rgb_pred_path = os.path.join(current_dir, 'rgb_fit_predictions.csv')
    if not os.path.exists(rgb_pred_path):
        print(f"File not found: {rgb_pred_path}")
        return
    df_rgb = pd.read_csv(rgb_pred_path)
    
    # 3. Load Fit Parameters for G1
    params_path = os.path.join(current_dir, 'rgb_fit_params.csv')
    if not os.path.exists(params_path):
        print(f"File not found: {params_path}")
        return
    df_params = pd.read_csv(params_path)
    
    # Extract G1 parameters
    # Assuming columns: channel,a,b,c,r2,rmse
    g1_params = df_params[df_params['channel'] == 'G1_corr'].iloc[0]
    a = g1_params['a']
    b = g1_params['b']
    c = g1_params['c']
    
    print(f"G1 Parameters: a={a}, b={b}, c={c}")
    
    # 4. Calculate Predicted Concentration from G1 Intensity (Inverse Model)
    # Model: I = a * exp(b * C) + c
    # Inverse: C = (1/b) * ln((I - c) / a)
    
    # Merge df_rgb into df_ridge or vice-versa to align data
    # Both have filename
    df_merged = pd.merge(df_ridge, df_rgb[['filename', 'G1_corr']], on='filename', how='inner')

    # Remove points with concentration > 200
    df_merged = df_merged[df_merged['conc_true'] <= 200]
    
    g1_intensity = df_merged['G1_corr']
    
    # Handle potential domain errors for log
    # (I - c) / a must be > 0.
    term = (g1_intensity - c) / a
    valid_mask = term > 0
    
    if not valid_mask.all():
        print(f"Warning: { (~valid_mask).sum() } points have invalid values for log (Intensity outside fit range).")
    
    # Initialize with NaNs
    df_merged['conc_pred_g1_exp'] = np.nan
    df_merged.loc[valid_mask, 'conc_pred_g1_exp'] = (1 / b) * np.log(term[valid_mask])
    
    # 5. Plot
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    
    # True values from Ridge CSV (conc_true)
    true_vals = df_merged['conc_true']
    
    # Plot Ridge CV Predictions
    ax.scatter(
        true_vals,
        df_merged["conc_pred_cv"],
        color="tab:blue",
        label="岭回归（LOO-CV）",
        alpha=0.8,
        s=36,
        linewidths=0,
    )
    
    # Plot G1 Exponential Predictions
    ax.scatter(
        true_vals,
        df_merged["conc_pred_g1_exp"],
        color="tab:red",
        marker="x",
        label="指数拟合（G1）",
        alpha=0.8,
        s=42,
        linewidths=1.2,
    )
    
    # Plot y=x line
    all_vals = pd.concat(
        [true_vals, df_merged["conc_pred_cv"], df_merged["conc_pred_g1_exp"]],
        ignore_index=True,
    ).dropna()
    min_val = float(all_vals.min())
    max_val = float(all_vals.max())
    
    # Add a bit of padding
    buffer = (max_val - min_val) * 0.05
    limit_min = min_val - buffer
    limit_max = max_val + buffer
    
    ax.plot(
        [limit_min, limit_max],
        [limit_min, limit_max],
        "k--",
        linewidth=1.0,
        label="y=x",
        zorder=0,
    )
    
    ax.set_xlim(limit_min, limit_max)
    ax.set_ylim(limit_min, limit_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("真实浓度（μM）")
    ax.set_ylabel("预测浓度（μM）")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    
    out_file = os.path.join(current_dir, 'comparison_plot_ridge_vs_g1.png')
    fig.savefig(out_file, dpi=300)
    print(f"Plot saved to {out_file}")
    plt.show()

if __name__ == '__main__':
    main()
