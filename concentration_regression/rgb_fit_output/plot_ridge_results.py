import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os


def format_plot_number(value, _pos=None):
    text = f"{value:.6g}"
    return "0" if text in ("", "-0") else text

def set_mpl_style():
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'ridge_predictions.csv')
    set_mpl_style()

    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if df.shape[1] < 2:
        print("Error: CSV file has fewer than 2 columns.")
        return

    col1_name = df.columns[0]
    col2_name = df.columns[1]

    x = df[col1_name]
    y = df[col2_name]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    ax.plot(
        x.astype(str),
        y,
        marker="o",
        linestyle="-",
        linewidth=1.0,
        markersize=3.5,
        label=f"{col2_name}",
    )

    ax.set_xlabel(col1_name)
    ax.set_ylabel(col2_name)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=True)
    ax.yaxis.set_major_formatter(FuncFormatter(format_plot_number))
    fig.tight_layout()

    output_image_path = os.path.join(current_dir, 'ridge_predictions_plot.png')
    fig.savefig(output_image_path, dpi=300)
    print(f"Plot saved to {output_image_path}")
    
    plt.show()

if __name__ == '__main__':
    main()
