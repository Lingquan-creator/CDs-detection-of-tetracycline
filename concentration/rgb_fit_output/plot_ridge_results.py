import pandas as pd
import matplotlib.pyplot as plt
import os

def set_mpl_style():
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun", "STSong"],
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
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'ridge_predictions.csv')
    set_mpl_style()

    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    # 读取CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if df.shape[1] < 2:
        print("Error: CSV file has fewer than 2 columns.")
        return

    # 获取前两列的数据
    col1_name = df.columns[0]
    col2_name = df.columns[1]

    x = df[col1_name]
    y = df[col2_name]

    # 绘制曲线
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    # 尝试将x轴转换为字符串以避免自动排序导致的混乱（如果它们不是数值型）
    # 但如果为了看趋势，如果是数值型更好。
    # 既然第一列是 filename，通常作为类别轴或者仅仅是标签
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
    fig.tight_layout()

    # 保存图片
    output_image_path = os.path.join(current_dir, 'ridge_predictions_plot.png')
    fig.savefig(output_image_path, dpi=300)
    print(f"Plot saved to {output_image_path}")
    
    # 显示图片
    plt.show()

if __name__ == '__main__':
    main()
