"""
在 conc_test/fit_results_batch.csv 上做线性拟合（去除 Blank）

默认对 Peak1 的强度相关参数做线性拟合，并输出：
- conc_test/linear_fit_results_no_blank.csv（拟合统计量与方程）
- conc_test/parameter_plots_no_blank/*.png（散点+拟合线、残差图）

用法：
  python conc_test/linear_fit_no_blank.py
  python conc_test/linear_fit_no_blank.py --input conc_test/fit_results_batch.csv --no-plots
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class FitSummary:
    param_col: str
    param_label: str
    n: int
    slope: float
    intercept: float
    r_value: float
    r_squared: float
    p_value: float
    std_err: float
    rmse: float

    @property
    def equation_y_of_c(self) -> str:
        return f"y = {self.slope:.6e} * C + {self.intercept:.6e}"

    @property
    def equation_c_of_y(self) -> str:
        # C = (y - intercept) / slope
        return f"C = (y - {self.intercept:.6e}) / ({self.slope:.6e})"


def parse_concentration_to_float(value: object) -> float:
    """
    解析浓度字符串为浮点数（μM）。
    支持：'Blank' -> 0，'0.2μM'/'0.2¦ĚM'/'0.2uM' 等各种编码情况。
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    text = str(value).strip()
    if text.lower() == "blank":
        return 0.0
    match = re.search(r"([0-9]*\.?[0-9]+)", text)
    if not match:
        raise ValueError(f"无法解析浓度字段: {value!r}")
    return float(match.group(1))


def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[FitSummary, np.ndarray]:
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_fit = slope * x + intercept
    residuals = y - y_fit
    rmse = float(np.sqrt(np.mean(residuals**2)))
    summary = FitSummary(
        param_col="",
        param_label="",
        n=int(len(x)),
        slope=float(slope),
        intercept=float(intercept),
        r_value=float(r_value),
        r_squared=float(r_value**2),
        p_value=float(p_value),
        std_err=float(std_err),
        rmse=rmse,
    )
    return summary, y_fit


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_fit(
    x: np.ndarray,
    y: np.ndarray,
    y_fit: np.ndarray,
    summary: FitSummary,
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(x, y, s=40, alpha=0.75, label="data")
    ax1.plot(x, y_fit, "r-", linewidth=2, label="linear fit")
    ax1.set_xlabel("Concentration (μM)")
    ax1.set_ylabel(summary.param_label)
    ax1.set_title(f"{summary.param_label} vs Concentration (no Blank)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    txt = (
        f"{summary.equation_y_of_c}\n"
        f"R2 = {summary.r_squared:.6f}\n"
        f"p = {summary.p_value:.3e}\n"
        f"RMSE = {summary.rmse:.6g}\n"
        f"{summary.equation_c_of_y}"
    )
    ax1.text(
        0.03,
        0.97,
        txt,
        transform=ax1.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    residuals = y - y_fit
    ax2.scatter(x, residuals, s=40, alpha=0.75)
    ax2.axhline(0, color="r", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Concentration (μM)")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residuals")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def default_parameters() -> List[Tuple[str, str]]:
    return [
        ("Peak1_A_prime", "Peak1_A_prime"),
        ("Peak1_max_intensity", "Peak1_max_intensity"),
        ("Peak1_integrated_intensity", "Peak1_integrated_intensity"),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Linear fit (remove Blank) for fit_results_batch.csv")
    parser.add_argument(
        "--input",
        default=os.path.join("conc_test", "fit_results_batch.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("conc_test", "linear_fit_results_no_blank.csv"),
        help="Output CSV path for fit summaries",
    )
    parser.add_argument(
        "--plots-dir",
        default=os.path.join("conc_test", "parameter_plots_no_blank"),
        help="Directory to save plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable generating PNG plots",
    )
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input)
    if "Concentration" not in df.columns:
        raise ValueError(f"CSV 缺少 'Concentration' 列: {args.input}")

    df["C_uM"] = df["Concentration"].apply(parse_concentration_to_float)

    # 去除 Blank（以及所有 C<=0 的点）
    df_nb = df[df["C_uM"] > 0].copy()
    if df_nb.empty:
        raise ValueError("去除 Blank 后数据为空，请检查输入文件。")

    results: List[FitSummary] = []
    if not args.no_plots:
        ensure_dir(args.plots_dir)

    for param_col, label in default_parameters():
        if param_col not in df_nb.columns:
            raise ValueError(f"CSV 缺少列: {param_col}")

        x = df_nb["C_uM"].to_numpy(dtype=float)
        y = df_nb[param_col].to_numpy(dtype=float)
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        summary, y_fit = linear_fit(x, y)
        summary = FitSummary(
            param_col=param_col,
            param_label=label,
            n=summary.n,
            slope=summary.slope,
            intercept=summary.intercept,
            r_value=summary.r_value,
            r_squared=summary.r_squared,
            p_value=summary.p_value,
            std_err=summary.std_err,
            rmse=summary.rmse,
        )
        results.append(summary)

        print(f"[{label}] n={summary.n}")
        print(f"  {summary.equation_y_of_c}")
        print(f"  R2={summary.r_squared:.6f}  RMSE={summary.rmse:.6g}")
        print(f"  inverse: {summary.equation_c_of_y}")

        if not args.no_plots:
            out_png = os.path.join(args.plots_dir, f"linear_fit_no_blank_{param_col}.png")
            plot_fit(x, y, y_fit, summary, out_png)

    out_df = pd.DataFrame(
        [
            {
                "param_col": r.param_col,
                "param_label": r.param_label,
                "n": r.n,
                "slope": r.slope,
                "intercept": r.intercept,
                "r_value": r.r_value,
                "r_squared": r.r_squared,
                "p_value": r.p_value,
                "std_err": r.std_err,
                "rmse": r.rmse,
                "equation_y_of_c": r.equation_y_of_c,
                "equation_c_of_y": r.equation_c_of_y,
            }
            for r in results
        ]
    )
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {args.output}")
    if not args.no_plots:
        print(f"Plots: {args.plots_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
