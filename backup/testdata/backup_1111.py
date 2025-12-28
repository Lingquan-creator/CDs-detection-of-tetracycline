"""
Python rewrite of fit_spectrum_general.m for multi-peak spectrum fitting.

Dependencies: numpy, scipy, matplotlib. Pandas is not required.
Configure the constants below, then run:
    python fit_spectrum_general.py
The script will read the CSV, fit the spectrum, print results, and save a plot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

# --------------------------------------------------------------------------- #
# User configuration (mirrors the MATLAB script settings)
# --------------------------------------------------------------------------- #

# Data file settings
DATA_FILE = "concentration.csv"  # CSV file to read
SKIP_ROWS = 1  # Number of header lines to skip (header row with concentration labels)
X_COLUMN = 1  # 1-based index for wavelength column
Y_COLUMNS = "all"  # "all" means process all columns after X_COLUMN, or specify list like [2,3,4]

# Peak count
N_PEAKS = 2  # 拟合峰数量

# Peak parameter presets: list of (k_m, lambda0_m, sigma_E_m).
# 峰参数预设：每个峰的(k_m, lambda0_m, sigma_E_m)元组列表
# Use math.nan to let the fitter optimize that parameter.
# 使用math.nan表示该参数需要拟合优化
#
# Examples 示例:
# - Empty list: all parameters auto-detected and fitted
#   空列表：所有参数自动检测并拟合
#   PEAK_PARAMS = []
#
# - Explicit 3 peaks with k=0 (recommended), other params free:
#   明确指定3峰，k=0（推荐），其他参数自由：
#   PEAK_PARAMS = [(0.0, math.nan, math.nan), (0.0, math.nan, math.nan), (0.0, math.nan, math.nan)]
#
# - Fix specific peak position:
#   固定某个峰位置：
#   PEAK_PARAMS = [(0.0, 450.0, math.nan), (0.0, math.nan, math.nan)]
#
PEAK_PARAMS: Sequence[Tuple[float, float, float]] = []

# Background
B_INITIAL = math.nan  # math.nan means auto-estimate from data minimum

# Parameter bounds
A_MIN = 0.0
A_MAX = math.inf
K_MIN = -0.5
K_MAX = 0.5
SIGMA_E_MIN = 0.01  # eV
SIGMA_E_MAX = 0.5  # eV
B_MIN = 0.0
B_MAX_RATIO = 0.5  # B upper bound = max(I_exp) * B_MAX_RATIO

# Fitting options
MAX_ITERATIONS = 2000
MAX_FUNCTION_EVALS = 20000
DISPLAY_ITER = False  # Set False to silence optimizer progress

# Output
SAVE_RESULTS = True
RESULTS_FILE = "fit_results.txt"
SAVE_CSV = True  # Save batch results to CSV
CSV_OUTPUT_FILE = "fit_results_batch.csv"  # CSV output filename
SAVE_FIGURE = True
FIGURE_FILE = "fit_result.png"

# Constant: photon energy-wavelength conversion (eV*nm)
C_EV_NM = 1239.8


# --------------------------------------------------------------------------- #
# Helper data structures
# --------------------------------------------------------------------------- #

@dataclass
class ParamInfo:
    k_fixed: bool
    lambda0_fixed: bool
    sigma_E_fixed: bool


# --------------------------------------------------------------------------- #
# Core calculations
# --------------------------------------------------------------------------- #

def load_data(path: Path, skip_rows: int, x_col: int, y_col: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load wavelength and intensity columns from CSV.
    Columns are 1-based in the config to match the MATLAB script.
    """
    data = np.genfromtxt(path, delimiter=",", skip_header=skip_rows)
    if data.ndim == 1:  # Handle single-row files gracefully
        data = data[np.newaxis, :]

    x_idx = x_col - 1
    y_idx = y_col - 1
    try:
        lambda_exp = data[:, x_idx]
        I_exp = data[:, y_idx]
    except IndexError as exc:
        raise IndexError(
            f"Requested columns X={x_col}, Y={y_col} but file has {data.shape[1]} columns"
        ) from exc

    mask = np.isfinite(lambda_exp) & np.isfinite(I_exp)
    lambda_exp = lambda_exp[mask]
    I_exp = I_exp[mask]

    if lambda_exp.size == 0:
        raise ValueError("No valid data points after filtering NaN/Inf.")

    return lambda_exp, I_exp


def load_batch_data(path: Path, skip_rows: int, x_col: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load wavelength and multiple intensity columns from CSV.
    Returns: (wavelength, intensity_matrix, column_headers)
    """
    # Read header
    with open(path, 'r', encoding='utf-8') as f:
        for _ in range(skip_rows - 1):
            f.readline()
        header_line = f.readline().strip()
    
    headers = [h.strip() for h in header_line.split(',')]
    
    # Load data
    data = np.genfromtxt(path, delimiter=",", skip_header=skip_rows)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    
    x_idx = x_col - 1
    lambda_exp = data[:, x_idx]
    
    # Get all intensity columns (everything except wavelength column)
    y_cols = [i for i in range(data.shape[1]) if i != x_idx]
    intensity_matrix = data[:, y_cols]
    
    # Get corresponding headers (skip wavelength column header)
    intensity_headers = [headers[i] for i in y_cols if i < len(headers)]
    
    # Ensure we have column names
    while len(intensity_headers) < intensity_matrix.shape[1]:
        intensity_headers.append(f"Column_{len(intensity_headers)+1}")
    
    return lambda_exp, intensity_matrix, intensity_headers


def auto_detect_peaks(I_exp: np.ndarray, lambda_exp: np.ndarray, n_peaks: int) -> Tuple[np.ndarray, np.ndarray]:
    """Find up to n_peaks by descending height; fall back to spaced guesses."""
    idx_peaks, properties = find_peaks(I_exp, height=None)
    peak_heights = I_exp[idx_peaks]
    peak_positions = lambda_exp[idx_peaks]

    order = np.argsort(peak_heights)[::-1]
    peak_heights = peak_heights[order][:n_peaks]
    peak_positions = peak_positions[order][:n_peaks]

    if peak_positions.size < n_peaks:
        # Evenly spread guesses inside the wavelength range
        lam_min, lam_max = lambda_exp.min(), lambda_exp.max()
        span = lam_max - lam_min
        margin = max(span * 0.1, 1.0)
        locs = np.linspace(lam_min + margin, lam_max - margin, n_peaks)
        pks = np.full(n_peaks, I_exp.max())
    else:
        locs = peak_positions
        pks = peak_heights

    order = np.argsort(locs)
    return locs[order], pks[order]


def is_free(value: float) -> bool:
    """Return True if the parameter should be fitted (value is NaN)."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def compute_single_peak(
    lambda_nm: np.ndarray,
    Ap: float,
    km: float,
    lambda0: float,
    sigma_E: float,
    C: float,
) -> np.ndarray:
    """Single-peak line shape."""
    lambda_peak = lambda0 if abs(km) < 1e-10 else km * Ap + lambda0
    E0 = C / lambda_peak - 2 * (sigma_E**2) / C * lambda_peak
    return (Ap / (lambda_nm**2)) * np.exp(-((C / lambda_nm - E0) ** 2) / (2 * sigma_E**2))


def multi_peak_model(params: np.ndarray, lambda_nm: np.ndarray, n_peaks: int, C: float) -> np.ndarray:
    """Sum of peaks plus background."""
    S = np.zeros_like(lambda_nm, dtype=float)
    B = params[-1]
    for i in range(n_peaks):
        idx = i * 4
        Ap, km, lambda0, sigma_E = params[idx : idx + 4]
        S += compute_single_peak(lambda_nm, Ap, km, lambda0, sigma_E, C)
    return S + B


def build_initial_params(
    lambda_exp: np.ndarray,
    I_exp: np.ndarray,
    n_peaks: int,
    peak_params: Sequence[Tuple[float, float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[ParamInfo], float]:
    """Construct initial guess and bounds."""
    locs, pks = auto_detect_peaks(I_exp, lambda_exp, n_peaks)

    B0 = np.nanmin(I_exp) if is_free(B_INITIAL) else B_INITIAL
    params0: List[float] = []
    lb: List[float] = []
    ub: List[float] = []
    infos: List[ParamInfo] = []

    lam_min, lam_max = lambda_exp.min(), lambda_exp.max()
    # SciPy requires lb < ub; give fixed parameters a tiny cushion.
    def fixed_bounds(val: float) -> Tuple[float, float]:
        eps = max(1e-12, abs(val) * 1e-9)
        return val - eps, val + eps

    for i in range(n_peaks):
        preset = peak_params[i] if i < len(peak_params) else (0.0, math.nan, math.nan)
        k_m, lambda0_m, sigma_E_m = preset

        # Amplitude
        if i < len(pks):
            A0 = (pks[i] - B0) * (locs[i] ** 2)
        else:
            A0 = (I_exp.max() - B0) * (locs[i] ** 2) / max(n_peaks, 1)
        A0 = max(A0, 1e6)

        # k_m
        if is_free(k_m):
            k0 = 0.0
            k_fixed = False
            lb_k, ub_k = K_MIN, K_MAX
        else:
            k0 = k_m
            k_fixed = True
            lb_k, ub_k = fixed_bounds(k0)

        # lambda0
        if is_free(lambda0_m):
            lambda0_init = locs[i]
            lambda0_fixed = False
            lb_lambda0, ub_lambda0 = lam_min, lam_max
        else:
            lambda0_init = lambda0_m
            lambda0_fixed = True
            lb_lambda0, ub_lambda0 = fixed_bounds(lambda0_init)

        # sigma_E
        if is_free(sigma_E_m):
            sigma_E_init = 0.12
            sigma_E_fixed = False
            lb_sigma, ub_sigma = SIGMA_E_MIN, SIGMA_E_MAX
        else:
            sigma_E_init = sigma_E_m
            sigma_E_fixed = True
            lb_sigma, ub_sigma = fixed_bounds(sigma_E_init)

        params0.extend([A0, k0, lambda0_init, sigma_E_init])
        lb.extend([A_MIN, lb_k, lb_lambda0, lb_sigma])
        ub.extend([A_MAX, ub_k, ub_lambda0, ub_sigma])

        infos.append(
            ParamInfo(
                k_fixed=k_fixed,
                lambda0_fixed=lambda0_fixed,
                sigma_E_fixed=sigma_E_fixed,
            )
        )

    params0.append(B0)
    lb.append(B_MIN)
    ub.append(I_exp.max() * B_MAX_RATIO)

    return np.array(params0, dtype=float), np.array(lb, dtype=float), np.array(ub, dtype=float), infos, B0


def fit_spectrum(lambda_exp: np.ndarray, I_exp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[ParamInfo]]:
    """Run the non-linear least squares fit."""
    params0, lb, ub, param_info, B0 = build_initial_params(lambda_exp, I_exp, N_PEAKS, PEAK_PARAMS)
    
    # Print initial parameters
    print("\n=== Initial Parameters ===")
    for i in range(N_PEAKS):
        idx = i * 4
        print(f"Peak {i+1}: A'={params0[idx]:.2e}, k={params0[idx+1]:.4f}, lambda0={params0[idx+2]:.2f} nm, sigma_E={params0[idx+3]:.4f} eV")
    print(f"Background B = {params0[-1]:.4e}\n")

    def residuals(p: np.ndarray) -> np.ndarray:
        return multi_peak_model(p, lambda_exp, N_PEAKS, C_EV_NM) - I_exp

    verbose = 2 if DISPLAY_ITER else 0
    result = least_squares(
        residuals,
        params0,
        bounds=(lb, ub),
        max_nfev=MAX_FUNCTION_EVALS,
        method='trf',  # Trust Region Reflective (similar to MATLAB's default)
        ftol=1e-10,
        xtol=None,  # Disable xtol to prevent premature termination
        gtol=1e-10,
        x_scale='jac',  # Automatic parameter scaling
        verbose=verbose,
    )

    params_fit = result.x
    I_fit = multi_peak_model(params_fit, lambda_exp, N_PEAKS, C_EV_NM)
    residual_vec = I_exp - I_fit
    return params_fit, I_fit, residual_vec, param_info


# --------------------------------------------------------------------------- #
# Reporting and plotting
# --------------------------------------------------------------------------- #

def compute_metrics(I_exp: np.ndarray, I_fit: np.ndarray, residuals: np.ndarray) -> Tuple[float, float]:
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((I_exp - I_exp.mean()) ** 2)
    R_squared = 1 - SS_res / SS_tot if SS_tot != 0 else float("nan")
    RMSE = math.sqrt(np.mean(residuals**2))
    return R_squared, RMSE


def peak_position(Ap: float, km: float, lambda0: float) -> float:
    return lambda0 if abs(km) < 1e-10 else km * Ap + lambda0


def summarize_fit(
    lambda_exp: np.ndarray,
    I_exp: np.ndarray,
    params_fit: np.ndarray,
    param_info: Sequence[ParamInfo],
) -> None:
    """Print fit parameters to stdout."""
    I_fit = multi_peak_model(params_fit, lambda_exp, N_PEAKS, C_EV_NM)
    residuals = I_exp - I_fit
    R_squared, RMSE = compute_metrics(I_exp, I_fit, residuals)

    print("\n========================================")
    print("             Fit Results")
    print("========================================\n")

    for i in range(N_PEAKS):
        idx = i * 4
        Ap = params_fit[idx]
        km = params_fit[idx + 1]
        lambda0 = params_fit[idx + 2]
        sigma_E = params_fit[idx + 3]
        peak_pos = peak_position(Ap, km, lambda0)
        E0 = C_EV_NM / peak_pos - 2 * (sigma_E**2) / C_EV_NM * peak_pos
        I_peak = compute_single_peak(lambda_exp, Ap, km, lambda0, sigma_E, C_EV_NM)
        peak_integral = np.trapezoid(I_peak, lambda_exp)

        info = param_info[i]
        print(f"Peak {i + 1}:")
        print(f"  A'{i + 1} = {Ap:.6e}")
        print(f"  k_{i + 1} = {km:.6f}" + (" (fixed)" if info.k_fixed else ""))
        print(f"  lambda0_{i + 1} = {lambda0:.4f} nm" + (" (fixed)" if info.lambda0_fixed else ""))
        print(f"  sigma_E_{i + 1} = {sigma_E:.6f} eV" + (" (fixed)" if info.sigma_E_fixed else ""))
        print(f"  peak position = {peak_pos:.4f} nm")
        print(f"  center energy E0 = {E0:.6f} eV")
        print(f"  integrated intensity = {peak_integral:.6e}\n")

    B_fit = params_fit[-1]
    print(f"Background B = {B_fit:.6e}\n")
    print("Fit quality:")
    print(f"  R^2 = {R_squared:.8f}")
    print(f"  RMSE = {RMSE:.6e}")
    print(f"  Relative RMSE = {100 * RMSE / I_exp.mean():.4f}%\n")


def save_results_file(
    path: Path,
    lambda_exp: np.ndarray,
    I_exp: np.ndarray,
    params_fit: np.ndarray,
    param_info: Sequence[ParamInfo],
) -> None:
    I_fit = multi_peak_model(params_fit, lambda_exp, N_PEAKS, C_EV_NM)
    residuals = I_exp - I_fit
    R_squared, RMSE = compute_metrics(I_exp, I_fit, residuals)

    with path.open("w", encoding="utf-8") as f:
        f.write("========================================\n")
        f.write("      Spectrum fitting results\n")
        f.write("========================================\n\n")
        f.write(f"Data file: {DATA_FILE}\n")
        f.write(f"Columns used: X={X_COLUMN}, Y={Y_COLUMN}\n")
        f.write(f"Number of peaks: {N_PEAKS}\n\n")

        for i in range(N_PEAKS):
            idx = i * 4
            Ap = params_fit[idx]
            km = params_fit[idx + 1]
            lambda0 = params_fit[idx + 2]
            sigma_E = params_fit[idx + 3]
            peak_pos = peak_position(Ap, km, lambda0)
            E0 = C_EV_NM / peak_pos - 2 * (sigma_E**2) / C_EV_NM * peak_pos
            I_peak = compute_single_peak(lambda_exp, Ap, km, lambda0, sigma_E, C_EV_NM)
            peak_integral = np.trapezoid(I_peak, lambda_exp)

            info = param_info[i]
            f.write(f"Peak {i + 1}:\n")
            f.write(f"  A'_{i + 1} = {Ap:.8e}\n")
            f.write(f"  k_{i + 1} = {km:.8f}" + (" (fixed)" if info.k_fixed else "") + "\n")
            f.write(f"  lambda0_{i + 1} = {lambda0:.6f} nm" + (" (fixed)" if info.lambda0_fixed else "") + "\n")
            f.write(f"  sigma_E_{i + 1} = {sigma_E:.8f} eV" + (" (fixed)" if info.sigma_E_fixed else "") + "\n")
            f.write(f"  peak position = {peak_pos:.6f} nm\n")
            f.write(f"  center energy E0 = {E0:.8f} eV\n")
            f.write(f"  integrated intensity = {peak_integral:.8e}\n\n")

        B_fit = params_fit[-1]
        f.write(f"Background B = {B_fit:.8e}\n\n")
        f.write("Fit quality:\n")
        f.write(f"  R^2 = {R_squared:.10f}\n")
        f.write(f"  RMSE = {RMSE:.8e}\n")
        f.write(f"  Relative RMSE = {100 * RMSE / I_exp.mean():.6f}%\n")

    print(f"Results saved to: {path}")


def save_figure(
    path: Path,
    lambda_exp: np.ndarray,
    I_exp: np.ndarray,
    params_fit: np.ndarray,
    residuals: np.ndarray,
) -> None:
    I_fit = multi_peak_model(params_fit, lambda_exp, N_PEAKS, C_EV_NM)
    rel_res = 100 * residuals / np.maximum(I_exp, 1e-12)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Plot data and total fit
    ax0 = axes[0]
    ax0.plot(lambda_exp, I_exp, "k.", markersize=3, label="Experimental data")
    ax0.plot(lambda_exp, I_fit, "r-", linewidth=2, label="Total fit")

    colors = plt.cm.tab10(np.linspace(0, 1, N_PEAKS))
    for i in range(N_PEAKS):
        idx = i * 4
        Ap = params_fit[idx]
        km = params_fit[idx + 1]
        lambda0 = params_fit[idx + 2]
        sigma_E = params_fit[idx + 3]
        I_peak = compute_single_peak(lambda_exp, Ap, km, lambda0, sigma_E, C_EV_NM)
        peak_pos = peak_position(Ap, km, lambda0)
        ax0.plot(lambda_exp, I_peak, "--", color=colors[i], linewidth=1.5, label=f"Peak {i + 1} ({peak_pos:.1f} nm)")

    B_fit = params_fit[-1]
    ax0.plot(lambda_exp, np.full_like(lambda_exp, B_fit), "k:", linewidth=1.0, label="Background")
    ax0.set_ylabel("Intensity (a.u.)", fontsize=11)
    R_squared, _ = compute_metrics(I_exp, I_fit, residuals)
    ax0.set_title(f"Multi-peak fit (R^2 = {R_squared:.6f}, {N_PEAKS} peaks)", fontsize=12)
    ax0.legend(loc="best", fontsize=9)
    ax0.grid(True, linewidth=0.3)

    # Residuals
    ax1 = axes[1]
    ax1.plot(lambda_exp, residuals, "b-", linewidth=1)
    ax1.axhline(0.0, color="k", linestyle="--", linewidth=0.5)
    ax1.set_ylabel("Residual", fontsize=11)
    ax1.set_title("Residuals", fontsize=12)
    ax1.grid(True, linewidth=0.3)

    # Relative residuals
    ax2 = axes[2]
    ax2.plot(lambda_exp, rel_res, "g-", linewidth=1)
    ax2.axhline(0.0, color="k", linestyle="--", linewidth=0.5)
    ax2.set_xlabel("Wavelength (nm)", fontsize=11)
    ax2.set_ylabel("Relative residual (%)", fontsize=11)
    ax2.set_title("Relative residuals", fontsize=12)
    ax2.grid(True, linewidth=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to: {path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def save_batch_results_csv(output_path: Path, results: List[dict]) -> None:
    """Save batch fitting results to CSV file."""
    import csv
    
    if not results:
        print("No results to save.")
        return
    
    # Prepare CSV rows
    rows = []
    for result in results:
        row = {
            'Concentration': result['concentration'],
            'R_squared': result['R_squared'],
            'RMSE': result['RMSE'],
            'Relative_RMSE_%': result['Relative_RMSE_%'],
            'Background_B': result['Background_B'],
        }
        
        # Add peak parameters
        for i, peak in enumerate(result['peaks'], 1):
            row[f'Peak{i}_A_prime'] = peak['A_prime']
            row[f'Peak{i}_k'] = peak['k']
            row[f'Peak{i}_lambda0_nm'] = peak['lambda0_nm']
            row[f'Peak{i}_sigma_E_eV'] = peak['sigma_E_eV']
            row[f'Peak{i}_peak_position_nm'] = peak['peak_position_nm']
            row[f'Peak{i}_E0_eV'] = peak['E0_eV']
            row[f'Peak{i}_integrated_intensity'] = peak['integrated_intensity']
        
        rows.append(row)
    
    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    print(f"\nBatch results saved to: {output_path}")


def main() -> None:
    data_path = Path(DATA_FILE)
    
    # Check if batch processing is needed
    try:
        lambda_exp, intensity_matrix, headers = load_batch_data(data_path, SKIP_ROWS, X_COLUMN)
        
        print(f"Loaded data from: {data_path}")
        print(f"Wavelength range: {lambda_exp.min():.1f} ~ {lambda_exp.max():.1f} nm")
        print(f"Found {intensity_matrix.shape[1]} concentration columns to process")
        print(f"Columns: {', '.join(headers)}\n")
        
        batch_results = []
        
        # Process each column
        for col_idx in range(intensity_matrix.shape[1]):
            I_exp = intensity_matrix[:, col_idx]
            concentration = headers[col_idx]
            
            # Filter valid data
            mask = np.isfinite(lambda_exp) & np.isfinite(I_exp)
            lambda_valid = lambda_exp[mask]
            I_valid = I_exp[mask]
            
            if lambda_valid.size == 0:
                print(f"⚠ Skipping '{concentration}': No valid data points")
                continue
            
            print(f"{'='*60}")
            print(f"Processing: {concentration}")
            print(f"{'='*60}")
            print(f"Valid points: {lambda_valid.size}")
            print(f"Intensity range: {I_valid.min():.2e} ~ {I_valid.max():.2e}")
            
            try:
                params_fit, I_fit, residuals, param_info = fit_spectrum(lambda_valid, I_valid)
                
                # Calculate metrics
                R_squared, RMSE = compute_metrics(I_valid, I_fit, residuals)
                rel_RMSE = 100 * RMSE / I_valid.mean()
                
                # Extract peak information
                peaks_info = []
                for i in range(N_PEAKS):
                    idx = i * 4
                    Ap = params_fit[idx]
                    km = params_fit[idx + 1]
                    lambda0 = params_fit[idx + 2]
                    sigma_E = params_fit[idx + 3]
                    peak_pos = peak_position(Ap, km, lambda0)
                    E0 = C_EV_NM / peak_pos - 2 * (sigma_E**2) / C_EV_NM * peak_pos
                    I_peak = compute_single_peak(lambda_valid, Ap, km, lambda0, sigma_E, C_EV_NM)
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
                
                # Store results
                batch_results.append({
                    'concentration': concentration,
                    'R_squared': R_squared,
                    'RMSE': RMSE,
                    'Relative_RMSE_%': rel_RMSE,
                    'Background_B': B_fit,
                    'peaks': peaks_info,
                })
                
                # Print summary
                print(f"\n✓ Fitting completed:")
                print(f"  R² = {R_squared:.6f}")
                print(f"  RMSE = {RMSE:.4e}")
                for i, peak in enumerate(peaks_info, 1):
                    print(f"  Peak {i}: λ = {peak['peak_position_nm']:.2f} nm, σ_E = {peak['sigma_E_eV']:.4f} eV")
                
            except Exception as e:
                print(f"✗ Error fitting '{concentration}': {e}")
                continue
        
        # Save batch results to CSV
        if SAVE_CSV and batch_results:
            save_batch_results_csv(Path(CSV_OUTPUT_FILE), batch_results)
        
        print(f"\n{'='*60}")
        print(f"Batch processing completed: {len(batch_results)}/{intensity_matrix.shape[1]} successful")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        print("Falling back to single-column mode...\n")
        
        # Fallback to original single-column processing
        lambda_exp, I_exp = load_data(data_path, SKIP_ROWS, X_COLUMN, Y_COLUMN if hasattr(__builtins__, 'Y_COLUMN') else 2)
        
        print(f"Loaded data from: {data_path}")
        print(f"Valid points: {lambda_exp.size}")
        print(f"Wavelength range: {lambda_exp.min():.1f} ~ {lambda_exp.max():.1f} nm")
        print(f"Intensity range: {I_exp.min():.2e} ~ {I_exp.max():.2e}")
        
        params_fit, I_fit, residuals, param_info = fit_spectrum(lambda_exp, I_exp)
        summarize_fit(lambda_exp, I_exp, params_fit, param_info)
        
        if SAVE_RESULTS:
            save_results_file(Path(RESULTS_FILE), lambda_exp, I_exp, params_fit, param_info)
        if SAVE_FIGURE:
            save_figure(Path(FIGURE_FILE), lambda_exp, I_exp, params_fit, residuals)


if __name__ == "__main__":
    main()
