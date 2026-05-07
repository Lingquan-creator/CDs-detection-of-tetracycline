"""
Integrate spectra and fit total intensity vs concentration.

Steps:
1) Read fit_data/data.csv (first column = wavelength, others = concentrations).
2) Plot all spectra.
3) Integrate intensity between 400-700 nm for each concentration.
4) Fit integrated intensity vs concentration using several base models and pick the best by AIC.

Run from repo root or fit_data:
    python fit_data/integrate_and_fit.py
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# -----------------------------------------------------------------------------
# Paths and configuration
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
DATA_FILE = BASE_DIR / "dataset/data.csv"
PLOT_DIR = BASE_DIR / "fit_plots"
INTEGRATED_CSV = RESULTS_DIR / "results_integrated_intensity.csv"
MODEL_SUMMARY_CSV = RESULTS_DIR / "results_intensity_models.csv"
SPECTRA_FIG = PLOT_DIR / "spectra_by_concentration.png"
FIT_FIG = PLOT_DIR / "integrated_intensity_fit.png"
INTEGRATION_RANGE = (400.0, 700.0)  # nm
EXCLUDE_CONCENTRATIONS: Tuple[float, ...] = (6.0,)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def parse_concentration(label: str) -> float:
    """Extract numeric concentration value from a column header (e.g., '0.2uM')."""
    match = re.search(r"[-+]?\d*\.?\d+", str(label))
    if not match:
        raise ValueError(f"Cannot parse concentration from header: {label}")
    return float(match.group(0))


def load_spectra(path: Path = DATA_FILE) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load wavelength vector and spectra (columns keyed by concentration)."""
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("Expected wavelength + at least one spectrum column.")

    wavelengths = df.iloc[:, 0].astype(float).to_numpy()
    spectra = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    concentrations = [parse_concentration(col) for col in spectra.columns]
    spectra.columns = concentrations
    return wavelengths, spectra


def ensure_output_dirs() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def plot_all_spectra(wavelengths: np.ndarray, spectra: pd.DataFrame, output_path: Path) -> None:
    """Plot spectra for each concentration on one figure."""
    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, len(spectra.columns)))

    for color, conc in zip(colors, spectra.columns):
        plt.plot(wavelengths, spectra[conc], color=color, linewidth=1.1, label=f"{conc} uM")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("Spectra by concentration")
    plt.xlim(wavelengths.min(), wavelengths.max())
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_intensity_fit(
    concentrations: np.ndarray,
    integrated: np.ndarray,
    model_results: List[Dict],
    output_path: Path,
) -> None:
    """Scatter of integrated intensities with all candidate model curves."""
    plt.figure(figsize=(8, 5))
    plt.scatter(concentrations, integrated, color="#111111", s=28, label="Integrated 400-700 nm")

    x_smooth = np.linspace(concentrations.min(), concentrations.max(), 400)
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_results)))

    for color, res in zip(colors, model_results):
        func = MODEL_FUNCTIONS[res["model"]]
        y_smooth = func(x_smooth, *res["params"])
        label = f"{res['model']} (R²={res['r_squared']:.4f}, AIC={res['aic']:.1f})"
        plt.plot(x_smooth, y_smooth, color=color, linewidth=2.0, label=label)

    plt.xlabel("Concentration (uM)")
    plt.ylabel("Integrated intensity (a.u.)")
    plt.title("Integrated intensity vs concentration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


# -----------------------------------------------------------------------------
# Models and fitting
# -----------------------------------------------------------------------------

def model_linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


def model_exponential(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    # Classic exponential form: A * exp(Bx) + C with clipping to avoid overflow
    return a * np.exp(np.clip(b * x, -700, 50)) + c


def model_power(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.power(x, b) + c


MODEL_FUNCTIONS: Dict[str, Callable[..., np.ndarray]] = {
    "exponential": model_exponential,
}

MODEL_FORMS: Dict[str, str] = {
    "exponential": "y = a * exp(b*x) + c",
}


def fit_candidate_models(x: np.ndarray, y: np.ndarray) -> List[Dict]:
    """Try multiple functional forms and rank by AIC."""
    y_range = y.max() - y.min()
    y_range = float(y_range if y_range > 0 else 1.0)

    candidates: List[Tuple[str, Callable, List[float], Tuple]] = [
        # Decaying exponential: enforce B<=0 so intensity decreases toward C
        ("exponential", model_exponential, [y_range, -0.01, y.min()], ([0, -np.inf, -np.inf], [np.inf, 0, np.inf])),
    ]

    results: List[Dict] = []
    for name, func, p0, bounds in candidates:
        try:
            popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=20000)
            y_pred = func(x, *popt)
        except Exception:
            continue

        residuals = y - y_pred
        rss = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - rss / ss_tot if ss_tot > 0 else 1.0
        rmse = float(np.sqrt(rss / len(y)))
        k = len(popt)
        aic = -math.inf if rss <= 0 else len(y) * math.log(rss / len(y)) + 2 * k

        results.append(
            {
                "model": name,
                "params": popt,
                "r_squared": r_squared,
                "rmse": rmse,
                "aic": aic,
                "formula": MODEL_FORMS[name],
            }
        )

    results.sort(key=lambda r: (r["aic"], -r["r_squared"]))
    return results


# -----------------------------------------------------------------------------
# Integration
# -----------------------------------------------------------------------------

def integrate_intensity(
    wavelengths: np.ndarray, spectra: pd.DataFrame, nm_range: Tuple[float, float]
) -> pd.DataFrame:
    """Integrate intensity over a wavelength window for each concentration."""
    mask = (wavelengths >= nm_range[0]) & (wavelengths <= nm_range[1])
    wl_window = wavelengths[mask]
    if wl_window.size == 0:
        raise ValueError(f"No data points within wavelength window {nm_range} nm.")

    rows = []
    for conc in spectra.columns:
        intensity = np.asarray(spectra[conc].to_numpy(dtype=float))
        area = float(np.trapezoid(intensity[mask], wl_window))
        rows.append({"concentration_uM": conc, "integrated_intensity": area})

    df = pd.DataFrame(rows).sort_values("concentration_uM").reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main() -> None:
    ensure_output_dirs()
    wavelengths, spectra = load_spectra()

    plot_all_spectra(wavelengths, spectra, SPECTRA_FIG)

    integrated_df = integrate_intensity(wavelengths, spectra, INTEGRATION_RANGE)
    if EXCLUDE_CONCENTRATIONS:
        keep_mask = ~integrated_df["concentration_uM"].isin(EXCLUDE_CONCENTRATIONS)
        integrated_df = integrated_df[keep_mask].reset_index(drop=True)
    integrated_df.to_csv(INTEGRATED_CSV, index=False)

    x = integrated_df["concentration_uM"].to_numpy(dtype=float)
    y = integrated_df["integrated_intensity"].to_numpy(dtype=float)

    model_results = fit_candidate_models(x, y)
    if not model_results:
        raise RuntimeError("Model fitting failed for all candidate functions.")

    best = model_results[0]
    export_rows = []
    for res in model_results:
        export_rows.append(
            {
                "model": res["model"],
                "formula": res["formula"],
                "params": "; ".join(f"{p:.6g}" for p in res["params"]),
                "r_squared": res["r_squared"],
                "rmse": res["rmse"],
                "aic": res["aic"],
            }
        )
    pd.DataFrame(export_rows).to_csv(MODEL_SUMMARY_CSV, index=False)

    plot_intensity_fit(x, y, model_results, FIT_FIG)

    print(f"Saved spectra plot to: {SPECTRA_FIG}")
    print(f"Saved integrated intensities to: {INTEGRATED_CSV}")
    print(f"Saved model comparison to: {MODEL_SUMMARY_CSV}")
    print(f"Best model: {best['model']} ({best['formula']})")
    print("Parameters:", ", ".join(f"{p:.6g}" for p in best["params"]))
    print(f"R^2 = {best['r_squared']:.5f}, RMSE = {best['rmse']:.5g}, AIC = {best['aic']:.3f}")


if __name__ == "__main__":
    main()
