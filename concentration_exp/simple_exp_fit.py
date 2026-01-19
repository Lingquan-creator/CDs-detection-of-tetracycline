
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from pathlib import Path
from sklearn.metrics import r2_score

def get_conc_from_filename(filename):
    try:
        return float(Path(filename).stem)
    except ValueError:
        return np.nan

def exp_decay_model(c, A, k, B):
    """
    Physical model: Signal = A * exp(-k * c) + B
    A: Amplitude (Signal at c=0, minus backgrouund/offset)
    k: Decay constant
    B: Offset (Background light not subtracted or electronic offset)
    """
    return A * np.exp(-k * c) + B

def fit_channel(c_data, i_data, channel_name="Channel"):
    """
    Fit I = A * exp(-k * C) + B
    """
    # Initial guess: 
    # B = min(I)
    # A = max(I) - B
    # k = 1/(mean(C)) * something? 
    # Approx: I(0) ~ A+B, I(inf) ~ B.
    
    p0 = [np.max(i_data) - np.min(i_data), 0.1, np.min(i_data)]
    bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]) # A>0
    
    try:
        popt, pcov = curve_fit(exp_decay_model, c_data, i_data, p0=p0, maxfev=10000)
    except RuntimeError:
        print(f"Fit failed for {channel_name}")
        return None

    y_pred = exp_decay_model(c_data, *popt)
    r2 = r2_score(i_data, y_pred)
    
    return {
        "params": popt,
        "r2": r2,
        "cov": pcov
    }

def calculate_concentration(signal, A, k, B):
    """
    Inverse model: C = -1/k * ln((Signal - B) / A)
    """
    # Clip signal to ensure valid log argument
    # (Signal - B) / A should be > 0
    # Ideally A > 0.
    # If k > 0, Signal decreases from A+B to B. So Signal - B goes from A to 0. (Signal-B)/A goes 1 to 0.
    
    val = (signal - B) / A
    # Avoid log of non-positive numbers
    val = np.maximum(val, 1e-9) 
    
    return - (1/k) * np.log(val)

def main():
    # 1. Load Data
    # Try multiple paths to locate the CSV
    possible_paths = [
        Path("pic_RGB/roi_results/roi_detailed_statistics.csv"),
        Path("../pic_RGB/roi_results/roi_detailed_statistics.csv"),
        Path("h:/000/CDs/pic_RGB/roi_results/roi_detailed_statistics.csv")
    ]
    
    csv_path = None
    for p in possible_paths:
        if p.exists():
            csv_path = p
            break
            
    if not csv_path:
        print("Error: Could not find roi_detailed_statistics.csv")
        return

    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 2. Extract Concentration
    df["conc"] = df["filename"].apply(get_conc_from_filename)
    df = df.dropna(subset=["conc"]).sort_values("conc")
    
    if df.empty:
        print("No valid concentration data found.")
        return

    # 3. Prepare Signals (Background Subtracted vs Raw)
    # We will compute Signal = ROI - BG and fit that.
    # If BG columns missing, use ROI directly.
    
    has_bg = "bg_R_mean" in df.columns
    
    # Analyze R, G, B
    # For G, we'll average G1 and G2 if available
    if "roi_G1_mean" in df.columns and "roi_G2_mean" in df.columns:
        df["roi_G_mean"] = (df["roi_G1_mean"] + df["roi_G2_mean"]) / 2
        if has_bg:
            df["bg_G_mean"] = (df["bg_G1_mean"] + df["bg_G2_mean"]) / 2
    else:
        # Fallback if only one G or named differently (adjust as needed)
        pass

    channels = ["R", "G", "B"]
    results = {}
    colors = {'R': 'red', 'G': 'green', 'B': 'blue'}

    plt.figure(figsize=(10, 6))

    label_suffix = ""

    for idx, ch in enumerate(channels):
        roi_col = f"roi_{ch}_mean"
        bg_col = f"bg_{ch}_mean"
        
        if roi_col not in df.columns:
            continue
            
        # Calculate Signal
        if has_bg and bg_col in df.columns:
            # We fit the NET signal usually, OR we fit the ROI directly but account for BG in the model parameter B?
            # User requirement: "No BG/Log target removed" -> "Backgound subtraction is required".
            # So we should use Net Signal.
            signal = df[roi_col] - df[bg_col]
            label_suffix = "(ROI - BG)"
        else:
            signal = df[roi_col]
            label_suffix = "(Raw ROI)"
            
        # Filter negative signals just in case? Or allow them (noise).
        # Physical intensity shouldn't be negative, but subtraction can cause it.
        # curve_fit handles negative y values fine for this model B can be negative.
        
        c_vals = df["conc"].values
        i_vals = signal.values
        
        fit_res = fit_channel(c_vals, i_vals, f"{ch}")
        
        if fit_res:
            A, k, B = fit_res["params"]
            r2 = fit_res["r2"]
            print(f"Channel {ch} Fit: I = {A:.2f} * exp(-{k:.4f} * C) + {B:.2f} | R2 = {r2:.4f}")
            
            # Inverse / Prediction Check
            c_pred = calculate_concentration(i_vals, A, k, B)
            # Filter out NaNs or Infs if any
            valid = np.isfinite(c_pred)
            if np.any(valid):
                rmse = np.sqrt(np.mean((c_pred[valid] - c_vals[valid])**2))
                print(f"  -> Prediction RMSE: {rmse:.4f} uM")
                print(f"  -> Inverse Formula: C = - (1 / {k:.4f}) * ln( (Signal - {B:.2f}) / {A:.2f} )")
            
            # Plot
            color = colors.get(ch, 'black')
            plt.scatter(c_vals, i_vals, color=color, alpha=0.5, label=f"{ch} Data")
            
            c_plot = np.linspace(c_vals.min(), c_vals.max(), 100)
            i_plot = exp_decay_model(c_plot, *fit_res["params"])
            
            plt.plot(c_plot, i_plot, color=color, linestyle='-', label=f"{ch} Fit ($k={k:.3f}$, $R^2={r2:.3f}$)")
            
            results[ch] = fit_res

    plt.title(f"Exponential Fit vs Concentration {label_suffix}")
    plt.xlabel("Concentration")
    plt.ylabel("Intensity (Signal)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = Path("concentration_exp/exponential_fit_plots.png")
    plt.savefig(plot_path)
    print(f"Plots saved to {plot_path.absolute()}")
    # plt.show() # Cannot show in this environment

if __name__ == "__main__":
    main()
