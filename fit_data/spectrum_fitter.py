"""
光谱拟合核心算法模块

包含峰检测、参数初始化、非线性拟合等核心功能。
Spectrum fitting core algorithms including peak detection, parameter initialization, and nonlinear fitting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks


# ============================================================================
# 数据结构 / Data Structures
# ============================================================================

@dataclass
class ParamInfo:
    """记录哪些参数是固定的 / Record which parameters are fixed"""
    k_fixed: bool
    lambda0_fixed: bool
    sigma_E_fixed: bool


# ============================================================================
# 辅助函数 / Helper Functions
# ============================================================================

def is_free(value: float) -> bool:
    """检查参数是否需要拟合 / Check if parameter should be fitted"""
    return value is None or (isinstance(value, float) and math.isnan(value))


def auto_detect_peaks(I_exp: np.ndarray, lambda_exp: np.ndarray, n_peaks: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    改进的峰检测：使用自适应阈值、突出度和峰宽度估计
    Improved peak detection with adaptive threshold, prominence, and width estimation
    
    Args:
        I_exp: 实验强度数据 / Experimental intensity data
        lambda_exp: 波长数据 / Wavelength data
        n_peaks: 需要检测的峰数量 / Number of peaks to detect
    
    Returns:
        (峰位置, 峰高度, 峰宽度) / (peak positions, peak heights, peak widths)
    """
    from scipy.signal import peak_widths
    
    # 方法1: 使用自适应高度阈值和突出度 / Method 1: Adaptive height threshold with prominence
    height_threshold = np.percentile(I_exp, 75)  # 前25%的强度 / Top 25% intensity
    min_peak_distance = max(len(I_exp) // (n_peaks * 3), 5)  # 最小峰间距 / Min peak distance
    
    idx_peaks, properties = find_peaks(
        I_exp,
        height=height_threshold,
        prominence=I_exp.max() * 0.05,  # 至少5%突出度 / At least 5% prominence
        distance=min_peak_distance
    )
    
    # 如果检测到的峰太少，降低阈值重试 / If too few peaks, retry with lower threshold
    if len(idx_peaks) < n_peaks:
        height_threshold = np.percentile(I_exp, 50)
        idx_peaks, properties = find_peaks(
            I_exp,
            height=height_threshold,
            prominence=I_exp.max() * 0.03,
            distance=max(min_peak_distance // 2, 3)
        )
    
    if len(idx_peaks) == 0:
        # 最后尝试：无任何约束 / Last resort: no constraints
        idx_peaks, properties = find_peaks(I_exp)
    
    # 按强度排序，取前n个最强峰 / Sort by intensity, take top n
    if len(idx_peaks) > 0:
        peak_heights = I_exp[idx_peaks]
        order = np.argsort(peak_heights)[::-1][:n_peaks]
        idx_peaks = idx_peaks[order]
        
        # 按波长重新排序 / Re-sort by wavelength
        order = np.argsort(lambda_exp[idx_peaks])
        idx_peaks = idx_peaks[order]
        peak_positions = lambda_exp[idx_peaks]
        peak_heights = I_exp[idx_peaks]
        
        # 估计峰宽度（FWHM）/ Estimate peak widths (FWHM)
        try:
            widths, _, _, _ = peak_widths(I_exp, idx_peaks, rel_height=0.5)
        except:
            widths = np.full(len(idx_peaks), len(I_exp) * 0.05)  # 默认5%数据范围 / Default 5% data range
    else:
        peak_positions = np.array([])
        peak_heights = np.array([])
        widths = np.array([])
    
    # 如果检测到的峰仍然不够，用均匀分布补充 / If still not enough peaks, supplement with uniform distribution
    if len(peak_positions) < n_peaks:
        lam_min, lam_max = lambda_exp.min(), lambda_exp.max()
        span = lam_max - lam_min
        margin = max(span * 0.1, 1.0)
        
        n_missing = n_peaks - len(peak_positions)
        # 避免与已有峰重叠 / Avoid overlapping with existing peaks
        if len(peak_positions) > 0:
            # 在空隙中插入 / Insert in gaps
            gaps = np.diff(np.concatenate([[lam_min], peak_positions, [lam_max]]))
            largest_gaps_idx = np.argsort(gaps)[::-1][:n_missing]
            
            new_positions = []
            for idx in largest_gaps_idx:
                if idx == 0:
                    pos = (lam_min + peak_positions[0]) / 2
                elif idx == len(peak_positions):
                    pos = (peak_positions[-1] + lam_max) / 2
                else:
                    pos = (peak_positions[idx-1] + peak_positions[idx]) / 2
                new_positions.append(pos)
            
            peak_positions = np.concatenate([peak_positions, new_positions])
            peak_heights = np.concatenate([peak_heights, np.full(n_missing, I_exp.max() * 0.5)])
            widths = np.concatenate([widths, np.full(n_missing, len(I_exp) * 0.05)])
        else:
            # 完全均匀分布 / Completely uniform distribution
            peak_positions = np.linspace(lam_min + margin, lam_max - margin, n_peaks)
            peak_heights = np.full(n_peaks, I_exp.max())
            widths = np.full(n_peaks, len(I_exp) * 0.05)
        
        # 重新排序 / Re-sort
        order = np.argsort(peak_positions)
        peak_positions = peak_positions[order]
        peak_heights = peak_heights[order]
        widths = widths[order]
    
    return peak_positions, peak_heights, widths


# ============================================================================
# 物理模型 / Physical Models
# ============================================================================

def compute_single_peak(
    lambda_nm: np.ndarray,
    Ap: float,
    km: float,
    lambda0: float,
    sigma_E: float,
    C: float,
) -> np.ndarray:
    """
    计算单个峰的谱线形状 / Compute single peak line shape
    
    Physical model:
    - Peak position: λ_peak = k_m × A'_m + λ_0,m
    - Center energy: E_0 = C/λ_peak - 2σ²_E/C × λ_peak
    - Intensity: I(λ) = (A'/λ²) × exp[-(C/λ - E_0)²/(2σ²_E)]
    """
    lambda_peak = lambda0 if abs(km) < 1e-10 else km * Ap + lambda0
    E0 = C / lambda_peak - 2 * (sigma_E**2) / C * lambda_peak
    return (Ap / (lambda_nm**2)) * np.exp(-((C / lambda_nm - E0) ** 2) / (2 * sigma_E**2))


def multi_peak_model(params: np.ndarray, lambda_nm: np.ndarray, n_peaks: int, C: float) -> np.ndarray:
    """
    多峰模型：所有峰的总和加背景 / Multi-peak model: sum of all peaks plus background
    
    Args:
        params: 参数数组 [A'_1, k_1, λ_0,1, σ_E,1, ..., A'_n, k_n, λ_0,n, σ_E,n, B]
        lambda_nm: 波长数组 / Wavelength array
        n_peaks: 峰数量 / Number of peaks
        C: 光子能量-波长转换常数 / Photon energy-wavelength conversion constant
    """
    S = np.zeros_like(lambda_nm, dtype=float)
    B = params[-1]
    for i in range(n_peaks):
        idx = i * 4
        Ap, km, lambda0, sigma_E = params[idx : idx + 4]
        S += compute_single_peak(lambda_nm, Ap, km, lambda0, sigma_E, C)
    return S + B


def peak_position(Ap: float, km: float, lambda0: float) -> float:
    """计算实际峰位置 / Calculate actual peak position"""
    return lambda0 if abs(km) < 1e-10 else km * Ap + lambda0


# ============================================================================
# 参数初始化 / Parameter Initialization
# ============================================================================

def build_initial_params(
    lambda_exp: np.ndarray,
    I_exp: np.ndarray,
    n_peaks: int,
    peak_params: Sequence[Tuple[float, float, float]],
    manual_initials: Sequence[Tuple[float, float, float, float]] | None,
    B_initial: float,
    A_min: float,
    A_max: float,
    allow_negative_A: bool,
    K_min: float,
    K_max: float,
    sigma_E_min: float,
    sigma_E_max: float,
    B_min: float,
    B_max_ratio: float,
    C: float = 1239.8,
    peak_specific_bounds: dict | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[ParamInfo], float]:
    """
    构建初始参数猜测和边界（改进版）/ Build initial parameter guess and bounds (improved)
    
    Args:
        manual_initials: 可选的手动初始值列表 [(A', k, lambda0, sigma_E)]，math.nan 表示自动估计 / Optional manual initials; use math.nan to auto-estimate
        allow_negative_A: 是否允许 A' 为负数 / Whether negative amplitudes are allowed
        peak_specific_bounds: 可选的峰特定边界字典 {peak_idx: {'param': value, ...}} / Optional peak-specific bounds dict
    
    Returns:
        (初始参数, 下界, 上界, 参数信息, 背景初值)
        (initial params, lower bounds, upper bounds, param info, initial background)
    """
    # 使用改进的峰检测，获取峰位、高度和宽度 / Use improved peak detection
    locs, pks, widths = auto_detect_peaks(I_exp, lambda_exp, n_peaks)

    # 使用百分位数估计背景，更鲁棒 / Use percentile for background estimation (more robust)
    B0 = np.percentile(I_exp, 5) if is_free(B_initial) else B_initial  # 第5百分位数 / 5th percentile
    params0: List[float] = []
    lb: List[float] = []
    ub: List[float] = []
    infos: List[ParamInfo] = []

    lam_min, lam_max = lambda_exp.min(), lambda_exp.max()
    manual_defaults = (math.nan, math.nan, math.nan, math.nan)
    
    def fixed_bounds(val: float) -> Tuple[float, float]:
        """为固定参数创建微小边界 / Create tiny bounds for fixed parameters"""
        eps = max(1e-12, abs(val) * 1e-9)
        return val - eps, val + eps

    # 动态调整A'的上界 / Dynamically adjust A' upper bound
    I_range = I_exp.max() - I_exp.min()
    A_max_dynamic = I_range * (lam_max**2) * 10
    A_max_effective = min(A_max, A_max_dynamic) if np.isfinite(A_max) else A_max_dynamic
    if allow_negative_A:
        # 若 A_min 未给出有限值（例如 -inf），默认用对称边界限制负振幅，避免极端负值。
        A_min_effective = A_min if np.isfinite(A_min) else -A_max_effective
    else:
        # 禁止负振幅时强制下界不小于 0 / Enforce non-negative amplitude when disabled
        A_min_effective = A_min if (np.isfinite(A_min) and A_min >= 0) else 0.0
    
    for i in range(n_peaks):
        preset = peak_params[i] if i < len(peak_params) else (0.0, math.nan, math.nan)
        k_m, lambda0_m, sigma_E_m = preset
        manual_vals = manual_initials[i] if (manual_initials is not None and i < len(manual_initials)) else manual_defaults
        manual_A, manual_k, manual_lambda0, manual_sigma = manual_vals
        
        # 获取峰特定边界（如果有）/ Get peak-specific bounds (if any)
        peak_bounds = peak_specific_bounds.get(i, {}) if peak_specific_bounds else {}
        A_min_peak = peak_bounds.get('A_min', A_min_effective)
        A_max_peak = peak_bounds.get('A_max', A_max_effective)
        K_min_peak = peak_bounds.get('k_min', K_min)
        K_max_peak = peak_bounds.get('k_max', K_max)
        sigma_E_min_peak = peak_bounds.get('sigma_E_min', sigma_E_min)
        sigma_E_max_peak = peak_bounds.get('sigma_E_max', sigma_E_max)

        # Amplitude A' - 改进估计 / Improved estimation
        if is_free(lambda0_m):
            lambda0_init_temp = manual_lambda0 if not is_free(manual_lambda0) else locs[i]
        else:
            lambda0_init_temp = lambda0_m
        
        # 在lambda0附近找到局部极值（支持正峰/负峰）/ Local extrema near lambda0 (supports positive/negative peaks)
        idx_near = np.argmin(np.abs(lambda_exp - lambda0_init_temp))
        search_range = max(int(widths[i] * 2) if i < len(widths) else 10, 5)
        idx_start = max(0, idx_near - search_range)
        idx_end = min(len(I_exp), idx_near + search_range + 1)
        window = I_exp[idx_start:idx_end]
        delta_max = window.max() - B0
        delta_min = window.min() - B0
        peak_delta = delta_max if abs(delta_max) >= abs(delta_min) else delta_min

        if not np.isfinite(peak_delta) or abs(peak_delta) < 1e-18:
            peak_delta = (pks[i] - B0) if i < len(pks) else (I_exp.max() - B0) * 0.3
        if not np.isfinite(peak_delta) or abs(peak_delta) < 1e-18:
            peak_delta = I_exp.max() * 0.3

        A0 = peak_delta * (lambda0_init_temp ** 2)
        if not is_free(manual_A):
            A0 = manual_A
        A0 = float(np.clip(A0, A_min_peak, A_max_peak))

        # k_m
        if is_free(k_m):
            k0 = manual_k if not is_free(manual_k) else 0.0
            k_fixed = False
            lb_k, ub_k = K_min_peak, K_max_peak
        else:
            k0 = k_m
            k_fixed = True
            lb_k, ub_k = fixed_bounds(k0)

        # lambda0
        if is_free(lambda0_m):
            lambda0_init = manual_lambda0 if not is_free(manual_lambda0) else locs[i]
            lambda0_fixed = False
            # 使用峰特定边界（如果提供）/ Use peak-specific bounds (if provided)
            lambda0_min_peak = peak_bounds.get('lambda0_min', lam_min)
            lambda0_max_peak = peak_bounds.get('lambda0_max', lam_max)
            lb_lambda0, ub_lambda0 = lambda0_min_peak, lambda0_max_peak
            # 确保初始值在边界内 / Ensure initial value is within bounds
            lambda0_init = float(np.clip(lambda0_init, lb_lambda0, ub_lambda0))
        else:
            lambda0_init = lambda0_m
            lambda0_fixed = True
            lb_lambda0, ub_lambda0 = fixed_bounds(lambda0_init)

        # sigma_E - 从峰宽度估计 / Estimate from peak width
        if is_free(sigma_E_m):
            if not is_free(manual_sigma):
                sigma_E_init = manual_sigma
            else:
                # 使用检测到的峰宽度估计 / Use detected peak width
                if i < len(widths):
                    width_lambda = widths[i] * (lambda_exp[1] - lambda_exp[0]) if len(lambda_exp) > 1 else 10.0
                    E_center = C / lambda0_init
                    sigma_E_init = abs((E_center / lambda0_init) * (width_lambda / 2.355))
                    sigma_E_init = np.clip(sigma_E_init, 0.05, 0.4)
                else:
                    sigma_E_init = 0.12
            sigma_E_fixed = False
            # 动态调整sigma_E上界 / Dynamically adjust sigma_E upper bound
            E_typical = C / lambda0_init
            sigma_max_default = min(sigma_E_max, E_typical * 0.3)
            # 使用峰特定边界（如果提供）/ Use peak-specific bounds (if provided)
            sigma_max_for_peak = min(sigma_E_max_peak, sigma_max_default)
            lb_sigma, ub_sigma = sigma_E_min_peak, sigma_max_for_peak
            sigma_E_init = float(np.clip(sigma_E_init, lb_sigma, ub_sigma))
        else:
            sigma_E_init = sigma_E_m
            sigma_E_fixed = True
            lb_sigma, ub_sigma = fixed_bounds(sigma_E_init)

        params0.extend([A0, k0, lambda0_init, sigma_E_init])
        lb.extend([A_min_peak, lb_k, lb_lambda0, lb_sigma])
        ub.extend([A_max_peak, ub_k, ub_lambda0, ub_sigma])

        infos.append(
            ParamInfo(
                k_fixed=k_fixed,
                lambda0_fixed=lambda0_fixed,
                sigma_E_fixed=sigma_E_fixed,
            )
        )

    # Background
    params0.append(B0)
    lb.append(B_min)
    ub.append(I_exp.max() * B_max_ratio)

    return np.array(params0, dtype=float), np.array(lb, dtype=float), np.array(ub, dtype=float), infos, B0


# ============================================================================
# 数据预处理 / Data Preprocessing
# ============================================================================

def preprocess_data(
    lambda_exp: np.ndarray,
    I_exp: np.ndarray,
    smooth: bool = True,
    remove_outliers: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    数据预处理：平滑和异常值移除
    Data preprocessing: smoothing and outlier removal
    
    Args:
        lambda_exp: 波长数据 / Wavelength data
        I_exp: 强度数据 / Intensity data
        smooth: 是否平滑 / Whether to smooth
        remove_outliers: 是否移除异常值 / Whether to remove outliers
    
    Returns:
        (处理后的波长, 处理后的强度) / (processed wavelength, processed intensity)
    """
    lambda_out = lambda_exp.copy()
    I_out = I_exp.copy()
    
    if smooth and len(I_exp) >= 7:
        from scipy.signal import savgol_filter
        # 自适应窗口长度 / Adaptive window length
        window_length = min(15, len(I_exp) // 10)
        if window_length % 2 == 0:
            window_length += 1
        if window_length >= 5:
            try:
                I_smoothed = savgol_filter(I_out, window_length, polyorder=3)
            except:
                I_smoothed = I_out
        else:
            I_smoothed = I_out
    else:
        I_smoothed = I_out
    
    if remove_outliers and len(I_exp) >= 10:
        # 使用四分位数范围检测异常值 / Use IQR for outlier detection
        residuals = I_out - I_smoothed
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1
        threshold = 3 * iqr if iqr > 0 else 3 * np.std(residuals)
        
        # 保留非异常值点 / Keep non-outlier points
        mask = np.abs(residuals) < threshold
        if mask.sum() >= len(I_exp) * 0.5:  # 至少保留50%数据 / Keep at least 50% data
            lambda_out = lambda_out[mask]
            I_out = I_out[mask]
    
    return lambda_out, I_out


# ============================================================================
# 非线性拟合 / Nonlinear Fitting
# ============================================================================

def fit_spectrum(
    lambda_exp: np.ndarray,
    I_exp: np.ndarray,
    n_peaks: int,
    peak_params: Sequence[Tuple[float, float, float]],
    manual_initials: Sequence[Tuple[float, float, float, float]] | None,
    C_eV_nm: float,
    max_func_evals: int,
    display_iter: bool,
    B_initial: float,
    A_min: float,
    A_max: float,
    allow_negative_A: bool,
    K_min: float,
    K_max: float,
    sigma_E_min: float,
    sigma_E_max: float,
    B_min: float,
    B_max_ratio: float,
    peak_specific_bounds: dict | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[ParamInfo]]:
    """
    执行非线性最小二乘拟合 / Run nonlinear least squares fitting
    
    Args:
        lambda_exp: 波长数据 / Wavelength data
        I_exp: 强度数据 / Intensity data
        n_peaks: 峰数量 / Number of peaks
        peak_params: 峰参数预设 / Peak parameter presets
        manual_initials: 手动初始值 (A', k, lambda0, sigma_E); 使用 None/math.nan 时自动估计
        C_eV_nm: 能量-波长转换常数 / Energy-wavelength conversion constant
        max_func_evals: 最大函数评估次数 / Maximum function evaluations
        display_iter: 是否显示迭代过程 / Whether to display iterations
        allow_negative_A: 是否允许 A' 为负，影响振幅下界 / Whether A' can be negative
        ...: 其他参数边界 / Other parameter bounds
    
    Returns:
        (拟合参数, 拟合强度, 残差, 参数信息)
        (fitted params, fitted intensity, residuals, param info)
    """
    params0, lb, ub, param_info, B0 = build_initial_params(
        lambda_exp, I_exp, n_peaks, peak_params, manual_initials, B_initial,
        A_min, A_max, allow_negative_A, K_min, K_max, sigma_E_min, sigma_E_max, B_min, B_max_ratio, C_eV_nm,
        peak_specific_bounds
    )
    
    # 打印初始参数 / Print initial parameters
    print("\n=== Initial Parameters ===")
    for i in range(n_peaks):
        idx = i * 4
        print(f"Peak {i+1}: A'={params0[idx]:.2e}, k={params0[idx+1]:.4f}, "
              f"lambda0={params0[idx+2]:.2f} nm, sigma_E={params0[idx+3]:.4f} eV")
    print(f"Background B = {params0[-1]:.4e}\n")

    def residuals(p: np.ndarray) -> np.ndarray:
        return multi_peak_model(p, lambda_exp, n_peaks, C_eV_nm) - I_exp

    verbose = 2 if display_iter else 0
    result = least_squares(
        residuals,
        params0,
        bounds=(lb, ub),
        max_nfev=max_func_evals,
        method='trf',  # Trust Region Reflective
        ftol=1e-10,
        xtol=None,  # Disable to prevent premature termination
        gtol=1e-10,
        x_scale='jac',  # Automatic parameter scaling
        verbose=verbose,
    )

    params_fit = result.x
    I_fit = multi_peak_model(params_fit, lambda_exp, n_peaks, C_eV_nm)
    residual_vec = I_exp - I_fit
    return params_fit, I_fit, residual_vec, param_info


def fit_spectrum_multistart(
    lambda_exp: np.ndarray,
    I_exp: np.ndarray,
    n_peaks: int,
    peak_params: Sequence[Tuple[float, float, float]],
    manual_initials: Sequence[Tuple[float, float, float, float]] | None,
    C_eV_nm: float,
    max_func_evals: int,
    display_iter: bool,
    B_initial: float,
    A_min: float,
    A_max: float,
    allow_negative_A: bool,
    K_min: float,
    K_max: float,
    sigma_E_min: float,
    sigma_E_max: float,
    B_min: float,
    B_max_ratio: float,
    n_starts: int = 3,
    peak_specific_bounds: dict | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[ParamInfo]]:
    """
    多起点全局优化：尝试多个初始点，选择最佳结果
    Multi-start global optimization: try multiple starting points, select best result
    
    Args:
        ...: 与fit_spectrum相同的参数 / Same parameters as fit_spectrum
        n_starts: 尝试的起点数量 / Number of starting points to try
    
    Returns:
        最佳拟合结果 / Best fitting result (same as fit_spectrum)
    """
    if n_starts <= 1:
        # 单起点，直接调用原函数 / Single start, call original function
        return fit_spectrum(
            lambda_exp, I_exp, n_peaks, peak_params, manual_initials, C_eV_nm,
            max_func_evals, display_iter, B_initial,
            A_min, A_max, allow_negative_A, K_min, K_max, sigma_E_min, sigma_E_max,
            B_min, B_max_ratio, peak_specific_bounds
        )
    
    print(f"\n🔄 多起点优化 / Multi-start optimization: {n_starts} attempts")
    
    best_result = None
    best_residual_norm = np.inf
    best_R_squared = -np.inf
    
    for attempt in range(n_starts):
        print(f"\n{'='*60}")
        print(f"🎯 起点 {attempt + 1}/{n_starts} / Attempt {attempt + 1}/{n_starts}")
        print(f"{'='*60}")
        
        # 为每次尝试创建不同的参数扰动 / Create different parameter perturbations
        if attempt == 0:
            # 第一次：使用标准检测 / First: standard detection
            peak_params_attempt = peak_params
        elif attempt == 1:
            # 第二次：峰位置±5nm扰动 / Second: peak position ±5nm perturbation
            peak_params_attempt = []
            for pp in peak_params if len(peak_params) > 0 else [(math.nan, math.nan, math.nan)] * n_peaks:
                k, lam0, sig = pp
                if not is_free(lam0):
                    lam0_new = lam0 + np.random.uniform(-5, 5)
                else:
                    lam0_new = lam0
                peak_params_attempt.append((k, lam0_new, sig))
        else:
            # 第三次及以后：sigma_E ±20%扰动 / Third+: sigma_E ±20% perturbation
            peak_params_attempt = []
            for pp in peak_params if len(peak_params) > 0 else [(math.nan, math.nan, math.nan)] * n_peaks:
                k, lam0, sig = pp
                if not is_free(sig):
                    sig_new = sig * np.random.uniform(0.8, 1.2)
                else:
                    sig_new = sig
                peak_params_attempt.append((k, lam0, sig_new))
        
        try:
            params_fit, I_fit, residuals, param_info = fit_spectrum(
                lambda_exp, I_exp, n_peaks, peak_params_attempt, manual_initials, C_eV_nm,
                max_func_evals, False,  # 禁用子迭代显示 / Disable sub-iteration display
                B_initial, A_min, A_max, allow_negative_A, K_min, K_max,
                sigma_E_min, sigma_E_max, B_min, B_max_ratio, peak_specific_bounds
            )
            
            # 评估结果 / Evaluate result
            residual_norm = np.sum(residuals**2)
            R_squared, RMSE = compute_metrics(I_exp, I_fit, residuals)
            
            print(f"✓ 拟合成功 / Fitting succeeded:")
            print(f"  残差平方和 / Residual sum of squares = {residual_norm:.4e}")
            print(f"  R² = {R_squared:.6f}")
            print(f"  RMSE = {RMSE:.4e}")
            
            # 保存最佳结果 / Save best result
            if residual_norm < best_residual_norm:
                best_residual_norm = residual_norm
                best_R_squared = R_squared
                best_result = (params_fit, I_fit, residuals, param_info)
                print(f"  🌟 新的最佳结果！/ New best result!")
        
        except Exception as e:
            print(f"✗ 拟合失败 / Fitting failed: {e}")
            continue
    
    if best_result is None:
        raise RuntimeError("所有起点均失败 / All starting points failed")
    
    print(f"\n{'='*60}")
    print(f"🏆 最终最佳结果 / Final best result:")
    print(f"  R² = {best_R_squared:.6f}")
    print(f"  残差平方和 / Residual sum of squares = {best_residual_norm:.4e}")
    print(f"{'='*60}\n")
    
    return best_result


# ============================================================================
# 拟合质量评估 / Fitting Quality Assessment
# ============================================================================

def compute_metrics(I_exp: np.ndarray, I_fit: np.ndarray, residuals: np.ndarray) -> Tuple[float, float]:
    """
    计算拟合质量指标 / Compute fitting quality metrics
    
    Returns:
        (R², RMSE)
    """
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((I_exp - I_exp.mean()) ** 2)
    R_squared = 1 - SS_res / SS_tot if SS_tot != 0 else float("nan")
    RMSE = math.sqrt(np.mean(residuals**2))
    return R_squared, RMSE
