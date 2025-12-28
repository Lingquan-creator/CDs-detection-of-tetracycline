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


def auto_detect_peaks(I_exp: np.ndarray, lambda_exp: np.ndarray, n_peaks: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    自动检测峰位置和高度 / Automatically detect peak positions and heights
    
    Args:
        I_exp: 实验强度数据 / Experimental intensity data
        lambda_exp: 波长数据 / Wavelength data
        n_peaks: 需要检测的峰数量 / Number of peaks to detect
    
    Returns:
        (峰位置, 峰高度) / (peak positions, peak heights)
    """
    idx_peaks, properties = find_peaks(I_exp, height=None)
    peak_heights = I_exp[idx_peaks]
    peak_positions = lambda_exp[idx_peaks]

    order = np.argsort(peak_heights)[::-1]
    peak_heights = peak_heights[order][:n_peaks]
    peak_positions = peak_positions[order][:n_peaks]

    if peak_positions.size < n_peaks:
        # 如果检测到的峰少于需要的数量，均匀分布初始猜测
        # If detected peaks < needed, evenly spread initial guesses
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
    B_initial: float,
    A_min: float,
    A_max: float,
    K_min: float,
    K_max: float,
    sigma_E_min: float,
    sigma_E_max: float,
    B_min: float,
    B_max_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[ParamInfo], float]:
    """
    构建初始参数猜测和边界 / Build initial parameter guess and bounds
    
    Returns:
        (初始参数, 下界, 上界, 参数信息, 背景初值)
        (initial params, lower bounds, upper bounds, param info, initial background)
    """
    locs, pks = auto_detect_peaks(I_exp, lambda_exp, n_peaks)

    B0 = np.nanmin(I_exp) if is_free(B_initial) else B_initial
    params0: List[float] = []
    lb: List[float] = []
    ub: List[float] = []
    infos: List[ParamInfo] = []

    lam_min, lam_max = lambda_exp.min(), lambda_exp.max()
    
    def fixed_bounds(val: float) -> Tuple[float, float]:
        """为固定参数创建微小边界 / Create tiny bounds for fixed parameters"""
        eps = max(1e-12, abs(val) * 1e-9)
        return val - eps, val + eps

    for i in range(n_peaks):
        preset = peak_params[i] if i < len(peak_params) else (0.0, math.nan, math.nan)
        k_m, lambda0_m, sigma_E_m = preset

        # Amplitude A'
        if i < len(pks):
            A0 = (pks[i] - B0) * (locs[i] ** 2)
        else:
            A0 = (I_exp.max() - B0) * (locs[i] ** 2) / max(n_peaks, 1)
        A0 = max(A0, 1e6)

        # k_m
        if is_free(k_m):
            k0 = 0.0
            k_fixed = False
            lb_k, ub_k = K_min, K_max
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
            lb_sigma, ub_sigma = sigma_E_min, sigma_E_max
        else:
            sigma_E_init = sigma_E_m
            sigma_E_fixed = True
            lb_sigma, ub_sigma = fixed_bounds(sigma_E_init)

        params0.extend([A0, k0, lambda0_init, sigma_E_init])
        lb.extend([A_min, lb_k, lb_lambda0, lb_sigma])
        ub.extend([A_max, ub_k, ub_lambda0, ub_sigma])

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
# 非线性拟合 / Nonlinear Fitting
# ============================================================================

def fit_spectrum(
    lambda_exp: np.ndarray,
    I_exp: np.ndarray,
    n_peaks: int,
    peak_params: Sequence[Tuple[float, float, float]],
    C_eV_nm: float,
    max_func_evals: int,
    display_iter: bool,
    B_initial: float,
    A_min: float,
    A_max: float,
    K_min: float,
    K_max: float,
    sigma_E_min: float,
    sigma_E_max: float,
    B_min: float,
    B_max_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[ParamInfo]]:
    """
    执行非线性最小二乘拟合 / Run nonlinear least squares fitting
    
    Args:
        lambda_exp: 波长数据 / Wavelength data
        I_exp: 强度数据 / Intensity data
        n_peaks: 峰数量 / Number of peaks
        peak_params: 峰参数预设 / Peak parameter presets
        C_eV_nm: 能量-波长转换常数 / Energy-wavelength conversion constant
        max_func_evals: 最大函数评估次数 / Maximum function evaluations
        display_iter: 是否显示迭代过程 / Whether to display iterations
        ...: 其他参数边界 / Other parameter bounds
    
    Returns:
        (拟合参数, 拟合强度, 残差, 参数信息)
        (fitted params, fitted intensity, residuals, param info)
    """
    params0, lb, ub, param_info, B0 = build_initial_params(
        lambda_exp, I_exp, n_peaks, peak_params, B_initial,
        A_min, A_max, K_min, K_max, sigma_E_min, sigma_E_max, B_min, B_max_ratio
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
