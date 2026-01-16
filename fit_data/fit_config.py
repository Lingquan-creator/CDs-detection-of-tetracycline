"""
配置文件 - 量子点光谱拟合参数设置

修改此文件中的参数来自定义拟合行为。
"""

import math
from pathlib import Path
from typing import Sequence, Tuple

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

# ============================================================================
# 数据文件设置 / Data File Settings
# ============================================================================

DATA_FILE = BASE_DIR / "data.csv"  # CSV文件名 / CSV file to read
SKIP_ROWS = 1  # 跳过的头部行数 / Number of header lines to skip (including concentration labels)
X_COLUMN = 1   # 波长列索引（从1开始） / 1-based index for wavelength column
Y_COLUMNS = "all"  # "all"表示处理所有强度列，或指定列表如[2,3,4] / "all" or list like [2,3,4]

# ============================================================================
# 拟合参数 / Fitting Parameters
# ============================================================================

N_PEAKS = 3  # 拟合峰数量 / Number of peaks to fit

# 峰参数预设 / Peak parameter presets: list of (k_m, lambda0_m, sigma_E_m)
# 使用 math.nan 表示该参数需要自动拟合 / Use math.nan for parameters to be fitted
#
# 示例 / Examples:
# - 完全自动 / Fully automatic:
#   PEAK_PARAMS = []
#
# - 固定 k=0（推荐），其他自动 / Fixed k=0 (recommended), others automatic:
#   PEAK_PARAMS = [(0.0, math.nan, math.nan), (0.0, math.nan, math.nan)]
#
# - 固定特定峰位 / Fix specific peak positions:
#   PEAK_PARAMS = [(0.0, 450.0, math.nan), (0.0, 480.0, math.nan)]
#
PEAK_PARAMS: Sequence[Tuple[float, float, float]] = [(0.0, math.nan, math.nan)]

# 手动初始值 / Manual initial guesses for peaks
# 格式: [(A_prime, k, lambda0_nm, sigma_E_eV), ...]; 使用 math.nan 表示自动估计
MANUAL_INITIAL_GUESSES: Sequence[Tuple[float, float, float, float]] = [(20000, math.nan, 420, math.nan),(35000, math.nan, 570, math.nan)]

# 背景初值 / Background initial value
# math.nan 表示自动估计（使用数据最小值） / math.nan means auto-estimate from data minimum
B_INITIAL = math.nan

# ============================================================================
# 参数边界 / Parameter Bounds
# ============================================================================

# 振幅边界 / Amplitude bounds
# 切换是否允许 A' 为负数（例如用于拟合“负峰/吸收”或差分信号）。
ALLOW_NEGATIVE_A = False
A_MIN = 0.0 if not ALLOW_NEGATIVE_A else -math.inf
A_MAX = math.inf

# k_m 边界（峰位-强度斜率）/ k_m bounds (peak position-intensity slope)
K_MIN = -0.1
K_MAX = 0.1

# 能量展宽边界 (eV) / Energy broadening bounds (eV)
SIGMA_E_MIN = 0.03  # 允许更窄的峰 / Allow narrower peaks
SIGMA_E_MAX = 0.8   # 允许更宽的峰 / Allow broader peaks

# 背景边界 / Background bounds
B_MIN = 0.0
B_MAX_RATIO = 0.5  # 背景上界 = max(强度) × B_MAX_RATIO / B upper bound = max(I) × B_MAX_RATIO

# ============================================================================
# 特定峰的参数边界（可选）/ Peak-Specific Parameter Bounds (Optional)
# ============================================================================

# 手动限制特定峰的参数范围 / Manually limit parameter ranges for specific peaks
# 格式: 字典，键为峰索引（从0开始），值为参数边界字典
# Format: dict with peak index (0-based) as key, parameter bounds dict as value
# 
# 可用的参数键 / Available parameter keys:
#   'A_min', 'A_max'       - 振幅边界 / Amplitude bounds
#   'k_min', 'k_max'       - k_m边界 / k_m bounds
#   'lambda0_min', 'lambda0_max' - 峰位边界 / Peak position bounds
#   'sigma_E_min', 'sigma_E_max' - 能量展宽边界 / Energy broadening bounds
#
# 未指定的峰将使用全局边界 / Unspecified peaks will use global bounds
# 未指定的参数将使用对应的全局边界 / Unspecified parameters will use global bounds
# 空字典表示全部使用全局边界 / Empty dict means use global bounds for all

PEAK_SPECIFIC_BOUNDS = {
    0: {},  # 峰1
    1: {},  # 峰2
}

# ============================================================================
# 优化器选项 / Optimizer Options
# ============================================================================

MAX_ITERATIONS = 5000      # 最大迭代次数 / Maximum iterations
MAX_FUNCTION_EVALS = 50000 # 最大函数评估次数 / Maximum function evaluations
DISPLAY_ITER = False       # 是否显示优化过程 / Whether to display optimization progress

# 多起点优化 / Multi-start optimization
N_MULTI_START = 3          # 尝试的起点数量（1=单次，3=多起点）/ Number of starting points (1=single, 3=multi-start)

# 数据预处理 / Data preprocessing
SMOOTH_DATA = False         # 是否平滑数据以减少噪声 / Smooth data to reduce noise
REMOVE_OUTLIERS = False     # 是否移除异常值 / Remove outliers

# ============================================================================
# 输出设置 / Output Settings
# ============================================================================

SAVE_RESULTS = False                           # 保存文本结果 / Save text results
RESULTS_FILE = RESULTS_DIR / "results_spec.txt"              # 文本结果文件名 / Text results filename

SAVE_CSV = True                               # 保存批量结果为CSV / Save batch results to CSV
CSV_OUTPUT_FILE = RESULTS_DIR / "results_spec.csv"     # CSV输出文件名 / CSV output filename

SAVE_FIGURE = True                            # 保存图形 / Save figure
FIGURE_FILE = RESULTS_DIR / "results_spec.png"                # 图形文件名 / Figure filename

# ============================================================================
# 物理常数 / Physical Constants
# ============================================================================

C_EV_NM = 1239.8  # 光子能量-波长转换常数 (eV·nm) / Photon energy-wavelength conversion (eV·nm)
