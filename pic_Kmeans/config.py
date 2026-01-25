"""
K-means 分割配置文件
"""

from pathlib import Path

# ============================================
# 路径配置
# ============================================
DATA_DIR = Path(r"h:\000\CDs\pic_Kmeans\a")       # 输入DNG文件目录
OUTPUT_DIR = Path(r"h:\000\CDs\pic_Kmeans\roi_results")  # 输出结果目录

# ============================================
# K-means 参数
# ============================================
N_CLUSTERS = 3  # 聚类数量（3表示背景、瓶体、高亮液体）- 已弃用，使用下面的分层参数
RANDOM_STATE = 42  # 随机种子，保证结果可重复
N_INIT = 10  # K-means初始化次数

# 嵌套K-means分层聚类数量
N_CLUSTERS_LEVEL1 = 2  # 第一层聚类数量（背景 vs 高亮区域）
N_CLUSTERS_LEVEL2 = 4  # 第二层聚类数量（瓶体 vs 高亮液体）

# 位置权重参数
USE_POSITION_WEIGHT = True  # 是否使用位置权重
POSITION_WEIGHT_FACTOR = 0.3  # 位置权重因子（0-1，越大越倾向中心）

# 液体区域约束
MIN_LIQUID_AREA_RATIO = 0.007  # 液体最小面积比例（0.7%）
MAX_LIQUID_AREA_RATIO = 0.05   # 液体最大面积比例（5%）
MAX_EDGE_DISTANCE_RATIO = 0.15  # 液体中心离边缘的最大距离比例（15%）

# 阴影填充参数
FILL_LIQUID_HOLES = True  # 是否填充液体内部的空洞（阴影）
HOLE_FILL_KERNEL_SIZE = 15  # 空洞填充的形态学核大小

# ============================================
# 图像处理参数
# ============================================
# DNG转换参数
USE_CAMERA_WB = True  # 使用相机白平衡
OUTPUT_BPS = 16  # 输出位深度
NO_AUTO_BRIGHT = True  # 不自动调整亮度

# 形态学操作参数（用于优化掩码）
MORPHOLOGY_KERNEL_SIZE = 5  # 形态学操作的核大小
APPLY_MORPHOLOGY = True  # 是否应用形态学操作来平滑掩码

# ============================================
# 可视化参数
# ============================================
SAVE_VISUALIZATION = True  # 是否保存可视化结果
SAVE_MASKS = False  # 是否保存掩码图像（已关闭，仅保存可视化）
VISUALIZATION_DPI = 150  # 可视化图像DPI
FIGURE_SIZE = (18, 12)  # 图像尺寸

# ============================================
# 高级分析参数
# ============================================
# 是否计算高亮区域的连通组件
ANALYZE_CONNECTED_COMPONENTS = True

# 连通组件的最小面积阈值（像素）
MIN_COMPONENT_AREA = 500

# 是否分析颜色通道
ANALYZE_COLOR_CHANNELS = True

# ============================================
# 输出配置
# ============================================
SAVE_JSON_RESULTS = True  # 保存JSON格式的详细结果
SAVE_CSV_SUMMARY = True  # 保存CSV格式的汇总表（推荐用Excel打开）
