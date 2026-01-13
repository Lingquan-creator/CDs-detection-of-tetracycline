"""
ROI 提取器配置
- 仅保留默认配置 DEFAULT_CONFIG 与可手动修改的 MANUAL_CONFIG（均为 dict）
- 修改 MANUAL_CONFIG 可覆盖默认值，CURRENT_CONFIG 指向生效配置
"""

from pathlib import Path

# ============================================
# 路径配置
# ============================================
DATA_DIR = Path(r'h:\000\CDs\pic_RGB\data_pic')       # 输入 DNG 目录
OUTPUT_DIR = Path(r'h:\000\CDs\pic_RGB\roi_results')  # 输出目录

# ============================================
# ROI 参数（字典形式）
# ============================================
DEFAULT_CONFIG = {
    # ROI识别模式："manual" 使用配置参数，"auto" 自动识别
    "roi_mode": "auto",  # "manual" 或 "auto"
    
    # 矩形 ROI 尺寸（像素），仅在manual模式下使用
    "fixed_roi_width": 120,
    "fixed_roi_height": 400,

    # ROI 旋转角度（度），逆时针为正，仅在manual模式下使用
    "roi_angle_deg": 0.0,

    # 背景环带（像素）：避开荧光主体的矩形环
    # bg_ring_inner: 内矩形相对ROI的扩展距离
    # bg_ring_outer: 外矩形相对ROI的扩展距离
    # 背景采样区域 = 外矩形 - 内矩形
    "bg_ring_inner": 80,
    "bg_ring_outer": 240,

    # 分割与去噪
    "min_area": 1000,          # 过滤过小连通域（噪声）
    "morph_kernel_size": 5,    # 形态学核大小，控制开/闭运算强度

    # 可选：不用 Otsu 时的阈值百分位（默认仍用 Otsu）
    "otsu_percentile": 0.85,
    
    # 中央区域优先：优先选择靠近图像中心的荧光区域
    "prefer_center": True,
    "center_weight": 0.3,  # 中心距离的权重（0-1），越大越优先中央
    "center_region_ratio": 0.7,  # 中心区域范围（0-1），只在图像中心这个比例的区域内搜索
    "edge_exclusion_ratio": 0.15,  # 边缘排除比例（0-1），排除图像边缘这个比例的区域
    
    # 自动模式参数
    "auto_roi_margin": 0.8,  # ROI边界收缩系数（0-1），避免包含边缘噪声
    "auto_color_threshold": 0.6,  # 黄白色判定阈值，R+G > B*threshold
    "auto_brightness_percentile": 95,  # 亮度百分位，用于判定高亮区域
    "auto_remove_cap": True,  # 自动移除瓶盖
    "cap_width_ratio": 1.3,  # 瓶盖宽度阈值：如果上部宽度 > 中部宽度 * ratio，认为是瓶盖
    "cap_detection_height": 0.3,  # 检测上部30%区域作为可能的瓶盖
}

# 手动配置：在此修改以覆盖默认值
MANUAL_CONFIG = {
    **DEFAULT_CONFIG,
    "roi_mode": "auto",  # 切换为"manual"使用固定尺寸
    "fixed_roi_width": 120,
    "fixed_roi_height": 120,
    "bg_ring_inner": 90,
    "bg_ring_outer": 140,
    "min_area": 800,
    "morph_kernel_size": 5,
    # "roi_angle_deg": 0.0,  # 需要时可覆盖角度
    "prefer_center": True,
    "center_weight": 0.7,  # 强烈优先中央区域
    "center_region_ratio": 0.6,  # 限制在中心60%区域搜索
    "edge_exclusion_ratio": 0.2,  # 排除边缘20%区域
    "auto_roi_margin": 0.85,  # 自动ROI收缩比例
}

# 当前生效配置（可切换为 DEFAULT_CONFIG 以回退）
CURRENT_CONFIG = MANUAL_CONFIG

# ============================================
# 输出控制
# ============================================
SAVE_VISUALIZATION = True
VISUALIZATION_DPI = 150
VISUALIZATION_FIGSIZE = (15, 10)

# ============================================
# 辅助选项
# ============================================
VERBOSE = True
IMAGE_EXTENSION = '*.dng'

# ============================================
# 辅助函数
# ============================================
def get_config_dict():
    """返回当前配置的浅拷贝"""
    return dict(CURRENT_CONFIG)

def print_config():
    """打印当前配置"""
    print("=" * 50)
    print("当前 ROI 提取器配置")
    print("=" * 50)
    print(f"数据路径: {DATA_DIR}")
    print(f"输出路径: {OUTPUT_DIR}")
    print(f"\nROI 参数:")
    print(f"  固定 ROI 尺寸: {CURRENT_CONFIG['fixed_roi_width']} x {CURRENT_CONFIG['fixed_roi_height']} px")
    print(f"  ROI 旋转角度: {CURRENT_CONFIG['roi_angle_deg']} deg")
    print(f"  背景环内径: {CURRENT_CONFIG['bg_ring_inner']} px")
    print(f"  背景环外径: {CURRENT_CONFIG['bg_ring_outer']} px")
    print(f"\n分割参数:")
    print(f"  最小区域面积: {CURRENT_CONFIG['min_area']} px^2")
    print(f"  形态学核大小: {CURRENT_CONFIG['morph_kernel_size']} px")
    print(f"\n输出设置:")
    print(f"  保存可视化: {SAVE_VISUALIZATION}")
    print(f"  图像 DPI: {VISUALIZATION_DPI}")
    print("=" * 50)

if __name__ == '__main__':
    print_config()
