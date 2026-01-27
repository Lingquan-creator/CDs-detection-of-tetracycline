"""
配置文件：跨设备RGB颜色空间映射
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class MappingConfig:
    """颜色空间映射配置类"""

    # ==================== 数据路径 ====================
    pic1_dir: Path = Path("h:/000/CDs/multi_device_rgb/pic_1")
    pic2_dir: Path = Path("h:/000/CDs/multi_device_rgb/pic_2")
    output_dir: Path = Path("h:/000/CDs/multi_device_rgb/device_mapping/results")

    # ==================== 匹配浓度 ====================
    # 两个设备都有的浓度样本（µM）
    matching_concentrations: Tuple[float, ...] = (40, 80, 120, 160, 200, 240)

    # ==================== 特征设置 ====================
    # 是否使用4通道（R, G1, G2, B）还是3通道（R, G, B）
    use_4_channels: bool = True

    # 多项式阶数（用于多项式岭回归）
    polynomial_degree: int = 2

    # 是否包含比值特征（R/G, B/G等）
    include_ratios: bool = False

    # 是否包含对数特征（log(R), log(G), log(B)）
    include_log_features: bool = False

    # ==================== 模型设置 ====================
    # 映射器类型："ridge", "gp", "interpolation"
    mapper_type: str = "ridge"

    # 岭回归的alpha网格（正则化强度）
    alpha_grid: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

    # 交叉验证策略："loo" (Leave-One-Out)
    cv_strategy: str = "loo"

    # ==================== 高斯过程设置 ====================
    # GP核函数的长度尺度范围
    gp_length_scale_bounds: Tuple[float, float] = (1.0, 100.0)

    # GP信号方差范围
    gp_signal_variance_bounds: Tuple[float, float] = (0.1, 10.0)

    # GP噪声方差范围
    gp_noise_variance_bounds: Tuple[float, float] = (1e-5, 1.0)

    # GP优化器重启次数
    gp_n_restarts: int = 10

    # ==================== 评估设置 ====================
    # 是否计算颜色差异（ΔE）
    compute_delta_e: bool = True

    # 是否测试端到端浓度预测
    test_concentration_prediction: bool = True

    # 浓度模型路径（如果test_concentration_prediction=True）
    concentration_model_path: Path = Path("h:/000/CDs/concentration/rgb_fit_output/concentration_model_v1.pkl")

    # ==================== 可视化设置 ====================
    # 是否保存可视化图表
    save_visualizations: bool = True

    # 图表DPI
    visualization_dpi: int = 150

    # 图表尺寸（英寸）
    visualization_figsize: Tuple[float, float] = (12, 8)

    # ==================== 其他设置 ====================
    # 随机种子（用于可重复性）
    random_seed: int = 42

    # 是否显示详细日志
    verbose: bool = True

    def __post_init__(self):
        """初始化后处理：确保路径存在"""
        # 转换为Path对象
        self.pic1_dir = Path(self.pic1_dir)
        self.pic2_dir = Path(self.pic2_dir)
        self.output_dir = Path(self.output_dir)
        self.concentration_model_path = Path(self.concentration_model_path)

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_pic1_stats_file(self) -> Path:
        """获取pic_1的统计文件路径"""
        return self.pic1_dir / "roi_results" / "roi_detailed_statistics.csv"

    def get_pic2_stats_file(self) -> Path:
        """获取pic_2的统计文件路径"""
        return self.pic2_dir / "roi_results" / "roi_detailed_statistics.csv"

    def validate(self) -> bool:
        """验证配置是否有效"""
        errors = []

        # 检查数据目录
        if not self.pic1_dir.exists():
            errors.append(f"pic_1目录不存在: {self.pic1_dir}")

        if not self.pic2_dir.exists():
            errors.append(f"pic_2目录不存在: {self.pic2_dir}")

        # 检查统计文件
        if not self.get_pic1_stats_file().exists():
            errors.append(f"pic_1统计文件不存在: {self.get_pic1_stats_file()}")
            errors.append("请先运行pic_RGB流程: python -m pic_RGB.extraction.roi_processor")

        if not self.get_pic2_stats_file().exists():
            errors.append(f"pic_2统计文件不存在: {self.get_pic2_stats_file()}")
            errors.append("请先运行pic_RGB流程: python -m pic_RGB.extraction.roi_processor")

        # 检查匹配浓度
        if len(self.matching_concentrations) < 3:
            errors.append(f"匹配浓度太少: {len(self.matching_concentrations)}，至少需要3对")

        # 检查多项式阶数
        if self.polynomial_degree < 1 or self.polynomial_degree > 3:
            errors.append(f"多项式阶数应在1-3之间: {self.polynomial_degree}")

        # 检查mapper类型
        valid_mappers = ["ridge", "gp", "interpolation"]
        if self.mapper_type not in valid_mappers:
            errors.append(f"无效的mapper类型: {self.mapper_type}，应为{valid_mappers}之一")

        # 如果有错误，打印并返回False
        if errors:
            print("配置验证失败:")
            for error in errors:
                print(f"  - {error}")
            return False

        return True

    def print_summary(self):
        """打印配置摘要"""
        print("=" * 60)
        print("跨设备RGB颜色空间映射配置")
        print("=" * 60)
        print(f"pic_1目录: {self.pic1_dir}")
        print(f"pic_2目录: {self.pic2_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"匹配浓度: {self.matching_concentrations}")
        print(f"通道数: {'4 (R, G1, G2, B)' if self.use_4_channels else '3 (R, G, B)'}")
        print(f"映射器类型: {self.mapper_type}")
        if self.mapper_type == "ridge":
            print(f"多项式阶数: {self.polynomial_degree}")
            print(f"Alpha网格: {self.alpha_grid}")
        print(f"交叉验证: {self.cv_strategy}")
        print(f"计算ΔE: {self.compute_delta_e}")
        print(f"测试浓度预测: {self.test_concentration_prediction}")
        print("=" * 60)


# 默认配置实例
default_config = MappingConfig()


if __name__ == "__main__":
    # 测试配置
    config = MappingConfig()
    config.print_summary()

    print("\n验证配置...")
    if config.validate():
        print("✓ 配置有效")
    else:
        print("✗ 配置无效")
