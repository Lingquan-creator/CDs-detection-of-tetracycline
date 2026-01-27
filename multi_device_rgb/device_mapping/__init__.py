"""
跨设备RGB颜色空间映射模块

将不同手机设备拍摄的RGB颜色空间进行映射对齐，
使得在参考设备上训练的浓度预测模型可以应用于其他设备的数据。
"""

from .config import MappingConfig

__version__ = "1.0.0"
__all__ = ["MappingConfig"]
