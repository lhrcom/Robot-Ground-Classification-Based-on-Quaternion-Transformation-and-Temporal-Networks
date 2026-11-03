"""兼容原欧拉角工具导入路径，计算实现位于公共特征模块。"""

from ground_classification.features import (
    cal_euler_angle,
    calculate_unit_quaternion,
    quaternion_to_euler,
)

__all__ = ["cal_euler_angle", "calculate_unit_quaternion", "quaternion_to_euler"]
