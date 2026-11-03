"""四元数归一化、欧拉角转换及传统模型的派生特征。角度单位为弧度。"""

import math

import numpy as np

MOTION_COLUMNS = (
    "angular_velocity_X", "angular_velocity_Y", "angular_velocity_Z",
    "linear_acceleration_X", "linear_acceleration_Y", "linear_acceleration_Z",
)
EULER_COLUMNS = ("euler_x", "euler_y", "euler_z")
QUATERNION_COLUMNS = ("orientation_X", "orientation_Y", "orientation_Z", "orientation_W")
CLASSICAL_COLUMNS = EULER_COLUMNS + MOTION_COLUMNS[:3] + (
    "norm_quat", "mod_quat", "norm_X", "norm_Y", "norm_Z", "norm_W",
) + MOTION_COLUMNS[3:] + ("totl_anglr_vel", "totl_linr_acc", "totl_xyz", "acc_vs_vel")


def feature_columns(model):
    if model in ("svm", "xgboost"):
        return CLASSICAL_COLUMNS
    if model == "lstm":
        # 原 LSTM 使用 z/y/x 顺序，保留以兼容已有权重和 scaler。
        return ("euler_z", "euler_y", "euler_x") + MOTION_COLUMNS
    if model in ("gru", "rnn"):
        return EULER_COLUMNS + MOTION_COLUMNS
    raise ValueError(f"不支持的模型：{model}")


def quaternion_to_euler(x, y, z, w):
    """将单位四元数 (x, y, z, w) 转换为 (roll, pitch, yaw)。"""
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def calculate_unit_quaternion(frame):
    """返回含归一化四元数的新表，不修改传入数据。"""
    frame = frame.copy()
    values = frame[list(QUATERNION_COLUMNS)].to_numpy(dtype=float)
    norms_squared = (values ** 2).sum(axis=1)
    if not np.isfinite(norms_squared).all() or (norms_squared <= 0).any():
        raise ValueError("四元数包含零模长或非有限值，无法归一化")
    frame["norm_quat"] = norms_squared
    frame["mod_quat"] = np.sqrt(norms_squared)
    for axis in "XYZW":
        frame[f"norm_{axis}"] = frame[f"orientation_{axis}"] / frame["mod_quat"]
    return frame


def add_euler_angles(frame):
    """对已归一化的四元数表添加欧拉角。"""
    frame = frame.copy()
    x, y, z, w = (frame[f"norm_{axis}"].to_numpy() for axis in "XYZW")
    frame["euler_x"] = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    frame["euler_y"] = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    frame["euler_z"] = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return frame


def cal_euler_angle(frame):
    return add_euler_angles(calculate_unit_quaternion(frame))


def basic_features(frame):
    frame = frame.copy()
    frame["totl_anglr_vel"] = np.sqrt((frame[list(MOTION_COLUMNS[:3])] ** 2).sum(axis=1))
    frame["totl_linr_acc"] = np.sqrt((frame[list(MOTION_COLUMNS[3:])] ** 2).sum(axis=1))
    frame["totl_xyz"] = np.sqrt((frame[list(QUATERNION_COLUMNS[:3])] ** 2).sum(axis=1))
    # 静止时角速度可能为零，约定比值为零，避免产生 inf。
    frame["acc_vs_vel"] = np.divide(
        frame["totl_linr_acc"], frame["totl_anglr_vel"],
        out=np.zeros(len(frame)), where=frame["totl_anglr_vel"].to_numpy() != 0,
    )
    return frame
