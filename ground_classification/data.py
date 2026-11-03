"""按 series_id 对齐样本与标签，并按 measurement_number 排列时间步。"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import DATA_DIR
from .features import MOTION_COLUMNS, QUATERNION_COLUMNS, basic_features, cal_euler_angle, feature_columns


def load_sequences(csv_path, columns, sequence_length=128):
    frame = pd.read_csv(csv_path)
    required = {"series_id", "measurement_number", *MOTION_COLUMNS, *QUATERNION_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} 缺少列：{sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("传感器数据为空或含有缺失值")
    if frame.duplicated(["series_id", "measurement_number"]).any():
        raise ValueError("同一序列存在重复的 measurement_number")
    frame = cal_euler_angle(frame)
    if "acc_vs_vel" in columns:
        frame = basic_features(frame)

    series_ids, sequences = [], []
    for series_id, group in frame.groupby("series_id", sort=False):
        group = group.sort_values("measurement_number")
        if len(group) != sequence_length:
            raise ValueError(f"序列 {series_id} 长度为 {len(group)}，应为 {sequence_length}")
        if not np.array_equal(group["measurement_number"].to_numpy(), np.arange(sequence_length)):
            raise ValueError(f"序列 {series_id} 的 measurement_number 必须连续且从 0 开始")
        series_ids.append(series_id)
        sequences.append(group[list(columns)].to_numpy(dtype=float))
    values = np.stack(sequences)
    if not np.isfinite(values).all():
        raise ValueError("传感器特征包含 NaN 或无穷值")
    return values, np.asarray(series_ids)


def load_training_data(model="lstm", data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    values, series_ids = load_sequences(data_dir / "X_train.csv", feature_columns(model))
    labels = pd.read_csv(data_dir / "y_train.csv")
    if not {"series_id", "surface"}.issubset(labels.columns):
        raise ValueError("标签文件必须包含 series_id 和 surface")
    if labels[["series_id", "surface"]].isna().any().any() or labels["series_id"].duplicated().any():
        raise ValueError("标签存在空值或重复的 series_id")
    labels = labels.set_index("series_id")
    if set(labels.index) != set(series_ids):
        raise ValueError("传感器数据与标签文件的 series_id 不一致")
    encoder = LabelEncoder()
    targets = encoder.fit_transform(labels.loc[series_ids, "surface"])
    if model in ("svm", "xgboost"):
        values = values.reshape(len(values), -1)
    return values, targets, encoder


def scale_sequences(scaler, values, fit=False):
    """仅训练集 fit，验证集和预测数据复用同一 scaler。"""
    shape = values.shape
    transform = scaler.fit_transform if fit else scaler.transform
    return transform(values.reshape(-1, shape[-1])).reshape(shape)
