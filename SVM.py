import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import os

def quaternion_to_euler(x, y, z, w):
    import math
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z


def fe_step0(actual):
    actual['norm_quat'] = (actual['orientation_X'] ** 2 + actual['orientation_Y'] ** 2 + actual['orientation_Z'] ** 2 + actual[
            'orientation_W'] ** 2)
    actual['mod_quat'] = (actual['norm_quat']) ** 0.5
    actual['norm_X'] = actual['orientation_X'] / actual['mod_quat']
    actual['norm_Y'] = actual['orientation_Y'] / actual['mod_quat']
    actual['norm_Z'] = actual['orientation_Z'] / actual['mod_quat']
    actual['norm_W'] = actual['orientation_W'] / actual['mod_quat']

    return actual


def fe_step1(actual):
    x, y, z, w = actual['norm_X'].tolist(), actual['norm_Y'].tolist(), actual['norm_Z'].tolist(), actual[
        'norm_W'].tolist()
    nx, ny, nz = [], [], []
    for i in range(len(x)):
        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])
        nx.append(xx)
        ny.append(yy)
        nz.append(zz)

    actual['euler_x'] = nx
    actual['euler_y'] = ny
    actual['euler_z'] = nz
    return actual


def Basicfeature(data):
    data['totl_anglr_vel'] = (data['angular_velocity_X'] ** 2 + data['angular_velocity_Y'] ** 2 + data[
        'angular_velocity_Z'] ** 2) ** 0.5
    data['totl_linr_acc'] = (data['linear_acceleration_X'] ** 2 + data['linear_acceleration_Y'] ** 2 + data[
        'linear_acceleration_Z'] ** 2) ** 0.5
    data['totl_xyz'] = (data['orientation_X'] ** 2 + data['orientation_Y'] ** 2 + data['orientation_Z'] ** 2) ** 0.5
    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']

    return data


def load_data():
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")


    X_train = fe_step0(X_train)
    X_train = fe_step1(X_train)
    X_train = Basicfeature(X_train)

    sensor_cols = [
        'euler_x', 'euler_y', 'euler_z',
        'angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z',
        'norm_quat', 'mod_quat', 'norm_X', 'norm_Y', 'norm_Z', 'norm_W',
        'linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z',
        'totl_anglr_vel', 'totl_linr_acc', 'totl_xyz', 'acc_vs_vel'

    ]


    # 按 'series_id' 分组，提取所有时序数据并展平为一维数组
    grouped = X_train.groupby('series_id')[sensor_cols].apply(lambda x: x.values.flatten())

    # 转换为 NumPy 数组（形状为 [样本数, 特征数]）
    X_t = np.vstack(grouped.values)

    # 标签编码
    le = LabelEncoder()
    y = le.fit_transform(y_train['surface'])

    return X_t, y, le

def train_model():

    X, y, le = load_data()
    # print(pd.Series(y).value_counts(sort=True))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # param_grid = {
    #     'C':[0.1,1,10],
    #     'gamma':[0.001,0.01]
    # }
    #
    # grid = GridSearchCV(
    #     SVC(kernel='rbf'),
    #     param_grid,
    #     cv=5,
    #     # class_weight='balanced',
    #     scoring='f1_micro',
    #     n_jobs=-1
    # )
    #
    # grid.fit(X_train,y_train)
    #
    # print("Best Parameters:", grid.best_params_)
    # print("Test Accuracy:", grid.score(X_val, y_val))

    model = SVC(
        # C=8, 0.7441
        C = 5,
        kernel='rbf',
        class_weight='balanced',
        gamma='scale',
        probability=True
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    true_label = le.inverse_transform(y_val)
    preds_label = le.inverse_transform(y_pred)

    accuracy = accuracy_score(true_label, preds_label)
    precision_score_average_micro = precision_score(true_label, preds_label, average='macro')
    recall_score_average_micro = recall_score(true_label, preds_label, average='macro')
    f1_score_average_micro = f1_score(true_label, preds_label, average='macro')

    # 打印日志
    print(f"acc:{accuracy:.4f}| "
          f"pre:{precision_score_average_micro:.4f}| "
          f"recall:{recall_score_average_micro:.4f}| "
          f"f1:{f1_score_average_micro:.4f}")


if __name__ == "__main__":
     train_model()

