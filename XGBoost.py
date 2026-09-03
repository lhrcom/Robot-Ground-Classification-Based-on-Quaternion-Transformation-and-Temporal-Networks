import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from collections import Counter
import os
import xgboost as xgb



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

    # grouped = X_train.groupby('series_id')[sensor_cols].agg(['mean', 'std', 'max', 'min'])
    #
    # # 展开多层列名（例如 'euler_x_mean'）
    # grouped.columns = [f'{col}_{stat}' for col in sensor_cols for stat in ['mean', 'std', 'max', 'min']]
    # X_t = grouped.values


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

    # class_weights = len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))
    # sample_weights = np.array([class_weights[y]for y in y_train])
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    dtrain = xgb.DMatrix(X_train,label=y_train,weight=sample_weights)
    dtest = xgb.DMatrix(X_val,label=y_val)



    params = {
        'booster': 'gbtree',
        'verbosity': 1,

        'eta': 0.05,
        'max_depth': 7,

        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 1.0,

        'objective': 'multi:softmax',
        'num_class': len(np.unique(y)),
        'eval_metric': 'mlogloss'

    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dtest, 'eval')],
        early_stopping_rounds=20,
        verbose_eval=1
    )


    y_pred = model.predict(dtest).astype(int)

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

    # 查看测试集的类别分布
    print("测试集分布:", Counter(y_val))
    # 查看预测结果的类别分布
    print("预测结果分布:", Counter(y_pred))

if __name__ == "__main__":
     train_model()

