import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    # X_train = Basicfeature(X_train)

    sensor_cols = [
        'euler_x', 'euler_y', 'euler_z',
        'angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z',

        'linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z',


    ]

    series_ids = X_train['series_id'].unique()
    X_3d = []
    for sid in series_ids:
        seq_data = X_train[X_train['series_id'] == sid][sensor_cols].values
        X_3d.append(seq_data)
    X_3d = np.stack(
        X_3d)  # X_3d = np.stack(X_3d) 适配深度学习模型（如 LSTM）的输入格式要求，即 (batch_size, sequence_length, input_features)

    # 标签编码
    le = LabelEncoder()
    y = le.fit_transform(y_train['surface'])

    return X_3d, y, le


class SensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity='tanh',
            bias=True,
            num_layers=3,  # 关键修改：层数>1才能使用dropout
            batch_first=True,
            dropout=0.05
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Sequential(

            nn.Linear(128, 256),  # 过渡层
            nn.LayerNorm(256),  # 添加层标准化
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        rnn_out, _ = self.rnn(x)  # out: (batch, seq_len, hidden_size)
        attn_out, _ = self.attention(rnn_out, rnn_out, rnn_out)
        out = self.fc(attn_out[:, -1, :])  # 取最后一个时间步
        return out


def train_model():
    # 加载数据
    X, y, le = load_data()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, 9)).reshape(-1, 128, 9)

    # 验证集使用相同的scaler转换
    X_val = scaler.transform(X_val.reshape(-1, 9)).reshape(-1, 128, 9)



    # 保存预处理对象
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')

    # 创建DataLoader
    train_loader = DataLoader(SensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(SensorDataset(X_val, y_val), batch_size=128)

    # 初始化模型
    model = RNNModel(
        input_size=9,
        hidden_size=128,
        num_classes=len(le.classes_)
    ).to(device)

    # 训练配置
    criterion = nn.CrossEntropyLoss()  # 平均损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,
        T_mult=1,
        eta_min=1e-6
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    best_acc = 0.0  # 训练准确率历史最佳
    early_stop_counter = 0

    epoch_list = []
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    grad_norm_list = []  # 新增：记录梯度范数

    # 训练循环
    for epoch in range(200):
        model.train()
        train_loss = 0.0

        total_grad_norm = 0.0;

        # 训练阶段
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 计算全局梯度L2范数（不裁剪）
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1e6, norm_type=2
            )
            total_grad_norm += grad_norm.item()

            optimizer.step()
            train_loss += loss.item()

            # 计算平均梯度范数
        avg_grad_norm = total_grad_norm / len(train_loader)
        print(f"Epoch {epoch + 1:03d} | Avg Grad Norm: {avg_grad_norm:.4f}")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        true_label = []
        preds_label = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                preds_label.append(preds)
                true_label.append(labels)
                correct += (preds == labels).sum().item()

        preds_label = torch.cat(preds_label).cpu().numpy()
        true_label = torch.cat(true_label).cpu().numpy()

        # 计算指标
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / len(val_loader.dataset)
        precision_score_average_micro = precision_score(true_label, preds_label, average='macro')
        recall_score_average_micro = recall_score(true_label, preds_label, average='macro')
        f1_score_average_micro = f1_score(true_label, preds_label, average='macro')

        epoch_list.append(epoch)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        # 在训练循环中保存梯度范数
        grad_norm_list.append(avg_grad_norm)

        # 学习率调整
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= 60:
                print(f"Early stopping at epoch {epoch + 1}")
                # return
                break
        # 打印日志
        print(f"Epoch {epoch + 1:03d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}| "
              f"pre:{precision_score_average_micro:.4f}| "
              f"recall:{recall_score_average_micro:.4f}| "
              f"f1:{f1_score_average_micro:.4f}")

        print(f"当前显存占用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    fig, ax1 = plt.subplots()
    ax1.plot(epoch_list, train_loss_list, 'b-', label='Train Loss')
    ax1.plot(epoch_list, val_loss_list, 'g-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(epoch_list, val_acc_list, 'r-', label='Val Acc')
    ax2.plot(epoch_list, grad_norm_list, 'm--', label='Grad Norm')  # 新增梯度范数曲线
    ax2.set_ylabel('Accuracy', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.legend(loc='upper right')
    plt.show()


# -------------------- 预测函数 --------------------

def predict():
    torch.cuda.empty_cache()  # 清空缓存
    # 确保模型文件存在
    if not os.path.exists("best_model.pth"):
        raise FileNotFoundError("未找到模型文件，请先运行训练流程")

    import joblib
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')

    # 定义传感器列（与训练时一致）
    sensor_cols = [
        'orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W',
        'angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z',
        'linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z',
        'norm_quat', 'mod_quat', 'norm_X', 'norm_Y', 'norm_Z', 'norm_W'
    ]

    # 加载预处理对象
    # _, _, scaler, le = load_data()

    # 加载测试数据
    X_test = pd.read_csv("X_test.csv")
    X_test = fe_step0(X_test)
    test_series_ids = X_test['series_id'].unique()
    X_test_3d = np.stack([
        X_test[X_test['series_id'] == sid][sensor_cols].values
        for sid in test_series_ids
    ])
    X_test_3d = scaler.transform(X_test_3d.reshape(-1, 10)).reshape(-1, 128, 10)
    # test_tensor = torch.FloatTensor(X_test_3d).to(device)

    # 转换为TensorDataset
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_test_3d))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)  # 调整batch_size

    model = LSTMModel(
        input_size=16,
        hidden_size=128,
        num_classes=len(le.classes_)
    ).to(device)
    model.load_state_dict(torch.load("best_model.pth"))  # 加载保存的模型权重

    # 预测
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)

    submission = pd.read_csv("sample_submission.csv")
    submission['surface'] = le.inverse_transform(all_preds)
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":

    train_model()

    try:
        predict()
        print("预测完成，结果已保存至 submission_1.csv")
    except FileNotFoundError as e:
        print(str(e))