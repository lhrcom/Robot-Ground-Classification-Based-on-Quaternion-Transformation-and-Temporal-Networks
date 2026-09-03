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
import math
from euler_angle import *
from visual_training_proccess import train_process_picture

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_model_parameters(model):
    """
    return model parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params


def load_data():

    X_train = pd.read_csv("dataset/X_train.csv")
    y_train = pd.read_csv("dataset/y_train.csv")

    X_train = cal_euler_angle(X_train)

    sensor_cols = [
         'euler_z', 'euler_y', 'euler_x',
        'angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z',
        'linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z',
        ]

    series_ids = X_train['series_id'].unique()
    
    
    X_3d = []

    for sid in series_ids:
        seq_data = X_train[X_train['series_id'] == sid][sensor_cols].values
        X_3d.append(seq_data)

    X_3d = np.stack(X_3d)  # X_3d = np.stack(X_3d) 适配深度学习模型（如 LSTM）的输入格式要求，即 (batch_size, sequence_length, input_features)

    # 标签编码
    le = LabelEncoder()
    y = le.fit_transform(y_train['surface'])

    return X_3d, y, le, len(sensor_cols)



class SensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """
    LSTM
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = 3,  # 关键修改：层数>1才能使用dropout
            batch_first = True,
            dropout = 0.05
        )
        self.attention = nn.MultiheadAttention(
            embed_dim = hidden_size,
            num_heads = 4,
            batch_first = True
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
        lstm_out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.fc(attn_out[:, -1, :])  # 取最后一个时间步
        return out


def train_model():
    # 加载数据
    X, y, le, len_var = load_data()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    # class_weights = torch.FloatTensor(class_weights).to(device)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, len_var)).reshape(-1, 128, len_var)
    X_val = scaler.transform(X_val.reshape(-1, len_var)).reshape(-1, 128, len_var)

    # 保存预处理对象
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')

    
    train_loader = DataLoader(SensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(SensorDataset(X_val, y_val), batch_size=128)

    
    model = LSTMModel(
        input_size = len_var,
        hidden_size = 128,
        num_classes = len(le.classes_)
    ).to(device)

    count_model_parameters(model)

    # 训练配置
    criterion = nn.CrossEntropyLoss()  # 平均损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

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
        
        # print training log
        print(f"Epoch {epoch + 1:03d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}| "
              f"pre:{precision_score_average_micro:.4f}| "
              f"recall:{recall_score_average_micro:.4f}| "
              f"f1:{f1_score_average_micro:.4f}")

        print(f"当前显存占用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    
    # visualize picture
    train_process_picture(epoch_list, train_loss_list, val_loss_list, val_acc_list, grad_norm_list)



if __name__ == "__main__":

    train_model()
    