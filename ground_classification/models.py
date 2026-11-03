"""三层时序网络 + 四头自注意力 + 全连接分类头。"""

import torch
from torch import nn
from torch.utils.data import Dataset


class SensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class TemporalModel(nn.Module):
    def __init__(self, kind, input_size, hidden_size, num_classes):
        super().__init__()
        if hidden_size <= 0 or hidden_size % 4:
            raise ValueError("hidden_size 必须为正数且能被 4 整除")
        self.kind = kind
        recurrent = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[kind]
        # 保留 lstm/gru/rnn、attention、fc 的权重键，兼容原 state_dict。
        setattr(self, kind, recurrent(
            input_size=input_size, hidden_size=hidden_size, num_layers=3,
            batch_first=True, dropout=0.05,
        ))
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs):
        sequence, _ = getattr(self, self.kind)(inputs)
        attended, _ = self.attention(sequence, sequence, sequence)
        return self.fc(attended[:, -1, :])


class LSTMModel(TemporalModel):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__("lstm", input_size, hidden_size, num_classes)


class GRUModel(TemporalModel):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__("gru", input_size, hidden_size, num_classes)


class RNNModel(TemporalModel):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__("rnn", input_size, hidden_size, num_classes)


def build_model(kind, input_size, hidden_size, num_classes):
    return {"lstm": LSTMModel, "gru": GRUModel, "rnn": RNNModel}[kind](
        input_size, hidden_size, num_classes,
    )


def count_model_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}\nTrainable parameters: {trainable:,}")
    return total, trainable


def select_device(requested="auto"):
    # 默认保持原实验的 CUDA / CPU 行为；Apple 芯片可显式选择 mps。
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前环境无法使用 CUDA，请选择 cpu 或 auto")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("当前环境无法使用 MPS，请选择 cpu 或 auto")
    return torch.device(requested)
