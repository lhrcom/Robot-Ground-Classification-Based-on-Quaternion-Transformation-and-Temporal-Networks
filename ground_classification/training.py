"""LSTM / GRU / RNN 共用训练流程与实验产物。"""

import csv
import random

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainConfig, write_json
from .data import load_training_data, scale_sequences
from .features import feature_columns
from .metrics import classification_metrics
from .models import SensorDataset, build_model, count_model_parameters, select_device
from .plotting import train_process_picture


def train_temporal(config=None):
    config = config or TrainConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    values, targets, encoder = load_training_data(config.model, config.data_dir)
    X_train, X_val, y_train, y_val = train_test_split(
        values, targets, test_size=config.test_size, random_state=config.seed,
    )
    scaler = StandardScaler()
    X_train = scale_sequences(scaler, X_train, fit=True)
    X_val = scale_sequences(scaler, X_val)
    train_loader = DataLoader(SensorDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(SensorDataset(X_val, y_val), batch_size=config.batch_size)
    model = build_model(config.model, values.shape[-1], config.hidden_size, len(encoder.classes_)).to(device)
    count_model_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.model == "lstm":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-6)

    run_dir = config.create_run_dir(feature_columns(config.model))
    joblib.dump(scaler, run_dir / "scaler.pkl")
    joblib.dump(encoder, run_dir / "label_encoder.pkl")
    print(f"Device: {device} | Output: {run_dir}")
    best_accuracy, stale_epochs = -1.0, 0
    history = []

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for epoch in range(1, config.epochs + 1):
            model.train()
            train_loss, total_grad_norm = 0.0, 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
                total_grad_norm += norm.item()
                optimizer.step()
                train_loss += loss.item() * len(labels)

            model.eval()
            validation_loss, predictions, actual = 0.0, [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    validation_loss += criterion(outputs, labels).item() * len(labels)
                    predictions.extend(outputs.argmax(dim=1).cpu().tolist())
                    actual.extend(labels.cpu().tolist())
            metrics = classification_metrics(actual, predictions)
            row = {
                "epoch": epoch,
                "train_loss": train_loss / len(train_loader.dataset),
                "val_loss": validation_loss / len(val_loader.dataset),
                **metrics,
                "grad_norm": total_grad_norm / len(train_loader),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()

            # Plateau 接收验证损失；余弦重启接收训练轮次。
            scheduler.step(row["val_loss"] if config.model == "lstm" else epoch)
            if metrics["accuracy"] > best_accuracy:
                best_accuracy, stale_epochs = metrics["accuracy"], 0
                torch.save(model.state_dict(), run_dir / "best_model.pth")
                write_json(run_dir / "metrics.json", row)
            else:
                stale_epochs += 1
            print(
                f"Epoch {epoch:03d} | Train Loss: {row['train_loss']:.4f} | "
                f"Val Loss: {row['val_loss']:.4f} | Acc: {metrics['accuracy']:.4f} | "
                f"F1 (macro): {metrics['f1_macro']:.4f}"
            )
            if stale_epochs >= config.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    train_process_picture(
        [row["epoch"] for row in history], [row["train_loss"] for row in history],
        [row["val_loss"] for row in history], [row["accuracy"] for row in history],
        [row["grad_norm"] for row in history], output_path=run_dir / "training.png", show=False,
    )
    return run_dir
