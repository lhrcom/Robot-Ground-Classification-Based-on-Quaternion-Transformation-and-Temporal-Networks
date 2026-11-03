"""统一命令入口；--help 无需导入机器学习依赖。"""

import argparse
from pathlib import Path

from .config import DATA_DIR, MODELS, OUTPUT_DIR, TEST_CSV, TrainConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description="机器人地面分类：训练与预测")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="训练一个模型并保存独立实验目录")
    train.add_argument("--model", choices=MODELS, default="lstm")
    train.add_argument("--data-dir", type=Path, default=DATA_DIR)
    train.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--hidden-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--patience", type=int, default=60)
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--grad-clip", type=float, default=1e6)
    train.add_argument("--boost-rounds", type=int, default=1000)
    train.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    prediction = commands.add_parser("predict", help="加载一次实验的配置和权重进行预测")
    prediction.add_argument("--run-dir", type=Path, required=True)
    prediction.add_argument("--input-csv", type=Path, default=TEST_CSV)
    prediction.add_argument("--output-csv", type=Path)
    prediction.add_argument("--model", choices=MODELS, help="可选，用于核对训练配置中的模型类型")
    prediction.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    prediction.add_argument("--batch-size", type=int, default=128)
    arguments = vars(parser.parse_args(argv))
    command = arguments.pop("command")
    if command == "train":
        config = TrainConfig(**arguments)
        if config.model in ("svm", "xgboost"):
            from .baselines import train_baseline
            return train_baseline(config)
        from .training import train_temporal
        return train_temporal(config)
    from .prediction import predict
    return predict(**arguments)
