"""GRU 兼容入口；公共实现位于 ground_classification。"""

from ground_classification.config import DATA_DIR, TrainConfig
from ground_classification.data import load_training_data
from ground_classification.models import GRUModel, SensorDataset
from ground_classification.prediction import predict as predict_run
from ground_classification.training import train_temporal


def load_data(data_dir=DATA_DIR):
    data = load_training_data("gru", data_dir)
    return data


def train_model(**kwargs):
    return train_temporal(TrainConfig(model="gru", **kwargs))


def predict(run_dir, **kwargs):
    return predict_run(run_dir, model="gru", **kwargs)


if __name__ == "__main__":
    import sys
    from ground_classification.cli import main
    main(["train", "--model", "gru", *sys.argv[1:]])
