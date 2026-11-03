"""SVM 兼容入口；公共实现位于 ground_classification。"""

from ground_classification.baselines import train_baseline
from ground_classification.config import DATA_DIR, TrainConfig
from ground_classification.data import load_training_data


def load_data(data_dir=DATA_DIR):
    return load_training_data("svm", data_dir)


def train_model(**kwargs):
    return train_baseline(TrainConfig(model="svm", **kwargs))


if __name__ == "__main__":
    import sys
    from ground_classification.cli import main
    main(["train", "--model", "svm", *sys.argv[1:]])
