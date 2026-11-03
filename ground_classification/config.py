"""项目路径与训练参数；默认路径不依赖启动时的工作目录。"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "dataset"
TEST_CSV = PROJECT_ROOT / "X_test.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS = ("lstm", "gru", "rnn", "svm", "xgboost")


@dataclass
class TrainConfig:
    model: str = "lstm"
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR
    epochs: int = 200
    batch_size: int = 128
    hidden_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 60
    test_size: float = 0.2
    seed: int = 42
    device: str = "auto"
    # 保留原实验近似不裁剪的设置；可显式设置为 1.0 等值。
    grad_clip: float = 1e6
    boost_rounds: int = 1000

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        if self.model not in MODELS:
            raise ValueError(f"不支持的模型：{self.model}")
        if min(self.epochs, self.batch_size, self.patience, self.boost_rounds) < 1:
            raise ValueError("epochs、batch_size、patience、boost_rounds 必须为正数")
        if self.hidden_size < 1 or self.hidden_size % 4:
            raise ValueError("hidden_size 必须为正数且能被 4 个注意力头整除")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size 必须在 0 和 1 之间")
        if self.learning_rate <= 0 or self.grad_clip <= 0 or self.weight_decay < 0:
            raise ValueError("学习率和梯度阈值必须为正数，weight_decay 不能为负数")

    def create_run_dir(self, feature_columns):
        run_dir = self.output_dir / self.model / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata = asdict(self)
        metadata.update(feature_columns=list(feature_columns), sequence_length=128)
        write_json(run_dir / "config.json", metadata)
        return run_dir


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
