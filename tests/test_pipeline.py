"""使用小型合成数据检查特征与端到端流程，不改动仓库数据或权重。"""

import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ground-classification-matplotlib"))

import numpy as np
import pandas as pd
import torch

from ground_classification.baselines import train_baseline
from ground_classification.config import TrainConfig
from ground_classification.data import load_sequences, load_training_data
from ground_classification.features import (
    basic_features, cal_euler_angle, feature_columns, quaternion_to_euler,
)
from ground_classification.models import build_model
from ground_classification.prediction import predict
from ground_classification.training import train_temporal


def make_data(path, count=16):
    rng = np.random.default_rng(7)
    rows = []
    for series_id in range(count):
        for step in range(128):
            rows.append({
                "series_id": series_id, "measurement_number": step,
                "orientation_X": 0.0, "orientation_Y": 0.0,
                "orientation_Z": math.sin(step / 256), "orientation_W": math.cos(step / 256),
                **{name: rng.normal(series_id % 2, 0.1) for name in feature_columns("lstm")[3:]},
            })
    frame = pd.DataFrame(rows)
    labels = pd.DataFrame({"series_id": np.arange(count), "surface": ["wood" if i % 2 else "concrete" for i in range(count)]})
    # 输入表和标签表都故意打乱顺序。
    frame.sample(frac=1, random_state=3).to_csv(path / "X_train.csv", index=False)
    labels.sample(frac=1, random_state=5).to_csv(path / "y_train.csv", index=False)
    frame[frame.series_id.isin([3, 1])].sample(frac=1, random_state=4).to_csv(path / "X_test.csv", index=False)
    return frame, labels


class FeatureTests(unittest.TestCase):
    def test_euler_known_rotation_and_input_unchanged(self):
        frame = pd.DataFrame({"orientation_X": [0.0], "orientation_Y": [0.0],
                              "orientation_Z": [2 * math.sin(math.pi / 4)],
                              "orientation_W": [2 * math.cos(math.pi / 4)]})
        original = frame.copy()
        transformed = cal_euler_angle(frame)
        np.testing.assert_allclose(transformed[["euler_x", "euler_y", "euler_z"]], [[0, 0, math.pi / 2]], atol=1e-12)
        pd.testing.assert_frame_equal(frame, original)

    def test_vectorized_angles_match_scalar_reference(self):
        values = np.random.default_rng(42).normal(size=(20, 4))
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        frame = pd.DataFrame(values, columns=[f"orientation_{axis}" for axis in "XYZW"])
        expected = np.array([quaternion_to_euler(*row) for row in values])
        np.testing.assert_allclose(cal_euler_angle(frame)[["euler_x", "euler_y", "euler_z"]], expected, atol=1e-12)

    def test_zero_quaternion_rejected(self):
        frame = pd.DataFrame({f"orientation_{axis}": [0.0] for axis in "XYZW"})
        with self.assertRaisesRegex(ValueError, "零模长"):
            cal_euler_angle(frame)

    def test_zero_velocity_ratio_is_finite(self):
        frame = pd.DataFrame({name: [0.0] for name in feature_columns("lstm")[3:]})
        for axis in "XYZ":
            frame[f"orientation_{axis}"] = 0.0
        self.assertEqual(basic_features(frame)["acc_vs_vel"].iloc[0], 0.0)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.frame, self.labels = make_data(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_order_and_label_alignment(self):
        values, ids = load_sequences(self.path / "X_train.csv", feature_columns("gru"))
        expected = cal_euler_angle(self.frame)
        for index, series_id in enumerate(ids):
            reference = expected.loc[expected.series_id == series_id, list(feature_columns("gru"))].to_numpy()
            np.testing.assert_allclose(values[index], reference)
        loaded, targets, encoder = load_training_data("gru", self.path)
        np.testing.assert_allclose(loaded, values)
        self.assertEqual(encoder.inverse_transform(targets).tolist(), self.labels.set_index("series_id").loc[ids, "surface"].tolist())

    def test_classical_shape(self):
        values, _, _ = load_training_data("svm", self.path)
        self.assertEqual(values.shape, (16, 128 * 19))

    def test_invalid_measurements_rejected(self):
        for change, message in [
            (lambda df: pd.concat([df, df.iloc[:1]]), "重复"),
            (lambda df: df.iloc[1:], "长度"),
            (lambda df: df.assign(measurement_number=df.measurement_number + 1), "连续"),
        ]:
            with self.subTest(message=message):
                change(self.frame).to_csv(self.path / "invalid.csv", index=False)
                with self.assertRaisesRegex(ValueError, message):
                    load_sequences(self.path / "invalid.csv", feature_columns("lstm"))

    def test_mismatched_labels_rejected(self):
        self.labels.iloc[:-1].to_csv(self.path / "y_train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "series_id 不一致"):
            load_training_data("lstm", self.path)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_temporal_models_train_save_load_predict(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            make_data(path)
            for model in ("lstm", "gru", "rnn"):
                with self.subTest(model=model):
                    config = TrainConfig(model=model, data_dir=path, output_dir=path / "outputs",
                                         epochs=1, hidden_size=16, batch_size=8, device="cpu")
                    run_dir = train_temporal(config)
                    self.assertTrue((run_dir / "training.png").is_file())
                    self.assertTrue((run_dir / "history.csv").is_file())
                    metrics = json.loads((run_dir / "metrics.json").read_text())
                    self.assertEqual(metrics["epoch"], 1)
                    self.assertTrue(math.isfinite(metrics["val_loss"]))
                    result = predict(run_dir, path / "X_test.csv", device="cpu", batch_size=1)
                    self.assertEqual(set(result.series_id), {1, 3})
                    self.assertTrue(set(result.surface).issubset({"wood", "concrete"}))
                    repeated = predict(run_dir, path / "X_test.csv", device="cpu", batch_size=2)
                    pd.testing.assert_frame_equal(result, repeated)
                    with self.assertRaisesRegex(ValueError, "不一致"):
                        predict(run_dir, model="svm")

    def check_baseline(self, model):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            make_data(path)
            config = TrainConfig(model=model, data_dir=path, output_dir=path / "outputs", boost_rounds=2)
            run_dir = train_baseline(config)
            result = predict(run_dir, path / "X_test.csv")
            self.assertEqual(set(result.series_id), {1, 3})
            self.assertTrue(set(result.surface).issubset({"wood", "concrete"}))

    def test_svm_train_save_load_predict(self):
        self.check_baseline("svm")

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "未安装可选依赖 xgboost")
    def test_xgboost_train_save_load_predict(self):
        self.check_baseline("xgboost")

    def test_non_default_hidden_size_and_original_weight_keys(self):
        for kind in ("lstm", "gru", "rnn"):
            model = build_model(kind, 9, 16, 9).eval()
            self.assertIn(f"{kind}.weight_ih_l0", model.state_dict())
            self.assertEqual(tuple(model(torch.zeros(2, 128, 9)).shape), (2, 9))
        with self.assertRaises(ValueError):
            build_model("lstm", 9, 15, 9)

    def test_missing_metadata_rejected_before_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "缺少 config.json"):
                predict(temporary, model="lstm")


if __name__ == "__main__":
    unittest.main()
