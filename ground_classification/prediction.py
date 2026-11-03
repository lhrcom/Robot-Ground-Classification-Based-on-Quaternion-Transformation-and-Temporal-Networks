"""使用训练目录内的配置、预处理器和最佳权重预测。"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import TEST_CSV
from .data import load_sequences, scale_sequences
from .features import feature_columns


def predict(run_dir, input_csv=TEST_CSV, output_csv=None, model=None, device="auto", batch_size=128):
    if batch_size < 1:
        raise ValueError("batch_size 必须为正数")
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if model is not None and model != config["model"]:
            raise ValueError("指定模型与训练目录中的 config.json 不一致")
        model = config["model"]
    else:
        raise ValueError(
            "训练目录缺少 config.json，无法确认模型参数和特征顺序。"
            "仓库根目录的历史权重为 19 维输入，与当前时序代码的 9 维输入不同；"
            "请使用重新训练生成的完整实验目录，或先补齐历史训练配置。"
        )
    columns = config.get("feature_columns", feature_columns(model))
    values, series_ids = load_sequences(input_csv, columns, config.get("sequence_length", 128))
    encoder = joblib.load(run_dir / "label_encoder.pkl")

    if model in ("svm", "xgboost"):
        values = values.reshape(len(values), -1)
        if model == "svm":
            scaler = joblib.load(run_dir / "scaler.pkl")
            estimator = joblib.load(run_dir / "best_model.joblib")
            predictions = estimator.predict(scaler.transform(values)).astype(int)
        else:
            import xgboost as xgb

            estimator = xgb.Booster()
            estimator.load_model(run_dir / "best_model.json")
            predictions = estimator.predict(
                xgb.DMatrix(values), iteration_range=(0, estimator.best_iteration + 1),
            ).astype(int)
    else:
        import torch
        from .models import build_model, select_device

        scaler = joblib.load(run_dir / "scaler.pkl")
        values = scale_sequences(scaler, values)
        target_device = select_device(device)
        estimator = build_model(model, len(columns), config["hidden_size"], len(encoder.classes_))
        state = torch.load(run_dir / "best_model.pth", map_location="cpu", weights_only=True)
        estimator.load_state_dict(state)
        estimator.to(target_device).eval()
        batches = []
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                inputs = torch.as_tensor(values[start:start + batch_size], dtype=torch.float32, device=target_device)
                batches.append(estimator(inputs).argmax(dim=1).cpu().numpy())
        predictions = np.concatenate(batches)

    result = pd.DataFrame({"series_id": series_ids, "surface": encoder.inverse_transform(predictions)})
    output_csv = Path(output_csv) if output_csv is not None else run_dir / "submission.csv"
    # 避免误将预测结果写回原始输入文件。
    if output_csv.resolve() == Path(input_csv).resolve():
        raise ValueError("预测输出路径不能与输入 CSV 相同")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"预测完成：{len(result)} 个序列 → {output_csv}")
    return result
