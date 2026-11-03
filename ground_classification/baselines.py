"""SVM 与 XGBoost：保留原有展平特征与主要超参数。"""

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from .config import write_json
from .data import load_training_data
from .features import feature_columns
from .metrics import classification_metrics


def train_baseline(config):
    if config.model == "xgboost":
        # 可选依赖，仅运行 XGBoost 时导入。
        import xgboost as xgb

    values, targets, encoder = load_training_data(config.model, config.data_dir)
    X_train, X_val, y_train, y_val = train_test_split(
        values, targets, test_size=config.test_size, random_state=config.seed,
    )
    run_dir = config.create_run_dir(feature_columns(config.model))
    joblib.dump(encoder, run_dir / "label_encoder.pkl")
    if config.model == "svm":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        model = SVC(C=5, kernel="rbf", class_weight="balanced", gamma="scale", probability=True,
                    random_state=config.seed)
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        joblib.dump(scaler, run_dir / "scaler.pkl")
        joblib.dump(model, run_dir / "best_model.joblib")
    elif config.model == "xgboost":
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        train = xgb.DMatrix(X_train, label=y_train, weight=weights)
        validation = xgb.DMatrix(X_val, label=y_val)
        params = {
            "booster": "gbtree", "verbosity": 1, "eta": 0.05, "max_depth": 7,
            "gamma": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0,
            "objective": "multi:softmax", "num_class": len(encoder.classes_),
            "eval_metric": "mlogloss", "seed": config.seed,
        }
        model = xgb.train(
            params, train, num_boost_round=config.boost_rounds,
            evals=[(train, "train"), (validation, "validation")],
            early_stopping_rounds=20, verbose_eval=True,
        )
        predictions = model.predict(validation, iteration_range=(0, model.best_iteration + 1)).astype(int)
        model.save_model(run_dir / "best_model.json")
        write_json(run_dir / "xgboost_params.json", params)
    else:
        raise ValueError(f"不支持的传统模型：{config.model}")
    metrics = classification_metrics(y_val, np.asarray(predictions))
    write_json(run_dir / "metrics.json", metrics)
    print(f"Metrics: {metrics}\nOutput: {run_dir}")
    return run_dir
