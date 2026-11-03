"""统一的验证指标，precision / recall / F1 均使用 macro 平均。"""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(targets, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }
