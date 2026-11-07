"""Tính chỉ số, lưu báo cáo và vẽ biểu đồ sau huấn luyện."""

import json

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

matplotlib.use("Agg")  # Lưu ảnh được cả khi không có cửa sổ đồ họa.
import matplotlib.pyplot as plt

from common import ID2LABEL


def calculate_metrics(actual_ids, predicted_ids):
    """Hai danh sách đầu vào đều dùng ID gốc 1–5."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_ids,
        predicted_ids,
        labels=list(ID2LABEL),
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(actual_ids, predicted_ids),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def decode_predictions(prediction):
    """Đổi kết quả nội bộ của Trainer về ID nhãn trong dataset."""
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    actual_ids = prediction.label_ids + 1
    predicted_ids = np.argmax(logits, axis=-1) + 1
    return actual_ids, predicted_ids


def compute_metrics(prediction):
    actual_ids, predicted_ids = decode_predictions(prediction)
    return calculate_metrics(actual_ids, predicted_ids)


def save_figure(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_confusion_matrix(actual_ids, predicted_ids, split, output_dir):
    figure, axis = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        actual_ids,
        predicted_ids,
        labels=list(ID2LABEL),
        display_labels=list(ID2LABEL.values()),
        ax=axis,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
    )
    axis.set_title(f"{split}: confusion matrix")
    save_figure(figure, output_dir / f"{split}_confusion_matrix.png")


def plot_metrics(metrics, split, output_dir):
    names = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    scores = list(metrics.values())

    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(names, scores)
    axis.set_ylim(0, 1.1)
    axis.bar_label(bars, labels=[f"{score:.3f}" for score in scores])
    axis.set_title(f"{split}: evaluation metrics")
    save_figure(figure, output_dir / f"{split}_metrics.png")


def save_reports(trainer, dataset, split, output_dir):
    """Dự đoán một tập, lưu chỉ số, CSV và hai biểu đồ đánh giá."""
    result = trainer.predict(dataset, metric_key_prefix=split)
    trainer.save_metrics(split, result.metrics)
    actual_ids, predicted_ids = decode_predictions(result)

    report = classification_report(
        actual_ids,
        predicted_ids,
        labels=list(ID2LABEL),
        target_names=list(ID2LABEL.values()),
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).T.to_csv(output_dir / f"{split}_report.csv")

    predictions = pd.DataFrame({
        "actual_id": actual_ids,
        "predicted_id": predicted_ids,
        "actual_label": [ID2LABEL[int(label)] for label in actual_ids],
        "predicted_label": [ID2LABEL[int(label)] for label in predicted_ids],
    })
    predictions.to_csv(output_dir / f"{split}_predictions.csv", index=False)

    metrics = calculate_metrics(actual_ids, predicted_ids)
    plot_confusion_matrix(actual_ids, predicted_ids, split, output_dir)
    plot_metrics(metrics, split, output_dir)
    print(f"\n{split} metrics: {metrics}", flush=True)


def plot_history(axis, history, key, label):
    rows = [row for row in history if key in row]
    steps = [row["step"] for row in rows]
    values = [row[key] for row in rows]
    axis.plot(steps, values, marker="o", label=label)


def save_training_charts(trainer, output_dir):
    history = trainer.state.log_history
    history_path = output_dir / "training_history.json"
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    figure, (loss_axis, metrics_axis) = plt.subplots(1, 2, figsize=(13, 5))
    plot_history(loss_axis, history, "loss", "Train loss")
    plot_history(loss_axis, history, "eval_loss", "Validation loss")

    for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
        plot_history(metrics_axis, history, f"eval_{key}", key)

    loss_axis.set_title("Training / validation loss")
    metrics_axis.set_title("Validation metrics")
    metrics_axis.set_ylim(0, 1.05)
    for axis in (loss_axis, metrics_axis):
        axis.set_xlabel("Optimizer step")
        axis.legend()
        axis.grid(alpha=0.3)

    save_figure(figure, output_dir / "training_curves.png")
