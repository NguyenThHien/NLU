import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


from common import (
    BASE_MODEL_DIR as MODEL_DIR,
    PROCESSED_DIR as DATA_DIR,
    ROOT_DIR, LABEL2ID, ID2LABEL, MODEL_LABEL2ID, MODEL_ID2LABEL,
    MAX_LENGTH, SEED,
)


def compute_metrics(prediction):
    """Đánh giá kết quả bằng ID gốc 1–5."""
    logits = prediction.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    # Đổi chỉ số nội bộ về ID gốc trước khi tính metric
    predicted_labels = np.argmax(logits, axis=-1) + 1
    true_labels = prediction.label_ids + 1

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=list(ID2LABEL),
        average="macro", zero_division=0,
    )
    return {
        "accuracy": accuracy_score(
            true_labels,
            predicted_labels,
        ),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def save_reports(trainer, dataset, split, output_dir):
    """Save predictions, per-class metrics and evaluation charts."""
    result = trainer.predict(dataset, metric_key_prefix=split)
    trainer.save_metrics(split, result.metrics)
    logits = result.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predicted = np.argmax(logits, axis=-1) + 1
    actual = result.label_ids + 1
    report = classification_report(
        actual, predicted, labels=list(ID2LABEL),
        target_names=list(LABEL2ID), zero_division=0, output_dict=True,
    )
    pd.DataFrame(report).T.to_csv(output_dir / f"{split}_report.csv")
    pd.DataFrame({
        "actual_id": actual, "predicted_id": predicted,
        "actual_label": [ID2LABEL[int(x)] for x in actual],
        "predicted_label": [ID2LABEL[int(x)] for x in predicted],
    }).to_csv(output_dir / f"{split}_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        actual, predicted, labels=list(ID2LABEL),
        display_labels=list(LABEL2ID), ax=ax, cmap="Blues",
        xticks_rotation=45, colorbar=False,
    )
    ax.set_title(f"{split}: confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / f"{split}_confusion_matrix.png", dpi=160)
    plt.close(fig)

    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    scores = [result.metrics[f"{split}_{key}"] for key in keys]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"], scores)
    ax.set_ylim(0, 1.1)
    ax.bar_label(bars, labels=[f"{x:.3f}" for x in scores])
    ax.set_title(f"{split}: evaluation metrics")
    fig.tight_layout()
    fig.savefig(output_dir / f"{split}_metrics.png", dpi=160)
    plt.close(fig)
    print(f"\n{split} metrics:", dict(zip(keys, scores)), flush=True)


def save_training_charts(trainer, output_dir):
    """Save loss and validation metrics against optimizer steps."""
    history = trainer.state.log_history
    (output_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for key, label in [("loss", "Train loss"), ("eval_loss", "Validation loss")]:
        rows = [row for row in history if key in row]
        axes[0].plot([r["step"] for r in rows], [r[key] for r in rows],
                     marker="o", label=label)
    for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
        rows = [row for row in history if f"eval_{key}" in row]
        axes[1].plot([r["step"] for r in rows],
                     [r[f"eval_{key}"] for r in rows], marker="o", label=key)
    for ax in axes:
        ax.set_xlabel("Optimizer step")
        ax.legend()
        ax.grid(alpha=0.3)
    axes[0].set_title("Training / validation loss")
    axes[1].set_title("Validation metrics")
    axes[1].set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Chay thu 2 buoc voi it du lieu.",
    )
    args = parser.parse_args()

    set_seed(SEED)

    # Lưu kết quả chạy thử và huấn luyện thật riêng nhau
    run_name = "smoke_test" if args.smoke else "phobert_intent"
    output_dir = ROOT_DIR / "outputs" / run_name

    # Chỉ dùng train và validation để huấn luyện, chọn model
    data = load_dataset(
        "csv",
        data_files={
            "train": str(DATA_DIR / "train_clean.csv"),
            "validation": str(DATA_DIR / "val_clean.csv"),
            "test": str(DATA_DIR / "test_clean.csv"),
        },
    )

    # Kiểm tra dữ liệu đầu vào
    for split in data:
        required_columns = {"text_segmented", "labels", "label"}

        if not required_columns.issubset(data[split].column_names):
            raise ValueError(
                f"{split}: thieu cot text_segmented, labels hoac label."
            )

        if len(data[split]) == 0:
            raise ValueError(f"{split}: khong co du lieu.")

        for position, row in enumerate(data[split], start=1):
            if row["labels"] not in ID2LABEL:
                raise ValueError(
                    f"{split}, dong {position}: "
                    "cot labels phai chua ID tu 1 den 5."
                )

            # Kiểm tra tên nhãn khớp ID gốc
            if LABEL2ID.get(row["label"]) != row["labels"]:
                raise ValueError(
                    f"{split}, dong {position}: "
                    "ten nhan khong khop ID nhan."
                )

            if (
                not isinstance(row["text_segmented"], str)
                or not row["text_segmented"].strip()
            ):
                raise ValueError(
                    f"{split}, dong {position}: "
                    "text_segmented rong hoac khong hop le."
                )

    # Chạy thử với một phần nhỏ dữ liệu
    if args.smoke:
        for split in data:
            data[split] = data[split].shuffle(seed=SEED).select(
                range(min(16, len(data[split])))
            )

    # Nạp tokenizer đã lưu trên máy
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        use_fast=False,
        local_files_only=True,
    )

    def tokenize_batch(batch):
        encoded = tokenizer(
            batch["text_segmented"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

        # CSV giữ nhãn 1–5.
        # Chỉ đổi trong bộ nhớ sang 0–4 cho hàm loss.
        encoded["labels"] = [
            int(label_id) - 1
            for label_id in batch["labels"]
        ]

        return encoded

    # Tokenize và chỉ giữ các đầu vào cần cho model
    for split in data:
        data[split] = data[split].map(
            tokenize_batch,
            batched=True,
            remove_columns=data[split].column_names,
        )

    # Nạp PhoBERT và thêm phần phân loại 5 lớp
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR),
        num_labels=5,
        label2id=MODEL_LABEL2ID,
        id2label=MODEL_ID2LABEL,
        local_files_only=True,
    )

    # Lưu cả bảng nhãn gốc trong cấu hình model
    model.config.dataset_label2id = LABEL2ID
    model.config.dataset_id2label = ID2LABEL

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        use_cpu=True,
        num_train_epochs=3,
        max_steps=2 if args.smoke else -1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=1 if args.smoke else 50,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
        ),
        compute_metrics=compute_metrics,
    )

    print("\nBat dau huan luyen...", flush=True)
    trainer.train()

    # Lưu model tốt nhất theo validation macro-F1
    save_dir = output_dir / "final_model"
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(save_dir)

    save_training_charts(trainer, output_dir)
    save_reports(trainer, data["validation"], "validation", output_dir)
    # Test is evaluated only after training and best-model selection.
    save_reports(trainer, data["test"], "test", output_dir)

    print(f"\nDa luu model tai: {save_dir}")
    print(f"Bao cao va bieu do: {output_dir}")

    if args.smoke:
        print(
            "\nDay chi la chay thu, "
            "chua phai model da huan luyen day du."
        )


if __name__ == "__main__":
    main()
