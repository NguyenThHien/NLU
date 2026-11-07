"""Huấn luyện PhoBERT: đọc dữ liệu → tokenize → train → lưu và đánh giá."""

import argparse

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from common import (
    BASE_MODEL_DIR,
    ID2LABEL,
    LABEL2ID,
    MAX_LENGTH,
    MODEL_ID2LABEL,
    MODEL_LABEL2ID,
    PROCESSED_DIR,
    ROOT_DIR,
    SEED,
)
from evaluation import compute_metrics, save_reports, save_training_charts


def validate_dataset(dataset, split):
    required_columns = {"text_segmented", "labels", "label"}
    if not required_columns.issubset(dataset.column_names):
        raise ValueError(f"{split}: thiếu cột text_segmented, labels hoặc label.")
    if len(dataset) == 0:
        raise ValueError(f"{split}: không có dữ liệu.")

    for position, row in enumerate(dataset, start=1):
        location = f"{split}, dòng {position}"
        if row["labels"] not in ID2LABEL:
            raise ValueError(f"{location}: labels phải là ID từ 1 đến 5.")
        if LABEL2ID.get(row["label"]) != row["labels"]:
            raise ValueError(f"{location}: tên nhãn không khớp ID.")

        text = row["text_segmented"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{location}: text_segmented trống hoặc không hợp lệ.")


def load_training_data(smoke=False):
    files = {
        "train": str(PROCESSED_DIR / "train_clean.csv"),
        "validation": str(PROCESSED_DIR / "val_clean.csv"),
        "test": str(PROCESSED_DIR / "test_clean.csv"),
    }
    datasets = load_dataset("csv", data_files=files)

    for split, dataset in datasets.items():
        validate_dataset(dataset, split)
        if smoke:
            sample_size = min(16, len(dataset))
            datasets[split] = dataset.shuffle(seed=SEED).select(range(sample_size))

    return datasets


def tokenize_data(datasets, tokenizer):
    def tokenize_batch(batch):
        encoded = tokenizer(
            batch["text_segmented"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
        # Cross-entropy dùng chỉ số 0–4; CSV vẫn giữ ID 1–5.
        encoded["labels"] = [int(label) - 1 for label in batch["labels"]]
        return encoded

    for split, dataset in datasets.items():
        datasets[split] = dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=dataset.column_names,
        )
    return datasets


def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        str(BASE_MODEL_DIR),
        num_labels=len(LABEL2ID),
        label2id=MODEL_LABEL2ID,
        id2label=MODEL_ID2LABEL,
        local_files_only=True,
    )
    model.config.dataset_label2id = LABEL2ID
    model.config.dataset_id2label = ID2LABEL
    return model


def build_training_args(output_dir, smoke=False):
    return TrainingArguments(
        output_dir=str(output_dir),
        use_cpu=True,
        num_train_epochs=3,
        max_steps=2 if smoke else -1,
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
        logging_steps=1 if smoke else 50,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="none",
        seed=SEED,
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT cho 5 ý định.")
    parser.add_argument("--smoke", action="store_true", help="Chạy thử 2 bước.")
    args = parser.parse_args()
    set_seed(SEED)

    run_name = "smoke_test" if args.smoke else "phobert_intent"
    output_dir = ROOT_DIR / "outputs" / run_name

    # 1. Chuẩn bị dữ liệu và model.
    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_MODEL_DIR),
        use_fast=False,
        local_files_only=True,
    )
    datasets = load_training_data(smoke=args.smoke)
    datasets = tokenize_data(datasets, tokenizer)
    model = load_model()

    # 2. Train và chọn checkpoint theo validation macro-F1.
    trainer = Trainer(
        model=model,
        args=build_training_args(output_dir, smoke=args.smoke),
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # 3. Lưu model tốt nhất rồi đánh giá validation và test.
    model_dir = output_dir / "final_model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)
    save_training_charts(trainer, output_dir)
    for split in ["validation", "test"]:
        save_reports(trainer, datasets[split], split, output_dir)

    print(f"Model: {model_dir}")
    print(f"Báo cáo và biểu đồ: {output_dir}")
    if args.smoke:
        print("Đây là chạy thử, chưa phải kết quả huấn luyện đầy đủ.")


if __name__ == "__main__":
    main()
