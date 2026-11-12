"""Huấn luyện PhoBERT gán nhãn slot (BIO): OBJECT, LOCATION, ATTRIBUTE, PLACE.

Dữ liệu: dataset/{train,val,test}.jsonl (nhãn theo âm tiết). Nhãn được chuyển sang từ
VnCoreNLP (nhãn từ = nhãn âm tiết đầu) để khớp đầu vào lúc dự đoán.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    set_seed,
)
from vncorenlp import VnCoreNLP

from common import (
    BASE_MODEL_DIR,
    ROOT_DIR,
    SEED,
    SLOT_LABELS,
    SLOT_TYPES,
    VNCORENLP_JAR,
    bio_spans,
    encode_words,
)
from augment import augment
from train import build_training_args

SPLITS = {"train": "train", "validation": "val", "test": "test"}
SLOT_LABEL2ID = {label: index for index, label in enumerate(SLOT_LABELS)}


def align_to_words(syllables, syllable_tags, segmenter):
    """Tách từ câu âm tiết; trả (words, word_tags, số từ cắt ngang ranh giới slot) hoặc None nếu lệch."""
    words = [w for sentence in segmenter.tokenize(" ".join(syllables)) for w in sentence]
    if sum(len(w.split("_")) for w in words) != len(syllables):
        return None

    word_tags, crossing, position = [], 0, 0
    for word in words:
        tags = syllable_tags[position:position + len(word.split("_"))]
        position += len(tags)
        inner = "O" if tags[0] == "O" else "I-" + tags[0].split("-", 1)[1]
        crossing += any(tag != inner for tag in tags[1:])
        word_tags.append(tags[0])
    return words, word_tags, crossing


def load_split(path, segmenter, tokenizer, smoke, augment_fraction=0.0):
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if smoke:
        rows = rows[:16]
    if augment_fraction:
        rng = random.Random(SEED)
        extra = []
        for row in rows:
            if row["tags"] and rng.random() < augment_fraction:
                tokens, tags = augment(row["tokens"], row["tags"], rng)
                extra.append({"tokens": tokens, "tags": tags})
        print(f"{path.name}: thêm {len(extra)} câu tăng cường")
        rows = rows + extra

    records, skipped, crossing, total_words = [], 0, 0, 0
    for row in rows:
        aligned = align_to_words(row["tokens"], row["tags"], segmenter)
        if aligned is None:
            skipped += 1
            continue
        words, word_tags, row_crossing = aligned
        crossing += row_crossing
        total_words += len(words)

        input_ids, first_positions = encode_words(words, tokenizer)
        labels = [-100] * len(input_ids)
        for position, tag in zip(first_positions, word_tags):
            labels[position] = SLOT_LABEL2ID[tag]
        records.append({"input_ids": input_ids, "labels": labels})

    print(f"{path.name}: {len(records)} câu, bỏ {skipped} câu lệch âm tiết, "
          f"{crossing}/{total_words} từ cắt ngang ranh giới slot ({crossing / max(total_words, 1):.2%})")
    return Dataset.from_list(records)


def span_scores(gold_sequences, pred_sequences):
    """Span-level precision/recall/F1 (khớp đúng loại + biên) tổng và theo từng loại slot."""
    counts = {name: [0, 0, 0] for name in ["all"] + SLOT_TYPES}  # tp, n_pred, n_gold
    for gold_tags, pred_tags in zip(gold_sequences, pred_sequences):
        gold, pred = set(bio_spans(gold_tags)), set(bio_spans(pred_tags))
        for name in counts:
            g = gold if name == "all" else {s for s in gold if s[0] == name}
            p = pred if name == "all" else {s for s in pred if s[0] == name}
            counts[name][0] += len(g & p)
            counts[name][1] += len(p)
            counts[name][2] += len(g)

    metrics = {}
    for name, (tp, n_pred, n_gold) in counts.items():
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_gold if n_gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        prefix = "" if name == "all" else f"{name.lower()}_"
        metrics.update({f"{prefix}precision": precision, f"{prefix}recall": recall, f"{prefix}f1": f1})
    return metrics


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    gold_sequences, pred_sequences = [], []
    for pred_row, label_row in zip(predictions, eval_pred.label_ids):
        mask = label_row != -100
        gold_sequences.append([SLOT_LABELS[i] for i in label_row[mask]])
        pred_sequences.append([SLOT_LABELS[i] for i in pred_row[mask]])
    return span_scores(gold_sequences, pred_sequences)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT gán nhãn slot.")
    parser.add_argument("--smoke", action="store_true", help="Chạy thử 2 bước.")
    parser.add_argument("--run-name", default="phobert_slots", help="Thư mục con trong outputs/.")
    parser.add_argument("--augment", type=float, default=0.0,
                        help="Tỉ lệ câu train được thêm bản tăng cường (vd 0.5). Chỉ áp cho train.")
    parser.add_argument("--data-dir", default=str(ROOT_DIR / "dataset"), help="Thư mục chứa train/val/test.jsonl.")
    args = parser.parse_args()
    set_seed(SEED)

    output_dir = ROOT_DIR / "outputs" / ("smoke_test" if args.smoke else args.run_name)
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR), use_fast=False, local_files_only=True)

    with VnCoreNLP(str(VNCORENLP_JAR), annotators="wseg", max_heap_size="-Xmx1g") as segmenter:
        datasets = {split: load_split(Path(args.data_dir) / f"{name}.jsonl", segmenter, tokenizer, args.smoke,
                                   args.augment if split == "train" else 0.0)
                    for split, name in SPLITS.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        str(BASE_MODEL_DIR),
        num_labels=len(SLOT_LABELS),
        label2id=SLOT_LABEL2ID,
        id2label=dict(enumerate(SLOT_LABELS)),
        local_files_only=True,
    )
    trainer = Trainer(
        model=model,
        args=build_training_args(output_dir, smoke=args.smoke, metric="f1"),
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    model_dir = output_dir / "final_model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)

    results = {split: trainer.evaluate(datasets[split], metric_key_prefix=split) for split in ("validation", "test")}
    (output_dir / "slot_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for split, metrics in results.items():
        print(f"{split}: F1 {metrics[f'{split}_f1']:.2%} | OBJECT F1 {metrics[f'{split}_object_f1']:.2%}")
    print(f"Model: {model_dir}")


if __name__ == "__main__":
    main()
