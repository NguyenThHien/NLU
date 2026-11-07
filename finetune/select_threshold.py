"""Chọn ngưỡng tin cậy trên validation, báo cáo trên test, ghi thresholds.json cạnh model.

Nhãn top-1 có điểm softmax dưới ngưỡng sẽ chuyển thành UNKNOWN.
END_SESSION được phép có ngưỡng riêng, cao hơn, vì kết thúc phiên nhầm gây hại nhiều nhất.
"""

import csv
import json

import torch
from sklearn.metrics import classification_report, f1_score

from common import LABEL2ID, MAX_LENGTH, PROCESSED_DIR, TRAINED_MODEL_DIR
from predict import FALLBACK_LABEL, THRESHOLDS_FILE, apply_thresholds, load_classifier

GRID = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998]
GUARDED_LABEL = "END_SESSION"
BUSINESS_LABELS = [name for name in LABEL2ID if name != FALLBACK_LABEL]


def load_split(name):
    with open(PROCESSED_DIR / f"{name}_clean.csv", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    return [row["text_segmented"] for row in rows], [row["label"] for row in rows]


def rank_all(texts, tokenizer, model, batch_size=32):
    ranked = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            for row in model(**inputs).logits.softmax(dim=-1).tolist():
                pairs = [(model.config.id2label[i], p) for i, p in enumerate(row)]
                ranked.append(sorted(pairs, key=lambda pair: -pair[1]))
    return ranked


def make_thresholds(base, guarded):
    thresholds = {name: base for name in BUSINESS_LABELS}
    thresholds[GUARDED_LABEL] = guarded
    return thresholds


def macro_f1(labels, ranked, thresholds):
    predicted = [apply_thresholds(scores, thresholds) for scores in ranked]
    return f1_score(labels, predicted, average="macro"), predicted


def search(labels, ranked):
    """Duyệt lưới, giữ ngưỡng nhỏ nhất đạt macro-F1 cao nhất trên validation."""
    best_f1, best = -1.0, None
    for base in GRID:
        for guarded in (value for value in GRID if value >= base):
            thresholds = make_thresholds(base, guarded)
            score, _ = macro_f1(labels, ranked, thresholds)
            if score > best_f1 + 1e-9:
                best_f1, best = score, thresholds
    return best_f1, best


def report(title, labels, predicted):
    print(f"\n== {title}")
    print(classification_report(labels, predicted, digits=4, zero_division=0))


def main():
    tokenizer, model = load_classifier()
    val_texts, val_labels = load_split("val")
    test_texts, test_labels = load_split("test")
    val_ranked = rank_all(val_texts, tokenizer, model)
    test_ranked = rank_all(test_texts, tokenizer, model)

    base_f1, _ = macro_f1(val_labels, val_ranked, {})
    best_f1, thresholds = search(val_labels, val_ranked)
    print(f"Validation macro-F1: không ngưỡng {base_f1:.4f} → có ngưỡng {best_f1:.4f}")
    print(f"Ngưỡng chọn: {thresholds}")

    _, test_plain = macro_f1(test_labels, test_ranked, {})
    _, test_thresholded = macro_f1(test_labels, test_ranked, thresholds)
    report("Test, không ngưỡng", test_labels, test_plain)
    report("Test, có ngưỡng", test_labels, test_thresholded)

    output = TRAINED_MODEL_DIR / THRESHOLDS_FILE
    output.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    print(f"Đã ghi {output}")


if __name__ == "__main__":
    main()
