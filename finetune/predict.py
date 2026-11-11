"""Nhập câu qua terminal và dự đoán một trong năm ý định."""

import json
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer
from vncorenlp import VnCoreNLP

from common import (
    LABEL2ID,
    MAX_LENGTH,
    SLOT_MODEL_DIR,
    TRAINED_MODEL_DIR,
    VNCORENLP_JAR,
    bio_spans,
    encode_words,
    expand_word_tags,
    normalize_text,
    segment_text,
)


def load_classifier(model_dir=TRAINED_MODEL_DIR, model_class=AutoModelForSequenceClassification):
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy model: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        use_fast=False,
        local_files_only=True,
    )
    model = model_class.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    model.to("cpu")
    model.eval()
    return tokenizer, model


def load_slot_tagger(model_dir=SLOT_MODEL_DIR):
    return load_classifier(model_dir, AutoModelForTokenClassification)


def original_syllables(text, segmented):
    """Âm tiết theo câu gốc (VnCoreNLP đổi 'hóa' → 'hoá'); số âm tiết lệch thì dùng bản đã tách từ."""
    original = re.findall(r"\w+|[^\w\s]", normalize_text(text))
    segmented_syllables = segmented.replace("_", " ").split()
    return original if len(original) == len(segmented_syllables) else segmented_syllables


def extract_slots(text, segmented, tokenizer, model):
    """Câu gốc + câu đã tách từ → [{"type": "OBJECT", "value": "hóa đơn"}, ...] theo thứ tự xuất hiện."""
    words = segmented.split()
    input_ids, first_positions = encode_words(words, tokenizer)
    with torch.inference_mode():
        predicted = model(input_ids=torch.tensor([input_ids])).logits[0].argmax(-1).tolist()

    word_tags = [model.config.id2label[predicted[i]] for i in first_positions]
    _, tags = expand_word_tags(words[:len(word_tags)], word_tags)
    syllables = original_syllables(text, segmented)
    return [{"type": name, "value": " ".join(syllables[start:end])} for name, start, end in bio_spans(tags)]


def target_object(slots):
    return next((slot["value"] for slot in slots if slot["type"] == "OBJECT"), None)


THRESHOLDS_FILE = "thresholds.json"
FALLBACK_LABEL = "UNKNOWN"


def load_thresholds(model_dir=TRAINED_MODEL_DIR):
    """Ngưỡng tối thiểu theo nhãn (do select_threshold.py sinh); thiếu file → không áp ngưỡng."""
    path = model_dir / THRESHOLDS_FILE
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def apply_thresholds(ranked_scores, thresholds):
    """Trả nhãn cuối: nhãn top-1 có điểm dưới ngưỡng của nó thì chuyển thành UNKNOWN."""
    label, score = ranked_scores[0]
    if score < thresholds.get(label, 0.0):
        return FALLBACK_LABEL
    return label


def predict_batch(texts, tokenizer, model, batch_size=16):
    """Nhận câu đã tách từ, trả ID gốc 1–5 và điểm softmax."""
    predicted_ids = []
    confidence_scores = []

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            probabilities = model(**inputs).logits.softmax(dim=-1)
            scores, model_ids = probabilities.max(dim=-1)

            for model_id in model_ids.tolist():
                label = model.config.id2label[model_id]
                predicted_ids.append(LABEL2ID[label])
            confidence_scores.extend(scores.tolist())

    return predicted_ids, confidence_scores


def predict_text(text, segmenter, tokenizer, model):
    segmented = segment_text(segmenter, text)
    if not segmented.strip():
        raise ValueError("Không tách được từ, bạn thử câu khác nhé.")

    token_ids = tokenizer(segmented, truncation=False)["input_ids"]
    was_truncated = len(token_ids) > MAX_LENGTH

    inputs = tokenizer(
        segmented,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.inference_mode():
        probabilities = model(**inputs).logits.softmax(dim=-1)[0]

    ranked_scores = []
    for model_id in torch.argsort(probabilities, descending=True).tolist():
        label = model.config.id2label[model_id]
        score = float(probabilities[model_id].item())
        ranked_scores.append((label, score))

    return segmented, ranked_scores, was_truncated


def print_prediction(segmented, ranked_scores, was_truncated, thresholds):
    label = apply_thresholds(ranked_scores, thresholds)
    score = ranked_scores[0][1]
    if was_truncated:
        print(f"Lưu ý: model chỉ nhận {MAX_LENGTH} token đầu.")

    print(f"Đã tách từ: {segmented}")
    if label != ranked_scores[0][0]:
        print(f"Model chọn {ranked_scores[0][0]} nhưng dưới ngưỡng → {label}")
    print(f"Nhãn dự đoán: {label}")
    print(f"ID nhãn: {LABEL2ID[label]}")
    print(f"Điểm softmax: {score:.2%}")
    print("\nĐiểm của cả 5 nhãn:")
    for name, probability in ranked_scores:
        print(f"  {name}: {probability:.2%}")


def main():
    print("Đang nạp model...", flush=True)
    tokenizer, model = load_classifier()
    thresholds = load_thresholds()
    slot_tokenizer, slot_model = load_slot_tagger()

    with VnCoreNLP(
        str(VNCORENLP_JAR),
        annotators="wseg",
        max_heap_size="-Xmx1g",
    ) as segmenter:
        print("Nhập câu để kiểm tra. Gõ /exit để thoát.")
        while True:
            text = input("\nVăn bản: ").strip()
            if text.lower() == "/exit":
                break
            if not text:
                print("Bạn chưa nhập nội dung.")
                continue

            try:
                result = predict_text(text, segmenter, tokenizer, model)
            except ValueError as error:
                print(error)
                continue
            print_prediction(*result, thresholds)
            slots = extract_slots(text, result[0], slot_tokenizer, slot_model)
            print(f"\nSlots: {slots}")
            print(f"target_obj: {target_object(slots)}")


if __name__ == "__main__":
    main()
