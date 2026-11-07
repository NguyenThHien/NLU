"""Evaluate memorization risk without training or modifying the dataset."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import (
    ROOT_DIR, PROCESSED_DIR, TRAINED_MODEL_DIR, VNCORENLP_JAR,
    MAX_LENGTH, LABEL2ID, ID2LABEL, normalize_text,
)

# Manually assigned expectations for new, single-intent requests.
CASES = {
    "READ_TEXT": [
        "hãy đọc nguyên văn những chữ in trên vỏ hộp thuốc này",
        "tôi muốn nghe nội dung tấm biển trước mặt",
        "giúp tôi đọc lại các dòng trên phiếu khám bệnh",
        "đọc nội dung hóa đơn thành tiếng cho mình",
        "đừng mô tả cảnh nữa, hãy đọc chữ trên màn hình",
    ],
    "DESCRIBE_SCENE": [
        "kể cho tôi nghe khung cảnh trong bức ảnh này",
        "xung quanh chỗ tôi đứng đang có những gì",
        "hãy cho tôi hình dung tổng thể căn phòng trước mặt",
        "mô tả toàn cảnh sân ga giúp tôi",
        "không cần đọc chữ, chỉ mô tả khung cảnh thôi",
    ],
    "ANSWER_VISUAL_QUESTION": [
        "trong ảnh có mấy chiếc xe đạp",
        "người đứng cạnh cửa đang mặc áo màu gì",
        "chiếc cốc nằm bên trái hay bên phải cái đĩa",
        "con mèo trong hình có đang nằm ngủ không",
        "đừng đọc cả hóa đơn, chỉ cho tôi biết tổng tiền là bao nhiêu",
    ],
    "END_SESSION": [
        "tôi không cần hỗ trợ nữa, kết thúc tại đây nhé",
        "đóng phiên trợ giúp hiện tại giúp tôi",
        "cảm ơn bạn, chúng ta dừng cuộc trò chuyện ở đây",
        "thoát khỏi phiên này đi",
        "đừng đọc tiếp nữa, kết thúc phiên giúp tôi",
    ],
    "UNKNOWN": [
        "giải phương trình hai x cộng ba bằng bảy",
        "làm sao để luộc trứng lòng đào",
        "hẹn giờ báo thức lúc sáu giờ sáng mai",
        "hãy sáng tác một bài thơ về mùa thu",
        "từ kết thúc trong tiếng Anh viết thế nào",
    ],
}


def metrics(actual, predicted):
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=list(ID2LABEL), average="macro", zero_division=0,
    )
    return dict(accuracy=accuracy_score(actual, predicted),
                precision_macro=precision, recall_macro=recall, f1_macro=f1)


def main():
    output_dir = ROOT_DIR / "outputs" / "model_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        split: pd.read_csv(PROCESSED_DIR / f"{split}_clean.csv")
        for split in ["train", "val", "test"]
    }
    normalized = {
        split: frame["text"].map(lambda x: normalize_text(x).casefold())
        for split, frame in frames.items()
    }
    summary = {"duplicates_within_split": {
        split: int(texts.duplicated().sum()) for split, texts in normalized.items()
    }, "overlap": {}}
    for first, second in [("train", "val"), ("train", "test"), ("val", "test")]:
        summary["overlap"][f"{first}_{second}"] = len(
            set(normalized[first]) & set(normalized[second])
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(TRAINED_MODEL_DIR), use_fast=False, local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(TRAINED_MODEL_DIR), local_files_only=True,
    ).eval()

    def predict(texts):
        predictions, confidence = [], []
        with torch.inference_mode():
            for start in range(0, len(texts), 16):
                inputs = tokenizer(texts[start:start + 16], padding=True,
                                   truncation=True, max_length=MAX_LENGTH,
                                   return_tensors="pt")
                probabilities = model(**inputs).logits.softmax(dim=-1)
                scores, indices = probabilities.max(dim=-1)
                predictions.extend((indices + 1).tolist())
                confidence.extend(scores.tolist())
        return np.array(predictions), confidence

    for split, frame in frames.items():
        predicted, _ = predict(frame["text_segmented"].tolist())
        summary[split] = metrics(frame["labels"].to_numpy(), predicted)
        print(split, summary[split], flush=True)

    # Lexical similarity is a proxy, not proof of semantic/template leakage.
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    train_vectors = vectorizer.fit_transform(normalized["train"])
    test_vectors = vectorizer.transform(normalized["test"])
    distances, indices = NearestNeighbors(n_neighbors=1, metric="cosine").fit(
        train_vectors
    ).kneighbors(test_vectors)
    similarities = 1 - distances[:, 0]
    summary["test_train_lexical_similarity"] = {
        "mean": float(similarities.mean()),
        "at_least_0.9": int((similarities >= 0.9).sum()),
    }
    pd.DataFrame({
        "test_text": frames["test"]["text"],
        "nearest_train_text": frames["train"].iloc[indices[:, 0]]["text"].to_numpy(),
        "cosine_similarity": similarities,
    }).to_csv(output_dir / "nearest_train.csv", index=False, encoding="utf-8-sig")

    rows = []
    # Use the already verified Java CLI, avoiding a local HTTP server.
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        for label, texts in CASES.items():
            for text in texts:
                source, target = folder / "input.txt", folder / "output.txt"
                source.write_text(text, encoding="utf-8")
                subprocess.run([
                    "java", "-Xmx1g", "-Dfile.encoding=UTF-8", "-jar",
                    str(VNCORENLP_JAR), "-fin", str(source), "-fout", str(target),
                    "-annotators", "wseg",
                ], cwd=VNCORENLP_JAR.parent, check=True, capture_output=True,
                   timeout=60)
                words = [line.split("\t")[1] for line in
                         target.read_text(encoding="utf-8").splitlines()
                         if line.strip() and len(line.split("\t")) >= 2]
                rows.append({
                    "text": text, "expected": label,
                    "text_segmented": " ".join(words),
                    "seen_in_train": normalize_text(text).casefold()
                    in set(normalized["train"]),
                })
    predicted, scores = predict([row["text_segmented"] for row in rows])
    for row, predicted_id, score in zip(rows, predicted, scores):
        row.update(predicted=ID2LABEL[int(predicted_id)], confidence=score)
        row["correct"] = row["expected"] == row["predicted"]
    summary["manual_new_sentences"] = metrics(
        [LABEL2ID[row["expected"]] for row in rows], predicted,
    )
    summary["manual_case_count"] = len(rows)
    summary["manual_cases_seen_in_train"] = sum(row["seen_in_train"] for row in rows)
    pd.DataFrame(rows).to_csv(output_dir / "manual_predictions.csv", index=False,
                             encoding="utf-8-sig")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    for row in rows:
        if not row["correct"]:
            print("MISS:", row["text"], row["expected"], "->", row["predicted"], flush=True)


if __name__ == "__main__":
    main()
