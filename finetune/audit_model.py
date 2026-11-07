"""Kiểm tra trùng dữ liệu và khả năng dự đoán câu mới, không train lại."""

import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from common import (
    ID2LABEL,
    LABEL2ID,
    PROCESSED_DIR,
    ROOT_DIR,
    VNCORENLP_JAR,
    normalize_text,
)
from evaluation import calculate_metrics
from predict import load_classifier, predict_batch


# Nhãn kỳ vọng được gán thủ công trước khi xem dự đoán.
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


def metrics(actual_ids, predicted_ids):
    # Giữ tên cột của báo cáo audit đã có.
    scores = calculate_metrics(actual_ids, predicted_ids)
    return {
        "accuracy": scores["accuracy"],
        "precision_macro": scores["macro_precision"],
        "recall_macro": scores["macro_recall"],
        "f1_macro": scores["macro_f1"],
    }


def load_datasets():
    frames = {}
    for split in ["train", "val", "test"]:
        path = PROCESSED_DIR / f"{split}_clean.csv"
        frames[split] = pd.read_csv(path, encoding="utf-8-sig")
    return frames


def normalize_for_comparison(text):
    return normalize_text(text).casefold()


def check_duplicates(frames):
    normalized = {}
    duplicate_counts = {}
    for split, frame in frames.items():
        texts = frame["text"].map(normalize_for_comparison)
        normalized[split] = texts
        duplicate_counts[split] = int(texts.duplicated().sum())

    overlap = {}
    for first, second in [("train", "val"), ("train", "test"), ("val", "test")]:
        common_texts = set(normalized[first]) & set(normalized[second])
        overlap[f"{first}_{second}"] = len(common_texts)

    return normalized, {
        "duplicates_within_split": duplicate_counts,
        "overlap": overlap,
    }


def evaluate_datasets(frames, tokenizer, model):
    results = {}
    for split, frame in frames.items():
        predicted_ids, _ = predict_batch(
            frame["text_segmented"].tolist(), tokenizer, model,
        )
        results[split] = metrics(frame["labels"].to_numpy(), predicted_ids)
        print(f"{split}: {results[split]}", flush=True)
    return results


def compare_with_train(frames, normalized, output_dir):
    """Độ giống từ vựng không khẳng định hai câu trùng ý nghĩa."""
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    train_vectors = vectorizer.fit_transform(normalized["train"])
    test_vectors = vectorizer.transform(normalized["test"])

    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine")
    neighbors.fit(train_vectors)
    distances, indices = neighbors.kneighbors(test_vectors)
    similarities = 1 - distances[:, 0]

    nearest_rows = frames["train"].iloc[indices[:, 0]]
    report = pd.DataFrame({
        "test_text": frames["test"]["text"],
        "nearest_train_text": nearest_rows["text"].to_numpy(),
        "cosine_similarity": similarities,
    })
    report.to_csv(
        output_dir / "nearest_train.csv", index=False, encoding="utf-8-sig",
    )
    return {
        "mean": float(similarities.mean()),
        "at_least_0.9": int((similarities >= 0.9).sum()),
    }


def segment_with_java(text, temporary_dir):
    """Dùng CLI để bộ audit không cần mở HTTP server cục bộ."""
    input_path = temporary_dir / "input.txt"
    output_path = temporary_dir / "output.txt"
    input_path.write_text(text, encoding="utf-8")

    command = [
        "java", "-Xmx1g", "-Dfile.encoding=UTF-8",
        "-jar", str(VNCORENLP_JAR),
        "-fin", str(input_path),
        "-fout", str(output_path),
        "-annotators", "wseg",
    ]
    subprocess.run(
        command,
        cwd=VNCORENLP_JAR.parent,
        check=True,
        capture_output=True,
        timeout=60,
    )

    words = []
    for line in output_path.read_text(encoding="utf-8").splitlines():
        columns = line.split("\t")
        if line.strip() and len(columns) >= 2:
            words.append(columns[1])
    return " ".join(words)


def evaluate_manual_cases(train_texts, tokenizer, model, output_dir):
    rows = []
    with tempfile.TemporaryDirectory() as temporary:
        for expected_label, texts in CASES.items():
            for text in texts:
                rows.append({
                    "text": text,
                    "expected": expected_label,
                    "text_segmented": segment_with_java(text, Path(temporary)),
                    "seen_in_train": normalize_for_comparison(text) in train_texts,
                })

    texts = [row["text_segmented"] for row in rows]
    predicted_ids, confidence_scores = predict_batch(texts, tokenizer, model)
    expected_ids = []

    for row, predicted_id, score in zip(rows, predicted_ids, confidence_scores):
        row["predicted"] = ID2LABEL[predicted_id]
        row["confidence"] = score
        row["correct"] = row["expected"] == row["predicted"]
        expected_ids.append(LABEL2ID[row["expected"]])

        if not row["correct"]:
            print(
                f"Sai: {row['text']} | "
                f"{row['expected']} -> {row['predicted']}",
                flush=True,
            )

    pd.DataFrame(rows).to_csv(
        output_dir / "manual_predictions.csv", index=False, encoding="utf-8-sig",
    )
    return {
        "manual_new_sentences": metrics(expected_ids, predicted_ids),
        "manual_case_count": len(rows),
        "manual_cases_seen_in_train": sum(row["seen_in_train"] for row in rows),
    }


def main():
    output_dir = ROOT_DIR / "outputs" / "model_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Đọc dữ liệu và kiểm tra câu trùng.
    frames = load_datasets()
    normalized, summary = check_duplicates(frames)

    # 2. Đánh giá model hiện có trên ba tập.
    tokenizer, model = load_classifier()
    summary.update(evaluate_datasets(frames, tokenizer, model))

    # 3. Tìm câu test gần train và thử các câu mới.
    summary["test_train_lexical_similarity"] = compare_with_train(
        frames, normalized, output_dir,
    )
    summary.update(evaluate_manual_cases(
        set(normalized["train"]), tokenizer, model, output_dir,
    ))

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Đã lưu báo cáo: {output_dir}")


if __name__ == "__main__":
    main()
