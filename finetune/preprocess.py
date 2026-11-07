"""Tạo CSV đã chuẩn hóa, tách từ và xáo trộn cho từng tập."""

import pandas as pd
from vncorenlp import VnCoreNLP

from common import (
    DATA_DIR,
    LABEL2ID,
    PROCESSED_DIR,
    SEED,
    VNCORENLP_JAR,
    normalize_text,
    segment_text,
)


def check_segmenter_files():
    model_dir = VNCORENLP_JAR.parent / "models" / "wordsegmenter"
    required_files = [
        VNCORENLP_JAR,
        model_dir / "vi-vocab",
        model_dir / "wordsegmenter.rdr",
    ]
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy: {path}")


def read_dataset(path):
    data = pd.read_csv(path, encoding="utf-8-sig")
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError(f"{path.name}: thiếu cột text hoặc label.")
    if data.empty:
        raise ValueError(f"{path.name}: không có dữ liệu.")
    if data[["text", "label"]].isna().any().any():
        raise ValueError(f"{path.name}: có text hoặc label bị trống.")

    for position, text in enumerate(data["text"], start=1):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path.name}, dòng {position}: text không hợp lệ.")

    invalid_labels = data.loc[~data["label"].isin(LABEL2ID), "label"].unique()
    if len(invalid_labels):
        raise ValueError(f"{path.name}: nhãn không hợp lệ: {invalid_labels}")
    return data


def preprocess_dataset(data, segmenter, split):
    data = data.copy()
    data["text_clean"] = data["text"].map(normalize_text)
    data["labels"] = data["label"].map(LABEL2ID).astype(int)

    segmented_texts = []
    total = len(data)
    for position, text in enumerate(data["text_clean"], start=1):
        segmented = segment_text(segmenter, text)
        if not segmented.strip():
            raise ValueError(f"{split}, dòng {position}: kết quả tách từ rỗng.")

        segmented_texts.append(segmented)
        if position % 500 == 0 or position == total:
            print(f"{split}: đã tách từ {position}/{total}", flush=True)

    data["text_segmented"] = segmented_texts
    return data.sample(frac=1, random_state=SEED).reset_index(drop=True)


def main():
    check_segmenter_files()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Dùng chung một tiến trình Java cho cả ba tập.
    with VnCoreNLP(
        str(VNCORENLP_JAR),
        annotators="wseg",
        max_heap_size="-Xmx1g",
    ) as segmenter:
        for split in ["train", "val", "test"]:
            data = read_dataset(DATA_DIR / f"{split}.csv")
            processed = preprocess_dataset(data, segmenter, split)

            output_path = PROCESSED_DIR / f"{split}_clean.csv"
            processed.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"Đã lưu {len(processed)} câu: {output_path}")
            print(processed["label"].value_counts().to_string())

    print("Hoàn tất tiền xử lý.")


if __name__ == "__main__":
    main()
