
import pandas as pd
from vncorenlp import VnCoreNLP

from common import (
    DATA_DIR, PROCESSED_DIR as OUTPUT_DIR, VNCORENLP_JAR,
    SEED, LABEL2ID, normalize_text, segment_text,
)


def main():
    # Kiểm tra các file cần cho bộ tách từ
    required_files = [
        VNCORENLP_JAR,
        VNCORENLP_JAR.parent / "models" / "wordsegmenter" / "vi-vocab",
        VNCORENLP_JAR.parent
        / "models"
        / "wordsegmenter"
        / "wordsegmenter.rdr",
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy: {file_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Khởi động Java một lần, chỉ dùng chức năng tách từ.
    # Tự đóng kết nối khi thoát khối with.
    with VnCoreNLP(
        str(VNCORENLP_JAR),
        annotators="wseg",
        max_heap_size="-Xmx1g",
    ) as segmenter:

        for split in ["train", "val", "test"]:
            input_path = DATA_DIR / f"{split}.csv"
            df = pd.read_csv(
                input_path,
                encoding="utf-8-sig",
            )

            # Kiểm tra cấu trúc và dữ liệu đầu vào
            required_columns = {"text", "label"}

            if not required_columns.issubset(df.columns):
                raise ValueError(
                    f"{input_path.name}: thiếu cột text hoặc label"
                )

            if df.empty:
                raise ValueError(
                    f"{input_path.name}: không có dòng dữ liệu"
                )

            if df[["text", "label"]].isna().any().any():
                raise ValueError(
                    f"{input_path.name}: có text hoặc label bị trống"
                )

            if not df["text"].map(
                lambda value: isinstance(value, str)
            ).all():
                raise ValueError(
                    f"{input_path.name}: có text không phải chuỗi"
                )

            # Chuẩn hóa câu và lưu vào cột mới
            df["text_clean"] = df["text"].map(normalize_text)

            if df["text_clean"].eq("").any():
                raise ValueError(
                    f"{input_path.name}: có câu rỗng sau chuẩn hóa"
                )

            # Chuyển nhãn về 0–4 theo tên nhãn
            # Giữ nguyên cột label_id gốc
            df["labels"] = df["label"].map(LABEL2ID)

            if df["labels"].isna().any():
                unknown_labels = df.loc[
                    df["labels"].isna(), "label"
                ].unique()

                raise ValueError(
                    f"{input_path.name}: nhãn không hợp lệ: "
                    f"{unknown_labels}"
                )

            df["labels"] = df["labels"].astype(int)

            # Tách từ từng dòng
            segmented_texts = []
            total = len(df)

            for position, text in enumerate(
                df["text_clean"],
                start=1,
            ):
                segmented = segment_text(segmenter, text)

                if not segmented.strip():
                    raise ValueError(
                        f"{input_path.name}: kết quả tách từ rỗng "
                        f"ở dòng dữ liệu {position}"
                    )

                segmented_texts.append(segmented)

                if position % 500 == 0 or position == total:
                    print(
                        f"{split}: da tach tu {position}/{total}",
                        flush=True,
                    )

            df["text_segmented"] = segmented_texts

            # Xáo trộn từng tập, giữ câu đi cùng nhãn
            df = df.sample(
                frac=1,
                random_state=SEED,
            ).reset_index(drop=True)

            # Lưu kết quả, không ghi đè CSV gốc
            output_path = OUTPUT_DIR / f"{split}_clean.csv"

            df.to_csv(
                output_path,
                index=False,
                encoding="utf-8-sig",
            )

            print(f"\nDa luu: {output_path}")
            print(f"So cau: {len(df)}")
            print(df["label"].value_counts().to_string())

            print("\nVi du sau xu ly:")
            print(
                df[["text_clean", "text_segmented", "labels"]]
                .head(3)
                .to_string(index=False)
            )

    print("\nHoan tat tien xu ly ca ba tap.")


if __name__ == "__main__":
    main()
