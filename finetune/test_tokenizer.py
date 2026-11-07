"""Xem tokenizer biến một câu đã tách từ thành ID và attention mask."""

import pandas as pd
from transformers import AutoTokenizer

from common import BASE_MODEL_DIR, MAX_LENGTH, PROCESSED_DIR


def main():
    data = pd.read_csv(PROCESSED_DIR / "train_clean.csv", encoding="utf-8-sig")
    sample = data.iloc[0]

    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_MODEL_DIR),
        use_fast=False,
        local_files_only=True,
    )
    encoded = tokenizer(
        sample["text_segmented"],
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"][0].tolist()

    print("Câu đã tách từ:", sample["text_segmented"])
    print("Nhãn:", sample["label"])
    print("ID nhãn trong dataset:", sample["labels"])
    print("Chỉ số nhãn đưa vào model:", int(sample["labels"]) - 1)
    print("Các token:", tokenizer.convert_ids_to_tokens(token_ids))
    print("input_ids:", encoded["input_ids"])
    print("attention_mask:", encoded["attention_mask"])


if __name__ == "__main__":
    main()
