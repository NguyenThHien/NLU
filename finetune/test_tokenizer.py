from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT_DIR / "model" / "phobert-base-v2"
DATA_FILE = ROOT_DIR / "dataset" / "processed" / "train_clean.csv"

# Nạp tokenizer đã tải về máy
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    use_fast=False,
    local_files_only=True,
)

# Đọc một câu đã tách từ
df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
text = df.loc[0, "text_segmented"]

# Chuyển câu thành đầu vào của model
encoded = tokenizer(
    text,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

print("Cau da tach tu:", text)
print("Nhan:", df.loc[0, "label"])
print("ID nhan cho model:", df.loc[0, "labels"])

print("\nCac token:")
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist()))

print("\ninput_ids:")
print(encoded["input_ids"])

print("\nattention_mask:")
print(encoded["attention_mask"])