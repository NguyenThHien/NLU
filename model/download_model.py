from pathlib import Path

from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "vinai/phobert-base-v2"
MODEL_DIR = Path(__file__).resolve().parent
SAVE_DIR = MODEL_DIR / "phobert-base-v2"

def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
    )

    print("Đang tải model...")
    model = AutoModel.from_pretrained(MODEL_NAME)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)

    print(f"Đã tải xong model và lưu vào thư mục: {SAVE_DIR}")


if __name__ == "__main__":
    main()