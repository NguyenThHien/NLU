
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vncorenlp import VnCoreNLP


from common import (
    TRAINED_MODEL_DIR as MODEL_DIR, VNCORENLP_JAR,
    LABEL2ID, MAX_LENGTH, segment_text,
)


def main():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Khong tim thay model: {MODEL_DIR}")

    print("Dang nap model...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        use_fast=False,
        local_files_only=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

    model.to("cpu")
    model.eval()  # Chuyển sang chế độ dự đoán

    with VnCoreNLP(
        str(VNCORENLP_JAR),
        annotators="wseg",
        max_heap_size="-Xmx1g",
    ) as segmenter:

        print("\nNhap cau can kiem tra.")
        print("Go /exit de thoat.")

        while True:
            text = input("\nVan ban: ").strip()

            if text.lower() == "/exit":
                break

            if not text:
                print("Ban chua nhap noi dung.")
                continue

            # Chuẩn hóa và tách từ giống lúc train
            segmented_text = segment_text(segmenter, text)

            if not segmented_text.strip():
                print("Khong tach duoc tu, ban thu cau khac nhe.")
                continue

            # Giữ cùng giới hạn token như lúc train
            token_count = len(
                tokenizer(
                    segmented_text,
                    truncation=False,
                )["input_ids"]
            )

            if token_count > MAX_LENGTH:
                print("Luu y: van ban dai, model chi nhan 128 token dau.")

            inputs = tokenizer(
                segmented_text,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            # Không tính gradient vì chỉ dự đoán
            with torch.inference_mode():
                logits = model(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1)[0]

            model_id = int(probabilities.argmax().item())
            label = model.config.id2label[model_id]
            original_id = LABEL2ID[label]
            score = float(probabilities[model_id].item())

            print(f"Da tach tu: {segmented_text}")
            print(f"Nhan du doan: {label}")
            print(f"ID nhan: {original_id}")
            print(f"Diem softmax: {score:.2%}")

            print("\nDiem cua ca 5 nhan:")
            ranked_ids = torch.argsort(
                probabilities,
                descending=True,
            ).tolist()

            for index in ranked_ids:
                name = model.config.id2label[index]
                probability = float(probabilities[index].item())
                print(f"  {name}: {probability:.2%}")


if __name__ == "__main__":
    main()
