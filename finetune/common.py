"""Cấu hình dùng chung cho tiền xử lý, huấn luyện và dự đoán."""

import os
import unicodedata
from pathlib import Path

# Đường dẫn tính từ thư mục dự án, không phụ thuộc nơi mở terminal.
ROOT_DIR = Path(__file__).resolve().parent.parent
# NLU_DATA_DIR: train trên bộ dữ liệu khác (vd dataset/v3) mà không đụng dataset gốc.
DATA_DIR = Path(os.environ.get("NLU_DATA_DIR", ROOT_DIR / "dataset"))
PROCESSED_DIR = DATA_DIR / "processed"
BASE_MODEL_DIR = ROOT_DIR / "model" / "phobert-base-v2"
# NLU_MODEL_DIR: chọn model cho predict/api/select_threshold (vd outputs/phobert_intent_v3/final_model).
TRAINED_MODEL_DIR = Path(
    os.environ.get("NLU_MODEL_DIR", ROOT_DIR / "outputs" / "phobert_intent" / "final_model")
).resolve()
VNCORENLP_JAR = ROOT_DIR / "model" / "vncorenlp" / "VnCoreNLP-1.2.jar"
MAX_LENGTH = 128
SEED = 42

LABEL2ID = {
    "READ_TEXT": 1,
    "DESCRIBE_SCENE": 2,
    "ANSWER_VISUAL_QUESTION": 3,
    "END_SESSION": 4,
    "UNKNOWN": 5,
}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}
MODEL_LABEL2ID = {name: value - 1 for name, value in LABEL2ID.items()}
MODEL_ID2LABEL = {value: name for name, value in MODEL_LABEL2ID.items()}


def normalize_text(text):
    """Giữ dấu và dấu câu, chỉ chuẩn hóa Unicode và khoảng trắng."""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def segment_text(segmenter, text):
    """Ghép các câu đã tách từ thành một chuỗi đầu vào cho PhoBERT."""
    sentences = segmenter.tokenize(normalize_text(text))
    words = []
    for sentence in sentences:
        words.extend(sentence)
    return " ".join(words)


# --- Slot (BIO) ---
# NLU_SLOT_MODEL_DIR: chọn model slot cho predict/api.
SLOT_MODEL_DIR = Path(
    os.environ.get("NLU_SLOT_MODEL_DIR", ROOT_DIR / "outputs" / "phobert_slots" / "final_model")
).resolve()
SLOT_TYPES = ["OBJECT", "LOCATION", "ATTRIBUTE", "PLACE"]
SLOT_LABELS = ["O"] + [f"{prefix}-{name}" for name in SLOT_TYPES for prefix in ("B", "I")]


def encode_words(words, tokenizer):
    """input_ids cho PhoBERT + vị trí subword đầu của từng từ (từ bị cắt do MAX_LENGTH thì bỏ)."""
    input_ids = [tokenizer.cls_token_id]
    first_positions = []
    for word in words:
        pieces = tokenizer.encode(word, add_special_tokens=False)
        if not pieces or len(input_ids) + len(pieces) > MAX_LENGTH - 1:
            break
        first_positions.append(len(input_ids))
        input_ids.extend(pieces)
    input_ids.append(tokenizer.sep_token_id)
    return input_ids, first_positions


def bio_spans(tags):
    """['B-X','I-X','O'] → [('X', start, end_exclusive)]. I-X lạc loài được coi là bắt đầu span mới."""
    spans = []
    for index, tag in enumerate(tags):
        prefix, _, name = tag.partition("-")
        if prefix == "I" and spans and spans[-1][0] == name and spans[-1][2] == index:
            spans[-1] = (name, spans[-1][1], index + 1)
        elif prefix in ("B", "I"):
            spans.append((name, index, index + 1))
    return spans


def expand_word_tags(words, word_tags):
    """Nhãn theo từ VnCoreNLP ('hóa_đơn') → (âm tiết, nhãn): B-X → B-X, I-X, ...; I-X → I-X, ...; O → O."""
    syllables, tags = [], []
    for word, tag in zip(words, word_tags):
        parts = word.split("_")
        inner = "O" if tag == "O" else "I-" + tag.split("-", 1)[1]
        syllables.extend(parts)
        tags.extend([tag] + [inner] * (len(parts) - 1))
    return syllables, tags


if __name__ == "__main__":
    assert bio_spans(["B-OBJECT", "I-OBJECT", "O", "I-PLACE", "B-OBJECT"]) == [
        ("OBJECT", 0, 2), ("PLACE", 3, 4), ("OBJECT", 4, 5)]
    assert expand_word_tags(["đọc", "hóa_đơn", "này"], ["O", "B-OBJECT", "O"]) == (
        ["đọc", "hóa", "đơn", "này"], ["O", "B-OBJECT", "I-OBJECT", "O"])
    print("ok")
