"""Cấu hình dùng chung cho tiền xử lý, huấn luyện và dự đoán."""

import unicodedata
from pathlib import Path

# Đường dẫn tính từ thư mục dự án, không phụ thuộc nơi mở terminal.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "dataset"
PROCESSED_DIR = DATA_DIR / "processed"
BASE_MODEL_DIR = ROOT_DIR / "model" / "phobert-base-v2"
TRAINED_MODEL_DIR = ROOT_DIR / "outputs" / "phobert_intent" / "final_model"
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
