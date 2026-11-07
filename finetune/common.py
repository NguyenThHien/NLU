"""Shared paths, original label IDs and text normalization."""

from pathlib import Path
import unicodedata

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
    return " ".join(unicodedata.normalize("NFC", text).split())


def segment_text(segmenter, text):
    return " ".join(
        word
        for sentence in segmenter.tokenize(normalize_text(text))
        for word in sentence
    )
