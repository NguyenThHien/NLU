"""HTTP API phân loại ý định: POST /v1/predict, GET /health, GET /metrics."""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from vncorenlp import VnCoreNLP

from common import LABEL2ID, VNCORENLP_JAR
from predict import apply_thresholds, load_classifier, load_thresholds, predict_text

MAX_TEXT_CHARS = 500
MODEL_VERSION = os.environ.get("MODEL_VERSION", "local")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("nlu-intent")

PREDICTIONS = Counter("nlu_predictions_total", "Số dự đoán theo nhãn cuối", ["label"])
FALLBACKS = Counter("nlu_threshold_fallbacks_total", "Số lần dưới ngưỡng, chuyển UNKNOWN", ["model_label"])
LATENCY = Histogram("nlu_predict_seconds", "Thời gian xử lý một request /v1/predict")
CONFIDENCE = Histogram(
    "nlu_confidence", "Điểm softmax top-1",
    buckets=(0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0),
)

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["tokenizer"], state["model"] = load_classifier()
    state["thresholds"] = load_thresholds()
    logger.info("model_version=%s thresholds=%s", MODEL_VERSION, state["thresholds"])
    with VnCoreNLP(
        str(VNCORENLP_JAR),
        annotators="wseg",
        max_heap_size="-Xmx1g",
    ) as segmenter:
        state["segmenter"] = segmenter
        yield


app = FastAPI(title="Vietnamese Intent API", version=MODEL_VERSION, lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


@app.post("/v1/predict")
def predict(request: PredictRequest) -> dict:
    started = time.perf_counter()
    try:
        _, ranked_scores, was_truncated = predict_text(
            request.text, state["segmenter"], state["tokenizer"], state["model"]
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    model_label, confidence = ranked_scores[0]
    label = apply_thresholds(ranked_scores, state["thresholds"])
    elapsed = time.perf_counter() - started

    PREDICTIONS.labels(label).inc()
    CONFIDENCE.observe(confidence)
    LATENCY.observe(elapsed)
    if label != model_label:
        FALLBACKS.labels(model_label).inc()
    # Không log nguyên văn câu người dùng (có thể chứa dữ liệu cá nhân).
    logger.info("label=%s model_label=%s confidence=%.4f latency_ms=%.1f",
                label, model_label, confidence, elapsed * 1000)

    return {
        "label": label,
        "label_id": LABEL2ID[label],
        "model_label": model_label,
        "confidence": confidence,
        "scores": dict(ranked_scores),
        "truncated": was_truncated,
        "model_version": MODEL_VERSION,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
