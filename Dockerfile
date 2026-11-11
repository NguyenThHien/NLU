# syntax=docker/dockerfile:1
FROM python:3.11-slim

# VnCoreNLP chạy bằng Java.
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# ponytail: torch không pin version (giống README); pin khi cần build tái lập.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        transformers==5.22.2 \
        huggingface_hub==2.4.0 \
        vncorenlp==1.0.3 \
        fastapi==0.115.6 \
        uvicorn==0.34.0 \
        prometheus-client==0.21.1

WORKDIR /app
COPY model/vncorenlp model/vncorenlp

# Model lấy từ Hugging Face theo tag/commit; token chỉ dùng lúc build, không lưu trong image.
ARG HF_REPO=Hienne/phobert-vi-intent
ARG HF_REVISION=v1.0
ARG HF_SLOT_REPO=Hienne/phobert-vi-slots
ARG HF_SLOT_REVISION=v1.0
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token)" python -c "\
from huggingface_hub import snapshot_download; \
skip = ['README.md', '.gitattributes']; \
snapshot_download('${HF_REPO}', revision='${HF_REVISION}', \
local_dir='outputs/phobert_intent/final_model', ignore_patterns=skip); \
snapshot_download('${HF_SLOT_REPO}', revision='${HF_SLOT_REVISION}', \
local_dir='outputs/phobert_slots/final_model', ignore_patterns=skip)"

COPY finetune/common.py finetune/predict.py finetune/api.py finetune/

RUN useradd --system app
USER app
ENV HF_HOME=/tmp/hf \
    MODEL_VERSION=${HF_REVISION}+slots-${HF_SLOT_REVISION}
WORKDIR /app/finetune
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
