FROM nvcr.io/nvidia/pytorch:25.09-py3

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/hf
ENV TRANSFORMERS_CACHE=/runpod-volume/hf
ENV TORCH_HOME=/runpod-volume/torch
ENV NEMOTRON_LANG=multi
ENV NEMOTRON_MODEL_DIR=/opt/nemotron-ocr-v2/v2_multilingual
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    ninja-build \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir hatchling editables "setuptools>=68" ninja

RUN git lfs install && \
    git clone https://huggingface.co/nvidia/nemotron-ocr-v2 /opt/nemotron-ocr-v2 && \
    cd /opt/nemotron-ocr-v2/nemotron-ocr && \
    pip install --no-build-isolation -v .

COPY handler.py /app/handler.py
COPY scripts /app/scripts

CMD ["python", "-u", "/app/handler.py"]
