# Образ сервиса: веса внутри, интернет после сборки не нужен (требование переносимости, разд. 7 ТЗ).
# Сборка:  docker build -t cashcheck .
# Запуск:  docker run --gpus all -p 8000:8000 -e API_KEYS=<ключ_жюри>,<ключ_ui> \
#            -e UI_KEY=<ключ_ui> -v $PWD/jobs:/data/jobs cashcheck
# Версии взяты из карточки VideoJev-4B (Python 3.12, PyTorch 2.8, CUDA 12.8).
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/* && python3 -m venv /opt/venv

RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir transformers==5.14.1 accelerate safetensors numpy pillow av==18.1.0 \
        flash-linear-attention==0.4.2 fla-core==0.4.2 \
        fastapi "uvicorn[standard]" python-multipart opencv-python-headless
# Сборка из исходников долгая (десятки минут). Если найдёте готовые wheel'ы под torch 2.8 + cu128, ставьте их.
RUN MAX_JOBS=4 pip install --no-cache-dir flash-attn causal-conv1d --no-build-isolation

# Каталог модели должен содержать веса И python-файлы кастомного кода (trust_remote_code).
COPY models/videojev /models/videojev
COPY service/ /app/
COPY web/ /web/
WORKDIR /app

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    MODEL_DIR=/models/videojev RULES_PATH=/app/rules.json CALIB_PATH=/app/calibration.json \
    DATA_DIR=/data/jobs WEB_INDEX=/web/index.html VJ_FPS=2.0 VJ_CONTEXT=32768
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
