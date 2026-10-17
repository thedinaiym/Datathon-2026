#!/usr/bin/env bash
# Установка на машине Vast: любой образ PyTorch от Vast, CUDA 12 или 13.
# Ничего не компилирует: flash-attn и causal-conv1d ставятся готовыми сборками под PyTorch 2.8.
# Если сборок не найдётся, работа модели (не обучение) всё равно возможна на встроенном внимании PyTorch (sdpa).
#   bash scripts/setup_vast.sh
set -eo pipefail
cd "$(dirname "$0")/.."

echo "== 1/5 системные пакеты"
apt-get update >/dev/null 2>&1 || true
apt-get install -y --no-install-recommends ffmpeg tmux >/dev/null
python -m pip --version >/dev/null 2>&1 || python -m ensurepip --upgrade
PIP="python -m pip"

echo "== 2/5 PyTorch 2.8 (CUDA 12.8)"
python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.8') and (torch.version.cuda or '').startswith('12') else 1)" 2>/dev/null \
  || $PIP install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

echo "== 3/5 библиотеки"
$PIP install transformers==5.14.1 accelerate safetensors numpy pillow av==18.1.0 \
  flash-linear-attention==0.4.2 fla-core==0.4.2 \
  fastapi "uvicorn[standard]" python-multipart opencv-python-headless pandas openpyxl "huggingface_hub[cli]"

echo "== 4/5 ускорители flash-attn и causal-conv1d (готовые сборки, без компиляции)"
PYTAG=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
TV=$(python -c "import torch; print('.'.join(torch.__version__.split('+')[0].split('.')[:2]))")
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch${TV}cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"
CC_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch${TV}cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"
export FLASH_ATTN_CUDA_ARCHS="80;86;89" MAX_JOBS=${MAX_JOBS:-16}
python -c "import flash_attn" 2>/dev/null || $PIP install "$FA_URL" \
  || timeout 25m $PIP install flash-attn --no-build-isolation \
  || echo "[!] flash-attn не установлен (для обучения он нужен, пришлите этот вывод)"
python -c "import causal_conv1d" 2>/dev/null || $PIP install "$CC_URL" \
  || timeout 15m $PIP install causal-conv1d --no-build-isolation \
  || echo "[!] causal-conv1d не установлен (пришлите этот вывод)"
python -c "import flash_attn, causal_conv1d; print('flash-attn', flash_attn.__version__, '| causal-conv1d OK')" || true

echo "== 5/5 модель и проверка"
mkdir -p models
[ -f models/VideoJev-4B/config.json ] || hf download ZhuYuhan/VideoJev-4B --local-dir models/VideoJev-4B

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'PY'
import glob, json, os, sys, time, torch
sys.path.insert(0, "service")
from analyzer import pick_attention
from transformers import AutoModel, AutoProcessor
d = "models/VideoJev-4B"
attn = pick_attention()
print("attention:", attn)
proc = AutoProcessor.from_pretrained(d, trust_remote_code=True, local_files_only=True)
model = AutoModel.from_pretrained(d, trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16,
                                  device_map={"": "cuda:0"}, attn_implementation=attn)
model.prepare_for_inference()
videos = sorted(glob.glob("data/videos/*.mov") + glob.glob("data/videos/*.MOV"))
if not videos:
    sys.exit("Нет видео в data/videos: проверка пропущена")
video = max(videos, key=os.path.getsize)
qs = [{"id": "g", "task": "grounding", "query": "The cashier puts money into the safe."},
      {"id": "c", "task": "choice", "query": "Is the safe open?", "options": ["A. Yes.", "B. No."]}]
t0 = time.time()
out = model.predict_video(proc, video, qs, fps=2.0, max_frames=2048, context=32768)
print(json.dumps(out, indent=1)[:800])
print(f"\nOK  video: {os.path.basename(video)}  seconds: {time.time() - t0:.1f}  "
      f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB  attention: {attn}")
PY
