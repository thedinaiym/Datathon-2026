#!/usr/bin/env bash
# One-time setup of a Vast.ai instance (template image with PyTorch + CUDA 12.x, "devel" variant for nvcc).
# Run from the repo root on the instance:  bash scripts/setup_vast.sh
# Expects videos in data/videos/. Prints peak GPU memory at the end: that tells you whether
# a cheaper 24 GB card is enough for serving.
set -euo pipefail

apt-get update && apt-get install -y --no-install-recommends ffmpeg tmux && rm -rf /var/lib/apt/lists/*

# VideoJev was tested with PyTorch 2.8 / CUDA 12.8; install it if the image has another version.
python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.8') else 1)" 2>/dev/null || \
  pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

pip install transformers==5.14.1 accelerate safetensors numpy pillow av==18.1.0 \
  flash-linear-attention==0.4.2 fla-core==0.4.2 \
  fastapi "uvicorn[standard]" python-multipart opencv-python-headless pandas openpyxl "huggingface_hub[cli]"

# Builds from source: can take 30+ minutes. Do NOT rent the expensive GPU for this step.
MAX_JOBS=${MAX_JOBS:-16} pip install flash-attn causal-conv1d --no-build-isolation

mkdir -p models
hf download ZhuYuhan/VideoJev-4B --local-dir models/VideoJev-4B   # add --revision <commit> to pin

# Smoke test with the internet switched off for Hugging Face: proves the model runs locally.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'PY'
import glob, json, time, torch
from transformers import AutoModel, AutoProcessor
d = "models/VideoJev-4B"
proc = AutoProcessor.from_pretrained(d, trust_remote_code=True, local_files_only=True)
model = AutoModel.from_pretrained(d, trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16,
                                  device_map={"": "cuda:0"}, attn_implementation="flash_attention_2")
model.prepare_for_inference()
videos = sorted(glob.glob("data/videos/*.mov") + glob.glob("data/videos/*.MOV"))
video = max(videos, key=lambda p: __import__("os").path.getsize(p))  # the longest-looking file
qs = [{"id": "g", "task": "grounding", "query": "The cashier puts money into the safe."},
      {"id": "c", "task": "choice", "query": "Is the safe open?", "options": ["A. Yes.", "B. No."]}]
t0 = time.time()
print(video)
print(json.dumps(model.predict_video(proc, video, qs, fps=2.0, max_frames=2048, context=32768), indent=1))
print(f"seconds: {time.time() - t0:.1f}   peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
PY
