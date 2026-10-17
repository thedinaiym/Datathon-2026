#!/usr/bin/env bash
# Окружение для обучения VideoJev (запускать после scripts/setup_vast.sh).
# Ставит репозиторий VideoJev, XTuner нужного коммита и базовую Qwen3.5-4B.
# torch, flash-attn и causal-conv1d НЕ переустанавливаются: они уже стоят готовыми сборками.
set -eo pipefail
PIP="python -m pip"
cd /workspace
[ -d VideoJev ] || git clone https://github.com/MCG-NJU/VideoJev.git
[ -d xtuner ] || git clone https://github.com/InternLM/xtuner.git
(cd xtuner && git checkout -q 76e705134521eff867b409f3b3451df1c4d8dd36)   # коммит из README VideoJev

cd VideoJev
grep -viE '^\s*(torch|torchvision|torchaudio|triton|flash[-_]attn|causal[-_]conv1d)\b' requirements.txt > /tmp/vj_requirements.txt || true
echo "Ставлю из requirements.txt (без torch/flash-attn/causal-conv1d):"; cat /tmp/vj_requirements.txt
$PIP install -r /tmp/vj_requirements.txt pytest
mkdir -p models
[ -f models/Qwen3.5-4B/config.json ] || hf download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B

export XTUNER_ROOT=/workspace/xtuner
set +e; source scripts/env.sh; set -e
VIDEOJEV_TEST_MODEL=models/Qwen3.5-4B python -m pytest -q tests \
  || echo "[!] Часть тестов VideoJev не прошла: пришлите вывод выше"
echo "Training environment OK"
