#!/usr/bin/env bash
# Окружение для ОБУЧЕНИЯ VideoJev на A100 (запускать ПОСЛЕ scripts/setup_vast.sh).
# Ставит репозиторий VideoJev, XTuner нужного коммита и базовую модель Qwen3.5-4B, затем гоняет тесты репозитория.
#   bash scripts/setup_training.sh
set -euo pipefail
cd /workspace
[ -d VideoJev ] || git clone https://github.com/MCG-NJU/VideoJev.git
[ -d xtuner ] || git clone https://github.com/InternLM/xtuner.git
(cd xtuner && git checkout 76e705134521eff867b409f3b3451df1c4d8dd36)   # коммит из README VideoJev

cd VideoJev
pip install -r requirements.txt pytest
mkdir -p models
[ -f models/Qwen3.5-4B/config.json ] || hf download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B

export XTUNER_ROOT=/workspace/xtuner
set +u; source scripts/env.sh; set -u
VIDEOJEV_TEST_MODEL=models/Qwen3.5-4B python -m pytest -q tests
echo
echo "Training environment OK"
