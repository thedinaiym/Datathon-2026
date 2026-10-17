#!/usr/bin/env bash
# Дообучение VideoJev-4B на ваших видео, 1 GPU. Запуск из /workspace/cashcheck после setup_training.sh:
#   REPEAT=4 bash scripts/train_videojev.sh
# Результат: models/VideoJev-4B-ft (готовая папка для сервиса).
set -euo pipefail
cd "$(dirname "$0")/.."
CC=$(pwd); VJ=/workspace/VideoJev
REPEAT=${REPEAT:-4}
mkdir -p logs

# 1. Манифест. Сплит фиксируется при первом запуске (data/vj/splits), дальше переиспользуется.
VAL_ARG=""
[ -f data/vj/splits/val_videos.txt ] && VAL_ARG="--val_list data/vj/splits/val_videos.txt"
python tools/make_manifest.py --videos_dir data/videos --labels data/labels.csv --rules service/rules.json \
  --out data/vj --augment hflip tilt rotzoom --repeat "$REPEAT" $VAL_ARG

# 2. Подготовка данных и два этапа обучения (команды из docs/DATA.md VideoJev, world-size 1).
cd "$VJ"
export XTUNER_ROOT=/workspace/xtuner
set +u; source scripts/env.sh; set -u
rm -rf data/custom4b runs/custom4b
python scripts/prepare_data.py --manifest "$CC/data/vj/train.jsonl" --base models/Qwen3.5-4B \
  --out data/custom4b --world-size 1 2>&1 | tee "$CC/logs/prepare.log"
torchrun --standalone --nproc-per-node=1 scripts/train.py \
  --config data/custom4b/configs/head_only.json --out runs/custom4b/head_only 2>&1 | tee "$CC/logs/train_head.log"
torchrun --standalone --nproc-per-node=1 scripts/train.py \
  --config data/custom4b/configs/joint.json --out runs/custom4b/joint \
  --init-head runs/custom4b/head_only/head.pt 2>&1 | tee "$CC/logs/train_joint.log"

# 3. Готовая папка для сервиса: веса + код модели и файлы процессора (нужны для trust_remote_code).
CKPT=$VJ/runs/custom4b/joint/hf_checkpoint
[ -f "$CKPT/config.json" ] || { echo "Нет HF-чекпойнта в $CKPT, см. logs/train_joint.log"; exit 1; }
for f in "$CC"/models/VideoJev-4B/*.py "$CC"/models/VideoJev-4B/*token* \
         "$CC"/models/VideoJev-4B/*processor* "$CC"/models/VideoJev-4B/chat_template*; do
  [ -e "$f" ] || continue
  [ -e "$CKPT/$(basename "$f")" ] || cp "$f" "$CKPT/"
done
rm -rf "$CC/models/VideoJev-4B-ft"
cp -r "$CKPT" "$CC/models/VideoJev-4B-ft"
du -sh "$CC/models/VideoJev-4B-ft"
echo "Готово: $CC/models/VideoJev-4B-ft"
