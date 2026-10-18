#!/usr/bin/env bash
# Дообучение VideoJev-4B на ваших видео, от установки до сравнения с исходной моделью.
# Запуск из /workspace/cashcheck:   bash scripts/finetune.sh
# Можно перезапускать после ошибки: готовые этапы пропускаются (отметки logs/stage_*.done).
# Настройки: REPEAT (повторы обучающих видео, 4), LR_SCALE (множитель скорости обучения, 0.1),
#            HEAD_FRACTION (доля видео на вспомогательный этап головы, 0.15).
set -eo pipefail
cd "$(dirname "$0")/.."
CC=$(pwd); VJ=/workspace/VideoJev
REPEAT=${REPEAT:-4}; LR_SCALE=${LR_SCALE:-1}; HEAD_FRACTION=${HEAD_FRACTION:-0.15}
mkdir -p logs
stage() { if [ -f "$CC/logs/stage_$1.done" ]; then echo "== $1: уже готово"; return 1; fi; echo; echo "== $1"; }
done_() { touch "$CC/logs/stage_$1.done"; }

if stage 01_setup; then bash scripts/setup_vast.sh 2>&1 | tee logs/01_setup.log; done_ 01_setup; fi

if stage 02_labels; then
  python tools/xlsx_to_labels.py --xlsx data/Dataset.xlsx --rules service/rules.json \
    --videos_dir data/videos --out data/labels.csv --unlabeled_as_clean
  done_ 02_labels
fi

if stage 03_training_env; then bash scripts/setup_training.sh 2>&1 | tee logs/03_training_env.log; done_ 03_training_env; fi

if stage 04_manifest; then   # 9 видео откладываются для честной проверки и в обучение не попадают
  python tools/make_manifest.py --videos_dir data/videos --labels data/labels.csv --rules service/rules.json \
    --out data/vj --augment hflip tilt rotzoom --repeat "$REPEAT"
  done_ 04_manifest
fi

if stage 05_baseline; then   # исходная VideoJev-4B на отложенных видео: с ней сравниваем
  python tools/evaluate.py --model_dir models/VideoJev-4B --rules service/rules.json --labels data/labels.csv \
    --videos_dir data/videos --split data/vj/splits/val_videos.txt --out runs/base_val 2>&1 | tee logs/05_baseline.log
  done_ 05_baseline
fi

if stage 06_init_lm; then    # старт с весов VideoJev-4B, а не с чистой Qwen3.5-4B
  python tools/videojev_init.py lm --qwen "$VJ/models/Qwen3.5-4B" --videojev models/VideoJev-4B \
    --out "$VJ/models/Qwen3.5-4B-vj" 2>&1 | tee logs/06_init_lm.log
  done_ 06_init_lm
fi

cd "$VJ"
export XTUNER_ROOT=/workspace/xtuner VIDEOJEV_BASE_4B="$VJ/models/Qwen3.5-4B-vj"
set +e; source scripts/env.sh; set -e

if stage 07_prepare; then
  rm -rf data/custom4b
  python scripts/prepare_data.py --manifest "$CC/data/vj/train.jsonl" --base "$VIDEOJEV_BASE_4B" \
    --out data/custom4b --world-size 1 --head-fraction "$HEAD_FRACTION" 2>&1 | tee "$CC/logs/07_prepare.log"
  python "$CC/tools/scale_lr.py" data/custom4b/configs/joint.json "$LR_SCALE" | tee -a "$CC/logs/07_prepare.log"
  done_ 07_prepare
fi

if stage 08_head; then       # нужен как шаблон формата головы
  rm -rf runs/custom4b/head_only
  torchrun --standalone --nproc-per-node=1 scripts/train.py \
    --config data/custom4b/configs/head_only.json --out runs/custom4b/head_only 2>&1 | tee "$CC/logs/08_head.log"
  done_ 08_head
fi

if stage 09_init_head; then  # голова из VideoJev-4B вместо головы, обученной с нуля
  python "$CC/tools/videojev_init.py" head --template runs/custom4b/head_only/head.pt \
    --videojev "$CC/models/VideoJev-4B" --out runs/custom4b/head_from_videojev.pt 2>&1 | tee "$CC/logs/09_init_head.log"
  done_ 09_init_head
fi

if stage 10_joint; then
  rm -rf runs/custom4b/joint
  torchrun --standalone --nproc-per-node=1 scripts/train.py \
    --config data/custom4b/configs/joint.json --out runs/custom4b/joint \
    --init-head runs/custom4b/head_from_videojev.pt 2>&1 | tee "$CC/logs/10_joint.log"
  done_ 10_joint
fi

if stage 11_export; then     # папка для сервиса: веса + код модели и файлы процессора
  CKPT="$VJ/runs/custom4b/joint/hf_checkpoint"
  [ -f "$CKPT/config.json" ] || { echo "Нет $CKPT/config.json, см. logs/10_joint.log"; exit 1; }
  rm -rf "$CC/models/VideoJev-4B-ft"; cp -r "$CKPT" "$CC/models/VideoJev-4B-ft"
  for f in "$CC"/models/VideoJev-4B/*.py "$CC"/models/VideoJev-4B/*token* \
           "$CC"/models/VideoJev-4B/*processor* "$CC"/models/VideoJev-4B/chat_template*; do
    [ -e "$f" ] || continue
    [ -e "$CC/models/VideoJev-4B-ft/$(basename "$f")" ] || cp "$f" "$CC/models/VideoJev-4B-ft/"
  done
  du -sh "$CC/models/VideoJev-4B-ft"
  done_ 11_export
fi

cd "$CC"
if stage 12_eval; then
  python tools/evaluate.py --model_dir models/VideoJev-4B-ft --rules service/rules.json --labels data/labels.csv \
    --videos_dir data/videos --split data/vj/splits/val_videos.txt --out runs/ft_val 2>&1 | tee logs/12_eval.log
  done_ 12_eval
fi

echo
echo "================ СРАВНЕНИЕ НА ОТЛОЖЕННЫХ ВИДЕО ================"
echo "Исходная VideoJev-4B (лучшая строка):"; grep -A2 "alpha mode" logs/05_baseline.log | sed -n 2p
echo "Дообученная (лучшая строка):";         grep -A2 "alpha mode" logs/12_eval.log | sed -n 2p
echo "Столбцы: alpha mode thr TP FN FP TN type tc_med tc_max. Важнее всего FN, затем FP."
