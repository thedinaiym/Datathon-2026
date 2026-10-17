#!/usr/bin/env bash
# Starts the API (web UI at /) and a public HTTPS tunnel, each in its own tmux session.
#   bash scripts/run_service.sh              # real model
#   ANALYZER=stub bash scripts/run_service.sh  # without the model, for a quick check
#   MODEL_DIR=/path/to/model bash scripts/run_service.sh  # e.g. the fine-tuned model
# Keys are created once in .env and reused: the jury key must never change. Copy .env to your laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
mkdir -p logs jobs

if [ ! -f .env ]; then
  python - > .env <<'PY'
import secrets
print(f"JURY_KEY={secrets.token_urlsafe(24)}")
print(f"UI_KEY={secrets.token_urlsafe(24)}")
PY
  echo "Created .env with new keys."
fi
set -a; source .env; set +a
MODEL_DIR=${MODEL_DIR:-$ROOT/models/VideoJev-4B}
[ "${ANALYZER:-videojev}" = "stub" ] || [ -f "$MODEL_DIR/config.json" ] || { echo "No model in $MODEL_DIR"; exit 1; }
echo "Model: $MODEL_DIR"

if ! command -v cloudflared >/dev/null; then
  curl -fsSL -o /usr/local/bin/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /usr/local/bin/cloudflared
fi

tmux kill-session -t api 2>/dev/null || true
tmux new -d -s api "cd $ROOT/service && \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ANALYZER=${ANALYZER:-videojev} \
  API_KEYS=$JURY_KEY,$UI_KEY UI_KEY=$UI_KEY \
  MODEL_DIR=$MODEL_DIR CALIB_PATH=$ROOT/service/calibration.json DATA_DIR=$ROOT/jobs \
  uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 2>&1 | tee -a $ROOT/logs/api.log"

echo "Loading the model (up to a few minutes)..."
for _ in $(seq 1 120); do curl -fs localhost:8000/healthz >/dev/null 2>&1 && break; sleep 5; done
curl -fs localhost:8000/healthz || { echo; echo "API did not start. See logs/api.log"; exit 1; }
echo

# The tunnel is started once and left running: its address changes only if it restarts.
if ! tmux has-session -t tunnel 2>/dev/null; then
  : > logs/tunnel.log
  tmux new -d -s tunnel "cloudflared tunnel --no-autoupdate --url http://localhost:8000 2>&1 | tee -a $ROOT/logs/tunnel.log"
fi
URL=""
for _ in $(seq 1 30); do
  URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' logs/tunnel.log | tail -1 || true)
  [ -n "$URL" ] && break
  sleep 2
done
[ -n "$URL" ] || { echo "Tunnel did not start. See logs/tunnel.log"; exit 1; }
echo "$URL" > logs/public_url.txt

cat <<EOF

  UI (for people):  $URL/
  API (for jury):   $URL/api/v1/analyze   and   $URL/api/v1/jobs/<job_id>
  Jury key:         $JURY_KEY

  Logs: tmux attach -t api   (detach: Ctrl+B, then D)
EOF
