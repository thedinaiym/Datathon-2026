#!/usr/bin/env bash
set -eo pipefail
PIP="python -m pip"
cd /workspace
[ -d VideoJev ] || git clone https://github.com/MCG-NJU/VideoJev.git
[ -d xtuner ] || git clone https://github.com/InternLM/xtuner.git
(cd xtuner && git checkout -q 76e705134521eff867b409f3b3451df1c4d8dd36)
cd VideoJev
python - <<'PY'
import pathlib, re
skip = re.compile(r"^\s*(torch|torchvision|torchaudio|triton|flash[-_]attn|causal[-_]conv1d)\b", re.I)
def expand(path, seen):
    path = pathlib.Path(path).resolve()
    if path in seen or not path.exists():
        return []
    seen.add(path)
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split(" #")[0].strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r"^(-r|--requirement)\s+(\S+)", s)
        if m:
            out += expand(path.parent / m.group(2), seen)
        elif skip.match(s):
            print("  пропускаю (уже установлено):", s)
        else:
            out.append(s)
    return out
lines = expand("requirements.txt", set())
pathlib.Path("/tmp/vj_requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("Ставлю:"); print("\n".join("  " + l for l in lines))
PY
$PIP install -r /tmp/vj_requirements.txt pytest
mkdir -p models
[ -f models/Qwen3.5-4B/config.json ] || hf download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
export XTUNER_ROOT=/workspace/xtuner
set +e; source scripts/env.sh; set -e
VIDEOJEV_TEST_MODEL=models/Qwen3.5-4B python -m pytest -q tests \
  || echo "[!] Часть тестов VideoJev не прошла: пришлите вывод выше"
echo "Training environment OK"
