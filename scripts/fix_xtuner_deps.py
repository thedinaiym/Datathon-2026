import re, subprocess, sys
VJ = "/workspace/VideoJev"
PREFIX = f"cd {VJ} && export XTUNER_ROOT=/workspace/xtuner && source scripts/env.sh >/dev/null 2>&1; "
CHECKS = [
    "python -c 'import xtuner.v1, xtuner.v1.datasets.collator, xtuner.v1.datasets.packing, xtuner.v1.datasets.sampler'",
    "python -c 'import timejev.packing, timejev.timestamp_option_data, timejev.skip_full_safety'",
    "python scripts/prepare_data.py --help",
    "python scripts/train.py --help",
]
PIP_NAME = {"cv2": "opencv-python-headless", "yaml": "pyyaml", "PIL": "pillow", "sklearn": "scikit-learn",
            "skimage": "scikit-image", "attr": "attrs", "google": "protobuf", "dateutil": "python-dateutil"}
NEVER = {"torch", "torchvision", "triton", "flash_attn", "causal_conv1d", "xtuner", "timejev", "videojev_hf"}
installed = []
for _ in range(30):
    for check in CHECKS:
        r = subprocess.run(["bash", "-c", PREFIX + check], capture_output=True, text=True)
        if r.returncode == 0:
            continue
        m = re.search(r"No module named '([^']+)'", r.stderr)
        mod = m.group(1).split(".")[0] if m else None
        if mod is None or mod in NEVER:
            print(f"\nПроверка не прошла: {check}\n"); print("\n".join(r.stderr.strip().splitlines()[-25:]))
            sys.exit("Пришлите этот вывод.")
        pkg = PIP_NAME.get(mod, mod.replace("_", "-"))
        print(f"Не хватает модуля {mod}: ставлю {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg]); installed.append(pkg)
        break
    else:
        print("\nВсе импорты XTuner и VideoJev работают.", "Доставлено:", ", ".join(installed)); sys.exit(0)
sys.exit("Слишком много недостающих модулей. Пришлите этот вывод.")
