#!/usr/bin/env python3
"""Уменьшает скорость обучения в сгенерированном конфиге VideoJev.

Конфиги рассчитаны на обучение с нуля на больших датасетах. Мы продолжаем с готовой VideoJev-4B
на 39 видео, поэтому шаги должны быть мягче, иначе модель забудет то, что уже умеет.

  python tools/scale_lr.py data/custom4b/configs/joint.json 0.1
"""
import json
import re
import sys

LR_KEY = re.compile(r"(^|_)(lr|learning_rate)($|_)", re.IGNORECASE)


def scale(node, factor, path=""):
    changes = []
    if isinstance(node, dict):
        for k, val in node.items():
            p = f"{path}.{k}" if path else k
            if LR_KEY.search(k) and isinstance(val, (int, float)) and not isinstance(val, bool) and val > 0:
                node[k] = val * factor
                changes.append((p, val, node[k]))
            else:
                changes += scale(val, factor, p)
    elif isinstance(node, list):
        for i, val in enumerate(node):
            changes += scale(val, factor, f"{path}[{i}]")
    return changes


def main():
    path, factor = sys.argv[1], float(sys.argv[2])
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    changes = scale(cfg, factor)
    if not changes:
        print(f"[!] В {path} не найдено параметров скорости обучения: оставлено как есть")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    for p, old, new in changes:
        print(f"  {p}: {old} -> {new}")


if __name__ == "__main__":
    main()
