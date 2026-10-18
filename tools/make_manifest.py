#!/usr/bin/env python3
"""labels.csv + rules.json -> манифесты VideoJev (формат docs/DATA.md репозитория) со сплитом по видео.

labels.csv (UTF-8, с заголовком): video,label,start_sec,end_sec,type
  одна строка на нарушение; для чистого видео одна строка с label=0;
  type = id правила из rules.json; end_sec необязателен (по умолчанию start_sec + 2 с).

На каждое видео и каждое правило:
  choice    "Does ...?" -> A. Yes / B. No   (есть ли нарушение этого типа)
  grounding запрос + spans                  (только если нарушение этого типа есть)

  python tools/make_manifest.py --videos_dir data/videos --labels data/labels.csv \
      --rules service/rules.json --out data/vj --augment hflip tilt rotzoom
Выход: data/vj/{train,val}.jsonl, data/vj/splits/{train,val}_videos.txt, data/vj/aug/*.mp4
"""
import argparse
import csv
import json
import os
import random
import subprocess
from collections import defaultdict
from pathlib import Path

# Только для train: имитация другого ракурса. Таймкоды при этих преобразованиях не меняются.
AUGS = {
    "hflip": "hflip",
    "tilt": "perspective=x0=W*0.08:y0=0:x1=W*0.92:y1=H*0.05:x2=0:y2=H:x3=W:y3=H*0.95",
    "rotzoom": ("rotate=8*PI/180:fillcolor=black,"
                "scale=trunc(iw*1.15/2+0.5)*2:trunc(ih*1.15/2+0.5)*2,"
                "crop=trunc(iw/1.15/2+0.5)*2:trunc(ih/1.15/2+0.5)*2"),
}


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def read_labels(path, default_dur=2.0):
    """-> {video: [{"type", "start", "end"}]}; пустой список = видео без нарушений."""
    videos = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row["video"].strip()
            videos.setdefault(v, [])
            if int(row["label"]):
                s = float(row["start_sec"])
                e = float(row["end_sec"]) if (row.get("end_sec") or "").strip() else s + default_dur
                videos[v].append({"type": row["type"].strip(), "start": s, "end": max(e, s + 0.5)})
    return videos


def split_videos(videos, val_frac, seed):
    """Стратифицированно: доля чистых и «грязных» видео в train и val одинаковая."""
    rng = random.Random(seed)
    train, val = [], []
    for has_violation in (False, True):
        group = sorted(v for v, evs in videos.items() if bool(evs) == has_violation)
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_frac)) if len(group) > 1 else 0
        val += group[:n_val]
        train += group[n_val:]
    return sorted(train), sorted(val)


def questions(key, rel_path, dur, events, rules):
    base = {"video_key": key, "video_path": rel_path, "duration": round(dur, 3)}
    present = {e["type"] for e in events}
    qs = []
    for r in rules:
        qs.append({**base, "id": f"{key}/choice/{r['id']}", "task": "choice",
                   "query": r["choice_en"], "options": ["A. Yes.", "B. No."],
                   "answer_index": 0 if r["id"] in present else 1})
        spans = [[round(e["start"], 2), round(min(e["end"], dur), 2)]
                 for e in events if e["type"] == r["id"]]
        if spans:
            qs.append({**base, "id": f"{key}/grounding/{r['id']}", "task": "grounding",
                       "query": r["grounding_en"], "spans": spans})
    return qs


def augment(src, dst, vf):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vf", vf, "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                    str(dst)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--out", default="data/vj")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--val_list", help="готовый список val-видео: сплит не меняется между прогонами")
    ap.add_argument("--augment", nargs="*", default=[], choices=sorted(AUGS))
    ap.add_argument("--repeat", type=int, default=1,
                    help="повторить обучающие видео N раз под разными video_key (аналог эпох)")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.rules, encoding="utf-8") as f:
        rules = json.load(f)["rules"]
    videos = read_labels(args.labels)
    unknown = {e["type"] for evs in videos.values() for e in evs} - {r["id"] for r in rules}
    if unknown:
        raise SystemExit(f"Типы из labels.csv отсутствуют в rules.json: {sorted(unknown)}")

    if args.val_list:
        with open(args.val_list, encoding="utf-8") as f:
            val = sorted(l.strip() for l in f if l.strip())
        train = sorted(set(videos) - set(val))
    else:
        train, val = split_videos(videos, args.val_frac, args.seed)
    (out / "splits").mkdir(exist_ok=True)
    for name, lst in (("train", train), ("val", val)):
        (out / "splits" / f"{name}_videos.txt").write_text("\n".join(lst) + "\n", encoding="utf-8")

    stats = defaultdict(int)
    for split, names in (("train", train), ("val", val)):
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for v in names:
                src = Path(args.videos_dir) / v
                dur = probe_duration(src)
                variants = [(Path(v).stem, src)]
                if split == "train":
                    for a in args.augment:
                        dst = out / "aug" / f"{Path(v).stem}__{a}.mp4"
                        augment(src, dst, AUGS[a])
                        variants.append((f"{Path(v).stem}__{a}", dst))
                    base_variants = list(variants)
                    for i in range(1, args.repeat):
                        for k, p in base_variants:
                            repeat_path = out / "repeat" / f"{k}__r{i}{p.suffix}"
                            repeat_path.parent.mkdir(parents=True, exist_ok=True)
                            if not repeat_path.exists():
                                os.link(p, repeat_path)
                            variants.append((f"{k}__r{i}", repeat_path))
                for key, path in variants:
                    rel = os.path.relpath(path.resolve(), out.resolve())  # пути относительно JSONL
                    for q in questions(key, rel, dur, videos[v], rules):
                        f.write(json.dumps(q, ensure_ascii=False) + "\n")
                        stats[f"{split}/{q['task']}"] += 1
                        if q["task"] == "choice" and q["answer_index"] == 0:
                            stats[f"{split}/choice_yes"] += 1
                stats[f"{split}/videos"] += len(variants)
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
