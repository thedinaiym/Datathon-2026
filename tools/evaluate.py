#!/usr/bin/env python3
"""Оценка на отложенных видео + калибровка порогов -> calibration.json для сервиса.

Модель прогоняется по каждому видео один раз, сырые ответы кэшируются в <out>/raw/.
Перебор порогов идёт по кэшу, повторный запуск с --from_cache работает без GPU.
Метрики считаются так же, как у жюри: вердикт по всему видео, тип, ошибка таймкода.

  python tools/evaluate.py --model_dir models/VideoJev-4B --rules service/rules.json \
      --labels data/labels.csv --videos_dir data/videos \
      --split data/vj/splits/val_videos.txt --out runs/eval_zeroshot
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "service"))
sys.path.insert(0, str(ROOT))
from analyzer import DEFAULT_CALIB, decide, load_rules, probe_duration  # noqa: E402
from make_manifest import read_labels  # noqa: E402


def score_config(cache, labels, rules, calib):
    s = {"TP": 0, "FN": 0, "FP": 0, "TN": 0, "type_exact": 0}
    errs = []
    for v, (dur, raw) in cache.items():
        found, events = decide(raw, rules, dur, calib), labels[v]
        pred, truth = bool(found), bool(events)
        s[("T" if pred == truth else "F") + ("P" if pred else "N")] += 1
        if pred and truth:
            s["type_exact"] += {f["rule"]["id"] for f in found} == {e["type"] for e in events}
            for f in found:
                same = [e for e in events if e["type"] == f["rule"]["id"]] or events
                errs.append(min(abs(f["timecode_sec"] - e["start"]) for e in same))
    s["tc_median"] = round(statistics.median(errs), 2) if errs else None
    s["tc_max"] = round(max(errs), 2) if errs else None
    return s


def per_video_report(cache, labels, rules, calib):
    """What went wrong on which video: this is what you read to fix the queries."""
    print("\nPer video with the chosen thresholds (truth -> prediction):")
    for v, (dur, raw) in sorted(cache.items()):
        found, events = decide(raw, rules, dur, calib), labels[v]
        truth = sorted({e["type"] for e in events}) or ["clean"]
        pred = [f"{f['rule']['id']}@{f['timecode_sec']:.0f}s({f['score']:.2f})" for f in found] or ["clean"]
        if events and not found:
            mark = "MISS  "
        elif found and not events:
            mark = "FALSE "
        elif events and {f["rule"]["id"] for f in found} != set(truth):
            mark = "type? "
        else:
            mark = "ok    "
        start = f" @{events[0]['start']:.0f}s" if events else ""
        print(f"  {mark}{v}: {', '.join(truth)}{start} -> {', '.join(pred)}")


def rank_key(stats):
    # приоритет из FAQ 16: без пропусков -> минимум ложных -> точный тип -> точный таймкод
    tc = stats["tc_median"] if stats["tc_median"] is not None else 1e9
    return (stats["FN"], stats["FP"], -stats["type_exact"], tc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir")
    ap.add_argument("--rules", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--from_cache", action="store_true")
    args = ap.parse_args()

    rules, labels = load_rules(args.rules), read_labels(args.labels)
    with open(args.split, encoding="utf-8") as f:
        names = [l.strip() for l in f if l.strip()]
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    cache, analyzer = {}, None
    for v in names:
        p = out / "raw" / f"{Path(v).stem}.json"
        if not p.exists():
            if args.from_cache:
                raise SystemExit(f"нет кэша для {v}")
            if analyzer is None:
                from analyzer import VideoJevAnalyzer
                analyzer = VideoJevAnalyzer(args.model_dir, args.rules, fps=args.fps, context=args.context)
            path = Path(args.videos_dir) / v
            t0 = time.time()
            raw = analyzer.raw(path)
            p.write_text(json.dumps({"duration": probe_duration(path), "raw": raw,
                                     "seconds": round(time.time() - t0, 1)}, ensure_ascii=False))
        d = json.loads(p.read_text())
        cache[v] = (d["duration"], d["raw"])
        print(f"{v}: длительность {d['duration']:.0f} с, модель {d.get('seconds', 0):.1f} с")

    results = []
    for alpha in (1.0, 0.75, 0.5, 0.25, 0.0):        # вес choice против максимума grounding
        for mode in ("peak", "start"):               # таймкод: пик вероятности или начало интервала
            for k in range(1, 20):
                calib = {**DEFAULT_CALIB, "alpha": alpha, "timecode_mode": mode, "threshold": round(0.05 * k, 2)}
                results.append((calib, score_config(cache, labels, rules, calib)))
    results.sort(key=lambda r: rank_key(r[1]))

    print("\nalpha mode   thr    TP  FN  FP  TN  type  tc_med  tc_max")
    for c, s in results[:15]:
        print(f"{c['alpha']:5.2f} {c['timecode_mode']:5s} {c['threshold']:5.2f} {s['TP']:4d}{s['FN']:4d}"
              f"{s['FP']:4d}{s['TN']:4d}{s['type_exact']:6d}  {s['tc_median']}  {s['tc_max']}")

    # среди равноценных порогов берём середину диапазона: запас и в сторону пропуска, и в сторону ложных
    top_c, top_s = results[0]
    ties = sorted((c for c, s in results if rank_key(s) == rank_key(top_s)
                   and c["alpha"] == top_c["alpha"] and c["timecode_mode"] == top_c["timecode_mode"]),
                  key=lambda c: c["threshold"])
    best = ties[len(ties) // 2]
    per_video_report(cache, labels, rules, best)
    (out / "calibration.json").write_text(json.dumps(best, indent=2, ensure_ascii=False))
    print(f"\nВыбрано: {best}\nСохранено в {out / 'calibration.json'}")
    if len(cache) < 15:
        print(f"[!] В валидации {len(cache)} видео: одно видео = {100 / len(cache):.0f}%. Смотрите на порядок, а не на десятые.")


if __name__ == "__main__":
    main()
