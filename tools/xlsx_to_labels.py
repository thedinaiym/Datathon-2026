#!/usr/bin/env python3
"""Dataset.xlsx (sheet "labels") -> labels.csv for make_manifest.py / evaluate.py.

  python tools/xlsx_to_labels.py --xlsx data/Dataset.xlsx --rules service/rules.json \
      --videos_dir data/videos --out data/labels.csv

- Matches sheet names to files on disk: Latin X vs Cyrillic Х, case, extension.
- Parses timecodes like "22 секунда", "1минута 8 секунда", "1:08".
- A row with several categories separated by ";" becomes several events with the same timecode.
- Writes <out_dir>/splits/all_videos.txt: with no fine-tuning, every labeled video is valid test data.
Columns are read by position: A file, B timecode, C verdict, D note, E category.
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

VIDEO_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}


def norm_name(name):
    s = Path(str(name).strip()).stem.lower().replace("х", "x")  # Cyrillic х -> Latin x
    return re.sub(r"\s+", "", s)


def norm_cat(text):
    """Category key: text before "(" so article numbers in the label do not matter."""
    s = str(text).split("(")[0].strip().lower().replace("ё", "е")
    return re.sub(r"\s+", " ", s)


def parse_time(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
        return None
    if hasattr(v, "hour"):  # a cell formatted as time
        return v.hour * 3600 + v.minute * 60 + v.second
    s = str(v).strip().lower()
    m = re.fullmatch(r"(\d+):(\d{1,2}(?:\.\d+)?)", s)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    mins = re.search(r"(\d+(?:[.,]\d+)?)\s*мин", s)
    secs = re.search(r"(\d+(?:[.,]\d+)?)\s*сек", s)
    if mins or secs:
        num = lambda x: float(x.group(1).replace(",", ".")) if x else 0.0
        return num(mins) * 60 + num(secs)
    return float(s.replace(",", "."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--sheet", default="labels")
    ap.add_argument("--rules", required=True)
    ap.add_argument("--videos_dir", help="folder with videos: real file names go into labels.csv")
    ap.add_argument("--out", default="data/labels.csv")
    ap.add_argument("--unlabeled_as_clean", action="store_true",
                    help="files in videos_dir that are missing from the sheet are clean videos")
    args = ap.parse_args()

    with open(args.rules, encoding="utf-8") as f:
        rules = json.load(f)["rules"]
    cat_to_id = {norm_cat(r["label_ru"]): r["id"] for r in rules}

    on_disk = {}
    if args.videos_dir:
        for p in sorted(Path(args.videos_dir).iterdir()):
            if p.suffix.lower() in VIDEO_EXT:
                on_disk[norm_name(p.name)] = p.name

    df = pd.read_excel(args.xlsx, sheet_name=args.sheet, header=0)
    rows, problems, seen, types = [], [], set(), Counter()
    for _, r in df.iterrows():
        raw_name = str(r.iloc[0]).strip()
        if not raw_name or raw_name.lower() == "nan":
            continue
        key = norm_name(raw_name)
        seen.add(key)
        video = on_disk.get(key)
        if video is None:
            video = f"{raw_name}.mov"
            if args.videos_dir:
                problems.append(f"{raw_name}: no matching file in {args.videos_dir}")
        verdict = str(r.iloc[2]).strip().lower()
        cats = [c.strip() for c in str(r.iloc[4]).split(";") if c.strip() and c.strip().lower() != "nan"]
        clean = verdict.startswith("без") or (len(cats) == 1 and norm_cat(cats[0]) == "без нарушения")
        if clean:
            rows.append({"video": video, "label": 0, "start_sec": "", "end_sec": "", "type": "", "note": ""})
            types["no_violation"] += 1
            continue
        t = parse_time(r.iloc[1])
        if t is None:
            problems.append(f"{raw_name}: marked as violation but has no timecode")
            continue
        note = "" if pd.isna(r.iloc[3]) else str(r.iloc[3]).strip()
        for c in cats:
            rid = cat_to_id.get(norm_cat(c))
            if rid is None:
                problems.append(f"{raw_name}: category not in rules.json: {c!r}")
                continue
            rows.append({"video": video, "label": 1, "start_sec": t, "end_sec": "", "type": rid, "note": note})
            types[rid] += 1

    unlabeled = sorted(v for k, v in on_disk.items() if k not in seen)
    if args.unlabeled_as_clean:
        for v in unlabeled:
            rows.append({"video": v, "label": 0, "start_sec": "", "end_sec": "", "type": "", "note": "not in sheet: clean"})
            types["no_violation"] += 1
        unlabeled = []

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video", "label", "start_sec", "end_sec", "type", "note"])
        w.writeheader()
        w.writerows(rows)
    videos = sorted({r["video"] for r in rows})
    (out.parent / "splits").mkdir(exist_ok=True)
    (out.parent / "splits" / "all_videos.txt").write_text("\n".join(videos) + "\n", encoding="utf-8")

    n_viol = len({r["video"] for r in rows if r["label"] == 1})
    print(f"labeled videos: {len(videos)} (with violation: {n_viol}, clean: {len(videos) - n_viol})")
    for k, c in types.most_common():
        print(f"  {c:3d}  {k}")
    if unlabeled:
        print(f"\nVideos WITHOUT labels ({len(unlabeled)}): label them or keep as a blind test")
        for v in unlabeled:
            print("  ", v)
    if problems:
        print("\nProblems:")
        for p in problems:
            print("  ", p)
    print(f"\n-> {out}\n-> {out.parent / 'splits' / 'all_videos.txt'}")


if __name__ == "__main__":
    main()
