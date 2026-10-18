"""VideoJev -> вердикт в формате API жюри (раздел 10 ТЗ).

На одно видео модель вызывается один раз: все вопросы (choice + grounding по каждому правилу)
идут одним predict_video, визуальная часть считается один раз (joint inference).
torch/transformers импортируются только в VideoJevAnalyzer, поэтому decide() и заглушка
работают без GPU (их использует tools/evaluate.py и режим ANALYZER=stub).
"""
import base64
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import cv2

os.environ.setdefault("HF_HUB_OFFLINE", "1")        # во время работы ничего не скачиваем
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
log = logging.getLogger("analyzer")

DEFAULT_CALIB = {"threshold": 0.5, "alpha": 1.0, "timecode_mode": "peak",
                 "grounding_threshold": 0.25, "per_rule": {}}
YES_NO = ["A. Yes.", "B. No."]


def pick_attention():
    """flash-attn, если он установлен и работает; иначе встроенное внимание PyTorch (sdpa).

    Принудительно: VJ_ATTN=sdpa или VJ_ATTN=flash_attention_2.
    """
    forced = os.environ.get("VJ_ATTN")
    if forced:
        return forced
    try:
        from flash_attn import flash_attn_func  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_rules(path):
    with open(path, encoding="utf-8") as f:
        rules = json.load(f)["rules"]
    for r in rules:
        if not (r.get("source") or {}).get("document"):
            log.warning("Правило %s без источника: заполните source в rules.json", r["id"])
    return rules


def build_questions(rules):
    qs = []
    for r in rules:
        qs.append({"id": f"choice/{r['id']}", "task": "choice", "query": r["choice_en"], "options": YES_NO})
        qs.append({"id": f"grounding/{r['id']}", "task": "grounding", "query": r["grounding_en"]})
    return qs


def probe_duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True)
        return float(out.stdout.strip())
    except Exception:
        cap = cv2.VideoCapture(str(path))
        fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return n / fps if fps else 0.0


def peak_time(probs, duration):
    """Момент максимума вероятности grounding.

    Предположение: бины равномерно покрывают видео (в демо модели 10 бинов на 10-секундный ролик).
    Сверьте с полем interval на своих видео в первый же прогон.
    """
    if not probs or duration <= 0:
        return None
    i = max(range(len(probs)), key=lambda k: probs[k])
    return (i + 0.5) * duration / len(probs), float(probs[i])


def decide(raw, rules, duration, calib):
    """Сырые ответы модели -> список найденных нарушений (по одному на правило)."""
    by_id = {r["id"]: r for r in raw}
    found = []
    for rule in rules:
        c = by_id.get(f"choice/{rule['id']}") or {}
        g = by_id.get(f"grounding/{rule['id']}") or {}
        p_yes = float(c["probabilities"][0]) if c.get("probabilities") else 0.0
        peak = peak_time(g.get("probabilities"), duration)
        alpha = calib.get("alpha", 1.0)
        score = alpha * p_yes + (1 - alpha) * (peak[1] if peak else 0.0)
        if score < calib.get("per_rule", {}).get(rule["id"], calib.get("threshold", 0.5)):
            continue
        interval = g.get("interval")
        if interval and calib.get("timecode_mode") == "start":
            t = float(interval[0])
        elif peak:
            t = peak[0]
        else:
            t = float(interval[0]) if interval else 0.0
        t = min(max(t, 0.0), max(duration - 0.05, 0.0))
        found.append({"rule": rule, "score": score, "timecode_sec": round(t, 2), "interval": interval})
    k = calib.get("max_violations")
    if k:  # только k самых уверенных типов; вердикт от этого не меняется
        found = sorted(found, key=lambda x: -x["score"])[:int(k)]
    return sorted(found, key=lambda x: x["timecode_sec"])


def describe(rule, item):
    parts = [rule["description_ru"].rstrip(".") + "."]
    if item["interval"]:
        a, b = item["interval"][:2]
        parts.append(f"Фрагмент: {float(a):.1f}–{float(b):.1f} с.")
    parts.append(f"Кадр-доказательство: {item['timecode_sec']:.1f} с.")
    src = rule.get("source") or {}
    if src.get("document"):
        punkt = src.get("punkt") or ""
        parts.append(f"Основание: {src['document']}{', ' + punkt if punkt else ''}.")
    parts.append("Это подсказка для проверяющего, а не окончательный вывод.")
    return " ".join(parts)


def evidence_times(t, duration):
    hi = max(duration - 0.05, 0.0)
    out = []
    for x in (t, t - 1.0, t + 1.0):  # первый кадр = таймкод
        x = round(min(max(x, 0.0), hi), 2)
        if x not in out:
            out.append(x)
    return out


def grab_frames_b64(path, times, max_side=960, quality=85):
    cap = cv2.VideoCapture(str(path))
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            frames.append({"timestamp_sec": t, "image_base64": base64.b64encode(buf.tobytes()).decode()})
    cap.release()
    return frames


def build_result(raw, rules, duration, calib, video_path):
    violations = []
    for item in decide(raw, rules, duration, calib):
        rule = item["rule"]
        violations.append({
            "type": rule["name_ru"],
            "timecode_sec": item["timecode_sec"],
            "description": describe(rule, item),
            "evidence_frames": grab_frames_b64(video_path, evidence_times(item["timecode_sec"], duration)),
        })
    return {"verdict": "violation" if violations else "no_violation", "violations": violations}


class VideoJevAnalyzer:
    def __init__(self, model_dir, rules_path, calib_path=None, device="cuda:0",
                 fps=2.0, max_frames=2048, context=32768):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.rules = load_rules(rules_path)
        self.questions = build_questions(self.rules)
        self.calib = dict(DEFAULT_CALIB)
        if calib_path and Path(calib_path).exists():
            with open(calib_path, encoding="utf-8") as f:
                self.calib.update(json.load(f))
        self.fps, self.max_frames, self.context = fps, max_frames, context
        # local_files_only + каталог с весами и кодом модели: работает без интернета
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
        attn = pick_attention()
        log.info("attention: %s", attn)
        self.model = AutoModel.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16,
            device_map={"": device}, attn_implementation=attn)
        self.model.prepare_for_inference()

    def raw(self, path, fps_scale=1.0):
        return self.model.predict_video(
            self.processor, str(path), self.questions,
            fps=self.fps * fps_scale, max_frames=self.max_frames, context=self.context,
            threshold=self.calib["grounding_threshold"])

    def analyze(self, path):
        t0 = time.time()
        duration = probe_duration(path)
        try:
            raw = self.raw(path)
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            log.warning("OOM на %s, повтор с половинным fps", path)
            raw = self.raw(path, fps_scale=0.5)
        result = build_result(raw, self.rules, duration, self.calib, path)
        log.info("%s: %.1f с видео, %d нарушений, обработка %.1f с",
                 path, duration, len(result["violations"]), time.time() - t0)
        return result, raw


class StubAnalyzer:
    """Без модели: отладка фронта, API и деплоя (ANALYZER=stub)."""

    def analyze(self, path):
        t = round(probe_duration(path) / 2, 2)
        time.sleep(2)
        return {"verdict": "violation", "violations": [{
            "type": "Тест (заглушка)", "timecode_sec": t,
            "description": "Ответ заглушки: модель не загружена.",
            "evidence_frames": grab_frames_b64(path, [t]),
        }]}, []


def make_analyzer():
    if os.environ.get("ANALYZER", "videojev") == "stub":
        return StubAnalyzer()
    return VideoJevAnalyzer(
        model_dir=os.environ["MODEL_DIR"],
        rules_path=os.environ.get("RULES_PATH", "rules.json"),
        calib_path=os.environ.get("CALIB_PATH", "calibration.json"),
        fps=float(os.environ.get("VJ_FPS", "2.0")),
        context=int(os.environ.get("VJ_CONTEXT", "32768")),
    )
