"""HTTP-сервис по разделу 10 ТЗ.

  POST /api/v1/analyze        поле file (.mov), заголовок X-API-Key  -> {"job_id", "status"}
  GET  /api/v1/jobs/{job_id}  заголовок X-API-Key                     -> статус и результат

  GET  /                       веб-интерфейс (ключ UI_KEY подставляется в страницу)

Запуск: проще всего через scripts/run_service.sh. Вручную:
  API_KEYS=<ключ_жюри>,<ключ_ui> UI_KEY=<ключ_ui> MODEL_DIR=/models/videojev \
  uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
ANALYZER=stub запускает сервис без модели (отладка фронта и деплоя).
Один процесс = одна модель в памяти GPU = одна очередь. Не ставьте --workers > 1.
"""
import json
import logging
import os
import queue
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from analyzer import make_analyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

DATA = Path(os.environ.get("DATA_DIR", "jobs"))
API_KEYS = {k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()}
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "300")) * 1024 * 1024
ALLOWED = {".mov", ".mp4"}
UI_KEY = os.environ.get("UI_KEY", "")  # ключ для веб-страницы; должен быть и в API_KEYS
WEB_INDEX = Path(os.environ.get("WEB_INDEX", Path(__file__).resolve().parent.parent / "web" / "index.html"))

jobs: dict = {}
tasks: queue.Queue = queue.Queue()
state = {"analyzer": None}


def save(job):
    public = {k: v for k, v in job.items() if k != "path"}
    (DATA / job["job_id"] / "job.json").write_text(json.dumps(public, ensure_ascii=False), encoding="utf-8")


def worker():
    while True:
        job_id = tasks.get()
        job = jobs[job_id]
        t0 = time.time()
        try:
            result, raw = state["analyzer"].analyze(job["path"])
            job.update(status="done", result=result)
            (DATA / job_id / "raw.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        except Exception as e:  # сервис не падает из-за одного видео
            log.exception("job %s failed", job_id)
            job.update(status="failed", error=str(e)[:500])
        job["seconds"] = round(time.time() - t0, 1)
        job["waited"] = round(t0 - job["created"], 1)
        log.info("job %s: %s, ожидание %.1f с, обработка %.1f с", job_id, job["status"], job["waited"], job["seconds"])
        save(job)
        tasks.task_done()


def restore():
    """После перезапуска: готовые задачи снова доступны, незавершённые возвращаются в очередь."""
    for p in sorted(DATA.glob("*/job.json"), key=lambda p: p.stat().st_mtime):
        job = json.loads(p.read_text(encoding="utf-8"))
        inputs = list(p.parent.glob("input.*"))
        if not inputs:
            continue
        job["path"] = str(inputs[0])
        jobs[job["job_id"]] = job
        if job["status"] == "processing":
            tasks.put(job["job_id"])


@asynccontextmanager
async def lifespan(_app):
    DATA.mkdir(parents=True, exist_ok=True)
    if not API_KEYS:
        log.warning("API_KEYS не задан: все запросы будут отклонены")
    state["analyzer"] = make_analyzer()  # модель грузится один раз при старте
    restore()
    threading.Thread(target=worker, daemon=True).start()
    yield


app = FastAPI(title="Проверка кассовой дисциплины по видео", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


def check_key(key):
    if not key or key not in API_KEYS:
        raise HTTPException(status_code=401, detail="invalid API key")


@app.post("/api/v1/analyze")
def analyze(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(status_code=415, detail=f"ожидается .mov, получено {ext or 'без расширения'}")
    job_id = uuid.uuid4().hex[:12]
    folder = DATA / job_id
    folder.mkdir(parents=True)
    path, size = folder / f"input{ext}", 0
    with path.open("wb") as f:
        while chunk := file.file.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                f.close()
                shutil.rmtree(folder, ignore_errors=True)
                raise HTTPException(status_code=413, detail="файл слишком большой")
            f.write(chunk)
    job = {"job_id": job_id, "status": "processing", "path": str(path), "created": time.time(),
           "filename": file.filename}
    jobs[job_id] = job
    save(job)
    tasks.put(job_id)
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str, x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] == "done":
        return {"job_id": job_id, "status": "done", **job["result"]}
    if job["status"] == "failed":
        return {"job_id": job_id, "status": "failed", "error": job.get("error", "")}
    return {"job_id": job_id, "status": "processing"}


@app.get("/", response_class=HTMLResponse)
def ui():
    """Веб-интерфейс на том же адресе, что и API: один URL для жюри, без CORS."""
    if not WEB_INDEX.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    html = WEB_INDEX.read_text(encoding="utf-8")
    return HTMLResponse(html.replace("__UI_KEY__", UI_KEY).replace("__API_BASE__", ""))


@app.get("/healthz")
def healthz():
    return {"ok": state["analyzer"] is not None, "queue": tasks.qsize(),
            "analyzer": type(state["analyzer"]).__name__}
