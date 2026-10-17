# Cash discipline check from video (Datathon 26)

Local AI service: upload a cashier video, get a verdict, violation type, timecode, evidence frames and the rule reference.
All models run inside the service; no cloud AI APIs.

## Layout
- `service/` — API (`app.py`), model wrapper and decision logic (`analyzer.py`), rules (`rules.json`), thresholds (`calibration.json`)
- `tools/` — `xlsx_to_labels.py` (labels sheet -> labels.csv), `make_manifest.py` (VideoJev training data), `evaluate.py` (metrics + threshold calibration)
- `scripts/setup_vast.sh` — GPU machine setup and offline smoke test
- `scripts/finetune.sh` — fine-tuning VideoJev-4B on the labeled videos, starting from VideoJev-4B weights; compares with the base model on held-out videos
- `tools/videojev_init.py`, `tools/scale_lr.py` — start fine-tuning from VideoJev-4B weights, gentler learning rate
- `scripts/run_service.sh` — starts API + UI + Cloudflare tunnel
- `web/index.html` — web UI, served by the API at `/`
- `Dockerfile` — portable offline image (weights inside)

## Run
```bash
bash scripts/setup_vast.sh          # once per GPU machine
bash scripts/run_service.sh         # API + web UI + public HTTPS URL; keys are kept in .env
ANALYZER=stub bash scripts/run_service.sh   # same, without the model
```
Web UI: `https://<host>/`. API: `https://<host>/api/v1/...`.

## API (section 10 of the brief)
```bash
curl -X POST https://<host>/api/v1/analyze -H "X-API-Key: <key>" -F "file=@video.mov"
curl https://<host>/api/v1/jobs/<job_id> -H "X-API-Key: <key>"
```

## TODO for the submission (section 8)
- [ ] Architecture and models
- [ ] Sources: document, article, and how each rule maps to what is visible on video
- [ ] List of detected violation types
- [ ] Limitations and known failure cases
