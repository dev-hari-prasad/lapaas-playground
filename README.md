# Lapaas Playground

A modular web playground to test image understanding models side-by-side:

- FastVLM (Apple) — image-to-text description
- MobileCLIP (Apple) — zero-shot image classification

## Features

- Side-by-side input/output with timing (total, prediction, first-load)
- Separate pages per tool with shared layout and Lapaas branding
- REST APIs for programmatic access

## Quickstart

1) Create and activate a virtualenv
```bash
python3 -m venv fastvlm_env
source fastvlm_env/bin/activate
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Run
```bash
python app.py
# open http://localhost:5001
```

## Endpoints

Pages
- `GET /` — Home
- `GET /fastvlm` — FastVLM page (describe)
- `GET /mobileclip` — MobileCLIP page (classify)

APIs
- `POST /api/fastvlm` — fields: `image`
- `POST /api/mobileclip` — fields: `image`, `labels` (comma-separated)

## Services
- `services/fastvlm_service.py` — loads `apple/FastVLM-0.5B` and generates descriptions
- `services/mobileclip_service.py` — loads `MobileCLIP-S2 (datacompdr)` and classifies labels

## Development
- Shared layout in `templates/base.html`
- Pages in `templates/*.html`
- Healthcheck: `GET /healthz`

## Open Source
- License: MIT (see `LICENSE`)
- Contributing: see `CONTRIBUTING.md`
- Code of Conduct: see `CODE_OF_CONDUCT.md`

## Notes
- CPU inference is enabled by default; GPU can be wired later.
- First FastVLM request may include model load time.
