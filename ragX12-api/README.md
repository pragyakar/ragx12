# ragX12-api

Simple FastAPI service with a summarization stub for X12 content.

## Endpoints

- `GET /health` -> `{ "status": "ok" }`
- `POST /summarize` -> Request body: `{ "x12": "<string>" }`
  - Response: `{ "summary": "<static>", "raw": "<echoed input>" }`

## Development

### Setup
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run
```
PYTHONPATH=src make ragx12-api
```
The server reloads on code changes.

### Host with Ngrok for testing
```
ngrok http 8000
```
Use the URL in lovable for live testing with Lovable frontend app.

### Environment
Variables loaded from `.env` using `python-dotenv`. Ensure `PYTHONPATH=src` (or run with a virtual environment that sets it) so `ragx12_core` can be imported.

## Notes
- Core logic lives in `src/ragx12_core.py`.
