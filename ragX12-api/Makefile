# Makefile for ragX12-api

.PHONY: ragx12-api install tunnel-ragx12-api venv clean

VENV := .venv
PY := python3
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn

$(VENV):
	$(PY) -m venv $(VENV)
	$(PIP) install -U pip

install: $(VENV)
	$(PIP) install -r requirements.txt

ragx12-api: install
	$(UVICORN) main:app --reload --port 8000

tunnel-ragx12-api: install
	($(UVICORN) main:app --reload --port 9000 &) ; \
	UVICORN_PID=$$! ; \
	echo "Uvicorn PID $$UVICORN_PID" ; \
	ngrok http 9000 ; \
	echo "Stopping uvicorn $$UVICORN_PID" ; \
	kill $$UVICORN_PID || true

clean:
	rm -rf $(VENV) __pycache__ **/__pycache__