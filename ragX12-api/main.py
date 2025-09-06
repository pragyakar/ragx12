from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from fastapi import HTTPException

# Import core logic from src package. Requires running with PYTHONPATH=src or setting sys.path.
try:
    from ragx12_core import generate_summary  # when PYTHONPATH includes ./src
except ModuleNotFoundError:
    import sys as _sys, pathlib as _pl
    _src_path = _pl.Path(__file__).parent / 'src'
    if str(_src_path) not in _sys.path:
        _sys.path.insert(0, str(_src_path))
    from ragx12_core import generate_summary

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="ragX12 API", version="0.1.0")

# Access a dummy variable (not strictly needed, but shows it's loaded)
DUMMY_VARIABLE = os.getenv("DUMMY_VARIABLE")


class SummarizeRequest(BaseModel):
    x12: str


@app.get("/health")
async def health():
    return {"status": "OK", "version": "0.1.0", "description": "RagX12 - AI-powered healthcare EDI summarization tool for processing X12 files. Convert complex EDI data into clear, human-readable summaries using advanced AI."}


@app.post("/summarize")
async def summarize(payload: SummarizeRequest):
    # Wrapper around core generator to convert errors to HTTP responses.
    try:
        result = generate_summary(payload.x12)
        return result  # includes summary, data, possible_actions, code_meanings
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")
