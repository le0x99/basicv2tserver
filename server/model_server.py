import os
import tempfile
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse

from .model import VoiceModel



@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading voice model into GPU...")
    app.state.model = VoiceModel()
    print("Model loaded.")
    yield
    print("Shutting down model server...")


app = FastAPI(
    title="Voice-to-Text Model API",
    lifespan=lifespan
)


# ----------------------------
# Concurrency control
# ----------------------------

GPU_CONCURRENCY = 1
gpu_lock = asyncio.Semaphore(GPU_CONCURRENCY)


# ----------------------------
# Endpoints
# ----------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(request: Request, file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    allowed_extensions = (".wav", ".mp3", ".m4a", ".flac")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    tmp_path = None

    try:
        # Save upload to temp file (streaming)
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name

            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)

        # Get model from app state
        model = request.app.state.model

        # Serialize GPU access
        async with gpu_lock:
            result_text = model.transcribe(tmp_path)

        return JSONResponse({"text": result_text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
