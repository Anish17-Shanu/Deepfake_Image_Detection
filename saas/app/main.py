from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from .analysis import detect_faces, forensic_stats, quality_stats, score_summary
from .config import (
    ALLOWED_CONTENT_TYPES,
    BASE_DIR,
    MAX_UPLOAD_BYTES,
    MODEL_EVAL_SUPPORT,
    MODEL_ID,
    MODEL_LIMITATION,
    MODEL_PAGE_URL,
    MODEL_REPORTED_ACCURACY,
    REPORT_DIR,
    UPLOAD_DIR,
)
from .model import DeepfakeModel
from .reports import build_report
from .storage import append_history, ensure_storage, find_record, load_history, write_report

app = FastAPI(title="Free Deepfake Image Analyzer", version="1.0.0")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
ensure_storage()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/data", StaticFiles(directory=str(BASE_DIR / "data")), name="data")

model: DeepfakeModel | None = None


@app.on_event("startup")
def startup() -> None:
    ensure_storage()


def get_model() -> DeepfakeModel:
    global model
    if model is None:
        model = DeepfakeModel()
    return model


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "history": load_history()[:8],
            "model_id": MODEL_ID,
            "model_page_url": MODEL_PAGE_URL,
            "model_accuracy": MODEL_REPORTED_ACCURACY,
            "model_eval_support": MODEL_EVAL_SUPPORT,
            "model_limitation": MODEL_LIMITATION,
        },
    )


@app.post("/analyze")
async def analyze(request: Request, image: UploadFile = File(...)) -> RedirectResponse:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WEBP image.")
    payload = await image.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 16 MB.")
    try:
        pil_image = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    scan_id = uuid4().hex[:12]
    filename = safe_filename(image.filename or "upload.jpg")
    stored_name = f"{scan_id}_{filename}"
    upload_path = UPLOAD_DIR / stored_name
    upload_path.write_bytes(payload)

    face_detection = detect_faces(pil_image)
    quality = quality_stats(pil_image, face_detection["boxes"])
    prediction = get_model().predict(pil_image)
    forensics = forensic_stats(pil_image)
    scores = score_summary(prediction, quality, forensics)
    record = {
        "scan_id": scan_id,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "filename": filename,
        "image_url": f"/data/uploads/{stored_name}",
        "face_detection": face_detection,
        "quality": quality,
        "prediction": prediction,
        "forensics": forensics,
        "scores": scores,
        "model_stats": {
            "model_id": MODEL_ID,
            "model_page_url": MODEL_PAGE_URL,
            "reported_accuracy": MODEL_REPORTED_ACCURACY,
            "evaluation_support": MODEL_EVAL_SUPPORT,
            "limitation": MODEL_LIMITATION,
        },
    }
    report = build_report(record)
    report_path = write_report(scan_id, report)
    record["report_url"] = f"/reports/{report_path.name}"
    append_history(record)
    return RedirectResponse(url=f"/results/{scan_id}", status_code=303)


@app.get("/results/{scan_id}", response_class=HTMLResponse)
def result(request: Request, scan_id: str) -> HTMLResponse:
    record = find_record(scan_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Result not found.")
    return templates.TemplateResponse("result.html", {"request": request, "record": record})


@app.get("/reports/{filename}")
def report(filename: str) -> FileResponse:
    path = REPORT_DIR / safe_filename(filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(path, media_type="text/markdown", filename=path.name)


def safe_filename(filename: str) -> str:
    clean = Path(filename).name.replace(" ", "_")
    return "".join(char for char in clean if char.isalnum() or char in {"-", "_", "."})[:160] or "upload.jpg"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
