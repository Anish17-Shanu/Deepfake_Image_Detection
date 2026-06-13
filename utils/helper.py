from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_ID = "dima806/deepfake_vs_real_image_detection"
MODEL_PAGE_URL = "https://huggingface.co/dima806/deepfake_vs_real_image_detection"
MODEL_WEIGHT_URL = "https://huggingface.co/dima806/deepfake_vs_real_image_detection/resolve/main/model.safetensors?download=true"
MODEL_CONFIG_URL = "https://huggingface.co/dima806/deepfake_vs_real_image_detection/resolve/main/config.json?download=true"
MODEL_PROCESSOR_URL = "https://huggingface.co/dima806/deepfake_vs_real_image_detection/resolve/main/preprocessor_config.json?download=true"
MODEL_ACCURACY = 0.9927
MODEL_NAME = "Deepfake vs Real Image Detection (ViT-Base 224)"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def ensure_directories(*directories: Path) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_history(history_path: Path, record: dict[str, Any]) -> None:
    history = load_json(history_path, default=[])
    history.insert(0, record)
    save_json(history_path, history[:100])


def get_history(history_path: Path) -> list[dict[str, Any]]:
    data = load_json(history_path, default=[])
    return data if isinstance(data, list) else []


def get_record_by_id(history_path: Path, prediction_id: str) -> dict[str, Any] | None:
    for record in get_history(history_path):
        if record.get("prediction_id") == prediction_id:
            return record
    return None


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def generate_report_text(record: dict[str, Any]) -> str:
    lines = [
        "Deepfake Image Detection Report",
        "================================",
        f"Prediction ID : {record.get('prediction_id')}",
        f"Timestamp     : {record.get('timestamp')}",
        f"Filename      : {record.get('original_filename')}",
        f"Result        : {record.get('predicted_label')}",
        f"Confidence    : {record.get('confidence_percent')}",
        f"Real Probability: {record.get('real_probability_percent')}",
        f"Fake Probability: {record.get('fake_probability_percent')}",
        f"Model         : {record.get('model_name')}",
        f"Model Source  : {record.get('model_page_url')}",
        "",
        "Interpretation",
        "--------------",
        record.get("interpretation", ""),
        "",
        "Notes",
        "-----",
        "This report is generated automatically for final-year project demonstration.",
        "The model is a public pretrained checkpoint and should be used as a screening tool, not a forensic verdict.",
    ]
    return "\n".join(lines)


def build_history_record(
    *,
    prediction_id: str,
    original_filename: str,
    stored_filename: str,
    predicted_label: str,
    confidence: float,
    real_probability: float,
    fake_probability: float,
    upload_url: str,
    result_url: str,
    heatmap_url: str | None,
    report_url: str,
    model_name: str,
    model_page_url: str,
) -> dict[str, Any]:
    interpretation = (
        "The uploaded image is most likely genuine." if predicted_label == "REAL" else "The uploaded image shows strong evidence of manipulation or synthetic generation."
    )
    return {
        "prediction_id": prediction_id,
        "timestamp": now_iso(),
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "predicted_label": predicted_label,
        "confidence": round(confidence, 6),
        "confidence_percent": format_percent(confidence),
        "real_probability": round(real_probability, 6),
        "real_probability_percent": format_percent(real_probability),
        "fake_probability": round(fake_probability, 6),
        "fake_probability_percent": format_percent(fake_probability),
        "upload_url": upload_url,
        "result_url": result_url,
        "heatmap_url": heatmap_url,
        "report_url": report_url,
        "model_name": model_name,
        "model_page_url": model_page_url,
        "interpretation": interpretation,
    }
