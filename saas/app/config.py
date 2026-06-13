import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
REPORT_DIR = DATA_DIR / "reports"
HISTORY_PATH = DATA_DIR / "history.json"
MODEL_DIR = BASE_DIR / "model"

MODEL_ID = os.getenv("MODEL_ID", "dima806/deepfake_vs_real_image_detection")
MODEL_NAME = "Deepfake vs Real Image Detection ViT"
MODEL_PAGE_URL = f"https://huggingface.co/{MODEL_ID}"
MODEL_REPORTED_ACCURACY = 0.9927
MODEL_EVAL_SUPPORT = 76161
MODEL_LIMITATION = (
    "The public model card reports 99.27% evaluation accuracy, but also warns that the "
    "training data is several years old and newer generators can cause concept drift."
)
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
