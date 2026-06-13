from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from .config import MODEL_DIR, MODEL_ID, MODEL_NAME


class DeepfakeModel:
    def __init__(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        self.processor = AutoImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_DIR, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image) -> dict:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=0)

        real_idx, fake_idx = self._label_indexes()
        real_probability = float(probabilities[real_idx].item())
        fake_probability = float(probabilities[fake_idx].item())
        label = "DEEPFAKE" if fake_probability >= real_probability else "REAL"
        confidence = max(real_probability, fake_probability)
        return {
            "model": MODEL_NAME,
            "label": label,
            "confidence": confidence,
            "real_probability": real_probability,
            "fake_probability": fake_probability,
        }

    def _ensure_model(self) -> None:
        required = [MODEL_DIR / "config.json", MODEL_DIR / "preprocessor_config.json", MODEL_DIR / "model.safetensors"]
        if all(path.exists() for path in required):
            return
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(MODEL_DIR),
            allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors", "README.md"],
        )

    def _label_indexes(self) -> tuple[int, int]:
        labels = {int(key): str(value).lower() for key, value in self.model.config.id2label.items()}
        real_idx = next((key for key, label in labels.items() if "real" in label), min(labels))
        fake_idx = next((key for key, label in labels.items() if "fake" in label), max(labels))
        return real_idx, fake_idx
