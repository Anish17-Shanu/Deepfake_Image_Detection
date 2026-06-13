from __future__ import annotations

from PIL import Image

from .analysis import compression_score, frequency_score
from .config import HEURISTIC_MODEL_NAME, MODEL_BACKEND, MODEL_DIR, MODEL_ID, MODEL_NAME


class DeepfakeModel:
    def __init__(self) -> None:
        self.backend = MODEL_BACKEND
        self.engine = VitDeepfakeModel() if self.backend == "vit" else HeuristicDeepfakeModel()

    def predict(self, image: Image.Image) -> dict:
        return self.engine.predict(image)


class HeuristicDeepfakeModel:
    def predict(self, image: Image.Image) -> dict:
        import cv2
        import numpy as np

        rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        texture_risk = np.clip(1 - blur / 700, 0, 1)
        compression_risk = compression_score(rgb)
        lighting_risk = np.clip(np.std([np.mean(rgb[:, :, channel]) for channel in range(3)]) / 45, 0, 1)
        edge_risk = np.clip(np.mean(cv2.Canny(gray, 80, 160) > 0) * 4, 0, 1)
        frequency_risk = frequency_score(gray)

        fake_probability = float(
            np.clip(
                texture_risk * 0.26
                + compression_risk * 0.20
                + lighting_risk * 0.16
                + edge_risk * 0.18
                + frequency_risk * 0.20,
                0.03,
                0.97,
            )
        )
        real_probability = float(1 - fake_probability)
        label = "DEEPFAKE" if fake_probability >= 0.5 else "REAL"
        confidence = max(real_probability, fake_probability)

        return {
            "model": HEURISTIC_MODEL_NAME,
            "backend": "heuristic",
            "label": label,
            "confidence": confidence,
            "real_probability": real_probability,
            "fake_probability": fake_probability,
        }


class VitDeepfakeModel:
    def __init__(self) -> None:
        from huggingface_hub import snapshot_download
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.torch = torch
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_model(snapshot_download)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        self.processor = AutoImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_DIR, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image) -> dict:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probabilities = self.torch.softmax(logits, dim=0)

        real_idx, fake_idx = self._label_indexes()
        real_probability = float(probabilities[real_idx].item())
        fake_probability = float(probabilities[fake_idx].item())
        label = "DEEPFAKE" if fake_probability >= real_probability else "REAL"
        confidence = max(real_probability, fake_probability)
        return {
            "model": MODEL_NAME,
            "backend": "vit",
            "label": label,
            "confidence": confidence,
            "real_probability": real_probability,
            "fake_probability": fake_probability,
        }

    def _ensure_model(self, snapshot_download) -> None:
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
