from __future__ import annotations

from pathlib import Path
import shutil

import torch
from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor, AutoModelForImageClassification

from .gradcam import generate_gradcam_overlay
from .preprocess import open_rgb_image
from .helper import MODEL_CONFIG_URL, MODEL_ID, MODEL_NAME, MODEL_PAGE_URL, MODEL_PROCESSOR_URL, MODEL_WEIGHT_URL


class DeepfakePredictor:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cpu")
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        self._ensure_model_files()
        self.processor = None
        self.model = None
        self._load_model_assets()
        self.model.to(self.device)
        self.model.eval()

    def _ensure_model_files(self) -> None:
        needed = [
            self.model_dir / "config.json",
            self.model_dir / "preprocessor_config.json",
            self.model_dir / "model.safetensors",
        ]
        if all(path.exists() for path in needed):
            return
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(self.model_dir),
            local_dir_use_symlinks=False,
            allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors", "README.md"],
        )

    def _clear_model_cache(self) -> None:
        for path in self.model_dir.iterdir():
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            else:
                shutil.rmtree(path, ignore_errors=True)

    def _load_model_assets(self) -> None:
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_dir, local_files_only=True)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_dir, local_files_only=True)
        except Exception:
            self._clear_model_cache()
            self._ensure_model_files()
            self.processor = AutoImageProcessor.from_pretrained(self.model_dir, local_files_only=True)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_dir, local_files_only=True)

    def predict(self, image_path: str | Path, heatmap_path: str | Path | None = None) -> dict:
        pil_image = open_rgb_image(image_path)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probabilities = torch.softmax(logits, dim=0)

        real_probability = float(probabilities[0].item())
        fake_probability = float(probabilities[1].item())
        predicted_class = int(torch.argmax(probabilities).item())
        class_map = self.model.config.id2label
        predicted_label = class_map.get(str(predicted_class), class_map.get(predicted_class, str(predicted_class)))
        confidence = float(probabilities[predicted_class].item())

        heatmap_result = None
        if heatmap_path is not None:
            heatmap_result = generate_gradcam_overlay(
                model=self.model,
                image_processor=self.processor,
                image_path=image_path,
                output_path=heatmap_path,
                class_index=predicted_class,
            )

        return {
            "predicted_class": predicted_class,
            "predicted_label": predicted_label.upper(),
            "confidence": confidence,
            "real_probability": real_probability,
            "fake_probability": fake_probability,
            "heatmap_path": heatmap_result,
            "model_name": MODEL_NAME,
            "model_page_url": MODEL_PAGE_URL,
            "model_weight_url": MODEL_WEIGHT_URL,
            "model_config_url": MODEL_CONFIG_URL,
            "model_processor_url": MODEL_PROCESSOR_URL,
        }
