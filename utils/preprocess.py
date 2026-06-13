from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def open_rgb_image(image_path: str | Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def prepare_image_for_model(image_path: str | Path, size: tuple[int, int] = (224, 224)) -> Image.Image:
    image = open_rgb_image(image_path)
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)


def image_to_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(image).astype("float32") / 255.0


def resize_preview(image_path: str | Path, size: tuple[int, int] = (640, 640)) -> Image.Image:
    image = open_rgb_image(image_path)
    image.thumbnail(size, Image.Resampling.LANCZOS)
    return image
