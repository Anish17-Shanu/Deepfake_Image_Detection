from __future__ import annotations

from pathlib import Path

import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .preprocess import image_to_numpy, open_rgb_image


def _reshape_transform_factory(image_size: int, patch_size: int):
    grid_size = image_size // patch_size

    def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor[:, 1:, :]
        batch_size, tokens, channels = tensor.size()
        if tokens != grid_size * grid_size:
            grid = int(tokens ** 0.5)
        else:
            grid = grid_size
        return tensor.reshape(batch_size, grid, grid, channels).permute(0, 3, 1, 2)

    return reshape_transform


def generate_gradcam_overlay(model, image_processor, image_path: str | Path, output_path: str | Path, class_index: int) -> str | None:
    try:
        pil_image = open_rgb_image(image_path)
        input_tensor = image_processor(images=pil_image, return_tensors="pt")["pixel_values"]
        target_layers = [model.vit.encoder.layer[-1].layernorm_before]
        reshape_transform = _reshape_transform_factory(model.config.image_size, model.config.patch_size)

        cam = GradCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        )

        with torch.enable_grad():
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(class_index)],
            )[0]

        rgb_image = image_to_numpy(pil_image)
        visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        return str(output_path)
    except Exception:
        return None
