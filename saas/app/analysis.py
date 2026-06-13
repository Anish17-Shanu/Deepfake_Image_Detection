from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter


def detect_faces(image: Image.Image) -> dict:
    rgb = np.asarray(image.convert("RGB").resize(_bounded_size(image.size, 480)))
    red = rgb[:, :, 0].astype(float)
    green = rgb[:, :, 1].astype(float)
    blue = rgb[:, :, 2].astype(float)

    # Lightweight skin-tone region estimate. This is intentionally conservative
    # and avoids OpenCV so the app fits Render's 512 MB free instance.
    channel_spread = np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])
    skin = (
        (red > 80)
        & (green > 30)
        & (blue > 15)
        & (channel_spread > 15)
        & (red > green * 1.03)
        & (red > blue * 1.18)
        & ((red - blue) > 25)
    )
    if skin.mean() < 0.015:
        return {"face_count": 0, "boxes": []}

    ys, xs = np.where(skin)
    if len(xs) == 0:
        return {"face_count": 0, "boxes": []}

    scale_x = image.width / rgb.shape[1]
    scale_y = image.height / rgb.shape[0]
    box = [
        int(xs.min() * scale_x),
        int(ys.min() * scale_y),
        int((xs.max() + 1) * scale_x),
        int((ys.max() + 1) * scale_y),
    ]
    return {"face_count": 1, "boxes": [box]}


def quality_stats(image: Image.Image, boxes: list[list[int]]) -> dict:
    gray = _gray_array(image)
    blur = laplacian_variance(gray)
    brightness = float(np.mean(gray))
    resolution = int(image.width * image.height)
    face_ratio = 0.0
    if boxes:
        largest = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        face_ratio = ((largest[2] - largest[0]) * (largest[3] - largest[1])) / max(1, resolution)

    reasons: list[str] = []
    if not boxes:
        reasons.append("No clear face-like region detected.")
    if len(boxes) > 1:
        reasons.append(f"{len(boxes)} faces detected. Upload one clear face for best reliability.")
    if resolution < 224 * 224:
        reasons.append("Resolution is below 224x224.")
    if blur < 75:
        reasons.append("Image is too blurry for reliable analysis.")
    if brightness < 40:
        reasons.append("Image is too dark.")
    if brightness > 220:
        reasons.append("Image is overexposed.")
    if boxes and face_ratio < 0.03:
        reasons.append("Face-like region is too small in the image.")

    return {
        "blur_score": blur,
        "brightness": brightness,
        "resolution": resolution,
        "face_size_ratio": face_ratio,
        "accepted": not reasons,
        "reasons": reasons,
    }


def forensic_stats(image: Image.Image) -> list[dict]:
    rgb = np.asarray(image.convert("RGB").resize(_bounded_size(image.size, 640)))
    gray = _gray_array(image)
    return [
        indicator("Texture anomalies", np.clip(1 - laplacian_variance(gray) / 700, 0, 1), "Micro-texture smoothness measured from local contrast."),
        indicator("Compression inconsistencies", compression_score(rgb), "Residual differences after controlled recompression."),
        indicator("Lighting mismatch", np.clip(np.std([np.mean(rgb[:, :, i]) for i in range(3)]) / 45, 0, 1), "Color-channel illumination imbalance."),
        indicator("Edge artifacts", edge_score(gray), "High-contrast edge density across the image."),
        indicator("Frequency signal", frequency_score(gray), "Frequency-domain periodicity often seen in generated imagery."),
    ]


def score_summary(prediction: dict, quality: dict, forensics: list[dict]) -> dict:
    forensic_risk = sum(item["score"] for item in forensics) / len(forensics)
    model_risk = prediction["fake_probability"]
    quality_penalty = 0 if quality["accepted"] else 0.2
    risk_score = float(np.clip(max(model_risk, forensic_risk * 0.75) + quality_penalty, 0, 1))
    authenticity_score = float(np.clip(prediction["real_probability"] * (1 - quality_penalty), 0, 1))
    return {
        "authenticity_score": authenticity_score,
        "risk_score": risk_score,
        "confidence_score": prediction["confidence"],
        "quality_score": float(np.clip((quality["blur_score"] / 400 + quality["face_size_ratio"] * 2) / 2, 0, 1)),
        "forensic_risk": forensic_risk,
    }


def compression_score(rgb: np.ndarray) -> float:
    source = Image.fromarray(rgb.astype("uint8"), "RGB")
    buffer = io.BytesIO()
    source.save(buffer, format="JPEG", quality=85, optimize=False)
    buffer.seek(0)
    decoded = np.asarray(Image.open(buffer).convert("RGB"))
    delta = np.mean(np.abs(decoded.astype(float) - rgb.astype(float)))
    return float(np.clip(delta / 18, 0, 1))


def frequency_score(gray: np.ndarray) -> float:
    small = gray
    if max(gray.shape) > 384:
        small_image = Image.fromarray(gray.astype("uint8"), "L")
        small = np.asarray(small_image.resize(_bounded_size((gray.shape[1], gray.shape[0]), 384))).astype(float)
    spectrum = np.fft.fftshift(np.fft.fft2(small))
    magnitude = np.log(np.abs(spectrum) + 1)
    return float(np.clip(np.std(magnitude) / 4.5, 0, 1))


def laplacian_variance(gray: np.ndarray) -> float:
    center = gray[1:-1, 1:-1] * -4
    neighbors = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(np.var(center + neighbors))


def edge_score(gray: np.ndarray) -> float:
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    return float(np.clip(((gx > 30).mean() + (gy > 30).mean()) * 1.6, 0, 1))


def indicator(name: str, score: float, evidence: str) -> dict:
    value = float(score)
    return {
        "name": name,
        "score": value,
        "severity": "high" if value >= 0.67 else "medium" if value >= 0.34 else "low",
        "evidence": evidence,
    }


def _gray_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L").resize(_bounded_size(image.size, 640))).astype(float)


def _bounded_size(size: tuple[int, int], max_side: int) -> tuple[int, int]:
    width, height = size
    scale = min(1.0, max_side / max(width, height))
    return max(1, int(width * scale)), max(1, int(height * scale))
