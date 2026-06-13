from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def detect_faces(image: Image.Image) -> dict:
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(48, 48))
    boxes = [[int(x), int(y), int(x + w), int(y + h)] for x, y, w, h in faces]
    return {"face_count": len(boxes), "boxes": boxes}


def quality_stats(image: Image.Image, boxes: list[list[int]]) -> dict:
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    resolution = int(image.width * image.height)
    face_ratio = 0.0
    if boxes:
        largest = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        face_ratio = ((largest[2] - largest[0]) * (largest[3] - largest[1])) / max(1, resolution)
    reasons: list[str] = []
    if not boxes:
        reasons.append("No face detected.")
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
    if boxes and face_ratio < 0.05:
        reasons.append("Face is too small in the image.")
    return {
        "blur_score": blur,
        "brightness": brightness,
        "resolution": resolution,
        "face_size_ratio": face_ratio,
        "accepted": not reasons,
        "reasons": reasons,
    }


def forensic_stats(image: Image.Image) -> list[dict]:
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return [
        indicator("Texture anomalies", np.clip(1 - cv2.Laplacian(gray, cv2.CV_64F).var() / 700, 0, 1), "Micro-texture smoothness measured from local contrast."),
        indicator("Compression inconsistencies", compression_score(rgb), "Residual differences after controlled recompression."),
        indicator("Lighting mismatch", np.clip(np.std([np.mean(rgb[:, :, i]) for i in range(3)]) / 45, 0, 1), "Color-channel illumination imbalance."),
        indicator("Edge artifacts", np.clip(np.mean(cv2.Canny(gray, 80, 160) > 0) * 4, 0, 1), "Abnormal edge density around high-contrast regions."),
        indicator("GAN frequency signal", frequency_score(gray), "Frequency-domain periodicity often seen in generated imagery."),
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
    encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1]
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    delta = np.mean(np.abs(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(float) - rgb.astype(float)))
    return float(np.clip(delta / 18, 0, 1))


def frequency_score(gray: np.ndarray) -> float:
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log(np.abs(spectrum) + 1)
    return float(np.clip(np.std(magnitude) / 4.5, 0, 1))


def indicator(name: str, score: float, evidence: str) -> dict:
    value = float(score)
    return {
        "name": name,
        "score": value,
        "severity": "high" if value >= 0.67 else "medium" if value >= 0.34 else "low",
        "evidence": evidence,
    }
