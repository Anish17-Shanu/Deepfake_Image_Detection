# Free Local Deepfake Analysis Web App

This is the simplified production-grade version of the deepfake image analyzer. It runs locally, uses free open-source libraries, stores data on disk, and does not require paid services, cloud accounts, external databases, queues, or authentication.

## What It Includes

- Upload JPEG, PNG, or WEBP face images
- Face detection and face count
- Image quality validation
- Real deepfake inference with the public ViT model
- Authenticity, confidence, risk, and quality scores
- Local forensic indicators
- Local scan history
- Downloadable Markdown reports
- Docker support
- Render deployment through the repository `render.yaml`
- Lightweight Vercel landing deployment from the repository root

## Verified Model Stats

The Hugging Face model card reports 99.27% evaluation accuracy on 76,161 images. It also warns that the training data is several years old, so newer AI generators can reduce real-world reliability through concept drift.

The app therefore reports authenticity and risk as screening scores based on model probability, quality checks, and local forensic indicators. They are not a legal or forensic verdict.

## What Was Removed

- JWT authentication
- RBAC
- PostgreSQL
- Redis
- Celery
- Cloudinary / S3
- Prometheus / Grafana / OpenTelemetry
- Paid-service integration points

## Run Locally

```bash
cd saas
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m app.main
```

Open:

```text
http://127.0.0.1:8000
```

## Run With Docker

```bash
cd saas
docker compose up --build
```

Open:

```text
http://127.0.0.1:8000
```

## Model

The app uses the free public model:

```text
dima806/deepfake_vs_real_image_detection
```

On first analysis, the model is downloaded into the local `saas/model/` folder if the full checkpoint is missing. After that, inference can run from the local files.
