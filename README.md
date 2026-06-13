# Deepfake Image Detection

Production-ready deepfake image detection project for uploading a face image and receiving:

- REAL / DEEPFAKE prediction from a public ViT checkpoint
- Confidence, real probability, and fake probability
- Authenticity, risk, quality, and forensic indicator stats
- Face count and image quality validation
- Local scan history and downloadable Markdown reports
- Docker and Render deployment support
- Lightweight Vercel landing deployment

## Model And Verified Stats

The app uses the public Hugging Face model:

```text
dima806/deepfake_vs_real_image_detection
```

Model page: https://huggingface.co/dima806/deepfake_vs_real_image_detection

The model card reports **99.27% evaluation accuracy** on **76,161 images**:

- Real precision/recall/F1: 0.9921 / 0.9933 / 0.9927
- Fake precision/recall/F1: 0.9933 / 0.9921 / 0.9927
- Overall accuracy: 0.9927

Important limitation: the same model card warns that the training data is several years old, so newer image generators can cause concept drift. Treat the output as a screening signal, not a legal or forensic verdict.

## Recommended App

Use the FastAPI app in `saas/` for deployment and demos.

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

Health check:

```text
http://127.0.0.1:8000/health
```

## Docker

```bash
cd saas
docker compose up --build
```

Open:

```text
http://127.0.0.1:8000
```

## Deploy To Render

The repository includes `render.yaml`. On Render:

1. Create a new Blueprint from this GitHub repository.
2. Render will use `saas/Dockerfile`.
3. Health check path is `/health`.
4. The app listens on Render's `PORT` environment variable.

The model downloads into `saas/model/` on first inference if it is not already cached.

## Deploy To Vercel

Vercel is included as a lightweight static project page via `vercel.json` and `public/index.html`.

The full analyzer is not deployed on Vercel because PyTorch, OpenCV, and the 343 MB local checkpoint are not a good fit for Vercel serverless limits. Use Render or Docker for the real inference app.

## Legacy Flask App

The original Flask implementation is still available at the repository root:

```bash
python app.py
```

For deployment, prefer the `saas/` FastAPI app.

## Repository Notes

Large model files, uploads, reports, cache folders, and runtime data are ignored by git. This keeps GitHub pushes deployable while preserving runtime model download support.
