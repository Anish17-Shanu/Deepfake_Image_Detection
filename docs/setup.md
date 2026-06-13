# Setup Guide

## Windows commands

```bash
cd Deepfake_Image_Detection
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## What happens on first run

The app checks the `model/` folder and downloads the pretrained checkpoint if the files are missing. After that, predictions run locally on your machine.

## If the model download is slow

The checkpoint is large, so the first startup may take time. After the model is cached locally, future runs are much faster.

## Supported input

- JPG
- JPEG
- PNG

## Output files

- Uploaded image copy in `static/uploads/`
- Result report in `static/results/<prediction_id>/report.txt`
- JSON record in `static/results/<prediction_id>/report.json`
- Grad-CAM heatmap in `static/results/<prediction_id>/heatmap.jpg`
