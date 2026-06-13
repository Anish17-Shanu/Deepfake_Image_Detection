# Model Choice Document

## Selected model

`dima806/deepfake_vs_real_image_detection`

## Reason for selection

This is a public pretrained image-classification checkpoint with a published evaluation score and a straightforward Hugging Face download path. It is more practical for a final-year project than a checkpoint that requires retraining, dataset reconstruction, or a separate training pipeline.

## Why not train from scratch

Training a deepfake detector from scratch is expensive, dataset-dependent, and difficult to reproduce on a normal student laptop. The requirement for this project is immediate usability, so a public pretrained checkpoint is the correct choice.

## Why not the video Xception model

XceptionNet is a strong deepfake baseline, but the public Keras model discovered in the research phase is video-oriented and expects sequences of frames. The requested project is image-based, so a dedicated image classifier is a better fit.

## Public download links

- https://huggingface.co/dima806/deepfake_vs_real_image_detection
- https://huggingface.co/dima806/deepfake_vs_real_image_detection/resolve/main/model.safetensors?download=true
- https://huggingface.co/dima806/deepfake_vs_real_image_detection/resolve/main/config.json?download=true
- https://huggingface.co/dima806/deepfake_vs_real_image_detection/resolve/main/preprocessor_config.json?download=true
