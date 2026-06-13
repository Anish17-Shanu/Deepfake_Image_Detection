from __future__ import annotations

import uuid
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from utils.helper import (
    MODEL_ACCURACY,
    MODEL_CONFIG_URL,
    MODEL_ID,
    MODEL_NAME,
    MODEL_PAGE_URL,
    MODEL_PROCESSOR_URL,
    MODEL_WEIGHT_URL,
    allowed_file,
    append_history,
    build_history_record,
    ensure_directories,
    generate_report_text,
    get_history,
    get_record_by_id,
    save_json,
)
from utils.predictor import DeepfakePredictor


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULTS_DIR = STATIC_DIR / "results"
MODEL_DIR = BASE_DIR / "model"
HISTORY_PATH = RESULTS_DIR / "history.json"


app = Flask(__name__)
app.secret_key = "deepfake-image-detection-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ensure_directories(UPLOAD_DIR, RESULTS_DIR, MODEL_DIR)


_predictor: DeepfakePredictor | None = None


def get_predictor() -> DeepfakePredictor:
    global _predictor
    if _predictor is None:
        _predictor = DeepfakePredictor(MODEL_DIR)
    return _predictor


def result_file_paths(prediction_id: str) -> dict[str, Path]:
    result_dir = RESULTS_DIR / prediction_id
    ensure_directories(result_dir)
    return {
        "dir": result_dir,
        "report_txt": result_dir / "report.txt",
        "report_json": result_dir / "report.json",
        "heatmap": result_dir / "heatmap.jpg",
    }


@app.route("/")
def index():
    latest_history = get_history(HISTORY_PATH)[:3]
    return render_template(
        "index.html",
        model_name=MODEL_NAME,
        model_accuracy=MODEL_ACCURACY,
        latest_history=latest_history,
        model_page_url=MODEL_PAGE_URL,
    )


@app.route("/predict", methods=["POST"])
def predict():
    uploaded_file = request.files.get("image")
    if not uploaded_file or uploaded_file.filename == "":
        flash("Please choose a JPG, JPEG, or PNG image before running prediction.")
        return redirect(url_for("index"))

    if not allowed_file(uploaded_file.filename):
        flash("Unsupported file type. Please upload a JPG, JPEG, or PNG image.")
        return redirect(url_for("index"))

    prediction_id = uuid.uuid4().hex[:12]
    safe_name = secure_filename(uploaded_file.filename)
    stored_filename = f"{prediction_id}_{safe_name}"
    upload_path = UPLOAD_DIR / stored_filename
    uploaded_file.save(upload_path)

    paths = result_file_paths(prediction_id)

    try:
        result = get_predictor().predict(upload_path, heatmap_path=paths["heatmap"])
    except Exception as exc:
        app.logger.exception("Prediction failed")
        flash(f"Prediction failed: {exc}")
        return redirect(url_for("index"))

    upload_url = url_for("static", filename=f"uploads/{stored_filename}")
    result_url = url_for("result", prediction_id=prediction_id)
    heatmap_url = url_for("static", filename=f"results/{prediction_id}/heatmap.jpg") if paths["heatmap"].exists() else None
    report_url = url_for("download_report", prediction_id=prediction_id)

    record = build_history_record(
        prediction_id=prediction_id,
        original_filename=safe_name,
        stored_filename=stored_filename,
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        real_probability=result["real_probability"],
        fake_probability=result["fake_probability"],
        upload_url=upload_url,
        result_url=result_url,
        heatmap_url=heatmap_url,
        report_url=report_url,
        model_name=result["model_name"],
        model_page_url=result["model_page_url"],
    )

    save_json(paths["report_json"], record)
    paths["report_txt"].write_text(generate_report_text(record), encoding="utf-8")
    append_history(HISTORY_PATH, record)

    return redirect(url_for("result", prediction_id=prediction_id))


@app.route("/result/<prediction_id>")
def result(prediction_id: str):
    record = get_record_by_id(HISTORY_PATH, prediction_id)
    if record is None:
        abort(404)
    return render_template("result.html", record=record, model_name=MODEL_NAME)


@app.route("/history")
def history():
    records = get_history(HISTORY_PATH)
    return render_template("history.html", records=records, model_name=MODEL_NAME)


@app.route("/about")
def about():
    return render_template(
        "about.html",
        model_name=MODEL_NAME,
        model_accuracy=MODEL_ACCURACY,
        model_id=MODEL_ID,
        model_page_url=MODEL_PAGE_URL,
        model_weight_url=MODEL_WEIGHT_URL,
        model_config_url=MODEL_CONFIG_URL,
        model_processor_url=MODEL_PROCESSOR_URL,
    )


@app.route("/report/<prediction_id>/download")
def download_report(prediction_id: str):
    report_path = RESULTS_DIR / prediction_id / "report.txt"
    if not report_path.exists():
        abort(404)
    return send_file(report_path, as_attachment=True, download_name=f"deepfake_report_{prediction_id}.txt")


@app.route("/api/latest")
def api_latest():
    records = get_history(HISTORY_PATH)
    return {"count": len(records), "latest": records[:1]}


@app.errorhandler(404)
def not_found(_error):
    return render_template("404.html", model_name=MODEL_NAME), 404


@app.errorhandler(500)
def server_error(_error):
    return render_template("500.html", model_name=MODEL_NAME), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)