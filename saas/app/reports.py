from __future__ import annotations


def build_report(record: dict) -> str:
    lines = [
        "# Deepfake Image Analysis Report",
        "",
        f"Scan ID: `{record['scan_id']}`",
        f"Filename: `{record['filename']}`",
        f"Timestamp: `{record['timestamp']}`",
        f"Prediction: `{record['prediction']['label']}`",
        f"Confidence: `{percent(record['prediction']['confidence'])}`",
        f"Real Probability: `{percent(record['prediction']['real_probability'])}`",
        f"Deepfake Probability: `{percent(record['prediction']['fake_probability'])}`",
        f"Authenticity Score: `{percent(record['scores']['authenticity_score'])}`",
        f"Risk Score: `{percent(record['scores']['risk_score'])}`",
        f"Quality Score: `{percent(record['scores']['quality_score'])}`",
        f"Model Source: `{record['model_stats']['model_page_url']}`",
        f"Reported Evaluation Accuracy: `{percent(record['model_stats']['reported_accuracy'])}` on {record['model_stats']['evaluation_support']} images",
        "",
        "## Authenticity Notes",
        record["model_stats"]["limitation"],
        "Authenticity and risk scores combine model probability, quality checks, and local forensic indicators. They are screening signals, not a legal or forensic verdict.",
        "",
        "## Face Detection",
        f"- Faces detected: {record['face_detection']['face_count']}",
        "",
        "## Quality Validation",
        *(f"- {reason}" for reason in record["quality"]["reasons"]),
        "",
        "## Forensic Indicators",
        *(f"- {item['name']}: {percent(item['score'])} ({item['severity']}) - {item['evidence']}" for item in record["forensics"]),
    ]
    return "\n".join(lines)


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"
