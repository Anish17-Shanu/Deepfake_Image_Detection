from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import DATA_DIR, HISTORY_PATH, REPORT_DIR, UPLOAD_DIR


def ensure_storage() -> None:
    for directory in (DATA_DIR, UPLOAD_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def save_history(records: list[dict[str, Any]]) -> None:
    HISTORY_PATH.write_text(json.dumps(records[:200], indent=2), encoding="utf-8")


def append_history(record: dict[str, Any]) -> None:
    records = load_history()
    records.insert(0, record)
    save_history(records)


def find_record(scan_id: str) -> dict[str, Any] | None:
    return next((record for record in load_history() if record["scan_id"] == scan_id), None)


def write_report(scan_id: str, report: str) -> Path:
    path = REPORT_DIR / f"{scan_id}.md"
    path.write_text(report, encoding="utf-8")
    return path
