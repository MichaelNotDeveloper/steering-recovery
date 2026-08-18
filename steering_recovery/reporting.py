from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable


def load_prompt_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    elif path.suffix == ".json":
        payload = json.loads(path.read_text())
        records = payload if isinstance(payload, list) else payload.get("records", [])
    else:
        records = [
            {"id": f"line-{index}", "prompt": line}
            for index, line in enumerate(path.read_text().splitlines())
            if line.strip()
        ]
    normalized = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("prompt"), str):
            raise ValueError(f"record {index} must be an object with a string 'prompt'")
        normalized.append({"id": str(record.get("id", index)), **record})
    if not normalized:
        raise ValueError("prompt file contains no records")
    return normalized


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def write_html_report(
    rows: list[dict[str, Any]], path: str | Path, settings: dict[str, Any]
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    setting_text = ", ".join(f"{key}={value!r}" for key, value in settings.items())
    articles = []
    for row in rows:
        prompt = str(row["prompt"])
        context = prompt[-400:]
        continuation = str(row["generated_text"])
        meta = (
            f"seed={row['seed']} · interventions={row['intervention_steps']}"
            f"/{row['forward_calls']}"
        )
        articles.append(
            "<article><h2>"
            + html.escape(str(row["prompt_id"]))
            + "</h2><div class='setting'>"
            + html.escape(setting_text)
            + "</div><pre><span class='context'>"
            + html.escape(context)
            + "</span>"
            + html.escape(continuation)
            + "</pre><small>"
            + html.escape(meta)
            + "</small></article>"
        )
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Steering baseline report</title><style>
body{font:15px/1.5 system-ui;max-width:1100px;margin:32px auto;padding:0 16px;background:#f4f5f7;color:#18212f}
article{background:white;border:1px solid #d9dee6;border-radius:12px;padding:18px;margin:14px 0}
h1,h2{margin:0 0 8px}.setting,small{color:#667085}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#fbfcfd;padding:14px;border-radius:8px}.context{color:#c62828;background:#ffebee}
</style></head><body><h1>Steering baseline report</h1>"""
    document += f"<p>{len(rows)} samples · {html.escape(setting_text)}</p>"
    document += "".join(articles) + "</body></html>"
    path.write_text(document, encoding="utf-8")
    return path
