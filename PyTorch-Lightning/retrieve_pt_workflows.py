#!/usr/bin/env python3
"""List PyTorch-Lightning runs from Drona job history for monitor dynamicSelect."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime


def format_submitted_at(value) -> str:
    if not value:
        return "unknown"
    s = str(value).strip().replace("Z", "").split("+")[0].split(".")[0]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    if len(s) >= 19:
        return s[:19].replace("T", " ")
    return s[:10] if s else "unknown"


def main() -> None:
    env_name = os.environ.get("DRONA_ENV_NAME") or os.environ.get("ENV_NAME") or "PyTorch-Lightning"
    runtime = (os.environ.get("DRONA_RUNTIME_DIR") or "").strip()
    records = []

    if runtime:
        retriever = os.path.join(runtime, "db_access", "drona_db_retriever.py")
        try:
            raw = subprocess.check_output(
                [sys.executable, retriever, "-e", env_name],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            parsed = json.loads(raw) if raw.strip() else []
            records = parsed if isinstance(parsed, list) else []
        except Exception:
            records = []

    candidates = []
    for r in records:
        if not isinstance(r, dict):
            continue

        drona_id = r.get("drona_id", "")
        if not drona_id:
            continue

        record_env = r.get("env") or r.get("environment") or r.get("env_name") or ""
        if record_env and record_env != env_name:
            continue

        location = (r.get("location") or "").strip()
        if not location:
            continue
        location = os.path.expandvars(os.path.expanduser(location))
        if not os.path.isdir(location):
            continue

        name = r.get("name") or "unnamed"
        submitted_at = format_submitted_at(r.get("start_time") or "")
        status = r.get("status") or "submitted"
        label = f"{name}  ({submitted_at})  [{status}]"
        start_time = r.get("start_time") or ""
        candidates.append((start_time, drona_id, label))

    candidates.sort(key=lambda item: item[0], reverse=True)
    options = [{"value": drona_id, "label": label} for _, drona_id, label in candidates]
    print(json.dumps(options))


if __name__ == "__main__":
    main()
