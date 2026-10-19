#!/usr/bin/env python3
"""Resolve TensorBoard log directory for a PyTorch-Lightning workflow run.

Reads JOB_DIR / WORKFLOW_ID from the environment (Drona retrieverParams).
Python is tolerant of CRLF, so this stays reliable when the env is synced from Windows.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys


def _resolve_job_dir() -> str:
    job_dir = (os.environ.get("JOB_DIR") or "").strip()
    if job_dir:
        return job_dir

    workflow_id = (os.environ.get("WORKFLOW_ID") or "").strip()
    if not workflow_id:
        return ""

    runtime = (os.environ.get("DRONA_RUNTIME_DIR") or "").strip()
    if not runtime:
        return ""

    retriever = os.path.join(runtime, "db_access", "drona_db_retriever.py")
    try:
        raw = subprocess.check_output(
            [sys.executable, retriever, "-i", workflow_id],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        record = json.loads(raw) if raw.strip() else {}
        return str(record.get("location", "") or "").strip()
    except Exception:
        return ""


def _log_dir_name(job_dir: str) -> str:
    name = "lightning_logs"
    if not job_dir:
        return name
    train_py = os.path.join(job_dir, "train.py")
    if not os.path.isfile(train_py):
        return name
    try:
        with open(train_py, "r", encoding="utf-8") as f:
            content = f.read()
        match = re.search(r'LOG_DIR\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            val = match.group(1)
            if val.startswith("./"):
                val = val[2:]
            return val or name
    except Exception:
        pass
    return name


def main() -> None:
    job_dir = _resolve_job_dir()
    log_dir_name = _log_dir_name(job_dir)

    candidates = []
    if job_dir:
        candidates.append(os.path.join(job_dir, log_dir_name))
        candidates.append(os.path.join(job_dir, "lightning_logs"))

    for path in candidates:
        if path and os.path.isdir(path):
            sys.stdout.write(path)
            return

    if job_dir:
        sys.stdout.write(os.path.join(job_dir, log_dir_name))
    else:
        sys.stdout.write("")


if __name__ == "__main__":
    main()
