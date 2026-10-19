#!/usr/bin/env python3
"""Emit the job output location for $DRONA_ID (monitor mode hidden field)."""
from __future__ import annotations

import json
import os
import subprocess
import sys


def main() -> None:
    drona_id = (os.environ.get("DRONA_ID") or "").strip()
    runtime = (os.environ.get("DRONA_RUNTIME_DIR") or "").strip()
    location = ""

    if drona_id and runtime:
        retriever = os.path.join(runtime, "db_access", "drona_db_retriever.py")
        try:
            raw = subprocess.check_output(
                [sys.executable, retriever, "-i", drona_id],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            record = json.loads(raw) if raw.strip() else {}
            location = str(record.get("location", "") or "")
        except Exception:
            location = ""

    sys.stdout.write(location)


if __name__ == "__main__":
    main()
