"""Audit trail (alignment_plan SEC-09): who did what, written as one JSON object per line.

Queries are recorded as a hash plus their length, not as text, so the audit log does not become a second
copy of sensitive content. Super-role activity is flagged so it can be reviewed.
"""
import hashlib
import json
import os
import threading

from settings import get_settings
from datetime import datetime, timezone

DEFAULT_PATH = get_settings().audit_log_path


def text_fingerprint(text: str) -> dict:
    return {"sha256": hashlib.sha256(text.encode()).hexdigest()[:16], "chars": len(text)}


class AuditLog:
    def __init__(self, path=DEFAULT_PATH):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def log(self, event: str, **fields):
        row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, **fields}
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row
