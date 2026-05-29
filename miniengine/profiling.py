from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any


_lock = Lock()


def now() -> float:
    return time.time()


def engine_log_path() -> str:
    return os.getenv("MINIENGINE_PROFILE_LOG", "logs/engine_events.jsonl")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def log_jsonl(obj: dict[str, Any]) -> None:
    path = Path(engine_log_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {key: _jsonable(value) for key, value in obj.items() if value is not None}
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def request_profile_fields(req: Any) -> dict[str, Any]:
    metadata = getattr(req, "profile_metadata", {}) or {}
    fields = {
        "request_id": getattr(req, "request_id", None),
        "run_id": metadata.get("run_id"),
        "workflow_id": metadata.get("workflow_id"),
        "task_id": metadata.get("task_id"),
        "sample_index": metadata.get("sample_index"),
        "step_id": metadata.get("step_id"),
        "agent_type": metadata.get("agent_type"),
        "workload": metadata.get("workload"),
        "call_type": metadata.get("call_type"),
        "is_optional": metadata.get("is_optional"),
        "is_critical": metadata.get("is_critical"),
        "branch_id": metadata.get("branch_id"),
        "attempt_id": metadata.get("attempt_id"),
    }
    return {key: value for key, value in fields.items() if value is not None}


def log_request_event(req: Any, event: str, **extra: Any) -> None:
    log_jsonl(
        {
            "event": event,
            "ts": now(),
            **request_profile_fields(req),
            **extra,
        }
    )
