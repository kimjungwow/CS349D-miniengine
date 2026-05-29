"""Normalize AgentBench joined-call CSVs into replayable JSONL traces.

The AgentBench profiling artifacts already contain the serving-shape fields we
need for MiniEngine replay: workflow ids, call types, optional/critical flags,
prompt/completion token counts, and engine cache statistics.  They do not always
include raw prompts, so this converter supports two modes:

* If a row contains ``messages`` or ``prompt_text``, preserve it.
* Otherwise, synthesize a deterministic prompt with a workflow-shared prefix and
  a call-specific suffix sized from the observed prompt-token count.

The synthetic path is intentional for serving-engine experiments: it preserves
agentic repeated-prefix pressure without requiring AgentBench runtime tools.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_WORKLOADS = ("math", "humaneval")
DEFAULT_AGENT_TYPES = ("react", "reflexion", "lats")


def _split_csv(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_bool(value: Any, default: bool = False) -> bool:
    text = _clean(value).lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y"}


def _to_int(value: Any, default: int = 0) -> int:
    text = _clean(value)
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    text = _clean(value)
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _repeat_words(seed: list[str], target_words: int) -> str:
    target_words = max(1, target_words)
    words: list[str] = []
    while len(words) < target_words:
        words.extend(seed)
    return " ".join(words[:target_words])


def _short_id(value: Any, fallback: str) -> str:
    text = _clean(value)
    return text.replace("/", "_").replace(":", "_") if text else fallback


def _synthetic_messages(row: dict[str, str], prompt_tokens: int) -> list[dict[str, str]]:
    """Build a deterministic prompt with an agentic shared-prefix shape."""

    workflow_id = _short_id(row.get("workflow_id"), "workflow")
    task_id = _short_id(row.get("task_id"), "task")
    step_id = _short_id(row.get("step_id"), "step")
    agent_type = _short_id(row.get("agent_type"), "agent")
    workload = _short_id(row.get("workload"), "workload")
    call_type = _short_id(row.get("call_type"), "llm_call")
    branch_id = _short_id(row.get("branch_id"), "main")
    attempt_id = _short_id(row.get("attempt_id"), "first")
    observed_hit = _to_int(row.get("cache_hit_tokens"), 0)

    if prompt_tokens <= 24:
        shared_words = max(4, prompt_tokens // 2)
    else:
        shared_words = min(prompt_tokens - 8, max(observed_hit, prompt_tokens // 2))
    suffix_words = max(4, prompt_tokens - shared_words)

    shared_seed = [
        "AgentBench",
        "workflow",
        workflow_id,
        "task",
        task_id,
        "agent",
        agent_type,
        "workload",
        workload,
        "shared",
        "scratchpad",
        "context",
    ]
    suffix_seed = [
        "step",
        step_id,
        "call",
        call_type,
        "branch",
        branch_id,
        "attempt",
        attempt_id,
        "request",
        "specific",
        "details",
    ]
    prompt = (
        _repeat_words(shared_seed, shared_words)
        + "\n\n"
        + _repeat_words(suffix_seed, suffix_words)
    )
    return [{"role": "user", "content": prompt}]


def _messages_from_row(row: dict[str, str], prompt_tokens: int) -> list[dict[str, str]]:
    messages_raw = _clean(row.get("messages"))
    if messages_raw:
        try:
            messages = json.loads(messages_raw)
            if isinstance(messages, list):
                return messages
        except json.JSONDecodeError:
            pass

    prompt_text = _clean(row.get("prompt_text"))
    if prompt_text:
        return [{"role": "user", "content": prompt_text}]

    return _synthetic_messages(row, prompt_tokens)


def _record_from_row(row: dict[str, str], source: str, index: int) -> dict[str, Any]:
    prompt_tokens = _to_int(row.get("prompt_tokens"), 0)
    if prompt_tokens <= 0:
        prompt_tokens = _to_int(row.get("prompt_tokens_est"), 256)

    completion_tokens = _to_int(row.get("completion_tokens"), 0)
    if completion_tokens <= 0:
        completion_tokens = _to_int(row.get("output_tokens_est"), 64)
    max_tokens = max(1, completion_tokens)

    run_id = _clean(row.get("run_id")) or Path(source).stem
    workflow_id = _clean(row.get("workflow_id")) or f"workflow-{index}"
    step_id = _clean(row.get("step_id")) or str(index)

    return {
        "record_id": f"{run_id}:{workflow_id}:{step_id}:{index}",
        "source_csv": source,
        "source_index": index,
        "run_id": run_id,
        "workflow_id": workflow_id,
        "task_id": _clean(row.get("task_id")),
        "sample_index": _to_int(row.get("sample_index"), index),
        "step_id": step_id,
        "agent_type": _clean(row.get("agent_type")).lower(),
        "workload": _clean(row.get("workload")).lower(),
        "call_type": _clean(row.get("call_type")) or "llm_call",
        "is_optional": _to_bool(row.get("is_optional")),
        "is_critical": _to_bool(row.get("is_critical")),
        "branch_id": _clean(row.get("branch_id")),
        "attempt_id": _clean(row.get("attempt_id")),
        "messages": _messages_from_row(row, prompt_tokens),
        "max_tokens": max_tokens,
        "observed_prompt_tokens": prompt_tokens,
        "observed_completion_tokens": completion_tokens,
        "observed_cache_hit_tokens": _to_int(row.get("cache_hit_tokens"), 0),
        "observed_cache_hit_ratio": _to_float(row.get("cache_hit_ratio"), 0.0),
        "original_llm_call_start_ts": _to_float(row.get("llm_call_start_ts"), 0.0),
        "original_agent_api_latency": _to_float(row.get("agent_api_latency"), 0.0),
    }


def load_records(
    input_csvs: list[Path],
    workloads: set[str],
    agent_types: set[str],
    max_calls: int | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for csv_path in input_csvs:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for index, row in enumerate(reader):
                workload = _clean(row.get("workload")).lower()
                agent_type = _clean(row.get("agent_type")).lower()
                if workload not in workloads or agent_type not in agent_types:
                    continue
                out.append(_record_from_row(row, str(csv_path), index))
                if max_calls is not None and len(out) >= max_calls:
                    return _sort_records(out)
    return _sort_records(out)


def _sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def step_value(row: dict[str, Any]) -> int:
        return _to_int(row.get("step_id"), 0)

    return sorted(
        records,
        key=lambda row: (
            float(row.get("original_llm_call_start_ts") or 0.0),
            str(row.get("workflow_id") or ""),
            int(row.get("sample_index") or 0),
            step_value(row),
            int(row.get("source_index") or 0),
        ),
    )


def write_jsonl(records: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare AgentBench joined-call CSVs for MiniEngine replay."
    )
    p.add_argument("--input-csv", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workloads", default=",".join(DEFAULT_WORKLOADS))
    p.add_argument("--agent-types", default=",".join(DEFAULT_AGENT_TYPES))
    p.add_argument("--max-calls", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(
        input_csvs=[Path(path) for path in args.input_csv],
        workloads=_split_csv(args.workloads),
        agent_types=_split_csv(args.agent_types),
        max_calls=args.max_calls,
    )
    write_jsonl(records, Path(args.output))
    print(f"Wrote {len(records)} replay records to {args.output}")


if __name__ == "__main__":
    main()
