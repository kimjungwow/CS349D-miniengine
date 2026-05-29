"""Replay normalized AgentBench LLM-call traces against MiniEngine.

This benchmark isolates serving-engine behavior from tool execution.  It reads
JSONL produced by ``benchmark.prepare_agentic_trace`` and sends each recorded
LLM call as an OpenAI-compatible chat completion request, carrying AgentBench
workflow metadata through HTTP headers.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import aiohttp


AGENT_HEADER_MAP = {
    "run_id": "x-agentbench-run-id",
    "workflow_id": "x-agentbench-workflow-id",
    "task_id": "x-agentbench-task-id",
    "sample_index": "x-agentbench-sample-index",
    "step_id": "x-agentbench-step-id",
    "agent_type": "x-agentbench-agent-type",
    "workload": "x-agentbench-workload",
    "call_type": "x-agentbench-call-type",
    "is_optional": "x-agentbench-is-optional",
    "is_critical": "x-agentbench-is-critical",
    "branch_id": "x-agentbench-branch-id",
    "attempt_id": "x-agentbench-attempt-id",
}


@dataclass
class ReplayResult:
    record_id: str
    workflow_id: str
    step_id: str
    agent_type: str
    workload: str
    call_type: str
    is_optional: bool
    is_critical: bool
    status: int
    error: str
    start_time: float
    first_token_time: float | None
    end_time: float
    output_chunks: int
    prompt_tokens: int
    completion_tokens: int
    cache_hit_tokens: int

    @property
    def latency(self) -> float:
        return self.end_time - self.start_time

    @property
    def ttft(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.start_time

    @property
    def cache_hit_ratio(self) -> float:
        if self.prompt_tokens <= 0:
            return 0.0
        return self.cache_hit_tokens / self.prompt_tokens


def _split_csv(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def _header_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)


def _headers(record: dict[str, Any]) -> dict[str, str]:
    return {
        header: _header_value(record[key])
        for key, header in AGENT_HEADER_MAP.items()
        if record.get(key) not in (None, "")
    }


def load_trace(
    path: Path,
    workloads: set[str],
    agent_types: set[str],
    max_calls: int | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            workload = str(record.get("workload", "")).lower()
            agent_type = str(record.get("agent_type", "")).lower()
            if workload not in workloads or agent_type not in agent_types:
                continue
            record.setdefault("record_id", f"{path.name}:{line_no}")
            out.append(record)
            if max_calls is not None and len(out) >= max_calls:
                break
    return out


async def replay_one(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    record: dict[str, Any],
) -> ReplayResult:
    payload = {
        "model": model,
        "messages": record["messages"],
        "max_tokens": int(record.get("max_tokens") or 1),
        "temperature": 0,
        "stream": True,
    }
    start = time.perf_counter()
    first_token: float | None = None
    end = start
    output_chunks = 0
    usage: dict[str, Any] = {}
    status = 0
    error = ""

    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers=_headers(record),
        ) as resp:
            status = resp.status
            if resp.status != 200:
                error = await resp.text()
            else:
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    body = line[6:]
                    if body == "[DONE]":
                        break
                    data = json.loads(body)
                    if "usage" in data:
                        usage = data["usage"]
                    delta = data["choices"][0].get("delta", {})
                    if delta.get("content"):
                        output_chunks += 1
                        if first_token is None:
                            first_token = time.perf_counter()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        end = time.perf_counter()

    return ReplayResult(
        record_id=str(record.get("record_id", "")),
        workflow_id=str(record.get("workflow_id", "")),
        step_id=str(record.get("step_id", "")),
        agent_type=str(record.get("agent_type", "")),
        workload=str(record.get("workload", "")),
        call_type=str(record.get("call_type", "")),
        is_optional=bool(record.get("is_optional", False)),
        is_critical=bool(record.get("is_critical", False)),
        status=status,
        error=error,
        start_time=start,
        first_token_time=first_token,
        end_time=end,
        output_chunks=output_chunks,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        cache_hit_tokens=int(usage.get("cache_hit_tokens") or 0),
    )


async def run_closed_loop(
    records: list[dict[str, Any]],
    base_url: str,
    model: str,
    concurrency: int,
) -> list[ReplayResult]:
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for record in records:
        await queue.put(record)

    results: list[ReplayResult] = []
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def worker() -> None:
            while True:
                try:
                    record = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                result = await replay_one(session, base_url, model, record)
                results.append(result)
                done = len(results)
                if done % 10 == 0 or done == len(records):
                    print(f"  completed {done}/{len(records)}", flush=True)

        await asyncio.gather(*(worker() for _ in range(max(1, concurrency))))
    return results


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((p / 100.0) * (len(values) - 1))))
    return values[idx]


def _summarize_group(label: str, results: list[ReplayResult]) -> None:
    ok = [r for r in results if r.status == 200 and not r.error]
    ttfts = [r.ttft for r in ok if r.ttft is not None]
    latencies = [r.latency for r in ok]
    prompt_tokens = sum(r.prompt_tokens for r in ok)
    completion_tokens = sum(r.completion_tokens for r in ok)
    cache_hits = sum(r.cache_hit_tokens for r in ok)
    hit_ratio = cache_hits / prompt_tokens if prompt_tokens else 0.0
    print(
        f"{label:<24} calls={len(results):>4} ok={len(ok):>4} "
        f"ttft_p50={_percentile(ttfts, 50):>7.3f}s "
        f"ttft_p90={_percentile(ttfts, 90):>7.3f}s "
        f"lat_p50={_percentile(latencies, 50):>7.3f}s "
        f"cache_hit={hit_ratio:>6.2%} "
        f"out_tok={completion_tokens}",
        flush=True,
    )


def print_summary(results: list[ReplayResult], wall_time: float) -> None:
    ok = [r for r in results if r.status == 200 and not r.error]
    print("")
    print("=" * 72)
    print("Agentic Replay Summary")
    print(f"Total calls: {len(results)}")
    print(f"Successful: {len(ok)}")
    print(f"Wall time: {wall_time:.2f} sec")
    if wall_time > 0:
        print(f"Request throughput: {len(ok) / wall_time:.2f} req/s")
        print(
            "Generation throughput: "
            f"{sum(r.completion_tokens for r in ok) / wall_time:.2f} tok/s"
        )
    _summarize_group("all", results)

    by_call_type: dict[str, list[ReplayResult]] = {}
    for result in results:
        by_call_type.setdefault(result.call_type or "unknown", []).append(result)
    print("")
    print("By call_type:")
    for call_type in sorted(by_call_type):
        _summarize_group(call_type, by_call_type[call_type])


def write_csv(results: list[ReplayResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "workflow_id",
        "step_id",
        "agent_type",
        "workload",
        "call_type",
        "is_optional",
        "is_critical",
        "status",
        "error",
        "latency",
        "ttft",
        "output_chunks",
        "prompt_tokens",
        "completion_tokens",
        "cache_hit_tokens",
        "cache_hit_ratio",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["latency"] = result.latency
            row["ttft"] = result.ttft
            row["cache_hit_ratio"] = result.cache_hit_ratio
            writer.writerow({field: row.get(field) for field in fieldnames})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay AgentBench traces to MiniEngine.")
    p.add_argument("--trace-path", required=True)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default="default")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max-calls", type=int, default=None)
    p.add_argument("--workloads", default="math,humaneval")
    p.add_argument("--agent-types", default="react,reflexion,lats")
    p.add_argument("--output", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = load_trace(
        Path(args.trace_path),
        workloads=_split_csv(args.workloads),
        agent_types=_split_csv(args.agent_types),
        max_calls=args.max_calls,
    )
    if not records:
        raise SystemExit("No replay records matched the requested filters.")

    print(
        f"Replaying {len(records)} calls to {args.base_url} "
        f"(concurrency={args.concurrency})",
        flush=True,
    )
    start = time.perf_counter()
    results = asyncio.run(
        run_closed_loop(
            records=records,
            base_url=args.base_url.rstrip("/"),
            model=args.model,
            concurrency=args.concurrency,
        )
    )
    wall_time = time.perf_counter() - start
    print_summary(results, wall_time)
    if args.output:
        write_csv(results, Path(args.output))
        print(f"Wrote replay results to {args.output}")


if __name__ == "__main__":
    main()
