"""
Client workload for the Milestone 3 retraction bonus.

Run this against a MiniEngine server configured with a deliberately small KV
pool.  The workload sends many long-generation streaming requests at once, so
prefill should fit but decode-time KV growth can exhaust the pool.

Usage:
    python -m benchmark.bench_retraction_client \
        --num-requests 16 --concurrency 16 --max-tokens 4096

Run the same command twice:
  1. server without --enable-retraction: expect stream/HTTP failures
  2. server with    --enable-retraction: expect completed=N/N
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp


DEFAULT_PROMPT = (
    "Write a very long numbered list. Continue until you are stopped by the "
    "token limit. Do not summarize. Do not end early. Each line should contain "
    "one number and a short sentence about synthetic benchmark data."
)


@dataclass
class Result:
    request_id: int
    status: int | None = None
    ok: bool = False
    chunks: int = 0
    chars: int = 0
    start_time: float = 0.0
    first_token_time: float | None = None
    end_time: float = 0.0
    error: str | None = None

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.start_time) * 1000.0

    @property
    def latency_s(self) -> float:
        return self.end_time - self.start_time


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    idx = min(len(xs) - 1, max(0, int(q * (len(xs) - 1))))
    return xs[idx]


async def run_one(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    args: argparse.Namespace,
    request_id: int,
) -> Result:
    result = Result(request_id=request_id, start_time=time.perf_counter())
    prompt = (
        f"Request {request_id}. {args.prompt} "
        f"Use request id {request_id} in each item so the output is unique."
    )
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
    }
    text_parts: list[str] = []

    async with sem:
        if args.stagger_ms > 0:
            await asyncio.sleep(
                (request_id % args.concurrency) * args.stagger_ms / 1000.0
            )
        try:
            async with session.post(
                f"{args.base_url}/v1/chat/completions", json=payload
            ) as resp:
                result.status = resp.status
                print(f"request {request_id}: status={resp.status}", flush=True)
                if resp.status != 200:
                    body = await resp.text()
                    result.error = f"HTTP {resp.status}: {body[:500]}"
                else:
                    async for raw in resp.content:
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data: "):
                            continue
                        body = line[6:]
                        if body == "[DONE]":
                            result.ok = True
                            break
                        try:
                            data = json.loads(body)
                        except json.JSONDecodeError:
                            continue

                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            if result.first_token_time is None:
                                result.first_token_time = time.perf_counter()
                            result.chunks += 1
                            result.chars += len(content)
                            if args.save_text_dir is not None:
                                text_parts.append(content)

                if resp.status == 200 and not result.ok:
                    result.error = "stream ended before [DONE]"
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            result.end_time = time.perf_counter()

    if args.save_text_dir is not None and text_parts:
        out_dir = Path(args.save_text_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"request_{request_id:03d}.txt").write_text(
            "".join(text_parts), encoding="utf-8"
        )

    status = "complete" if result.ok else "failed"
    print(
        f"request {request_id}: {status} chunks={result.chunks} "
        f"chars={result.chars} seconds={result.latency_s:.2f} "
        f"error={result.error or '-'}",
        flush=True,
    )
    return result


async def async_main(args: argparse.Namespace) -> list[Result]:
    timeout = aiohttp.ClientTimeout(total=args.timeout if args.timeout > 0 else None)
    sem = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [run_one(session, sem, args, i) for i in range(args.num_requests)]
        return await asyncio.gather(*tasks)


def report(results: list[Result], wall_time: float) -> None:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    lats = [r.latency_s for r in ok]
    total_chunks = sum(r.chunks for r in results)

    print("\n" + "=" * 72)
    print("Retraction client summary")
    print(f"  Requests       : {len(results)}")
    print(f"  Completed      : {len(ok)}/{len(results)}")
    print(f"  Failed         : {len(failed)}")
    print(f"  Wall time      : {wall_time:.2f} s")
    print(f"  Output chunks  : {total_chunks}")
    if ok:
        print(f"  Throughput     : {len(ok) / wall_time:.2f} completed req/s")
        if ttfts:
            print(
                f"  TTFT p50 / p99 : {statistics.median(ttfts):.0f} ms / "
                f"{percentile(ttfts, 0.99):.0f} ms"
            )
        print(
            f"  Lat p50 / p99  : {statistics.median(lats):.2f} s / "
            f"{percentile(lats, 0.99):.2f} s"
        )
    if failed:
        print("\nFailures:")
        for r in failed:
            print(f"  request {r.request_id}: status={r.status} error={r.error}")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Long-decode client workload for testing paged KV retraction.",
    )
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default="default")
    p.add_argument("--num-requests", type=int, default=16)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text shared by all requests; request id is appended.",
    )
    p.add_argument(
        "--stagger-ms",
        type=float,
        default=0.0,
        help="Optional per-request launch staggering within each concurrency wave.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Total aiohttp timeout in seconds. 0 means no total timeout.",
    )
    p.add_argument(
        "--save-text-dir",
        default=None,
        help="Optional directory for full streamed text, useful for duplicate checks.",
    )
    args = p.parse_args()
    if args.num_requests <= 0:
        p.error("--num-requests must be positive")
    if args.concurrency <= 0:
        p.error("--concurrency must be positive")
    if args.max_tokens <= 0:
        p.error("--max-tokens must be positive")
    return args


def main() -> None:
    args = parse_args()
    print("Retraction client")
    print(f"  Server       : {args.base_url}")
    print(f"  Requests     : {args.num_requests}")
    print(f"  Concurrency  : {args.concurrency}")
    print(f"  Max tokens   : {args.max_tokens}")
    print(f"  Temperature  : {args.temperature}")
    if args.save_text_dir is not None:
        print(f"  Save text dir: {args.save_text_dir}")

    t0 = time.perf_counter()
    results = asyncio.run(async_main(args))
    report(results, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
