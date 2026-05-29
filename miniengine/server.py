"""
OpenAI-compatible HTTP server.

Endpoints:
  POST /v1/chat/completions   — chat completion (streaming & non-streaming)
  GET  /v1/models             — list available models
  GET  /health                — liveness check

The server is intentionally thin — it only translates HTTP ↔ internal
Request objects.  All scheduling and model logic lives elsewhere.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from miniengine.core import Request as EngineRequest
from miniengine.core import SamplingParams, TokenOutput
from miniengine.engine import Engine
from miniengine.profiling import log_request_event, now
from miniengine.scheduler import Scheduler

logger = logging.getLogger(__name__)

# ── FastAPI app ─────────────────────────────────────────────────────────

app = FastAPI(title="MiniEngine", version="0.1.0")

# These are set by __main__.py before the server starts.
engine: Engine | None = None
scheduler: Scheduler | None = None
model_id: str = "unknown"

# ── Request / Response schemas (OpenAI-compatible subset) ───────────────


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=256)
    temperature: float | None = Field(default=0.6)
    top_p: float | None = Field(default=0.95)
    top_k: int | None = Field(default=20)
    repetition_penalty: float | None = Field(default=1.0)
    stream: bool = False


# ── Endpoints ───────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/cache_stats")
async def cache_stats():
    """Snapshot of radix-cache effectiveness counters.

    Returns ``{"enabled": False}`` if the engine doesn't have a radix
    cache wired up yet — students see this until they implement Part B.
    """
    cache = getattr(engine, "radix_cache", None) if engine else None
    if cache is None:
        return {"enabled": False}
    m = cache.metrics
    pool = engine.pool
    return {
        "enabled": True,
        "hit_rate": m.hit_rate,
        "total_lookups": m.total_lookups,
        "total_query_tokens": m.total_query_tokens,
        "total_hit_tokens": m.total_hit_tokens,
        "total_inserted_pages": m.total_inserted_pages,
        "total_evicted_pages": m.total_evicted_pages,
        "num_cached_pages": getattr(cache, "num_cached_pages", 0),
        "pool_num_free": pool.num_free if pool is not None else 0,
        "pool_num_evictable": getattr(pool, "num_evictable", 0),
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "miniengine",
            }
        ],
    }


def _bool_header(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.lower() in {"1", "true", "yes", "y"}


def _profile_metadata(http_request: FastAPIRequest) -> dict:
    headers = http_request.headers
    metadata = {
        "run_id": headers.get("x-agentbench-run-id"),
        "workflow_id": headers.get("x-agentbench-workflow-id"),
        "task_id": headers.get("x-agentbench-task-id"),
        "sample_index": headers.get("x-agentbench-sample-index"),
        "step_id": headers.get("x-agentbench-step-id"),
        "agent_type": headers.get("x-agentbench-agent-type"),
        "workload": headers.get("x-agentbench-workload"),
        "call_type": headers.get("x-agentbench-call-type"),
        "is_optional": _bool_header(headers.get("x-agentbench-is-optional")),
        "is_critical": _bool_header(headers.get("x-agentbench-is-critical")),
        "branch_id": headers.get("x-agentbench-branch-id"),
        "attempt_id": headers.get("x-agentbench-attempt-id"),
    }
    return {key: value for key, value in metadata.items() if value is not None}


@app.post("/v1/chat/completions")
async def chat_completions(raw: ChatCompletionRequest, http_request: FastAPIRequest):
    assert engine is not None and scheduler is not None

    # Tokenize the conversation using the model's chat template
    messages = [{"role": m.role, "content": m.content} for m in raw.messages]
    input_ids = engine.tokenize_messages(messages)

    sampling_params = SamplingParams(
        max_new_tokens=raw.max_tokens or 256,
        temperature=raw.temperature if raw.temperature is not None else 0.6,
        top_p=raw.top_p if raw.top_p is not None else 0.95,
        top_k=raw.top_k if raw.top_k is not None else 20,
        repetition_penalty=(
            raw.repetition_penalty if raw.repetition_penalty is not None else 1.0
        ),
    )

    req = EngineRequest(
        request_id=str(uuid.uuid4()),
        input_ids=input_ids,
        sampling_params=sampling_params,
        profile_metadata=_profile_metadata(http_request),
    )
    req.profile_received_ts = now()
    log_request_event(
        req,
        "request_received",
        prompt_tokens=req.num_input_tokens,
        max_new_tokens=sampling_params.max_new_tokens,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
        stream=raw.stream,
    )
    scheduler.add_request(req)

    if raw.stream:
        return StreamingResponse(
            _stream_response(req, raw.model or model_id),
            media_type="text/event-stream",
        )

    # Non-streaming: collect full response
    full_text, usage = await _collect_full_response(req)
    return _make_completion_response(
        req.request_id, raw.model or model_id, full_text, usage
    )


# ── Streaming helpers ───────────────────────────────────────────────────


async def _stream_response(req: EngineRequest, model: str) -> AsyncGenerator[str, None]:
    """Yield SSE chunks as tokens arrive from the scheduler.

    The final chunk includes ``usage`` with prefix-cache hit information
    so clients can record cache effectiveness per request.
    """
    loop = asyncio.get_event_loop()
    while True:
        output: TokenOutput = await loop.run_in_executor(None, req.token_queue.get)
        if output.finished:
            chunk = _make_stream_chunk(
                req.request_id,
                model,
                content="",
                finish_reason="stop",
                usage=_usage_from_request(req),
            )
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return
        chunk = _make_stream_chunk(req.request_id, model, content=output.token_text)
        yield f"data: {json.dumps(chunk)}\n\n"


async def _collect_full_response(req: EngineRequest) -> tuple[str, dict]:
    """Block until the request finishes; return (text, usage dict)."""
    loop = asyncio.get_event_loop()
    parts: list[str] = []
    while True:
        output: TokenOutput = await loop.run_in_executor(None, req.token_queue.get)
        if output.finished:
            return "".join(parts), _usage_from_request(req)
        parts.append(output.token_text)


def _usage_from_request(req: EngineRequest) -> dict:
    """Build the OpenAI-style ``usage`` block straight off the request.

    ``cache_hit_tokens`` is our extension — number of prompt tokens
    served from the radix prefix cache (page-aligned).
    """
    p = req.num_input_tokens
    c = req.num_output_tokens
    return {
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": p + c,
        "cache_hit_tokens": req.cache_hit_tokens,
    }


# ── Response builders ───────────────────────────────────────────────────


def _make_stream_chunk(
    request_id: str,
    model: str,
    content: str,
    finish_reason: str | None = None,
    usage: dict | None = None,
) -> dict:
    chunk: dict = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _make_completion_response(
    request_id: str, model: str, text: str, usage: dict | None = None
) -> dict:
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage
        or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_hit_tokens": 0,
        },
    }
