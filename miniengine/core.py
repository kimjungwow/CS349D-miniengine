"""
Core data structures for the serving engine.

These structures flow through the entire system:
  Client request → Server → Scheduler → Engine → Scheduler → Server → Client response

Key types:
  - SamplingParams: controls token generation (temperature, top_k, top_p, etc.)
  - Request: tracks the full lifecycle of a single generation request
  - TokenOutput: a single streamed token delivered to the client
"""

import queue
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class RequestStatus(Enum):
    """Lifecycle states for a generation request."""

    WAITING = auto()  # Queued, not yet scheduled
    RUNNING = auto()  # Actively being prefilled or decoded
    FINISHED = auto()  # Generation complete


@dataclass
class SamplingParams:
    """Parameters that control how the next token is sampled."""

    max_new_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    repetition_penalty: float = 1.0
    # If temperature == 0 → greedy decoding (argmax)


@dataclass
class Request:
    """
    A single generation request flowing through the engine.

    The scheduler moves requests through WAITING → RUNNING → FINISHED.
    During RUNNING, the engine fills output_ids one token at a time and
    pushes TokenOutput objects into token_queue for streaming.
    """

    request_id: str
    input_ids: list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    status: RequestStatus = RequestStatus.WAITING
    arrival_time: float = field(default_factory=time.time)

    # Per-request KV cache (set by Engine during prefill, updated on each decode)
    kv_cache: Any = None

    # Streaming output channel — scheduler pushes, server consumes
    token_queue: queue.Queue = field(default_factory=queue.Queue)

    # ── Milestone 3 perf counters (set by scheduler; read by server) ───
    # Number of prompt tokens served from the radix prefix cache.  Stays
    # 0 when the cache is disabled or the lookup missed.  Surfaced in
    # the response's ``usage`` block so clients can record per-request
    # cache effectiveness.
    cache_hit_tokens: int = 0

    # ── Retraction bookkeeping (paged mode only) ──────────────────────
    # ``needs_rehydrate`` means the scheduler retracted this request's
    # paged KV state after some output had already been streamed.  The
    # engine must rebuild KV for ``input_ids + output_ids[:-1]`` without
    # sampling or streaming another token before decode resumes.
    needs_rehydrate: bool = False
    num_retractions: int = 0

    # ── AgentBench profiling metadata/timestamps ─────────────────────
    profile_metadata: dict[str, Any] = field(default_factory=dict)
    profile_received_ts: float | None = None
    profile_scheduled_ts: float | None = None
    profile_prefill_start_ts: float | None = None
    profile_prefill_end_ts: float | None = None
    profile_first_token_ts: float | None = None
    profile_decode_end_ts: float | None = None
    profile_finished_ts: float | None = None

    # ── Scheduler policy observability (Milestone 4) ─────────────────
    # Populated by agent-aware scheduling before admission. These fields
    # are diagnostic only; they do not affect sampling or response shape.
    estimated_cache_hit_tokens: int = 0
    scheduler_priority: float = 0.0

    # ── Derived properties ─────────────────────────────────────────────

    @property
    def num_input_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_ids)

    @property
    def is_finished(self) -> bool:
        return self.num_output_tokens >= self.sampling_params.max_new_tokens


@dataclass
class TokenOutput:
    """One streamed token pushed from the scheduler to the server."""

    token_id: int
    token_text: str
    finished: bool
