"""
Request scheduler — the core orchestrator of the serving engine.

The scheduler sits between the HTTP server and the model engine:

    Server  ──add_request()──▶  Scheduler  ──prefill/decode──▶  Engine
      ▲                            │
      └─── token_queue (stream) ◄──┘

It runs in a background thread, repeatedly calling step() which:
  1. Admits waiting requests and prefills them  (WAITING → RUNNING)
  2. Runs one decode step on every running request
  3. Retires finished requests                    (RUNNING → FINISHED)

"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

from miniengine.core import Request, RequestStatus, TokenOutput
from miniengine.engine import Engine
from miniengine.profiling import log_request_event, now

logger = logging.getLogger(__name__)

SCHEDULER_POLICIES = ("fcfs", "agent-cache-aware")
AGE_WEIGHT = 1.0
CACHE_WEIGHT = 0.002
CRITICAL_WEIGHT = 5.0
OPTIONAL_PENALTY = 2.0


class Scheduler:
    """
    FCFS scheduler with two modes:

      baseline : process one request to completion before the next.
      batched  : iteration-level batching — admit + prefill many requests,
                 then advance all running requests by one token in a
                 single batched forward pass.  New requests can join the
                 batch the same step they finish prefill.

    Public API (thread-safe):
        add_request(req)   — enqueue a new request
        start()            — launch the background scheduling loop
        stop()             — gracefully shut down
    """

    def __init__(
        self,
        engine: Engine,
        max_running: int = 16,
        mode: str = "paged",
        prefill_chunk_size: int = 0,
        enable_retraction: bool = False,
        scheduler_policy: str = "fcfs",
    ):
        if scheduler_policy not in SCHEDULER_POLICIES:
            raise ValueError(
                f"scheduler_policy must be one of {SCHEDULER_POLICIES}, "
                f"got {scheduler_policy!r}"
            )
        self.engine = engine
        self.max_running = max_running
        self.mode = mode
        self.prefill_chunk_size = prefill_chunk_size
        self.enable_retraction = enable_retraction
        self.scheduler_policy = scheduler_policy

        # Queues
        self.waiting: deque[Request] = deque()
        self.retracted: deque[Request] = deque()
        self.prefilling: deque[Request] = deque()
        self.running: list[Request] = []

        # Thread control
        self._lock = threading.Lock()
        self._running_flag = False
        self._thread: threading.Thread | None = None

        # Stats
        self.total_finished: int = 0
        self.total_generated_tokens: int = 0

    # ── Public API (thread-safe) ────────────────────────────────────────

    def add_request(self, request: Request) -> None:
        """Enqueue a request for scheduling."""
        with self._lock:
            self.waiting.append(request)
            logger.info(
                "Enqueued request %s  (prompt_len=%d, waiting=%d)",
                request.request_id,
                request.num_input_tokens,
                len(self.waiting),
            )

    def start(self) -> None:
        """Start the scheduler loop in a background daemon thread."""
        self._running_flag = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started  (policy=%s)", self.scheduler_policy)

    def stop(self) -> None:
        """Signal the scheduler to stop and wait for the thread to join."""
        self._running_flag = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("Scheduler stopped")

    # ── Main loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running_flag:
            has_work = (
                bool(self.waiting)
                or bool(self.retracted)
                or bool(self.prefilling)
                or bool(self.running)
            )
            if not has_work:
                time.sleep(0.005)  # idle sleep to avoid busy-waiting
                continue
            try:
                self.step()
            except Exception:
                logger.exception("Scheduler step failed")

    # ── Scheduling step ─────────────────────────────────────────────────

    def step(self) -> list[Request]:
        """
        One scheduling iteration.  Behaviour depends on self.mode.

        Returns list of requests that finished in this step.
        """
        if self.mode == "baseline":
            return self._step_baseline()
        if self.mode == "paged":
            return self._step_paged()
        return self._step_batched()

    def _step_baseline(self) -> list[Request]:
        """One request to completion per step. Maximally naive."""
        finished: list[Request] = []

        with self._lock:
            if not self.waiting:
                return finished
            req = self.waiting.popleft()

        req.status = RequestStatus.RUNNING
        self._mark_scheduled(req)
        self._mark_prefill_start(req)
        token_id = self.engine.prefill(req)
        self._mark_prefill_end(req)
        req.output_ids.append(token_id)
        self._stream_token(req, token_id)

        while not self._check_finished(req, token_id):
            token_id = self.engine.decode_step(req)
            req.output_ids.append(token_id)
            self._stream_token(req, token_id)

        self._finish_request(req, finished)
        return finished

    def _step_batched(self) -> list[Request]:
        """
        Iteration-level batched step:
          Phase 1 — admit waiting requests and prefill them (per-request).
          Phase 2 — batched decode: one token for every running request.
        Newly prefilled requests join the decode batch in the same step.
        """
        finished: list[Request] = []

        # ── Phase 1: admit + prefill ────────────────────────────────────
        with self._lock:
            to_prefill: list[Request] = []
            while (
                self.waiting and len(self.running) + len(to_prefill) < self.max_running
            ):
                to_prefill.append(self.waiting.popleft())

        for req in to_prefill:
            req.status = RequestStatus.RUNNING
            self._mark_scheduled(req)
            self._mark_prefill_start(req)
            token_id = self.engine.prefill(req)
            self._mark_prefill_end(req)
            req.output_ids.append(token_id)
            self._stream_token(req, token_id)
            if self._check_finished(req, token_id):
                self._finish_request(req, finished)
            else:
                self.running.append(req)

        # ── Phase 2: batched decode ─────────────────────────────────────
        if self.running:
            token_ids = self.engine.batched_decode(self.running)
            still_running: list[Request] = []
            for req, token_id in zip(self.running, token_ids):
                req.output_ids.append(token_id)
                self._stream_token(req, token_id)
                if self._check_finished(req, token_id):
                    self._finish_request(req, finished)
                else:
                    still_running.append(req)
            self.running = still_running

        return finished

    def _step_paged(self) -> list[Request]:
        """
        Paged-mode step — same shape as ``_step_batched`` but:

          - admission is gated by KV-pool page availability,
          - prefill is varlen-batched, optionally in per-request chunks,
          - decode is paged + flash_attn (with optional CUDA graph),
          - finishing a request commits/free its KV pages via
            ``engine.free_paged_state``.
        """
        finished: list[Request] = []
        pool = self.engine.pool
        assert pool is not None, "scheduler in paged mode but engine has no KV pool"

        # ── Phase 1: resume/admit + batched paged prefill ───────────────
        rehydrated_ready: list[Request] = []
        with self._lock:
            to_prefill: list[Request] = []
            pages_reserved = 0

            while self.prefilling:
                req = self.prefilling[0]
                needed = self.engine.paged_prefill_additional_pages_needed(
                    req, self.prefill_chunk_size
                )
                pages_available = pool.num_free + pool.num_evictable - pages_reserved
                if pages_available < needed:
                    break
                pages_reserved += needed
                to_prefill.append(self.prefilling.popleft())

            active = len(self.running) + len(self.prefilling) + len(to_prefill)
            while (
                self.retracted
                and not self.prefilling
                and active < self.max_running
            ):
                req = self.retracted[0]
                self.engine.prepare_paged_prefill(req)
                if self.engine.paged_prefill_complete(req):
                    rehydrated_ready.append(self.retracted.popleft())
                    active += 1
                    continue

                needed = self.engine.paged_prefill_additional_pages_needed(
                    req, self.prefill_chunk_size
                )
                pages_available = pool.num_free + pool.num_evictable - pages_reserved
                if pages_available < needed:
                    self.engine.release_paged_prefill(req)
                    break
                pages_reserved += needed
                to_prefill.append(self.retracted.popleft())
                active += 1

            while (
                self.waiting
                and not self.prefilling
                and not self.retracted
                and active < self.max_running
            ):
                picked = self._select_waiting_request_for_paged_prefill(
                    pages_reserved, pool
                )
                if picked is None:
                    break  # can't fit; wait for pages to free
                req, needed = picked
                pages_reserved += needed
                to_prefill.append(req)
                active += 1

        for req in rehydrated_ready:
            req.status = RequestStatus.RUNNING
            self._mark_scheduled(req)
            req.needs_rehydrate = False
            self.running.append(req)
            logger.info(
                "Rehydrated request %s from cached pages  (retractions=%d)",
                req.request_id,
                req.num_retractions,
            )

        if to_prefill:
            for req in to_prefill:
                req.status = RequestStatus.RUNNING
                self._mark_scheduled(req)
                self._mark_prefill_start(req)
            for req, token_id in zip(
                to_prefill,
                self.engine.paged_batched_prefill(
                    to_prefill, chunk_size=self.prefill_chunk_size
                ),
            ):
                if req.needs_rehydrate:
                    if self.engine.paged_prefill_complete(req):
                        req.needs_rehydrate = False
                        self.running.append(req)
                        logger.info(
                            "Rehydrated request %s  (retractions=%d)",
                            req.request_id,
                            req.num_retractions,
                        )
                    else:
                        self.prefilling.append(req)
                    continue

                if token_id is None:
                    self.prefilling.append(req)
                    continue
                self._mark_prefill_end(req)
                req.output_ids.append(token_id)
                self._stream_token(req, token_id)
                if self._check_finished(req, token_id):
                    self._finish_request(req, finished)
                else:
                    self.engine.commit_paged_cache(req, finished=False)
                    self.running.append(req)

        # ── Phase 2: paged batched decode ───────────────────────────────
        if self.running:
            token_ids = self._paged_decode_with_retraction()
            if token_ids is not None:
                still_running: list[Request] = []
                for req, token_id in zip(self.running, token_ids):
                    req.output_ids.append(token_id)
                    self._stream_token(req, token_id)
                    if self._check_finished(req, token_id):
                        self._finish_request(req, finished)
                    else:
                        still_running.append(req)
                self.running = still_running

        return finished

    def _select_waiting_request_for_paged_prefill(
        self,
        pages_reserved: int,
        pool,
    ) -> tuple[Request, int] | None:
        """Pick the next waiting request for paged prefill admission."""
        if self.scheduler_policy != "agent-cache-aware":
            req = self.waiting[0]
            self.engine.prepare_paged_prefill(req)
            needed = self.engine.paged_prefill_additional_pages_needed(
                req, self.prefill_chunk_size
            )
            pages_available = pool.num_free + pool.num_evictable - pages_reserved
            if pages_available < needed:
                self.engine.release_paged_prefill(req)
                return None
            return self.waiting.popleft(), needed

        candidates: list[tuple[float, int, Request, int]] = []
        for idx, req in enumerate(self.waiting):
            priority = self._agent_cache_priority(req)
            estimated_needed = self.engine.estimate_paged_prefill_additional_pages_needed(
                req, self.prefill_chunk_size
            )
            candidates.append((priority, -idx, req, estimated_needed))
        candidates.sort(reverse=True)

        for _priority, neg_idx, req, estimated_needed in candidates:
            pages_available = pool.num_free + pool.num_evictable - pages_reserved
            if pages_available < estimated_needed:
                continue

            self.engine.prepare_paged_prefill(req)
            needed = self.engine.paged_prefill_additional_pages_needed(
                req, self.prefill_chunk_size
            )
            pages_available = pool.num_free + pool.num_evictable - pages_reserved
            if pages_available < needed:
                self.engine.release_paged_prefill(req)
                continue

            del self.waiting[-neg_idx]
            return req, needed

        return None

    def _agent_cache_priority(self, req: Request) -> float:
        """Score waiting agentic requests by age, cache reuse, and criticality."""
        metadata = req.profile_metadata or {}
        queue_age = max(0.0, time.time() - req.arrival_time)
        estimated_hit = self.engine.estimate_paged_cache_hit_tokens(req)
        is_optional = self._metadata_bool(metadata, "is_optional")
        is_critical = self._metadata_bool(metadata, "is_critical")

        priority = (
            AGE_WEIGHT * queue_age
            + CACHE_WEIGHT * estimated_hit
            + (CRITICAL_WEIGHT if is_critical else 0.0)
            - (OPTIONAL_PENALTY if is_optional else 0.0)
        )
        req.estimated_cache_hit_tokens = estimated_hit
        req.scheduler_priority = priority
        return priority

    def _metadata_bool(self, metadata: dict, key: str) -> bool:
        value = metadata.get(key, False)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def _paged_decode_with_retraction(self) -> list[int] | None:
        """Decode running requests, retracting victims on KV-pool exhaustion."""
        while self.running:
            try:
                return self.engine.paged_batched_decode(self.running)
            except RuntimeError as exc:
                if not self.enable_retraction or not self._is_kv_pool_exhausted(exc):
                    raise

                victim = self._choose_retraction_victim()
                if victim is None:
                    logger.warning(
                        "KV pool exhausted during decode, but no request has "
                        "retractable pages"
                    )
                    raise

                self.running.remove(victim)
                freed = self.engine.retract_paged_state(victim)
                victim.needs_rehydrate = True
                victim.num_retractions += 1
                victim.status = RequestStatus.WAITING
                self.retracted.appendleft(victim)
                log_request_event(
                    victim,
                    "request_retracted",
                    freed_pages=freed,
                    num_retractions=victim.num_retractions,
                    remaining_running=len(self.running),
                )
                logger.warning(
                    "Retracted request %s after decode allocation failure "
                    "(freed_pages=%d, retractions=%d, remaining_running=%d)",
                    victim.request_id,
                    freed,
                    victim.num_retractions,
                    len(self.running),
                )

        return None

    def _choose_retraction_victim(self) -> Request | None:
        """Pick the largest useful victim, then highest remaining work, youngest."""
        candidates = [
            req for req in self.running if self.engine.paged_retractable_pages(req) > 0
        ]
        if not candidates:
            return None

        def key(req: Request) -> tuple[int, int, float]:
            remaining = req.sampling_params.max_new_tokens - req.num_output_tokens
            return (
                self.engine.paged_retractable_pages(req),
                remaining,
                req.arrival_time,
            )

        return max(candidates, key=key)

    def _is_kv_pool_exhausted(self, exc: RuntimeError) -> bool:
        return "KV pool exhausted" in str(exc)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _mark_scheduled(self, req: Request) -> None:
        if req.profile_scheduled_ts is not None:
            return
        req.profile_scheduled_ts = now()
        log_request_event(
            req,
            "request_scheduled",
            scheduler_policy=self.scheduler_policy,
            estimated_cache_hit_tokens=req.estimated_cache_hit_tokens,
            scheduler_priority=req.scheduler_priority,
            waiting=len(self.waiting),
            running=len(self.running),
            prefilling=len(self.prefilling),
            retracted=len(self.retracted),
        )

    def _mark_prefill_start(self, req: Request) -> None:
        if req.profile_prefill_start_ts is not None:
            return
        req.profile_prefill_start_ts = now()
        log_request_event(
            req,
            "prefill_start",
            prompt_tokens=req.num_input_tokens,
            cache_hit_tokens=req.cache_hit_tokens,
            estimated_cache_hit_tokens=req.estimated_cache_hit_tokens,
            scheduler_priority=req.scheduler_priority,
        )

    def _mark_prefill_end(self, req: Request) -> None:
        if req.profile_prefill_end_ts is not None:
            return
        req.profile_prefill_end_ts = now()
        log_request_event(
            req,
            "prefill_end",
            prompt_tokens=req.num_input_tokens,
            cache_hit_tokens=req.cache_hit_tokens,
            estimated_cache_hit_tokens=req.estimated_cache_hit_tokens,
            scheduler_priority=req.scheduler_priority,
        )

    def _check_finished(self, req: Request, token_id: int) -> bool:
        """Decide whether a request should stop generating."""
        if req.is_finished:
            return True
        if self.engine.is_stop_token(token_id):
            return True
        return False

    def _stream_token(self, req: Request, token_id: int) -> None:
        """Push a generated token into the request's streaming queue."""
        text = self.engine.decode_token(token_id)
        if req.profile_first_token_ts is None:
            req.profile_first_token_ts = now()
            log_request_event(
                req,
                "first_token",
                output_tokens=req.num_output_tokens,
                token_id=token_id,
            )
        req.token_queue.put(
            TokenOutput(token_id=token_id, token_text=text, finished=False)
        )

    def _finish_request(self, req: Request, finished_list: list[Request]) -> None:
        """Mark a request as finished and free its resources."""
        req.status = RequestStatus.FINISHED
        if req.profile_decode_end_ts is None:
            req.profile_decode_end_ts = now()
            log_request_event(
                req,
                "decode_end",
                completion_tokens=req.num_output_tokens,
            )
        req.kv_cache = None  # release GPU memory (baseline / batched modes)
        if self.mode == "paged":
            self.engine.free_paged_state(req)
        req.token_queue.put(TokenOutput(token_id=-1, token_text="", finished=True))
        finished_list.append(req)
        req.profile_finished_ts = now()
        finish_reason = "length" if req.is_finished else "stop"
        log_request_event(
            req,
            "request_finished",
            prompt_tokens=req.num_input_tokens,
            completion_tokens=req.num_output_tokens,
            total_tokens=req.num_input_tokens + req.num_output_tokens,
            cache_hit_tokens=req.cache_hit_tokens,
            estimated_cache_hit_tokens=req.estimated_cache_hit_tokens,
            scheduler_priority=req.scheduler_priority,
            num_retractions=req.num_retractions,
            finish_reason=finish_reason,
        )

        self.total_finished += 1
        self.total_generated_tokens += req.num_output_tokens
        logger.info(
            "Finished request %s  (output_len=%d, running=%d, waiting=%d)",
            req.request_id,
            req.num_output_tokens,
            len(self.running),
            len(self.waiting),
        )
