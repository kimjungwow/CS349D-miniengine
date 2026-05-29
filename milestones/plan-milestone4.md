# Milestone 4 Plan: Agent-Aware Cache-Benefit Scheduling

## Summary

This project implements a new MiniEngine scheduling feature for serving engines
under agentic workloads.

The supporting benchmark work is to prepare AgentBench Math/HumanEval traces and
replay them against MiniEngine. The main contribution is not just the dataset.
The new serving-engine feature is an opt-in scheduler policy:

```text
--scheduler-policy agent-cache-aware
```

This scheduler uses agent workflow metadata and predicted radix-cache reuse to
prioritize requests that are likely to reduce prefill work or unblock critical
agent progress.

Baseline:

```text
Milestone-3 paged scheduler + chunked prefill + radix cache
```

New feature:

```text
Agent-aware cache-benefit scheduling on top of the existing paged/radix-cache engine
```

No tests or benchmarks need to be run by the assistant. Evaluation commands and
metrics are listed only for the user to run later.

## Why This Is Meaningful

MiniEngine already has strong serving mechanisms:

- continuous batching
- paged KV cache
- chunked prefill
- radix prefix cache
- CUDA graph support
- optional retraction

But the existing scheduler does not understand agentic workflow structure. It
treats a ReAct step, Reflexion retry, LATS branch generation, and LATS
verification mostly like ordinary chat requests.

Agentic workloads expose extra semantics:

- many LLM calls belong to the same workflow
- later calls often resend a growing scratchpad
- branch calls share long prefixes
- some calls are critical-path, while others are optional/speculative
- cache reuse differs strongly by call type

The new feature uses this metadata to make better scheduling decisions. This is
a real engine change because it changes which request gets admitted/prefilled
next, not merely how requests are measured.

Project claim:

> I built an AgentBench trace-replay benchmark for MiniEngine and implemented an
> agent-aware cache-benefit scheduler that uses workflow metadata and predicted
> prefix-cache reuse to reduce TTFT and prefill overhead for agentic workloads.

## Implementation Plan

1. **Prepare AgentBench trace format**

   Convert existing AgentBench joined profiling CSVs into a normalized replay
   JSONL format.

   Each LLM-call row should preserve:

   - `workflow_id`
   - `step_id`
   - `agent_type`
   - `workload`
   - `call_type`
   - `is_optional`
   - `is_critical`
   - `branch_id`
   - `attempt_id`
   - prompt/messages or synthetic prompt reconstruction fields
   - observed prompt tokens
   - observed completion tokens
   - original cache-hit fields when available

   Initial scope:

   - workloads: `math`, `humaneval`
   - agents: `react`, `reflexion`, `lats`
   - exclude WebShop and HotpotQA for v1
   - exclude LLMCompiler for v1 because current checkout supports it only for
     HotpotQA/WebShop

2. **Add AgentBench trace replay benchmark**

   Add a MiniEngine-side benchmark script:

   ```text
   benchmark/bench_agentic_replay.py
   ```

   It should:

   - read normalized AgentBench replay JSONL
   - send OpenAI-compatible chat requests to MiniEngine
   - attach agent metadata as HTTP headers
   - support a simple closed-loop replay first
   - not execute tools during replay

3. **Propagate agent metadata into MiniEngine requests**

   Extend the chat completion endpoint to read optional AgentBench headers:

   ```text
   x-agentbench-workflow-id
   x-agentbench-step-id
   x-agentbench-agent-type
   x-agentbench-workload
   x-agentbench-call-type
   x-agentbench-is-optional
   x-agentbench-is-critical
   x-agentbench-branch-id
   x-agentbench-attempt-id
   ```

   Store these values on `Request`. If headers are absent, existing behavior
   must remain unchanged.

4. **Extend request scheduling metadata**

   Add internal scheduling fields:

   ```text
   estimated_cache_hit_tokens
   scheduler_priority
   ```

5. **Add scheduler policy CLI**

   Add:

   ```text
   --scheduler-policy {fcfs,agent-cache-aware}
   ```

   Default should preserve current behavior:

   ```text
   --scheduler-policy fcfs
   ```

   The new policy should be opt-in.

6. **Implement cache-hit preview**

   Add a non-mutating cache preview helper:

   ```text
   estimate_cache_hit_tokens(req) -> int
   ```

   It should:

   - use the same page-aligned prefix semantics as the radix cache
   - not lock cache nodes
   - not mutate cache metrics
   - not alter request state except storing the estimate

7. **Implement priority scoring**

   For waiting requests in paged mode, compute:

   ```text
   priority =
       age_weight * queue_age_seconds
     + cache_weight * estimated_cache_hit_tokens
     + critical_weight * is_critical
     - optional_penalty * is_optional
   ```

   Recommended initial weights:

   ```text
   age_weight = 1.0
   cache_weight = 0.002
   critical_weight = 5.0
   optional_penalty = 2.0
   ```

   Interpretation:

   - high predicted cache reuse improves priority
   - critical-path calls are prioritized
   - optional branch calls are deprioritized
   - queue age prevents starvation

8. **Integrate with paged admission**

   The new policy should only change selection order among waiting requests.

   It must preserve existing safety rules:

   - `max_running`
   - KV page availability
   - cache eviction constraints
   - chunked prefill behavior
   - retracted request handling
   - already-prefilling request priority

   Retracted and currently-prefilling requests should remain higher priority
   than newly waiting requests.

9. **Add observability**

   Record enough information to explain scheduler decisions:

   - selected scheduler policy
   - request metadata
   - estimated cache-hit tokens
   - final scheduler priority
   - actual `cache_hit_tokens`
   - queue time
   - TTFT
   - prefill time
   - call type
   - optional/critical status

   This can be surfaced through replay output CSV and existing profiling logs.

## User-Run Evaluation

The user can run these later.

Baseline server:

```bash
python -m miniengine \
  --model Qwen/Qwen3-8B \
  --mode paged \
  --prefill-chunk-size 512
```

Agent-aware scheduler server:

```bash
python -m miniengine \
  --model Qwen/Qwen3-8B \
  --mode paged \
  --prefill-chunk-size 512 \
  --scheduler-policy agent-cache-aware
```

Optional ablation:

```bash
python -m miniengine \
  --model Qwen/Qwen3-8B \
  --mode paged \
  --prefill-chunk-size 512 \
  --scheduler-policy agent-cache-aware \
  --disable-radix-cache
```

Replay benchmark:

```bash
python -m benchmark.bench_agentic_replay \
  --trace-path <normalized-agentbench-trace.jsonl> \
  --base-url http://localhost:8000 \
  --concurrency 16 \
  --workloads math,humaneval \
  --agent-types react,reflexion,lats \
  --output <results.csv>
```

Metrics to report:

- TTFT p50/p90/p99
- queue time p50/p90/p99
- prefill time
- decode time
- throughput
- cache hit ratio
- cache hit tokens per request
- critical vs optional latency
- per-call-type latency
- ordinary `bench_serving` regression check

Success target:

- Show a meaningful improvement on AgentBench replay, ideally at least 20% lower
  TTFT or queue+prefill latency for Math/HumanEval agentic traces.
- Show no major regression on ordinary serving workloads.

## Assumptions

- The assistant will not run tests or benchmarks unless explicitly asked.
- The assistant must never execute `rm`.
- The new scheduler is opt-in.
- Existing default scheduler behavior remains available.
- Dataset/replay is supporting infrastructure; the main Milestone 4 feature is
  the scheduler policy.
