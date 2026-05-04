"""
Model engine — wraps the bare-bone CausalLM for serving.

The engine is a "black box" that the scheduler calls into.  It handles:
  1. Model loading and GPU placement (via model.py + safetensors)
  2. Tokenization / detokenization (chat-template aware via AutoTokenizer)
  3. Prefill (prompt → first token + KV cache)
  4. Decode  (previous token + KV cache → next token + updated KV cache)
  5. Token sampling (delegated to sampler.py)

Two decode paths:
  - decode_step(req)        : one request, used by baseline scheduler
  - batched_decode(reqs)    : many requests, one forward pass with padded
                              KV + attention mask, used by batched mode

Prefill stays per-request — variable prompt lengths make batched prefill
complex, and decode is where the throughput gain lives.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from miniengine.core import Request
from miniengine.model import CausalLM, ModelConfig, load_weights
from miniengine.sampler import sample_token
from miniengine.kv_memory_pool import KVMemoryPool

logger = logging.getLogger(__name__)


class Engine:
    """Model wrapper supporting baseline (per-request) and batched decode."""

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        mode: str = "batched",
        page_size: int = 32,
        mem_fraction_static: float = 0.85,
    ):
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.page_size = page_size
        self.mem_fraction_static = mem_fraction_static

        # ── Tokenizer (still from HF — it's just a tokenizer) ──────────
        logger.info("Loading tokenizer from %s …", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # ── Model (bare-bone PyTorch, loaded from safetensors) ──────────
        logger.info("Loading model config from %s …", model_path)
        config = ModelConfig.from_pretrained(model_path)
        logger.info(
            "Config: layers=%d, hidden=%d, heads=%d, kv_heads=%d, head_dim=%d, "
            "intermediate=%d, vocab=%d, tie_embed=%s",
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.intermediate_size,
            config.vocab_size,
            config.tie_word_embeddings,
        )

        # Build on meta device — load_weights replaces parameters with
        # GPU tensors directly, so we never allocate a CPU fp32 copy.
        with torch.device("meta"):
            self.model = CausalLM(config)
        load_weights(self.model, model_path, dtype=dtype, device=device)
        if mode == "paged":
            if page_size % 256 != 0:
                raise ValueError(
                    f"--page-size must be a multiple of 256 for paged mode "
                    f"(flash_attn_with_kvcache kernel tile constraint); got {page_size}"
                )
            torch.cuda.synchronize()
            total = torch.cuda.get_device_properties(device).total_memory
            used  = torch.cuda.memory_allocated(device)
            budget = int(self.mem_fraction_static * total) - used
            self.kv_pool = KVMemoryPool.from_budget(
                num_layers=config.num_hidden_layers,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                page_size=page_size,
                dtype=dtype,
                device=device,
                bytes_budget=budget,
            )
        self.model.eval()

        # ── Stop tokens ─────────────────────────────────────────────────
        self.stop_token_ids: set[int] = set()
        if self.tokenizer.eos_token_id is not None:
            self.stop_token_ids.add(self.tokenizer.eos_token_id)
        for tok_name in ("eos_token", "pad_token"):
            tid = getattr(self.tokenizer, f"{tok_name}_id", None)
            if tid is not None:
                self.stop_token_ids.add(tid)
        for token_str in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
            tid = self.tokenizer.convert_tokens_to_ids(token_str)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                self.stop_token_ids.add(tid)

        logger.info(
            "Engine ready  —  vocab=%d, stop_ids=%s, params=%dM",
            len(self.tokenizer),
            self.stop_token_ids,
            sum(p.numel() for p in self.model.parameters()) // 1_000_000,
        )

    # ── Page lifecycle (paged mode) ─────────────────────────────────────

    def acquire_pages_for(self, request: Request) -> None:
        """Allocate just enough pool pages to hold the prompt and store the
        page indices on `request.kv_cache`. Decode-time growth allocates one
        more page per page_size tokens written. No-op outside paged mode.
        """
        if self.mode != "paged":
            return
        num_pages = self.kv_pool.pages_needed(request.num_input_tokens)
        pages = self.kv_pool.allocate(num_pages) if num_pages > 0 else []
        request.kv_cache = pages
        logger.debug(
            "Acquired %d pages for %s (prompt_len=%d, pool_free=%d)",
            num_pages,
            request.request_id,
            request.num_input_tokens,
            self.kv_pool.num_free,
        )

    def can_admit(self, request: Request) -> bool:
        """True if the pool currently has enough pages to fit `request`'s
        prompt. Returns True for non-paged modes (no pool to gate on).
        """
        if self.mode != "paged":
            return True
        needed = self.kv_pool.pages_needed(request.num_input_tokens)
        return self.kv_pool.num_free >= needed

    def release_pages_for(self, request: Request) -> None:
        """Return any pool pages held by `request` to the free list.
        No-op outside paged mode or if nothing was acquired.
        """
        if self.mode != "paged":
            return
        pages = request.kv_cache
        if not pages:
            return
        self.kv_pool.free(pages)
        logger.debug(
            "Released %d pages from %s (pool_free=%d)",
            len(pages),
            request.request_id,
            self.kv_pool.num_free,
        )

    # ── Tokenization ────────────────────────────────────────────────────

    def tokenize_messages(self, messages: list[dict[str, str]]) -> list[int]:
        """Apply the model's chat template and tokenize into ids."""
        kwargs: dict[str, Any] = dict(
            tokenize=False,
            add_generation_prompt=True,
        )
        # Qwen3 models support enable_thinking; silently ignore if unsupported
        try:
            text = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, **kwargs)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token id back to a string."""
        return self.tokenizer.decode([token_id], skip_special_tokens=True)

    # ── Forward passes ──────────────────────────────────────────────────

    @torch.inference_mode()
    def prefill(self, request: Request) -> int:
        """
        Run the prefill phase for one request.

        Processes the full prompt in a single forward pass, stores the
        resulting KV cache on the request, and samples the first output
        token.

        Returns:
            The first generated token id.
        """
        input_ids = torch.tensor(
            [request.input_ids], dtype=torch.long, device=self.device
        )
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        logits, kv_caches = self.model(input_ids, position_ids, kv_caches=None)
        request.kv_cache = kv_caches

        # Sample from the last position
        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    @torch.inference_mode()
    def batched_prefill(self, requests: list[Request]) -> list[int]:
        """
        Packed paged prefill: scatter K/V into pool pages, run varlen attention.

        Pages must already be allocated on each request via acquire_pages_for.
        """
        seq_lens = [len(req.input_ids) for req in requests]

        cu_seqlens = torch.zeros(len(requests) + 1, dtype=torch.int32, device=self.device)
        cu_seqlens[1:] = torch.tensor(seq_lens, dtype=torch.int32, device=self.device).cumsum(0)
        max_seqlen = max(seq_lens)

        packed_ids = torch.tensor(
            [tok for req in requests for tok in req.input_ids],
            dtype=torch.long, device=self.device,
        ).unsqueeze(0)

        packed_pos = torch.cat([
            torch.arange(l, device=self.device) for l in seq_lens
        ]).unsqueeze(0)

        # Slot mapping: for each packed token, the flat index
        # (page_idx * page_size + slot_in_page) into the pool's per-layer
        # K/V tensors.
        slot_mapping_list: list[int] = []
        for req in requests:
            pages = req.kv_cache  # list[int]
            for t in range(req.num_input_tokens):
                page = pages[t // self.page_size]
                slot_mapping_list.append(page * self.page_size + (t % self.page_size))
        slot_mapping = torch.tensor(
            slot_mapping_list, dtype=torch.int64, device=self.device,
        )

        logits, _ = self.model(
            packed_ids, packed_pos,
            kv_caches=None,
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
            kv_pool=self.kv_pool.kv_caches,
            slot_mapping=slot_mapping,
        )

        token_ids = []
        for i, req in enumerate(requests):
            end = int(cu_seqlens[i + 1].item())
            req.num_kv_tokens = req.num_input_tokens
            token_ids.append(
                sample_token(logits[:, end - 1, :], req.sampling_params, req.output_ids)
            )

        return token_ids

    @torch.inference_mode()
    def batched_decode(self, requests: list[Request]) -> list[int]:
        """
        Paged batched decode using flash_attn_with_kvcache.

        Reads existing KV from the pool through each request's page table
        (req.kv_cache) and writes the new step's K/V at cache_seqlens[b]
        in-place. Increments req.num_kv_tokens by 1 per request.
        """
        if not requests:
            return []
        batch_size = len(requests)

        input_ids = torch.tensor(
            [[req.output_ids[-1]] for req in requests],
            dtype=torch.long, device=self.device,
        )

        cache_seqlens = torch.tensor(
            [req.num_kv_tokens for req in requests],
            dtype=torch.int32, device=self.device,
        )
        # RoPE position = current cache length (where the new token is written)
        position_ids = cache_seqlens.long().unsqueeze(1)  # (B, 1)

        max_pages = max(len(req.kv_cache) for req in requests)
        block_table = torch.zeros(
            batch_size, max_pages, dtype=torch.int32, device=self.device,
        )
        for i, req in enumerate(requests):
            block_table[i, :len(req.kv_cache)] = torch.tensor(
                req.kv_cache, dtype=torch.int32, device=self.device,
            )

        logits, _ = self.model(
            input_ids, position_ids,
            kv_caches=None,
            kv_pool=self.kv_pool.kv_caches,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
        )

        token_ids: list[int] = []
        for i, req in enumerate(requests):
            token_ids.append(
                sample_token(logits[i:i+1, -1, :], req.sampling_params, req.output_ids)
            )
            req.num_kv_tokens += 1
            # Grow page table if the next decode step would write into a
            # page this request doesn't own yet. If the pool is exhausted,
            # terminate the request gracefully instead of crashing the step.
            next_page_idx = req.num_kv_tokens // self.page_size
            if next_page_idx >= len(req.kv_cache):
                try:
                    req.kv_cache.extend(self.kv_pool.allocate(1))
                except ValueError:
                    req.sampling_params.max_new_tokens = req.num_output_tokens
                    logger.warning(
                        "Pool exhausted mid-decode; terminating %s at output_len=%d",
                        req.request_id, req.num_output_tokens,
                    )

        return token_ids

    @torch.inference_mode()
    def decode_step(self, request: Request) -> int:
        """
        Run one decode step for a request that has already been prefilled.

        Feeds the last generated token through the model together with the
        cached KV values, updates the cache, and samples the next token.

        Returns:
            The next generated token id.
        """
        input_ids = torch.tensor(
            [[request.output_ids[-1]]], dtype=torch.long, device=self.device
        )
        # Position = current KV cache length (= num tokens already processed)
        cache_len = request.kv_cache[0][0].shape[2]  # layer 0, key tensor, seq dim
        position_ids = torch.tensor([[cache_len]], device=self.device)

        logits, kv_caches = self.model(
            input_ids, position_ids, kv_caches=request.kv_cache
        )
        request.kv_cache = kv_caches

        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    def is_stop_token(self, token_id: int) -> bool:
        return token_id in self.stop_token_ids

    # ── Legacy batched decode (milestone-1 batched mode) ────────────────

    @torch.inference_mode()
    def batched_decode_legacy(self, requests: list[Request]) -> list[int]:
        """
        Legacy SDPA-based batched decode for milestone-1 `--mode batched`.

        Pads per-request KV caches to the longest in the batch, builds a
        float attention mask that ignores padding, runs the model once,
        then extracts each request's actual KV (real prefix + new token)
        and samples its next token.
        """
        if not requests:
            return []

        batch_size = len(requests)
        num_layers = len(requests[0].kv_cache)

        # Stack last generated token from each request → (batch, 1)
        input_ids = torch.tensor(
            [[req.output_ids[-1]] for req in requests],
            dtype=torch.long,
            device=self.device,
        )

        # Each request's current KV length and the per-request RoPE position
        cache_lens = [req.kv_cache[0][0].shape[2] for req in requests]
        max_cache_len = max(cache_lens)
        position_ids = torch.tensor(
            [[cl] for cl in cache_lens],
            dtype=torch.long,
            device=self.device,
        )

        # Pad and stack KV caches per layer to (batch, kv_heads, max_cache_len, head_dim)
        padded_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            k_list, v_list = [], []
            for req in requests:
                k, v = req.kv_cache[layer_idx]
                pad_len = max_cache_len - k.shape[2]
                if pad_len > 0:
                    k = F.pad(k, (0, 0, 0, pad_len))
                    v = F.pad(v, (0, 0, 0, pad_len))
                k_list.append(k)
                v_list.append(v)
            padded_kv_caches.append(
                (torch.cat(k_list, dim=0), torch.cat(v_list, dim=0))
            )

        # Mask shape (batch, 1, 1, max_cache_len + 1): the attention forward
        # appends the new token to the cache, so kv_len = max_cache_len + 1.
        # Mask only the padding window [cl, max_cache_len) per request.
        attention_mask = torch.zeros(
            batch_size,
            1,
            1,
            max_cache_len + 1,
            device=self.device,
            dtype=self.dtype,
        )
        for i, cl in enumerate(cache_lens):
            attention_mask[i, 0, 0, cl:max_cache_len] = float("-inf")

        logits, new_kv_caches = self.model(
            input_ids,
            position_ids,
            kv_caches=padded_kv_caches,
            attention_mask=attention_mask,
        )

        # Extract each request's real KV (actual prefix + new token at -1).
        token_ids: list[int] = []
        for i, req in enumerate(requests):
            cl = cache_lens[i]
            per_req_kv = []
            for layer_idx in range(num_layers):
                k_full = new_kv_caches[layer_idx][0][i : i + 1]
                v_full = new_kv_caches[layer_idx][1][i : i + 1]
                k_new = torch.cat([k_full[:, :, :cl, :], k_full[:, :, -1:, :]], dim=2)
                v_new = torch.cat([v_full[:, :, :cl, :], v_full[:, :, -1:, :]], dim=2)
                per_req_kv.append((k_new, v_new))
            req.kv_cache = per_req_kv
            token_ids.append(
                sample_token(
                    logits[i : i + 1, -1, :], req.sampling_params, req.output_ids
                )
            )
        return token_ids
