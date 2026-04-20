"""
Model engine — wraps the bare-bone CausalLM for serving.

The engine is a "black box" that the scheduler calls into.  It handles:
  1. Model loading and GPU placement (via model.py + safetensors)
  2. Tokenization / detokenization (chat-template aware via AutoTokenizer)
  3. Prefill (prompt → first token + KV cache)
  4. Decode  (previous token + KV cache → next token + updated KV cache)
  5. Token sampling (delegated to sampler.py)

Design note:
  The current API is single-request (prefill / decode_step).  A natural
  first optimisation is to add batched versions that pad sequences and run
  multiple requests through a single forward pass.

  For tensor parallelism, the bare-bone nn.Linear layers in model.py can
  be sharded directly: Q/K/V/gate/up column-wise, O/down row-wise, with
  an all-reduce after the row-parallel matmul.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoTokenizer

from miniengine.core import Request
from miniengine.model import CausalLM, ModelConfig, load_weights
from miniengine.sampler import sample_token
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Engine:
    """Bare-bone model wrapper for single-request prefill and decode."""

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.device = device
        self.dtype = dtype

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

        self.model = CausalLM(config)
        load_weights(self.model, model_path, dtype=dtype, device=device)
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
    def batched_decode(self, requests: list[Request]) -> int:

        batch_size = len(requests)

        cache_lens = [req.kv_cache[0][0].shape[2] for req in requests]
        
        # scalar
        batched_cache_len = max(cache_lens)
        
        # (batch, 1)
        batched_input_ids = torch.tensor(
            [req.output_ids[-1] for req in requests],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)

        # (batch, 1)

        batched_position_ids = torch.tensor(
            cache_lens,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)
        # batched_position_ids = torch.tensor([[batched_cache_len for _ in range(len(requests))]], device=self.device)

        # (batch, batched_cache_len)
        # attn_mask = torch.ones(len(requests),(batched_cache_len+1),dtype=torch.bool, device=self.device)
        # for i in range(len(requests)):
        #     attn_mask[i,requests[i].kv_cache[0][0].shape[2]:] = False
        cache_lens_t = torch.tensor(cache_lens, device=self.device)  # (B,)
        attn_mask = torch.arange(batched_cache_len + 1, device=self.device)[None, :] <= cache_lens_t[:, None]
        num_layers = len(requests[0].kv_cache)
        attn_mask = attn_mask[:, None, None, :].expand(-1, 32, 1, -1) # FIXME: hardcoded 32 maybe number of heads?
        

        #TODO: Pads per-request KV caches to the max cache length in the batch
        padded_kvcache = []
        
        for i in range(num_layers):
            per_layer_batched_k = []
            per_layer_batched_v = []
            for req in requests:
                pad_length = batched_cache_len-req.kv_cache[0][0].shape[2]
                k, v = req.kv_cache[i]
                if pad_length > 0:
                    k = F.pad(k, (0, 0, 0, pad_length))  # seq dim pad
                    v = F.pad(v, (0, 0, 0, pad_length))
                per_layer_batched_k.append(k[0])
                per_layer_batched_v.append(v[0])
            padded_kvcache.append((torch.stack(per_layer_batched_k, dim=0),torch.stack(per_layer_batched_v, dim=0)))
        

        if False:
            print(">>",attn_mask.shape)
            print(batched_input_ids.shape)
            print(batched_position_ids.shape)
            print(padded_kvcache[0][0].shape)
        logits, new_kv = self.model(input_ids=batched_input_ids, position_ids=batched_position_ids, kv_caches=padded_kvcache, attn_mask=attn_mask)
        if False:
            print(len(padded_kvcache))
            print(len(padded_kvcache[0]))
            print(padded_kvcache[0][0].shape)
            
            print(len(requests[0].kv_cache))
            print(len(requests[0].kv_cache[0]))
            print(requests[0].kv_cache[0][0].shape)
            print("@@")
            print(logits.shape)
            print(new_kv[0][0].shape)
        for i in range(len(requests)):
            for l in range(num_layers):
                new_token_k = new_kv[l][0][i][:, -1, :]  
                new_token_k = new_token_k.unsqueeze(0).unsqueeze(2)
                new_token_v = new_kv[l][1][i][:, -1, :]  
                new_token_v = new_token_v.unsqueeze(0).unsqueeze(2)
                req.kv_cache[l] = (torch.cat([req.kv_cache[l][0], new_token_k], dim=2),torch.cat([req.kv_cache[l][1], new_token_v], dim=2))
        
        # return sample_token(
            # logits[:, -1, :], request.sampling_params, request.output_ids
        # )
        return [sample_token(logits[i,-1,:],requests[i].sampling_params, requests[i].output_ids) for i in range(len(requests)) ]

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

        logits, kv_caches = self.model(input_ids, position_ids, kv_caches=request.kv_cache)
        request.kv_cache = kv_caches

        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    def is_stop_token(self, token_id: int) -> bool:
        return token_id in self.stop_token_ids
