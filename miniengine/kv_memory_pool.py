"""Pre-allocated paged KV cache memory pool — Milestone 2, Part A.

This is a SKELETON. Implement the methods below.

The pool owns a fixed amount of GPU memory, divided into equal-size
**pages**. Each page holds the KV state for `page_size` tokens for one
layer. Requests acquire pages as their KV grows and return them when
they finish; the cache itself never reallocates.

Storage layout (page-major vs token-major, contiguous K+V vs separate,
shape conventions, etc.) is YOUR design decision — pick something and
document the tradeoffs.
"""

from __future__ import annotations

import torch
import math

from collections import deque


class KVMemoryPool:
    """Pre-allocated paged KV cache pool.

    Args:
        num_pages:    Total pages in the pool (capacity).
        page_size:    Tokens per page. Tunable knob — exposed as
                      `--page-size` on the CLI. Smaller = less
                      fragmentation, bigger page tables; larger = the
                      opposite.
        num_layers:   Number of transformer layers.
        num_kv_heads: KV heads per layer (GQA).
        head_dim:     Per-head dimension.
        dtype:        KV dtype (typically bfloat16).
        device:       e.g. "cuda".
    """

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Indices of currently-free pages
        self.free: deque[int] = deque(range(num_pages))

        # Fused K+V tensor: [num_layers, 2, num_pages, page_size, num_kv_heads, head_dim]
        # Indexed as cache[layer, 0/1, page_idx, slot, kv_head, :] for K/V respectively.
        self.cache = torch.zeros(
            num_layers,
            2,
            num_pages,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self._kv_caches = [
            (self.cache[layer, 0], self.cache[layer, 1])
            for layer in range(num_layers)
        ]

    def allocate(self, num_pages: int) -> list[int]:
        """Reserve `num_pages` pages and return their indices.

        Raises if the pool cannot satisfy the request.
        """
        if num_pages > len(self.free):
            raise ValueError("num_pages is bigger than available pages.")
        ret = []
        for _ in range(num_pages):
            ret.append(self.free.popleft())
        raise ret

    def free(self, page_indices: list[int]) -> None:
        """Return the listed pages to the free pool."""
        for i in page_indices:
            self.free.append(i)
            # TODO: Zeroize?

    def pages_needed(self, seq_len: int) -> int:
        """How many pages are required to store `seq_len` tokens."""
        return math.ceil(seq_len / self.page_size)

    @property
    def num_free(self) -> int:
        """Pages currently available for allocation."""
        return len(self.free)

    @property
    def kv_caches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer (K, V) cache tensors.

        The attention path holds references to these and indexes into
        them via per-request page tables. The exact shape is up to your
        design — but it must be STABLE: no reallocation, no resizing,
        no swapping out the tensors after construction.
        """
        return self._kv_caches

    @classmethod
    def from_budget(
        cls,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        bytes_budget: int,
    ) -> KVMemoryPool:
        """Convenience: derive `num_pages` from a memory budget."""
        element_size = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else 2
        # cache shape: (num_layers, 2, num_pages, page_size, num_kv_heads, head_dim)
        bytes_per_page = 2 * num_layers * page_size * num_kv_heads * head_dim * element_size
        num_pages = max(1, bytes_budget // bytes_per_page)
        return cls(num_pages, page_size, num_layers, num_kv_heads, head_dim, dtype, device)
