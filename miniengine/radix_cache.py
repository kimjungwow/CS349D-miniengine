"""Radix-tree prefix cache — Milestone 3, Part B.

Stores already-computed KV pages keyed by token prefix so a new request whose
prompt starts with a cached prefix can reuse those pages instead of
recomputing them.  This file is the **skeleton** — fill in the methods marked
``TODO``.

The data structure is a radix tree whose nodes own KV pages from the
``KVMemoryPool``.  Pages held by the cache are *not* in the pool's free list;
they return there only when the cache evicts them (LRU) or when an in-flight
insert chooses to free a redundant duplicate.

Performance counters in ``CacheMetrics`` are read by the ``/cache_stats``
endpoint and by the scheduler's per-prefill-batch INFO log line.  Update them
inside your implementation so those observability hooks light up.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miniengine.kv_memory_pool import KVMemoryPool

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Aggregate cache statistics — surfaced via ``/cache_stats``."""

    total_lookups: int = 0
    total_query_tokens: int = 0
    total_hit_tokens: int = 0
    total_inserted_pages: int = 0
    total_evicted_pages: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_query_tokens == 0:
            return 0.0
        return self.total_hit_tokens / self.total_query_tokens


class RadixNode:
    """A radix-tree node.  Design hints — adjust to fit your implementation:

    * ``parent`` / ``children`` form the tree.
    * ``key`` carries the tokens on the edge from the parent.
    * ``pages`` are the KV pages corresponding to those tokens; for safety
      against partial-page sharing, keep ``len(key)`` a multiple of
      ``page_size`` and ``len(pages) == len(key) // page_size``.
    * ``ref_count`` should reflect "number of locked leaves in this
      subtree" so eviction can check a single field.  Manipulated by
      ``inc_lock_ref`` / ``dec_lock_ref``.
    * ``last_access`` drives LRU.
    """

    __slots__ = ("parent", "children", "key", "pages", "ref_count", "last_access")

    def __init__(self) -> None:
        self.parent: RadixNode | None = None
        self.children: dict = {}
        self.key: list[int] = []
        self.pages: list[int] = []
        self.ref_count: int = 0
        self.last_access: float = time.monotonic()


@dataclass
class MatchResult:
    """Result of a prefix lookup.

    ``matched_tokens`` is page-aligned (multiple of ``page_size``);
    ``matched_pages`` carries the KV pages for those tokens.
    ``last_node`` is the deepest node the walk reached — callers lock it
    (``inc_lock_ref``) for the lifetime of the borrowing request.
    """

    matched_pages: list[int] = field(default_factory=list)
    matched_tokens: int = 0
    last_node: RadixNode | None = None


class RadixCache:
    """Token-prefix → KV-pages cache backed by a radix tree.

    Required behaviours (see milestone 3 doc):
      * page-aligned matching — never return a partial-page result
      * LRU eviction of unlocked subtrees
      * eviction-on-allocate: ``KVMemoryPool.allocate`` should call
        ``cache.evict(n)`` when the free list is short
      * ``inc_lock_ref`` / ``dec_lock_ref`` protect in-flight requests
        (same names as sglang's radix cache).
    """

    def __init__(self, pool: "KVMemoryPool") -> None:
        self.pool = pool
        self.page_size = pool.page_size
        self.root = RadixNode()
        self.metrics = CacheMetrics()
        self._num_cached_pages = 0

    @property
    def num_cached_pages(self) -> int:
        """Total pages currently held by the tree."""
        return self._num_cached_pages

    def num_evictable_pages(self) -> int:
        """Pages that an LRU sweep could free right now."""
        total = 0
        for node in self._iter_nodes():
            if node is not self.root and node.ref_count == 0:
                total += len(node.pages)
        return total

    # ── Lookup ─────────────────────────────────────────────────────────

    def match_prefix(self, tokens: list[int]) -> MatchResult:
        """Find the longest page-aligned prefix of ``tokens`` in the tree.

        Update ``metrics.total_lookups`` / ``total_query_tokens`` /
        ``total_hit_tokens`` so the perf counters are accurate.
        """
        self.metrics.total_lookups += 1
        self.metrics.total_query_tokens += len(tokens)

        node, matched_tokens = self._tree_walk(tokens)
        matched_pages = self.pages_for_node(node) if matched_tokens > 0 else []
        self.metrics.total_hit_tokens += matched_tokens
        return MatchResult(
            matched_pages=matched_pages,
            matched_tokens=matched_tokens,
            last_node=node if matched_tokens > 0 else None,
        )

    # ── Lock ref counting (sglang-style) ───────────────────────────────

    def inc_lock_ref(self, node: RadixNode | None) -> None:
        """Lock ``node`` (and the path to root) against eviction."""
        while node is not None:
            node.ref_count += 1
            node = node.parent

    def dec_lock_ref(self, node: RadixNode | None) -> None:
        """Release a lock.  Refresh ``last_access`` while walking."""
        now = time.monotonic()
        while node is not None:
            node.ref_count -= 1
            if node.ref_count < 0:
                raise RuntimeError("radix cache lock refcount went negative")
            node.last_access = now
            node = node.parent

    # ── Insertion ──────────────────────────────────────────────────────

    def insert_and_return(
        self, tokens: list[int], pages: list[int]
    ) -> tuple[RadixNode, list[int]]:
        """Insert (tokens, pages) into the tree.

        Returns ``(leaf_node, redundant_pages)``: ``redundant_pages`` are
        pages the caller handed in that were duplicates of pages already
        cached at the same prefix — the caller should return them to the
        pool.  Update ``metrics.total_inserted_pages`` to reflect what
        actually got added.
        """
        insert_pages = min(len(tokens) // self.page_size, len(pages))
        insert_len = insert_pages * self.page_size
        if insert_len == 0:
            return self.root, []

        tokens = tokens[:insert_len]
        pages = pages[:insert_pages]
        node, prefix_len = self._tree_walk(tokens)
        prefix_pages = prefix_len // self.page_size
        redundant_pages = pages[:prefix_pages]

        if prefix_len < insert_len:
            new_node = RadixNode()
            new_node.parent = node
            new_node.key = tokens[prefix_len:]
            new_node.pages = pages[prefix_pages:]
            node.children[self._child_key(new_node.key)] = new_node
            node = new_node

            inserted = len(new_node.pages)
            self._num_cached_pages += inserted
            self.metrics.total_inserted_pages += inserted

        return node, redundant_pages

    # ── Eviction ───────────────────────────────────────────────────────

    def evict(self, n_pages_needed: int) -> int:
        """LRU-evict at least ``n_pages_needed`` pages (best effort).

        Return the number actually freed.  Bump
        ``metrics.total_evicted_pages``.  Never touch a locked node.
        """
        if n_pages_needed <= 0:
            return 0

        freed = 0
        candidates = self._evictable_leaves()
        while freed < n_pages_needed and candidates:
            candidates.sort(key=lambda n: n.last_access)
            node = candidates.pop(0)
            if node.parent is None or node.ref_count != 0 or node.children:
                continue

            parent = node.parent
            pages = list(node.pages)
            if pages:
                self.pool.free(pages)
                freed += len(pages)
                self._num_cached_pages -= len(pages)
                self.metrics.total_evicted_pages += len(pages)

            del parent.children[self._child_key(node.key)]
            node.parent = None
            node.children = {}

            if (
                parent is not self.root
                and parent.ref_count == 0
                and not parent.children
            ):
                candidates.append(parent)

        return freed

    # ── Maintenance ────────────────────────────────────────────────────

    def reset(self) -> None:
        """Drop the whole tree, return every page to the pool."""
        pages: list[int] = []
        for node in self._iter_nodes():
            if node is not self.root:
                pages.extend(node.pages)
        self.pool.free(pages)
        self.root = RadixNode()
        self._num_cached_pages = 0

    def pages_for_node(self, node: RadixNode | None) -> list[int]:
        """Return cached pages along the path from root to ``node``."""
        if node is None:
            return []
        parts: list[list[int]] = []
        while node is not None and node is not self.root:
            parts.append(node.pages)
            node = node.parent
        out: list[int] = []
        for pages in reversed(parts):
            out.extend(pages)
        return out

    def _iter_nodes(self) -> list[RadixNode]:
        nodes = [self.root]
        out: list[RadixNode] = []
        while nodes:
            node = nodes.pop()
            out.append(node)
            nodes.extend(node.children.values())
        return out

    def _evictable_leaves(self) -> list[RadixNode]:
        return [
            node
            for node in self._iter_nodes()
            if node is not self.root and node.ref_count == 0 and not node.children
        ]

    def _tree_walk(self, tokens: list[int]) -> tuple[RadixNode, int]:
        limit = self._align_down(len(tokens))
        node = self.root
        prefix_len = 0
        now = time.monotonic()

        while prefix_len < limit:
            child = node.children.get(
                self._child_key(tokens[prefix_len : prefix_len + self.page_size])
            )
            if child is None:
                break

            match_len = self._common_prefix_len(child.key, tokens[prefix_len:limit])
            match_len = self._align_down(match_len)
            if match_len == 0:
                break

            prefix_len += match_len
            child.last_access = now
            if match_len < len(child.key):
                node = self._split_node(child, match_len)
                node.last_access = now
                break

            node = child

        return node, prefix_len

    def _split_node(self, node: RadixNode, split_len: int) -> RadixNode:
        if node.parent is None:
            raise RuntimeError("cannot split radix root")
        if split_len <= 0 or split_len >= len(node.key):
            raise ValueError("split_len must be inside the node key")
        if split_len % self.page_size != 0:
            raise ValueError("radix splits must be page-aligned")

        parent = node.parent
        prefix = RadixNode()
        prefix.parent = parent
        prefix.key = node.key[:split_len]
        prefix.pages = node.pages[: split_len // self.page_size]
        prefix.ref_count = node.ref_count
        prefix.last_access = node.last_access

        parent.children[self._child_key(prefix.key)] = prefix

        node.key = node.key[split_len:]
        node.pages = node.pages[split_len // self.page_size :]
        node.parent = prefix
        prefix.children[self._child_key(node.key)] = node

        return prefix

    def _align_down(self, n: int) -> int:
        return n - (n % self.page_size)

    def _child_key(self, tokens: list[int]) -> tuple[int, ...]:
        if len(tokens) < self.page_size:
            raise ValueError("radix child key requires at least one page")
        return tuple(tokens[: self.page_size])

    def _common_prefix_len(self, a: list[int], b: list[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i
