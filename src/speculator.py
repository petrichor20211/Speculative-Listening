"""Lossless speculation inside a chunk-interleaved target's own decoding loop."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import Tensor

Redraft = Literal["none", "lookup", "recycle", "lookahead", "pool"]

LOOKUP_NGRAM = 3  # prompt lookup decoding: longest n-gram matched in the context
LOOKUP_TOKENS = 10  # and the draft length copied after a match
RECYCLE_CANDIDATES = 8  # Token Recycling: next-token candidates kept per token
RECYCLE_TOKENS = 10
LOOKAHEAD_WINDOW = 7  # lookahead decoding: Jacobi window W (its paper's 7B setting)
LOOKAHEAD_NGRAM = 5  # and n-gram size N
LOOKAHEAD_KEEP = 7  # n-grams kept per first token
POOL_NGRAM = 4  # hypothesis pool: longest n-gram matched against the output so far
POOL_TOKENS = 128  # and the longest continuation proposed from it
MAX_PREFILL_DRAFT = 1024  # a draft is at most one chunk's output, tool calls included


class Backend(Protocol):
    """What the speculator needs from a target: its KV cache and one forward over new positions."""

    def cache_length(self) -> int: ...

    def forward(self, embeds: Tensor) -> Tensor:
        """Run ``embeds`` [T, H] after the cache; returns the last hidden states [T, H]."""

    def crop(self, length: int) -> None: ...

    def logits(self, hidden: Tensor) -> Tensor: ...

    def embed(self, ids: list[int]) -> Tensor: ...


@dataclass
class Pending:
    token: int
    logits: Tensor  # [1, V]: the distribution after this draft token
    hidden: Tensor  # [H]


@dataclass
class ChunkStats:
    forwards: int = 0
    drafted: int = 0
    accepted: int = 0


class Speculator:
    """Replaces the target's one-token feed; the target's loop still picks every token itself."""

    def __init__(self, backend: Backend, *, stop: set[int], redraft: Redraft = "none") -> None:
        self.backend = backend
        self.stop = stop
        self.redraft: Redraft = redraft
        self.history: Callable[[], list[int]] | None = None  # the session's token ids so far
        self.pool: Callable[[int], list[int]] | None = None  # token -> continuation from the hypothesis pool
        self.reset()

    def reset(self) -> None:
        self.pending: list[Pending] = []
        self.prefill_draft: list[int] = []
        self.stats = ChunkStats()
        self.recycled: dict[int, list[int]] = {}  # token -> its most recent top-k next-token candidates
        self.lookahead_pool: dict[int, list[list[int]]] = {}
        self.trajectory: list[list[int]] = []
        self.window: list[int] = []

    def begin_chunk(self, draft: list[int] | None = None) -> None:
        """Before a chunk: ``draft`` rides in the chunk's prefill forward and is verified there."""
        self.stats = ChunkStats()
        self.prefill_draft = list(draft or [])[:MAX_PREFILL_DRAFT]

    def peek(self) -> int | None:
        """The next pending draft token: what ``feed_token`` accepts without a forward."""
        return self.pending[0].token if self.pending else None

    def drop_pending(self) -> None:
        if self.pending:
            self.backend.crop(self.backend.cache_length() - len(self.pending))
            self.pending.clear()

    def run(self, embeds: Tensor) -> tuple[Tensor, Tensor]:
        """One forward over new positions; returns (logits [T, V], hidden [T, H])."""
        self.stats.forwards += 1
        hidden = self.backend.forward(embeds)
        return self.backend.logits(hidden), hidden

    def feed_prefill(self, embeds: Tensor) -> tuple[Tensor, Tensor]:
        """The chunk's input. Returns the logits after it [1, V] and its hidden states [T, H]."""
        draft, self.prefill_draft = self.prefill_draft, []
        if not draft:
            logits, hidden = self.run(embeds)
            return logits[-1:], hidden
        n = len(embeds)
        logits, hidden = self.run(torch.cat((embeds, self.backend.embed(draft))))
        self.stage(draft, logits[n:], hidden[n:])
        self.update_recycled(draft, logits[n:])
        return logits[n - 1 : n], hidden[:n]

    def feed_token(self, token: int, embed: Tensor) -> tuple[Tensor, Tensor]:
        """One generated token: the pending draft position's logits if it matches, else one forward with the
        ``redraft`` policy's draft attached. Returns the logits after it [1, V] and its hidden state [1, H]."""
        if self.pending and self.pending[0].token == token:
            hit = self.pending.pop(0)
            self.stats.accepted += 1
            return hit.logits, hit.hidden[None]
        self.drop_pending()
        draft = self.propose(token) if token not in self.stop else []
        x = torch.cat((embed[None], self.backend.embed(draft))) if draft else embed[None]
        if self.redraft == "lookahead" and token not in self.stop:
            window = self.jacobi_window(token)
            logits, hidden = self.run(torch.cat((x, self.backend.embed(window))))
            self.backend.crop(self.backend.cache_length() - len(window))
            self.update_lookahead(window, logits[len(x) :].argmax(-1).tolist())
            logits, hidden = logits[: len(x)], hidden[: len(x)]
        else:
            logits, hidden = self.run(x)
        self.update_recycled([token] + draft, logits)
        if draft:
            self.stage(draft, logits[1:], hidden[1:])
        return logits[:1], hidden[:1]

    def stage(self, tokens: list[int], logits: Tensor, hidden: Tensor) -> None:
        self.pending = [Pending(t, logits[i : i + 1], hidden[i]) for i, t in enumerate(tokens)]
        self.stats.drafted += len(tokens)

    def propose(self, token: int) -> list[int]:
        """The draft that follows ``token`` under the active redraft policy."""
        if self.redraft == "lookup":
            return self.lookup(token)
        if self.redraft == "recycle":
            return self.recycle(token)
        if self.redraft == "lookahead":
            return self.lookahead(token)
        if self.redraft == "pool" and self.pool is not None:
            return self.pool(token)
        return []

    def cut(self, draft: list[int]) -> list[int]:
        """``draft`` up to and including its first stop token."""
        for i, t in enumerate(draft):
            if t in self.stop:
                return draft[: i + 1]
        return draft

    def lookup(self, token: int) -> list[int]:
        """Prompt lookup: the continuation of the most recent earlier occurrence of the longest n-gram ending in ``token``."""
        history = list(self.history()) if self.history is not None else []
        if not history or history[-1] != token:
            history.append(token)
        for n in range(min(LOOKUP_NGRAM, len(history) - 1), 0, -1):
            pattern = history[-n:]
            for start in range(len(history) - n - 1, -1, -1):
                if history[start : start + n] == pattern:
                    return self.cut(history[start + n : start + n + LOOKUP_TOKENS])
        return []

    def recycle(self, token: int) -> list[int]:
        """Token Recycling: the chain of top-1 candidates recorded after ``token``."""
        out: list[int] = []
        while len(out) < RECYCLE_TOKENS:
            candidates = self.recycled.get(token)
            if not candidates:
                break
            token = candidates[0]
            out.append(token)
            if token in self.stop:
                break
        return out

    def update_recycled(self, tokens: list[int], logits: Tensor) -> None:
        if self.redraft != "recycle" or not tokens:
            return
        top = logits[: len(tokens)].topk(RECYCLE_CANDIDATES, dim=-1).indices.tolist()
        for t, candidates in zip(tokens, top, strict=True):
            self.recycled[t] = candidates

    def lookahead(self, token: int) -> list[int]:
        """Lookahead decoding: the most recent pool n-gram that starts with ``token``."""
        candidates = self.lookahead_pool.get(token)
        return self.cut(list(candidates[0])) if candidates else []

    def jacobi_window(self, token: int) -> list[int]:
        if len(self.window) != LOOKAHEAD_WINDOW:
            history = list(self.history()) if self.history is not None else []
            seed = (history + [token])[-LOOKAHEAD_WINDOW:]
            self.window = (seed * LOOKAHEAD_WINDOW)[:LOOKAHEAD_WINDOW]
        return list(self.window)

    def update_lookahead(self, window: list[int], outputs: list[int]) -> None:
        """Collect the n-grams along the window's trajectory and advance the window."""
        self.trajectory = (self.trajectory + [window])[-(LOOKAHEAD_NGRAM - 1) :]
        if len(self.trajectory) == LOOKAHEAD_NGRAM - 1:
            for i in range(LOOKAHEAD_WINDOW):
                gram = [level[i] for level in self.trajectory] + [outputs[i]]
                candidates = self.lookahead_pool.setdefault(gram[0], [])
                tail = gram[1:]
                if tail in candidates:
                    candidates.remove(tail)
                candidates.insert(0, tail)
                del candidates[LOOKAHEAD_KEEP:]
        self.window = list(outputs)


def pool_lookup(history: list[int], pool: list[list[int]], stop: int) -> list[int]:
    """The continuation of the longest n-gram ending ``history``, searched in the session's drafts (most
    recent first) and then in ``history`` itself; cut after ``stop``."""
    for n in range(min(POOL_NGRAM, len(history)), 0, -1):
        pattern = history[-n:]
        for sequence, own in [(draft, False) for draft in reversed(pool)] + [(history, True)]:
            last = len(sequence) - n - (1 if own else 0)
            for i in range(last, -1, -1):
                if sequence[i : i + n] == pattern and i + n < len(sequence):
                    out = sequence[i + n : i + n + POOL_TOKENS]
                    if stop in out:
                        out = out[: out.index(stop) + 1]
                    return out
    return []
