"""Shared replay machinery of the MiniCPM-o-based targets: one unit of the released runtime per chunk."""
from __future__ import annotations

import collections
import copy
import random
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from speclisten.audio import RATE
from speclisten.speculator import Speculator
from speclisten.targets.base import Kind, wall_ms

ATOMS = (int, float, bool, str, bytes, type(None), complex, range, torch.dtype, torch.device)
IMMUTABLE_DEPTH = 3  # nested tuples checked for immutability
TENSOR_DEPTH = 8  # containers walked when registering shared tensors
Phase = Literal["prefill", "generate"]


class BudgetExhausted(Exception):
    """Raised inside the runtime's generation when a draft's budget runs out."""


def is_heavy(value: object) -> bool:
    name = type(value).__name__
    return isinstance(value, torch.nn.Module) or "Tokenizer" in name or "Processor" in name or callable(value)


def is_immutable(value: object, depth: int = 0) -> bool:
    if isinstance(value, ATOMS):
        return True
    return isinstance(value, (tuple, frozenset)) and depth < IMMUTABLE_DEPTH and all(is_immutable(x, depth + 1) for x in value)


def register_tensors(value: object, memo: dict[int, object], depth: int = 0) -> None:
    """Every tensor reachable through containers and plain objects goes into deepcopy's memo, so it is shared."""
    if torch.is_tensor(value):
        memo[id(value)] = value
        return
    if depth > TENSOR_DEPTH or isinstance(value, ATOMS) or id(value) in memo:
        return
    if isinstance(value, dict):
        for x in value.values():
            register_tensors(x, memo, depth + 1)
    elif isinstance(value, (list, tuple, set, frozenset, collections.deque)):
        for x in value:
            register_tensors(x, memo, depth + 1)
    elif hasattr(value, "__dict__") and not is_heavy(value):
        for x in vars(value).values():
            register_tensors(x, memo, depth + 1)


def is_token_feed(last: tuple[int, Tensor] | None, embeds: Tensor) -> bool:
    """Whether ``embeds`` is the embedding of the token the runtime just looked up (a one-token generation step)."""
    return last is not None and embeds.size(0) == 1 and last[1].data_ptr() == embeds.data_ptr()


def padded(chunk: np.ndarray) -> np.ndarray:
    return np.pad(chunk, (0, RATE - len(chunk))).astype(np.float32)


class MiniCPMOBackend:
    def __init__(self, decoder: object) -> None:
        self.decoder = decoder

    def cache_length(self) -> int:
        return self.decoder.get_cache_length()

    def forward(self, embeds: Tensor) -> Tensor:
        decoder = self.decoder
        past = decoder.get_cache_length()
        positions = torch.arange(past, past + len(embeds), device=embeds.device)[None]
        out = decoder.m(inputs_embeds=embeds[None], position_ids=positions, past_key_values=decoder.cache, return_dict=True,
                        output_hidden_states=True)
        decoder.cache = out.past_key_values
        return out.hidden_states[-1][0]

    def crop(self, length: int) -> None:
        self.decoder.cache.crop(length)

    def logits(self, hidden: Tensor) -> Tensor:
        return self.decoder.m.lm_head(hidden)

    def embed(self, ids: list[int]) -> Tensor:
        return self.decoder.embed_tokens(ids)


class MiniCPMOTarget:
    """Real chunks run with the runtime's LLM feeds routed through the installed speculator; a draft runs the
    same unit routine on a snapshot of the session, with a budget, and the snapshot is restored afterwards."""

    tool_start = -1  # no tool call unless a subclass says otherwise
    feed_returns_tuple = True  # whether the runtime's ``feed`` returns (logits, hidden) or logits alone

    def __init__(self, duplex: object, tokenizer: object, control: set[int], runner: object | None = None) -> None:
        self.duplex = duplex
        self.tokenizer = tokenizer
        self.control = control
        self.runner = runner
        self.decoder = duplex.decoder
        self.backend = MiniCPMOBackend(self.decoder)
        self.stop = tokenizer.convert_tokens_to_ids("</unit>")
        # deepcopy memo for snapshots: the model, tokenizer and session holders are shared, never copied
        self.shared: dict[int, object] = {}
        holders = (duplex, self.decoder, duplex.model, self.decoder.m, tokenizer, runner, getattr(duplex, "tokenizer", None),
                   getattr(duplex, "processor", None), getattr(duplex.model, "processor", None), getattr(runner, "bundle", None))
        for obj in holders:
            if obj is None:
                continue
            self.shared[id(obj)] = obj
            if isinstance(obj, torch.nn.Module):
                self.shared.update({id(module): module for module in obj.modules()})
                self.shared.update({id(tensor): tensor for tensor in list(obj.parameters()) + list(obj.buffers())})
        self.original = (self.decoder.feed, self.decoder.embed_token)
        self.routed = self.original
        self.phase: Phase | None = None
        self.prefilled = False
        self.last: tuple[int, Tensor] | None = None

    def prepare(self) -> None: ...

    def prefill(self, chunk: np.ndarray) -> None: ...

    def generate(self) -> None: ...

    def trim(self, ids: list[int]) -> list[int]:
        """A draft's tokens from the unit's recorded ids."""
        return ids

    def drain(self) -> None:
        """Drop what an interrupted draft left in the runtime's queues."""

    def copy(self, value: object) -> object:
        """A copy that shares tensors and immutable values (the runtimes never write a session tensor in place)."""
        if is_immutable(value):
            return value
        if type(value) is list and all(type(x) in (int, float, bool, str) for x in value):
            return list(value)
        memo = dict(self.shared)
        register_tensors(value, memo)
        return copy.deepcopy(value, memo)

    def save(self, obj: object, skip: tuple[str, ...]) -> dict[str, object]:
        saved = {}
        for key, value in vars(obj).items():
            if key in skip or is_heavy(value):
                continue
            try:
                saved[key] = self.copy(value)
            except Exception:  # locks and events cannot be copied; they are not session state
                saved[key] = value
        return saved

    def snapshot(self) -> dict[str, object]:
        model = self.duplex.model
        snap = {
            "duplex": self.save(self.duplex, skip=("decoder", "model", "processor", "tokenizer")),
            "decoder": self.save(self.decoder, skip=("cache", "m", "tokenizer", "feed", "decode")),
            "cache_length": self.decoder.get_cache_length(),
            "audio_kv": self.copy(getattr(model, "audio_past_key_values", None)),
            "audio_chunk": getattr(model, "audio_chunk_idx", None),
            "mel": model.processor.get_streaming_snapshot() if getattr(model, "processor", None) is not None else None,
            # feature extraction may dither with random noise: the draft must not consume the real chunk's noise
            "rng": (torch.get_rng_state(), torch.cuda.get_rng_state(), np.random.get_state(), random.getstate()),
        }
        if self.runner is not None:
            snap["runner"] = self.save(self.runner, skip=("bundle", "duplex", "params"))
        return snap

    def restore(self, snap: dict[str, object]) -> None:
        assert not snap.get("restored"), "a snapshot shares its objects and is restored once"
        snap["restored"] = True
        model = self.duplex.model
        if self.decoder.get_cache_length() > snap["cache_length"]:
            self.decoder.cache.crop(snap["cache_length"])
        for key, value in snap["duplex"].items():
            setattr(self.duplex, key, value)
        for key, value in snap["decoder"].items():
            setattr(self.decoder, key, value)
        model.audio_past_key_values = snap["audio_kv"]
        if snap["audio_chunk"] is not None:
            model.audio_chunk_idx = snap["audio_chunk"]
        if snap["mel"] is not None:
            model.processor.restore_streaming_snapshot(snap["mel"])
        if "runner" in snap:
            for key, value in snap["runner"].items():
                setattr(self.runner, key, value)
        cpu, cuda, numpy_state, python_state = snap["rng"]
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(cuda)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        assert self.decoder.get_cache_length() == snap["cache_length"]

    def output(self, logits: Tensor, hidden: Tensor, return_logits: bool, return_hidden: bool) -> object:
        """A speculator result shaped as the runtime's ``feed`` returns it."""
        if not return_logits:
            return None
        return (logits, hidden[None]) if (self.feed_returns_tuple or return_hidden) else logits

    def install(self, spec: Speculator) -> None:
        _, embed_token = self.original

        def tracked(tid: Tensor | int) -> Tensor:
            embedding = embed_token(tid)
            self.last = (int(tid.item() if torch.is_tensor(tid) else tid), embedding)
            return embedding

        def routed(embeds: Tensor, return_logits: bool = False, return_hidden: bool = False) -> object:
            last, self.last = self.last, None
            if self.phase == "generate" and is_token_feed(last, embeds):
                logits, hidden = spec.feed_token(last[0], embeds[0])
            elif return_logits and not self.prefilled and (self.phase == "prefill" or embeds.size(0) > 1):
                # the unit's prefill: the audio feed, or the first multi-position feed of a unit routine that
                # feeds the audio and generates in one call
                self.prefilled = True
                logits, hidden = spec.feed_prefill(embeds)
            else:
                spec.drop_pending()  # runtime markers must not follow stale draft positions
                logits, hidden = spec.run(embeds)
                logits = logits[-1:]
            return self.output(logits, hidden, return_logits, return_hidden)

        self.decoder.feed, self.decoder.embed_token = routed, tracked
        self.routed = (routed, tracked)

    def start_session(self, task: str) -> None:
        torch.manual_seed(0)  # the same random stream (feature dithering) for every method
        np.random.seed(0)
        random.seed(0)
        self.prepare()

    def history(self) -> list[int]:
        return self.duplex.total_ids

    def chunk_type(self, ids: list[int]) -> Kind:
        if self.tool_start in ids:
            return "T"
        return "S" if any(t not in self.control for t in ids) else "L"

    def step(self, chunk: np.ndarray, temperature: float = 0.0, generator: torch.Generator | None = None) -> list[int]:
        assert temperature == 0.0, "speculative sampling is implemented for the Streaming SLM only"
        before = len(self.duplex.total_ids)
        self.phase, self.prefilled = "prefill", False
        self.prefill(padded(chunk))
        self.phase = "generate"
        self.generate()
        self.phase = None
        return list(self.duplex.total_ids[before:])

    def draft(self, chunk: np.ndarray, budget_ms: float, propose: Callable[[int], list[int]] | None,
              ban: int | None = None, real: bool = False) -> list[int]:
        if budget_ms <= 0:
            return []
        feed, embed_token = self.original
        deadline = wall_ms() + budget_ms
        spec = None
        if propose is not None:  # the draft itself verifies pool proposals: the same output in fewer forwards
            spec = Speculator(self.backend, stop=set(), redraft="pool")
            spec.pool = propose
        generating = banned = False
        last: tuple[int, Tensor] | None = None

        def tracked(tid: Tensor | int) -> Tensor:
            nonlocal last
            embedding = embed_token(tid)
            last = (int(tid.item() if torch.is_tensor(tid) else tid), embedding)
            return embedding

        def budgeted(embeds: Tensor, return_logits: bool = False, return_hidden: bool = False) -> object:
            nonlocal last, banned
            if generating and wall_ms() >= deadline:
                raise BudgetExhausted
            token, last = last, None
            if spec is not None and generating and is_token_feed(token, embeds):
                logits, hidden = spec.feed_token(token[0], embeds[0])
                return self.output(logits, hidden, return_logits, return_hidden)
            if spec is not None:
                spec.drop_pending()
            out = feed(embeds, return_logits, return_hidden) if return_hidden else feed(embeds, return_logits)
            # the runner-up branch bans the unit's first decision: the logits of its prefill feed
            if ban is not None and return_logits and not banned and (not generating or embeds.size(0) > 1):
                banned = True
                logits = (out[0] if isinstance(out, tuple) else out).clone()
                logits[..., ban] = float("-inf")
                out = (logits, out[1]) if isinstance(out, tuple) else logits
            return out

        self.decoder.feed, self.decoder.embed_token = budgeted, tracked
        snap = self.snapshot()
        before = len(self.duplex.total_ids)
        try:
            self.prefill(padded(chunk) if real else np.zeros(RATE, np.float32))
            generating = True
            self.generate()
        except BudgetExhausted:
            pass
        finally:
            ids = self.trim(list(self.duplex.total_ids[before:]))
            self.drain()
            self.restore(snap)
            self.decoder.feed, self.decoder.embed_token = self.routed
        return ids
