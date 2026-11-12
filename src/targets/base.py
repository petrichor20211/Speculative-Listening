"""What the replay needs from a target: a chunk-interleaved model behind its released runtime."""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal, Protocol

import numpy as np
import torch

from speclisten.speculator import Backend, Speculator

Kind = Literal["L", "S", "T", "W"]  # listen, speak, tool call; LWS also writes without speaking
Dtype = Literal["bfloat16", "float32"]


class Target(Protocol):
    backend: Backend
    stop: int  # the token that ends a chunk's output
    control: set[int]  # tokens that carry no content (a chunk of control tokens alone is a listen chunk)

    def install(self, spec: Speculator) -> None:
        """Route the runtime's LLM calls for real chunks through ``spec``."""

    def start_session(self, task: str) -> None: ...

    def history(self) -> list[int]:
        """The session's token ids so far, the drafts of the current chunk included."""

    def chunk_type(self, ids: list[int]) -> Kind: ...

    def draft(self, chunk: np.ndarray, budget_ms: float, propose: Callable[[int], list[int]] | None,
              ban: int | None = None, real: bool = False) -> list[int]:
        """The chunk's output decoded before its audio arrives, under the silence hypothesis (or, for the
        oracle, with the real audio), within ``budget_ms``; ``ban`` excludes a first token. Leaves no trace."""

    def step(self, chunk: np.ndarray, temperature: float = 0.0, generator: torch.Generator | None = None) -> list[int]:
        """Prefill the chunk's audio and decode its output through the installed speculator."""


def wall_ms() -> float:
    torch.cuda.synchronize()
    return time.perf_counter() * 1000
