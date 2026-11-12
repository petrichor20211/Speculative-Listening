"""Streaming SLM through its training repository's model code (``src``, on ``PYTHONPATH``)."""
from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
import torch
from omegaconf import OmegaConf
from src.infer.streaming_slm import load_model
from src.models.streaming_slm import DEFAULT_MAX_TOKENS_PER_CHUNK, MAX_TOOL_CALL_TOKENS, TOOL_CALL_END, StreamingSLMModel
from torch import Tensor
from transformers import DynamicCache

from speclisten.audio import RATE
from speclisten.paths import path
from speclisten.speculator import Speculator
from speclisten.targets.base import Dtype, Kind, wall_ms

DEVICE = "cuda"


def load(dtype: Dtype) -> StreamingSLMModel:
    checkpoint = path("slm_checkpoint")
    if dtype != "float32":
        return load_model(checkpoint, dtype, DEVICE)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cfg = OmegaConf.create(payload["hyper_parameters"]["cfg"])
    cfg.model.pretrained_weights = False
    cfg.model.pretrained_s2s_model = None
    cfg.model.activation_checkpointing = False
    cfg.model.attention.implementation = "sdpa"  # FlashAttention-2 has no float32 kernels
    cfg.exp_manager.explicit_log_dir = os.environ.get("DUPLEX_LOG_DIR", str(path("outputs") / "logs"))
    model = StreamingSLMModel(OmegaConf.to_container(cfg, resolve=True))
    model.load_state_dict(payload["state_dict"], strict=True, assign=True)
    del payload
    model.to(device=DEVICE, dtype=torch.float32).eval()
    model.perception.to(torch.bfloat16)  # the audio encoder is FlashAttention-only; its output is the same for every method
    model.llm.config.use_cache = True
    return model


def choose_token(logits: Tensor, temperature: float, pending: int | None, generator: torch.Generator | None) -> int:
    """Greedy, or speculative sampling against the pending draft token (Leviathan et al. 2023): accept it
    with probability p(x), else sample from p with x removed."""
    if temperature <= 0:
        return int(logits.argmax(-1).item())
    probs = torch.softmax(logits[0].float() / temperature, -1)
    if pending is not None:
        if torch.rand((), device=probs.device, generator=generator) < probs[pending]:
            return pending
        probs[pending] = 0.0
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=generator).item())


class ThinkerBackend:
    def __init__(self, model: StreamingSLMModel) -> None:
        self.model = model
        self.cache = DynamicCache(config=model.llm.config)

    def new_session(self) -> None:
        self.cache = DynamicCache(config=self.model.llm.config)

    def cache_length(self) -> int:
        return self.cache.get_seq_length()

    def forward(self, embeds: Tensor) -> Tensor:
        out = self.model.llm(inputs_embeds=embeds[None], past_key_values=self.cache, use_cache=True, return_dict=True)
        self.cache = out.past_key_values
        return out.last_hidden_state[0]

    def crop(self, length: int) -> None:
        self.cache.crop(length)

    def logits(self, hidden: Tensor) -> Tensor:
        return self.model.lm_head(hidden).masked_fill(~self.model.allowed_tokens, float("-inf"))

    def embed(self, ids: list[int]) -> Tensor:
        return self.model.embed_tokens(torch.tensor(ids, device=self.model.device))


class StreamingSLM:
    """The model's streaming loop (8 text tokens or a 512-token tool call per chunk, then ``<chunk_eos>``),
    reimplemented so that every LLM call goes through a speculator."""

    def __init__(self, dtype: Dtype = "bfloat16") -> None:
        self.model = load(dtype)
        self.backend = ThinkerBackend(self.model)
        self.stop = self.model.chunk_eos_id
        self.control = {self.stop}
        self.tool_start = self.model.chunk_token_ids.tool_call_start
        self.tool_end = self.model.tokenizer.text_to_ids(TOOL_CALL_END)[0]
        self.spec: Speculator | None = None
        self.ids: list[int] = []  # the session's tokens: the prompt and every chunk's output
        self.retained: list[Tensor] = []  # the audio encoder's history window
        self.chunks = 0
        self.silence: tuple[int, Tensor] | None = None  # (chunk index, embedding of a silent chunk after the history)

    def install(self, spec: Speculator) -> None:
        self.spec = spec

    def start_session(self, task: str) -> None:
        self.backend.new_session()
        self.ids = self.model._prompt_tokens(task, "").tolist()
        self.backend.forward(self.backend.embed(self.ids))
        self.retained = []
        self.chunks = 0
        self.silence = None

    def history(self) -> list[int]:
        return self.ids

    def chunk_type(self, ids: list[int]) -> Kind:
        return "T" if self.tool_start in ids else ("S" if len(ids) > 1 else "L")

    def encode(self, chunk: np.ndarray) -> Tensor:
        """The chunk's audio embedding from the encoder's window: the retained history and the chunk."""
        signal = torch.from_numpy(chunk)[None].to(self.model.device)
        window = torch.cat((*self.retained, torch.nn.functional.pad(signal, (0, RATE - len(chunk)))), 1)
        lengths = torch.tensor([window.shape[1] - RATE + len(chunk)], device=window.device)
        perception = self.model.perception
        tokens, counts = perception.encode(window, lengths)
        mask = torch.arange(perception.tokens_per_chunk, device=window.device) < counts[:, None]
        embeds = perception.proj(tokens) * mask[..., None]
        return embeds[0, : int(counts[0])].to(self.model.lm_head.weight.dtype)

    def decode(self, spec: Speculator, prefix: Tensor, temperature: float, generator: torch.Generator | None,
               drafting: bool = False, deadline: float = float("inf"), ban: int | None = None) -> list[int]:
        """The model's chunk loop after ``prefix`` through ``spec``. A draft stops when its budget runs out and
        does not feed its last token: the KV is cropped afterwards anyway."""
        logits, _ = spec.feed_prefill(prefix)
        if ban is not None:
            logits = logits.clone()
            logits[0, ban] = float("-inf")
        ids: list[int] = []
        text_tokens = call_length = 0
        in_call = False
        while True:
            if drafting and wall_ms() >= deadline:
                return ids
            token = choose_token(logits, temperature, spec.peek(), generator)
            ids.append(token)
            self.ids.append(token)
            if token == self.stop:
                if not drafting:
                    spec.feed_token(token, self.backend.embed([token])[0])
                return ids
            if token == self.tool_start:
                in_call, call_length = True, 0
            if in_call:
                call_length += 1
                in_call = token != self.tool_end
            else:
                text_tokens += 1
            capped = call_length >= MAX_TOOL_CALL_TOKENS if in_call else text_tokens >= DEFAULT_MAX_TOKENS_PER_CHUNK
            if capped and drafting:
                ids.append(self.stop)
                self.ids.append(self.stop)
                return ids
            logits, _ = spec.feed_token(token, self.backend.embed([token])[0])
            if capped:
                ids.append(self.stop)
                self.ids.append(self.stop)
                spec.feed_token(self.stop, self.backend.embed([self.stop])[0])
                return ids

    def draft(self, chunk: np.ndarray, budget_ms: float, propose: Callable[[int], list[int]] | None,
              ban: int | None = None, real: bool = False) -> list[int]:
        if budget_ms <= 0:
            return []
        if real:
            prefix = self.encode(chunk)
        else:
            if self.silence is None or self.silence[0] != self.chunks:
                self.silence = (self.chunks, self.encode(np.zeros_like(chunk)))
            prefix = self.silence[1]
        deadline = wall_ms() + budget_ms
        spec = Speculator(self.backend, stop={self.stop}, redraft="pool" if propose is not None else "none")
        spec.pool = propose
        start, known = self.backend.cache_length(), len(self.ids)
        ids = self.decode(spec, prefix, 0.0, None, drafting=True, deadline=deadline, ban=ban)
        self.backend.crop(start)
        del self.ids[known:]
        return ids

    def step(self, chunk: np.ndarray, temperature: float = 0.0, generator: torch.Generator | None = None) -> list[int]:
        assert self.spec is not None, "install a speculator first"
        prefix = self.encode(chunk)
        signal = torch.from_numpy(chunk)[None].to(self.model.device)
        self.retained.append(torch.nn.functional.pad(signal, (0, RATE - len(chunk))))
        self.retained = self.retained[-self.model.aut_history_chunks :] if self.model.aut_history_chunks else []
        self.chunks += 1
        return self.decode(self.spec, prefix, temperature, generator)
