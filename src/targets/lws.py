"""LWS (Listen-Write-Speak) through its released demo runtime (``triplex_runtime``, on ``PYTHONPATH``)."""
from __future__ import annotations

import queue
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from triplex_runtime.async_model import MiniCPMOTriplexAsyncStreamingModel
from triplex_runtime.decoder import DuplexWindowConfig

from speclisten.audio import read_mono_16k
from speclisten.paths import path
from speclisten.targets.base import Dtype, Kind
from speclisten.targets.minicpmo import MiniCPMOTarget

PROFILE = "haitian_triplex"  # the demo's tri-channel profile: system prompt and reference voice
REPETITION_PENALTY = 1.05  # the runtime's sampling settings that greedy decoding keeps
REPETITION_WINDOW = 256


class LWS(MiniCPMOTarget):
    """The runtime is driven unit by unit without its worker threads: ``prefill`` pushes the chunk through its
    streaming front end and takes the embeddings off its queue, ``generate`` runs its unit routine on them."""

    feed_returns_tuple = False

    def __init__(self, dtype: Dtype = "bfloat16") -> None:
        assets = path("lws_assets")
        duplex = MiniCPMOTriplexAsyncStreamingModel(
            device="cuda", ls_mode="explicit", checkpoint_path=str(assets / "checkpoint" / "triplex_checkpoint.pt"),
            model_path=str(path("minicpmo_base")), tokenizer_path=str(assets / "tokenizer"),
            tts_checkpoint_path=str(assets / "checkpoint" / "tts.pt"), token2wav_model_dir=str(assets / "token2wav"),
        )
        if dtype == "float32":  # the runtime always loads bfloat16
            duplex.model.to(torch.float32)
        decoder = duplex.decoder
        decoder.set_window_config(DuplexWindowConfig(sliding_window_mode="off"))  # sessions are shorter than its window
        device = decoder.m.lm_head.weight.device
        decoder.embed_tokens = lambda ids: decoder.m.model.embed_tokens(torch.tensor(ids, device=device))
        profile = assets / "profiles" / PROFILE
        self.prefix = (profile / "prefix_prompt.txt").read_text(encoding="utf-8").strip()
        self.suffix = (profile / "suffix_prompt.txt").read_text(encoding="utf-8").strip()
        self.ref_audio = read_mono_16k(profile / "ref_audio.wav")
        self.speak_id = duplex.speak_token_id
        self.chunk_eos_id = duplex.tts_eos_token_id
        self.think_bos = set(duplex.thinking_start_token_ids)
        self.think_eos = set(duplex.thinking_end_token_ids)
        self.embeds: Tensor | None = None
        self.unit_time = 0
        unit_ids = {duplex._unit_start_id, duplex._unit_end_id}
        super().__init__(duplex, duplex.tokenizer, set(duplex.special_token_ids) | unit_ids | {duplex.tokenizer.eos_token_id})
        self.install_greedy()

    def install_greedy(self) -> None:
        """The runtime's ``decode`` without sampling: the same masks and repetition penalty, then the argmax."""
        decoder = self.decoder

        def decode(logits: Tensor, temperature: float = 0.7, mode: str = "sampling", top_k: int = 20, top_p: float = 0.8,
                   listen_top_k: int | None = None, listen_prob_scale: float = 1.0,
                   text_repetition_penalty: float = REPETITION_PENALTY, text_repetition_window_size: int = REPETITION_WINDOW,
                   **_: object) -> Tensor:
            logits = logits.clone()
            logits[0, decoder.tts_pad_id] = -float("inf")
            if text_repetition_penalty != 1.0 and decoder.generated_tokens:
                recent = torch.tensor(sorted(set(decoder.generated_tokens[-text_repetition_window_size:])), device=logits.device)
                logits[0, recent] /= text_repetition_penalty
            if listen_prob_scale != 1.0:
                logits[0, decoder.listen_id] *= listen_prob_scale
            next_id = torch.argmax(logits, dim=-1)
            value = int(next_id.item())
            (decoder.generated_special_tokens if value in decoder._special_token_set else decoder.generated_tokens).append(value)
            return next_id

        decoder.decode = decode

    def prepare(self) -> None:
        duplex = self.duplex
        duplex.model.reset_session(reset_token2wav_cache=False)
        suffix = f"\n{self.suffix}" if self.suffix else ""
        duplex.prepare(prefix_system_prompt=f"<|im_start|>system\n{self.prefix}\n<|audio_start|>",
                       suffix_system_prompt=f"<|audio_end|>{suffix}<|im_end|>", ref_audio=self.ref_audio, prompt_wav_path=None)
        duplex._stop_async_pipeline()
        duplex._clear_queues()
        self.embeds, self.unit_time = None, 0

    def prefill(self, chunk: np.ndarray) -> None:
        assert self.duplex.streaming_prefill(chunk), "the runtime produced no unit for this chunk"
        self.embeds, self.unit_time = self.duplex.audio_embed_queue.get_nowait()

    def generate(self) -> None:
        self.duplex._llm_generate_unit(self.embeds, self.unit_time)
        empty(self.duplex.llm_to_tts_queue)  # no TTS worker consumes the unit's text

    def drain(self) -> None:
        for q in (self.duplex.llm_to_tts_queue, self.duplex.audio_embed_queue, getattr(self.duplex, "output_queue", None)):
            if q is not None:
                empty(q)

    def trim(self, ids: list[int]) -> list[int]:
        """A unit's ids are ``<unit>``, tokens, ``</unit>``: the draft is the tokens."""
        if ids and ids[0] == self.duplex._unit_start_id:
            ids = ids[1:]
        if ids and ids[-1] == self.duplex._unit_end_id:
            ids = ids[:-1]
        return ids

    def chunk_type(self, ids: list[int]) -> Kind:
        """``S`` when the unit speaks (with or without writing), ``W`` when it only writes, else ``L``."""
        speak = write = False
        channel: Literal["speak", "write"] | None = None
        for t in ids:
            if t == self.speak_id:
                channel = "speak"
            elif t in self.think_bos:
                channel = "write"
            elif t in self.think_eos or t == self.chunk_eos_id:
                channel = None
            elif t not in self.control:
                speak |= channel == "speak"
                write |= channel == "write"
        return "S" if speak else ("W" if write else "L")


def empty(q: queue.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return
