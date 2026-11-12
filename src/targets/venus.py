"""Realtime-Venus through its released model code and demo (``venus_model`` and ``venus_repo`` on ``PYTHONPATH``)."""
from __future__ import annotations

import numpy as np
import torch
from demos.model.decoding import configure_length_penalty
from transformers import AutoModel

from speclisten.paths import path
from speclisten.targets.base import Dtype
from speclisten.targets.minicpmo import MiniCPMOTarget

CONTROL = ("<|listen|>", "<|speak|>", "<|chunk_eos|>", "<|turn_eos|>", "<unit>", "</unit>", "<|tts_bos|>", "<|chunk_tts_eos|>")
LENGTH_PENALTY = 0.8  # the demo's turn-ending control (DuplexSettings.length_penalty)
MAX_SPEAK_TOKENS = 50  # the demo's per-unit cap


class Venus(MiniCPMOTarget):
    def __init__(self, dtype: Dtype = "bfloat16") -> None:
        model = AutoModel.from_pretrained(str(path("venus_model")), trust_remote_code=True, local_files_only=True,
                                          attn_implementation="sdpa", torch_dtype=getattr(torch, dtype)).eval().cuda()
        duplex = model.as_duplex(generate_audio=False)
        tokenizer = duplex.tokenizer
        self.tool_start = tokenizer.convert_tokens_to_ids("<delegate>")
        self.generate_kwargs: dict[str, object] = {}
        control = {tokenizer.convert_tokens_to_ids(n) for n in CONTROL} | {tokenizer.eos_token_id}
        super().__init__(duplex, tokenizer, control)

    def prepare(self) -> None:
        self.duplex.model.reset_session(reset_token2wav_cache=False)  # drops the previous session's streaming audio KV
        self.duplex.prepare(prefix_system_prompt="Streaming Omni Conversation.", prompt_wav_path=None)
        self.generate_kwargs, _ = configure_length_penalty(self.duplex, LENGTH_PENALTY)

    def prefill(self, chunk: np.ndarray) -> None:
        self.duplex.streaming_prefill(audio_waveform=chunk)

    def generate(self) -> None:
        self.duplex.streaming_generate(decode_mode="greedy", max_new_speak_tokens_per_chunk=MAX_SPEAK_TOKENS,
                                       prompt_wav_path=None, **self.generate_kwargs)
