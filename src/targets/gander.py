"""Gander through its released runtime (``mcpmft``, on ``PYTHONPATH``)."""
from __future__ import annotations

import numpy as np
from mcpmft.args import ModelArguments
from mcpmft.infer.common import load_for_infer
from mcpmft.infer.online import DuplexParams, OnlineRunner
from mcpmft.prompts import GANDER_DUPLEX_SYSTEM_PROMPT
from mcpmft.tool_protocol import ensure_lean_task_tools

from speclisten.paths import path
from speclisten.targets.base import Dtype
from speclisten.targets.minicpmo import MiniCPMOTarget

CONTROL = ("<|listen|>", "<|speak|>", "<|interrupt|>", "<|backchannel|>", "<|chunk_eos|>", "<|turn_eos|>", "<unit>",
           "</unit>", "<|tts_bos|>", "<|chunk_tts_eos|>")


class Gander(MiniCPMOTarget):
    def __init__(self, dtype: Dtype = "bfloat16") -> None:
        bundle = load_for_infer(ModelArguments(model_name_or_path=str(path("gander_thinker")), torch_dtype=dtype),
                                init_token2wav=False)
        # the release's serving settings (serve.example.yaml, Unit-8 checkpoint); greedy decoding ignores the
        # sampling parameters, which the runtime still requires
        self.params = DuplexParams(
            generate_audio=False, decode_mode="greedy", sliding_window_mode="context_slate",
            speak_text_tokens_per_unit=8, talker_speech_tokens_per_unit=50, max_new_speak_tokens_per_chunk=11,
            max_new_tool_tokens=256, max_tool_response_tokens=256, context_max_units=128,
            context_previous_max_tokens=1500, memory_slate_max_tokens=256, temperature=0.7, top_p=0.8, top_k=20,
        )
        runner = OnlineRunner(bundle, self.params)
        tokenizer = bundle.tokenizer
        self.tool_start = tokenizer.convert_tokens_to_ids("<tool_call>")
        self.prefix: object | None = None  # the prepared system prompt and tool schemas, captured on the first session
        control = {tokenizer.convert_tokens_to_ids(n) for n in CONTROL} | {tokenizer.eos_token_id}
        super().__init__(runner.duplex, tokenizer, control, runner)

    def prepare(self) -> None:
        if self.prefix is not None:  # the system-prompt prefill is session-independent: captured once, restored afterwards
            self.runner.restore_prefix_snapshot(self.prefix)
            return
        self.runner.prepare(system_prompt=GANDER_DUPLEX_SYSTEM_PROMPT, tools=ensure_lean_task_tools(None), ref_audio_path=None)
        self.prefix = self.runner.capture_prefix_snapshot()

    def prefill(self, chunk: np.ndarray) -> None:
        self.duplex.streaming_prefill(audio_waveform=chunk)

    def generate(self) -> None:
        params = self.params
        self.runner.streaming_generate(max_new_speak_tokens_per_chunk=params.max_new_speak_tokens_per_chunk, decode_mode="greedy",
                                       temperature=params.temperature, top_p=params.top_p, top_k=params.top_k)
