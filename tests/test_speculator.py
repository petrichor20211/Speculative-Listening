"""Every method emits AR's tokens and leaves AR's KV on a toy deterministic LM (CPU): ``pytest tests``."""
from __future__ import annotations

import random

import torch

from speclisten.speculator import Redraft, Speculator, pool_lookup

VOCAB, STOP, CAP = 23, 0, 6
REDRAFTS: tuple[Redraft, ...] = ("lookup", "recycle", "lookahead", "pool")


def next_token(prefix: list[int]) -> int:
    return (sum((i + 1) * t for i, t in enumerate(prefix)) * 7 + len(prefix)) % VOCAB


class ToyBackend:
    """Hidden states are one-hot codes of the next token; the KV is the list of fed token ids."""

    def __init__(self) -> None:
        self.kv: list[int] = []

    def cache_length(self) -> int:
        return len(self.kv)

    def embed(self, ids: list[int]) -> torch.Tensor:
        return torch.nn.functional.one_hot(torch.tensor(ids, dtype=torch.long), VOCAB).float()

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        hidden = []
        for t in embeds.argmax(-1).tolist():
            self.kv.append(t)
            hidden.append(torch.nn.functional.one_hot(torch.tensor(next_token(self.kv)), VOCAB).float())
        return torch.stack(hidden)

    def crop(self, length: int) -> None:
        del self.kv[length:]

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * 10


def ar_chunk(prefix: list[int], audio: list[int]) -> list[int]:
    """What AR emits for a chunk: greedy tokens after the audio until STOP or the cap."""
    kv, out = prefix + audio, []
    while True:
        t = next_token(kv)
        out.append(t)
        kv.append(t)
        if t == STOP:
            return out
        if len(out) >= CAP:
            return out + [STOP]


def corrupt(ids: list[int], rng: random.Random, p: float) -> list[int]:
    return [(t + 1) % VOCAB if rng.random() < p else t for t in ids]


def replay(redraft: Redraft | None, chunks: list[list[int]], seed: int,
           silence_draft: bool) -> tuple[list[list[int]], list[int], int]:
    """Returns the emitted tokens, the final KV and the forward count. With ``silence_draft`` the chunk's
    AR output, randomly corrupted, rides in the prefill as the idle-window draft would."""
    rng = random.Random(seed)
    backend = ToyBackend()
    spec = Speculator(backend, stop={STOP}, redraft=redraft or "none")
    history: list[int] = []
    hypotheses: list[list[int]] = []
    spec.history = lambda: history
    spec.pool = lambda token: pool_lookup(history, hypotheses, STOP)
    outputs, forwards = [], 0
    for audio in chunks:
        draft = None
        if silence_draft:
            draft = corrupt(ar_chunk(list(backend.kv), audio), rng, 0.2)
            hypotheses.append(draft)
        spec.begin_chunk(draft)
        history += audio
        logits, _ = spec.feed_prefill(backend.embed(audio))
        emitted = []
        while True:
            token = int(logits.argmax(-1).item())
            emitted.append(token)
            history.append(token)
            if token == STOP:
                spec.feed_token(token, backend.embed([token])[0])
                break
            logits, _ = spec.feed_token(token, backend.embed([token])[0])
            if len(emitted) >= CAP:
                spec.feed_token(STOP, backend.embed([STOP])[0])
                emitted.append(STOP)
                history.append(STOP)
                break
        spec.drop_pending()
        forwards += spec.stats.forwards
        outputs.append(emitted)
    return outputs, list(backend.kv), forwards


def test_every_method_is_lossless() -> None:
    rng = random.Random(0)
    for trial in range(200):
        chunks = [[rng.randrange(1, VOCAB) for _ in range(rng.randrange(1, 4))] for _ in range(rng.randrange(1, 8))]
        reference, kv, _ = replay(None, chunks, trial, silence_draft=False)
        for redraft in REDRAFTS:
            for silence_draft in (False, True):
                outputs, method_kv, _ = replay(redraft, chunks, trial, silence_draft)
                assert outputs == reference, (redraft, silence_draft, trial)
                assert method_kv == kv, (redraft, silence_draft, trial)


def test_drafting_saves_forwards() -> None:
    rng = random.Random(1)
    chunks = [[rng.randrange(1, VOCAB) for _ in range(2)] for _ in range(50)]
    assert replay("pool", chunks, 1, silence_draft=True)[2] < replay(None, chunks, 1, silence_draft=False)[2]
    repetitive = [[3, 5]] * 30  # the same chunk 30 times: n-grams and recycled candidates reoccur
    ar = replay(None, repetitive, 2, silence_draft=False)[2]
    for redraft in ("lookup", "recycle", "lookahead"):
        assert replay(redraft, repetitive, 2, silence_draft=False)[2] < ar, redraft
