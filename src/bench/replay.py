"""TickTock-Bench replay: every method on the same sessions, chunk by chunk, through the target's own runtime.

  python -m speclisten.bench.replay --target gander --sessions config/eval/sessions/llamaq.jsonl --out OUT \\
      --methods ar,after_audio_lookup,self_silence_h_b
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from speclisten.audio import RATE, has_speech
from speclisten.bench.report import CHUNK_MS
from speclisten.paths import session_wav
from speclisten.speculator import Redraft, Speculator, pool_lookup
from speclisten.targets.base import Target, wall_ms

TARGETS = {"slm": "StreamingSLM", "gander": "Gander", "venus": "Venus", "lws": "LWS"}  # module -> class
Record = dict[str, object]
SEED_STRIDE = 100_000  # sampling seeds of a repeat never overlap another repeat's
SAMPLING_SEED = 1000  # base of the per-session sampling seeds


@dataclass(frozen=True)
class Method:
    name: str
    silence: bool = False  # draft the chunk under the silence hypothesis in the idle window
    pool: bool = False  # keep every draft of the session in the hypothesis pool and redraft from it
    branch: bool = False  # also draft the branch from the runner-up first token
    unbudgeted: bool = False  # oracle: no budget, and the chunk decoded again with the real audio
    redraft: Redraft = "none"
    temperature: float = 0.0  # greedy drafts verified by speculative sampling at this temperature (0 = greedy)


METHODS = {
    "ar": Method("ar"),
    "after_audio_lookup": Method("after_audio_lookup", redraft="lookup"),
    "after_audio_recycle": Method("after_audio_recycle", redraft="recycle"),
    "after_audio_lookahead": Method("after_audio_lookahead", redraft="lookahead"),
    "self_silence": Method("self_silence", silence=True),
    "self_silence_h": Method("self_silence_h", silence=True, pool=True, redraft="pool"),
    "self_silence_h_b": Method("self_silence_h_b", silence=True, pool=True, branch=True, redraft="pool"),
    "self_silence_u": Method("self_silence_u", silence=True, unbudgeted=True),
}


def parse_method(name: str) -> Method:
    """``<method>`` or ``<method>_t<NN>`` (speculative sampling at temperature NN/100)."""
    match = re.fullmatch(r"(.+?)(?:_t(\d+))?", name)
    if match is None or match.group(1) not in METHODS:
        raise ValueError(f"unknown method {name!r}; known: {', '.join(METHODS)}")
    temperature = int(match.group(2)) / 100 if match.group(2) else 0.0
    return dataclasses.replace(METHODS[match.group(1)], name=name, temperature=temperature)


def has_content(draft: list[int], control: set[int]) -> bool:
    """Whether a draft is worth keeping in the hypothesis pool: more than its end token, and not control tokens only."""
    return len(draft) > 1 and any(t not in control for t in draft)


@torch.inference_mode()
def run_session(target: Target, spec: Speculator, method: Method, task: str, audio: np.ndarray, seed: int) -> list[Record]:
    """One session under one method; returns one record per chunk."""
    target.start_session(task)
    spec.reset()
    spec.redraft = method.redraft
    spec.history = target.history
    pool: list[list[int]] = []

    def propose(token: int) -> list[int]:
        history = list(target.history())
        if not history or history[-1] != token:  # a runtime may record the token after feeding it
            history.append(token)
        return pool_lookup(history, pool, target.stop)

    spec.pool = propose if method.pool else None
    generator = torch.Generator(device="cuda").manual_seed(SAMPLING_SEED + seed) if method.temperature > 0 else None
    rows = []
    busy_ms = 0.0  # how long the previous chunk's work ran into this chunk's second
    for k in range(0, len(audio), RATE):
        chunk = audio[k : k + RATE]
        t0 = wall_ms()
        draft, meta = None, {}
        if method.silence:
            budget_ms = float("inf") if method.unbudgeted else max(0.0, CHUNK_MS - busy_ms)
            draft = target.draft(chunk, budget_ms, propose if method.pool else None)
            if method.pool and has_content(draft, target.control):
                pool.append(draft)
            rest_ms = budget_ms - (wall_ms() - t0)
            if method.branch and draft and rest_ms > 0:
                branch = target.draft(chunk, rest_ms, propose, ban=draft[0])
                if has_content(branch, target.control):
                    pool.append(branch)
                meta["branch_len"] = len(branch)
            meta["draft_ids"] = draft
            if method.unbudgeted:
                meta["real_ids"] = target.draft(chunk, float("inf"), None, real=True)
        spec.begin_chunk(draft)
        t1 = wall_ms()
        ids = target.step(chunk, method.temperature, generator)
        t2 = wall_ms()
        spec.drop_pending()
        t3 = wall_ms()
        busy_ms = t3 - t1
        rows.append({"chunk": k // RATE, "type": target.chunk_type(ids), "ids": ids, "ms": t2 - t1, "pre_ms": t1 - t0,
                     "post_ms": t3 - t2, "forwards": spec.stats.forwards, "drafted": spec.stats.drafted,
                     "accepted": spec.stats.accepted, "speech": has_speech(chunk), **meta})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--sessions", required=True, help="session manifest, config/eval/sessions/<task>.jsonl")
    parser.add_argument("--out", required=True, help="run directory: <method>.<shard>.jsonl per method")
    parser.add_argument("--methods", required=True, help="comma-separated method names")
    parser.add_argument("--shard", default="0/1", help="i/n: every n-th session starting at i")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="float32 for the exactness check")
    parser.add_argument("--seed", type=int, default=0, help="offset of the sampling seeds (a repeat of a sampling run)")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index, count = (int(x) for x in args.shard.split("/"))
    sessions = [json.loads(line) for line in open(args.sessions)][index::count]
    methods = [parse_method(name) for name in args.methods.split(",")]
    target: Target = getattr(importlib.import_module(f"speclisten.targets.{args.target}"), TARGETS[args.target])(args.dtype)
    spec = Speculator(target.backend, stop={target.stop})
    target.install(spec)
    files = {m.name: (out / f"{m.name}.{index}.jsonl").open("a") for m in methods}

    def read(session: Record) -> np.ndarray:
        audio, rate = sf.read(session_wav(session["wav"]), dtype="float32")
        assert rate == RATE and audio.ndim == 1, f"{session['wav']}: 16 kHz mono expected"
        return audio

    if sessions:  # untimed warm-up of every method, so first-use costs fall on none of them
        audio = read(sessions[0])
        for method in methods:
            run_session(target, spec, method, sessions[0]["task"], audio, seed=0)
    # Long-lived model objects leave the cyclic GC's scans: a collection over them would land inside timed units.
    gc.collect()
    gc.freeze()
    for n, session in enumerate(sessions):
        audio = read(session)
        results = {}
        for method in methods if n % 2 == 0 else methods[::-1]:  # alternate the order on the wall clock
            started = time.perf_counter()
            seed = n + SEED_STRIDE * args.seed
            rows = results[method.name] = run_session(target, spec, method, session["task"], audio, seed)
            print(f"[{n + 1}/{len(sessions)}] {session['id']} {method.name}: {len(rows)} chunks, "
                  f"mean ms {np.mean([r['ms'] for r in rows]):.1f}, forwards {sum(r['forwards'] for r in rows)}, "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
        for method in methods:
            reference = results.get("ar") if method.temperature == 0 else None
            for k, row in enumerate(results[method.name]):
                row.update(session=session["id"], task=session["task"], method=method.name)
                if reference is not None:
                    row["exact"] = k < len(reference) and row["ids"] == reference[k]["ids"]
                files[method.name].write(json.dumps(row) + "\n")
            files[method.name].flush()


if __name__ == "__main__":
    main()
