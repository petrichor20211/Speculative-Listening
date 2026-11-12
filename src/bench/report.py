"""TickTock-Bench statistics from per-chunk records (unpaired means, session-level bootstrap; docs/BENCH.md).

  python -m speclisten.bench.report speedup --run RUN --method self_silence_h_b --kind S
  python -m speclisten.bench.report acceptance --run ORACLE_RUN
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

CHUNK_MS = 1000.0
Record = dict[str, object]
Rows = dict[str, dict[int, Record]]  # session -> chunk index -> record


def load(run: Path | str, method: str) -> Rows:
    rows: Rows = collections.defaultdict(dict)
    for file in sorted(glob.glob(str(Path(run) / f"{method}.*.jsonl"))):
        for line in open(file):
            record = json.loads(line)
            rows[record["session"]][record["chunk"]] = record
    if not rows:
        raise FileNotFoundError(f"no {method}.*.jsonl in {run}")
    return dict(rows)


def drafts_ahead(method: str) -> bool:
    return method.startswith("self_silence")


def charged_ms(chunks: dict[int, Record], ahead: bool) -> dict[int, float]:
    """Post-audio latency per chunk; a drafting method also pays the drafting that overran the idle window."""
    out = {}
    for k in sorted(chunks):
        ms = chunks[k]["ms"]
        if ahead:
            previous = chunks.get(k - 1)
            idle = CHUNK_MS - (previous["ms"] + previous["post_ms"]) if previous else CHUNK_MS
            ms += max(0.0, chunks[k]["pre_ms"] - max(idle, 0.0))
        out[k] = ms
    return out


def latencies(rows: Rows, method: str, kind: str) -> dict[str, list[float]]:
    """session -> charged latencies of its chunks of ``kind`` (``all`` for every chunk)."""
    out = {}
    for session, chunks in rows.items():
        ms = charged_ms(chunks, drafts_ahead(method))
        values = [ms[k] for k, r in chunks.items() if kind == "all" or r["type"] == kind]
        if values:
            out[session] = values
    return out


def bootstrap(stat: Callable[[list[str]], float], sessions: list[str], n: int = 2000, seed: int = 0) -> list[float]:
    """95% interval of ``stat`` over sessions resampled with replacement."""
    rng = np.random.default_rng(seed)
    values = [stat([sessions[i] for i in rng.integers(len(sessions), size=len(sessions))]) for _ in range(n)]
    values = [v for v in values if np.isfinite(v)]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def speedup(ar: Rows, run: Rows, method: str, kind: str) -> dict[str, object]:
    """Mean AR latency over mean method latency on chunks of ``kind``."""
    reference, measured = latencies(ar, "ar", kind), latencies(run, method, kind)
    sessions = sorted(set(reference) | set(measured))

    def ratio(chosen: list[str]) -> float:
        ar_ms = [v for s in chosen for v in reference.get(s, [])]
        ms = [v for s in chosen for v in measured.get(s, [])]
        return float(np.mean(ar_ms) / np.mean(ms)) if ar_ms and ms else float("nan")

    return {"speedup": ratio(sessions), "ci95": bootstrap(ratio, sessions),
            "ar_ms": float(np.mean([v for x in reference.values() for v in x])),
            "ms": float(np.mean([v for x in measured.values() for v in x])), "chunks": sum(len(x) for x in measured.values())}


def miss_rate(rows: Rows, method: str, kind: str) -> float:
    values = [v for x in latencies(rows, method, kind).values() for v in x]
    return float(np.mean([v > CHUNK_MS for v in values])) if values else float("nan")


def tokens_per_forward(rows: Rows) -> float:
    """Output tokens committed per forward pass over all chunks (the end-of-chunk token is not counted)."""
    tokens = sum(max(0, len(r["ids"]) - 1) for chunks in rows.values() for r in chunks.values())
    return tokens / max(1, sum(r["forwards"] for chunks in rows.values() for r in chunks.values()))


def max_lag(rows: Rows, method: str) -> list[float]:
    """Per session, the largest backlog behind real time: b_t = max(0, b_{t-1} + l_t - chunk duration)."""
    out = []
    for chunks in rows.values():
        ms, backlog, largest = charged_ms(chunks, drafts_ahead(method)), 0.0, 0.0
        for k in sorted(chunks):
            backlog = max(0.0, backlog + ms[k] - CHUNK_MS)
            largest = max(largest, backlog)
        out.append(largest)
    return out


def perceived(rows: Rows, method: str) -> dict[str, list[float]]:
    """Response onset latency (first speak chunk after a non-speak chunk) and tool-call latency (first
    tool-call chunk after a non-tool chunk): backlog carried in plus the chunk's own latency."""
    out: dict[str, list[float]] = {"onset": [], "tool": []}
    for chunks in rows.values():
        ms, backlog, previous = charged_ms(chunks, drafts_ahead(method)), 0.0, "L"
        for k in sorted(chunks):
            kind = chunks[k]["type"]
            if kind == "S" and previous != "S":
                out["onset"].append(backlog + ms[k])
            if kind == "T" and previous != "T":
                out["tool"].append(backlog + ms[k])
            backlog = max(0.0, backlog + ms[k] - CHUNK_MS)
            previous = kind
    return out


def acceptance(oracle: Rows) -> dict[str, dict[str, object]]:
    """Chunk-level acceptance of the silence draft from an oracle run (``draft_ids`` under silence against
    ``real_ids`` under the real audio), by type of the real output and over chunks with user speech, with the
    share of mismatched chunks that diverge at the first token."""
    groups: dict[str, dict[str, list[bool]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    first: dict[str, list[bool]] = collections.defaultdict(list)
    for session, chunks in oracle.items():
        for record in chunks.values():
            draft, real = record["draft_ids"], record["real_ids"]
            ok = draft == real
            for g in ("all", record["type"]) + (("speech",) if record["speech"] else ()):
                groups[g][session].append(ok)
            if not ok:
                first[session].append(not draft or not real or draft[0] != real[0])
    return {g: session_rate(by_session) for g, by_session in list(groups.items()) + [("first token", first)] if by_session}


def session_rate(by_session: dict[str, list[bool]]) -> dict[str, object]:
    sessions = sorted(by_session)

    def rate(chosen: list[str]) -> float:
        return float(np.mean([x for s in chosen for x in by_session[s]]))

    return {"rate": rate(sessions), "ci95": bootstrap(rate, sessions), "chunks": sum(len(v) for v in by_session.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["speedup", "acceptance"])
    parser.add_argument("--run", required=True)
    parser.add_argument("--ar", help="run directory with ar.*.jsonl (default: --run)")
    parser.add_argument("--method", default="self_silence_h_b")
    parser.add_argument("--kind", default="S", choices=["L", "S", "T", "W", "all"])
    args = parser.parse_args()
    if args.what == "acceptance":
        result = acceptance(load(args.run, "self_silence_u"))
    else:
        result = speedup(load(args.ar or args.run, "ar"), load(args.run, args.method), args.method, args.kind)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
