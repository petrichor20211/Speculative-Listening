"""Every table of the paper from the run directories of ``scripts/evaluate/jobs/`` (docs/REPRODUCTION.md).

  python -m speclisten.bench.analyses <outputs> <results dir>    -> analyses.json, analyses.txt
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from speclisten.bench.report import (
    CHUNK_MS,
    Rows,
    acceptance,
    latencies,
    load,
    max_lag,
    miss_rate,
    perceived,
    speedup,
    tokens_per_forward,
)

TASKS = ("webq", "llamaq", "triviaqa", "alpaca", "bba", "fdbv1", "fdbv15", "fdbv3")
MODELS = ("slm", "gander", "venus", "lws")
METHODS = {"PLD": "after_audio_lookup", "Token Recycling": "after_audio_recycle",
           "Lookahead": "after_audio_lookahead", "Ours": "self_silence_h_b"}
LENGTH_BINS = ((1, 1), (2, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128))
TEMPERATURES = ("30", "70", "100")
CONCURRENCY = (1, 2, 4)


def kind_of(model: str, task: str) -> str:
    """Tool-call chunks on FDB v3 for the models that issue tool calls, speak chunks otherwise."""
    return "T" if task == "fdbv3" and model in ("slm", "gander") else "S"


def mean_ms(rows: Rows, method: str, kind: str) -> tuple[float, int]:
    values = [v for x in latencies(rows, method, kind).values() for v in x]
    return (statistics.mean(values) if values else float("nan")), len(values)


Table = dict[str, object]


def main_table(runs: Path) -> Table:
    out: Table = {}
    for model in MODELS:
        for task in TASKS:
            ar = load(runs / f"{model}_{task}", "ar")
            ar_tpf = tokens_per_forward(ar)
            for name, method in METHODS.items():
                rows = load(runs / f"{model}_{task}", method)
                cell = speedup(ar, rows, method, kind_of(model, task))
                cell["mat"] = tokens_per_forward(rows) / ar_tpf
                out.setdefault(model, {}).setdefault(name, {})[task] = cell
        for name in METHODS:
            row = out[model][name]
            row["avg7"] = statistics.mean(row[t]["speedup"] for t in TASKS if t != "fdbv3")
            row["mat_avg"] = statistics.mean(row[t]["mat"] for t in TASKS)
    return out


def lag_table(runs: Path) -> Table:
    out = {}
    for model in MODELS:
        out[model] = {}
        for label, method in (("AR", "ar"), ("Ours", "self_silence_h_b")):
            lags = max_lag(load(runs / f"{model}_fdbv3", method), method)
            out[model][label] = {"mean": statistics.mean(lags), "max": max(lags),
                                 "lagging_pct": 100 * sum(x > CHUNK_MS for x in lags) / len(lags)}
    return out


def profile(runs: Path) -> Table:
    """AR latency by chunk type and the idle share of listen chunks (FDB v3; LWS on Llama Questions)."""
    out = {}
    for model in MODELS:
        task = "llamaq" if model == "lws" else "fdbv3"
        ar = load(runs / f"{model}_{task}", "ar")
        row = {"task": task}
        for kind in "LST":
            ms, n = mean_ms(ar, "ar", kind)
            row[kind] = {"ms": ms, "n": n, "miss_pct": 100 * miss_rate(ar, "ar", kind)}
        row["idle_pct"] = 100 * (1 - row["L"]["ms"] / CHUNK_MS)
        out[model] = row
    return out


def ablation(runs: Path) -> Table:
    out = {}
    for model in ("slm", "gander"):
        for task, kind in (("llamaq", "S"), ("fdbv3", "T")):
            ar = load(runs / f"{model}_{task}", "ar")
            steps = (("+ silence-hypothesis draft", f"abl_{model}_{task}", "self_silence"),
                     ("+ hypothesis pool", f"abl_{model}_{task}", "self_silence_h"),
                     ("+ runner-up branch", f"{model}_{task}", "self_silence_h_b"))
            out[f"{model}_{task}"] = {label: speedup(ar, load(runs / run, method), method, kind) for label, run, method in steps}
    return out


def user_perceived(runs: Path) -> Table:
    """Response onset latency averaged over the tasks; tool-call latency on FDB v3."""
    out = {}
    for model in MODELS:
        out[model] = {}
        for label, method in (("AR", "ar"), ("Ours", "self_silence_h_b")):
            per_task = [perceived(load(runs / f"{model}_{task}", method), method) for task in TASKS]
            onsets = [statistics.mean(p["onset"]) for p in per_task if p["onset"]]
            tool = per_task[TASKS.index("fdbv3")]["tool"]
            out[model][label] = {"onset": statistics.mean(onsets) if onsets else None,
                                 "tool": statistics.mean(tool) if tool else None}
    return out


def forwards_by_length(runs: Path) -> Table:
    """Post-audio forward passes per chunk by output length (Streaming SLM, FDB v3)."""
    out = {}
    for label, run, method in (("AR", "slm_fdbv3", "ar"), ("PLD", "slm_fdbv3", "after_audio_lookup"),
                               ("Token Recycling", "slm_fdbv3", "after_audio_recycle"),
                               ("Lookahead", "slm_fdbv3", "after_audio_lookahead"),
                               ("Silence draft", "abl_slm_fdbv3", "self_silence"), ("+ pool", "abl_slm_fdbv3", "self_silence_h"),
                               ("Ours", "slm_fdbv3", "self_silence_h_b")):
        bins: dict[str, list[int]] = {f"{a}-{b}": [] for a, b in LENGTH_BINS}
        for chunks in load(runs / run, method).values():
            for r in chunks.values():
                for a, b in LENGTH_BINS:
                    if a <= len(r["ids"]) <= b:
                        bins[f"{a}-{b}"].append(r["forwards"])
        out[label] = {k: {"passes": statistics.mean(v), "n": len(v)} for k, v in bins.items() if v}
    return out


def sampling(runs: Path) -> Table:
    out = {}
    for t in TEMPERATURES:
        ar, rows = load(runs / "sampling_slm_fdbv3", f"ar_t{t}"), load(runs / "sampling_slm_fdbv3", f"self_silence_h_b_t{t}")
        out[f"T={int(t) / 100}"] = {kind: speedup(ar, rows, f"self_silence_h_b_t{t}", kind) for kind in "ST"}
    return out


def acceptance_table(runs: Path) -> Table:
    """Chunk-level acceptance of the silence draft (Llama Questions; tool-call chunks on FDB v3) and pooled rates."""
    out = {}
    pooled = {"all": [0, 0], "first token": [0, 0]}
    for model in MODELS:
        rows = acceptance(load(runs / f"{model}_oracle_llamaq", "self_silence_u"))
        for key in pooled:
            if key in rows:
                pooled[key][0] += rows[key]["rate"] * rows[key]["chunks"]
                pooled[key][1] += rows[key]["chunks"]
        if (runs / f"{model}_oracle_fdbv3").exists():
            rows["T"] = acceptance(load(runs / f"{model}_oracle_fdbv3", "self_silence_u")).get("T")
        out[model] = rows
    out["pooled"] = {key: total / n for key, (total, n) in pooled.items() if n}
    return out


def concurrency(runs: Path) -> Table:
    """N concurrent sessions on one GPU: speak and tool-call speedup, and the tool-call deadline miss rate."""
    out = {}
    for n in CONCURRENCY:
        ar, rows = load(runs / f"conc_n{n}_slm_fdbv3", "ar"), load(runs / f"conc_n{n}_slm_fdbv3", "self_silence_h_b")
        out[f"N={n}"] = {"speak": speedup(ar, rows, "self_silence_h_b", "S"), "tool": speedup(ar, rows, "self_silence_h_b", "T"),
                         "tool_miss_ar": miss_rate(ar, "ar", "T"), "tool_miss": miss_rate(rows, "self_silence_h_b", "T")}
    return out


def render(res: dict[str, Table]) -> str:
    lines = []

    def section(title: str, key: str) -> bool:
        lines.append(f"== {title}")
        if "missing" in res[key]:
            lines.append(f"   missing: {res[key]['missing']}")
            return False
        return True

    if section("main table: speedup per task | #MAT (avg over tasks) | Avg (7 tasks without FDB v3)", "main"):
        for model in MODELS:
            for name in METHODS:
                row = res["main"][model][name]
                lines.append(f"{model:6s} {name:16s} " + " ".join(f"{row[t]['speedup']:6.2f}" for t in TASKS)
                             + f" | {row['mat_avg']:.2f} | {row['avg7']:.2f}")
        lines.append("   #MAT of Ours per task")
        for model in MODELS:
            lines.append(f"{model:6s} " + " ".join(f"{res['main'][model]['Ours'][t]['mat']:6.2f}" for t in TASKS))
    if section("maximum lag per session (FDB v3): AR mean / max / lagging% | Ours mean / max / lagging%", "lag"):
        for model in MODELS:
            ar, ours = res["lag"][model]["AR"], res["lag"][model]["Ours"]
            lines.append(f"{model:6s} {ar['mean']:6.0f} {ar['max']:6.0f} {ar['lagging_pct']:4.0f} | "
                         f"{ours['mean']:5.0f} {ours['max']:5.0f} {ours['lagging_pct']:4.0f}")
    if section("chunk profile under AR: mean ms by type, idle share of listen chunks, deadline misses", "profile"):
        for model in MODELS:
            row = res["profile"][model]
            by_kind = "  ".join(f"{kind} {row[kind]['ms']:5.0f}" for kind in "LST")
            lines.append(f"{model:6s} {row['task']:6s} {by_kind}  idle {row['idle_pct']:.1f}%  "
                         f"miss S {row['S']['miss_pct']:.1f}%  miss T {row['T']['miss_pct']:.1f}%")
    if section("ablation: speedup as components are added (speak: Llama Questions; tool: FDB v3)", "ablation"):
        for key, row in res["ablation"].items():
            lines.append(f"{key:14s} " + "  ".join(f"{name} {c['speedup']:.2f}" for name, c in row.items()))
    if section("user-perceived latency (ms): onset over tasks, tool call on FDB v3", "perceived"):
        for model in MODELS:
            cells = []
            for label in ("AR", "Ours"):
                cell = res["perceived"][model][label]
                onset, tool = (f"{cell[k]:5.0f}" if cell[k] is not None else "    -" for k in ("onset", "tool"))
                cells.append(f"{label} onset {onset} tool {tool}")
            lines.append(f"{model:6s} " + " | ".join(cells))
    if section("forward passes per chunk by output length (Streaming SLM, FDB v3)", "length"):
        for label, bins in res["length"].items():
            lines.append(f"{label:16s} " + " ".join(f"{b}:{c['passes']:.1f}({c['n']})" for b, c in bins.items()))
    if section("speculative sampling (Streaming SLM, FDB v3): speedup of speak / tool-call chunks", "sampling"):
        for key, row in res["sampling"].items():
            lines.append(f"{key:6s} speak {row['S']['speedup']:.2f}  tool {row['T']['speedup']:.2f}")
    if section("acceptance of the silence draft (%): all, listen, speak, tool call, user speech, first-token mismatches",
               "acceptance"):
        for model in MODELS:
            cells = [res["acceptance"][model].get(g) for g in ("all", "L", "S", "T", "speech", "first token")]
            lines.append(f"{model:6s} " + " ".join(f"{100 * c['rate']:5.1f}" if c else "    -" for c in cells))
        pooled = res["acceptance"]["pooled"]
        lines.append(f"pooled all {100 * pooled['all']:.1f}  first token {100 * pooled['first token']:.1f}")
    if section("concurrency (Streaming SLM, FDB v3): speak / tool speedup, tool-call miss rate AR -> Ours", "concurrency"):
        for key, row in res["concurrency"].items():
            lines.append(f"{key:4s} speak {row['speak']['speedup']:.2f}  tool {row['tool']['speedup']:.2f}  "
                         f"miss {100 * row['tool_miss_ar']:.1f}% -> {100 * row['tool_miss']:.1f}%")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, help="the outputs directory holding the run directories")
    parser.add_argument("out", type=Path, help="where analyses.json and analyses.txt are written")
    args = parser.parse_args()
    runs, out = args.runs, args.out
    out.mkdir(parents=True, exist_ok=True)
    res: dict[str, Table] = {}
    for key, table in (("main", main_table), ("lag", lag_table), ("profile", profile), ("ablation", ablation),
                       ("perceived", user_perceived), ("length", forwards_by_length), ("sampling", sampling),
                       ("acceptance", acceptance_table), ("concurrency", concurrency)):
        try:
            res[key] = table(runs)
        except FileNotFoundError as e:
            res[key] = {"missing": str(e)}
    (out / "analyses.json").write_text(json.dumps(res, indent=1))
    text = render(res)
    (out / "analyses.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
