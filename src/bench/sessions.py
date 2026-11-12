"""Build the replay audio of TickTock-Bench from the public releases of the source benchmarks (docs/DATA.md).

  python -m speclisten.bench.sessions --task webq llamaq triviaqa alpaca bba fdbv1 fdbv15 fdbv3
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download

from speclisten.audio import RATE, read_mono_16k
from speclisten.paths import ROOT, path

LEAD, TAIL = int(0.5 * RATE), int(10 * RATE)
DURATION_TOLERANCE = 0.05  # seconds between the built audio and the manifest before a mismatch is reported
OPENAUDIOBENCH = {"webq": "web_questions", "llamaq": "llama_questions", "triviaqa": "trivia_qa"}
TASKS = ("webq", "llamaq", "triviaqa", "alpaca", "bba", "fdbv1", "fdbv15", "fdbv3")


def question_session(question: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros(LEAD, np.float32), question, np.zeros(TAIL, np.float32)])


def find_one(root: Path, pattern: str) -> Path:
    hits = sorted(root.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"no {pattern} under {root}")
    return hits[0]


def normalized(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


class Builder:
    def __init__(self, task: str, out_root: Path) -> None:
        self.task = task
        self.rows = [json.loads(line) for line in (ROOT / "config" / "eval" / "sessions" / f"{task}.jsonl").open()]
        self.out = out_root / task
        self.out.mkdir(parents=True, exist_ok=True)
        self.mismatches: list[str] = []

    def write(self, row: dict[str, object], wav: np.ndarray, question: str | None = None) -> None:
        sf.write(self.out / f"{row['id']}.wav", wav, RATE)
        if question is not None and normalized(question) != normalized(row.get("question")):
            self.mismatches.append(f"{row['id']}: manifest {row.get('question')!r} != source {question!r}")
        if abs(len(wav) / RATE - row["seconds"]) > DURATION_TOLERANCE:
            self.mismatches.append(f"{row['id']}: duration {len(wav) / RATE:.3f} s, manifest {row['seconds']} s")

    def openaudiobench(self) -> None:
        subset = OPENAUDIOBENCH[self.task]
        root = Path(snapshot_download("baichuan-inc/OpenAudioBench", repo_type="dataset")) / "eval_datas" / subset
        with (root / f"{subset}.csv").open(newline="", encoding="utf-8") as f:
            by_stem = {Path(r["audio_filename"]).stem: r for r in csv.DictReader(f)}
        for row in self.rows:
            stem = row["id"].split("_", 1)[1] if self.task in ("webq", "triviaqa") else row["id"]
            audio = find_one(root / "audios", f"{stem}.*")
            self.write(row, question_session(read_mono_16k(audio)), by_stem[stem].get("question"))

    def voicebench(self) -> None:
        dataset = load_dataset("hlt-lab/voicebench", "alpacaeval_full", split="test").cast_column("audio", Audio(decode=False))
        for row in self.rows:
            item = dataset[int(row["id"].rsplit("_", 1)[1])]
            source = io.BytesIO(item["audio"]["bytes"]) if item["audio"].get("bytes") else item["audio"]["path"]
            self.write(row, question_session(read_mono_16k(source)), item.get("prompt"))

    def big_bench_audio(self) -> None:
        root = Path(snapshot_download("ArtificialAnalysis/big_bench_audio", repo_type="dataset"))
        meta = {r["id"]: r for r in map(json.loads, (root / "metadata.jsonl").open())}
        for row in self.rows:
            source = meta[int(row["id"].rsplit("_", 1)[1])]
            if source["category"] != row["bba_category"]:
                self.mismatches.append(f"{row['id']}: category {source['category']} != {row['bba_category']}")
            self.write(row, question_session(read_mono_16k(root / source["file_name"])))

    def fdb_stream(self) -> None:
        """v1 / v1.5: ``<category>/<sample>/input.wav`` under ``fdb_data``; the id is ``<category>_<sample>``."""
        for row in self.rows:
            category, sample = row["id"].rsplit("_", 1)
            self.write(row, read_mono_16k(find_one(path("fdb_data"), f"{category}/{sample}/input.wav")))

    def fdb_v3(self) -> None:
        """v3: the user recording named after the session id, anywhere under ``fdb_data``."""
        for row in self.rows:
            self.write(row, read_mono_16k(find_one(path("fdb_data"), f"{row['id']}.*")))

    def run(self) -> None:
        {"webq": self.openaudiobench, "llamaq": self.openaudiobench, "triviaqa": self.openaudiobench,
         "alpaca": self.voicebench, "bba": self.big_bench_audio,
         "fdbv1": self.fdb_stream, "fdbv15": self.fdb_stream, "fdbv3": self.fdb_v3}[self.task]()
        print(f"{self.task}: {len(self.rows)} sessions -> {self.out}", flush=True)
        for mismatch in self.mismatches:
            print("  mismatch", mismatch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", nargs="+", required=True, choices=TASKS)
    parser.add_argument("--out", help="output root (default: the 'sessions' key of config/eval/local.yaml)")
    args = parser.parse_args()
    for task in args.task:
        Builder(task, Path(args.out) if args.out else path("sessions")).run()


if __name__ == "__main__":
    main()
