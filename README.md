<a name="readme-top"></a>

<div align="center">
  <h1 align="center">Decode Before You Hear: Speculative Listening for Chunk-Interleaved Full-Duplex Speech Language Models</h1>
</div>
<div align="center">
  <a href="https://arxiv.org/abs/2611.06982"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv Paper" height="22"></a>
  <a href="https://huggingface.co/papers/2611.06982"><img src="https://img.shields.io/badge/HuggingFace-Paper-fcc21b?style=for-the-badge&logo=huggingface&logoColor=white" alt="HF Papers" height="22"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue?style=for-the-badge" alt="License" height="22"></a>
</div>

---

Welcome to **Speculative Listening**! 👋 This repository accompanies *"Decode Before You Hear: Speculative Listening for Chunk-Interleaved Full-Duplex Speech Language Models"*. It contains **Speculative Listening**, a training-free and lossless framework that lets a chunk-interleaved full-duplex speech language model decode the next time chunk before its audio arrives, and **TickTock-Bench**, a benchmark that measures per-chunk latency of such models against the chunk deadline. The repository also provides the replay drivers for four models through their released runtimes, the session manifests and builders, and the scripts that reproduce every table of the paper.

<img src="./assets/overview.png" alt="overview" style="max-width: 70%; height: auto;">

### 🪐 Key Features <span id="key-features"></span>

👂 **Speculative Listening** During the idle time of a listen chunk, the model drafts the next chunk under the silence hypothesis; the chunk's own audio prefill verifies the draft, so a correct draft costs no extra forward pass. A runner-up branch at the first token and a cross-chunk hypothesis pool keep every draft of the session available. Greedy outputs are identical to autoregressive decoding.

⏱️ **TickTock-Bench** Eight tasks (speech reasoning, conversational dynamics, agentic tool calls), 760 sessions, four chunk-interleaved models through their own runtimes, latency per 1 s chunk by chunk type, maximum lag behind real time, response onset and tool-call latency, with a deterministic calibrated clock for shared GPUs.

🧩 **Drop-in speculator** One `Speculator` class sits inside a runtime's own decoding loop and implements autoregressive decoding, prompt lookup decoding, Token Recycling, lookahead decoding and Speculative Listening behind the same two calls, so new models and new drafting policies are a thin adapter each.

## 🔥 News <span id="news"></span>

<div style="max-height: 240px; overflow-y: auto;">

* **[2026.11]** Paper released on [arXiv](https://arxiv.org/abs/2611.06982). Code and TickTock-Bench release in preparation.

</div>

## 📑 Table of Contents <span id="table-of-contents"></span>

* [🚀 Quick Start](#quick-start)
    * [1. Installation](#installation)
    * [2. Models and Paths](#models)
    * [3. Sessions](#sessions)
    * [4. Running a Replay](#replay)
    * [5. Reproduction](#reproduction)
* [✨ How It Works](#how-it-works)
* [📊 Results](#results)
* [🌱 Acknowledgements](#acknowledgements)
* [📚 Citation](#citation)

## 🚀 Quick Start <span id="quick-start"></span>

### 1. Installation <span id="installation"></span>

The replay runs each model inside the Python environment of its released runtime, so there are three
environments, one per runtime family. The shared tools need only the packages in `requirements.txt`.

```bash
git clone https://github.com/petrichor20211/Speculative-Listening.git
cd Speculative-Listening
pip install -r requirements.txt

# Streaming SLM            -> envs/duplex.yml
# Gander, Realtime-Venus   -> envs/realtime-venus.yml   (plus the requirements of the two runtimes)
# LWS                      -> envs/lws.md               (the released demo's own virtual environment)
conda env create -f envs/duplex.yml
conda env create -f envs/realtime-venus.yml
```

Check the speculator on CPU:

```bash
python tests/test_speculator.py
```

### 2. Models and Paths <span id="models"></span>

Every location (model weights, runtime repositories, session audio, outputs, the three interpreters) is a
key of one untracked file:

```bash
cp config/eval/local.example.yaml config/eval/local.yaml
```

```yaml
# Streaming SLM (weights released separately)
duplex_repo: /path/to/Duplex
slm_checkpoint: /path/to/streaming-slm.ckpt
slm_export: /path/to/streaming_slm_export

# MiniCPM-o-based targets, each through its released runtime
venus_model: /path/to/Realtime-Venus/Realtime-Venus-Audio
venus_repo: /path/to/Realtime-Venus
gander_thinker: /path/to/Gander/thinker
gander_talker: /path/to/Gander/talker
gander_repo: /path/to/Omni-Interaction-Agent/minicpm_ft
lws_repo: /path/to/lws_demo
lws_assets: /path/to/lws-triplex-demo
minicpmo_base: /path/to/MiniCPM-o-4_5

# Data and outputs
fdb_data: /path/to/Full-Duplex-Bench
sessions: /path/to/sessions
outputs: /path/to/outputs

# Interpreters
python_duplex: /path/to/envs/duplex/bin/python
python_venus: /path/to/envs/realtime-venus/bin/python
python_lws: /path/to/lws_demo/.venv/bin/python
```

Any key can be overridden by an environment variable `SPECLISTEN_<KEY>` (upper case).

### 3. Sessions <span id="sessions"></span>

> [!TIP]
> The session manifests of the paper are in `config/eval/sessions/` (ids, order, metadata). Audio is not
> redistributed; the builder recreates it from the public releases of the source benchmarks.

```bash
# question-answering sets, downloaded from Hugging Face
python -m speclisten.bench.sessions --task webq llamaq triviaqa alpaca bba
# Full-Duplex-Bench v1 / v1.5 / v3, from your copy of the benchmark (fdb_data in the config)
python -m speclisten.bench.sessions --task fdbv1 fdbv15 fdbv3
```

A question-answering session is 0.5 s of silence, the spoken question and 10 s of silence; a
Full-Duplex-Bench session is the released user stream as is. Sources, recipes and licences are in
[docs/DATA.md](docs/DATA.md).

### 4. Running a Replay <span id="replay"></span>

One job replays one session set through one model with a list of methods and writes one JSON line per
chunk and method:

```bash
# scripts/evaluate/run.sh <target> <gpu> <shard i/n> <run> <sessions.jsonl> <methods>
MODE=wall scripts/evaluate/run.sh gander 0 0/1 gander_llamaq config/eval/sessions/llamaq.jsonl ar,after_audio_lookup,self_silence_h_b
```

* `MODE=wall` (the paper): end-to-end wall clock on an exclusive GPU, from the arrival of a chunk's audio to
  the end of the Thinker's output; the audio encoder, runtime overheads and drafting past the idle window
  are included. The Talker is not timed.
* `MODE=clock`: the calibrated clock. Forward passes are charged from `config/eval/cost_models/<target>.json`
  (measured on an A100 80 GB), so the run is deterministic and several lanes can share a GPU.
* `MODE=talker`: wall clock with each release's Talker and Token2wav after the unit, on the same GPU.

Targets are `slm`, `gander`, `venus` and `lws`. Methods:

| name | method |
|---|---|
| `ar` | autoregressive decoding |
| `after_audio_lookup` | prompt lookup decoding (PLD) |
| `after_audio_recycle` | Token Recycling, chain form |
| `after_audio_lookahead` | lookahead decoding, chain form (W = 7, N = 5) |
| `self_silence` | silence-hypothesis draft in the idle window |
| `self_silence_h` | + hypothesis pool |
| `self_silence_h_b` | + runner-up first-token branch (**Speculative Listening**) |

Statistics from the records:

```bash
python -m speclisten.eval.report speedup --ar OUT/gander_llamaq --run OUT/gander_llamaq --method self_silence_h_b --kind S
python -m speclisten.eval.report acceptance --run OUT/oracle_run        # self_silence_u records
python scripts/evaluate/compare_runs.py OUT/new_run OUT/old_run         # chunk-by-chunk agreement
```

The protocol, metrics, record format and how to add a model or a method are in [docs/BENCH.md](docs/BENCH.md).

### 5. Reproduction <span id="reproduction"></span>

The job files under `scripts/evaluate/jobs/` list every run of the paper: the main table (4 models x 8
tasks x 5 methods), the ablation and the speculative-sampling runs. A queue lane takes the next unclaimed
job from a file and runs it; start one lane per GPU.

```bash
for g in 0 1 2 3; do scripts/evaluate/queue.sh scripts/evaluate/jobs/main_table.txt $g lane$g & done
scripts/evaluate/queue.sh scripts/evaluate/jobs/ablation.txt 0 abl
scripts/evaluate/queue.sh scripts/evaluate/jobs/sampling.txt 1 sampling
```

Every table of the paper is then computed from the run directories:

```bash
python -m speclisten.eval.analyses /path/to/outputs results/     # -> results/analyses.txt, analyses.json
python scripts/evaluate/lag_audit.py /path/to/outputs             # per-chunk lag audit (FDB v3)
DEEPSEEK_API_KEY=... python -m speclisten.eval.judge --model gander --ar OUT/gander_alpaca --run OUT/gander_alpaca \
    --sessions config/eval/sessions/alpaca.jsonl --cache OUT/judge_cache.jsonl   # response quality
```

Notes on reproducing the paper's numbers (wall clock vs calibrated clock, bfloat16 non-determinism,
known differences) are in [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

## ✨ How It Works <span id="how-it-works"></span>

<img src="./assets/method.png" alt="method" style="max-width: 90%; height: auto;">

A chunk-interleaved full-duplex model listens and speaks in fixed time chunks, and each chunk must be
processed before the next one arrives. Listen chunks leave compute idle while the model waits for audio;
speak and tool-call chunks need serial generation, which delays responses and tool calls. Using the idle
time for the next chunk means decoding it before its audio arrives, under an assumed input.

- **Two observations.** Taking silence as the assumed input and comparing the output with the output under
  the real audio on four models: the next audio chunk does not change the output in 95.3% of chunks, and
  88% of the changes begin at the first output token.
- **Silence-hypothesis draft.** In the idle time, the model decodes the next chunk as if the incoming audio
  were silence. The draft rides in the chunk's audio prefill forward pass, where it is verified token by
  token; an accepted draft costs no further pass.
- **Runner-up branch.** Since changes start at the first token, a second draft continues from the runner-up
  first token.
- **Hypothesis pool.** Every draft of the session is kept; after a rejection, the continuation is proposed
  from the pool by n-gram lookup before the model falls back to one token per pass.
- **Lossless.** Verification is exact token comparison inside the runtime's own loop, so greedy outputs equal
  autoregressive decoding; speculative sampling covers the sampling case.

## 📊 Results <span id="results"></span>

Speedup over autoregressive decoding in mean post-audio latency of speak chunks (FDB v3: tool-call chunks for
the two models that issue tool calls), wall clock, one A100 80 GB. Full table and user-perceived latency in
[docs/LEADERBOARD.md](docs/LEADERBOARD.md).

| Model | Method | Web Q. | Llama Q. | TriviaQA | AlpacaEval | BBA | FDB v1 | FDB v1.5 | FDB v3 | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|
| Streaming SLM | Speculative Listening | 2.80 | 2.87 | 3.04 | 2.75 | 2.67 | 2.62 | 2.57 | 23.41 | 2.76 |
| Gander | Speculative Listening | 2.82 | 3.00 | 3.26 | 2.18 | 2.47 | 2.18 | 2.27 | 3.45 | 2.60 |
| Realtime-Venus | Speculative Listening | 2.75 | 2.81 | 2.73 | 2.85 | 2.59 | 2.57 | 2.54 | 2.41 | 2.69 |
| LWS | Speculative Listening | 2.43 | 2.95 | 2.74 | 2.24 | 3.46 | 2.37 | 2.55 | 1.60 | 2.68 |

Across the four models, Speculative Listening speeds up speak chunks by 2.6-2.8x on average and tool-call
chunks by up to 23.4x, reduces the decoding delay before a response starts by 73% on average (to 74-241 ms),
and reduces the maximum lag behind real time by 94.4%.

## 🌱 **Acknowledgements** <span id="acknowledgements"></span>

[![Gander](https://img.shields.io/badge/Model-Gander-blue?style=flat&logo=arxiv)](https://arxiv.org/abs/2609.08977)
[![Realtime-Venus](https://img.shields.io/badge/Model-Realtime--Venus-blue?style=flat&logo=arxiv)](https://arxiv.org/abs/2609.13814)
[![LWS](https://img.shields.io/badge/Model-LWS-blue?style=flat&logo=arxiv)](https://arxiv.org/abs/2606.07547)
[![MiniCPM-o](https://img.shields.io/badge/Model-MiniCPM--o-blue?style=flat&logo=github)](https://github.com/OpenBMB/MiniCPM-o)
[![OpenAudioBench](https://img.shields.io/badge/Dataset-OpenAudioBench-blue?style=flat&logo=huggingface)](https://huggingface.co/datasets/baichuan-inc/OpenAudioBench)
[![VoiceBench](https://img.shields.io/badge/Dataset-VoiceBench-blue?style=flat&logo=huggingface)](https://huggingface.co/datasets/hlt-lab/voicebench)
[![Big Bench Audio](https://img.shields.io/badge/Dataset-Big--Bench--Audio-blue?style=flat&logo=huggingface)](https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio)
[![Full-Duplex-Bench](https://img.shields.io/badge/Dataset-Full--Duplex--Bench-blue?style=flat&logo=github)](https://github.com/DanielLin94144/Full-Duplex-Bench)

The replay drivers run the released runtimes of Gander, Realtime-Venus and LWS unchanged; the baselines
follow prompt lookup decoding (Saxena, 2023), Token Recycling (Luo et al., 2024) and lookahead decoding
(Fu et al., 2024) in chain form. This project is licensed under the **Apache License 2.0**; see the
[LICENSE](./LICENSE) file. Audio of the source benchmarks is not redistributed (see [docs/DATA.md](docs/DATA.md)).

## 📚 **Citation** <span id="citation"></span>

```bibtex
@misc{bian2026decodehearspeculativelistening,
      title={Decode Before You Hear: Speculative Listening for Chunk-Interleaved Full-Duplex Speech Language Models}, 
      author={Yutong Bian and Yanglin Zhang and Yuxiang Wang and Luoyuan Zhang and Rongshen He and Zihao Wang and Junwen Qiu and Jiaqi Li and Zhizheng Wu},
      year={2026},
      eprint={2611.06982},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2611.06982}, 
}
```

---

<div align="center">
  <a href="https://github.com/petrichor20211/Speculative-Listening"><img src="https://img.shields.io/badge/⭐ Star%20us%20on%20GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/></a>
  <a href="https://github.com/petrichor20211/Speculative-Listening/issues"><img src="https://img.shields.io/badge/🐞 Report%20Issues-e74c3c?style=for-the-badge&logo=github"/></a>
</div>
