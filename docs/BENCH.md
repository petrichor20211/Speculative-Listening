# TickTock-Bench

A benchmark for decoding latency in chunk-interleaved full-duplex speech language models: fixed session sets,
one replay protocol, one set of metrics, every decoding method run through the model's **own released
runtime** on the same hardware. It is the evaluation protocol of *Decode Before You Hear: Speculative Listening
for Chunk-Interleaved Full-Duplex Speech Language Models*. Any method that changes how a chunk is decoded can
be scored with it.

It measures latency **per time chunk**, against the chunk duration as a deadline, by chunk type (listen /
speak / tool call), together with the lag a session accumulates behind the incoming audio, the delays a user
perceives, and the chunk-level acceptance rate of a draft decoded before the audio arrives.

## Protocol

| item | setting |
|---|---|
| models | chunk-interleaved full-duplex models with a chunk duration of 1 s, through their released runtimes: Streaming SLM (from the Qwen2.5-Omni-7B Thinker), Gander, Realtime-Venus and LWS (the three built on MiniCPM-o 4.5) |
| input | the user's audio stream of a session, 16 kHz mono, replayed one chunk at a time as recorded |
| per chunk | the runtime prefills the chunk's audio and decodes the chunk's output with its own rules (control tokens, caps, masks, tool grammar); the output classifies the chunk as listen (`L`), speak (`S`) or tool call (`T`) |
| decoding | greedy, batch size 1, bfloat16, one A100 80 GB; speculative sampling as a robustness setting |
| methods | autoregressive decoding (AR); prompt lookup decoding, Token Recycling and lookahead decoding, in chain form, drafting after the audio; Speculative Listening: silence-hypothesis draft, hypothesis pool, runner-up first-token branch |
| timing | end-to-end wall clock on an exclusive GPU, from the arrival of the chunk's audio to the end of the Thinker's output, audio encoder and runtime overheads included. Drafting starts when the previous chunk's work ends and gets the rest of the chunk duration as its budget; drafting that overruns the budget is charged to the chunk at analysis time. The Thinker runs alone and is what the clock measures |
| statistics | unpaired: a method's mean is over all chunks of a type in its own run; uncertainty is a session-level bootstrap |

## Sessions

Eight tasks in three groups, 760 sessions; recipes, sources and licences in [DATA.md](DATA.md), manifests in
`config/eval/sessions/`.

| group | task | source | sessions |
|---|---|---|---|
| speech reasoning | Web Questions (`webq`) | OpenAudioBench | 100 |
| | Llama Questions (`llamaq`) | OpenAudioBench | 100 |
| | TriviaQA (`triviaqa`) | OpenAudioBench | 100 |
| | AlpacaEval (`alpaca`) | VoiceBench | 100 |
| | Big Bench Audio (`bba`) | Artificial Analysis | 60 |
| conversational dynamics | Full-Duplex-Bench v1 (`fdbv1`) | FDB | 100 (20 per category) |
| | Full-Duplex-Bench v1.5 (`fdbv15`) | FDB | 100 (25 per category) |
| agentic | Full-Duplex-Bench v3 (`fdbv3`) | FDB | 100 |

## Metrics

| metric | definition |
|---|---|
| post-audio latency | from the arrival of the chunk's audio to the end of its output, by chunk type |
| speedup | mean post-audio latency of AR divided by that of the method, on chunks of a type |
| #MAT | output tokens per target forward pass over all chunks, divided by the same quantity for AR (AR = 1.00) |
| deadline miss rate | share of chunks whose post-audio latency exceeds the chunk duration |
| maximum lag | per session, the largest backlog behind real time, `b_t = max(0, b_{t-1} + l_t - 1000 ms)` |
| response onset latency | backlog plus post-audio latency of the first speak chunk of a response |
| tool-call latency | the same for the first chunk of a tool call |
| acceptance rate | share of chunks whose output decoded under the silence hypothesis equals the output decoded with the real audio (oracle run, method `self_silence_u`), by chunk type and over chunks with user speech; and the share of mismatches that start at the first token |

The benchmark reports latency. Every method it scores is lossless, so task scores stay those of the model;
the response-quality check of the paper (`speclisten.bench.judge`) confirms it.

## Methods

| name | method |
|---|---|
| `ar` | autoregressive decoding |
| `after_audio_lookup` | prompt lookup decoding: n-grams of n <= 3 in the session's tokens, drafts of up to 10 tokens |
| `after_audio_recycle` | Token Recycling, chain form: top-8 candidates per token, drafts of up to 10 tokens |
| `after_audio_lookahead` | lookahead decoding, chain form: W = 7, N = 5 |
| `self_silence` | silence-hypothesis draft in the idle window |
| `self_silence_h` | + hypothesis pool (n-grams of n <= 4, continuations of up to 128 tokens) |
| `self_silence_h_b` | + runner-up first-token branch (**Speculative Listening**) |
| `self_silence_u` | unbudgeted silence-hypothesis draft, and the chunk decoded again with the real audio (the oracle) |

A suffix `_t<NN>` (Streaming SLM) verifies the greedy drafts by speculative sampling at temperature NN/100;
`ar_t<NN>` is the sampled reference.

## Records

A run directory holds `<method>.<shard>.jsonl`, one JSON line per chunk:

| field | meaning |
|---|---|
| `session`, `task`, `method`, `chunk` | the session id and task, the method, the chunk index |
| `type`, `ids` | `L` / `S` / `T` (LWS also `W`: a unit that only writes) and the chunk's output tokens |
| `ms`, `pre_ms`, `post_ms` | post-audio latency as measured; drafting time before the audio (the part beyond the idle window is added to `ms` by the analysis); bookkeeping after the output |
| `forwards`, `drafted`, `accepted` | post-audio forward passes; draft tokens verified and accepted |
| `draft_ids`, `branch_len`, `real_ids` | the silence-hypothesis draft, the runner-up branch's length, the oracle's real-audio output |
| `speech`, `exact` | whether the chunk's audio has user speech; whether the output equals AR's on the same session (greedy methods, when `ar` is in the same job) |

## Running

```bash
# one job: scripts/evaluate/run.sh <target> <gpu> <shard i/n> <run> <sessions.jsonl> <methods> [replay args...]
scripts/evaluate/run.sh gander 0 0/1 gander_llamaq config/eval/sessions/llamaq.jsonl ar,after_audio_lookup,self_silence_h_b
# a queue lane that works through a job file (<target> <run> <nshards> <sessions.jsonl> <methods> [replay args...]); one lane per GPU
scripts/evaluate/queue.sh scripts/evaluate/jobs/main_table.txt 0 lane0
# statistics
python -m speclisten.bench.report speedup --run OUT/gander_llamaq --method self_silence_h_b --kind S
python -m speclisten.bench.analyses OUT results/      # every table of the paper
```

`<shard i/n>` takes every n-th session starting at i; the shards of a run write `<outputs>/<run>/<method>.<i>.jsonl`
and their logs go to `<outputs>/<run>/logs/` (or `$SPECLISTEN_LOGS/<run>/`). `run.sh` puts the target's runtime
repository on `PYTHONPATH` and runs `speclisten.bench.replay` in the target's interpreter, pinned to the GPU's NUMA
node when `numactl` exists (`PIN=0` turns it off); both locations come from `config/eval/local.yaml`. Replay
arguments after the methods are passed through (`--dtype float32`, `--seed N`).

## Adding a model

A target (`src/targets/`) is a thin adapter over the model's released runtime, see the `Target`
protocol in `base.py`. It installs the `Speculator` into the runtime's LLM calls, runs one unit per chunk
(`step`), and decodes a chunk before its audio under the silence hypothesis (`draft`) without leaving a
trace. The adapter keeps the runtime's decision rules, caps and masks; greedy decoding takes the argmax of
the same masked logits the runtime would sample from. `slm.py` reimplements the Streaming SLM's loop around
the speculator; `minicpmo.py` routes the runtime's own loop through it and snapshots the session for drafts
(Gander, Realtime-Venus, LWS).

## Adding a method

A method is a drafting policy: what rides in the chunk's prefill forward (before the audio) and what is
redrafted after a rejection (`Speculator.propose`, with its policy name in the `Redraft` literal). Verification,
KV cropping, timing and the records are shared, so a new method changes how many forward passes a chunk costs
and leaves its tokens as they are. Register its name in `src/bench/replay.py` (`METHODS`); a method that
drafts in the idle window keeps the `self_silence` prefix, which tells the analysis to charge drafting overruns.

## Scope

- The user's audio is replayed as recorded, so interruptions and overlaps are those of the recording.
- Latencies are wall clock; the paper's numbers come from one otherwise idle A100 80 GB per job, and on other
  hardware the speedups carry over more closely than the latencies.
- In bfloat16, kernels vary slightly with the pass length, so a few chunks per session can diverge from AR
  under any speculative method; `exact` in the records shows it and `--dtype float32` removes it (untimed).
- The three public models run from their own releases; the Streaming SLM's weights are released separately.
