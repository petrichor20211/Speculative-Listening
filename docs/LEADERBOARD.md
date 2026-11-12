# TickTock-Bench leaderboard

Speedup over autoregressive decoding (AR) in mean post-audio latency of **speak chunks**: 1 s chunks, greedy
decoding, batch size 1, bfloat16, one A100 80 GB, wall clock from the arrival of the chunk's audio to the end
of the Thinker's output. FDB v3 (†) reports **tool-call chunks** for the two models that issue tool calls
(Streaming SLM, Gander) and speak chunks otherwise. #MAT: output tokens per target forward pass over all
chunks, relative to AR, averaged over the tasks. Avg.: mean speedup over the seven tasks without FDB v3.
Table 2 of the paper.

| Model | Method | Web Q. | Llama Q. | TriviaQA | AlpacaEval | BBA | FDB v1 | FDB v1.5 | FDB v3† | #MAT | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Streaming SLM | PLD | 1.01 | 1.01 | 1.02 | 1.01 | 1.03 | 1.03 | 1.02 | 1.01 | 1.01 | 1.02 |
| | Token Recycling | 1.01 | 1.00 | 1.02 | 1.01 | 1.03 | 1.03 | 1.02 | 1.02 | 1.01 | 1.02 |
| | Lookahead | 0.98 | 0.98 | 0.99 | 0.97 | 0.99 | 0.98 | 1.00 | 1.07 | 1.03 | 0.99 |
| | Speculative Listening | 2.80 | 2.87 | 3.04 | 2.75 | 2.67 | 2.62 | 2.57 | 23.41 | 2.63 | 2.76 |
| Gander | PLD | 1.04 | 1.02 | 1.01 | 1.07 | 1.03 | 1.07 | 1.06 | 1.07 | 1.21 | 1.04 |
| | Token Recycling | 1.01 | 0.99 | 1.00 | 1.02 | 0.98 | 1.01 | 1.02 | 1.08 | 1.01 | 1.00 |
| | Lookahead | 1.00 | 0.97 | 0.98 | 1.00 | 0.96 | 1.01 | 1.00 | 1.12 | 1.01 | 0.99 |
| | Speculative Listening | 2.82 | 3.00 | 3.26 | 2.18 | 2.47 | 2.18 | 2.27 | 3.45 | 2.20 | 2.60 |
| Realtime-Venus | PLD | 1.11 | 1.08 | 1.10 | 1.12 | 1.11 | 1.11 | 1.11 | 1.05 | 1.23 | 1.11 |
| | Token Recycling | 1.03 | 1.01 | 1.02 | 1.01 | 1.03 | 1.01 | 1.01 | 0.99 | 1.01 | 1.02 |
| | Lookahead | 1.00 | 0.99 | 1.00 | 1.01 | 0.99 | 0.99 | 0.99 | 0.96 | 1.01 | 1.00 |
| | Speculative Listening | 2.75 | 2.81 | 2.73 | 2.85 | 2.59 | 2.57 | 2.54 | 2.41 | 2.15 | 2.69 |
| LWS | PLD | 1.09 | 1.10 | 1.11 | 1.09 | 1.17 | 1.12 | 1.09 | 1.14 | 1.11 | 1.11 |
| | Token Recycling | 1.14 | 1.13 | 1.15 | 1.12 | 1.19 | 1.17 | 1.13 | 1.17 | 1.13 | 1.15 |
| | Lookahead | 1.14 | 1.13 | 1.14 | 1.14 | 1.23 | 1.14 | 1.12 | 1.15 | 1.15 | 1.15 |
| | Speculative Listening | 2.43 | 2.95 | 2.74 | 2.24 | 3.46 | 2.37 | 2.55 | 1.60 | 2.52 | 2.68 |

User-perceived latency (ms; Table 5 of the paper). Onset: decoding delay before a response starts, averaged
over the eight tasks. Tool: tool-call latency on FDB v3.

| Model | AR onset | AR tool | Speculative Listening onset | Speculative Listening tool |
|---|---|---|---|---|
| Streaming SLM | 417 | 2623 | 74 | 112 |
| Gander | 763 | 1827 | 228 | 513 |
| Realtime-Venus | 565 | – | 241 | – |
| LWS | 1089 | – | 204 | – |

To add a row, run the protocol in [BENCH.md](BENCH.md) on the eight tasks, compute each cell with
`python -m speclisten.bench.report speedup --run <outputs>/<model>_<task> --method <method> --kind S` (`--kind T`
for tool-call chunks on FDB v3), and open a pull request with the outputs and the per-chunk records of the run.
