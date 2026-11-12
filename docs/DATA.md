# Sessions, sources, licences

TickTock-Bench ships **session manifests** (sample ids, order, metadata) and a **builder** that recreates the
replay audio from the public releases of the source benchmarks, so the audio comes from its sources.

## Manifest format

`config/eval/sessions/<task>.jsonl`, one session per line:

```json
{"id": "llama_questions_0", "task": "aqa", "wav": "llamaq/llama_questions_0.wav", "seconds": 12.52, "category": "llama_questions", "question": "What is the capital of France?"}
```

| field | meaning |
|---|---|
| `id` | session id; for the question-answering sets it carries the source sample id |
| `task` | `aqa` (question answering, conversational dynamics) or `delegation` (Full-Duplex-Bench v3): the Streaming SLM's prompt |
| `wav` | the user's audio, 16 kHz mono, relative to the `sessions` root of `config/eval/local.yaml` |
| `seconds` | duration; it fixes the number of 1 s chunks |
| `category`, `question`, `bba_category`, `n_tools` | source metadata |

## Recipes

- **Question-answering sets** (Web Questions, Llama Questions, TriviaQA from OpenAudioBench; AlpacaEval from
  VoiceBench; Big Bench Audio): 0.5 s of silence, the spoken question, 10 s of silence. 100 sessions each
  (Big Bench Audio: 60, round-robin over its categories).
- **Full-Duplex-Bench v1 and v1.5**: the released user stream (`input.wav`) as is; 100 sessions each, split
  evenly over the categories (v1: 20 x 5, v1.5: 25 x 4), lowest sample ids first.
- **Full-Duplex-Bench v3**: the released user recordings, 100 sessions.

## Building the audio

```bash
pip install -e .[sessions]
# fill fdb_data and sessions in config/eval/local.yaml, then
python -m speclisten.bench.sessions --task webq llamaq triviaqa alpaca bba   # downloads from Hugging Face
python -m speclisten.bench.sessions --task fdbv1 fdbv15 fdbv3                # from your Full-Duplex-Bench copy
```

The builder reads every manifest, locates each session's source sample by its id, applies the recipe and
writes `<sessions>/<task>/<id>.wav`. The `question` field of the manifest is checked against the source text
and mismatches are reported, so a changed upstream release is noticed. Full-Duplex-Bench is fetched by hand
from its repository (v1 / v1.5) and the download link it gives (v3) into `fdb_data`; the builder finds
`<category>/<sample>/input.wav` (v1 / v1.5) and the v3 recording named after the session id anywhere under it.

## Sources and licences

| source | used for | licence | what we ship |
|---|---|---|---|
| [OpenAudioBench](https://huggingface.co/datasets/baichuan-inc/OpenAudioBench) | Web Questions, Llama Questions, TriviaQA | see the dataset card; underlying text: TriviaQA Apache-2.0, WebQuestions and Llama Questions as released by their authors | sample ids + builder |
| [VoiceBench](https://huggingface.co/datasets/hlt-lab/voicebench) | AlpacaEval | Apache-2.0 | sample ids + builder |
| [Big Bench Audio](https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio) | BBA | MIT | sample ids + builder |
| [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) v1 / v1.5 | conversational dynamics | as stated by its authors; the v1 pause-handling and turn-taking subsets derive from CANDOR, which is access-controlled | sample ids; fetch the data from the benchmark's repository |
| Full-Duplex-Bench v3 | agentic (tool calls) | as stated by its authors; human recordings | sample ids; fetch the data from the link in the benchmark's repository |
