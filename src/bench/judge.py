"""Response quality of a method against AR on AlpacaEval: VoiceBench's open-ended rubric, scored by an LLM judge.

  DEEPSEEK_API_KEY=... python -m speclisten.bench.judge --target gander --run OUT/gander_alpaca \\
      --sessions config/eval/sessions/alpaca.jsonl --cache OUT/judge_cache.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from speclisten.bench.report import bootstrap, load
from speclisten.paths import path

PROMPT = """
I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
Your task is to rate the model’s responses based on the provided user input transcription [Instruction] and the model’s output transcription [Response].

Please evaluate the response on a scale of 1 to 5:
1 point: The response is largely irrelevant, incorrect, or fails to address the user’s query. It may be off-topic or provide incorrect information.
2 points: The response is somewhat relevant but lacks accuracy or completeness. It may only partially answer the user’s question or include extraneous information.
3 points: The response is relevant and mostly accurate, but it may lack conciseness or include unnecessary details that don’t contribute to the main point.
4 points: The response is relevant, accurate, and concise, providing a clear answer to the user’s question without unnecessary elaboration.
5 points: The response is exceptionally relevant, accurate, and to the point. It directly addresses the user’s query in a highly effective and efficient manner, providing exactly the information needed.

Below are the transcription of user’s instruction and models’ response:
### [Instruction]: {question}
### [Response]: {prediction}

After evaluating, please output the score only without anything else.
You don’t need to provide any explanations.
"""  # VoiceBench's open-ended prompt

TOKENIZERS = {"slm": ("slm_export", ""), "gander": ("gander_thinker", ""), "venus": ("venus_model", ""),
              "lws": ("lws_assets", "tokenizer")}
JUDGE_MODEL = "deepseek-v4.1-flash"  # the paper's judge
JUDGE_URL = "https://api.deepseek.com/v1/chat/completions"
ATTEMPTS, TIMEOUT_S, SCORE_TOKENS = 5, 120, 8
SPECIAL = re.compile(r"<\|[^|>]*\|>|</?unit>")
TOOL = re.compile(r"<tool_call>.*?</tool_call>|<delegate>.*?</delegate>", re.S)
SPOKEN = re.compile(r"<\|speak\|>(.*?)(?=<\|chunk_eos\|>|<\|turn_eos\|>|</unit>|$)", re.S)


def spoken_text(target: str, tokenizer: object, chunks: list[list[int]]) -> str:
    """What the user hears: the Streaming SLM's chunk output is its spoken text; the MiniCPM-o-based models
    speak between ``<|speak|>`` and the end of the chunk."""
    parts = []
    for ids in chunks:
        text = TOOL.sub("", tokenizer.decode(ids, skip_special_tokens=False))
        parts += [text] if target == "slm" else SPOKEN.findall(text)
    return re.sub(r"\s+", " ", SPECIAL.sub("", "".join(parts))).strip()


def ask(question: str, answer: str, url: str, model: str, key: str) -> str:
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": SCORE_TOKENS, "thinking": {"type": "disabled"},
                       "messages": [{"role": "user", "content": PROMPT.format(question=question, prediction=answer)}]})
    request = urllib.request.Request(url, data=body.encode(), headers={"Authorization": f"Bearer {key}",
                                                                       "Content-Type": "application/json"})
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                return json.loads(response.read())["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
            time.sleep(2**attempt)
    raise RuntimeError(f"the judge API failed {ATTEMPTS} times")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True, choices=sorted(TOKENIZERS))
    parser.add_argument("--run", required=True, help="run directory with ar and the method's records")
    parser.add_argument("--method", default="self_silence_h_b")
    parser.add_argument("--sessions", required=True, help="manifest with the questions")
    parser.add_argument("--cache", required=True, help="jsonl of judgments, appended to; reruns cost nothing")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--judge-url", default=JUDGE_URL)
    args = parser.parse_args()
    api_key = os.environ["DEEPSEEK_API_KEY"]
    location, sub = TOKENIZERS[args.target]
    tokenizer = AutoTokenizer.from_pretrained(str(path(location) / sub), trust_remote_code=True)
    questions = {e["id"]: e["question"] for e in map(json.loads, open(args.sessions))}
    cache_path = Path(args.cache)
    cache = {c["key"]: c for c in map(json.loads, open(cache_path))} if cache_path.exists() else {}

    def judged(question: str, text: str) -> float:
        key = hashlib.sha1(f"{args.target}\0{args.judge_model}\0{question}\0{text}".encode()).hexdigest()
        if key not in cache:
            raw = ask(question, text, args.judge_url, args.judge_model, api_key)
            cache[key] = {"key": key, "target": args.target, "question": question, "text": text, "raw": raw}
            with cache_path.open("a") as f:
                f.write(json.dumps(cache[key], ensure_ascii=False) + "\n")
        score = re.search(r"[1-5](?:\.\d+)?", cache[key]["raw"])
        return float(score.group(0)) if score else float("nan")

    ar, run = load(args.run, "ar"), load(args.run, args.method)
    scores = {}
    for s in sorted(set(ar) & set(run) & set(questions)):
        text_ar = spoken_text(args.target, tokenizer, [ar[s][k]["ids"] for k in sorted(ar[s])])
        text = spoken_text(args.target, tokenizer, [run[s][k]["ids"] for k in sorted(run[s])])
        score_ar = judged(questions[s], text_ar)
        scores[s] = (score_ar, score_ar if text == text_ar else judged(questions[s], text))  # same text, same score
    sessions = [s for s, (a, b) in scores.items() if np.isfinite(a) and np.isfinite(b)]

    def mean_score(which: int) -> Callable[[list[str]], float]:
        return lambda chosen: float(np.mean([scores[s][which] for s in chosen]))

    def delta(chosen: list[str]) -> float:
        return float(np.mean([scores[s][1] - scores[s][0] for s in chosen]))

    print(json.dumps({"target": args.target, "sessions": len(sessions),
                      "ar": mean_score(0)(sessions), "ar_ci95": bootstrap(mean_score(0), sessions),
                      "method": mean_score(1)(sessions), "method_ci95": bootstrap(mean_score(1), sessions),
                      "delta": delta(sessions), "delta_ci95": bootstrap(delta, sessions)}, indent=2))


if __name__ == "__main__":
    main()
