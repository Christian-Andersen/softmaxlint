"""Spike: does FIM mid-training actually give a model usable bidirectional scores?

Self-contained -- it does not import ``softmaxlint`` and does not touch ``src/``.

Background
----------
The linter scores each token by asking the model to Fill-In-The-Middle: the
code after a token is supposed to make an off-by-one like ``<=`` look wrong. That
only works if the checkpoint was actually trained for FIM. Having the
``<|fim_*|>`` tokens in a tokenizer proves nothing -- Qwen3 base models declare
them too and score garbage.

``TIGER-Lab/FIM-Mid-8B`` is ``Qwen3-8B`` with function-aware FIM mid-training,
so it should. This script checks that claim cheaply before anyone rewrites the
linter: it sends a few FIM prompts and asks whether the model fills the hole
with what a code model should write.

A model with no FIM training degenerates here: it keeps predicting ``\\n`` (the
statistical favourite once the ``<|fim_*|>`` markers read as noise) regardless of
the context, which is exactly the failure we measured on Qwen3.8-27B.

Run it against a running vLLM server::

    uv run -- python scripts/spike_fim_probe.py
    uv run -- python scripts/spike_fim_probe.py --base-url http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import math
import sys

PREFIX = "<|fim_prefix|>"
SUFFIX = "<|fim_suffix|>"
MIDDLE = "<|fim_middle|>"

#: (prefix, suffix, what a code model should generate into the hole)
CASES = [
    (
        "def average(numbers):\n    total = 0\n    for i in range(len(numbers)):\n        if i ",
        " len(numbers):\n            total += numbers[i]\n    return total / len(numbers)\n",
        "<",
    ),
    (
        "import os\n\ndef read_config(path):\n    with open(path) as handle:\n        return json.load(handle)\n",
        "",
        "import json\n",
    ),
    (
        "def clamp(value, low, high):\n    return max(low, min(high, ",
        "))\n",
        "value",
    ),
]


def is_degenerate(top: str) -> bool:
    """True when the model's answer is nothing but whitespace.

    Measured on a real checkpoint: TIGER-Lab/FIM-Mid-8B answers ``'\\n\\n'`` at
    91-99% for every hole, while in plain (non-FIM) mode the same checkpoint
    completes code correctly. The markers are single tokens in the tokenizer, so
    the prompt is well formed -- the model simply does not act on them.

    Matching on "whitespace only" rather than an exact token list, because the
    degenerate answer is a *run* of newlines, not a single one.
    """
    return top.strip() == ""


def probe(base_url: str, model: str, *, top_logprobs: int) -> tuple[list[dict[str, object]], str]:
    """Send each case and collect the top candidate at the hole."""
    from openai import OpenAI

    client = OpenAI(api_key="EMPTY", base_url=base_url)
    rows: list[dict[str, object]] = []

    for prefix, suffix, expected in CASES:
        prompt = f"{PREFIX}{prefix}{SUFFIX}{suffix}{MIDDLE}"
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=1,
            temperature=0,
            logprobs=top_logprobs,
        )
        choice = response.model_dump()["choices"][0]
        logprobs = choice.get("logprobs") or {}
        top = (logprobs.get("top_logprobs") or [{}])[0] or {}
        if not top:
            msg = f"server returned no candidate logprobs for {model}"
            raise RuntimeError(msg)

        best = max(top.items(), key=lambda item: item[1])
        rows.append(
            {
                "expected": expected,
                "top": best[0],
                "top_prob": math.exp(best[1]),
                "hit": expected in {token.strip() for token in top},
            }
        )

    return rows, model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:18020/v1")
    parser.add_argument("--model", default="", help="defaults to whatever the endpoint advertises")
    parser.add_argument("--top-logprobs", type=int, default=20)
    args = parser.parse_args()

    try:
        from openai import OpenAI

        model = args.model or OpenAI(api_key="EMPTY", base_url=args.base_url).models.list().data[0].id
        rows, model = probe(args.base_url, model, top_logprobs=args.top_logprobs)
    except Exception as exc:  # noqa: BLE001
        print(f"\ncould not probe {args.base_url}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(f"\nmodel: {model}\n")
    print(f"{'expected':<14} {'top candidate':<18} {'prob':>8}  in top-k?")
    for row in rows:
        print(f"{row['expected']!r:<14} {row['top']!r:<18} {row['top_prob']:>8.3f}  {'yes' if row['hit'] else 'no'}")

    hits = sum(1 for row in rows if row["hit"])
    newlines = sum(1 for row in rows if is_degenerate(str(row["top"])))

    print("\n=== VERDICT ===")
    if newlines == len(rows):
        print(f"FAIL: all {len(rows)} holes were answered with whitespace only.")
        print("That is the signature of a checkpoint with no FIM training -- the markers read as")
        print("noise, so it falls back to the most common token. Do not build a linter on this.")
        return 1
    if newlines:
        print(f"NOTE: {newlines}/{len(rows)} holes got a whitespace-only answer.")
    if hits == len(rows):
        print(f"PASS: {hits}/{len(rows)} holes filled with the expected token.")
        print("This checkpoint understands FIM. Point MODEL_NAME at it and run `just calibrate`;")
        print("the real question is then whether the false-positive rate on corpus/correct is low.")
        return 0
    print(f"PARTIAL: {hits}/{len(rows)} hits, {newlines}/{len(rows)} degenerate.")
    print("Worth calibrating anyway -- check the false-positive rate before trusting it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
