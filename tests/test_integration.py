"""Live-server integration tests.

Skipped without a running server (`just up`), so CI -- which has no GPU --
never triggers them. They run under the normal `pytest` invocation, which is
what the `test` recipe executes. Full-corpus measurement stays in
`scripts/calibrate.py`; these cover the live path only (listing, two FIM
probes, one end-to-end lint).
"""

import math

import pytest
from openai import APIConnectionError, APITimeoutError, OpenAI

from softmaxlint.linter import SoftmaxLinter
from softmaxlint.settings import Settings
from tests.conftest import BUG_DIR

PREFIX = "<|fim_prefix|>"
SUFFIX = "<|fim_suffix|>"
MIDDLE = "<|fim_middle|>"

#: (code before the hole, code after it, the token a FIM-trained model fills in)
CASES = [
    (
        "def average(numbers):\n    total = 0\n    for i in range(len(numbers)):\n        if i ",
        " len(numbers):\n            total += numbers[i]\n    return total / len(numbers)\n",
        "<",
    ),
    (
        "def clamp(value, low, high):\n    return max(low, min(high, ",
        "))\n",
        "value",
    ),
]


def live_client() -> OpenAI:
    """A client for the serving stack, or a skip when nothing is listening."""
    settings = Settings()
    client = OpenAI(api_key=settings.api_key, base_url=settings.server_base_url, timeout=10)
    try:
        client.models.list()
    except (APIConnectionError, APITimeoutError) as exc:
        pytest.skip(f"no live server at {settings.server_base_url}: {exc}")
    return client


def test_server_advertises_exactly_one_model() -> None:
    """Model discovery -- which the linter depends on -- works against it."""
    client = live_client()

    names = [entry.id for entry in client.models.list().data if entry.id]

    assert len(names) == 1, f"set MODEL_NAME to choose one of {names}"


def test_fim_holes_predict_the_expected_token() -> None:
    """The served checkpoint was actually trained for FIM (cf. Qwen3)."""
    client = live_client()
    models = [entry.id for entry in client.models.list().data if entry.id]
    assert len(models) == 1, f"set MODEL_NAME to choose one of {models}"
    model = models[0]

    for prefix, suffix, expected in CASES:
        response = client.completions.create(
            model=model,
            prompt=f"{PREFIX}{prefix}{SUFFIX}{suffix}{MIDDLE}",
            max_tokens=1,
            temperature=0,
            logprobs=20,
        )
        top = response.model_dump()["choices"][0]["logprobs"]["top_logprobs"][0]
        best, best_logprob = max(top.items(), key=lambda item: item[1])

        assert expected in {token.strip() for token in top}, f"{expected!r} not in top-20 for {prefix[-20:]!r}"
        assert best.strip() != "", "degenerate whitespace-only answer means no FIM training"
        assert math.exp(best_logprob) > 0.3, f"model unsure at a textbook hole: {best!r}"


def test_end_to_end_check_flags_a_seeded_bug() -> None:
    """The whole path -- discovery, tokenization, batched FIM, grading.

    Uses a real corpus file rather than the tiny demo snippet: the model
    needs surrounding context to be confident about the fix. ``continue``
    flags in every run, hinted or not.
    """
    live_client()  # skip when no server is listening; the linter discovers it on its own
    linter = SoftmaxLinter(config=Settings())
    results = linter.evaluate_snippet((BUG_DIR / "control_flow.py").read_text())
    findings = [(linter.classify(res), res["token"]) for res in results if linter.classify(res)]

    assert any(level == "CRITICAL" and token.strip() == "continue" for level, token in findings)
