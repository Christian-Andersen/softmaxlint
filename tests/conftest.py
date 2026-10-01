"""Shared fixtures.

No test contacts vLLM. Instead the real ``openai.OpenAI`` client is pointed at
an in-memory transport that replies with vLLM's completion payload, so the
tests exercise the actual request and response contract rather than a
hand-rolled fake client.
"""

from __future__ import annotations

import io
import math
import re
from collections.abc import Callable
from typing import Any, NamedTuple

import httpx2
import pytest
from openai import OpenAI
from rich.console import Console

from softmaxlint.linter import SoftmaxLinter, TokenAnalysis
from softmaxlint.settings import Settings

# The snippet that used to be hardcoded into main() as the demo input. It lives
# here now as test data: an off-by-one boundary bug, `<=` where `<` belongs.
BUGGY_SNIPPET = (
    "def calculate_average(numbers):\n"
    "    total = 0\n"
    "    for i in range(len(numbers)):\n"
    "        if i <= len(numbers):\n"
    "            total += numbers[i]\n"
    "    return total / len(numbers)\n"
)
BUGGY_OPERATOR = "<="
CORRECT_OPERATOR = "<"

#: Log probabilities standing in for "the model fully expected this token".
CONFIDENT_LOGPROB = math.log(0.95)
#: ...and for "the model barely believed this token".
SURPRISED_LOGPROB = math.log(0.02)

BASE_URL = "http://localhost:8000/v1"

#: vLLM numbers logprob entries by how likely the token was overall.
RANK_TOP = 1
RANK_WRITTEN = 2

_TOKEN_RE = re.compile(r"\w+|\s+|[^\w\s]+")

# (written token, its logprob, top token, top token's logprob)
type Score = tuple[str, float, str, float]


def split_tokens(code: str) -> list[str]:
    """Split code into tokens that concatenate back into the original string."""
    return _TOKEN_RE.findall(code)


def completion_body(
    prompt_logprobs: list[dict[str, dict[str, Any]] | None] | None,
    **overrides: object,
) -> dict[str, Any]:
    """A vLLM ``/completions`` response body carrying prompt log probabilities.

    The leading ``None`` mirrors the server, which reports no distribution for
    the first prompt token. Each position maps a token id to a logprob, and the
    entry with ``rank == 1`` is the token the model actually wanted.
    """
    body = {
        "id": "cmpl-stub",
        "object": "text_completion",
        "created": 0,
        "model": "stub-model",
        "choices": [
            {
                "index": 0,
                "text": "",
                "logprobs": None,
                "finish_reason": "length",
                "prompt_logprobs": prompt_logprobs,
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    body.update(overrides)
    return body


def logprobs_payload(scores: list[Score]) -> dict[str, Any]:
    """``completion_body`` for ``scores``, prefixed with the leading ``None``."""
    positions: list[dict[str, dict[str, Any]] | None] = [None]
    for token, actual_logprob, top_token, top_logprob in scores:
        positions.append(
            {
                "0": {"logprob": actual_logprob, "decoded_token": token, "rank": RANK_WRITTEN},
                "1": {"logprob": top_logprob, "decoded_token": top_token, "rank": RANK_TOP},
            }
        )
    return completion_body(positions)


def confident_scores(code: str, *, surprised: str = "") -> list[Score]:
    """Score ``code`` as fully expected, except for tokens equal to ``surprised``."""
    return [
        (token, SURPRISED_LOGPROB, CORRECT_OPERATOR, CONFIDENT_LOGPROB)
        if token == surprised
        else (token, CONFIDENT_LOGPROB, token, CONFIDENT_LOGPROB)
        for token in split_tokens(code)
    ]


def token_analysis(token: str = "<=", *, ratio: float = 0.021) -> TokenAnalysis:
    """A single TokenAnalysis, for tests that only care about the level."""
    return {
        "token": token,
        "actual_prob": 0.02,
        "top_prob": 0.95,
        "top_token": CORRECT_OPERATOR,
        "ratio": ratio,
        "surprisal": -SURPRISED_LOGPROB,
        "rank": RANK_WRITTEN,
    }


class StubServer:
    """Replies to any completion request with a canned body."""

    def __init__(self, body: dict[str, Any] | None = None, *, error: Exception | None = None) -> None:
        self.body: dict[str, Any] = completion_body([]) if body is None else body
        self.error = error
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return httpx2.Response(200, json=self.body)

    def client(self) -> OpenAI:
        """A real OpenAI client wired to this server instead of a socket."""
        return OpenAI(
            api_key="EMPTY",
            base_url=BASE_URL,
            http_client=httpx2.Client(transport=httpx2.MockTransport(self)),
        )


class Recorder(NamedTuple):
    """A Console plus the buffer it writes to, for asserting on rendered text."""

    console: Console
    buffer: io.StringIO

    @property
    def text(self) -> str:
        return self.buffer.getvalue()


@pytest.fixture
def settings() -> Settings:
    """Settings that ignore ``.env``, so tests do not depend on the environment."""
    return Settings(_env_file=None)


@pytest.fixture
def console() -> Recorder:
    """A Console writing to memory with colour and width pinned for stable output."""
    buffer = io.StringIO()
    return Recorder(Console(file=buffer, no_color=True, width=120), buffer)


@pytest.fixture
def server() -> StubServer:
    """A vLLM endpoint that replies with an empty, well-formed completion."""
    return StubServer()


@pytest.fixture
def make_linter(settings: Settings, server: StubServer) -> Callable[..., SoftmaxLinter]:
    """Build a SoftmaxLinter talking to the stub server instead of a real one.

    The returned linter keeps a reference to the server so tests can assert on
    what was sent.
    """

    def _make(body: dict[str, Any] | None = None, *, error: Exception | None = None) -> SoftmaxLinter:
        server.body = completion_body([]) if body is None else body
        server.error = error
        linter = SoftmaxLinter(config=settings)
        linter.client = server.client()
        return linter

    return _make
