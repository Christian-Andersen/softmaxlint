"""Shared fixtures.

No test contacts vLLM. Instead the real ``openai.OpenAI`` client is pointed at
an in-memory transport that answers with vLLM-shaped completion bodies, so the
tests exercise the actual request and response contract rather than a
hand-rolled fake client.
"""

from __future__ import annotations

import io
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

import httpx2
import pytest
from openai import OpenAI
from rich.console import Console

from softmaxlint import fim
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

#: What the stub endpoint claims to serve.
SERVED_MODEL = "stub-served-model"

CORRECT_DIR = Path(__file__).resolve().parent.parent / "corpus" / "correct"
BUG_DIR = Path(__file__).resolve().parent.parent / "corpus" / "bugs"

#: "# BUG: <prose naming the expected token in quotes>"
BUG_MARKER = re.compile(r"#\s*BUG:.*?[\"`]([^\"`]+)[\"`]", re.ASCII)


def expected_bug_token(path: Path) -> str:
    """The token a corpus bug file is supposed to get flagged on."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if match := BUG_MARKER.search(line):
            return match.group(1)
    return ""


DEFAULT_MARKERS = fim.FimMarkers(
    prefix=fim.DEFAULT_PREFIX_MARKER,
    suffix=fim.DEFAULT_SUFFIX_MARKER,
    middle=fim.DEFAULT_MIDDLE_MARKER,
)

#: A stand-in for the model's BPE, used only to keep the stub server's token
#: boundaries and its candidate lists consistent with each other.
#:
#: It is emphatically *not* Qwen's tokenizer -- tests/qwen_bpe.json records the
#: real thing, and test_fim.py checks us against it. What matters here is only
#: that the offsets the stub reports and the candidates it returns agree.
STUB_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\s+|[^\s\w]+|\d+")


def stub_offsets(source: str) -> list[int]:
    """Token start offsets, as an echoed completion would report them."""
    return [match.start() for match in STUB_TOKEN_RE.finditer(source) if match.end() > match.start()]


def slice_of(text: str, start: int = 0) -> fim.Slice:
    """A Slice covering ``text``, for tests that do not care about positions."""
    return fim.Slice(start, start + len(text), text)


def after_of(prompt: str, markers: fim.FimMarkers = DEFAULT_MARKERS) -> str:
    """Recover the code that a FIM prompt places after the hole.

    The hole sits at the first character of this string, which is exactly what
    the model is being asked to predict.
    """
    body = prompt[len(markers.prefix) :]
    _, separator, rest = body.partition(markers.suffix)
    if not separator:
        msg = f"prompt is not a FIM prompt: {prompt!r}"
        raise AssertionError(msg)
    return rest.removesuffix(markers.middle)


def token_analysis(token: str = "<=", *, ratio: float = 0.021) -> TokenAnalysis:
    """A single TokenAnalysis, for tests that only care about the level."""
    return {
        "token": token,
        "actual_prob": 0.02,
        "top_prob": 0.95,
        "top_token": CORRECT_OPERATOR,
        "ratio": ratio,
        "surprisal": -SURPRISED_LOGPROB,
        "rank": 2,
    }


class StubServer:
    """Replies to completion requests with either a canned body or a responder."""

    def __init__(
        self,
        body: dict[str, Any] | None = None,
        *,
        error: Exception | None = None,
        responder: Callable[[str | list[str]], dict[str, Any]] | None = None,
        models: list[str] | None = None,
        models_status: int = 200,
    ) -> None:
        self.body: dict[str, Any] = fim_body([]) if body is None else body
        self.error = error
        self.responder = responder
        self.models: list[str] = [SERVED_MODEL] if models is None else models
        self.models_status = models_status
        self.requests: list[dict[str, Any]] = []
        self.model_lookups: int = 0

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET":
            self.model_lookups += 1
            if self.models_status != 200:
                return httpx2.Response(self.models_status, json={"error": "no such endpoint"})
            return httpx2.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": name, "object": "model", "created": 0, "owned_by": "stub"} for name in self.models],
                },
            )

        payload = json.loads(request.content)
        self.requests.append(payload)
        if self.error is not None:
            raise self.error
        if self.responder is not None:
            return httpx2.Response(200, json=self.responder(payload["prompt"]))
        return httpx2.Response(200, json=self.body)

    def client(self) -> OpenAI:
        """A real OpenAI client wired to this server instead of a socket."""
        return OpenAI(
            api_key="EMPTY",
            base_url=BASE_URL,
            http_client=httpx2.Client(transport=httpx2.MockTransport(self)),
        )

    @property
    def prompts_sent(self) -> list[list[str]]:
        """The prompt list of every request received."""
        return [request["prompt"] for request in self.requests]


def fim_choice(index: int, candidates: dict[str, float], *, with_logprobs: bool = True) -> dict[str, Any]:
    """One completion choice carrying a single generated token's candidates."""
    top_token = max(candidates, key=lambda token: candidates[token]) if candidates else ""
    choice: dict[str, Any] = {
        "index": index,
        "text": top_token,
        "finish_reason": "length",
    }
    if with_logprobs:
        choice["logprobs"] = {
            "text_offset": [0],
            "tokens": [top_token],
            "token_logprobs": [candidates.get(top_token, 0.0)],
            "top_logprobs": [dict(candidates)],
        }
    return choice


def echo_choice(source: str, *, with_offsets: bool = True) -> dict[str, Any]:
    """One completion choice echoing ``source`` back as a tokenization."""
    offsets = stub_offsets(source) if with_offsets else []
    choice: dict[str, Any] = {"index": 0, "text": "", "finish_reason": "length"}
    if with_offsets:
        choice["logprobs"] = {
            "text_offset": offsets,
            "tokens": [
                source[offset : (offsets[i + 1] if i + 1 < len(offsets) else len(source))]
                for i, offset in enumerate(offsets)
            ],
            "token_logprobs": [0.0] * len(offsets),
            "top_logprobs": [],
        }
    return choice


def fim_body(choices: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    """A vLLM completion body with the given choices."""
    body: dict[str, Any] = {
        "id": "cmpl-stub",
        "object": "text_completion",
        "created": 0,
        "model": "stub-model",
        "choices": choices,
        "usage": {"prompt_tokens": 0, "completion_tokens": len(choices), "total_tokens": len(choices)},
    }
    body.update(overrides)
    return body


@dataclass
class FimStub:
    """A stub vLLM that answers both kinds of request softmaxlint makes.

    The tokenization request (a bare string prompt) is echoed back with
    ``text_offset``, so the linter's probes land on boundaries this stub's
    candidate lists agree with. The scoring requests (a list of FIM prompts) are
    answered from the source each prompt carries.

    ``surprising`` names a token the model disbelieves: it is returned as a
    candidate with a low logprob, behind ``alternative``. Setting ``censored``
    drops it from the candidate list altogether, which is what the server does
    when a token falls outside the top-k we asked for. Note that censoring only
    bites if ``alternative`` is not itself a prefix of ``surprising`` -- ``<``
    still "matches" a written ``<=``, so use something that cannot appear there.

    ``reverse_choices`` returns the batch backwards while keeping each choice's
    ``index`` attached to its own prompt, reproducing the out-of-order batches
    real servers produce.
    """

    surprising: str = ""
    alternative: str = CORRECT_OPERATOR
    censored: bool = False
    reverse_choices: bool = False
    echo_offsets: bool = True
    offset_shift: int = 0
    """Shift every echoed boundary, to simulate a server whose tokenization the
    linter's probes disagree with."""
    markers: fim.FimMarkers = DEFAULT_MARKERS

    def with_offset(self, shift: int) -> FimStub:
        """A copy whose echoed boundaries are shifted by ``shift`` characters."""
        return replace(self, offset_shift=shift)

    _offsets: dict[str, list[int]] = field(default_factory=dict, init=False, repr=False)

    def boundaries(self, source: str) -> list[int]:
        """Token offsets for ``source``, with any configured shift applied.

        The shift only moves interior boundaries: the first one stays at 0 so
        the result is still a plausible tokenization, which is what makes the
        linter trust it and probe one character off.
        """
        if source not in self._offsets:
            offsets = stub_offsets(source)
            self._offsets[source] = [
                offset + (self.offset_shift if index else 0) for index, offset in enumerate(offsets)
            ]
        return self._offsets[source]

    def candidates(self, prompt: str) -> dict[str, float]:
        after = after_of(prompt, self.markers)
        written = STUB_TOKEN_RE.match(after)
        if self.surprising and after.startswith(self.surprising):
            if self.censored:
                return {self.alternative: CONFIDENT_LOGPROB}
            return {self.alternative: CONFIDENT_LOGPROB, self.surprising: SURPRISED_LOGPROB}
        return {(written.group() if written else after): CONFIDENT_LOGPROB}

    def __call__(self, prompt: str | list[str]) -> dict[str, Any]:
        if isinstance(prompt, str):
            choice = echo_choice(prompt, with_offsets=self.echo_offsets)
            if self.echo_offsets:
                choice["logprobs"]["text_offset"] = self.boundaries(prompt)
            return fim_body([choice])

        choices = [fim_choice(index, self.candidates(item)) for index, item in enumerate(prompt)]
        if self.reverse_choices:
            choices.reverse()
        return fim_body(choices)


def fim_responder(**kwargs: Any) -> FimStub:
    """A responder built from :class:`FimStub` keyword arguments."""
    return FimStub(**kwargs)


def echo_then(body: dict[str, Any]) -> Callable[[str | list[str]], dict[str, Any]]:
    """Answer the tokenization request honestly, then serve ``body`` for scoring."""

    def respond(prompt: str | list[str]) -> dict[str, Any]:
        if isinstance(prompt, str):
            return fim_body([echo_choice(prompt)])
        return body

    return respond


def fixed_candidates(written: float, expected: float) -> Callable[[str | list[str]], dict[str, Any]]:
    """Answer every prompt with the same two candidates: ``foo`` and ``bar``.

    Handy for pinning an exact ratio without reasoning about the snippet.
    """

    def respond(prompt: str | list[str]) -> dict[str, Any]:
        if isinstance(prompt, str):
            return fim_body([echo_choice(prompt)])
        return fim_body([fim_choice(index, {"foo": written, "bar": expected}) for index in range(len(prompt))])

    return respond


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
    """A vLLM endpoint with no responder configured yet."""
    return StubServer()


@pytest.fixture
def make_linter(settings: Settings, server: StubServer) -> Callable[..., SoftmaxLinter]:
    """Build a SoftmaxLinter talking to the stub server instead of a real one.

    Keyword arguments other than ``error`` and ``responder`` are applied to the
    linter's settings, so a test can change chunking or thresholds.
    """

    def _make(
        *,
        error: Exception | None = None,
        responder: Callable[[str | list[str]], dict[str, Any]] | None = None,
        models: list[str] | None = None,
        models_status: int = 200,
        **overrides: Any,
    ) -> SoftmaxLinter:
        server.error = error
        server.responder = responder
        if models is not None:
            server.models = models
        server.models_status = models_status
        linter = SoftmaxLinter(config=settings.model_copy(update=overrides))
        linter.client = server.client()
        return linter

    return _make
