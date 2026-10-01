"""Bidirectional scoring with Fill-In-The-Middle (FIM).

Causal log probabilities only ever look left: the distribution reported for a
token is conditioned on everything before it and nothing after it. In source
code that is the wrong prior. ``if i <= len(numbers)`` looks fine given the
loop above it, and only betrays itself once the ``total += numbers[i]`` below
is visible.

FIM fixes the conditioning by reordering the context. For each slice of the
source we ask the model to fill a hole surrounded by the rest of the file::

    <|fim_prefix|>{code before}<|fim_suffix|>{code after}<|fim_middle|>

and read the distribution of the first token the model would emit into the
hole. The whole source is submitted in one batched request; vLLM schedules the
slices concurrently, and because they share a long prefix the server's prefix
cache does most of the work.

Probing one slice per token is inherently one request per token, so the number
of slices is the thing to keep an eye on.

**Slices must land on the model's own token boundaries.** That is not a detail:
see :func:`slices_from_offsets`.
"""

from __future__ import annotations

import math
import re
from itertools import pairwise
from typing import TYPE_CHECKING, Any, NamedTuple

from .errors import SoftmaxLintError

if TYPE_CHECKING:
    from .linter import TokenAnalysis

#: Qwen-style FIM markers. StarCoder and CodeLlama spell these differently,
#: which is why they live in settings rather than being baked in here.
DEFAULT_PREFIX_MARKER = "<|fim_prefix|>"
DEFAULT_SUFFIX_MARKER = "<|fim_suffix|>"
DEFAULT_MIDDLE_MARKER = "<|fim_middle|>"

#: Fallback splitter for when the server cannot report a tokenization.
#:
#: Deliberately coarse. It merges runs of punctuation and whitespace into single
#: slices so it is far less likely to invent a boundary the model never emits,
#: but it is still not the model's tokenizer and will disagree with it.
COARSE_RE = re.compile(r"\s+|[A-Za-z_]\w*|\d+|[^\sA-Za-z_0-9]+")

SPLITTERS = {"line": re.compile(r"[^\n]*\n|[^\n]+"), "coarse": COARSE_RE}


class FimMarkers(NamedTuple):
    """The three special tokens that delimit the FIM prompt."""

    prefix: str
    suffix: str
    middle: str


class Slice(NamedTuple):
    """One token of the source, and the character span it occupies."""

    start: int
    end: int
    text: str


def slices_from_offsets(source: str, offsets: list[int]) -> list[Slice]:
    """Turn the server's token boundaries into probe slices.

    ``offsets`` are the start index of each token, as reported by the model when
    it echoes a tokenization. Every probe then sits exactly where the model
    itself would split the text, so the candidate list it hands back really
    does contain the token the author wrote.

    Getting this wrong is not subtle. Probing one character off a boundary, or
    splitting a token the model emits as one piece, means the written token is
    simply absent from the candidate list; the slice is then censored, its
    ratio comes back as zero, and every ordinary line of code reports as
    CRITICAL. Qwen's BPE also folds the preceding space into the token
    (``' total'``, ``'(n'``, ``'):\\n'``), so a probe that starts *after* the
    whitespace can never match what the model returns.

    Raises:
        SoftmaxLintError: if the offsets are not a usable tokenization.
    """
    if not offsets:
        return []

    if offsets[0] != 0 or offsets != sorted(offsets) or offsets[-1] >= len(source):
        msg = (
            "server returned an unusable tokenization "
            f"(first={offsets[0]!r}, last={offsets[-1]!r}, source length={len(source)}); "
            "softmaxlint needs echo=True text offsets to place its probes"
        )
        raise SoftmaxLintError(msg)

    boundaries = [*offsets, len(source)]
    return [Slice(start, end, source[start:end]) for start, end in pairwise(boundaries) if end > start]


def slice_source(source: str, mode: str) -> list[Slice]:
    """Split ``source`` with a local regex, for when the server cannot help.

    Kept as a degraded fallback only; see :func:`slices_from_offsets` for why
    the model's own boundaries are strongly preferred.

    Raises:
        SoftmaxLintError: if ``mode`` is not a known splitter.
    """
    pattern = SPLITTERS.get(mode)
    if pattern is None:
        msg = f"unknown fim_chunk_mode {mode!r}, expected one of {sorted(SPLITTERS)}"
        raise SoftmaxLintError(msg)

    return [Slice(match.start(), match.end(), match.group()) for match in pattern.finditer(source) if match.group()]


def build_prompt(source: str, start: int, markers: FimMarkers) -> str:
    """Wrap ``source`` so that the slice at ``start`` is the hole to fill."""
    return f"{markers.prefix}{source[:start]}{markers.suffix}{source[start:]}{markers.middle}"


def parse_offsets(choice: dict[str, Any], source: str) -> list[int]:
    """Read a tokenization out of an echoed completion.

    ``text_offset`` is the character position of each prompt token. vLLM fills
    it in for ``echo=True`` requests, which is how softmaxlint learns the
    model's own token boundaries without shipping a tokenizer of its own.

    Raises:
        SoftmaxLintError: if the server echoed no usable offsets.
    """
    logprobs: dict[str, Any] | None = choice.get("logprobs")
    if not logprobs:
        msg = "server echoed no logprobs, so its tokenization is unavailable."
        raise SoftmaxLintError(msg)

    offsets: list[int] | None = logprobs.get("text_offset")
    if not offsets:
        msg = "server echoed no text_offset, so its tokenization is unavailable."
        raise SoftmaxLintError(msg)

    usable = [offset for offset in offsets if 0 <= offset < len(source)]
    if not usable:
        msg = f"server echoed text_offset outside the snippet (source length={len(source)})."
        raise SoftmaxLintError(msg)
    return usable


def _top_logprobs(choice: dict[str, Any]) -> dict[str, float]:
    """Pull the first generated token's candidate distribution out of a choice."""
    logprobs: dict[str, Any] | None = choice.get("logprobs")
    if not logprobs:
        msg = "server returned no logprobs for the FIM middle token."
        raise SoftmaxLintError(msg)

    top: list[dict[str, float]] | None = logprobs.get("top_logprobs")
    if not top:
        msg = "server returned no candidate logprobs for the FIM middle token."
        raise SoftmaxLintError(msg)

    return top[0] or {}


def _match_source(candidates: dict[str, float], slice_: Slice) -> tuple[str, float] | None:
    """Find the candidate that corresponds to the token the author actually wrote.

    An exact match against the slice is what we want and what the server will
    supply now that probes land on real token boundaries. The prefix fallback
    covers the model choosing a different way to split the *same* text -- it
    offered ``' total'`` where the author wrote ``' total = '`` -- which is not
    an anomaly, so the longest fit is taken.
    """
    exact = candidates.get(slice_.text)
    if exact is not None:
        return slice_.text, exact

    remainder = slice_.text
    matches = [(token, logprob) for token, logprob in candidates.items() if token and remainder.startswith(token)]
    if not matches:
        return None
    return max(matches, key=lambda item: (len(item[0]), item[1]))


def parse_choice(choice: dict[str, Any], slice_: Slice) -> TokenAnalysis:
    """Turn one scored FIM prompt into a :class:`TokenAnalysis`.

    The reported ``token`` is the source text this slice covers, so the heatmap
    still reassembles into the file the user actually wrote.

    When the written token is absent from the candidates the server returned,
    its true probability is below everything we asked for. That is reported as
    ``actual_prob == 0.0`` -- a ratio of zero, which grades as CRITICAL -- and
    ``surprisal`` carries the floor we can actually vouch for: the surprisal of
    the least surprising candidate we did receive.
    """
    candidates = _top_logprobs(choice)
    if not candidates:
        msg = "server returned an empty candidate set for the FIM middle token."
        raise SoftmaxLintError(msg)

    top_token, top_logprob = max(candidates.items(), key=lambda item: item[1])
    top_prob = math.exp(top_logprob)

    match = _match_source(candidates, slice_)
    if match is None:
        return {
            "token": slice_.text,
            "actual_prob": 0.0,
            "top_prob": top_prob,
            "top_token": top_token,
            "ratio": 0.0,
            "surprisal": -min(candidates.values()),
            # Ranked below everything the server was willing to name.
            "rank": len(candidates) + 1,
        }

    actual_token, actual_logprob = match
    actual_prob = math.exp(actual_logprob)
    ordered = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    rank = next(index for index, (token, _) in enumerate(ordered, start=1) if token == actual_token)

    return {
        "token": slice_.text,
        "actual_prob": actual_prob,
        "top_prob": top_prob,
        "top_token": top_token,
        "ratio": actual_prob / top_prob if top_prob > 0 else 1.0,
        "surprisal": -actual_logprob,
        "rank": rank,
    }


def order_choices(choices: list[dict[str, Any]], expected: int) -> list[dict[str, Any]]:
    """Return choices aligned with prompt order.

    vLLM makes no promise about the order it returns a batch in, and the OpenAI
    client preserves whatever order it received, so sorting on ``index`` is
    what keeps each score attached to the slice it belongs to.

    Raises:
        SoftmaxLintError: if the server returned the wrong number of choices.
    """
    ordered = sorted(choices, key=lambda choice: choice.get("index") or 0)
    if len(ordered) != expected:
        msg = f"expected {expected} scored prompts, server returned {len(ordered)}"
        raise SoftmaxLintError(msg)
    return ordered
