"""Score source code against a local vLLM server using Fill-In-The-Middle.

The engine asks the server where it splits a snippet into tokens, wraps each
token in a FIM prompt that carries the whole snippet as context on both sides,
submits the batch in as few requests as possible, and folds the returned
distributions back into per-token :class:`TokenAnalysis` records. Grading and
reporting are unchanged by any of that: see :meth:`SoftmaxLinter.classify` and
:mod:`softmaxlint.report`.
"""

import io
import tokenize
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from openai import NotFoundError, OpenAI, OpenAIError

from . import fim
from .errors import SoftmaxLintError
from .settings import Settings

if TYPE_CHECKING:
    from .fim import Slice

type AnomalyLevel = Literal["CRITICAL", "WARNING"]


class TokenAnalysis(TypedDict):
    """Per-slice comparison of what was written against what was expected."""

    token: str
    actual_prob: float
    top_prob: float
    top_token: str
    ratio: float
    surprisal: float
    rank: int


@dataclass(frozen=True, slots=True)
class Anomaly:
    """A slice whose probability fell far enough below the model's pick to report."""

    index: int
    level: AnomalyLevel
    analysis: TokenAnalysis


def _prose_spans(source: str) -> list[tuple[int, int]]:
    """Character spans of strings and comments in ``source``.

    Returns an empty list when the source does not tokenize, in which case
    everything scores.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError, SyntaxError, ValueError:
        return []
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    spans = []
    for token in tokens:
        if token.type not in (tokenize.STRING, tokenize.COMMENT):
            continue
        (start_line, start_col), (end_line, end_col) = token.start, token.end
        if start_line - 1 < len(offsets) and end_line - 1 < len(offsets):
            spans.append((offsets[start_line - 1] + start_col, offsets[end_line - 1] + end_col))
    return spans


class SoftmaxLinter:
    """Scores source text against a local vLLM server exposed over the OpenAI API.

    Scoring is bidirectional: every token is judged with the code after it as
    well as the code before it, which is the whole reason this uses FIM rather
    than plain prompt log probabilities.

    Probes are placed on the *model's* token boundaries, learned by asking the
    server to echo a tokenization. An unaligned probe cannot find the written
    token in the candidate list, so it is scored as censored and reported
    CRITICAL -- see :meth:`token_boundaries`.

    Unless :attr:`Settings.model_name` pins one, the served model is discovered
    from the endpoint -- see :attr:`model_name`.

    The server is contacted lazily, so a failing endpoint surfaces as a
    :class:`SoftmaxLintError` rather than at construction.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self.config: Settings = config if config is not None else Settings()
        self.client: OpenAI = OpenAI(base_url=self.config.server_base_url, api_key=self.config.api_key)
        self._discovered: str | None = None

    @property
    def model_name(self) -> str:
        """The model to request: the pinned name, else whatever the server serves.

        Discovery happens once per linter and is cached, so a run costs a single
        ``GET /v1/models``.

        Raises:
            SoftmaxLintError: if the endpoint cannot be reached, does not
                support model discovery, or advertises an ambiguous set of names.
        """
        if self.config.model_name is not None:
            return self.config.model_name
        if self._discovered is None:
            self._discovered = self._discover_model()
        return self._discovered

    def _discover_model(self) -> str:
        """Ask the endpoint which model it is serving.

        A server advertising several names is refused rather than guessed at.
        Picking one arbitrarily could silently score with a model whose
        tokenizer has no FIM support, which produces a confident-looking report
        full of nonsense.
        """
        try:
            listing = self.client.models.list()
        except NotFoundError as exc:
            msg = (
                f"{self.config.server_base_url} does not support model discovery. "
                "Set MODEL_NAME to the model it serves."
            )
            raise SoftmaxLintError(msg) from exc
        except OpenAIError as exc:
            msg = f"could not reach vLLM at {self.config.server_base_url}: {exc}"
            raise SoftmaxLintError(msg) from exc

        names = sorted({entry.id for entry in listing.data if entry.id})
        if not names:
            msg = f"{self.config.server_base_url} advertises no models. Set MODEL_NAME to name one explicitly."
            raise SoftmaxLintError(msg)
        if len(names) > 1:
            listed = ", ".join(names)
            msg = f"server advertises several models ({listed}); set MODEL_NAME to choose one."
            raise SoftmaxLintError(msg)
        return names[0]

    @property
    def markers(self) -> fim.FimMarkers:
        """The FIM delimiters this linter will write into its prompts."""
        return fim.FimMarkers(
            prefix=self.config.fim_prefix_marker,
            suffix=self.config.fim_suffix_marker,
            middle=self.config.fim_middle_marker,
        )

    def classify(self, analysis: TokenAnalysis) -> AnomalyLevel | None:
        """Grade one scored token, or ``None`` if it is fine.

        Skipped before the ratio thresholds: whitespace-only slices; tokens
        scored while the model was unpeaked (``top_prob`` below
        ``min_top_prob``); and tokens where the top pick is whitespace or a
        docstring delimiter, which asks for something to be added rather than
        grading what was written.

        Past those, thresholds are strict: a ratio exactly equal to a
        threshold is not an anomaly at that level. A censored score -- a slice
        the server could not even place in its candidate list -- comes through
        as a ratio of zero and so always grades CRITICAL.
        """
        if not analysis["token"].strip():
            return None
        if analysis["top_prob"] < self.config.min_top_prob:
            return None
        if not analysis["top_token"].strip() or analysis["top_token"].strip() in ('"""', "'''"):
            # Whitespace or a docstring delimiter on top: a request to add
            # something, not a verdict on the written token.
            return None
        ratio = analysis["ratio"]
        if ratio < self.config.threshold_crit:
            return "CRITICAL"
        if ratio < self.config.threshold_warn:
            return "WARNING"
        return None

    def _request(
        self,
        prompt: str | list[str],
        *,
        logprobs: int,
        echo: bool = False,
    ) -> list[dict[str, Any]]:
        """POST one completion request and hand back its choices."""
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=1,
                temperature=0,
                logprobs=logprobs,
                echo=echo,
            )
        except OpenAIError as exc:
            msg = f"could not reach vLLM at {self.config.server_base_url}: {exc}"
            raise SoftmaxLintError(msg) from exc

        data: dict[str, Any] = response.model_dump()
        choices: list[dict[str, Any]] | None = data.get("choices")
        if not choices:
            msg = "server returned no choices."
            raise SoftmaxLintError(msg)

        return choices

    def token_boundaries(self, source: str) -> list[Slice]:
        """Ask the server where it splits ``source`` into tokens.

        Probing has to land on the model's own boundaries: if a probe sits
        anywhere else, the token the author wrote is not in the candidate list
        that comes back, the slice gets censored, and ordinary code reports as
        CRITICAL. One cheap request per file buys exact alignment.
        """
        (choice,) = self._request(source, logprobs=0, echo=True)
        return fim.slices_from_offsets(source, fim.parse_offsets(choice, source))

    def _slices(self, source: str) -> list[Slice]:
        """Probe positions for ``source``, preferring the server's tokenization."""
        if self.config.fim_slice_mode in {"coarse", "line"}:
            return fim.slice_source(source, self.config.fim_slice_mode)
        try:
            return self.token_boundaries(source)
        except SoftmaxLintError:
            if not self.config.fim_allow_unaligned_probes:
                raise
            return fim.slice_source(source, "coarse")

    def _should_score(self, source: str, slice_: Slice) -> bool:
        """Whether ``slice_`` is worth a server round-trip.

        Whitespace is formatting, and (unless ``score_strings_and_comments``
        says otherwise) strings and comments are prose: surprisal there is not
        a code signal. Skipped slices are recorded as trivially expected, so
        this only shrinks the request count, never the report.
        """
        if not slice_.text.strip():
            return False
        if self.config.score_strings_and_comments:
            return True
        return not any(start <= slice_.start < end for start, end in _prose_spans(source))

    def evaluate_snippet(self, code_text: str) -> list[TokenAnalysis]:
        """Score every token of ``code_text`` with bidirectional context.

        Returns one record per token, in source order, so the tokens reassemble
        back into the original text.

        Raises:
            SoftmaxLintError: if the server is unreachable, cannot report its
                tokenization, returns the wrong number of choices, or omits
                candidate log probabilities.
        """
        if not code_text.strip():
            return []

        slices = self._slices(code_text)
        if not slices:
            return []

        markers = self.markers
        runnable = [index for index, slice_ in enumerate(slices) if self._should_score(code_text, slice_)]
        prompts = [fim.build_prompt(code_text, slices[index].start, markers) for index in runnable]

        scored: list[dict[str, Any]] = []
        batch_size = self.config.fim_batch_size
        for offset in range(0, len(prompts), batch_size):
            batch = prompts[offset : offset + batch_size]
            scored.extend(fim.order_choices(self._request(batch, logprobs=self.config.fim_top_logprobs), len(batch)))

        parsed = {
            index: fim.parse_choice(choice, slices[index]) for index, choice in zip(runnable, scored, strict=True)
        }
        return [
            parsed.get(
                index,
                {
                    # Unscored by policy (a string, comment, or whitespace): recorded
                    # as trivially expected so reporting still reassembles the file.
                    "token": slice_.text,
                    "actual_prob": 1.0,
                    "top_prob": 1.0,
                    "top_token": slice_.text,
                    "ratio": 1.0,
                    "surprisal": 0.0,
                    "rank": 1,
                },
            )
            for index, slice_ in enumerate(slices)
        ]
