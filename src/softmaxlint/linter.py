"""Ask a vLLM server how surprised it was by each token of a snippet."""

import math
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from openai import OpenAI, OpenAIError

from .errors import SoftmaxLintError
from .settings import Settings

type AnomalyLevel = Literal["CRITICAL", "WARNING"]


class TokenAnalysis(TypedDict):
    """Per-token comparison of what was written against what was expected."""

    token: str
    actual_prob: float
    top_prob: float
    top_token: str
    ratio: float
    surprisal: float
    rank: int


@dataclass(frozen=True, slots=True)
class Anomaly:
    """A token whose probability fell far enough below the model's pick to report."""

    index: int
    level: AnomalyLevel
    analysis: TokenAnalysis


class SoftmaxLinter:
    """Scores source text against a local vLLM server exposed over the OpenAI API.

    The server is contacted lazily, one request per snippet, so a failing
    endpoint surfaces as a :class:`SoftmaxLintError` rather than at construction.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self.config: Settings = config if config is not None else Settings()
        self.client: OpenAI = OpenAI(base_url=self.config.server_base_url, api_key=self.config.api_key)

    def classify(self, ratio: float) -> AnomalyLevel | None:
        """Grade a written/expected probability ratio, or ``None`` if it is fine.

        Thresholds are strict: a ratio exactly equal to a threshold is not an
        anomaly at that level.
        """
        if ratio < self.config.threshold_crit:
            return "CRITICAL"
        if ratio < self.config.threshold_warn:
            return "WARNING"
        return None

    def evaluate_snippet(self, code_text: str) -> list[TokenAnalysis]:
        """Send code to vLLM and extract token-level log probabilities.

        Returns one entry per prompt token, minus the leading token for which
        the server reports no distribution.

        Raises:
            SoftmaxLintError: if the server is unreachable, returns no choices,
                or omits prompt log probabilities.
        """
        try:
            response = self.client.completions.create(
                model=self.config.model_name,
                prompt=code_text,
                max_tokens=0,
                echo=True,
                extra_body={"prompt_logprobs": 1},
            )
        except OpenAIError as exc:
            msg = f"could not reach vLLM at {self.config.server_base_url}: {exc}"
            raise SoftmaxLintError(msg) from exc

        data: dict[str, Any] = response.model_dump()
        choices: list[dict[str, Any]] | None = data.get("choices")
        if not choices:
            msg = "server returned no choices."
            raise SoftmaxLintError(msg)

        prompt_logprobs: list[dict[str, dict[str, Any]] | None] | None = choices[0].get("prompt_logprobs")
        if not prompt_logprobs:
            msg = "server returned no prompt_logprobs."
            raise SoftmaxLintError(msg)

        analysis_results: list[TokenAnalysis] = []

        for item in prompt_logprobs:
            if item is None:
                continue

            candidates: list[dict[str, Any]] = list(item.values())

            actual_entry: dict[str, Any] = next((c for c in candidates if "decoded_token" in c), candidates[0])
            top_entry: dict[str, Any] = min(candidates, key=lambda c: c.get("rank", 999))

            actual_token: str = str(actual_entry.get("decoded_token", ""))
            actual_logprob: float = float(actual_entry["logprob"])
            top_logprob: float = float(top_entry["logprob"])

            actual_prob: float = math.exp(actual_logprob)
            top_prob: float = math.exp(top_logprob)

            ratio: float = actual_prob / top_prob if top_prob > 0 else 1.0
            surprisal: float = -actual_logprob

            analysis_results.append(
                {
                    "token": actual_token,
                    "actual_prob": actual_prob,
                    "top_prob": top_prob,
                    "top_token": str(top_entry.get("decoded_token", "")),
                    "ratio": ratio,
                    "surprisal": surprisal,
                    "rank": int(actual_entry.get("rank", 1)),
                }
            )

        return analysis_results
