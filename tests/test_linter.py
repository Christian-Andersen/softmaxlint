"""Extraction and grading of token log probabilities."""

import json
import math
from collections.abc import Callable

import httpx2
import pytest

from softmaxlint.errors import SoftmaxLintError
from softmaxlint.linter import SoftmaxLinter, TokenAnalysis
from softmaxlint.settings import Settings
from tests.conftest import (
    BASE_URL,
    BUGGY_OPERATOR,
    BUGGY_SNIPPET,
    CONFIDENT_LOGPROB,
    CORRECT_OPERATOR,
    RANK_TOP,
    RANK_WRITTEN,
    SURPRISED_LOGPROB,
    StubServer,
    completion_body,
    confident_scores,
    logprobs_payload,
    split_tokens,
)

MakeLinter = Callable[..., SoftmaxLinter]


def findings(linter: SoftmaxLinter, results: list[TokenAnalysis]) -> list[tuple[str | None, str]]:
    """Grade each result, keeping only the tokens the linter objects to."""
    return [(linter.classify(res["ratio"]), res["token"]) for res in results if linter.classify(res["ratio"])]


# ------------------------------------------------------------------------------
# The demo snippet, now exercised end to end instead of hardcoded into main()
# ------------------------------------------------------------------------------
def test_off_by_one_snippet_is_flagged_on_the_boundary_operator(make_linter: MakeLinter) -> None:
    """The `<=` the model disbelieves is reported, and nothing else is."""
    linter = make_linter(logprobs_payload(confident_scores(BUGGY_SNIPPET, surprised=BUGGY_OPERATOR)))

    findings_found = findings(linter, linter.evaluate_snippet(BUGGY_SNIPPET))

    assert findings_found == [("CRITICAL", BUGGY_OPERATOR)]


def test_off_by_one_snippet_keeps_its_tokens_verbatim(make_linter: MakeLinter) -> None:
    """Analysis covers the snippet exactly, losing no whitespace or punctuation."""
    linter = make_linter(logprobs_payload(confident_scores(BUGGY_SNIPPET)))

    results = linter.evaluate_snippet(BUGGY_SNIPPET)

    assert "".join(res["token"] for res in results) == BUGGY_SNIPPET


def test_correcting_the_off_by_one_makes_the_snippet_clean(make_linter: MakeLinter) -> None:
    """Swapping `<=` for `<` removes the only finding."""
    corrected = BUGGY_SNIPPET.replace(BUGGY_OPERATOR, CORRECT_OPERATOR)
    linter = make_linter(logprobs_payload(confident_scores(corrected)))

    findings_found = findings(linter, linter.evaluate_snippet(corrected))

    assert findings_found == []


# ------------------------------------------------------------------------------
# Ratio arithmetic
# ------------------------------------------------------------------------------
def test_ratio_compares_written_token_against_the_top_candidate(make_linter: MakeLinter) -> None:
    linter = make_linter(logprobs_payload([("foo", math.log(0.1), "bar", math.log(0.5))]))

    (result,) = linter.evaluate_snippet("foo")

    assert result["token"] == "foo"
    assert result["top_token"] == "bar"
    assert result["ratio"] == pytest.approx(0.2)
    assert result["actual_prob"] == pytest.approx(0.1)
    assert result["top_prob"] == pytest.approx(0.5)
    assert result["surprisal"] == pytest.approx(-math.log(0.1))
    assert result["rank"] == RANK_WRITTEN


def test_surprisal_of_a_certain_token_is_zero(make_linter: MakeLinter) -> None:
    linter = make_linter(logprobs_payload([("foo", 0.0, "foo", 0.0)]))

    (result,) = linter.evaluate_snippet("foo")

    assert result["surprisal"] == pytest.approx(0.0)


def test_ratio_is_one_when_the_expected_token_has_zero_probability(make_linter: MakeLinter) -> None:
    """Guards the divide-by-zero branch: a ratio of 0/0 must not blow up.

    The log probability is finite but so low that it underflows to zero once
    exponentiated, which is how a vanishing probability actually arrives over
    the wire -- JSON has no infinity.
    """
    linter = make_linter(logprobs_payload([("foo", -1e308, "bar", -1e308)]))

    (result,) = linter.evaluate_snippet("foo")

    assert result["actual_prob"] == 0.0
    assert result["ratio"] == 1.0


def test_leading_token_without_a_distribution_is_skipped(make_linter: MakeLinter) -> None:
    """vLLM reports no logprobs for the first prompt token; it is not analysed."""
    linter = make_linter(logprobs_payload([("a", 0.0, "a", 0.0), ("b", 0.0, "b", 0.0)]))

    results = linter.evaluate_snippet("ab")

    assert [res["token"] for res in results] == ["a", "b"]


def test_a_prompt_with_no_distributions_at_all_yields_no_results(make_linter: MakeLinter) -> None:
    linter = make_linter(completion_body([None, None]))

    assert linter.evaluate_snippet("ab") == []


def test_one_request_is_sent_per_snippet(make_linter: MakeLinter, server: StubServer) -> None:
    linter = make_linter(logprobs_payload(confident_scores(BUGGY_SNIPPET)))

    linter.evaluate_snippet(BUGGY_SNIPPET)

    assert len(server.requests) == 1
    request = server.requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{BASE_URL}/completions"
    assert json.loads(request.content) == {
        "model": Settings(_env_file=None).model_name,
        "prompt": BUGGY_SNIPPET,
        "max_tokens": 0,
        "echo": True,
        "prompt_logprobs": 1,
    }


# ------------------------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.09, "CRITICAL"),
        (0.099, "CRITICAL"),
        (0.10, "WARNING"),  # equal to the critical threshold, so a warning not a critical
        (0.11, "WARNING"),
        (0.29, "WARNING"),
        (0.30, None),  # equal to the warning threshold, so not a warning
        (0.31, None),
        (1.0, None),
    ],
)
def test_classify_grades_ratios_against_strict_thresholds(
    settings: Settings,
    ratio: float,
    expected: str | None,
) -> None:
    linter = SoftmaxLinter(config=settings)

    assert linter.classify(ratio) == expected


def test_classify_follows_configured_thresholds() -> None:
    linter = SoftmaxLinter(config=Settings(_env_file=None, threshold_warn=0.90, threshold_crit=0.80))

    assert linter.classify(0.85) == "WARNING"


# ------------------------------------------------------------------------------
# Error branches
# ------------------------------------------------------------------------------
def test_unreachable_server_raises_a_softmaxlint_error(
    make_linter: MakeLinter,
    settings: Settings,
) -> None:
    linter = make_linter(error=httpx2.ConnectError("connection refused"))

    with pytest.raises(SoftmaxLintError, match=f"could not reach vLLM at {settings.server_base_url}"):
        linter.evaluate_snippet("foo")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (completion_body([], choices=[]), "no choices"),
        ({}, "no choices"),
        (completion_body(None), "no prompt_logprobs"),
        (completion_body([]), "no prompt_logprobs"),
    ],
)
def test_malformed_server_responses_raise_a_softmaxlint_error(
    make_linter: MakeLinter,
    body: dict[str, object],
    message: str,
) -> None:
    linter = make_linter(body)

    with pytest.raises(SoftmaxLintError, match=message):
        linter.evaluate_snippet("foo")


# ------------------------------------------------------------------------------
# Fixture helpers
# ------------------------------------------------------------------------------
def test_split_tokens_reconstructs_the_input() -> None:
    assert "".join(split_tokens(BUGGY_SNIPPET)) == BUGGY_SNIPPET


def test_split_tokens_keeps_operators_whole() -> None:
    assert "<=" in split_tokens(BUGGY_SNIPPET)


def test_logprobs_payload_marks_the_top_candidate_with_rank_one() -> None:
    body = logprobs_payload([("foo", SURPRISED_LOGPROB, CORRECT_OPERATOR, CONFIDENT_LOGPROB)])

    choices = body["choices"]
    assert isinstance(choices, list)
    position = choices[0]["prompt_logprobs"]
    assert position[0] is None  # no distribution for the first prompt token
    assert position[1] == {
        "0": {"logprob": SURPRISED_LOGPROB, "decoded_token": "foo", "rank": RANK_WRITTEN},
        "1": {"logprob": CONFIDENT_LOGPROB, "decoded_token": CORRECT_OPERATOR, "rank": RANK_TOP},
    }
