"""Orchestration of FIM scoring, and grading of the results."""

from collections.abc import Callable

import httpx2
import pytest

from softmaxlint.errors import SoftmaxLintError
from softmaxlint.linter import SoftmaxLinter, TokenAnalysis
from softmaxlint.settings import Settings
from tests.conftest import (
    BUGGY_OPERATOR,
    BUGGY_SNIPPET,
    CONFIDENT_LOGPROB,
    DEFAULT_MARKERS,
    SERVED_MODEL,
    StubServer,
    after_of,
    echo_then,
    fim_body,
    fim_choice,
    fim_responder,
    token_analysis,
)

MakeLinter = Callable[..., SoftmaxLinter]


def graded(linter: SoftmaxLinter, results: list[TokenAnalysis]) -> list[tuple[str | None, str]]:
    """Grade each result, keeping only the slices the linter objects to."""
    return [(linter.classify(res), res["token"]) for res in results if linter.classify(res)]


# ------------------------------------------------------------------------------
# The demo snippet, now exercised end to end instead of hardcoded into main()
# ------------------------------------------------------------------------------
def test_off_by_one_snippet_is_flagged_on_the_boundary_operator(make_linter: MakeLinter) -> None:
    """The `<=` the model disbelieves is reported, and nothing else is."""
    linter = make_linter(responder=fim_responder(surprising=BUGGY_OPERATOR))

    findings = graded(linter, linter.evaluate_snippet(BUGGY_SNIPPET))

    assert findings == [("CRITICAL", BUGGY_OPERATOR)]


def test_off_by_one_snippet_keeps_its_slices_verbatim(make_linter: MakeLinter) -> None:
    """Scoring covers the snippet exactly, losing no whitespace or punctuation."""
    linter = make_linter(responder=fim_responder())

    results = linter.evaluate_snippet(BUGGY_SNIPPET)

    assert "".join(res["token"] for res in results) == BUGGY_SNIPPET


def test_correcting_the_off_by_one_makes_the_snippet_clean(make_linter: MakeLinter) -> None:
    """Swapping `<=` for `<` removes the only finding."""
    corrected = BUGGY_SNIPPET.replace(BUGGY_OPERATOR, "<")
    linter = make_linter(responder=fim_responder(surprising=BUGGY_OPERATOR))

    findings = graded(linter, linter.evaluate_snippet(corrected))

    assert findings == []


def test_a_token_the_server_cannot_place_is_still_caught(make_linter: MakeLinter) -> None:
    """A token outside the candidate list comes back censored, so it grades CRITICAL."""
    linter = make_linter(responder=fim_responder(surprising=BUGGY_OPERATOR, censored=True, alternative="while"))

    results = linter.evaluate_snippet(BUGGY_SNIPPET)

    assert graded(linter, results) == [("CRITICAL", BUGGY_OPERATOR)]
    assert next(res for res in results if res["token"] == BUGGY_OPERATOR)["actual_prob"] == 0.0


def test_an_alternative_that_is_only_a_prefix_masks_the_anomaly(make_linter: MakeLinter) -> None:
    """Next-token scoring cannot see past the first token, so a shorter candidate
    that fits the written token looks like a match. Documented, not fixed: the
    deviation shows up on the following probe instead."""
    linter = make_linter(responder=fim_responder(surprising=BUGGY_OPERATOR, censored=True, alternative="<"))

    results = linter.evaluate_snippet(BUGGY_SNIPPET)

    assert next(res for res in results if res["token"] == BUGGY_OPERATOR)["ratio"] == pytest.approx(1.0)


def test_line_slicing_scores_line_starts_only(make_linter: MakeLinter) -> None:
    """Documents the tradeoff behind ``fim_slice_mode="line"``: the hole is only
    ever placed at a line start, so a mid-line operator is never probed."""
    linter = make_linter(responder=fim_responder(surprising=BUGGY_OPERATOR), fim_slice_mode="line")

    results = linter.evaluate_snippet(BUGGY_SNIPPET)

    assert graded(linter, results) == []
    assert "".join(res["token"] for res in results) == BUGGY_SNIPPET


def test_an_empty_snippet_is_not_sent_to_the_server(make_linter: MakeLinter, server: StubServer) -> None:
    linter = make_linter(responder=fim_responder())

    assert linter.evaluate_snippet("") == []
    assert server.requests == []


# ------------------------------------------------------------------------------
# Batching
# ------------------------------------------------------------------------------
def test_the_tokenization_is_one_request_and_scoring_is_one_batch(make_linter: MakeLinter, server: StubServer) -> None:
    linter = make_linter(responder=fim_responder())

    results = linter.evaluate_snippet(BUGGY_SNIPPET)

    tokenization, scoring = server.requests
    assert tokenization["prompt"] == BUGGY_SNIPPET
    assert tokenization["echo"] is True
    assert isinstance(scoring["prompt"], list)
    # Whitespace slices are recorded without a round-trip, so the batch holds
    # one prompt per scorable slice rather than one per slice.
    assert len(scoring["prompt"]) == sum(1 for res in results if res["token"].strip())
    assert len(scoring["prompt"]) < len(results)


def test_each_prompt_carries_the_whole_snippet_on_both_sides(make_linter: MakeLinter, server: StubServer) -> None:
    """This is the point of FIM: the hole is surrounded by the rest of the file."""
    linter = make_linter(responder=fim_responder())

    linter.evaluate_snippet("ab cd")

    (prompts,) = server.prompts_sent[-1:]
    # Probing the first token puts everything in the suffix; probing the last
    # puts everything in the prefix. Either way the whole snippet is visible.
    assert prompts[0] == "<|fim_prefix|><|fim_suffix|>ab cd<|fim_middle|>"
    assert prompts[-1] == "<|fim_prefix|>ab <|fim_suffix|>cd<|fim_middle|>"


def test_scores_stay_attached_to_their_slice_when_the_server_reorders(
    make_linter: MakeLinter,
) -> None:
    linter = make_linter(responder=fim_responder(surprising=BUGGY_OPERATOR, reverse_choices=True))

    findings = graded(linter, linter.evaluate_snippet(BUGGY_SNIPPET))

    assert findings == [("CRITICAL", BUGGY_OPERATOR)]


def test_a_batch_size_smaller_than_the_file_splits_the_work(
    make_linter: MakeLinter,
    server: StubServer,
) -> None:
    linter = make_linter(responder=fim_responder(), fim_batch_size=4)

    results = linter.evaluate_snippet(BUGGY_SNIPPET)

    assert len(server.requests) > 2  # one tokenization plus several batches
    assert "".join(res["token"] for res in results) == BUGGY_SNIPPET


def test_the_request_asks_for_one_token_and_the_configured_candidate_count(
    make_linter: MakeLinter,
    server: StubServer,
) -> None:
    linter = make_linter(responder=fim_responder(), fim_top_logprobs=7)

    linter.evaluate_snippet("ab")

    (request,) = server.requests[-1:]
    assert request["max_tokens"] == 1
    assert request["temperature"] == 0
    assert request["logprobs"] == 7
    assert request["model"] == SERVED_MODEL


# ------------------------------------------------------------------------------
# Model discovery
# ------------------------------------------------------------------------------
def test_the_model_is_discovered_from_the_endpoint_by_default(
    make_linter: MakeLinter,
    server: StubServer,
    settings: Settings,
) -> None:
    assert settings.model_name is None

    assert make_linter().model_name == SERVED_MODEL
    assert server.model_lookups == 1


def test_an_explicit_model_name_wins_and_skips_discovery(make_linter: MakeLinter, server: StubServer) -> None:
    linter = make_linter(responder=fim_responder(), model_name="pinned/model")

    assert linter.model_name == "pinned/model"
    assert server.requests == []


def test_the_discovered_model_is_the_one_requested(make_linter: MakeLinter, server: StubServer) -> None:
    linter = make_linter(responder=fim_responder(), models=["served/other"])

    linter.evaluate_snippet("ab")

    assert server.requests[-1]["model"] == "served/other"


def test_discovery_happens_once_per_linter(make_linter: MakeLinter, server: StubServer) -> None:
    linter = make_linter(responder=fim_responder())

    for _ in range(3):
        linter.evaluate_snippet("ab")

    assert len(server.requests) == 6  # three calls, each a tokenization plus a scoring request
    assert server.model_lookups == 1  # ...but the model was listed only once


def test_an_ambiguous_endpoint_is_refused_rather_than_guessed(make_linter: MakeLinter) -> None:
    linter = make_linter(models=["b-model", "a-model"])

    with pytest.raises(SoftmaxLintError, match="advertises several models"):
        _ = linter.model_name


def test_an_endpoint_with_no_models_is_an_error(make_linter: MakeLinter) -> None:
    linter = make_linter(models=[])

    with pytest.raises(SoftmaxLintError, match="advertises no models"):
        _ = linter.model_name


def test_an_endpoint_without_discovery_asks_for_a_pinned_name(make_linter: MakeLinter) -> None:
    linter = make_linter(models_status=404)

    with pytest.raises(SoftmaxLintError, match="does not support model discovery"):
        _ = linter.model_name


def test_an_unreachable_endpoint_is_reported_as_such(make_linter: MakeLinter, server: StubServer) -> None:
    linter = make_linter(models_status=503)
    server.error = httpx2.ConnectError("connection refused")

    with pytest.raises(SoftmaxLintError, match="could not reach vLLM"):
        _ = linter.model_name


def test_scoring_still_works_when_discovery_fails(make_linter: MakeLinter) -> None:
    """A pinned name keeps the run alive when the endpoint cannot be listed."""
    linter = make_linter(responder=fim_responder(), models_status=404, model_name="pinned/model")

    assert linter.evaluate_snippet("ab") != []


def test_only_the_tokenization_request_echoes(make_linter: MakeLinter, server: StubServer) -> None:
    """FIM needs the distribution at the hole, which is a generation logprob."""
    linter = make_linter(responder=fim_responder())

    linter.evaluate_snippet("ab")

    tokenization, scoring = server.requests
    assert tokenization["echo"] is True
    assert not scoring["echo"]
    assert "prompt_logprobs" not in scoring


def test_configured_markers_reach_the_prompt(make_linter: MakeLinter) -> None:
    linter = make_linter(
        responder=fim_responder(markers=DEFAULT_MARKERS),
        fim_prefix_marker="PRE",
        fim_suffix_marker="SUF",
        fim_middle_marker="MID",
    )

    with pytest.raises(AssertionError, match="not a FIM prompt"):
        linter.evaluate_snippet("ab")


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
        (0.0, "CRITICAL"),  # a censored score always grades critical
    ],
)
def test_classify_grades_ratios_against_strict_thresholds(
    settings: Settings,
    ratio: float,
    expected: str | None,
) -> None:
    linter = SoftmaxLinter(config=settings)

    assert linter.classify(token_analysis(ratio=ratio)) == expected


def test_classify_follows_configured_thresholds() -> None:
    linter = SoftmaxLinter(config=Settings(_env_file=None, threshold_warn=0.90, threshold_crit=0.80))

    assert linter.classify(token_analysis(ratio=0.85)) == "WARNING"


def test_classify_ignores_whitespace_only_slices(settings: Settings) -> None:
    """Whitespace-only slices are never findings."""
    linter = SoftmaxLinter(config=settings)
    blank = token_analysis(ratio=0.0)
    blank["token"] = "   "

    assert linter.classify(blank) is None


def test_classify_needs_a_confident_model(settings: Settings) -> None:
    """A low ratio under a flat distribution is one guess among many."""
    linter = SoftmaxLinter(config=settings)
    unsure = token_analysis(ratio=0.01)
    unsure["top_prob"] = 0.2

    assert linter.classify(unsure) is None


def test_classify_ignores_a_missing_docstring_suggestion(settings: Settings) -> None:
    """A model asking for a docstring says nothing about the written token."""
    linter = SoftmaxLinter(config=settings)
    docstring_wanted = token_analysis(ratio=0.01)
    docstring_wanted["top_token"] = ' """'

    assert linter.classify(docstring_wanted) is None


# ------------------------------------------------------------------------------
# Error branches
# ------------------------------------------------------------------------------
def test_unreachable_server_raises_a_softmaxlint_error(make_linter: MakeLinter, settings: Settings) -> None:
    linter = make_linter(error=httpx2.ConnectError("connection refused"))

    with pytest.raises(SoftmaxLintError, match=f"could not reach vLLM at {settings.server_base_url}"):
        linter.evaluate_snippet("foo")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (fim_body([]), "no choices"),
        ({}, "no choices"),
        (fim_body([fim_choice(0, {"a": CONFIDENT_LOGPROB}, with_logprobs=False)]), "no logprobs"),
        (fim_body([fim_choice(0, {})]), "empty candidate set"),
    ],
)
def test_malformed_server_responses_raise_a_softmaxlint_error(
    make_linter: MakeLinter,
    body: dict[str, object],
    message: str,
) -> None:
    linter = make_linter(responder=echo_then(body))

    with pytest.raises(SoftmaxLintError, match=message):
        linter.evaluate_snippet("ab")


def test_the_wrong_number_of_choices_is_reported(make_linter: MakeLinter) -> None:
    linter = make_linter(responder=echo_then(fim_body([fim_choice(0, {"a": CONFIDENT_LOGPROB})])))

    # "ab cd" is three slices, but the whitespace one is never sent.
    with pytest.raises(SoftmaxLintError, match="expected 2 scored prompts, server returned 1"):
        linter.evaluate_snippet("ab cd")


# ------------------------------------------------------------------------------
# Markers
# ------------------------------------------------------------------------------
def test_the_linter_writes_the_configured_markers() -> None:
    config = Settings(_env_file=None, fim_prefix_marker="A", fim_suffix_marker="B", fim_middle_marker="C")

    assert SoftmaxLinter(config=config).markers.prefix == "A"


def test_default_markers_are_the_qwen_spelling() -> None:
    assert SoftmaxLinter(config=Settings(_env_file=None)).markers == (
        "<|fim_prefix|>",
        "<|fim_suffix|>",
        "<|fim_middle|>",
    )


def test_prompts_reassemble_the_snippet_around_the_hole(make_linter: MakeLinter, server: StubServer) -> None:
    linter = make_linter(responder=fim_responder())

    linter.evaluate_snippet("ab cd")

    (prompts,) = server.prompts_sent[-1:]
    # The whitespace slice between the words is recorded without a probe.
    assert [after_of(prompt) for prompt in prompts] == ["ab cd", "cd"]


def test_string_spans_are_recorded_without_a_probe(make_linter: MakeLinter, server: StubServer) -> None:
    """String literals are not scored by default."""
    source = 'x = "ab"\n'
    linter = make_linter(responder=fim_responder())

    results = linter.evaluate_snippet(source)

    assert "".join(res["token"] for res in results) == source
    (prompts,) = server.prompts_sent[-1:]
    assert len(prompts) == 2  # 'x' and '='; whitespace and the string are policy-skipped

    full = make_linter(responder=fim_responder(), score_strings_and_comments=True)
    full.evaluate_snippet(source)

    (all_prompts,) = server.prompts_sent[-1:]
    assert len(all_prompts) == 5  # everything but the whitespace
