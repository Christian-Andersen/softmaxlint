"""Slicing, prompt construction, and candidate parsing for FIM scoring."""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from softmaxlint import fim
from softmaxlint.errors import SoftmaxLintError
from tests.conftest import (
    BUGGY_OPERATOR,
    BUGGY_SNIPPET,
    CONFIDENT_LOGPROB,
    CORRECT_OPERATOR,
    DEFAULT_MARKERS,
    SURPRISED_LOGPROB,
    after_of,
    echo_choice,
    fim_choice,
    slice_of,
    stub_offsets,
)

SOURCE = "alpha beta\ngamma delta\n"

# Recorded from the real Qwen2.5-Coder tokenizer. Guards against a regression to
# a locally-invented tokenization, which is what made every line report.
QWEN_BPE = json.loads((Path(__file__).parent / "qwen_bpe.json").read_text())


def choice_for(candidates: dict[str, float]) -> dict[str, Any]:
    return fim_choice(0, candidates)


# ------------------------------------------------------------------------------
# Slicing from the model's own boundaries
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("case", sorted(QWEN_BPE))
def test_slices_reproduce_the_real_qwen_tokenization(case: str) -> None:
    """Probes must land where Qwen's BPE actually splits, not on a regex."""
    source = QWEN_BPE[case]["source"]
    offsets = QWEN_BPE[case]["offsets"]

    slices = fim.slices_from_offsets(source, offsets)

    assert [slice_.text for slice_ in slices] == QWEN_BPE[case]["tokens"]
    assert "".join(slice_.text for slice_ in slices) == source


@pytest.mark.parametrize("case", sorted(QWEN_BPE))
def test_every_slice_reports_the_exact_span_it_covers(case: str) -> None:
    source = QWEN_BPE[case]["source"]

    for slice_ in fim.slices_from_offsets(source, QWEN_BPE[case]["offsets"]):
        assert source[slice_.start : slice_.end] == slice_.text


def test_real_qwen_merges_pieces_a_regex_would_split() -> None:
    """The reason we ask the server rather than tokenizing locally."""
    tokens = QWEN_BPE["SMALL"]["tokens"]

    assert "(n" in tokens  # a regex would split this into '(' and 'n'
    assert "):\n" in tokens  # and this into '):' and '\n'
    assert " total" in tokens  # the leading space belongs to the token


def test_a_coarse_split_disagrees_with_the_real_tokenizer() -> None:
    """Documents the failure mode the server round-trip exists to avoid."""
    coarse = [slice_.text for slice_ in fim.slice_source(QWEN_BPE["SMALL"]["source"], "coarse")]

    assert coarse != QWEN_BPE["SMALL"]["tokens"]
    assert set(coarse) & set(QWEN_BPE["SMALL"]["tokens"]) != set(QWEN_BPE["SMALL"]["tokens"])


def test_offsets_that_do_not_start_at_zero_are_rejected() -> None:
    with pytest.raises(SoftmaxLintError, match="unusable tokenization"):
        fim.slices_from_offsets(SOURCE, [3, 7])


def test_offsets_that_run_backwards_are_rejected() -> None:
    with pytest.raises(SoftmaxLintError, match="unusable tokenization"):
        fim.slices_from_offsets(SOURCE, [0, 9, 4])


def test_offsets_past_the_end_are_rejected() -> None:
    with pytest.raises(SoftmaxLintError, match="unusable tokenization"):
        fim.slices_from_offsets("ab", [0, 99])


def test_no_offsets_means_no_slices() -> None:
    assert fim.slices_from_offsets(SOURCE, []) == []


def test_zero_width_tokens_are_dropped() -> None:
    assert [s.text for s in fim.slices_from_offsets("abc", [0, 0, 1])] == ["a", "bc"]


# ------------------------------------------------------------------------------
# Reading a tokenization off an echoed response
# ------------------------------------------------------------------------------
def test_offsets_are_read_from_an_echoed_choice() -> None:
    choice = echo_choice(SOURCE)

    assert fim.parse_offsets(choice, SOURCE) == stub_offsets(SOURCE)


def test_a_missing_logprobs_block_is_an_error() -> None:
    with pytest.raises(SoftmaxLintError, match="echoed no logprobs"):
        fim.parse_offsets({"index": 0}, SOURCE)


def test_missing_text_offsets_are_an_error() -> None:
    choice = echo_choice(SOURCE)
    del choice["logprobs"]["text_offset"]

    with pytest.raises(SoftmaxLintError, match="echoed no text_offset"):
        fim.parse_offsets(choice, SOURCE)


def test_offsets_outside_the_snippet_are_an_error() -> None:
    choice = echo_choice(SOURCE)
    choice["logprobs"]["text_offset"] = [500, 600]

    with pytest.raises(SoftmaxLintError, match="outside the snippet"):
        fim.parse_offsets(choice, SOURCE)


# ------------------------------------------------------------------------------
# Local fallback splitters
# ------------------------------------------------------------------------------
def test_line_slices_reassemble_into_the_source() -> None:
    assert "".join(s.text for s in fim.slice_source(SOURCE, "line")) == SOURCE


def test_line_slicing_keeps_line_endings_with_their_line() -> None:
    assert [s.text for s in fim.slice_source("one\ntwo\n", "line")] == ["one\n", "two\n"]


def test_line_slicing_does_not_invent_a_trailing_newline() -> None:
    assert [s.text for s in fim.slice_source("one\ntwo", "line")] == ["one\n", "two"]


def test_coarse_slices_reassemble_into_the_source() -> None:
    assert "".join(s.text for s in fim.slice_source(BUGGY_SNIPPET, "coarse")) == BUGGY_SNIPPET


def test_an_unknown_slice_mode_is_rejected() -> None:
    with pytest.raises(SoftmaxLintError, match="unknown fim_chunk_mode"):
        fim.slice_source(SOURCE, "sentence")


def test_an_empty_source_yields_no_slices() -> None:
    assert fim.slice_source("", "line") == []
    assert fim.slice_source("", "coarse") == []


# ------------------------------------------------------------------------------
# Prompt construction
# ------------------------------------------------------------------------------
def test_the_prompt_places_the_hole_between_prefix_and_suffix() -> None:
    prompt = fim.build_prompt(SOURCE, 11, DEFAULT_MARKERS)

    assert prompt == "<|fim_prefix|>alpha beta\n<|fim_suffix|>gamma delta\n<|fim_middle|>"


def test_the_hole_starts_exactly_where_the_slice_starts() -> None:
    slice_ = slice_of("gamma", 11)

    assert after_of(fim.build_prompt(SOURCE, slice_.start, DEFAULT_MARKERS)) == SOURCE[slice_.start :]


def test_markers_are_configurable() -> None:
    markers = fim.FimMarkers(prefix="PRE", suffix="SUF", middle="MID")

    assert fim.build_prompt("ab", 1, markers) == "PREaSUFbMID"


def test_a_probe_at_offset_zero_puts_all_code_in_the_suffix() -> None:
    assert after_of(fim.build_prompt(SOURCE, 0, DEFAULT_MARKERS)) == SOURCE


def test_a_probe_at_the_end_puts_all_code_in_the_prefix() -> None:
    prompt = fim.build_prompt(SOURCE, len(SOURCE), DEFAULT_MARKERS)

    assert prompt == f"<|fim_prefix|>{SOURCE}<|fim_suffix|><|fim_middle|>"
    assert after_of(prompt) == ""


# ------------------------------------------------------------------------------
# Candidate parsing
# ------------------------------------------------------------------------------
def test_a_matched_token_is_scored_against_the_best_candidate() -> None:
    slice_ = slice_of("alpha")
    choice = choice_for({"alpha": CONFIDENT_LOGPROB, "omega": math.log(0.5)})

    analysis = fim.parse_choice(choice, slice_)

    assert analysis["token"] == "alpha"
    assert analysis["top_token"] == "alpha"
    assert analysis["actual_prob"] == pytest.approx(0.95)
    assert analysis["top_prob"] == pytest.approx(0.95)
    assert analysis["ratio"] == pytest.approx(1.0)
    assert analysis["rank"] == 1


def test_an_unlikely_token_produces_a_low_ratio() -> None:
    slice_ = slice_of(BUGGY_OPERATOR)
    choice = choice_for({CORRECT_OPERATOR: CONFIDENT_LOGPROB, BUGGY_OPERATOR: SURPRISED_LOGPROB})

    analysis = fim.parse_choice(choice, slice_)

    assert analysis["actual_prob"] == pytest.approx(0.02)
    assert analysis["top_prob"] == pytest.approx(0.95)
    assert analysis["ratio"] == pytest.approx(0.02 / 0.95)
    assert analysis["surprisal"] == pytest.approx(-SURPRISED_LOGPROB)
    assert analysis["rank"] == 2


def test_an_exact_match_beats_a_longer_prefix() -> None:
    """The written token is preferred over a longer token that merely starts it."""
    slice_ = slice_of("foo")
    choice = choice_for({"foo(": SURPRISED_LOGPROB, "foo": CONFIDENT_LOGPROB})

    analysis = fim.parse_choice(choice, slice_)

    assert analysis["rank"] == 1
    assert analysis["ratio"] == pytest.approx(1.0)


def test_a_resegmented_source_falls_back_to_the_longest_fit() -> None:
    """The model splitting the same text differently is not an anomaly."""
    slice_ = slice_of(" total = ")
    choice = choice_for({" total": CONFIDENT_LOGPROB, "=": SURPRISED_LOGPROB})

    analysis = fim.parse_choice(choice, slice_)

    assert analysis["ratio"] == pytest.approx(1.0)


def test_an_absent_token_is_reported_as_censored() -> None:
    slice_ = slice_of(BUGGY_OPERATOR)
    choice = choice_for({"while": CONFIDENT_LOGPROB, "for": CONFIDENT_LOGPROB - 0.1})

    analysis = fim.parse_choice(choice, slice_)

    assert analysis["actual_prob"] == 0.0
    assert analysis["ratio"] == 0.0
    assert analysis["token"] == BUGGY_OPERATOR
    assert analysis["rank"] == 3  # ranked below everything the server named


def test_a_censored_surprisal_reports_the_floor_we_can_vouch_for() -> None:
    slice_ = slice_of(BUGGY_OPERATOR)
    weakest = CONFIDENT_LOGPROB - 0.5
    choice = choice_for({"while": CONFIDENT_LOGPROB, "for": weakest})

    assert fim.parse_choice(choice, slice_)["surprisal"] == pytest.approx(-weakest)


def test_an_empty_candidate_string_is_never_matched() -> None:
    choice = choice_for({"": CONFIDENT_LOGPROB, "a": CONFIDENT_LOGPROB})

    assert fim.parse_choice(choice, slice_of("alpha"))["ratio"] == pytest.approx(1.0)


def test_a_zero_probability_top_candidate_does_not_divide_by_zero() -> None:
    choice = choice_for({"alpha": -1e308, "omega": -1e308})

    assert fim.parse_choice(choice, slice_of("alpha"))["ratio"] == 1.0


def test_a_choice_without_logprobs_is_rejected() -> None:
    choice = fim_choice(0, {"alpha": CONFIDENT_LOGPROB}, with_logprobs=False)

    with pytest.raises(SoftmaxLintError, match="no logprobs"):
        fim.parse_choice(choice, slice_of("alpha"))


def test_an_empty_candidate_set_is_rejected() -> None:
    with pytest.raises(SoftmaxLintError, match="empty candidate set"):
        fim.parse_choice(choice_for({}), slice_of("alpha"))


# ------------------------------------------------------------------------------
# Batch alignment
# ------------------------------------------------------------------------------
def test_choices_are_reordered_back_into_prompt_order() -> None:
    choices = [{"index": 2, "text": "c"}, {"index": 0, "text": "a"}, {"index": 1, "text": "b"}]

    assert [c["text"] for c in fim.order_choices(choices, 3)] == ["a", "b", "c"]


def test_choices_already_in_order_are_left_alone() -> None:
    choices = [{"index": 0, "text": "a"}, {"index": 1, "text": "b"}]

    assert fim.order_choices(choices, 2) == choices


def test_a_missing_index_leaves_arrival_order_intact() -> None:
    assert [c["text"] for c in fim.order_choices([{"text": "a"}, {"text": "b"}], 2)] == ["a", "b"]


def test_the_wrong_number_of_choices_is_an_error() -> None:
    with pytest.raises(SoftmaxLintError, match="expected 3 scored prompts, server returned 2"):
        fim.order_choices([{"index": 0}, {"index": 1}], 3)


def test_no_choices_at_all_is_an_error() -> None:
    with pytest.raises(SoftmaxLintError, match="expected 1 scored prompts, server returned 0"):
        fim.order_choices([], 1)
