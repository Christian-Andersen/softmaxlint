"""Rich rendering of analysis results."""

import math
from collections.abc import Callable
from pathlib import Path

from softmaxlint.linter import Anomaly, SoftmaxLinter
from softmaxlint.report import render_file, render_summary
from tests.conftest import BUGGY_SNIPPET, Recorder, confident_scores, logprobs_payload, token_analysis

MakeLinter = Callable[..., SoftmaxLinter]

SOURCE = Path("pkg/example.py")
SIMPLE = "x = 1\n"


def render(linter: SoftmaxLinter, recorder: Recorder, code: str) -> list[Anomaly]:
    """Run one snippet through the stub server and render the result."""
    return render_file(recorder.console, SOURCE, linter.evaluate_snippet(code), linter)


# ------------------------------------------------------------------------------
# Per-file rendering
# ------------------------------------------------------------------------------
def test_a_file_with_a_critical_token_renders_an_anomaly_table(make_linter: MakeLinter, console: Recorder) -> None:
    linter = make_linter(logprobs_payload(confident_scores(BUGGY_SNIPPET, surprised="<=")))

    anomalies = render(linter, console, BUGGY_SNIPPET)

    assert [anomaly.level for anomaly in anomalies] == ["CRITICAL"]
    assert "FOUND 1 POTENTIAL ANOMALIES" in console.text


def test_the_table_shows_the_written_and_expected_tokens(make_linter: MakeLinter, console: Recorder) -> None:
    linter = make_linter(logprobs_payload(confident_scores(BUGGY_SNIPPET, surprised="<=")))

    render(linter, console, BUGGY_SNIPPET)

    assert "'<='" in console.text
    assert "'<'" in console.text
    assert "2.00%" in console.text  # what the model thought of the written token
    assert "95.00%" in console.text  # what the model wanted instead


def test_the_anomaly_row_reports_the_probability_ratio(make_linter: MakeLinter, console: Recorder) -> None:
    linter = make_linter(logprobs_payload([("foo", math.log(0.1), "bar", math.log(0.5))]))

    render(linter, console, "foo")

    assert "20.0%" in console.text


def test_a_mildly_surprising_token_is_not_reported(make_linter: MakeLinter, console: Recorder) -> None:
    linter = make_linter(logprobs_payload([("foo", math.log(0.4), "bar", math.log(0.5))]))  # ratio ~0.8

    anomalies = render(linter, console, "foo")

    assert anomalies == []
    assert "No token anomalies detected" in console.text


def test_a_warning_level_token_is_labelled_a_warning(make_linter: MakeLinter, console: Recorder) -> None:
    linter = make_linter(logprobs_payload([("foo", math.log(0.05), "bar", math.log(0.5))]))  # ratio ~0.1

    anomalies = render(linter, console, "foo")

    assert [anomaly.level for anomaly in anomalies] == ["WARNING"]
    assert "WARNING" in console.text


def test_a_clean_file_reports_no_findings(make_linter: MakeLinter, console: Recorder) -> None:
    linter = make_linter(logprobs_payload(confident_scores(BUGGY_SNIPPET)))

    anomalies = render(linter, console, BUGGY_SNIPPET)

    assert anomalies == []
    assert "No token anomalies detected" in console.text


def test_the_heading_names_the_file_that_was_checked(make_linter: MakeLinter, console: Recorder) -> None:
    linter = make_linter(logprobs_payload(confident_scores(SIMPLE)))

    render(linter, console, SIMPLE)

    assert str(SOURCE) in console.text


def test_a_file_with_no_tokens_renders_without_a_table(make_linter: MakeLinter, console: Recorder) -> None:
    linter = make_linter(logprobs_payload([]))

    anomalies = render(linter, console, "")

    assert anomalies == []
    assert "FOUND" not in console.text


# ------------------------------------------------------------------------------
# Run totals
# ------------------------------------------------------------------------------
def test_the_summary_counts_files_and_splits_levels_by_severity(console: Recorder) -> None:
    anomalies = [
        Anomaly(index=0, level="CRITICAL", analysis=token_analysis()),
        Anomaly(index=1, level="WARNING", analysis=token_analysis()),
        Anomaly(index=2, level="WARNING", analysis=token_analysis()),
    ]

    render_summary(console.console, 4, anomalies)

    assert "4 files checked" in console.text
    assert "1 critical" in console.text
    assert "2 warning" in console.text


def test_the_summary_is_singular_for_a_single_file(console: Recorder) -> None:
    render_summary(console.console, 1, [])

    assert "1 file checked" in console.text


def test_the_summary_reports_a_clean_run(console: Recorder) -> None:
    render_summary(console.console, 2, [])

    assert "no anomalies" in console.text
