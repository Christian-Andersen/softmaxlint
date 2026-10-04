"""Calibration: what the linter does to code that is known-good.

``corpus/correct`` holds ordinary Python with nothing wrong in it, so any
finding there is a false positive. These tests use a stub that behaves like a
competent model -- it echoes its tokenization and always ranks the token that
was actually written first -- and assert the linter stays silent. That is the
property a real run depends on, and it is what a misaligned probe destroys.

``corpus/bugs`` holds seeded defects whose expected token is named in a
``# BUG:`` comment, so recall is measured rather than eyeballed.
"""

import re
from pathlib import Path

import pytest

from softmaxlint.linter import SoftmaxLinter
from tests.conftest import BUG_DIR, CORRECT_DIR, FimStub, expected_bug_token

CORRECT_FILES = sorted(CORRECT_DIR.glob("*.py"))
BUG_FILES = sorted(BUG_DIR.glob("*.py"))


def flagged(linter: SoftmaxLinter, path: Path) -> list[tuple[str, str]]:
    """Every graded token in ``path``, as (level, token)."""
    results = linter.evaluate_snippet(path.read_text(encoding="utf-8"))
    return [(level, analysis["token"]) for analysis in results if (level := linter.classify(analysis))]


# ------------------------------------------------------------------------------
# False positives on known-good code
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("path", CORRECT_FILES, ids=lambda p: p.name)
def test_known_good_code_produces_no_findings(make_linter, path: Path) -> None:  # noqa: ANN001
    linter = make_linter(responder=FimStub())

    assert flagged(linter, path) == []


def test_the_corpus_is_actually_present() -> None:
    """Guards against a corpus that quietly went missing."""
    assert len(CORRECT_FILES) >= 3
    assert len(BUG_FILES) >= 3


def test_censoring_does_not_leak_into_good_code(make_linter) -> None:  # noqa: ANN001
    """A confident model should never produce a censored score on ordinary code."""
    linter = make_linter(responder=FimStub())

    for path in CORRECT_FILES:
        results = linter.evaluate_snippet(path.read_text(encoding="utf-8"))
        censored = [analysis["token"] for analysis in results if analysis["actual_prob"] == 0.0]
        assert censored == [], f"{path.name} censored {censored}"


def test_misaligned_probes_flood_the_report(make_linter) -> None:  # noqa: ANN001
    """Documents the failure mode the tokenization round-trip exists to prevent.

    A server that echoes boundaries one character off cannot name the written
    token, so nearly everything is censored and reported CRITICAL.
    """
    shifted = FimStub().with_offset(1)
    linter = make_linter(responder=shifted)

    findings = flagged(linter, CORRECT_FILES[0])

    assert len(findings) > len(flagged(make_linter(responder=FimStub()), CORRECT_FILES[0]))


# ------------------------------------------------------------------------------
# Recall on seeded defects
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("path", BUG_FILES, ids=lambda p: p.name)
def test_a_seeded_defect_is_flagged_on_its_marked_token(make_linter, path: Path) -> None:  # noqa: ANN001
    expected = expected_bug_token(path)
    assert expected, f"{path.name} has no # BUG: marker"

    linter = make_linter(responder=FimStub(surprising=expected, alternative="zzz"))

    assert any(level == "CRITICAL" and token.strip() == expected for level, token in flagged(linter, path))


@pytest.mark.parametrize("path", BUG_FILES, ids=lambda p: p.name)
def test_the_rest_of_a_buggy_file_stays_quiet(make_linter, path: Path) -> None:  # noqa: ANN001
    """One seeded defect should not drag the whole file into the report."""
    linter = make_linter(responder=FimStub())

    assert flagged(linter, path) == []


@pytest.mark.parametrize("path", CORRECT_FILES + BUG_FILES, ids=lambda p: p.name)
def test_every_corpus_file_is_valid_python(path: Path) -> None:
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_bug_markers_name_tokens_that_occur_in_the_file() -> None:
    """A stale marker would silently measure recall against the wrong token."""
    for path in BUG_FILES:
        token = expected_bug_token(path)
        assert token in path.read_text(encoding="utf-8"), f"{path.name}: {token!r} not in file"


def test_no_bug_marker_sits_in_the_good_corpus() -> None:
    for path in CORRECT_FILES:
        assert not re.search(r"#\s*BUG:", path.read_text(encoding="utf-8"))
