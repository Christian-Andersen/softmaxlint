"""The `check` command: argument handling, file selection, and exit codes."""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest
from typer.testing import CliRunner

from softmaxlint import cli
from softmaxlint.linter import SoftmaxLinter
from softmaxlint.settings import Settings
from tests.conftest import (
    BUGGY_OPERATOR,
    BUGGY_SNIPPET,
    StubServer,
    fim_responder,
    fixed_candidates,
)

runner = CliRunner()

CLEAN = fim_responder()
DIRTY = fim_responder(surprising=BUGGY_OPERATOR)

Serve = Callable[..., None]


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    """Stop the CLI from reading a developer's .env out of the working directory."""
    monkeypatch.setattr(cli, "Settings", lambda: settings)


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch) -> Serve:
    """Point the CLI at a stub vLLM endpoint."""

    def _serve(
        responder: Callable[[str | list[str]], dict[str, Any]] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        stub = StubServer(responder=responder, error=error)

        class _StubbedLinter(SoftmaxLinter):
            def __init__(self, config: Settings | None = None) -> None:
                super().__init__(config)
                self.client = stub.client()

        monkeypatch.setattr(cli, "SoftmaxLinter", _StubbedLinter)

    return _serve


# ------------------------------------------------------------------------------
# Exit codes
# ------------------------------------------------------------------------------
def test_check_exits_zero_when_everything_is_clean(tmp_path: Path, serve: Serve) -> None:
    serve(CLEAN)
    write(tmp_path / "average.py", BUGGY_SNIPPET)

    result = runner.invoke(cli.app, ["check", str(tmp_path)])

    assert result.exit_code == cli.EXIT_OK, result.output
    assert "no anomalies" in result.output


def test_check_exits_one_when_anomalies_are_found(tmp_path: Path, serve: Serve) -> None:
    serve(DIRTY)
    write(tmp_path / "average.py", BUGGY_SNIPPET)

    result = runner.invoke(cli.app, ["check", str(tmp_path)])

    assert result.exit_code == cli.EXIT_FINDINGS, result.output
    assert "CRITICAL" in result.output


def test_check_exits_two_on_a_missing_path(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["check", str(tmp_path / "absent.py")])

    assert result.exit_code == cli.EXIT_ERROR, result.output
    assert "does not exist" in result.output


def test_check_exits_two_when_the_server_is_unreachable(tmp_path: Path, serve: Serve) -> None:
    serve(error=httpx2.ConnectError("connection refused"))
    write(tmp_path / "average.py", BUGGY_SNIPPET)

    result = runner.invoke(cli.app, ["check", str(tmp_path)])

    assert result.exit_code == cli.EXIT_ERROR, result.output
    assert "could not reach vLLM" in result.output


def test_check_exits_two_for_contradictory_thresholds(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["check", str(tmp_path), "--threshold-crit", "0.9", "--threshold-warn", "0.1"])

    assert result.exit_code == cli.EXIT_ERROR, result.output
    assert "thresholds must satisfy" in result.output


def test_check_exits_two_when_a_file_cannot_be_decoded(tmp_path: Path, serve: Serve) -> None:
    serve(CLEAN)
    (tmp_path / "binary.py").write_bytes(b"\xff\xfe\x00\x01")

    result = runner.invoke(cli.app, ["check", str(tmp_path)])

    assert result.exit_code == cli.EXIT_ERROR, result.output


# ------------------------------------------------------------------------------
# File selection
# ------------------------------------------------------------------------------
def test_check_defaults_to_the_working_directory(
    tmp_path: Path,
    serve: Serve,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serve(CLEAN)
    write(tmp_path / "checked_here.py", BUGGY_SNIPPET)
    write(tmp_path / "skipped_here.py", BUGGY_SNIPPET)
    write(tmp_path / ".gitignore", "skipped_here.py\n")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(cli.app, ["check"])

    assert result.exit_code == cli.EXIT_OK, result.output
    assert "checked_here.py" in result.output
    assert "skipped_here.py" not in result.output


def test_check_skips_gitignored_files(tmp_path: Path, serve: Serve) -> None:
    serve(DIRTY)
    write(tmp_path / "kept.py", BUGGY_SNIPPET)
    write(tmp_path / "ignored.py", BUGGY_SNIPPET)
    write(tmp_path / ".gitignore", "ignored.py\n")

    result = runner.invoke(cli.app, ["check", str(tmp_path)])

    assert "ignored.py" not in result.output
    assert "kept.py" in result.output


def test_check_can_be_told_to_ignore_gitignore(tmp_path: Path, serve: Serve) -> None:
    serve(CLEAN)
    write(tmp_path / "ignored.py", BUGGY_SNIPPET)
    write(tmp_path / ".gitignore", "ignored.py\n")

    result = runner.invoke(cli.app, ["check", str(tmp_path), "--no-respect-gitignore"])

    assert "ignored.py" in result.output


def test_check_accepts_several_paths(tmp_path: Path, serve: Serve) -> None:
    serve(CLEAN)
    write(tmp_path / "one" / "first.py", BUGGY_SNIPPET)
    write(tmp_path / "two" / "second.py", BUGGY_SNIPPET)

    result = runner.invoke(cli.app, ["check", str(tmp_path / "one"), str(tmp_path / "two")])

    assert result.exit_code == cli.EXIT_OK, result.output
    assert "first.py" in result.output
    assert "second.py" in result.output


def test_check_reports_when_there_is_nothing_to_lint(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["check", str(tmp_path)])

    assert result.exit_code == cli.EXIT_OK, result.output
    assert "no Python files to check" in result.output


def test_check_skips_blank_files_without_asking_the_server(tmp_path: Path, serve: Serve) -> None:
    serve(CLEAN)
    write(tmp_path / "blank.py", "\n\n")
    write(tmp_path / "average.py", BUGGY_SNIPPET)

    result = runner.invoke(cli.app, ["check", str(tmp_path)])

    assert result.exit_code == cli.EXIT_OK, result.output
    assert "1 file checked" in result.output


# ------------------------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------------------------
def test_a_loosened_threshold_turns_findings_into_a_clean_run(tmp_path: Path, serve: Serve) -> None:
    serve(DIRTY)
    write(tmp_path / "average.py", BUGGY_SNIPPET)

    result = runner.invoke(cli.app, ["check", str(tmp_path), "--threshold-crit", "0.001", "--threshold-warn", "0.002"])

    assert result.exit_code == cli.EXIT_OK, result.output


def test_a_tightened_threshold_promotes_a_warning_to_critical(tmp_path: Path, serve: Serve) -> None:
    # ratio 0.2: a WARNING at the default thresholds, CRITICAL once crit is raised.
    serve(fixed_candidates(math.log(0.1), math.log(0.5)))
    write(tmp_path / "thing.py", "foo")

    result = runner.invoke(cli.app, ["check", str(tmp_path), "--threshold-warn", "0.9", "--threshold-crit", "0.5"])

    assert result.exit_code == cli.EXIT_FINDINGS, result.output
    assert "CRITICAL" in result.output


def test_threshold_options_win_over_the_settings_file(
    tmp_path: Path,
    serve: Serve,
    settings: Settings,
) -> None:
    serve(DIRTY)
    write(tmp_path / "average.py", BUGGY_SNIPPET)
    settings.threshold_crit = 0.5
    settings.threshold_warn = 0.9

    result = runner.invoke(cli.app, ["check", str(tmp_path), "--threshold-crit", "0.001", "--threshold-warn", "0.002"])

    assert result.exit_code == cli.EXIT_OK, result.output
