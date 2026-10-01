"""Command line entry point for softmaxlint."""

from pathlib import Path
from typing import Annotated, NamedTuple

import typer
from rich.console import Console

from .discovery import collect_python_files
from .errors import SoftmaxLintError
from .linter import Anomaly, SoftmaxLinter
from .report import render_file, render_summary
from .settings import Settings

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2


def display_path(path: Path) -> Path:
    """Shorten ``path`` for display, relative to the working directory.

    Reporting keeps output readable and clickable; paths outside the working
    tree are shown as given.
    """
    try:
        return path.relative_to(Path.cwd())
    except ValueError:
        return path


app = typer.Typer(
    name="softmaxlint",
    help="Lint Python source for tokens the model found improbable.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _root() -> None:
    """Lint Python source for tokens the model found improbable.

    A callback keeps `check` a real subcommand; without one, a single-command
    Typer app collapses into the root command and swallows the word "check".
    """


class FileOutcome(NamedTuple):
    """The result of checking a single file."""

    anomalies: list[Anomaly]
    counted: bool
    """False when the file was skipped as blank or could not be read."""
    failed: bool
    """True when the file could not be read at all."""


def _check_file(console: Console, linter: SoftmaxLinter, file: Path) -> FileOutcome:
    """Analyse and report one file.

    Raises:
        SoftmaxLintError: if the server fails. A dead endpoint will not recover
            on the next file, so the caller is expected to abandon the run.
    """
    try:
        source = file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        console.print(f"[bold red]error:[/bold red] {file}: {exc}")
        return FileOutcome(anomalies=[], counted=False, failed=True)

    if not source.strip():
        return FileOutcome(anomalies=[], counted=False, failed=False)

    anomalies = render_file(console, display_path(file), linter.evaluate_snippet(source), linter)
    return FileOutcome(anomalies=anomalies, counted=True, failed=False)


@app.command()
def check(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Files or directories to check. Directories are walked recursively."),
    ] = None,
    *,
    threshold_warn: Annotated[
        float | None,
        typer.Option("--threshold-warn", help="Ratio below which a token is a WARNING."),
    ] = None,
    threshold_crit: Annotated[
        float | None,
        typer.Option("--threshold-crit", help="Ratio below which a token is CRITICAL."),
    ] = None,
    respect_gitignore: Annotated[
        bool,
        typer.Option("--respect-gitignore/--no-respect-gitignore", help="Honour .gitignore files."),
    ] = True,
) -> None:
    """Check Python files for probabilistic anomalies.

    Exits 0 when clean, 1 when anomalies were found, and 2 on a hard error.
    """
    console = Console()
    settings = Settings()

    if threshold_warn is not None:
        settings.threshold_warn = threshold_warn
    if threshold_crit is not None:
        settings.threshold_crit = threshold_crit

    if not 0 <= settings.threshold_crit < settings.threshold_warn <= 1:
        console.print(
            "[bold red]error:[/bold red] thresholds must satisfy 0 <= crit < warn <= 1 "
            f"(got crit={settings.threshold_crit}, warn={settings.threshold_warn})"
        )
        raise typer.Exit(code=EXIT_ERROR)

    targets = list(paths) if paths else [Path.cwd()]

    try:
        files = collect_python_files(targets, respect_gitignore=respect_gitignore)
    except SoftmaxLintError as exc:
        console.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(code=EXIT_ERROR) from exc

    if not files:
        console.print("[dim]no Python files to check[/dim]")
        raise typer.Exit(code=EXIT_OK)

    linter = SoftmaxLinter(config=settings)
    anomalies: list[Anomaly] = []
    had_error = False
    checked = 0

    for file in files:
        try:
            outcome = _check_file(console, linter, file)
        except SoftmaxLintError as exc:
            console.print(f"[bold red]error:[/bold red] {file}: {exc}")
            raise typer.Exit(code=EXIT_ERROR) from exc
        had_error = had_error or outcome.failed
        checked += outcome.counted
        anomalies.extend(outcome.anomalies)

    render_summary(console, checked, anomalies)

    if had_error:
        raise typer.Exit(code=EXIT_ERROR)
    if anomalies:
        raise typer.Exit(code=EXIT_FINDINGS)
    raise typer.Exit(code=EXIT_OK)


def main() -> None:
    """Console script wrapper. Typer owns argument parsing and exit codes."""
    app()
