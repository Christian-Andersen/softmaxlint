"""Command line entry point for softmaxlint."""

import json
from pathlib import Path
from typing import Annotated, Literal, NamedTuple

import typer
from rich.console import Console

from . import lsp as lsp_support
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


def _apply_thresholds(
    console: Console,
    settings: Settings,
    threshold_warn: float | None,
    threshold_crit: float | None,
) -> None:
    """Layer CLI threshold options over settings, or exit 2 when they clash."""
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

    _apply_thresholds(console, settings, threshold_warn, threshold_crit)

    targets = list(paths) if paths else [Path.cwd()]

    files = _collect_files(console, targets, respect_gitignore=respect_gitignore)

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


lsp_app = typer.Typer(
    name="lsp",
    help="Language server protocol support for editor integration.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(lsp_app, name="lsp")


@lsp_app.command("check")
def lsp_check(
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
    output_format: Annotated[
        Literal["concise", "json"] | None,
        typer.Option("--output-format", help="Output format: concise (human) or json (machine)."),
    ] = None,
) -> None:
    """Check files and print the findings for editor consumption.

    Same scoring and grading as ``check``; only the output differs.
    ``--output-format json`` (the default) prints LSP diagnostics;
    ``concise`` prints one ruff-style line per finding. Exits 0 when clean,
    1 when anomalies were found, and 2 on a hard error.
    """
    console = Console(stderr=True)
    settings = Settings()

    _apply_thresholds(console, settings, threshold_warn, threshold_crit)
    if output_format is not None:
        settings.output_format = output_format

    targets = list(paths) if paths else [Path.cwd()]

    files = _collect_files(console, targets, respect_gitignore=respect_gitignore)

    linter = SoftmaxLinter(config=settings)
    try:
        findings, had_error = _lsp_check_files(console, linter, files)
    except SoftmaxLintError as exc:
        console.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(code=EXIT_ERROR) from exc

    if settings.output_format == "concise":
        for file, item in findings:
            print(lsp_support.format_concise(display_path(file), item))
    else:
        print(json.dumps([item for _, item in findings], indent=2))

    if had_error:
        raise typer.Exit(code=EXIT_ERROR)
    if findings:
        raise typer.Exit(code=EXIT_FINDINGS)
    raise typer.Exit(code=EXIT_OK)


def _collect_files(console: Console, targets: list[Path], *, respect_gitignore: bool) -> list[Path]:
    """Resolve targets to files, or exit 2 when a path is unusable."""
    try:
        return collect_python_files(targets, respect_gitignore=respect_gitignore)
    except SoftmaxLintError as exc:
        console.print(f"[bold red]error:[/bold red] {exc}")
        raise typer.Exit(code=EXIT_ERROR) from exc


def _lsp_check_files(
    console: Console, linter: SoftmaxLinter, files: list[Path]
) -> tuple[list[tuple[Path, dict[str, object]]], bool]:
    """Score each file; returns ((file, diagnostic) pairs, whether a file was unreadable)."""
    findings: list[tuple[Path, dict[str, object]]] = []
    had_error = False

    for file in files:
        try:
            source = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            console.print(f"[bold red]error:[/bold red] {file}: {exc}")
            had_error = True
            continue
        if not source.strip():
            continue
        results = linter.evaluate_snippet(source)
        uri = lsp_support.file_uri(file)
        findings.extend((file, {"uri": uri, **item}) for item in lsp_support.diagnostics_for(source, results, linter))

    return findings, had_error


@lsp_app.command("server")
def lsp_server() -> None:
    """Speak LSP over stdio until told to exit. Lints on open and save."""
    linter = SoftmaxLinter(config=Settings())
    raise typer.Exit(code=lsp_support.serve_forever(linter))


def main() -> None:
    """Console script wrapper. Typer owns argument parsing and exit codes."""
    app()
