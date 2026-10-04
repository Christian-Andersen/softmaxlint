"""Render analysis results as Rich panels and tables."""

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .linter import Anomaly

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rich.console import Console

    from .linter import AnomalyLevel, SoftmaxLinter, TokenAnalysis

_LEVEL_STYLES: dict[AnomalyLevel, str] = {
    "CRITICAL": "[bold white on red] CRITICAL [/bold white on red]",
    "WARNING": "[bold black on yellow] WARNING  [/bold black on yellow]",
}

_TOKEN_STYLES: dict[AnomalyLevel, str] = {
    "CRITICAL": "bold white on red",
    "WARNING": "bold black on yellow",
}


def render_file(console: Console, path: Path, results: Sequence[TokenAnalysis], linter: SoftmaxLinter) -> list[Anomaly]:
    """Print one file's heatmap plus its anomaly table.

    Returns the anomalies found so the caller can accumulate a total and decide
    an exit code without re-deriving them.
    """
    heatmap_text: Text = Text()
    anomalies: list[Anomaly] = []

    for idx, res in enumerate(results):
        level: AnomalyLevel | None = linter.classify(res)
        if level is None:
            heatmap_text.append(res["token"])
            continue
        heatmap_text.append(res["token"], style=_TOKEN_STYLES[level])
        anomalies.append(Anomaly(index=idx, level=level, analysis=res))

    # Printed on its own line: a long path in a panel title gets cropped, and
    # knowing which file a heatmap belongs to matters more than the branding.
    # soft_wrap keeps it that way on a narrow terminal too -- Rich would
    # otherwise fold the line mid-filename, leaving an unreadable path.
    console.print(f"[bold cyan]{path}[/bold cyan]", soft_wrap=True)
    console.print(
        Panel(
            heatmap_text,
            title="[bold cyan]SOFTMAXLINT: PROBABILISTIC CODE HEATMAP[/bold cyan]",
            subtitle="[dim]Highlights: Red = Severe Anomaly, Yellow = Mild Warning[/dim]",
            expand=False,
        )
    )

    if not anomalies:
        console.print(
            "\n[bold green]✅ No token anomalies detected. Code conforms to model expectations.[/bold green]\n"
        )
        return anomalies

    table = Table(
        title=f"⚠️  FOUND {len(anomalies)} POTENTIAL ANOMALIES",
        title_style="bold yellow",
        header_style="bold magenta",
    )
    table.add_column("Level", justify="center")
    table.add_column("Written Token", style="cyan")
    table.add_column("Probability", justify="right")
    table.add_column("Top Expected", style="green")
    table.add_column("Expected Prob", justify="right")
    table.add_column("Ratio", justify="right")

    for anomaly in anomalies:
        res = anomaly.analysis
        table.add_row(
            _LEVEL_STYLES[anomaly.level],
            repr(res["token"]),
            f"{res['actual_prob'] * 100:5.2f}%",
            repr(res["top_token"]),
            f"{res['top_prob'] * 100:5.2f}%",
            f"{res['ratio'] * 100:5.1f}%",
        )

    console.print()
    console.print(table)
    console.print()
    return anomalies


def render_summary(console: Console, files_checked: int, anomalies: Sequence[Anomaly]) -> None:
    """Print the run totals shown after every file has been reported."""
    critical = sum(anomaly.level == "CRITICAL" for anomaly in anomalies)
    warning = len(anomalies) - critical

    noun = "file" if files_checked == 1 else "files"
    parts = [f"{files_checked} {noun} checked"]
    if anomalies:
        parts.append(f"[bold red]{critical} critical[/bold red]")
        parts.append(f"[bold yellow]{warning} warning[/bold yellow]")
    else:
        parts.append("[bold green]no anomalies[/bold green]")

    console.print(" · ".join(parts))
