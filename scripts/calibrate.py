"""Measure whether the model is actually good enough to lint with.

Runs the linter over a labelled corpus and reports the two numbers that decide
whether the tool is usable:

* **false positives** on ``corpus/correct`` -- code with nothing wrong in it.
  Every finding here is noise, and the goal is zero.
* **recall** on ``corpus/bugs`` -- code with a seeded defect marked by a
  ``# BUG:`` comment naming the token expected to be flagged.

It also reports the **censoring rate**, which is the number to watch. A token
the server will not name has an unknown probability and is reported
CRITICAL, so a high censoring rate means the run is noise regardless of how
good the model is. That normally means probes are not landing on the model's
token boundaries.

Usage::

    just calibrate                    # against SERVER_BASE_URL
    just calibrate --threshold-warn 0.5 --threshold-crit 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from softmaxlint.discovery import collect_python_files
from softmaxlint.errors import SoftmaxLintError
from softmaxlint.linter import SoftmaxLinter
from softmaxlint.settings import Settings

CORPUS = Path(__file__).resolve().parent.parent / "corpus"
CORRECT = CORPUS / "correct"
BUGS = CORPUS / "bugs"

#: Above this, most tokens are unknown to the server and the run is noise.
CENSORING_ALARM = 0.10

#: Seeded bugs per file. Labels live beside the sources, never in them:
#: anything in the file is scoring context, including a comment naming a fix.
EXPECTED: dict[str, Any] = json.loads((BUGS / "expected.json").read_text(encoding="utf-8"))

console = Console()


@dataclass
class FileReport:
    """What one corpus file produced."""

    path: Path
    tokens: int = 0
    findings: Counter[str] = field(default_factory=Counter)
    censored: int = 0
    expected: list[dict[str, Any]] = field(default_factory=list)
    hits: set[int] = field(default_factory=set)

    @property
    def flagged(self) -> int:
        return sum(self.findings.values())


def line_of(source: str, offset: int) -> int:
    """1-based line number of a character ``offset``."""
    return source.count("\n", 0, offset) + 1


def score(path: Path, linter: SoftmaxLinter) -> FileReport:
    """Score one file and grade every token in it."""
    report = FileReport(path=path)
    if BUGS in path.parents:
        report.expected = list(EXPECTED.get(path.name, []))
    source = path.read_text(encoding="utf-8")
    results = linter.evaluate_snippet(source)

    for analysis in results:
        report.tokens += 1
        if analysis["actual_prob"] == 0.0:
            report.censored += 1
        level = linter.classify(analysis)
        if level is not None:
            report.findings[analysis["token"]] += 1
            # BPE folds preceding whitespace into the token (``' =='``),
            # so both sides compare stripped.
            line = line_of(source, analysis["start"])
            for index, bug in enumerate(report.expected):
                if bug["line"] == line and analysis["token"].strip() == bug["token"]:
                    report.hits.add(index)

    return report


def summarise(reports: list[FileReport], label: str) -> None:
    table = Table(title=f"{label}", title_style="bold cyan", header_style="bold magenta")
    for column in ("file", "tokens", "censored", "flagged", "top offender"):
        table.add_column(column)
    table.add_column("recall", justify="center")

    for report in sorted(reports, key=lambda item: (-item.flagged, item.path.name)):
        worst = report.findings.most_common(1)
        if not report.expected:
            recall = ""
        elif len(report.hits) == len(report.expected):
            recall = f"[green]{len(report.hits)}/{len(report.expected)}[/green]"
        else:
            recall = f"[red]{len(report.hits)}/{len(report.expected)}[/red]"
        table.add_row(
            report.path.name,
            str(report.tokens),
            f"[yellow]{report.censored}[/yellow]" if report.censored else "0",
            f"[red]{report.flagged}[/red]" if report.flagged else "0",
            repr(worst[0][0]) if worst else "-",
            recall,
        )
    console.print(table)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold-warn", type=float, default=None)
    parser.add_argument("--threshold-crit", type=float, default=None)
    parser.add_argument("--limit", type=int, default=0, help="only score the first N files per group")
    args = parser.parse_args()

    settings = Settings()
    if args.threshold_warn is not None:
        settings.threshold_warn = args.threshold_warn
    if args.threshold_crit is not None:
        settings.threshold_crit = args.threshold_crit

    console.print(f"[bold]server[/bold] {settings.server_base_url}")
    console.print(f"[bold]slice mode[/bold] {settings.fim_slice_mode}")
    console.print(
        f"[bold]thresholds[/bold] warn<{settings.threshold_warn} crit<{settings.threshold_crit}\n",
    )

    linter = SoftmaxLinter(config=settings)

    try:
        # Resolve up front so an unreachable endpoint or an ambiguous model
        # fails before any of the corpus is scored.
        console.print(f"[bold]model[/bold] {linter.model_name}\n")
        correct_files = sorted(collect_python_files([CORRECT]))
        bug_files = sorted(collect_python_files([BUGS]))
        if args.limit:
            correct_files = correct_files[: args.limit]
            bug_files = bug_files[: args.limit]

        correct = [score(path, linter) for path in correct_files]
        bugs = [score(path, linter) for path in bug_files]
    except SoftmaxLintError as exc:
        console.print(f"[bold red]error:[/bold red] {exc}")
        return 2

    summarise(correct, "corpus/correct -- every finding is a false positive")
    summarise(bugs, "corpus/bugs -- 'expected hit' is recall")

    clean_tokens = sum(report.tokens for report in correct)
    false_positives = sum(report.flagged for report in correct)
    censored = sum(report.censored for report in correct + bugs)
    scored = sum(report.tokens for report in correct + bugs)
    hits = sum(len(report.hits) for report in bugs)
    total_bugs = sum(len(report.expected) for report in bugs)
    dirty_files = sum(1 for report in bugs if report.flagged)

    console.print()
    console.print(
        f"[bold]false positive rate[/bold]  {false_positives}/{clean_tokens} tokens"
        f" ({100 * false_positives / max(clean_tokens, 1):.2f}%)"
    )
    console.print(f"[bold]files with any finding[/bold]  {sum(1 for r in correct if r.flagged)}/{len(correct)}")
    console.print(f"[bold]recall[/bold]  {hits}/{total_bugs} seeded bugs flagged")
    console.print(f"[bold]files with any finding[/bold]  {dirty_files}/{len(bugs)}")
    console.print(f"[bold]censoring rate[/bold]  {censored}/{scored} tokens ({100 * censored / max(scored, 1):.2f}%)")

    verdict = "unusable" if false_positives else "clean on known-good code"
    if censored / max(scored, 1) > CENSORING_ALARM:
        console.print(
            "\n[bold red]Censoring is high.[/bold red] Most tokens are unknown to the "
            "server, so their ratios are 0 and they all read as CRITICAL. Check that "
            "fim_slice_mode is 'server' and that the model supports the FIM markers.",
        )
    console.print(f"\n[bold]verdict:[/bold] {verdict}")
    return 0 if not false_positives else 1


if __name__ == "__main__":
    sys.exit(main())
