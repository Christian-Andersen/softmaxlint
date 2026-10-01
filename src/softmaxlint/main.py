import math
import sys
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from openai import OpenAI, OpenAIError
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from pathlib import Path


# ==============================================================================
# Settings Configuration via Pydantic-Settings
# ==============================================================================
class Settings(BaseSettings):
    server_base_url: str = "http://localhost:8000/v1"
    model_name: str = "Qwen/Qwen3.5-0.8B"
    api_key: str = "EMPTY"
    file_path: Path | None = None
    threshold_warn: float = 0.30
    threshold_crit: float = 0.10

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


# Instantiate Global Settings & Console
settings: Settings = Settings()
console: Console = Console()

# Type Aliases & TypedDicts (Python 3.14+ syntax)
type AnomalyLevel = Literal["CRITICAL", "WARNING"]


class TokenAnalysis(TypedDict):
    token: str
    actual_prob: float
    top_prob: float
    top_token: str
    ratio: float
    surprisal: float
    rank: int


# ==============================================================================
# Linter Core
# ==============================================================================
class SoftmaxLinter:
    def __init__(self, config: Settings = settings) -> None:
        self.config: Settings = config
        self.client: OpenAI = OpenAI(base_url=self.config.server_base_url, api_key=self.config.api_key)

    def evaluate_snippet(self, code_text: str) -> list[TokenAnalysis]:
        """Sends code to vLLM via OpenAI SDK and extracts token-level log probabilities."""
        try:
            response = self.client.completions.create(
                model=self.config.model_name,
                prompt=code_text,
                max_tokens=0,
                echo=True,
                extra_body={"prompt_logprobs": 1},
            )
        except OpenAIError as e:
            console.print(f"[bold red]Error communicating with vLLM server:[/bold red] {e}")
            sys.exit(1)

        data: dict[str, Any] = response.model_dump()
        choices: list[dict[str, Any]] | None = data.get("choices")
        if not choices:
            console.print("[bold red]Error:[/bold red] No choices returned by server.")
            sys.exit(1)

        prompt_logprobs: list[dict[str, dict[str, Any]] | None] | None = choices[0].get("prompt_logprobs")
        if not prompt_logprobs:
            console.print("[bold red]Error:[/bold red] No prompt_logprobs returned by server.")
            sys.exit(1)

        analysis_results: list[TokenAnalysis] = []

        for item in prompt_logprobs:
            if item is None:
                continue

            candidates: list[dict[str, Any]] = list(item.values())

            actual_entry: dict[str, Any] = next((c for c in candidates if "decoded_token" in c), candidates[0])
            top_entry: dict[str, Any] = min(candidates, key=lambda c: c.get("rank", 999))

            actual_token: str = str(actual_entry.get("decoded_token", ""))
            actual_logprob: float = float(actual_entry["logprob"])
            top_logprob: float = float(top_entry["logprob"])

            actual_prob: float = math.exp(actual_logprob)
            top_prob: float = math.exp(top_logprob)

            ratio: float = actual_prob / top_prob if top_prob > 0 else 1.0
            surprisal: float = -actual_logprob

            analysis_results.append(
                {
                    "token": actual_token,
                    "actual_prob": actual_prob,
                    "top_prob": top_prob,
                    "top_token": str(top_entry.get("decoded_token", "")),
                    "ratio": ratio,
                    "surprisal": surprisal,
                    "rank": int(actual_entry.get("rank", 1)),
                }
            )

        return analysis_results

    def print_heatmap(self, results: list[TokenAnalysis]) -> None:
        """Renders inline code with Rich styles based on anomaly severity."""
        heatmap_text: Text = Text()
        anomalies: list[tuple[int, TokenAnalysis, AnomalyLevel]] = []

        for idx, res in enumerate(results):
            token: str = res["token"]
            ratio: float = res["ratio"]

            if ratio < self.config.threshold_crit:
                # Critical anomaly highlight
                heatmap_text.append(token, style="bold white on red")
                anomalies.append((idx, res, "CRITICAL"))
            elif ratio < self.config.threshold_warn:
                # Warning anomaly highlight
                heatmap_text.append(token, style="bold black on yellow")
                anomalies.append((idx, res, "WARNING"))
            else:
                heatmap_text.append(token)

        # Print Heatmap Panel
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
            return

        # Print Anomalies Summary Table
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

        for _, res, level in anomalies:
            level_style: str = (
                "[bold white on red] CRITICAL [/bold white on red]"
                if level == "CRITICAL"
                else "[bold black on yellow] WARNING  [/bold black on yellow]"
            )
            table.add_row(
                level_style,
                repr(res["token"]),
                f"{res['actual_prob'] * 100:5.2f}%",
                repr(res["top_token"]),
                f"{res['top_prob'] * 100:5.2f}%",
                f"{res['ratio'] * 100:5.1f}%",
            )

        console.print()
        console.print(table)
        console.print()


# ==============================================================================
# Main Execution
# ==============================================================================
def main() -> None:
    code_snippet: str

    if settings.file_path is not None:
        file_path: Path = settings.file_path
        if not file_path.is_file():
            console.print(f"[bold red]Error:[/bold red] File '{file_path}' does not exist or is not a file.")
            sys.exit(1)
        code_snippet = file_path.read_text(encoding="utf-8")
    else:
        # Default test snippet containing an off-by-one boundary bug
        code_snippet = (
            "def calculate_average(numbers):\n"
            "    total = 0\n"
            "    for i in range(len(numbers)):\n"
            "        if i <= len(numbers):\n"
            "            total += numbers[i]\n"
            "    return total / len(numbers)\n"
        )

    linter: SoftmaxLinter = SoftmaxLinter(config=settings)
    results: list[TokenAnalysis] = linter.evaluate_snippet(code_snippet)
    linter.print_heatmap(results)


if __name__ == "__main__":
    main()
