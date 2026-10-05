"""Runtime configuration, sourced from the environment and an optional .env file."""

from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Tunables for talking to the vLLM server and scoring code.

    Every field is overridable through an environment variable of the same
    name in upper case (``THRESHOLD_WARN``, ``MODEL_NAME``, ...). A ``.env``
    file in the working directory is read as a fallback.
    """

    server_base_url: str = "http://127.0.0.1:18020/v1"

    # Which model to score with. Left unset by default: the served model is
    # discovered from the endpoint instead, so upgrading the server does not
    # also mean editing this file. Set it explicitly only to pin a name -- for
    # instance when the server advertises several aliases.
    model_name: str | None = None

    api_key: str = "EMPTY"
    threshold_warn: float = 0.30
    threshold_crit: float = 0.10

    # --- Fill-in-the-middle scoring -----------------------------------------
    # The markers delimit the prefix, suffix and middle of a FIM prompt. The
    # defaults are the Qwen spellings; StarCoder and CodeLlama differ, so
    # switching model family is a config change rather than a code change.
    fim_prefix_marker: str = "<|fim_prefix|>"
    fim_suffix_marker: str = "<|fim_suffix|>"
    fim_middle_marker: str = "<|fim_middle|>"

    # How many candidate tokens to ask for at the hole. Tokens the server does
    # not name are recorded as probability-below-the-floor, so a larger value
    # means fewer censored scores at the cost of a larger response. vLLM's
    # server-side default ceiling for this is 20.
    fim_top_logprobs: int = 20

    # How probes are placed.
    #
    # "server" asks vLLM to echo a tokenization and probes each real token
    # boundary. This is the only mode that does not manufacture false positives:
    # the candidate list at a real boundary always contains the token that was
    # actually written, so ordinary code scores near 1.0.
    #
    # "line" and "coarse" split the text locally instead. They cost nothing
    # extra but disagree with the model's tokenizer, so most probes are
    # censored and reported CRITICAL. They exist for offline testing and for
    # inspecting a snippet without a live server, not for real runs.
    fim_slice_mode: Literal["server", "line", "coarse"] = "server"

    # What to do if the server will not report a tokenization.
    #
    # Falling back keeps the run alive but guarantees a flood of false
    # positives, so the default is to fail loudly. Set true only when you are
    # deliberately inspecting an unaligned run.
    fim_allow_unaligned_probes: bool = False

    # Largest number of FIM prompts per HTTP request. Bounded so a big file
    # cannot produce one unstreamable payload; the server still runs the
    # slices concurrently.
    fim_batch_size: int = 256

    # Minimum top-candidate probability for a finding to count. Below it the
    # distribution is flat and a low ratio means one guess among many, not a
    # wrong token.
    min_top_prob: float = 0.5

    # Skip strings and comments without asking the server. Prose surprisal is
    # not a code signal; skipped spans are recorded as trivially expected.
    # Skipping also shrinks each file's request count.
    score_strings_and_comments: bool = False

    # Presentation for `lsp check`, mirroring `ruff check --output-format`.
    # Only concise (human) and json (machine) are offered; the SARIF/JUnit
    # CI-reporter formats have no editor use.
    output_format: Literal["concise", "json"] = "json"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("model_name", mode="before")
    @classmethod
    def _blank_model_name_means_discover(cls, value: object) -> object:
        """Treat a blank ``MODEL_NAME=`` in a .env file as unset, not as ""."""
        return None if isinstance(value, str) and not value.strip() else value
