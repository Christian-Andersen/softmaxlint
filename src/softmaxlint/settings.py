"""Runtime configuration, sourced from the environment and an optional .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Tunables for talking to the vLLM server and scoring tokens.

    Every field is overridable through an environment variable of the same
    name in upper case (``THRESHOLD_WARN``, ``MODEL_NAME``, ...). A ``.env``
    file in the working directory is read as a fallback.
    """

    server_base_url: str = "http://localhost:8000/v1"
    model_name: str = "Qwen/Qwen3.5-0.8B"
    api_key: str = "EMPTY"
    threshold_warn: float = 0.30
    threshold_crit: float = 0.10

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
