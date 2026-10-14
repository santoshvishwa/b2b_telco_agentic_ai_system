"""Central settings: provider routing, model names, token budget.

Everything comes from environment variables (optionally via a .env file) so the
same code runs unchanged on a laptop, in CI, or later on Azure/Databricks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

# Agents that call an LLM. Names are used for per-agent provider overrides
# (FIBERPULSE_PROVIDER_<NAME>) and for the token ledger.
AGENTS = (
    "context_researcher",  # Agent 3
    "paradox_triage",      # Agent 4
    "offer_architect",     # Agent 5
    "compliance",          # Agent 6
    "prioritizer",         # Agent 7
)

PROVIDERS = ("ollama", "anthropic")

# Synthetic placeholder - NOT a real e& policy value. Agent 6 enforces this in code.
MARGIN_FLOOR = 0.60


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _check_provider(value: str, source: str) -> str:
    value = value.strip().lower()
    if value not in PROVIDERS:
        raise ValueError(f"{source}={value!r} is not one of {PROVIDERS}")
    return value


@dataclass(frozen=True)
class Settings:
    default_provider: str = "ollama"
    agent_providers: dict[str, str] = field(default_factory=dict)

    ollama_model: str = "qwen3:8b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_think: bool = False
    ollama_num_ctx: int = 8192

    anthropic_model: str = "claude-haiku-5-5"
    anthropic_api_key: str | None = None
    api_token_cap: int = 100_000
    api_token_hard_limit: int = 90_000

    max_parse_retries: int = 2
    data_dir: Path = Path("data")

    @classmethod
    def load(cls, env_file: str | os.PathLike | None = None) -> "Settings":
        if load_dotenv is not None:
            load_dotenv(env_file, override=False)

        overrides: dict[str, str] = {}
        for agent in AGENTS:
            key = f"FIBERPULSE_PROVIDER_{agent.upper()}"
            if os.getenv(key):
                overrides[agent] = _check_provider(os.environ[key], key)

        return cls(
            default_provider=_check_provider(
                os.getenv("FIBERPULSE_PROVIDER", "ollama"), "FIBERPULSE_PROVIDER"
            ),
            agent_providers=overrides,
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_think=_env_bool("OLLAMA_THINK", False),
            ollama_num_ctx=_env_int("OLLAMA_NUM_CTX", 8192),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            api_token_cap=_env_int("API_TOKEN_CAP", 100_000),
            api_token_hard_limit=_env_int("API_TOKEN_HARD_LIMIT", 90_000),
            max_parse_retries=_env_int("FIBERPULSE_MAX_PARSE_RETRIES", 2),
            data_dir=Path(os.getenv("FIBERPULSE_DATA_DIR", "data")),
        )

    def provider_for(self, agent: str) -> str:
        """Provider for one agent: per-agent override, else the default."""
        return self.agent_providers.get(agent, self.default_provider)

    def model_for(self, provider: str) -> str:
        return self.ollama_model if provider == "ollama" else self.anthropic_model
