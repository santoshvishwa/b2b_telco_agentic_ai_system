"""Provider-agnostic LLM client with a token ledger and a disk cache.

Agents never talk to Ollama or Anthropic directly. They call
`LLMClient.invoke(agent=..., messages=..., schema=...)`. That gives us:

  * one place to switch provider (per agent, via env vars)
  * a hard token budget on the API provider (the 100K-token key)
  * a replay cache so re-running a test never spends tokens or time twice
  * structured (Pydantic) output with bounded retries
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import BaseModel

from .config import Settings

Message = tuple[str, str]  # ("system" | "human", text)


class BudgetExceeded(RuntimeError):
    """Raised before an API call that would push usage past the hard limit."""


class LLMOutputError(RuntimeError):
    """The model never produced output matching the requested schema."""


# --- Token ledger ------------------------------------------------------------


class TokenLedger:
    """Persistent per-provider / per-agent token accounting."""

    def __init__(self, path: Path, cap: int, hard_limit: int):
        self.path = Path(path)
        self.cap = cap
        self.hard_limit = hard_limit
        self.data: dict[str, dict[str, dict[str, int]]] = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def record(
        self, provider: str, agent: str, tokens_in: int, tokens_out: int, cached: bool = False
    ) -> None:
        row = self.data.setdefault(provider, {}).setdefault(
            agent, {"calls": 0, "cache_hits": 0, "tokens_in": 0, "tokens_out": 0}
        )
        if cached:
            row["cache_hits"] += 1
        else:
            row["calls"] += 1
            row["tokens_in"] += tokens_in
            row["tokens_out"] += tokens_out
        self._save()

    def tokens_used(self, provider: str) -> int:
        rows = self.data.get(provider, {}).values()
        return sum(r["tokens_in"] + r["tokens_out"] for r in rows)

    @property
    def api_tokens_used(self) -> int:
        return self.tokens_used("anthropic")

    def check_api_budget(self, estimate: int) -> None:
        used = self.api_tokens_used
        if used + estimate > self.hard_limit:
            raise BudgetExceeded(
                f"API call refused: {used:,} used + ~{estimate:,} estimated would exceed "
                f"the hard limit of {self.hard_limit:,} (cap {self.cap:,})."
            )

    def summary(self) -> str:
        lines = []
        for provider, agents in sorted(self.data.items()):
            total = self.tokens_used(provider)
            lines.append(f"{provider}: {total:,} tokens")
            for agent, r in sorted(agents.items()):
                lines.append(
                    f"  {agent:<20} calls={r['calls']:<4} cache_hits={r['cache_hits']:<4} "
                    f"in={r['tokens_in']:,} out={r['tokens_out']:,}"
                )
        if not lines:
            lines.append("no usage recorded yet")
        lines.append(
            f"API budget: {self.api_tokens_used:,} / {self.hard_limit:,} hard limit "
            f"({self.cap:,} cap)"
        )
        return "\n".join(lines)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2)
        os.replace(tmp, self.path)  # atomic: a crash never leaves a half-written ledger


# --- Result type -------------------------------------------------------------


@dataclass
class LLMResult:
    text: str
    parsed: BaseModel | None
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    duration_s: float
    cached: bool = False
    attempts: int = 1


# --- Client ------------------------------------------------------------------

ModelFactory = Callable[[str, float, int], Any]  # (provider, temperature, max_tokens) -> chat model


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)


def _rough_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class LLMClient:
    def __init__(
        self,
        settings: Settings | None = None,
        ledger: TokenLedger | None = None,
        model_factory: ModelFactory | None = None,
    ):
        self.settings = settings or Settings.load()
        self.ledger = ledger or TokenLedger(
            self.settings.data_dir / "token_ledger.json",
            cap=self.settings.api_token_cap,
            hard_limit=self.settings.api_token_hard_limit,
        )
        self.cache_dir = self.settings.data_dir / "llm_cache"
        self._factory = model_factory or self._default_factory
        self._needs_nothink_prefix = False  # set if the installed ChatOllama lacks `reasoning`

    # -- public ---------------------------------------------------------------

    def invoke(
        self,
        agent: str,
        messages: Sequence[Message],
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        use_cache: bool = True,
    ) -> LLMResult:
        provider = self.settings.provider_for(agent)
        model_name = self.settings.model_for(provider)
        messages = list(messages)

        key = self._cache_key(provider, model_name, messages, schema, temperature, max_tokens)
        if use_cache:
            hit = self._cache_read(key, schema)
            if hit is not None:
                self.ledger.record(provider, agent, 0, 0, cached=True)
                return LLMResult(
                    text=hit["text"],
                    parsed=hit["parsed"],
                    provider=provider,
                    model=model_name,
                    tokens_in=0,
                    tokens_out=0,
                    duration_s=0.0,
                    cached=True,
                )

        model = self._factory(provider, temperature, max_tokens)
        sent = self._apply_nothink(provider, messages)
        runner = model.with_structured_output(schema, include_raw=True) if schema else model

        attempts_allowed = 1 + (self.settings.max_parse_retries if schema else 0)
        total_in = total_out = 0
        started = time.perf_counter()
        last_error: str | None = None

        for attempt in range(1, attempts_allowed + 1):
            if provider == "anthropic":
                estimate = sum(_rough_tokens(m[1]) for m in sent) + max_tokens
                self.ledger.check_api_budget(estimate)

            raw_out = runner.invoke(sent)
            raw_msg = raw_out["raw"] if schema else raw_out
            usage = getattr(raw_msg, "usage_metadata", None) or {}
            t_in = int(usage.get("input_tokens") or sum(_rough_tokens(m[1]) for m in sent))
            t_out = int(
                usage.get("output_tokens") or _rough_tokens(_content_to_text(raw_msg.content))
            )
            total_in += t_in
            total_out += t_out
            self.ledger.record(provider, agent, t_in, t_out)

            if not schema:
                text = _content_to_text(raw_msg.content)
                self._cache_write(key, {"text": text, "parsed": None})
                return LLMResult(
                    text, None, provider, model_name, total_in, total_out,
                    time.perf_counter() - started, attempts=attempt,
                )

            parsed = raw_out.get("parsed")
            if parsed is not None and not raw_out.get("parsing_error"):
                text = parsed.model_dump_json()
                self._cache_write(key, {"text": text, "parsed": parsed.model_dump(mode="json")})
                return LLMResult(
                    text, parsed, provider, model_name, total_in, total_out,
                    time.perf_counter() - started, attempts=attempt,
                )

            last_error = str(raw_out.get("parsing_error"))
            sent = sent + [
                ("human", "Your previous reply did not match the required schema. "
                          "Reply again with valid structured output only.")
            ]

        raise LLMOutputError(
            f"{agent}: no valid {schema.__name__} after {attempts_allowed} attempts "
            f"({last_error})"
        )

    # -- internals ------------------------------------------------------------

    def _apply_nothink(self, provider: str, messages: list[Message]) -> list[Message]:
        if provider == "ollama" and self._needs_nothink_prefix and not self.settings.ollama_think:
            return [("system", "/no_think")] + messages
        return messages

    def _default_factory(self, provider: str, temperature: float, max_tokens: int) -> Any:
        s = self.settings
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            kwargs = dict(
                model=s.ollama_model,
                base_url=s.ollama_base_url,
                temperature=temperature,
                num_ctx=s.ollama_num_ctx,
                num_predict=max_tokens,
            )
            try:
                return ChatOllama(reasoning=s.ollama_think, **kwargs)
            except Exception:
                # Older langchain-ollama without `reasoning`: fall back to Qwen3's
                # "/no_think" text switch. Upgrading langchain-ollama is the cleaner fix.
                self._needs_nothink_prefix = True
                return ChatOllama(**kwargs)

        if provider == "anthropic":
            if not s.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example).")
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=s.anthropic_model,
                api_key=s.anthropic_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        raise ValueError(f"unknown provider {provider!r}")

    def _cache_key(
        self,
        provider: str,
        model: str,
        messages: list[Message],
        schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "provider": provider,
            "model": model,
            "think": self.settings.ollama_think if provider == "ollama" else None,
            "messages": [list(m) for m in messages],
            "schema": schema.model_json_schema() if schema else None,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_read(self, key: str, schema: type[BaseModel] | None) -> dict | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            parsed = schema.model_validate(entry["parsed"]) if schema and entry["parsed"] else None
            if schema and parsed is None:
                return None
            return {"text": entry["text"], "parsed": parsed}
        except Exception:
            return None  # corrupt or stale entry: just call the model again

    def _cache_write(self, key: str, entry: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps(entry, ensure_ascii=False), encoding="utf-8"
        )
