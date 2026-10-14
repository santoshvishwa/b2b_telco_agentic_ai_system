"""Phase 0 smoke test: is the whole toolchain wired up?

    python scripts/smoke_test.py                 # local Ollama only (free)
    python scripts/smoke_test.py --provider anthropic --yes-spend-tokens

Checks, in order: Python packages, Ollama reachable + model pulled, a plain call,
a structured-output call, and the token ledger. Paste the output back to Claude.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
import urllib.error
import urllib.request

from pydantic import BaseModel, Field

from fiberpulse.config import Settings
from fiberpulse.llm import LLMClient
from fiberpulse.state import Verdict


class SmokeTriage(BaseModel):
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="At most two sentences.")


PARADOX_PROMPT = (
    "An SMB fiber account has a churn-model score of 0.82 (high). But the customer upgraded "
    "their plan last month and had a positive support call last week. Decide whether the "
    "churn score should be trusted. verdict=confirmed means the score is right; "
    "verdict=paradox means the evidence contradicts the score."
)


def check_packages() -> bool:
    ok = True
    for pkg in ("pydantic", "langchain_core", "langchain_ollama", "langchain_anthropic", "langgraph"):
        try:
            importlib.import_module(pkg)
            print(f"  [ok]   {pkg}")
        except ImportError:
            print(f"  [MISS] {pkg}  -> pip install -r requirements.txt")
            ok = False
    return ok


def check_ollama(settings: Settings) -> bool:
    url = settings.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            names = [m["name"] for m in json.load(resp).get("models", [])]
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"  [FAIL] Ollama not reachable at {settings.ollama_base_url} ({exc})")
        print("         Start the Ollama app, then retry.")
        return False
    print(f"  [ok]   Ollama reachable, models: {', '.join(names) or 'none'}")
    if settings.ollama_model not in names:
        print(f"  [MISS] {settings.ollama_model} not pulled -> ollama pull {settings.ollama_model}")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["ollama", "anthropic"], default=None)
    parser.add_argument("--yes-spend-tokens", action="store_true",
                        help="required for --provider anthropic (uses your 100K budget)")
    args = parser.parse_args()

    settings = Settings.load()
    if args.provider:
        if args.provider == "anthropic" and not args.yes_spend_tokens:
            print("Refusing: --provider anthropic spends API tokens. Add --yes-spend-tokens.")
            return 2
        settings = dataclasses.replace(settings, default_provider=args.provider)

    print("1. Packages")
    if not check_packages():
        return 1

    provider = settings.default_provider
    print(f"\n2. Provider: {provider} ({settings.model_for(provider)})")
    if provider == "ollama" and not check_ollama(settings):
        return 1

    client = LLMClient(settings)

    print("\n3. Plain call")
    r = client.invoke("context_researcher", [("human", "Reply with the single word: ready")],
                      max_tokens=20, use_cache=False)
    print(f"   reply={r.text.strip()!r}  {r.duration_s:.1f}s  in={r.tokens_in} out={r.tokens_out}")

    print("\n4. Structured call (mini paradox test)")
    r = client.invoke(
        "paradox_triage",
        [("system", "You are a careful telecom analyst. Be concise."), ("human", PARADOX_PROMPT)],
        schema=SmokeTriage, max_tokens=300, use_cache=False,
    )
    print(f"   verdict={r.parsed.verdict.value}  confidence={r.parsed.confidence}")
    print(f"   reasoning={r.parsed.reasoning}")
    print(f"   {r.duration_s:.1f}s  in={r.tokens_in} out={r.tokens_out}  attempts={r.attempts}")

    print("\n5. Token ledger")
    print("   " + client.ledger.summary().replace("\n", "\n   "))
    print("\nSmoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
