"""LLM layer tests use a fake chat model: no Ollama, no API key, no tokens spent."""

import dataclasses

import pytest
from pydantic import BaseModel

from fiberpulse.config import Settings
from fiberpulse.llm import BudgetExceeded, LLMClient, LLMOutputError, TokenLedger


class Answer(BaseModel):
    verdict: str
    confidence: float


class FakeMsg:
    def __init__(self, content="ok", tin=100, tout=20):
        self.content = content
        self.usage_metadata = {"input_tokens": tin, "output_tokens": tout}


class FakeStructured:
    def __init__(self, chat, schema):
        self.chat, self.schema = chat, schema

    def invoke(self, messages):
        self.chat.calls += 1
        payload = self.chat.payloads.pop(0)
        if payload is None:
            return {"raw": FakeMsg("garbage"), "parsed": None, "parsing_error": "bad json"}
        return {"raw": FakeMsg("x"), "parsed": self.schema(**payload), "parsing_error": None}


class FakeChat:
    def __init__(self, payloads=()):
        self.payloads = list(payloads)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return FakeMsg("hello")

    def with_structured_output(self, schema, include_raw=True):
        return FakeStructured(self, schema)


def make_client(tmp_path, chat, **setting_overrides):
    settings = dataclasses.replace(Settings(), data_dir=tmp_path, **setting_overrides)
    ledger = TokenLedger(tmp_path / "ledger.json", cap=settings.api_token_cap,
                         hard_limit=settings.api_token_hard_limit)
    return LLMClient(settings, ledger=ledger, model_factory=lambda p, t, m: chat)


def test_plain_call_records_tokens(tmp_path):
    client = make_client(tmp_path, FakeChat())
    result = client.invoke("context_researcher", [("human", "hi")])
    assert result.text == "hello" and result.tokens_in == 100
    assert client.ledger.tokens_used("ollama") == 120


def test_cache_hit_spends_nothing_and_skips_the_model(tmp_path):
    chat = FakeChat()
    client = make_client(tmp_path, chat)
    client.invoke("context_researcher", [("human", "hi")])
    second = client.invoke("context_researcher", [("human", "hi")])
    assert second.cached and chat.calls == 1
    assert client.ledger.tokens_used("ollama") == 120
    assert client.ledger.data["ollama"]["context_researcher"]["cache_hits"] == 1


def test_structured_output_is_parsed(tmp_path):
    chat = FakeChat([{"verdict": "paradox", "confidence": 0.8}])
    client = make_client(tmp_path, chat)
    result = client.invoke("paradox_triage", [("human", "q")], schema=Answer)
    assert result.parsed.verdict == "paradox"


def test_parse_failure_is_retried_then_succeeds(tmp_path):
    chat = FakeChat([None, {"verdict": "confirmed", "confidence": 0.6}])
    client = make_client(tmp_path, chat)
    result = client.invoke("paradox_triage", [("human", "q")], schema=Answer)
    assert result.attempts == 2 and chat.calls == 2


def test_parse_failure_raises_after_retries(tmp_path):
    chat = FakeChat([None, None, None])
    client = make_client(tmp_path, chat, max_parse_retries=2)
    with pytest.raises(LLMOutputError):
        client.invoke("paradox_triage", [("human", "q")], schema=Answer)


def test_per_agent_provider_override():
    settings = Settings(default_provider="ollama", agent_providers={"paradox_triage": "anthropic"})
    assert settings.provider_for("paradox_triage") == "anthropic"
    assert settings.provider_for("offer_architect") == "ollama"


def test_api_budget_blocks_the_call_before_it_happens(tmp_path):
    chat = FakeChat()
    client = make_client(
        tmp_path, chat, default_provider="anthropic", api_token_cap=1000, api_token_hard_limit=500
    )
    with pytest.raises(BudgetExceeded):
        client.invoke("paradox_triage", [("human", "x" * 4000)], max_tokens=512)
    assert chat.calls == 0  # refused before reaching the model


def test_api_budget_is_cumulative_and_persisted(tmp_path):
    chat = FakeChat()
    client = make_client(
        tmp_path, chat, default_provider="anthropic", api_token_cap=1000, api_token_hard_limit=250
    )
    # Each fake call really spends 120 tokens; the pre-call estimate is ~51 (1 prompt + 50 max).
    client.invoke("paradox_triage", [("human", "a")], max_tokens=50, use_cache=False)  # 0+51 ok -> 120 used
    client.invoke("paradox_triage", [("human", "b")], max_tokens=50, use_cache=False)  # 120+51 ok -> 240 used
    with pytest.raises(BudgetExceeded):  # 240+51=291 > 250
        client.invoke("paradox_triage", [("human", "c")], max_tokens=50, use_cache=False)
    assert chat.calls == 2
    reloaded = TokenLedger(tmp_path / "ledger.json", cap=1000, hard_limit=250)
    assert reloaded.api_tokens_used == 240
