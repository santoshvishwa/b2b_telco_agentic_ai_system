# FiberPulse AI

A hands-on build of a multi-agent B2B fiber CVM system (LangGraph orchestration, RAG, human-in-the-loop),
developed one agent at a time on **synthetic data**, local-first.

> Everything here is invented for learning: account data, offers, prices, the 60% margin floor, policies.
> Do not put real e& tables, policies or pricing in this repo.

## Phase 0 contents

| Path | What it is |
|---|---|
| `src/fiberpulse/state.py` | `AccountState` and the typed output of every agent |
| `src/fiberpulse/config.py` | Settings, per-agent provider routing, the (synthetic) margin floor |
| `src/fiberpulse/llm.py` | Provider-agnostic LLM client: Ollama or Anthropic, structured output, retries, disk cache, **token ledger with a hard API budget** |
| `scripts/smoke_test.py` | Checks packages, Ollama, a plain call, a structured call, the ledger |
| `tests/` | State and LLM-layer tests (fake model, no tokens spent) |
| `docs/AGENT_CONTRACTS.md` | Contract sheet for Agents 3-7, the orchestrator and the exception exit |
| `docs/DECISIONS.md` | Decision log, including your laptop benchmarks |

## Setup on Windows (PowerShell, on your personal laptop)

```powershell
cd fiberpulse
py -3 -m venv .venv                      # any Python 3.10+ works
.\.venv\Scripts\Activate.ps1             # if blocked by execution policy, use: .\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e ".[dev]"                  # quotes matter in PowerShell
copy .env.example .env

ollama pull qwen3:8b
ollama pull nomic-embed-text

pytest
python scripts\smoke_test.py
```

`requirements.txt` is a flat copy of the same dependencies, in case you prefer `pip install -r requirements.txt`.

### What a good smoke test looks like

- Step 1: every package `[ok]`
- Step 2: Ollama reachable, `qwen3:8b` listed
- Step 3: reply `'ready'` in a few seconds
- Step 4: a verdict, a confidence, and a two-sentence reasoning, in roughly 10-30 s
- Step 5: ledger shows `ollama` tokens and `API budget: 0 / 90,000`

The structured call is also a first look at Agent 4's job: does the model challenge a 0.82 churn score when the
customer just upgraded and had a happy call, or does it just agree?

## Token budget (your 100K API key)

- Default provider is local Ollama, so day-to-day work spends **zero** API tokens.
- The API is only used when you set `FIBERPULSE_PROVIDER=anthropic` (or a per-agent override).
- Every call is checked first: if used + estimated tokens would pass `API_TOKEN_HARD_LIMIT` (90,000), the call is
  refused before it is sent. The 10,000 gap absorbs estimation error.
- Usage is saved in `data/token_ledger.json`. Responses are cached in `data/llm_cache/`, so repeating an identical
  call is free.
- Smoke test against the API (spends a few hundred tokens): `python scripts\smoke_test.py --provider anthropic --yes-spend-tokens`

## Known unknowns (not verified yet)

I could not run Ollama or Anthropic calls from where I wrote this, and PyPI was blocked there, so:

1. The 16 tests passed through a minimal stand-in for pytest. Please run real `pytest` once on your machine.
2. `smoke_test.py` has not been run against a live Ollama. Expect to paste me its output and we fix anything that breaks.
3. `llm.py` passes `reasoning=False` to `ChatOllama` to switch Qwen3 thinking off. Older `langchain-ollama` versions
   do not have that option; the code then falls back to a `/no_think` prefix. If thinking is still on (replies are
   slow and very long), run `pip install -U langchain-ollama`.
4. The default API model `claude-haiku-5-5` should be confirmed against what your key can access.

## Next: Phase 1 (synthetic data)

Customer_360, SR_Complaint, Addons, bills, Call_Intent_Signals, External_Enrichment, with hidden churn and
acquisition drivers and deliberately planted paradox cases, plus the policy documents, offer catalog and
Customer Outcomes knowledge base.
