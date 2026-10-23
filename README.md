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

## Phase 1: synthetic data (done)

```powershell
uv sync --extra dev                      # or: pip install -e ".[dev]"
uv run python -m fiberpulse.synth.build --n-accounts 20000 --seed 42 --out data/synthetic
uv run python -m fiberpulse.synth.validate --out data/synthetic
uv run pytest
```

Expect about 20-40 s for the build, then a table of row counts (customer_360 20,000; model_scores 40,000;
1,250 flagged per signal; 360 KB cases) and "all checks passed". Output goes to `data/synthetic/`:
`lakehouse.db`, `raw/*.csv`, `knowledge/` (policy handbook, outcomes KB), `eval/` (hidden truth), `manifest.json`.

Read `docs/DATA_DICTIONARY.md` first, then open `notebooks/day2_explore_data.ipynb`.
On Windows, close any program holding `lakehouse.db` open (DB browser, notebook) before rebuilding.

Not verified on my side: pandas 2.x specifically (the sandbox had pandas 3.0), real `pytest` (I used a
stand-in runner, 35 tests), and identical bytes for the same seed across library versions (see ADR-014).

## Phase 2: real models (done on branch `phase-2`)

```powershell
uv sync --extra dev                      # add --extra mlops if you want MLflow
uv run python -m fiberpulse.models.train --data data/synthetic          # XGBoost; ~1-2 min
uv run python -m fiberpulse.models.report --data data/synthetic
uv run pytest
```

What it does: builds 72 point-in-time features (data up to 2026-08-31 only), trains churn and acquisition models with
5-fold out-of-fold scoring, and writes `model_scores_v1` to the lakehouse plus `data/synthetic/models/`
(`metrics.json`, `importance_*.csv`, saved models, `labels_3m.csv`, `model_scores_v1.csv`). The report compares v1 with
the stand-in v0 on the 50 planted accounts.

Add `--mlflow` to log to a local store, then `uv run mlflow ui --backend-store-uri sqlite:///data/synthetic/mlflow.db`.

Numbers from my run (scikit-learn backend, 20K accounts): AUC 0.69 churn / 0.70 acquisition, top-6.25% lift 3.4-3.8x,
and 32 of 50 planted accounts still flagged. Yours may differ with XGBoost.

Not verified on my side: the XGBoost path, MLflow logging, and a run on Windows.
Read ADR-015 to ADR-018 in `docs/DECISIONS.md`.

## Next: Phase 3 (Agent 3 Researcher + minimal LangGraph)

Agent 3 builds each flagged account's context from the lakehouse; a two-node graph runs it end to end.
