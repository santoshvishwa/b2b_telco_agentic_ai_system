# Decision log

Short records of why things are the way they are. Add a new entry whenever a decision changes the build.

## ADR-001: Provider-agnostic LLM layer
**Decision:** agents call `LLMClient.invoke(agent, messages, schema)`; provider and model come from settings, with
per-agent overrides (`FIBERPULSE_PROVIDER_<AGENT>`).
**Why:** local models for iteration, an API model for the few places quality matters, and later Azure OpenAI, without
touching agent code.

## ADR-002: Local-first on a laptop, API as a scarce resource
**Decision:** default provider is Ollama. The API (100K-token key) is reserved for a benchmark of Agent 4 and one demo run.
**Why:** about 10 accounts' worth of full-pipeline tokens fit in 100K, so the API cannot be the development loop.
**Side benefit:** "runs fully on-prem on a laptop" is a real data-residency story.

## ADR-003: Deterministic first, LLM only for judgment
**Decision:** margin math, price/cost lookup, ranking arithmetic, SQL safety and ID validation are code. The LLM
selects, interprets, explains and retrieves.
**Why:** cheaper, faster on slow local hardware, testable with exact assertions, and it removes a whole class of
hallucinated numbers.

## ADR-004: Typed state, one writer per section
**Decision:** `AccountState` (Pydantic) is the only thing passed between graph nodes. Each agent writes exactly one section.
**Why:** a goal can be verified by validating a schema plus invariants instead of reading prose. Most agent bugs come
from loosely defined state.

## ADR-005: Token ledger, replay cache, hard limit below the cap
**Decision:** every call is metered per provider and agent; identical calls are served from a disk cache; API calls are
refused when used + estimate would pass 90,000 of the 100,000 tokens.
**Why:** a mistake in a loop must not be able to burn the whole key.

## ADR-006: Keep the project separate from the employer
**Decision:** personal machine, synthetic data only, invented prices/margins/policies.
**Why:** protects you if the repo or a write-up is ever shared, and avoids any IP or data-handling question.

## ADR-007: Qwen3 8B with thinking off as the local default
**Decision:** `qwen3:8b`, `OLLAMA_THINK=false`, `num_ctx=8192`; `llama3.2:3b` for plumbing tests only.
**Why:** measured on this laptop (Core Ultra 7 258V, 32 GB, integrated Arc GPU, CPU inference). The model name for
run 1 is assumed to be `qwen3:8b`; confirm with `ollama ps`.

| Run | Total | Output tokens | Generation | Prompt processing |
|---|---|---|---|---|
| Thinking on (default) | 75.6 s | 566 | 7.71 tok/s | 31.96 tok/s |
| Thinking off | 15.5 s | 126 | 9.59 tok/s | 32.21 tok/s |

**Consequence:** prompt processing is the real bottleneck (about 32 tok/s). A 1,000-token prompt costs about 30 s
before the first output token, so prompts are budgeted per agent (see the contract sheet), static instructions go
first so Ollama can reuse the cached prefix, and account context is passed as compact JSON.

## ADR-008: Planted ground truth instead of LLM judges
**Decision:** the synthetic data generator will plant known paradox/confirmed cases, margin violations and
ranking-relevant facts. Agent evals are deterministic code against that truth.
**Why:** zero tokens, exact, repeatable. An LLM judge is used only for explanation quality on about 10 cases, locally.

## ADR-009: SQLite as the lakehouse stand-in
**Decision:** all tables live in one `lakehouse.db` (SQLite) plus CSV copies in `raw/`. Agents query it with SQL.
**Why:** zero setup, one file, same SQL habits as Databricks. Porting later means swapping the connection only.

## ADR-010: Model view vs fresh events (how stale-score paradoxes arise)
**Decision:** the stand-in model ("simulated-v0") sees only features dated up to 2026-08-31. Events from 2026-09-01
to the scoring date (2026-10-01) shift the TRUE propensity but not the score. A flagged account whose fresh events
lower the true risk is a paradox; one whose events agree with the score is confirmed.
**Why:** this is how real paradoxes happen (the score is a month old). It gives Agent 4 a checkable reason to disagree.
**Phase 2 rule:** the real XGBoost model must be trained on features at or before 2026-08-31, scored out-of-fold,
or the paradox mechanism disappears.

## ADR-011: eval/ is hidden from agents; context truth is computed two ways
**Decision:** ground truth, the eval set, compliance cases and expected context live only under `eval/`. Agent code
never reads that folder. Expected context is produced by pandas (`truth.py`) and independently by SQL
(`validate.sql_context`); the build fails if they differ.
**Why:** a grader that shares code with the thing it grades hides bugs in both.

## ADR-012: Margin floor is inclusive (gross margin >= 60%)
**Decision:** `meets_margin_floor` passes at exactly 60%. Offer UP-10 sits exactly on the floor as a boundary test.
**Open:** your diagram says "above 60%". If you mean strictly above, change one line in `config.py`, flip UP-10's
expected result, and the tests will show what moves.

## ADR-013: Scaled counts, transcripts deferred
**Decision:** 20,000 accounts, 1,250 flagged per signal (6.25%), 360 KB cases, 42 compliance cases. Call
transcripts are not generated now (only call summaries); transcripts arrive with the Phase 7 extension.
**Why:** enough for statistics and fast builds (about 23 s) without the weight of free-text generation.

## ADR-014: Reproducibility is per library version
**Decision:** same seed gives identical files on the same numpy/pandas versions; `manifest.json` stores SHA-256
hashes to check this. Different library versions may give different bytes, though the same structure and checks.
**Why:** numpy does not promise bit-identical random streams across versions.

## Open decisions
- Final project name (Project Axon / Maestro / FiberMesh AI / Sentinel; none chosen).
- Whether to port Phases 5+ to Databricks Free Edition.
