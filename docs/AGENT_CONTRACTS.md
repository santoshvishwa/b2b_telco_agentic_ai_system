# Agent contract sheet (core pipeline: Agents 3-7 + orchestrator)

A contract is written **before** an agent's code. It defines the goal, what goes in, what comes out, how it can
fail, and how we test it with the planted ground truth from the synthetic data (Phase 1).

Conventions
- "Writes" is the one `AccountState` section the agent owns (see `src/fiberpulse/state.py`).
- "LLM" lists what the model is allowed to decide. Everything else is code.
- Token targets are prompt-size budgets so local runs stay fast (prompt processing is about 32 tok/s on this laptop).
- Put static instructions first in every prompt so Ollama can reuse the cached prefix.

## Pipeline at a glance

```
scored account (churn or acquisition)
        |
   [3 Researcher] -> [4 Paradox Triage] --confirmed--> [5 Offer Architect] -> [6 Compliance] -> [7 Prioritizer] -> ranked set
                            |
                         paradox
                            v
                     Exception Exit (suppress, route to Service Recovery)
```

| Step | Writes | Status after |
|---|---|---|
| 3 | `context` | `enriched` |
| 4 | `triage` | `triaged` or `exception_exit` |
| 5 | `offers` | `offers_drafted` |
| 6 | `compliance` | `compliance_checked` |
| 7 | `ranking` | `ranked` |

---

## Agent 3: Context Enrichment (The Researcher)

- **Goal:** build the complete account picture the instant a score is detected.
- **Phase:** 3 (first agent built)
- **Reads:** `account_id`, `score`
- **Writes:** `AccountContext`
- **Tools (read-only):** parameterized lookups over Customer_360, SR_Complaint, Addons, Call_Intent_Signals, External_Enrichment.
  Start with fixed query templates; add guarded free-form text-to-SQL later as a deliberate learning step.
- **LLM decides:** only which tools to call and in what order, and a short list of `data_gaps` when something is missing.
- **Deterministic:** every value in `AccountContext` comes straight from query results, never from model text.
- **Query guard:** SELECT only; allow-listed tables; row limit; timeout; reject multiple statements.
- **Token target:** under about 400 prompt tokens.
- **Failure modes:** missing table or account, stale data, SQL errors, tool loops, model inventing a value.
- **Tests (planted truth):**
  - every field equals the generator's ground-truth value for 50 accounts
  - `data_gaps` is populated when a source row is deliberately removed
  - guard blocks `DROP`, `UPDATE`, `DELETE`, stacked statements and non-allow-listed tables

## Agent 4: Paradox Triage (The Auditor)

- **Goal:** validate that the model's signal is real before anyone acts on it.
- **Phase:** 4, first reasoning agent and the heart of the system
- **Reads:** `score`, `context`; retrieves similar cases from the Customer Outcomes KB (RAG)
- **Writes:** `TriageDecision` (`confirmed` or `paradox`, confidence, reasoning, `evidence_case_ids`)
- **LLM decides:** whether the account's evidence agrees with the score, and the explanation.
- **Deterministic:** retrieval (top 3 cases), validation that every cited case ID exists in the KB, routing:
  `paradox` calls `state.exit_with_exception(...)`.
- **Behaviour:**
  - Confirmed: explain why, check similar past cases, pass forward.
  - Paradox: override the score, suppress the action, go to the Exception Exit.
- **Token target:** under about 1,000 prompt tokens (3 retrieved cases, compact context JSON).
- **Failure modes:** sycophancy (always agrees with the score), over-skepticism (flags everything), citing cases that
  do not exist, ignoring `data_gaps`, unstable verdicts on near-identical input.
- **Tests (planted truth):**
  - paradox recall and precision on about 25 planted paradox/confirmed accounts
  - zero cited case IDs outside the KB
  - same verdict on a re-run at temperature 0
  - a sycophancy probe: accounts whose score is high but whose evidence is contradictory must not all be `confirmed`
- **Model plan:** develop on local Qwen3 8B; run the benchmark once on an API model to see the quality gap.

## Agent 5: Offer Architect (The Planner)

- **Goal:** turn a validated signal into the right action.
- **Phase:** 4
- **Reads:** `score`, `context`, `triage`; offer catalog
- **Writes:** `list[OfferCandidate]`
- **LLM decides:** which catalog offers fit this account, and the rationale for each.
- **Deterministic:** the LLM returns offer IDs only; code looks up name, price and cost from the catalog.
  The model never types a price.
- **Logic:** churn signal gives the best save action that protects the account's margin; acquisition signal gives the
  best-fit upsell matched to the account.
- **Token target:** under about 700 prompt tokens.
- **Failure modes:** offer IDs not in the catalog, an offer type that does not match the signal, duplicate offers,
  offering something the account already has.
- **Tests:** 100% of IDs exist in the catalog; churn accounts get retention offers and acquisition accounts get upsells;
  no offer duplicates an existing add-on.

## Agent 6: RAG Compliance (The Risk Officer)

- **Goal:** protect policy and margin before anything reaches Pega.
- **Phase:** 4
- **Reads:** `offers`; corporate policy documents (RAG)
- **Writes:** `ComplianceResult` (one `OfferCheck` per offer)
- **Deterministic (hard rule):** gross margin = (price - cost) / price, computed in code. Margin below `MARGIN_FLOOR`
  (0.60, synthetic) fails regardless of anything the model says.
- **LLM decides:** only whether retrieved policy text applies to this offer and account (for example segment or
  contract-term rules), and the wording of a violation.
- **Token target:** under about 600 prompt tokens.
- **Failure modes:** model overriding the margin rule, citing policy text that was not retrieved, missing a
  segment restriction.
- **Tests:** property test: no offer with margin below the floor ever passes; every `policy_refs` ID exists in the
  policy index; planted policy-violating offers are all caught.

## Agent 7: Recommendation Ranking (The Prioritizer)

- **Goal:** prioritize by impact.
- **Phase:** 4
- **Reads:** compliant offers, `score`, `context`
- **Writes:** `list[RankedOffer]`
- **Deterministic:** expected revenue lift and propensity-to-accept are computed in code from catalog values and
  a propensity function; sorting and rank numbering are code.
- **LLM decides:** only the one-line reasoning attached to each rank.
- **Token target:** under about 500 prompt tokens.
- **Failure modes:** non-compliant offers appearing, rank gaps or ties, reasoning that contradicts the numbers.
- **Tests:** ranks are exactly 1..n; order matches expected lift; every ranked offer is in
  `compliance.compliant_offer_ids`; an empty compliant set yields an empty ranking and a clear status.

## Exception Exit

- **Trigger:** Agent 4 returns `paradox`.
- **Behaviour:** campaign suppressed, `status = exception_exit`, `exception_reason` recorded, Agents 5-7 skipped,
  account routed to the Service Recovery queue.
- **Test:** zero LLM calls for Agents 5-7 on any paradox account.

---

## Orchestrator (LangGraph)

- **Phase:** 3 (two-node skeleton), 5 (full wiring)
- **Graph:** 3 -> 4 -> (conditional) -> 5 -> 6 -> 7 -> done; paradox edge goes to the Exception Exit.
- **Goal verification between steps:** after each agent, run its schema validation plus the invariants listed in its
  contract. A step passes only when its goal check passes.
- **Retries:** up to 2 retries with the validation error fed back; then `status = failed`, error recorded,
  and the batch continues with the next account (one bad account never stops the run).
- **Checkpointing:** persist state per step so a crashed batch resumes instead of restarting.
- **Idempotency:** `AccountState.idempotency_key` (`account:month:signal`); re-running overwrites, never duplicates.
- **Observability:** every step appends a `StepRecord` (agent, duration, provider, model, tokens, ok).
- **Tests:** paradox path skips 5-7; a forced failure in Agent 5 leaves other accounts unaffected; a re-run
  produces identical output (cached) and no duplicate rows.

## Evaluation data (planted by the Phase 1 generator)

All files are under `data/synthetic/eval/`. Agents must never read this folder.

| Planted truth | File | Used to test |
|---|---|---|
| Per-account expected context values (200 accounts) | `context_truth.json` | Agent 3 accuracy |
| 25 paradox and 25 confirmed accounts, 11 archetypes, with labels | `eval_set.json` | Agent 4 recall/precision, sycophancy |
| Offer catalog with price and cost, 3 offers below the margin floor, 1 on the boundary | `lakehouse.db` table `offer_catalog` | Agents 5, 6 |
| 42 offer/account cases with expected rule violations | `compliance_cases.json` | Agent 6 |
| Full hidden drivers and true probabilities | `ground_truth.csv` | Agent 7 and sanity checks |
