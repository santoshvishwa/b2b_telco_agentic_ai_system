# Day 2 notes (2026-10-14): Phase 1, synthetic data, and GitHub

Everything here is a learning build on **synthetic data**. Prices, margins and policies are invented.

## 1. What we did today

1. Built the **Phase 1 synthetic data generator** (`src/fiberpulse/synth/`, 13 modules).
2. Wrote tests, a data dictionary, six new decision records (ADR-009 to ADR-014) and a day-2 notebook.
3. Set up **GitHub**: Phase 0 pushed and tagged, Phase 1 pushed on a branch, merged into `main` and tagged.
4. Ran the Phase 1 checks on your machine: build validated, **35 tests passed** under real pytest (Python 3.13.5).

Repo: `santoshvishwa/b2b_telco_agentic_ai_system` (currently **public**; change under GitHub Settings > Danger Zone if you want it private).

## 2. What the data contains

| Item | Count |
|---|---|
| Accounts (customer_360) | 20,000 (seed 42) |
| Scores (model_scores) | 40,000, with 1,250 flagged per signal (top 6.25%) |
| Tickets / add-ons / bills | about 37K / 20K / 119K |
| Calls / external signals / offer history | about 5K / 29K / 2.6K |
| Offers in the catalog | 16 (3 below the 60% margin floor, 1 exactly on it) |
| Planted eval accounts | 50: 25 paradox, 25 confirmed, in 11 archetypes |
| Compliance test cases | 42 (about 4 per policy rule) |
| Outcomes knowledge base | 360 past cases, about 10% with a contrary outcome |
| Expected-context truth | 200 accounts |

Build takes about 24 s. Layout: `lakehouse.db`, `raw/*.csv`, `knowledge/`, `eval/` (hidden truth), `manifest.json`.

## 3. The ideas worth remembering

- **Paradox = stale score.** The stand-in model only sees data up to 2026-08-31. Events from 1 Sep to 1 Oct (the scoring date) change the *true* risk but not the score. A flagged account whose fresh events lower the risk is a paradox; one whose fresh events agree is confirmed.
- **Planted truth instead of LLM judges.** Agents are graded by code against `eval/`. Agents must never read `eval/`.
- **Deterministic first.** Margins, prices, rule checks and ranking are code. The LLM judges and explains.
- **Two independent answers for context truth.** pandas (`truth.py`) and SQL (`validate.sql_context`) must agree, or the build fails.
- **Policy oracle.** `evaluate_offer` applies 8 rules (POL-MARGIN-01, DISC-02, FREQ-03, CARE-04, CREDIT-05, CONSENT-06, ELIG-07, DUP-08). Agent 6 is graded against it.
- **Margin floor is inclusive (>= 60%).** Your diagram says "above 60%". Offer UP-10 sits exactly at 60% as the boundary test. If you want strictly above, it is a one-line change in `config.py` (ADR-012). **Still your decision.**

## 4. Commands to remember

```powershell
uv sync --extra dev
uv run python -m fiberpulse.synth.build --n-accounts 20000 --seed 42 --out data/synthetic
uv run python -m fiberpulse.synth.validate --out data/synthetic
uv run pytest
```

Close any program holding `lakehouse.db` open (DB browser, notebook) before rebuilding on Windows.

## 5. Git workflow we settled on

- One repo, one folder. Each phase gets a **branch** `phase-N`, merged into `main` when tests pass.
- Tags use `vX.Y-phaseN` (for example `v0.1-phase1`), plus the existing `phase-0` tag.
- Copy the *contents* of an unzipped pack over the project root, never the folder itself (that creates a nested copy).
- Before `git add -A`, check `git status` does not list `.env`, `.venv/` or `data/`.

Mistakes we hit, so you recognise them:

| Error | Cause | Fix |
|---|---|---|
| `src refspec phase-1 does not match any` | the branch was never created or committed | `git checkout -b phase-1`, commit, then push |
| `src refspec phase-1 matches more than one` | a branch **and** a tag were both named `phase-1` | `git tag -d phase-1`; use `v0.1-phase1` for tags |

## 6. Status and open items

- Done and verified on your machine: Phase 1 build, validation, 35 tests.
- **Not yet confirmed:** `notebooks/day2_explore_data.ipynb` running in VS Code (you are running it next).
- **I cannot push to your repo.** The Claude GitHub App is not installed on your account (https://github.com/apps/claude/installations/select_target). Until then, new phases arrive as zips.
- My sandbox cannot install xgboost or mlflow, so Phase 2 code for them will be written but only partly testable on my side.
- Decision pending: the margin-floor wording (section 3).

## 7. Tomorrow: Phase 2 (models)

Plan, not started:
- Build features from data up to **2026-08-31** only, with a test proving later rows cannot change them (the leakage rule, ADR-010).
- Train XGBoost churn and acquisition models with out-of-fold scoring, tracked in MLflow.
- Take labels from the generator's 3-month outcomes, extracted once into a training file so the models never read `eval/` directly.
- Write the new scores to a separate table (`model_scores_v1`). The planted eval set was built on the old scores, so the agent evals stay on the old table for reproducibility.
- Report how many of the 50 planted accounts the real model still flags, and where the stale-score paradoxes appear.

First thing tomorrow: tell me what the notebook run showed.
