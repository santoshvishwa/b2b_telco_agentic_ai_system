# Data dictionary: Phase 1 synthetic data

Everything here is invented: accounts, prices, margins, policies, outcomes. Build it with

```
python -m fiberpulse.synth.build --n-accounts 20000 --seed 42 --out data/synthetic
```

## 1. Layout and who may read what

| Path | Purpose | Readable by agents? |
|---|---|---|
| `lakehouse.db` | SQLite stand-in for the Databricks tables (read-only for decision agents) | Yes |
| `raw/*.csv` | The same tables as CSV | Yes |
| `knowledge/policy_handbook.md` | Source for Agent 6's policy RAG index (each `## POL-...` section is one chunk) | Yes |
| `knowledge/outcomes_kb.jsonl` | Source for Agent 4's Customer Outcomes RAG index (historical cases) | Yes |
| `eval/` | Ground truth, eval sets, context truth. **Contains the answers.** | **Never** |
| `manifest.json` | Seed, row counts, file hashes, validation result | n/a |

## 2. The idea that creates paradox cases

| Date | Meaning |
|---|---|
| `2026-08-31` | **Model snapshot.** The predictive model only ever saw data up to here. |
| `2026-09-01` to `2026-09-30` | **Fresh window.** Events here are invisible to the model but visible in the tables. |
| `2026-10-01` | **Scoring date** (`AS_OF`). "Today" for every agent and every definition below. |

The stand-in model scores accounts from snapshot features only. The TRUE propensity (kept in `eval/ground_truth.csv`)
also depends on fresh events. A high score with fresh good news (upgrade, renewal, happy call) is a **paradox**; a high
score with fresh bad news is **confirmed**. For acquisition, fresh blockers (open High complaint, unpaid bill,
distress, a recent purchase) make a high score a paradox.

## 3. Tables

### `customer_360`

| Column | Type | Meaning |
|---|---|---|
| `account_id` | TEXT (PK) | Primary key, `ACC-000001` style. |
| `company_name` | TEXT | Invented company name (random word combination; any resemblance to a real company is coincidental). |
| `industry` | TEXT | One of 10 industries. |
| `emirate` | TEXT | Emirate of the main site. |
| `segment` | TEXT | `Micro` (1-9 employees), `Small` (10-49), `Medium` (50-249). |
| `employees` | INTEGER | Headcount. |
| `plan_name` | TEXT | Current fiber plan (after any plan change). |
| `plan_speed_mbps` | INTEGER | Plan speed. |
| `mrc_aed` | INTEGER | Current monthly recurring charge for the plan, in AED (invented prices). |
| `tenure_months` | INTEGER | Months since `customer_since`, 3-120. |
| `customer_since` | TEXT | Account start date. |
| `contract_term_months` | INTEGER | 0 = rolling monthly, otherwise 12 or 24. |
| `contract_months_remaining` | INTEGER | 0 = out of contract (rolling, or term finished). |
| `last_renewal_date` | TEXT | Date of the latest contract renewal, NULL for rolling accounts. |
| `payment_method` | TEXT | `Auto-debit`, `Invoice` or `Cheque`. |
| `marketing_opt_in` | INTEGER | 1 = consent to marketing offers, 0 = opted out (policy POL-CONSENT-06). |
| `avg_utilization_pct` | REAL | Average bandwidth utilisation, 10-98. |
| `addon_count` | INTEGER | Add-ons active at the scoring date (derived from `addons`). |
| `last_plan_change_date` | TEXT | Date of the latest plan change, NULL if none. |
| `last_plan_change_type` | TEXT | `upgrade` or `downgrade`, NULL if none. |

### `sr_complaint`

| Column | Type | Meaning |
|---|---|---|
| `ticket_id` | TEXT (PK) | Primary key. |
| `account_id` | TEXT | Foreign key. |
| `opened_date` | TEXT | Date opened. |
| `closed_date` | TEXT | Date closed, NULL while open. |
| `category` | TEXT | Complaints: Outage, Slow speed, Billing dispute, Installation delay, SLA breach. Requests: Service request, Upgrade request. |
| `is_complaint` | INTEGER | 1 for the five complaint categories, 0 for requests. |
| `severity` | TEXT | `Low`, `Medium`, `High` (requests are always Low). |
| `status` | TEXT | `Open` or `Closed` (Open exactly when `closed_date` is NULL). |
| `csat` | INTEGER | Customer satisfaction 1-5 for closed, surveyed tickets, otherwise NULL. |
| `resolution_hours` | INTEGER | Hours to resolve, NULL while open. |

### `addons`

| Column | Type | Meaning |
|---|---|---|
| `addon_row_id` | TEXT (PK) | Primary key. |
| `account_id` | TEXT | Foreign key. |
| `addon_name` | TEXT | One of 8 add-on products. |
| `monthly_price_aed` | INTEGER | Monthly price. |
| `start_date` | TEXT | Start date. |
| `end_date` | TEXT | NULL while active. |
| `is_upsell` | INTEGER | 1 if sold through an upsell, 0 if bundled at sale. |

### `bills`

| Column | Type | Meaning |
|---|---|---|
| `bill_id` | TEXT (PK) | Primary key. |
| `account_id` | TEXT | Foreign key. |
| `bill_month` | TEXT | `YYYY-MM`, 2026-04 to 2026-09. Months before the account existed have no bill; ~1.5% of other accounts miss one bill on purpose. |
| `amount_aed` | REAL | Bill amount (plan + add-ons, prorated for mid-month changes, small usage noise). |
| `paid` | INTEGER | 0 only for an unpaid, overdue September bill. |
| `days_late` | INTEGER | Days paid after the due date (0-3 is on time). For an unpaid bill, days overdue at the scoring date. |
| `disputed` | INTEGER | 1 if the customer disputed the bill. |

### `call_intent_signals`

| Column | Type | Meaning |
|---|---|---|
| `call_id` | TEXT (PK) | Primary key. |
| `account_id` | TEXT | Foreign key. |
| `call_date` | TEXT | Call date. |
| `direction` | TEXT | `inbound` or `outbound`. |
| `duration_sec` | INTEGER | Call length. |
| `intent` | TEXT | `acquisition`, `enquiry` or `complaint`. |
| `sentiment` | TEXT | `positive` (> 0.25), `negative` (< -0.25) or `neutral`. |
| `sentiment_score` | REAL | -1 to 1. |
| `summary` | TEXT | One-line summary. No personal data (masking is assumed done upstream). |

### `external_enrichment`

| Column | Type | Meaning |
|---|---|---|
| `signal_id` | TEXT (PK) | Primary key. |
| `account_id` | TEXT | Foreign key. |
| `signal_date` | TEXT | Date of the signal. |
| `source` | TEXT | `LinkedIn`, `X/Twitter`, `Financial services` or `Utility service`. |
| `signal_text` | TEXT | Short text of the signal. |
| `polarity` | TEXT | `positive`, `negative` or `neutral`. |

### `offer_history`

| Column | Type | Meaning |
|---|---|---|
| `history_id` | TEXT (PK) | Primary key. |
| `account_id` | TEXT | Foreign key. |
| `offer_id` | TEXT | Catalog offer that was made. |
| `offer_date` | TEXT | Date the offer was made (used for contact-frequency policy). |
| `accepted` | INTEGER | 1 if accepted. |
| `channel` | TEXT | `SMS`, `Email` or `Push`. |

### `model_scores`

| Column | Type | Meaning |
|---|---|---|
| `account_id` | TEXT | Part of the key. |
| `scored_month` | TEXT | `2026-10`. |
| `signal_type` | TEXT | `churn` or `acquisition`; every account has both. |
| `score` | REAL | Propensity score 0-1, rank-calibrated (NOT a calibrated probability). |
| `model_version` | TEXT | `simulated-v0` until Phase 2 replaces it. |
| `score_rank` | INTEGER | 1 = highest score for that signal. |
| `flagged` | INTEGER | 1 for the top 6.25% per signal (the accounts the agents work on). |

### `offer_catalog`

| Column | Type | Meaning |
|---|---|---|
| `offer_id` | TEXT (PK) | Primary key (`RET-` retention, `UP-` upsell). |
| `name` | TEXT | Offer name. |
| `offer_type` | TEXT | `retention` answers a churn signal, `upsell` an acquisition signal. |
| `monthly_price_aed` | REAL | What the customer pays per month for the offer. |
| `monthly_cost_aed` | REAL | Our cost to deliver it per month. Gross margin = (price - cost) / price, computed in code, not stored. |
| `discount_pct` | REAL | Discount embedded in the offer. |
| `term_months` | INTEGER | Commitment length. |
| `eligible_segments` | TEXT | Pipe-separated segments. |
| `min_tenure_months` | INTEGER | Minimum customer tenure. |
| `addon_name` | TEXT | Add-on the customer holds after accepting, NULL for plan changes and price locks. |
| `description` | TEXT | Short description. |

## 4. AccountContext definitions (what Agent 3 must reproduce)

All windows are relative to the scoring date `2026-10-01`. `eval/context_truth.json` holds the expected values for 200
accounts; `validate.py` recomputes them with SQL (`sql_context`) as an independent check. That SQL is the reference
for Agent 3's tools.

| Field | Definition |
|---|---|
| `tenure_months` | `customer_360.tenure_months` |
| `plan_name`, `mrc_aed` | Current values from `customer_360` |
| `addon_count` | Add-ons with `end_date` NULL or after the scoring date |
| `open_complaints_90d` | Complaints (`is_complaint=1`) with `status='Open'` opened after `2026-07-03` |
| `days_since_last_complaint` | Scoring date minus the latest complaint `opened_date`; NULL if none |
| `days_since_last_upsell` | Scoring date minus the latest of: add-on `start_date` where `is_upsell=1`, and `last_plan_change_date` where the type is `upgrade`; NULL if none |
| `bill_trend_pct_3m` | Needs all 6 bills: (mean of the latest 3 / mean of the earlier 3 - 1) x 100, one decimal; otherwise NULL |
| `call_intent` | `intent` of the latest call on or after `2026-09-01`; NULL if no call in that window |
| `external_signals` | `"source: signal_text"` for signals on or after `2026-08-02`, newest first |
| `data_gaps` | `bills_incomplete` if the account has fewer bills than months it existed in Apr-Sep; `external_enrichment_none` if it has no enrichment rows at all |

## 5. Scores

- Every account has a churn score and an acquisition score. The top **6.25% per signal** are flagged (1,250 each at
  20,000 accounts, so about 2,500 flagged in total, matching the "2-3K" in the architecture).
- Flagged scores range 0.70 to about 0.99. A score is a rank-calibrated propensity, not a probability.
- Real base rates: about 4% of accounts churn and 6% acquire in 3 months. Flagged accounts are only about 14% (churn)
  and 18% (acquisition) truly likely, so many high scores are naturally weak.
- **Phase 2 rule:** a real model must train only on information available up to `2026-08-31` (tickets, bills and
  add-ons dated after it are off limits), and scores must be out-of-fold so no account is scored by a model that
  trained on it. Eval accounts are drawn from the top 4% by the stand-in score so they should stay flagged.

## 6. Planted eval set (`eval/eval_set.json`)

50 flagged accounts, 25 paradox and 25 confirmed, every one opted in and with no open pre-snapshot ticket. Each has a
story written into the tables through September events.

| Archetype | Signal | Truth | Difficulty | Count |
|---|---|---|---|---|
| `churn_true_risk_easy` | churn | confirmed | easy | 10 |
| `churn_true_risk_hard` | churn | confirmed | hard | 5 |
| `churn_stale_recovered_easy` | churn | paradox | easy | 6 |
| `churn_stale_renewed_easy` | churn | paradox | easy | 4 |
| `churn_stale_hard` | churn | paradox | hard | 5 |
| `acq_true_opportunity_easy` | acquisition | confirmed | easy | 7 |
| `acq_true_opportunity_hard` | acquisition | confirmed | hard | 3 |
| `acq_blocked_complaint_easy` | acquisition | paradox | easy | 3 |
| `acq_blocked_financial_easy` | acquisition | paradox | easy | 2 |
| `acq_saturated_easy` | acquisition | paradox | easy | 2 |
| `acq_blocked_hard` | acquisition | paradox | hard | 3 |

Hard cases mix a few opposing minor signals into a clear net direction. Planted signal codes: `open_complaint`,
`open_high_complaint`, `minor_open_ticket`, `complaint_resolved_good_csat`, `negative_call`, `mild_negative_call`,
`positive_call`, `acquisition_call`, `bill_unpaid`, `bill_disputed`, `bill_late`, `bill_on_time`, `recent_upsell`,
`second_recent_purchase`, `recent_renewal`, `growth_signal`, `distress_signal`, `competitor_signal`.

Every confirmed case has a true probability above every paradox case (checked by the build). The rest of the flagged
accounts carry random September events and have no label; their truth is only the probability in `ground_truth.csv`.

## 7. Offer catalog and policy rules

| Offer | Type | Price | Cost | Margin | Discount | Term | Segments | Min tenure | Add-on |
|---|---|---|---|---|---|---|---|---|---|
| `RET-01` Loyalty Wi-Fi Bundle | retention | 260 | 70 | 73.1% | 20% | 12 | Micro/Small/Medium | 6 | Managed Wi-Fi |
| `RET-02` Speed Step-Up | retention | 180 | 55 | 69.4% | 10% | 12 | Micro/Small/Medium | 6 | - |
| `RET-03` Price Lock 24 | retention | 120 | 30 | 75.0% | 0% | 24 | Micro/Small/Medium | 12 | - |
| `RET-04` Premium Care Plus | retention | 320 | 150 | 53.1% **below floor** | 0% | 12 | Small/Medium | 12 | - |
| `RET-05` Mega Loyalty Rebate | retention | 400 | 140 | 65.0% | 30% | 12 | Micro/Small/Medium | 6 | - |
| `RET-06` Security Shield Loyalty Pack | retention | 200 | 55 | 72.5% | 15% | 12 | Micro/Small/Medium | 6 | Cybersecurity Bundle |
| `UP-01` Extra Bandwidth +200 | upsell | 350 | 110 | 68.6% | 0% | 12 | Micro/Small/Medium | 3 | Extra Bandwidth |
| `UP-02` Cybersecurity Bundle | upsell | 220 | 60 | 72.7% | 0% | 12 | Micro/Small/Medium | 3 | Cybersecurity Bundle |
| `UP-03` Cloud Backup 1TB | upsell | 150 | 35 | 76.7% | 0% | 12 | Micro/Small/Medium | 3 | Cloud Backup |
| `UP-04` VoIP 5 Lines | upsell | 280 | 120 | 57.1% **below floor** | 0% | 12 | Micro/Small/Medium | 3 | VoIP Lines |
| `UP-05` SD-WAN Lite | upsell | 900 | 330 | 63.3% | 0% | 24 | Medium | 12 | SD-WAN Lite |
| `UP-06` Managed Wi-Fi | upsell | 200 | 50 | 75.0% | 0% | 12 | Micro/Small/Medium | 3 | Managed Wi-Fi |
| `UP-07` Static IP Block | upsell | 90 | 15 | 83.3% | 0% | 12 | Micro/Small/Medium | 3 | Static IP |
| `UP-08` Symmetric Premium Upgrade | upsell | 600 | 250 | 58.3% **below floor** | 0% | 12 | Small/Medium | 6 | - |
| `UP-09` 4G Business Continuity | upsell | 240 | 95 | 60.4% | 0% | 12 | Micro/Small/Medium | 3 | 4G Backup |
| `UP-10` Business Analytics Pack | upsell | 500 | 200 | 60.0% **exactly on floor** | 0% | 12 | Micro/Small/Medium | 6 | - |

Policy rules (`catalog.evaluate_offer` is the oracle used for evals; Agent 6 must reason from
`knowledge/policy_handbook.md`):

| Rule | Violated when |
|---|---|
| `POL-MARGIN-01` | Gross margin below 60%. Exactly 60.0% passes (inclusive floor, see ADR-012). |
| `POL-DISC-02` | Offer discount above the segment cap (Micro 15%, Small 20%, Medium 25%) |
| `POL-FREQ-03` | Retention offer within 90 days of the last retention offer, or upsell within 30 days of the last upsell offer |
| `POL-CARE-04` | Upsell offered while a High-severity complaint is open |
| `POL-CREDIT-05` | 2+ late bills among the last 3, and the offer costs over 300 AED a month or runs 24+ months |
| `POL-CONSENT-06` | Customer has not opted in to marketing |
| `POL-ELIG-07` | Segment not eligible, or tenure below the offer's minimum |
| `POL-DUP-08` | Offer provides an add-on the customer already holds |

`eval/compliance_cases.json` holds 42 (account, offer) pairs with the exact expected set of violated rules: 4 per rule,
8 clean passes (including the 60.0% boundary and a 60.4% margin), and 2 multi-violation cases.

## 8. Outcomes knowledge base (`knowledge/outcomes_kb.jsonl`)

360 historical cases from a separate cohort (`H-0001` ...), so retrieval can never leak an eval account. Fields:
`case_id`, `hist_account_id`, `signal_type`, `segment`, `industry`, `tenure_months`, `score_at_flag`, `context`,
`action`, `outcome`, `signal_validity` (`real` or `stale_or_blocked`), `lesson`, and `text` (the string to embed).
About 10% of paradox-pattern cases have a contrary outcome on purpose, so the KB is helpful but not an oracle.

## 9. Reproducibility

- The same seed, numpy and pandas versions reproduce the same files (`manifest.json` holds SHA-256 hashes).
- Different numpy or pandas versions may change random streams: regenerate on each machine instead of comparing files
  across machines. The planted counts and all checks hold either way.
- Close any notebook or tool that has `lakehouse.db` open before rebuilding, or Windows will refuse to replace it.
- Call transcripts and PII masking are deferred to Phase 7 (Agent 2); `call_intent_signals.summary` is already clean text.
