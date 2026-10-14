"""Eval artifacts: planted eval accounts, policy-compliance cases, and context truth.

These files live in eval/ and must never be readable by the agents (they contain the answers).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .catalog import OFFERS, RULE_IDS, AccountFacts, evaluate_offer
from .truth import context_truth, facts_table


def build_eval_set(tables: dict[str, pd.DataFrame], gt: pd.DataFrame) -> list[dict]:
    scores = tables["model_scores"].set_index(["account_id", "signal_type"])["score"]
    planted = gt[gt["archetype"] != "unplanted"].sort_values("account_id")
    cases = []
    for k, r in enumerate(planted.itertuples(index=False), start=1):
        signal = r.planted_signal_type
        true_p = r.true_churn_prob_3m if signal == "churn" else r.true_acquisition_prob_3m
        cases.append({
            "case_id": f"EVAL-{k:03d}",
            "account_id": r.account_id,
            "signal_type": signal,
            "score": float(scores.loc[(r.account_id, signal)]),
            "expected_verdict": r.expected_verdict,
            "difficulty": r.difficulty,
            "archetype": r.archetype,
            "planted_signals": r.planted_signals.split("|"),
            "true_probability_3m": float(true_p),
        })
    return cases


def _facts_dict(f: AccountFacts) -> dict:
    return {
        "segment": f.segment, "tenure_months": f.tenure_months, "marketing_opt_in": f.marketing_opt_in,
        "open_high_ticket": f.open_high_ticket, "late_or_unpaid_last3": f.late_or_unpaid_last3,
        "days_since_last_retention_offer": f.days_since_last_retention_offer,
        "days_since_last_upsell_offer": f.days_since_last_upsell_offer,
        "active_addons": sorted(f.active_addons),
    }


def build_compliance_cases(tables: dict[str, pd.DataFrame], rng: np.random.Generator,
                           per_rule: int = 4, n_pass: int = 8) -> list[dict]:
    """(account, offer) pairs with the exact expected set of violated policy rules."""
    facts = facts_table(tables)
    accounts = [facts_id for facts_id in facts]
    order = rng.permutation(len(accounts))
    accounts = [accounts[i] for i in order]
    offers = list(OFFERS)
    used: set[str] = set()
    cases: list[dict] = []

    def find(target: list[str], k: int, prefer: list[str] | None = None) -> None:
        got = 0
        for acc in accounts:
            if acc in used:
                continue
            pool = ([o for o in offers if o.offer_id in prefer] if prefer else []) + \
                   [offers[j] for j in rng.permutation(len(offers))]
            for offer in pool:
                if evaluate_offer(offer, facts[acc]) == target:
                    used.add(acc)
                    cases.append({"account_id": acc, "offer_id": offer.offer_id,
                                  "expected_pass": not target, "expected_violations": target,
                                  "facts": _facts_dict(facts[acc])})
                    got += 1
                    break
            if got == k:
                return

    for rule in RULE_IDS:
        find([rule], per_rule)
    find([], 1, prefer=["UP-10"])        # boundary: margin exactly on the floor must pass
    find([], 1, prefer=["UP-09"])        # margin just above the floor must pass
    find([], n_pass - 2)
    find(["POL-MARGIN-01", "POL-CONSENT-06"], 1)
    find(["POL-MARGIN-01", "POL-ELIG-07"], 1)

    for k, case in enumerate(cases, start=1):
        case["case_id"] = f"COMP-{k:03d}"
    return [{"case_id": c.pop("case_id"), **c} for c in cases]


def build_context_truth(tables: dict[str, pd.DataFrame], gt: pd.DataFrame,
                        rng: np.random.Generator, n_random: int = 120, n_gap: int = 15) -> dict[str, dict]:
    """Truth for the planted accounts, random accounts, and accounts with deliberate data gaps."""
    planted = gt.loc[gt["archetype"] != "unplanted", "account_id"].tolist()
    planted_set = set(planted)
    rest = [a for a in gt["account_id"] if a not in planted_set]
    picks = rng.permutation(len(rest))
    random_ids = [rest[i] for i in picks[:n_random]]
    pool = [rest[i] for i in picks[n_random:n_random + 6000]]   # candidates for the gap cases

    chosen = planted + random_ids
    full = context_truth(tables, chosen)

    pool_truth = context_truth(tables, pool)
    for gap in ("bills_incomplete", "external_enrichment_none"):
        found = [a for a in pool if gap in pool_truth[a]["data_gaps"] and a not in full][:n_gap]
        for a in found:
            full[a] = pool_truth[a]
    return dict(sorted(full.items()))
