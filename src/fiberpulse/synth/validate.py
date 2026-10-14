"""Consistency checks on a generated dataset.

Most importantly it recomputes every AccountContext field in SQL and compares with the pandas
truth, so the metric definitions are verified two independent ways. The SQL below is also the
reference for Agent 3's tools in Phase 3.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from . import constants as K
from .catalog import OFFER_BY_ID, RULE_IDS, evaluate_offer
from .tables import SCHEMA
from .truth import AS_OF_ISO, expected_bill_months, facts_table

EXPECTED_DB_TABLES = set(SCHEMA)


def sql_context(con: sqlite3.Connection, acc: str) -> dict:
    """Reference SQL for every AccountContext field (definitions: docs/DATA_DICTIONARY.md)."""
    asof = AS_OF_ISO

    def one(sql, *args):
        return con.execute(sql, args).fetchone()[0]

    cust = con.execute("SELECT tenure_months, plan_name, mrc_aed, customer_since FROM customer_360 "
                       "WHERE account_id=?", (acc,)).fetchone()
    bills = [r[0] for r in con.execute(
        "SELECT amount_aed FROM bills WHERE account_id=? ORDER BY bill_month", (acc,))]
    trend = None
    if len(bills) == 6:
        trend = round((sum(bills[3:]) / 3 / (sum(bills[:3]) / 3) - 1.0) * 100.0, 1)
    intent = con.execute(
        "SELECT intent FROM call_intent_signals WHERE account_id=? AND call_date >= date(?, '-30 day') "
        "ORDER BY call_date DESC, call_id DESC LIMIT 1", (acc, asof)).fetchone()
    signals = [f"{s}: {t}" for s, t in con.execute(
        "SELECT source, signal_text FROM external_enrichment WHERE account_id=? "
        "AND signal_date >= date(?, '-60 day') ORDER BY signal_date DESC, signal_id DESC", (acc, asof))]
    gaps = []
    if len(bills) < expected_bill_months(cust[3]):
        gaps.append("bills_incomplete")
    if one("SELECT COUNT(*) FROM external_enrichment WHERE account_id=?", acc) == 0:
        gaps.append("external_enrichment_none")
    return {
        "tenure_months": cust[0],
        "plan_name": cust[1],
        "mrc_aed": float(cust[2]),
        "addon_count": one("SELECT COUNT(*) FROM addons WHERE account_id=? "
                           "AND (end_date IS NULL OR end_date > ?)", acc, asof),
        "open_complaints_90d": one(
            "SELECT COUNT(*) FROM sr_complaint WHERE account_id=? AND is_complaint=1 AND status='Open' "
            "AND opened_date > date(?, '-90 day')", acc, asof),
        "days_since_last_complaint": one(
            "SELECT CAST(julianday(?) - julianday(MAX(opened_date)) AS INTEGER) FROM sr_complaint "
            "WHERE account_id=? AND is_complaint=1", asof, acc),
        "days_since_last_upsell": one(
            "SELECT CAST(julianday(?) - julianday(MAX(d)) AS INTEGER) FROM ("
            "SELECT start_date AS d FROM addons WHERE account_id=? AND is_upsell=1 UNION ALL "
            "SELECT last_plan_change_date FROM customer_360 WHERE account_id=? "
            "AND last_plan_change_type='upgrade')", asof, acc, acc),
        "bill_trend_pct_3m": trend,
        "call_intent": intent[0] if intent else None,
        "external_signals": signals,
        "data_gaps": gaps,
    }


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) < 1e-9
    return a == b


def run_checks(out_dir: str | Path) -> list[str]:
    out = Path(out_dir)
    problems: list[str] = []
    con = sqlite3.connect(out / "lakehouse.db")
    try:
        tables = {t: pd.read_sql_query(f"SELECT * FROM {t}", con) for t in SCHEMA}
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if names != EXPECTED_DB_TABLES:
            problems.append(f"unexpected tables in lakehouse.db: {sorted(names ^ EXPECTED_DB_TABLES)}")
        for t, cols in SCHEMA.items():
            got = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
            if got != [c for c, _ in cols]:
                problems.append(f"{t}: columns differ from schema")

        cust = tables["customer_360"]
        n = len(cust)
        ids = set(cust["account_id"])

        # referential integrity
        for t in ("sr_complaint", "addons", "bills", "call_intent_signals", "external_enrichment",
                  "offer_history", "model_scores"):
            missing = set(tables[t]["account_id"]) - ids
            if missing:
                problems.append(f"{t}: {len(missing)} account_ids not in customer_360")
        bad_offers = set(tables["offer_history"]["offer_id"]) - set(OFFER_BY_ID)
        if bad_offers:
            problems.append(f"offer_history: unknown offers {sorted(bad_offers)}")

        # scores
        ms = tables["model_scores"]
        if len(ms) != 2 * n or ms.groupby("account_id").size().max() != 2:
            problems.append("model_scores: expected exactly two rows (churn, acquisition) per account")
        if not ms["score"].between(0, 1).all():
            problems.append("model_scores: score outside 0-1")
        for sig in ("churn", "acquisition"):
            flagged = int(ms.loc[ms["signal_type"] == sig, "flagged"].sum())
            if abs(flagged - n * K.FLAG_RATE) > 2:
                problems.append(f"model_scores: {sig} flagged={flagged}, expected about {n * K.FLAG_RATE:.0f}")
            low = ms[(ms["signal_type"] == sig) & (ms["flagged"] == 1)]["score"].min()
            if low < 0.69:
                problems.append(f"model_scores: a flagged {sig} account scores {low:.2f} (< 0.70)")

        # dates and values
        since = cust.set_index("account_id")["customer_since"]
        for t, col in (("sr_complaint", "opened_date"), ("call_intent_signals", "call_date"),
                       ("external_enrichment", "signal_date"), ("offer_history", "offer_date"),
                       ("addons", "start_date")):
            df = tables[t]
            early = int((df[col] < df["account_id"].map(since)).sum())
            if early:
                problems.append(f"{t}: {early} rows dated before customer_since")
            if (df[col] >= AS_OF_ISO).any():
                problems.append(f"{t}: rows dated on or after the scoring date")
        tk = tables["sr_complaint"]
        closed = tk[tk["closed_date"].notna()]
        if (closed["closed_date"] < closed["opened_date"]).any():
            problems.append("sr_complaint: closed before opened")
        if ((tk["status"] == "Open") != tk["closed_date"].isna()).any():
            problems.append("sr_complaint: status does not match closed_date")
        if (cust["contract_months_remaining"] > cust["contract_term_months"]).any():
            problems.append("customer_360: contract_months_remaining > contract_term_months")
        if (tables["bills"]["amount_aed"] <= 0).any():
            problems.append("bills: non-positive amount")
        if tables["bills"].duplicated(["account_id", "bill_month"]).any():
            problems.append("bills: duplicate account/month")

        # context truth: SQL vs pandas
        truth = json.loads((out / "eval" / "context_truth.json").read_text(encoding="utf-8"))
        mismatches = 0
        for acc, expected in truth.items():
            got = sql_context(con, acc)
            for field, value in expected.items():
                if not _same(value, got[field]):
                    mismatches += 1
                    if mismatches <= 5:
                        problems.append(f"context mismatch {acc}.{field}: truth={value!r} sql={got[field]!r}")
        if mismatches > 5:
            problems.append(f"... {mismatches} context mismatches in total")
        gap_kinds = {g for v in truth.values() for g in v["data_gaps"]}
        if gap_kinds != {"bills_incomplete", "external_enrichment_none"}:
            problems.append(f"context_truth does not exercise both data gaps (found {sorted(gap_kinds)})")

        # eval set
        ev = json.loads((out / "eval" / "eval_set.json").read_text(encoding="utf-8"))
        verdicts = pd.Series([e["expected_verdict"] for e in ev]).value_counts().to_dict()
        if verdicts != {"confirmed": 25, "paradox": 25}:
            problems.append(f"eval_set verdict counts are {verdicts}, expected 25/25")
        flagged_set = {(r.account_id, r.signal_type) for r in ms[ms["flagged"] == 1].itertuples()}
        for e in ev:
            if (e["account_id"], e["signal_type"]) not in flagged_set:
                problems.append(f"{e['case_id']}: account is not flagged for {e['signal_type']}")
        conf = [e["true_probability_3m"] for e in ev if e["expected_verdict"] == "confirmed"]
        para = [e["true_probability_3m"] for e in ev if e["expected_verdict"] == "paradox"]
        if min(conf) <= max(para):
            problems.append(f"eval labels overlap in truth: min confirmed {min(conf):.2f} <= max paradox {max(para):.2f}")

        # compliance cases: recompute the oracle from the database
        facts = facts_table(tables)
        comp = json.loads((out / "eval" / "compliance_cases.json").read_text(encoding="utf-8"))
        rules_hit = {r for c in comp for r in c["expected_violations"]}
        if rules_hit != set(RULE_IDS):
            problems.append(f"compliance_cases do not cover every rule (missing {sorted(set(RULE_IDS) - rules_hit)})")
        for c in comp:
            got = evaluate_offer(OFFER_BY_ID[c["offer_id"]], facts[c["account_id"]])
            if got != c["expected_violations"] or c["expected_pass"] != (not got):
                problems.append(f"{c['case_id']}: oracle gives {got}, case says {c['expected_violations']}")
        if not any(c["offer_id"] == "UP-10" and c["expected_pass"] for c in comp):
            problems.append("compliance_cases: no passing boundary case for UP-10 (margin exactly 60%)")

        # knowledge base
        kb = [json.loads(line) for line in (out / "knowledge" / "outcomes_kb.jsonl").read_text(
            encoding="utf-8").splitlines()]
        if len({k["case_id"] for k in kb}) != len(kb):
            problems.append("outcomes_kb: duplicate case_id")
        if any("ACC-" in k["text"] for k in kb):
            problems.append("outcomes_kb: contains a current account id (leak)")
    finally:
        con.close()
    return problems


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Validate a generated FiberPulse dataset")
    parser.add_argument("--out", default="data/synthetic")
    args = parser.parse_args()
    problems = run_checks(args.out)
    for p in problems:
        print("PROBLEM:", p)
    print("validation:", "FAILED" if problems else "all checks passed")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
