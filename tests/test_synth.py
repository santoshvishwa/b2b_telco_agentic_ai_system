"""Tests for the Phase 1 synthetic data: determinism, integrity, planted truth, policy oracle."""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from fiberpulse.config import MARGIN_FLOOR, meets_margin_floor
from fiberpulse.synth import constants as K
from fiberpulse.synth.archetypes import PLAN
from fiberpulse.synth.build import build
from fiberpulse.synth.catalog import (
    OFFER_BY_ID, OFFERS, RULE_IDS, AccountFacts, evaluate_offer, render_handbook,
)
from fiberpulse.synth.tables import SCHEMA
from fiberpulse.synth.validate import sql_context

N = 2000
_CACHE: dict = {}


def built(seed: int = 42, n: int = N):
    """Build once per (seed, n) into a temp dir and reuse across tests."""
    key = (seed, n)
    if key not in _CACHE:
        out = Path(tempfile.mkdtemp(prefix="fp_synth_"))
        _CACHE[key] = (out, build(n, seed, out, quiet=True))
    return _CACHE[key]


def clean_facts(**over):
    base = dict(segment="Small", tenure_months=24, marketing_opt_in=True, open_high_ticket=False,
                late_or_unpaid_last3=0, days_since_last_retention_offer=None,
                days_since_last_upsell_offer=None, active_addons=frozenset())
    base.update(over)
    return AccountFacts(**base)


# --- build, determinism, integrity -----------------------------------------------------


def test_full_build_passes_every_validation_check():
    _, manifest = built()
    assert manifest["validation_problems"] == []


def test_same_seed_gives_identical_files_and_a_new_seed_changes_them():
    _, a = built(42)
    out_b = Path(tempfile.mkdtemp(prefix="fp_synth_"))
    b = build(N, 42, out_b, quiet=True)
    assert a["files"] == b["files"]
    _, c = built(7)
    assert a["files"]["raw/customer_360.csv"] != c["files"]["raw/customer_360.csv"]


def test_minimum_account_count_is_enforced():
    with pytest.raises(ValueError):
        build(500, 1, tempfile.mkdtemp(prefix="fp_synth_"), quiet=True)


def test_row_counts_and_flag_rate():
    _, m = built()
    assert m["rows"]["customer_360"] == N
    assert m["rows"]["model_scores"] == 2 * N
    expected = N * K.FLAG_RATE
    assert abs(m["flagged"]["churn"] - expected) <= 2
    assert abs(m["flagged"]["acquisition"] - expected) <= 2


def test_every_schema_table_exists_with_the_declared_columns():
    out, _ = built()
    con = sqlite3.connect(out / "lakehouse.db")
    for table, cols in SCHEMA.items():
        got = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
        assert got == [c for c, _ in cols], table
    con.close()


def test_bills_follow_account_tenure():
    out, _ = built()
    con = sqlite3.connect(out / "lakehouse.db")
    rows = con.execute(
        "SELECT c.account_id, c.customer_since, COUNT(b.bill_id) FROM customer_360 c "
        "LEFT JOIN bills b ON b.account_id = c.account_id GROUP BY c.account_id").fetchall()
    con.close()
    from fiberpulse.synth.truth import expected_bill_months
    over = [r for r in rows if r[2] > expected_bill_months(r[1])]
    assert over == []
    full = sum(1 for r in rows if r[2] == expected_bill_months(r[1]))
    assert full / len(rows) > 0.95   # only deliberate gaps are missing


# --- no leakage ------------------------------------------------------------------------


def test_lakehouse_contains_no_ground_truth_columns():
    out, _ = built()
    con = sqlite3.connect(out / "lakehouse.db")
    forbidden = ("archetype", "true_", "expected_verdict", "signal_kind", "planted", "shift", "difficulty")
    for table in SCHEMA:
        for col in (r[1] for r in con.execute(f"PRAGMA table_info({table})")):
            assert not any(f in col for f in forbidden), f"{table}.{col}"
    con.close()


def test_knowledge_base_is_separate_from_current_accounts():
    out, _ = built()
    lines = (out / "knowledge" / "outcomes_kb.jsonl").read_text(encoding="utf-8").splitlines()
    kb = [json.loads(x) for x in lines]
    assert len(kb) == 360
    assert len({c["case_id"] for c in kb}) == 360
    assert not any("ACC-" in c["text"] for c in kb)
    assert {c["signal_type"] for c in kb} == {"churn", "acquisition"}
    real = sum(1 for c in kb if c["signal_validity"] == "real")
    # 160 confirmed patterns are always "real"; ~10% of the 200 paradox patterns carry a contrary outcome
    assert 0.46 < real / len(kb) < 0.60
    assert real > 160   # i.e. the deliberate noise is actually present


# --- planted eval set -------------------------------------------------------------------


def test_eval_set_matches_the_plan():
    out, _ = built()
    ev = json.loads((out / "eval" / "eval_set.json").read_text(encoding="utf-8"))
    assert len(ev) == 50
    assert sum(e["expected_verdict"] == "confirmed" for e in ev) == 25
    assert sum(e["expected_verdict"] == "paradox" for e in ev) == 25
    assert sum(e["signal_type"] == "churn" for e in ev) == 30
    for archetype, signal, verdict, difficulty, count in PLAN:
        got = [e for e in ev if e["archetype"] == archetype]
        assert len(got) == count, archetype
        assert all(e["signal_type"] == signal and e["expected_verdict"] == verdict for e in got)
    assert len({e["account_id"] for e in ev}) == 50   # one case per account
    assert sum(e["difficulty"] == "hard" for e in ev) >= 15


def test_confirmed_and_paradox_are_separated_in_true_probability():
    out, _ = built()
    ev = json.loads((out / "eval" / "eval_set.json").read_text(encoding="utf-8"))
    conf = [e["true_probability_3m"] for e in ev if e["expected_verdict"] == "confirmed"]
    para = [e["true_probability_3m"] for e in ev if e["expected_verdict"] == "paradox"]
    assert min(conf) > max(para)
    assert sum(conf) / len(conf) > 0.5
    assert sum(para) / len(para) < 0.15


def test_planted_accounts_are_flagged_with_high_scores():
    out, _ = built()
    con = sqlite3.connect(out / "lakehouse.db")
    ev = json.loads((out / "eval" / "eval_set.json").read_text(encoding="utf-8"))
    for e in ev:
        row = con.execute("SELECT score, flagged FROM model_scores WHERE account_id=? AND signal_type=?",
                          (e["account_id"], e["signal_type"])).fetchone()
        assert row[1] == 1 and row[0] >= 0.70 and abs(row[0] - e["score"]) < 1e-9
    con.close()


def test_context_truth_matches_independent_sql():
    out, _ = built()
    truth = json.loads((out / "eval" / "context_truth.json").read_text(encoding="utf-8"))
    assert len(truth) >= 150
    con = sqlite3.connect(out / "lakehouse.db")
    for acc, expected in list(truth.items())[:60]:
        got = sql_context(con, acc)
        for field, value in expected.items():
            if isinstance(value, float):
                assert abs(value - got[field]) < 1e-9, (acc, field)
            else:
                assert value == got[field], (acc, field)
    con.close()
    gaps = {g for v in truth.values() for g in v["data_gaps"]}
    assert gaps == {"bills_incomplete", "external_enrichment_none"}


# --- catalog and policy oracle ------------------------------------------------------------


def test_margin_catalog_has_designed_edge_cases():
    below = {o.offer_id for o in OFFERS if not meets_margin_floor(o.gross_margin)}
    assert below == {"RET-04", "UP-04", "UP-08"}
    assert OFFER_BY_ID["UP-10"].gross_margin == pytest.approx(MARGIN_FLOOR)
    assert meets_margin_floor(OFFER_BY_ID["UP-10"].gross_margin)      # exactly on the floor passes
    assert meets_margin_floor(OFFER_BY_ID["UP-09"].gross_margin)      # just above passes
    assert OFFER_BY_ID["RET-05"].discount_pct > 25                    # fails discount cap, not margin


def test_clean_account_passes_every_catalog_offer_that_is_eligible():
    facts = clean_facts(segment="Medium", tenure_months=60)
    for o in OFFERS:
        v = evaluate_offer(o, facts)
        assert set(v) <= {"POL-MARGIN-01", "POL-DISC-02"}, (o.offer_id, v)


def test_each_policy_rule_fires_on_its_own_trigger():
    ret = OFFER_BY_ID["RET-03"]      # price lock, passes everything for a clean account
    up = OFFER_BY_ID["UP-07"]        # static IP, cheap and short
    assert evaluate_offer(ret, clean_facts()) == []
    assert evaluate_offer(up, clean_facts()) == []
    assert evaluate_offer(OFFER_BY_ID["UP-04"], clean_facts()) == ["POL-MARGIN-01"]
    assert evaluate_offer(OFFER_BY_ID["RET-05"], clean_facts(segment="Medium")) == ["POL-DISC-02"]
    assert evaluate_offer(OFFER_BY_ID["RET-01"], clean_facts(segment="Micro")) == ["POL-DISC-02"]
    assert evaluate_offer(ret, clean_facts(days_since_last_retention_offer=30)) == ["POL-FREQ-03"]
    assert evaluate_offer(ret, clean_facts(days_since_last_retention_offer=90)) == []     # gap is exactly 90
    assert evaluate_offer(up, clean_facts(days_since_last_upsell_offer=10)) == ["POL-FREQ-03"]
    assert evaluate_offer(ret, clean_facts(days_since_last_upsell_offer=1)) == []         # types do not mix
    assert evaluate_offer(up, clean_facts(open_high_ticket=True)) == ["POL-CARE-04"]
    assert evaluate_offer(ret, clean_facts(open_high_ticket=True)) == []                  # retention still allowed
    assert evaluate_offer(OFFER_BY_ID["UP-01"], clean_facts(late_or_unpaid_last3=2)) == ["POL-CREDIT-05"]
    assert evaluate_offer(up, clean_facts(late_or_unpaid_last3=3)) == []                  # small and short is fine
    assert evaluate_offer(ret, clean_facts(marketing_opt_in=False)) == ["POL-CONSENT-06"]
    assert evaluate_offer(OFFER_BY_ID["UP-05"], clean_facts(segment="Small")) == ["POL-ELIG-07"]
    assert evaluate_offer(OFFER_BY_ID["RET-03"], clean_facts(tenure_months=5)) == ["POL-ELIG-07"]
    assert evaluate_offer(up, clean_facts(active_addons=frozenset({"Static IP"}))) == ["POL-DUP-08"]
    assert evaluate_offer(OFFER_BY_ID["UP-10"], clean_facts()) == []                      # boundary margin passes


def test_violations_are_reported_in_rule_order():
    v = evaluate_offer(OFFER_BY_ID["UP-04"], clean_facts(marketing_opt_in=False, open_high_ticket=True))
    assert v == ["POL-MARGIN-01", "POL-CARE-04", "POL-CONSENT-06"]
    assert v == sorted(v, key=RULE_IDS.index)


def test_handbook_documents_every_rule_and_the_real_numbers():
    text = render_handbook()
    for rule in RULE_IDS:
        assert f"## {rule}" in text
    assert "60%" in text and "Micro 15%" in text and "Medium 25%" in text


def test_compliance_cases_are_consistent_and_complete():
    out, _ = built()
    cases = json.loads((out / "eval" / "compliance_cases.json").read_text(encoding="utf-8"))
    assert len(cases) >= 38
    assert len({c["case_id"] for c in cases}) == len(cases)
    assert {r for c in cases for r in c["expected_violations"]} == set(RULE_IDS)
    assert sum(c["expected_pass"] for c in cases) >= 8
    assert any(len(c["expected_violations"]) >= 2 for c in cases)
    assert any(c["offer_id"] == "UP-10" and c["expected_pass"] for c in cases)
    assert all(c["expected_pass"] == (not c["expected_violations"]) for c in cases)


def test_data_dictionary_mentions_every_table_and_column():
    doc = (Path(__file__).resolve().parents[1] / "docs" / "DATA_DICTIONARY.md").read_text(encoding="utf-8")
    for table, cols in SCHEMA.items():
        assert table in doc, table
        for col, _ in cols:
            assert col in doc, f"{table}.{col}"
