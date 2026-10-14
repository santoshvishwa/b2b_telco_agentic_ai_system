"""Assemble the final tables, write the SQLite 'lakehouse' stand-in, and the eval-only ground truth."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import constants as K
from .catalog import OFFERS
from .events import EventBook
from .population import Population, iso, sigmoid

# --- Schema (column order = table order) ---------------------------------------------

SCHEMA: dict[str, list[tuple[str, str]]] = {
    "customer_360": [
        ("account_id", "TEXT PRIMARY KEY"), ("company_name", "TEXT"), ("industry", "TEXT"),
        ("emirate", "TEXT"), ("segment", "TEXT"), ("employees", "INTEGER"), ("plan_name", "TEXT"),
        ("plan_speed_mbps", "INTEGER"), ("mrc_aed", "INTEGER"), ("tenure_months", "INTEGER"),
        ("customer_since", "TEXT"), ("contract_term_months", "INTEGER"),
        ("contract_months_remaining", "INTEGER"), ("last_renewal_date", "TEXT"),
        ("payment_method", "TEXT"), ("marketing_opt_in", "INTEGER"),
        ("avg_utilization_pct", "REAL"), ("addon_count", "INTEGER"),
        ("last_plan_change_date", "TEXT"), ("last_plan_change_type", "TEXT"),
    ],
    "sr_complaint": [
        ("ticket_id", "TEXT PRIMARY KEY"), ("account_id", "TEXT"), ("opened_date", "TEXT"),
        ("closed_date", "TEXT"), ("category", "TEXT"), ("is_complaint", "INTEGER"),
        ("severity", "TEXT"), ("status", "TEXT"), ("csat", "INTEGER"), ("resolution_hours", "INTEGER"),
    ],
    "addons": [
        ("addon_row_id", "TEXT PRIMARY KEY"), ("account_id", "TEXT"), ("addon_name", "TEXT"),
        ("monthly_price_aed", "INTEGER"), ("start_date", "TEXT"), ("end_date", "TEXT"),
        ("is_upsell", "INTEGER"),
    ],
    "bills": [
        ("bill_id", "TEXT PRIMARY KEY"), ("account_id", "TEXT"), ("bill_month", "TEXT"),
        ("amount_aed", "REAL"), ("paid", "INTEGER"), ("days_late", "INTEGER"), ("disputed", "INTEGER"),
    ],
    "call_intent_signals": [
        ("call_id", "TEXT PRIMARY KEY"), ("account_id", "TEXT"), ("call_date", "TEXT"),
        ("direction", "TEXT"), ("duration_sec", "INTEGER"), ("intent", "TEXT"), ("sentiment", "TEXT"),
        ("sentiment_score", "REAL"), ("summary", "TEXT"),
    ],
    "external_enrichment": [
        ("signal_id", "TEXT PRIMARY KEY"), ("account_id", "TEXT"), ("signal_date", "TEXT"),
        ("source", "TEXT"), ("signal_text", "TEXT"), ("polarity", "TEXT"),
    ],
    "offer_history": [
        ("history_id", "TEXT PRIMARY KEY"), ("account_id", "TEXT"), ("offer_id", "TEXT"),
        ("offer_date", "TEXT"), ("accepted", "INTEGER"), ("channel", "TEXT"),
    ],
    "model_scores": [
        ("account_id", "TEXT"), ("scored_month", "TEXT"), ("signal_type", "TEXT"), ("score", "REAL"),
        ("model_version", "TEXT"), ("score_rank", "INTEGER"), ("flagged", "INTEGER"),
    ],
    "offer_catalog": [
        ("offer_id", "TEXT PRIMARY KEY"), ("name", "TEXT"), ("offer_type", "TEXT"),
        ("monthly_price_aed", "REAL"), ("monthly_cost_aed", "REAL"), ("discount_pct", "REAL"),
        ("term_months", "INTEGER"), ("eligible_segments", "TEXT"), ("min_tenure_months", "INTEGER"),
        ("addon_name", "TEXT"), ("description", "TEXT"),
    ],
}
INDEXES = {
    "sr_complaint": ["account_id"], "addons": ["account_id"], "bills": ["account_id"],
    "call_intent_signals": ["account_id"], "external_enrichment": ["account_id"],
    "offer_history": ["account_id"], "model_scores": ["account_id", "signal_type"],
}


def _ids(prefix: str, n: int, width: int) -> list[str]:
    return [f"{prefix}-{k + 1:0{width}d}" for k in range(n)]


def _ord(series) -> np.ndarray:
    return np.array([date.fromisoformat(s).toordinal() if isinstance(s, str) else 0 for s in series])


# --- Bills ---------------------------------------------------------------------------


def build_bills(pop: Population, book: EventBook, addons: pd.DataFrame, protect: np.ndarray) -> pd.DataFrame:
    rng, n = book.rng, pop.n
    cust = pop.cust
    mrc = cust["mrc_aed"].to_numpy(float)
    prev = cust["prev_mrc_aed"].to_numpy(float)
    chg = _ord(cust["last_plan_change_date"])
    a_i = addons["i"].to_numpy()
    a_start = _ord(addons["start_date"])
    a_end = np.where(addons["end_date"].isna().to_numpy(), 10**9, _ord(addons["end_date"]))
    a_price = addons["monthly_price_aed"].to_numpy(float)
    since = _ord(cust["customer_since"])

    frames = []
    for m, ym in enumerate(K.BILL_MONTHS):
        y, mo = int(ym[:4]), int(ym[5:])
        ms = date(y, mo, 1).toordinal()
        me = (date(y + (mo == 12), mo % 12 + 1, 1)).toordinal()
        days = me - ms
        # plan charge, prorated when a plan change falls inside the month
        plan = np.where(chg == 0, mrc, np.where(chg >= me, prev, np.where(chg < ms, mrc,
                        prev * (chg - ms) / days + mrc * (1 - (chg - ms) / days))))
        overlap = np.clip(np.minimum(a_end, me) - np.maximum(a_start, ms), 0, days) / days
        addon_amt = np.bincount(a_i, weights=a_price * overlap, minlength=n)
        amount = np.round((plan + addon_amt) * rng.normal(1.0, 0.02, size=n), 2)

        if m < 5:
            late = pop.late_flags[:, m]
            days_late = np.where(late, rng.integers(6, 41, size=n), rng.integers(0, 4, size=n))
            paid = np.ones(n, dtype=int)
            disputed = (rng.random(n) < 0.012).astype(int)
        else:
            s = book.sept
            late = (s == "late")
            unpaid = (s == "unpaid")
            days_late = np.where(late, rng.integers(8, 21, size=n),
                                 np.where(unpaid, K.LATE_DAYS + 1, rng.integers(0, 4, size=n)))
            paid = np.where(unpaid, 0, 1)
            disputed = (s == "disputed").astype(int)
        active = since < me   # the account existed for at least part of this month
        frames.append(pd.DataFrame({
            "i": np.arange(n), "bill_month": ym, "amount_aed": amount, "paid": paid,
            "days_late": days_late.astype(int), "disputed": disputed,
        })[active])
    bills = pd.concat(frames, ignore_index=True)

    # data gaps: ~1.5% of non-planted accounts lose one random monthly bill
    gap_acc = np.flatnonzero((rng.random(n) < 0.015) & ~protect)
    gap_month = rng.integers(0, 6, size=len(gap_acc))
    drop = set(zip(gap_acc.tolist(), [K.BILL_MONTHS[k] for k in gap_month]))
    keep = [(i, bm) not in drop for i, bm in zip(bills["i"].tolist(), bills["bill_month"].tolist())]
    return bills[np.array(keep)]


# --- Assembly ------------------------------------------------------------------------


def assemble(pop: Population, book: EventBook, scores: pd.DataFrame, masks: dict,
             planted: np.ndarray, rng: np.random.Generator) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    n = pop.n
    acct = pop.cust["account_id"].to_numpy()

    def with_acct(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.insert(0, "account_id", acct[df["i"].to_numpy()])
        return df.drop(columns=["i"])

    # tickets
    tk = pd.concat([pop.tickets, pd.DataFrame(book.tickets, columns=pop.tickets.columns)], ignore_index=True)
    tk = tk.sort_values(["opened_date", "i", "category"], kind="stable").reset_index(drop=True)
    tk["resolution_hours"] = tk["resolution_hours"].where(tk["resolution_hours"] >= 0, other=pd.NA)
    tk["csat"] = pd.array(tk["csat"].tolist(), dtype="Int64")
    tk["resolution_hours"] = pd.array(tk["resolution_hours"].tolist(), dtype="Int64")
    tk = with_acct(tk)
    tk.insert(0, "ticket_id", _ids("TKT", len(tk), 6))

    # add-ons
    ad_raw = pd.concat([pop.addons, pd.DataFrame(book.addons, columns=pop.addons.columns)], ignore_index=True)
    ad_raw = ad_raw.sort_values(["start_date", "i", "addon_name"], kind="stable").reset_index(drop=True)
    ad = with_acct(ad_raw)
    ad.insert(0, "addon_row_id", _ids("ADD", len(ad), 6))

    # bills (uses the raw add-on frame, which still carries the account index)
    bills = build_bills(pop, book, ad_raw, protect=planted)
    bills = bills.sort_values(["bill_month", "i"], kind="stable").reset_index(drop=True)
    bills = with_acct(bills)
    bills.insert(0, "bill_id", _ids("BIL", len(bills), 7))

    # calls
    calls = pd.DataFrame(book.calls)
    calls = calls.sort_values(["call_date", "i", "intent"], kind="stable").reset_index(drop=True)
    calls = with_acct(calls)
    calls.insert(0, "call_id", _ids("CALL", len(calls), 6))

    # external enrichment (signal_kind is generator-internal and not exposed)
    ext = pd.DataFrame(book.external).drop(columns=["signal_kind"])
    ext = ext.sort_values(["signal_date", "i", "source"], kind="stable").reset_index(drop=True)
    ext = with_acct(ext)
    ext.insert(0, "signal_id", _ids("EXT", len(ext), 6))

    # offer history
    oh = pop.offer_history.sort_values(["offer_date", "i"], kind="stable").reset_index(drop=True)
    oh = with_acct(oh)
    oh.insert(0, "history_id", _ids("OFH", len(oh), 6))

    # customer_360 (active add-ons at the scoring date)
    active = ad_raw[ad_raw["end_date"].isna() | (ad_raw["end_date"] > K.AS_OF.isoformat())]
    cust = pop.cust.copy()
    cust["addon_count"] = np.bincount(active["i"].to_numpy(), minlength=n)
    cust_cols = [c for c, _ in SCHEMA["customer_360"]]
    cust = cust[cust_cols]

    catalog = pd.DataFrame([{
        "offer_id": o.offer_id, "name": o.name, "offer_type": o.offer_type,
        "monthly_price_aed": float(o.monthly_price_aed), "monthly_cost_aed": float(o.monthly_cost_aed),
        "discount_pct": float(o.discount_pct), "term_months": o.term_months,
        "eligible_segments": "|".join(o.eligible_segments), "min_tenure_months": o.min_tenure_months,
        "addon_name": o.addon_name, "description": o.description,
    } for o in OFFERS])

    tables = {
        "customer_360": cust, "sr_complaint": tk, "addons": ad, "bills": bills,
        "call_intent_signals": calls, "external_enrichment": ext, "offer_history": oh,
        "model_scores": scores, "offer_catalog": catalog,
    }
    tables = {name: df[[c for c, _ in SCHEMA[name]]] for name, df in tables.items()}

    # --- eval-only ground truth ---
    p_c = sigmoid(pop.z_true_c + book.shift_c)
    p_a = sigmoid(pop.z_true_a + book.shift_a)
    meta = [book.planted.get(i) for i in range(n)]
    gt = pd.DataFrame({
        "account_id": acct,
        "archetype": [m["archetype"] if m else "unplanted" for m in meta],
        "planted_signal_type": [m["signal_type"] if m else None for m in meta],
        "expected_verdict": [m["expected_verdict"] if m else None for m in meta],
        "difficulty": [m["difficulty"] if m else None for m in meta],
        "planted_signals": ["|".join(m["planted_signals"]) if m else None for m in meta],
        "flagged_churn": masks["flagged_c"].astype(int),
        "flagged_acquisition": masks["flagged_a"].astype(int),
        "score_churn": np.round(masks["score_c"], 4),
        "score_acquisition": np.round(masks["score_a"], 4),
        "true_churn_prob_3m": np.round(p_c, 4),
        "true_acquisition_prob_3m": np.round(p_a, 4),
        "true_churned_3m": (rng.random(n) < p_c).astype(int),
        "true_acquired_3m": (rng.random(n) < p_a).astype(int),
        "churn_shift_from_fresh_events": np.round(book.shift_c, 3),
        "acquisition_shift_from_fresh_events": np.round(book.shift_a, 3),
    })
    return tables, gt


# --- SQLite ----------------------------------------------------------------------------


def _py(v):
    return None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)


def write_sqlite(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    path = Path(path)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    try:
        for name, cols in SCHEMA.items():
            ddl = ", ".join(f"{c} {t}" for c, t in cols)
            if name == "model_scores":
                ddl += ", PRIMARY KEY (account_id, scored_month, signal_type)"
            con.execute(f"CREATE TABLE {name} ({ddl})")
            df = tables[name]
            cols_only = [c for c, _ in cols]
            rows = [tuple(_py(v) for v in row) for row in df[cols_only].itertuples(index=False, name=None)]
            con.executemany(f"INSERT INTO {name} VALUES ({','.join('?' * len(cols_only))})", rows)
            for col in INDEXES.get(name, []):
                con.execute(f"CREATE INDEX idx_{name}_{col} ON {name}({col})")
        con.commit()
    finally:
        con.close()
