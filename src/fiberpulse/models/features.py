"""Point-in-time features: what a model could know on the snapshot date (2026-08-31).

Rule (ADR-010, ADR-015): a feature may only use information dated on or before the snapshot. The lakehouse
holds data up to the scoring date (2026-10-01), so every table is filtered by its own date columns, and the
few customer_360 fields that later events overwrite (plan, renewal) are rewound to their snapshot state.
`tests/test_models.py` proves this by deleting or hiding everything after the snapshot and checking the
features do not change.
"""

from __future__ import annotations

import sqlite3
from datetime import date

import numpy as np
import pandas as pd

from fiberpulse.synth import constants as K

SNAP = K.MODEL_SNAPSHOT
SNAP_S = SNAP.isoformat()
WIN90_S = K.PRE90_START.isoformat()
HIST_S = K.HISTORY_START.isoformat()
LAST_BILL_MONTH = "2026-08"          # September's bill is after the snapshot
EXIT_WORDS = ("terminat", "cancel", "exit", "comparing", "switch")
PLAN_IDX = {p[0]: i for i, p in enumerate(K.PLANS)}
SEGMENT_ORD = {s: i for i, s in enumerate(K.SEGMENTS)}
SAMPLE_SEED = 20260831
CATEGORICAL = ("industry", "emirate", "payment_method")


def _read(con: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, con)


def _months_between(a: pd.Series, b: date) -> pd.Series:
    d = pd.to_datetime(a)
    return (b.year - d.dt.year) * 12 + (b.month - d.dt.month) - (b.day < d.dt.day).astype(int)


def _account_features(con) -> pd.DataFrame:
    c = _read(con, "SELECT * FROM customer_360").set_index("account_id")
    out = pd.DataFrame(index=c.index)
    out["segment_ord"] = c["segment"].map(SEGMENT_ORD)
    out["employees"] = c["employees"]
    out["tenure_months"] = _months_between(c["customer_since"], SNAP).clip(lower=0)
    out["utilization_pct"] = c["avg_utilization_pct"]
    out["marketing_opt_in"] = c["marketing_opt_in"]

    # plan at the snapshot: undo a plan change that happened after it
    idx = c["plan_name"].map(PLAN_IDX).astype(int)
    changed_after = c["last_plan_change_date"].notna() & (c["last_plan_change_date"] > SNAP_S)
    undo = np.where(c["last_plan_change_type"] == "upgrade", -1, 1)
    idx = idx + np.where(changed_after, undo, 0)
    idx = idx.clip(0, len(K.PLANS) - 1)
    out["plan_speed_mbps"] = [K.PLANS[i][1] for i in idx]
    changed_before = c["last_plan_change_date"].notna() & (c["last_plan_change_date"] <= SNAP_S)
    out["plan_upgraded_recently"] = (changed_before & (c["last_plan_change_type"] == "upgrade")).astype(int)
    out["plan_downgraded_recently"] = (changed_before & (c["last_plan_change_type"] == "downgrade")).astype(int)

    # contract at the snapshot. A renewal after it overwrote the old term and we cannot know what the old
    # contract looked like, and any rule derived from the renewal date would leak the future. So those
    # accounts get (term, months remaining) sampled from accounts that did not renew after the snapshot.
    term = c["contract_term_months"].to_numpy()
    rem = np.where(term > 0, np.minimum(c["contract_months_remaining"].to_numpy() + 1, term), 0)
    renewed_after = (c["last_renewal_date"].notna() & (c["last_renewal_date"] > SNAP_S)).to_numpy()
    if renewed_after.any():
        pool = np.flatnonzero(~renewed_after)
        pick = np.random.default_rng(SAMPLE_SEED).choice(pool, size=int(renewed_after.sum()), replace=True)
        term = term.copy()
        rem = rem.copy()
        term[renewed_after] = term[pick]
        rem[renewed_after] = rem[pick]
    out["contract_term_months"] = term
    out["contract_months_remaining"] = rem
    renewed_before = c["last_renewal_date"].notna() & (c["last_renewal_date"] <= SNAP_S)
    out["renewed_last_6m"] = (renewed_before &
                              (_months_between(c["last_renewal_date"].where(renewed_before, "2000-01-01"), SNAP) < 6)
                              ).astype(int)
    for col in CATEGORICAL:
        dummies = pd.get_dummies(c[col], prefix=col, dtype=int)
        out = out.join(dummies)
    return out


def _ticket_features(con, ids) -> pd.DataFrame:
    t = _read(con, f"SELECT * FROM sr_complaint WHERE opened_date <= '{SNAP_S}'")
    # status as of the snapshot, not as of today
    t["open_at_snap"] = (t["closed_date"].isna() | (t["closed_date"] > SNAP_S)).astype(int)
    closed = t["closed_date"].notna() & (t["closed_date"] <= SNAP_S)
    t["csat_seen"] = t["csat"].where(closed)
    t["res_hours_seen"] = t["resolution_hours"].where(closed)
    t["in90"] = (t["opened_date"] >= WIN90_S).astype(int)
    t["sev_high"] = (t["severity"] == "High").astype(int)
    t["complaint90"] = t["in90"] * t["is_complaint"]
    t["high_open"] = t["open_at_snap"] * t["sev_high"]
    t["open_complaint"] = t["open_at_snap"] * t["is_complaint"]
    for cat, name in (("Outage", "outage"), ("SLA breach", "sla"), ("Billing dispute", "billing"),
                      ("Slow speed", "slow"), ("Upgrade request", "upgrade_req")):
        t[f"{name}90"] = t["in90"] * (t["category"] == cat)
    g = t.groupby("account_id")
    f = pd.DataFrame({
        "tickets_12m": g.size(), "tickets_90d": g["in90"].sum(), "complaints_90d": g["complaint90"].sum(),
        "open_tickets": g["open_at_snap"].sum(), "open_complaints": g["open_complaint"].sum(),
        "open_high_tickets": g["high_open"].sum(), "csat_mean": g["csat_seen"].mean(),
        "res_hours_mean": g["res_hours_seen"].mean(), "last_ticket": g["opened_date"].max(),
        **{k: g[k].sum() for k in ("outage90", "sla90", "billing90", "slow90", "upgrade_req90")},
    })
    f["days_since_ticket"] = (pd.Timestamp(SNAP) - pd.to_datetime(f.pop("last_ticket"))).dt.days
    return f.reindex(ids)


def _call_features(con, ids) -> pd.DataFrame:
    k = _read(con, f"SELECT * FROM call_intent_signals WHERE call_date <= '{SNAP_S}' AND call_date >= '{WIN90_S}'")
    k["neg"] = (k["sentiment"] == "negative").astype(int)
    k["pos"] = (k["sentiment"] == "positive").astype(int)
    k["exit_words"] = k["summary"].str.lower().apply(lambda s: int(any(w in s for w in EXIT_WORDS)))
    for intent in ("complaint", "enquiry", "acquisition"):
        k[f"i_{intent}"] = (k["intent"] == intent).astype(int)
    g = k.groupby("account_id")
    f = pd.DataFrame({
        "calls_90d": g.size(), "calls_neg": g["neg"].sum(), "calls_pos": g["pos"].sum(),
        "call_sentiment_mean": g["sentiment_score"].mean(), "calls_exit_words": g["exit_words"].sum(),
        "calls_complaint": g["i_complaint"].sum(), "calls_enquiry": g["i_enquiry"].sum(),
        "calls_acquisition": g["i_acquisition"].sum(),
    })
    return f.reindex(ids)


def _external_features(con, ids) -> pd.DataFrame:
    e = _read(con, f"SELECT * FROM external_enrichment WHERE signal_date <= '{SNAP_S}' "
                   f"AND signal_date >= '{WIN90_S}'")
    for p in ("negative", "positive", "neutral"):
        e[p] = (e["polarity"] == p).astype(int)
    g = e.groupby("account_id")
    f = pd.DataFrame({"ext_negative": g["negative"].sum(), "ext_positive": g["positive"].sum(),
                      "ext_total": g.size()})
    return f.reindex(ids)


def _addon_features(con, ids) -> pd.DataFrame:
    a = _read(con, f"SELECT * FROM addons WHERE start_date <= '{SNAP_S}' "
                   f"AND (end_date IS NULL OR end_date > '{SNAP_S}')")
    g = a.groupby("account_id")
    f = pd.DataFrame({"addons_active": g.size(), "addon_spend_aed": g["monthly_price_aed"].sum(),
                      "addons_upsell": g["is_upsell"].sum()})
    return f.reindex(ids)


def _bill_features(con, ids) -> pd.DataFrame:
    b = _read(con, f"SELECT * FROM bills WHERE bill_month <= '{LAST_BILL_MONTH}' ORDER BY bill_month")
    b["late"] = (b["days_late"] > K.LATE_DAYS).astype(int)
    g = b.groupby("account_id")
    first = g["amount_aed"].first()
    last = g["amount_aed"].last()
    f = pd.DataFrame({
        "bills_n": g.size(), "bill_avg_aed": g["amount_aed"].mean(), "bill_last_aed": last,
        "bill_trend": last / first.replace(0, np.nan), "late_n": g["late"].sum(),
        "days_late_mean": g["days_late"].mean(), "disputed_n": g["disputed"].sum(),
        "unpaid_n": g["paid"].apply(lambda s: int((s == 0).sum())),
    })
    last3 = b.groupby("account_id").tail(3).groupby("account_id")
    f["late_last3"] = last3["late"].sum()
    return f.reindex(ids)


def _offer_features(con, ids) -> pd.DataFrame:
    o = _read(con, f"SELECT * FROM offer_history WHERE offer_date <= '{SNAP_S}'")
    o["is_ret"] = o["offer_id"].str.startswith("RET").astype(int)
    o["days_ago"] = (pd.Timestamp(SNAP) - pd.to_datetime(o["offer_date"])).dt.days
    ret = o[o["is_ret"] == 1].groupby("account_id")
    up = o[o["is_ret"] == 0].groupby("account_id")
    f = pd.DataFrame(index=ids)
    f["days_since_retention_offer"] = ret["days_ago"].min().reindex(ids)
    f["days_since_upsell_offer"] = up["days_ago"].min().reindex(ids)
    f["offers_12m"] = o.groupby("account_id").size().reindex(ids)
    f["offers_accepted_12m"] = o.groupby("account_id")["accepted"].sum().reindex(ids)
    return f


FILL_ZERO = ("tickets_12m", "tickets_90d", "complaints_90d", "open_tickets", "open_complaints",
             "open_high_tickets", "outage90", "sla90", "billing90", "slow90", "upgrade_req90",
             "calls_90d", "calls_neg", "calls_pos", "calls_exit_words", "calls_complaint", "calls_enquiry",
             "calls_acquisition", "ext_negative", "ext_positive", "ext_total", "addons_active",
             "addon_spend_aed", "addons_upsell", "offers_12m", "offers_accepted_12m")


def build_features(con: sqlite3.Connection) -> pd.DataFrame:
    """One row per account (index account_id). Missing history is NaN where 'unknown' differs from 'none'."""
    base = _account_features(con)
    ids = base.index
    parts = [base, _ticket_features(con, ids), _call_features(con, ids), _external_features(con, ids),
             _addon_features(con, ids), _bill_features(con, ids), _offer_features(con, ids)]
    f = pd.concat(parts, axis=1)
    f[list(FILL_ZERO)] = f[list(FILL_ZERO)].fillna(0)
    f["days_since_ticket"] = f["days_since_ticket"].fillna(400)
    f["days_since_retention_offer"] = f["days_since_retention_offer"].fillna(999)
    f["days_since_upsell_offer"] = f["days_since_upsell_offer"].fillna(999)
    return f.astype(float).sort_index()
