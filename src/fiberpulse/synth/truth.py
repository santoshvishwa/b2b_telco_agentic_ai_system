"""Reference ("truth") computations over the final tables, written in pandas.

validate.py recomputes the same quantities with SQL and compares, so the metric definitions in
docs/DATA_DICTIONARY.md are checked two independent ways before any agent depends on them.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import constants as K
from .catalog import OFFER_BY_ID, AccountFacts

AS_OF_ISO = K.AS_OF.isoformat()


def _iso_days_before(days: int) -> str:
    return (K.AS_OF - timedelta(days=days)).isoformat()


def _days_since(iso_date: str | None):
    if not iso_date:
        return None
    return (K.AS_OF - date.fromisoformat(iso_date)).days


def expected_bill_months(customer_since: str) -> int:
    """Bill months (Apr-Sep) in which the account existed for at least part of the month."""
    count = 0
    for ym in K.BILL_MONTHS:
        y, mo = int(ym[:4]), int(ym[5:])
        month_end = date(y + (mo == 12), mo % 12 + 1, 1)
        if date.fromisoformat(customer_since) < month_end:
            count += 1
    return count


def context_truth(tables: dict[str, pd.DataFrame], ids: list[str]) -> dict[str, dict]:
    """Ground-truth AccountContext fields for the given accounts (definitions: DATA_DICTIONARY.md)."""
    idset = set(ids)
    cust = tables["customer_360"].set_index("account_id")
    tk = tables["sr_complaint"]
    tk = tk[tk["account_id"].isin(idset)]
    ad = tables["addons"]
    ad = ad[ad["account_id"].isin(idset)]
    bl = tables["bills"]
    bl = bl[bl["account_id"].isin(idset)]
    cl = tables["call_intent_signals"]
    cl = cl[cl["account_id"].isin(idset)]
    ex = tables["external_enrichment"]
    ex = ex[ex["account_id"].isin(idset)]

    g_tk = {k: v for k, v in tk.groupby("account_id")}
    g_ad = {k: v for k, v in ad.groupby("account_id")}
    g_bl = {k: v for k, v in bl.groupby("account_id")}
    g_cl = {k: v for k, v in cl.groupby("account_id")}
    g_ex = {k: v for k, v in ex.groupby("account_id")}
    empty = pd.DataFrame()

    out: dict[str, dict] = {}
    for acc in ids:
        c = cust.loc[acc]
        t = g_tk.get(acc, empty)
        a = g_ad.get(acc, empty)
        b = g_bl.get(acc, empty)
        k = g_cl.get(acc, empty)
        e = g_ex.get(acc, empty)

        addon_count = 0 if a.empty else int((a["end_date"].isna() | (a["end_date"] > AS_OF_ISO)).sum())

        open_90 = 0
        last_complaint = None
        if not t.empty:
            comp = t[t["is_complaint"] == 1]
            open_90 = int(((comp["status"] == "Open") & (comp["opened_date"] > _iso_days_before(90))).sum())
            if not comp.empty:
                last_complaint = comp["opened_date"].max()

        upsell_dates = []
        if not a.empty:
            upsell_dates += a.loc[a["is_upsell"] == 1, "start_date"].tolist()
        if c["last_plan_change_type"] == "upgrade" and isinstance(c["last_plan_change_date"], str):
            upsell_dates.append(c["last_plan_change_date"])
        last_upsell = max(upsell_dates) if upsell_dates else None

        trend = None
        if len(b) == 6:
            amounts = b.sort_values("bill_month")["amount_aed"].to_numpy(float)
            trend = round((amounts[3:].mean() / amounts[:3].mean() - 1.0) * 100.0, 1)

        call_intent = None
        if not k.empty:
            recent = k[k["call_date"] >= _iso_days_before(K.FRESH_DAYS)]
            if not recent.empty:
                call_intent = recent.sort_values(["call_date", "call_id"]).iloc[-1]["intent"]

        signals: list[str] = []
        if not e.empty:
            recent = e[e["signal_date"] >= _iso_days_before(60)].sort_values(
                ["signal_date", "signal_id"], ascending=False)
            signals = [f"{r.source}: {r.signal_text}" for r in recent.itertuples()]

        gaps = []
        if len(b) < expected_bill_months(c["customer_since"]):
            gaps.append("bills_incomplete")
        if e.empty:
            gaps.append("external_enrichment_none")

        out[acc] = {
            "tenure_months": int(c["tenure_months"]),
            "plan_name": c["plan_name"],
            "mrc_aed": float(c["mrc_aed"]),
            "addon_count": addon_count,
            "open_complaints_90d": open_90,
            "days_since_last_complaint": _days_since(last_complaint),
            "days_since_last_upsell": _days_since(last_upsell),
            "bill_trend_pct_3m": trend,
            "call_intent": call_intent,
            "external_signals": signals,
            "data_gaps": gaps,
        }
    return out


def facts_table(tables: dict[str, pd.DataFrame]) -> dict[str, AccountFacts]:
    """AccountFacts for every account, for the policy oracle."""
    cust = tables["customer_360"]
    tk, bl, oh, ad = tables["sr_complaint"], tables["bills"], tables["offer_history"], tables["addons"]

    open_high = set(tk.loc[(tk["is_complaint"] == 1) & (tk["severity"] == "High")
                           & (tk["status"] == "Open"), "account_id"])
    last3 = bl[bl["bill_month"].isin(K.BILL_MONTHS[-3:]) & (bl["days_late"] > K.LATE_DAYS)]
    late3 = last3.groupby("account_id").size().to_dict()

    oh = oh.assign(offer_type=oh["offer_id"].map(lambda o: OFFER_BY_ID[o].offer_type))
    last_off = oh.groupby(["account_id", "offer_type"])["offer_date"].max().to_dict()

    act = ad[ad["end_date"].isna() | (ad["end_date"] > AS_OF_ISO)]
    held = {k: frozenset(v) for k, v in act.groupby("account_id")["addon_name"].agg(list).items()}

    facts = {}
    for r in cust.itertuples(index=False):
        facts[r.account_id] = AccountFacts(
            segment=r.segment,
            tenure_months=int(r.tenure_months),
            marketing_opt_in=bool(r.marketing_opt_in),
            open_high_ticket=r.account_id in open_high,
            late_or_unpaid_last3=int(late3.get(r.account_id, 0)),
            days_since_last_retention_offer=_days_since(last_off.get((r.account_id, "retention"))),
            days_since_last_upsell_offer=_days_since(last_off.get((r.account_id, "upsell"))),
            active_addons=held.get(r.account_id, frozenset()),
        )
    return facts
