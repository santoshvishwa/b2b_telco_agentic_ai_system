"""Structural population: accounts, pre-snapshot history, hidden drivers, model-view features.

Design: every account has two hidden drivers (u_c churn frailty, u_a acquisition appetite).
The stand-in predictive model ("simulated-v0") only sees OBSERVED features frozen at
MODEL_SNAPSHOT. The true propensities also depend on hidden drivers and, later, on fresh
events the model never saw. That gap is what makes scores go stale.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import constants as K
from .catalog import OFFERS

SEG_ADJ_TICKETS = np.array([0.0, 0.15, 0.35])
ADDON_NAMES = list(K.ADDONS)
ADDON_P = np.array([K.ADDONS[n][1] for n in ADDON_NAMES])
ADDON_P = ADDON_P / ADDON_P.sum()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_intercept(z: np.ndarray, target_mean: float) -> float:
    """Intercept c such that mean(sigmoid(z + c)) == target_mean (bisection)."""
    lo, hi = -20.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if sigmoid(z + mid).mean() > target_mean:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def iso(ordinal: int) -> str:
    return date.fromordinal(int(ordinal)).isoformat()


@dataclass
class Population:
    n: int
    cust: pd.DataFrame           # customer_360 columns plus internal columns (see build_population)
    latent: pd.DataFrame         # hidden drivers and model-view (observed) features
    tickets: pd.DataFrame        # pre-snapshot tickets, keyed by account index `i`
    addons: pd.DataFrame         # add-on rows, keyed by account index `i`
    offer_history: pd.DataFrame  # past offers, keyed by account index `i`
    late_flags: np.ndarray       # n x 5 bool, bill months Apr..Aug
    z_true_c: np.ndarray         # true churn logit before fresh events
    z_true_a: np.ndarray         # true acquisition logit before fresh events
    z_sim_c: np.ndarray          # what the stand-in model "sees" (churn)
    z_sim_a: np.ndarray          # what the stand-in model "sees" (acquisition)


def _tickets(rng, idx, lo_ord, hi_ord, complaint: bool, p_open: float) -> pd.DataFrame:
    """Vectorised ticket rows for account indices `idx` (one row per entry)."""
    m = len(idx)
    cats = K.COMPLAINT_CATEGORIES if complaint else K.REQUEST_CATEGORIES
    cat_p = np.array(K.TICKET_CATEGORY_P[:5] if complaint else K.TICKET_CATEGORY_P[5:])
    cat_p = cat_p / cat_p.sum()
    opened = rng.integers(lo_ord, hi_ord + 1, size=m)
    category = np.array(cats)[rng.choice(len(cats), size=m, p=cat_p)]
    if complaint:
        sev = rng.choice(3, size=m, p=[0.50, 0.35, 0.15])
    else:
        sev = np.zeros(m, dtype=int)
    is_open = rng.random(m) < p_open
    res_days = np.where(sev == 2, rng.integers(1, 4, size=m),
                        np.where(sev == 1, rng.integers(1, 6, size=m), rng.integers(1, 8, size=m)))
    closed_ord = opened + res_days
    csat_raw = rng.choice([1, 2, 3, 4, 5], size=m, p=[0.07, 0.10, 0.20, 0.33, 0.30])
    csat_raw = np.where(res_days > 5, np.minimum(csat_raw, rng.integers(2, 5, size=m)), csat_raw)
    surveyed = rng.random(m) < 0.80
    status = np.where(is_open, "Open", "Closed")
    closed_date = np.array([None if o else iso(c) for o, c in zip(is_open, closed_ord)], dtype=object)
    csat = np.array([None if (o or not s) else int(c) for o, s, c in zip(is_open, surveyed, csat_raw)],
                    dtype=object)
    return pd.DataFrame({
        "i": idx.astype(int),
        "opened_date": [iso(o) for o in opened],
        "closed_date": closed_date,
        "category": category,
        "is_complaint": 1 if complaint else 0,
        "severity": np.array(K.SEVERITIES)[sev],
        "status": status,
        "csat": csat,
        "resolution_hours": np.where(is_open, -1, res_days * 24 + rng.integers(0, 24, size=m)),
    })


def build_population(n: int, rng: np.random.Generator, pyrng: random.Random) -> Population:
    asof = K.AS_OF.toordinal()
    snap = K.MODEL_SNAPSHOT.toordinal()

    # --- identity and static attributes --------------------------------------
    seg_idx = rng.choice(3, size=n, p=K.SEGMENT_WEIGHTS)
    segment = np.array(K.SEGMENTS)[seg_idx]
    employees = np.empty(n, dtype=int)
    for s, name in enumerate(K.SEGMENTS):
        m = seg_idx == s
        lo, hi = K.SEGMENT_EMPLOYEES[name]
        employees[m] = rng.integers(lo, hi + 1, size=int(m.sum()))
    ind_idx = rng.choice(len(K.INDUSTRIES), size=n, p=[x[1] for x in K.INDUSTRIES])
    industry = np.array([x[0] for x in K.INDUSTRIES])[ind_idx]
    industry_risk = np.array([x[2] for x in K.INDUSTRIES])[ind_idx]
    em_idx = rng.choice(len(K.EMIRATES), size=n, p=[x[1] for x in K.EMIRATES])
    emirate = np.array([x[0] for x in K.EMIRATES])[em_idx]
    plan_idx = np.empty(n, dtype=int)
    for s, name in enumerate(K.SEGMENTS):
        m = seg_idx == s
        plan_idx[m] = rng.choice(len(K.PLANS), size=int(m.sum()), p=K.PLAN_WEIGHTS_BY_SEGMENT[name])
    list_price = np.array([p[2] for p in K.PLANS])[plan_idx]
    price_ratio = rng.uniform(0.88, 1.04, size=n)
    mrc = np.round(list_price * price_ratio).astype(int)
    tenure = np.clip((3 + rng.gamma(2.0, 16.0, size=n)).astype(int), 3, 120)
    since_ord = asof - (tenure * 30.44).astype(int)

    names = [f"{pyrng.choice(K.NAME_ADJ)} {pyrng.choice(K.NAME_NOUN)} {pyrng.choice(K.NAME_SUFFIX)}"
             for _ in range(n)]

    term = rng.choice([0, 12, 24], size=n, p=[0.30, 0.40, 0.30])
    # baseline renewals are at least 2 months old: a renewal inside the fresh window may only come
    # from the event book, because fresh events move the true propensities
    elapsed = np.where(term > 0, (rng.random(n) * (term + 7)).astype(int), 0)
    elapsed = np.where(term > 0, np.minimum(np.maximum(elapsed, 2), tenure), 0)
    remaining = np.where(term > 0, np.maximum(term - elapsed, 0), 0)
    last_renewal = [iso(asof - int(e) * 30) if t > 0 else None for t, e in zip(term, elapsed)]

    pay_idx = rng.choice(3, size=n, p=K.PAYMENT_WEIGHTS)
    payment_method = np.array(K.PAYMENT_METHODS)[pay_idx]
    opt_in = rng.random(n) < 0.92

    u_c = rng.standard_normal(n)
    u_a = rng.standard_normal(n)
    util = np.clip(50 + 10 * u_a + 6 * seg_idx + rng.normal(0, 12, size=n), 10, 98).round(1)

    # --- baseline plan changes (before the fresh window) ------------------------
    chg_lo = np.maximum(K.HISTORY_START.toordinal(), since_ord + 30)
    chg_hi = snap - 31
    had_change = (rng.random(n) < 0.12) & (chg_lo < chg_hi)
    change_kind = np.where(rng.random(n) < 0.70, "upgrade", "downgrade")
    change_ord = chg_lo + (rng.random(n) * np.maximum(chg_hi - chg_lo, 1)).astype(int)
    # keep plan/mrc as they are; baseline changes only record history, with a consistent prev_mrc
    prev_mrc = np.where(had_change & (change_kind == "upgrade"), np.round(mrc * 0.85),
                        np.where(had_change, np.round(mrc * 1.15), mrc)).astype(int)
    last_change_date = [iso(o) if h else None for h, o in zip(had_change, change_ord)]
    last_change_type = [k if h else None for h, k in zip(had_change, change_kind)]

    cust = pd.DataFrame({
        "account_id": [f"ACC-{i + 1:06d}" for i in range(n)],
        "company_name": names,
        "industry": industry,
        "emirate": emirate,
        "segment": segment,
        "employees": employees,
        "plan_idx": plan_idx,
        "plan_name": np.array([p[0] for p in K.PLANS])[plan_idx],
        "plan_speed_mbps": np.array([p[1] for p in K.PLANS])[plan_idx],
        "mrc_aed": mrc,
        "prev_mrc_aed": prev_mrc,                  # internal: MRC before the last plan change
        "price_ratio": price_ratio,                # internal: keeps the account's discount on plan changes
        "tenure_months": tenure,
        "customer_since": [iso(o) for o in since_ord],
        "contract_term_months": term,
        "contract_months_remaining": remaining,
        "last_renewal_date": last_renewal,
        "payment_method": payment_method,
        "marketing_opt_in": opt_in.astype(int),
        "avg_utilization_pct": util,
        "last_plan_change_date": last_change_date,
        "last_plan_change_type": last_change_type,
    })

    # --- pre-snapshot tickets ---------------------------------------------------
    seg_adj = SEG_ADJ_TICKETS[seg_idx]
    lam_pre = 0.30 * np.exp(0.55 * u_c + seg_adj)
    k_pre = rng.poisson(lam_pre)
    idx_pre = np.repeat(np.arange(n), k_pre)
    t_pre = _tickets(rng, idx_pre, K.PRE90_START.toordinal(), snap, complaint=True, p_open=0.03)

    lam_old = 0.70 * np.exp(0.45 * u_c + seg_adj)
    idx_old = np.repeat(np.arange(n), rng.poisson(lam_old))
    t_old = _tickets(rng, idx_old, K.HISTORY_START.toordinal(), K.PRE90_START.toordinal() - 1,
                     complaint=True, p_open=0.0)
    idx_req = np.repeat(np.arange(n), rng.poisson(0.55, size=n))
    t_req = _tickets(rng, idx_req, K.HISTORY_START.toordinal(), snap, complaint=False, p_open=0.02)
    since_iso = np.array([iso(o) for o in since_ord])

    def after_since(df, col):
        ok = df[col].to_numpy().astype(str) >= since_iso[df["i"].to_numpy()]
        return df[ok].reset_index(drop=True)

    t_pre, t_old, t_req = after_since(t_pre, "opened_date"), after_since(t_old, "opened_date"), \
        after_since(t_req, "opened_date")
    tickets = pd.concat([t_pre, t_old, t_req], ignore_index=True)

    compl_pre = np.bincount(t_pre["i"].to_numpy(), minlength=n)
    sla_pre = np.bincount(t_pre.loc[t_pre["category"] == "SLA breach", "i"].to_numpy(), minlength=n)

    # --- baseline add-ons (history) ---------------------------------------------
    seg_lam = np.array([0.6, 1.0, 1.8])[seg_idx] * np.exp(0.2 * u_a)
    add_counts = np.minimum(rng.poisson(seg_lam), 6)
    rows = []
    floor_ord = date(2023, 1, 1).toordinal()
    for i in np.flatnonzero(add_counts):
        picks = rng.choice(len(ADDON_NAMES), size=int(add_counts[i]), replace=False, p=ADDON_P)
        for a in picks:
            lo = max(int(since_ord[i]), floor_ord)
            start = int(rng.integers(lo, snap - 14))
            ended = rng.random() < 0.08
            end = min(start + int(rng.integers(60, 400)), snap) if ended else None
            rows.append((int(i), ADDON_NAMES[a], K.ADDONS[ADDON_NAMES[a]][0], iso(start),
                         iso(end) if end else None, int(rng.random() < 0.60)))
    addons = pd.DataFrame(rows, columns=["i", "addon_name", "monthly_price_aed", "start_date",
                                         "end_date", "is_upsell"])
    act = addons[addons["end_date"].isna()]
    addon_pre = np.bincount(act["i"].to_numpy(), minlength=n) if len(act) else np.zeros(n, dtype=int)

    # --- baseline offer history ---------------------------------------------------
    has_offer = rng.random(n) < 0.12
    n_off = np.where(has_offer, 1 + (rng.random(n) < 0.15), 0)
    idx_off = np.repeat(np.arange(n), n_off)
    m_off = len(idx_off)
    off_pick = rng.integers(0, len(OFFERS), size=m_off)
    offer_history = pd.DataFrame({
        "i": idx_off,
        "offer_id": [OFFERS[k].offer_id for k in off_pick],
        "offer_date": [iso(o) for o in rng.integers(K.HISTORY_START.toordinal(), asof, size=m_off)],
        "accepted": (rng.random(m_off) < 0.45).astype(int),
        "channel": np.array(["SMS", "Email", "Push"])[rng.choice(3, size=m_off, p=[0.45, 0.35, 0.20])],
    })
    offer_history = after_since(offer_history, "offer_date")

    # --- late-payment pattern for bill months Apr..Aug (Sept comes from fresh events) ---
    auto = (pay_idx == 0).astype(float)
    cheque = (pay_idx == 2).astype(float)
    p_late = sigmoid(-2.6 + 0.9 * u_c - 0.5 * auto + 0.2 * cheque)
    late_flags = rng.random((n, 5)) < p_late[:, None]
    for m, ym in enumerate(K.BILL_MONTHS[:5]):
        y, mo = int(ym[:4]), int(ym[5:])
        month_end = date(y + (mo == 12), mo % 12 + 1, 1).toordinal()
        late_flags[:, m] &= since_ord < month_end
    late_pre = late_flags[:, 2:].sum(axis=1)   # Jun, Jul, Aug

    # --- hidden drivers -> true logits; observed features -> stand-in model logits ---
    out_of_contract = (remaining == 0).astype(float)
    z_c_obs = (0.45 * np.log1p(compl_pre) + 0.35 * sla_pre + 0.55 * late_pre + 0.60 * out_of_contract
               - 0.012 * tenure + industry_risk - 0.15 * addon_pre)
    z_c_hidden = 0.50 * u_c + rng.normal(0, 0.5, size=n)
    c0 = calibrate_intercept(z_c_obs + z_c_hidden, 0.04)
    z_true_c = z_c_obs + z_c_hidden + c0
    z_sim_c = z_c_obs + c0 + rng.normal(0, 0.25, size=n)

    z_a_obs = (0.030 * (util - 60) + np.array([0.0, 0.2, 0.4])[seg_idx] - 0.35 * addon_pre
               - 0.5 * late_pre - 0.4 * np.log1p(compl_pre) + 0.004 * np.minimum(tenure, 60)
               + 0.3 * (plan_idx <= 1))
    z_a_hidden = 0.55 * u_a + rng.normal(0, 0.5, size=n)
    a0 = calibrate_intercept(z_a_obs + z_a_hidden, 0.06)
    z_true_a = z_a_obs + z_a_hidden + a0
    z_sim_a = z_a_obs + a0 + rng.normal(0, 0.25, size=n)

    latent = pd.DataFrame({
        "u_c": u_c, "u_a": u_a, "seg_idx": seg_idx,
        "complaints_pre_90d": compl_pre, "sla_breaches_pre_90d": sla_pre,
        "late_bills_pre_3m": late_pre, "addon_count_pre": addon_pre,
        "open_pre_tickets": np.bincount(t_pre.loc[t_pre["status"] == "Open", "i"].to_numpy(), minlength=n),
    })

    return Population(n, cust, latent, tickets, addons, offer_history, late_flags,
                      z_true_c, z_true_a, z_sim_c, z_sim_a)
