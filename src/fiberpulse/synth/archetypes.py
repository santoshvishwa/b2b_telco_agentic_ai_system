"""Planted archetypes: the labelled eval accounts (25 paradox + 25 confirmed).

Each recipe writes September events for one flagged account so the story is unambiguous to a
careful reader, then returns a list of signal codes describing what was planted. "Hard" recipes
mix in opposing minor signals so the net direction is still clear but not obvious.

Signal codes (also listed in docs/DATA_DICTIONARY.md):
  open_complaint, open_high_complaint, minor_open_ticket, complaint_resolved_good_csat,
  negative_call, mild_negative_call, positive_call, acquisition_call,
  bill_unpaid, bill_disputed, bill_late, bill_on_time,
  recent_upsell, second_recent_purchase, recent_renewal,
  growth_signal, distress_signal, competitor_signal
"""

from __future__ import annotations

import numpy as np

from .events import EventBook

# (archetype, signal_type, expected_verdict, difficulty, count)
PLAN = (
    ("churn_true_risk_easy", "churn", "confirmed", "easy", 10),
    ("churn_true_risk_hard", "churn", "confirmed", "hard", 5),
    ("churn_stale_recovered_easy", "churn", "paradox", "easy", 6),
    ("churn_stale_renewed_easy", "churn", "paradox", "easy", 4),
    ("churn_stale_hard", "churn", "paradox", "hard", 5),
    ("acq_true_opportunity_easy", "acquisition", "confirmed", "easy", 7),
    ("acq_true_opportunity_hard", "acquisition", "confirmed", "hard", 3),
    ("acq_blocked_complaint_easy", "acquisition", "paradox", "easy", 3),
    ("acq_blocked_financial_easy", "acquisition", "paradox", "easy", 2),
    ("acq_saturated_easy", "acquisition", "paradox", "easy", 2),
    ("acq_blocked_hard", "acquisition", "paradox", "hard", 3),
)
TOTAL_PLANTED = sum(p[4] for p in PLAN)

# Underlying (hidden) propensity an account needs before it can be a "confirmed" eval case.
MIN_P_BEFORE = {"churn": 0.08, "acquisition": 0.10}


class _Ctx:
    """Small helper bundle so recipes read clearly."""

    def __init__(self, b: EventBook):
        self.b = b

    def d(self, lo, hi):          # random whole days ago
        return int(self.b.rng.integers(lo, hi + 1))

    def u(self, lo, hi):          # random float
        return float(self.b.rng.uniform(lo, hi))

    def pick(self, seq):
        return self.b.pyrng.choice(seq)


# --- churn recipes -----------------------------------------------------------------


def churn_true_risk_easy(c: _Ctx, i):
    b = c.b
    for _ in range(c.pick([1, 2])):
        b.ticket(i, c.d(2, 20), c.pick(["Outage", "SLA breach", "Slow speed", "Billing dispute"]),
                 c.pick(["High", "High", "Medium"]))
    b.call(i, c.d(1, 12), "complaint", -c.u(0.6, 0.95))
    status = c.pick(["unpaid", "disputed", "late"])
    b.sept_bill(i, status)
    codes = ["open_complaint", "negative_call", f"bill_{status}"]
    if b.rng.random() < 0.5:
        b.external_signal(i, c.d(3, 25), "competitor")
        codes.append("competitor_signal")
    return codes


def churn_true_risk_hard(c: _Ctx, i):
    b = c.b
    b.ticket(i, c.d(12, 25), c.pick(["Outage", "Slow speed", "Billing dispute"]), "Medium",
             closed_after=c.d(2, 5), csat=4)
    b.ticket(i, c.d(2, 9), c.pick(["Outage", "SLA breach"]), "High")
    b.call(i, c.d(1, 10), "complaint", -c.u(0.28, 0.45))
    b.sept_bill(i, "disputed")
    b.external_signal(i, c.d(3, 25), "competitor")
    return ["complaint_resolved_good_csat", "open_high_complaint", "mild_negative_call", "bill_disputed",
            "competitor_signal"]


def churn_stale_recovered_easy(c: _Ctx, i):
    b = c.b
    b.upsell_event(i, c.d(8, 24))
    b.call(i, c.d(1, 12), c.pick(["enquiry", "acquisition"]), c.u(0.6, 0.9))
    b.ticket(i, c.d(15, 28), c.pick(["Outage", "Slow speed", "SLA breach"]), c.pick(["High", "Medium"]),
             closed_after=c.d(2, 6), csat=c.pick([4, 5]))
    b.sept_bill(i, "paid")
    return ["recent_upsell", "positive_call", "complaint_resolved_good_csat", "bill_on_time"]


def churn_stale_renewed_easy(c: _Ctx, i):
    b = c.b
    b.renewal(i, c.d(5, 25), c.pick([12, 24]))
    codes = ["recent_renewal", "bill_on_time"]
    if b.rng.random() < 0.6:
        b.call(i, c.d(1, 12), "enquiry", c.u(0.3, 0.7))
        codes.append("positive_call")
    b.sept_bill(i, "paid")
    return codes


def churn_stale_hard(c: _Ctx, i):
    b = c.b
    b.renewal(i, c.d(10, 25), c.pick([12, 24]))
    b.upsell_event(i, c.d(8, 24))
    b.call(i, c.d(1, 10), "enquiry", c.u(0.3, 0.6))
    b.ticket(i, c.d(2, 8), "Slow speed", "Low")
    b.external_signal(i, c.d(5, 25), "competitor")
    return ["recent_renewal", "recent_upsell", "positive_call", "minor_open_ticket", "competitor_signal"]


# --- acquisition recipes -------------------------------------------------------------


def acq_true_opportunity_easy(c: _Ctx, i):
    b = c.b
    b.call(i, c.d(1, 14), "acquisition", c.u(0.6, 0.9))
    b.external_signal(i, c.d(3, 25), "growth")
    b.sept_bill(i, "paid")
    return ["acquisition_call", "growth_signal", "bill_on_time"]


def acq_true_opportunity_hard(c: _Ctx, i):
    b = c.b
    b.call(i, c.d(1, 14), "acquisition", c.u(0.0, 0.25))
    b.external_signal(i, c.d(3, 25), "growth")
    b.ticket(i, c.d(10, 25), "Slow speed", "Low", closed_after=c.d(1, 3), csat=c.pick([3, 4]))
    b.ticket(i, c.d(2, 8), "Installation delay", "Low")
    b.sept_bill(i, "paid")
    return ["acquisition_call", "growth_signal", "minor_open_ticket", "bill_on_time"]


def acq_blocked_complaint_easy(c: _Ctx, i):
    b = c.b
    b.ticket(i, c.d(2, 18), c.pick(["SLA breach", "Outage"]), "High")
    b.call(i, c.d(1, 10), "complaint", -c.u(0.5, 0.9))
    return ["open_high_complaint", "negative_call"]


def acq_blocked_financial_easy(c: _Ctx, i):
    b = c.b
    # POL-CREDIT-05 needs two late bills among the last three; selection prefers a prior late flag
    if not (b.late_flags[i, 3] or b.late_flags[i, 4]):
        b.late_flags[i, 4] = True
    b.sept_bill(i, "unpaid")
    b.external_signal(i, c.d(3, 25), "distress", index=0)
    return ["bill_unpaid", "distress_signal"]


def acq_saturated_easy(c: _Ctx, i):
    b = c.b
    b.upsell_event(i, c.d(3, 15))
    b.upsell_event(i, c.d(16, 24))
    free = [a for a in ("Cloud Backup", "Static IP", "Managed Wi-Fi", "Cybersecurity Bundle")
            if a not in b.held.get(int(i), set())]
    if free:
        b.addon(i, c.pick(free), c.d(5, 20), True)
    b.sept_bill(i, "paid")
    return ["recent_upsell", "second_recent_purchase", "bill_on_time"]


def acq_blocked_hard(c: _Ctx, i):
    b = c.b
    b.call(i, c.d(1, 12), "acquisition", c.u(0.4, 0.7))
    b.external_signal(i, c.d(3, 25), "distress", index=2)
    b.sept_bill(i, "unpaid")
    b.ticket(i, c.d(2, 9), "Billing dispute", "Medium")
    return ["acquisition_call", "distress_signal", "bill_unpaid", "open_complaint"]


RECIPES = {
    "churn_true_risk_easy": churn_true_risk_easy,
    "churn_true_risk_hard": churn_true_risk_hard,
    "churn_stale_recovered_easy": churn_stale_recovered_easy,
    "churn_stale_renewed_easy": churn_stale_renewed_easy,
    "churn_stale_hard": churn_stale_hard,
    "acq_true_opportunity_easy": acq_true_opportunity_easy,
    "acq_true_opportunity_hard": acq_true_opportunity_hard,
    "acq_blocked_complaint_easy": acq_blocked_complaint_easy,
    "acq_blocked_financial_easy": acq_blocked_financial_easy,
    "acq_saturated_easy": acq_saturated_easy,
    "acq_blocked_hard": acq_blocked_hard,
}


def select_candidates(book: EventBook, masks: dict) -> tuple[np.ndarray, np.ndarray]:
    """Core-flagged, single-signal, opted-in accounts with no open pre-snapshot tickets."""
    lat, cust = book.pop.latent, book.pop.cust
    ok = ((cust["marketing_opt_in"].to_numpy() == 1) & ~(masks["flagged_c"] & masks["flagged_a"])
          & (lat["open_pre_tickets"].to_numpy() == 0))
    pool_c = np.flatnonzero(masks["core_c"] & ok)
    pool_a = np.flatnonzero(masks["core_a"] & ok)
    return book.rng.permutation(pool_c), book.rng.permutation(pool_a)


def plant(book: EventBook, masks: dict) -> np.ndarray:
    """Write all archetype events. Returns a boolean mask of planted account indices."""
    pool_c, pool_a = select_candidates(book, masks)
    need_c = sum(p[4] for p in PLAN if p[1] == "churn")
    need_a = sum(p[4] for p in PLAN if p[1] == "acquisition")
    if len(pool_c) < need_c or len(pool_a) < need_a:
        raise ValueError(
            f"Not enough candidate accounts to plant the eval set (churn {len(pool_c)}/{need_c}, "
            f"acquisition {len(pool_a)}/{need_a}). Use more accounts (>= 1000)."
        )
    # the most restrictive recipe picks first: prefer accounts that already paid late in Jul/Aug
    late_hist = book.late_flags[:, 3] | book.late_flags[:, 4]
    pool_a = np.concatenate([pool_a[late_hist[pool_a]], pool_a[~late_hist[pool_a]]])

    pools = {"churn": list(pool_c), "acquisition": list(pool_a)}
    p_before = {"churn": 1 / (1 + np.exp(-book.pop.z_true_c)),
                "acquisition": 1 / (1 + np.exp(-book.pop.z_true_a))}
    ctx = _Ctx(book)
    order = sorted(PLAN, key=lambda p: p[0] != "acq_blocked_financial_easy")
    planted = np.zeros(book.n, dtype=bool)
    for archetype, signal, verdict, difficulty, count in order:
        for _ in range(count):
            pool = pools[signal]
            if verdict == "confirmed":
                # a "confirmed" label means the account really is at risk / ready to buy
                k = next((k for k, j in enumerate(pool) if p_before[signal][j] >= MIN_P_BEFORE[signal]), None)
                if k is None:
                    raise ValueError(f"No candidate with enough underlying {signal} propensity; use more accounts.")
            else:
                k = 0
            i = int(pool.pop(k))
            codes = RECIPES[archetype](ctx, i)
            planted[i] = True
            book.planted[i] = {
                "archetype": archetype, "signal_type": signal, "expected_verdict": verdict,
                "difficulty": difficulty, "planted_signals": codes,
            }
    return planted
