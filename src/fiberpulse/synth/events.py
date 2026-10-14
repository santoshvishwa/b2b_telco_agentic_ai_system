"""EventBook: records every post-snapshot event and its effect on the TRUE propensities.

Each event appends a row to the right table AND nudges the account's true churn / acquisition
logit (shift_c / shift_a). The stand-in model never sees these events, so they are what makes
a high score stale (paradox) or genuinely right (confirmed).
"""

from __future__ import annotations

import random
from datetime import timedelta

import numpy as np

from . import constants as K
from .population import ADDON_NAMES, Population, iso

# (churn shift, acquisition shift) per open ticket severity
OPEN_TICKET_SHIFT = {"High": (0.8, -1.4), "Medium": (0.4, -0.5), "Low": (0.15, -0.15)}
EXTERNAL_SHIFT = {            # kind -> (churn shift, acquisition shift)
    "growth": (-0.1, 0.9), "distress": (0.4, -1.4), "competitor": (0.7, -0.2),
    "positive_brand": (-0.3, 0.2), "neutral": (0.0, 0.0),
}
EXTERNAL_POLARITY = {"growth": "positive", "positive_brand": "positive", "neutral": "neutral",
                     "distress": "negative", "competitor": "negative"}
SEPT_BILL_SHIFT = {"paid": (0.0, 0.0), "late": (0.35, -0.6), "unpaid": (0.5, -0.8),
                   "disputed": (0.3, -0.2)}


def sentiment_label(s: float) -> str:
    return "pos" if s > 0.25 else "neg" if s < -0.25 else "neu"


class EventBook:
    def __init__(self, pop: Population, rng: np.random.Generator, pyrng: random.Random):
        self.pop, self.rng, self.pyrng, self.n = pop, rng, pyrng, pop.n
        self.cust = pop.cust
        self.late_flags = pop.late_flags
        self.tickets: list[dict] = []
        self.calls: list[dict] = []
        self.addons: list[dict] = []
        self.external: list[dict] = []
        self.shift_c = np.zeros(self.n)
        self.shift_a = np.zeros(self.n)
        self.sept = np.array(["paid"] * self.n, dtype=object)
        self.planted: dict[int, dict] = {}
        act = pop.addons[pop.addons["end_date"].isna()]
        self.held: dict[int, set[str]] = {}
        for i, name in zip(act["i"].to_numpy(), act["addon_name"].to_numpy()):
            self.held.setdefault(int(i), set()).add(name)

    # -- helpers ----------------------------------------------------------------
    def _date(self, days_ago: int):
        return K.AS_OF - timedelta(days=int(days_ago))

    @staticmethod
    def _fresh(days_ago: int) -> bool:
        return days_ago <= K.FRESH_DAYS

    def _shift(self, i: int, churn: float, acq: float, days_ago: int) -> None:
        if self._fresh(days_ago):
            self.shift_c[i] += churn
            self.shift_a[i] += acq

    def aid(self, i: int) -> str:
        return self.cust.at[i, "account_id"]

    # -- events -------------------------------------------------------------------
    def ticket(self, i, days_ago, category, severity, closed_after=None, csat=None):
        opened = self._date(days_ago)
        is_complaint = category in K.COMPLAINT_CATEGORIES
        closed = None
        if closed_after is not None:
            closed = min(opened + timedelta(days=int(closed_after)), K.AS_OF - timedelta(days=1))
        hours = -1 if closed is None else int((closed - opened).days * 24 + self.rng.integers(0, 24))
        self.tickets.append({
            "i": int(i), "opened_date": opened.isoformat(),
            "closed_date": closed.isoformat() if closed else None,
            "category": category, "is_complaint": int(is_complaint), "severity": severity,
            "status": "Closed" if closed else "Open", "csat": csat if closed else None,
            "resolution_hours": hours,
        })
        if is_complaint:
            if closed is None:
                self._shift(i, *OPEN_TICKET_SHIFT[severity], days_ago)
            elif csat is not None and csat >= 4:
                self._shift(i, -0.3, 0.1, days_ago)
            elif csat is not None and csat <= 2:
                self._shift(i, 0.3, -0.2, days_ago)

    def call(self, i, days_ago, intent, sentiment, direction=None):
        label = sentiment_label(sentiment)
        pool = K.CALL_SUMMARIES.get((intent, label)) or K.CALL_SUMMARIES[(intent, "neu")]
        if direction is None:
            direction = "outbound" if (intent == "acquisition" and self.rng.random() < 0.5) else "inbound"
        self.calls.append({
            "i": int(i), "call_date": self._date(days_ago).isoformat(), "direction": direction,
            "duration_sec": int(self.rng.integers(60, 900)), "intent": intent,
            "sentiment": {"pos": "positive", "neg": "negative", "neu": "neutral"}[label],
            "sentiment_score": round(float(sentiment), 2), "summary": self.pyrng.choice(pool),
        })
        churn = -0.8 * sentiment + (0.3 if intent == "complaint" and sentiment < 0 else 0.0)
        acq = 0.8 * sentiment + (1.0 if intent == "acquisition" else 0.0)
        self._shift(i, churn, acq, days_ago)

    def plan_change(self, i, days_ago, kind) -> bool:
        cur = int(self.cust.at[i, "plan_idx"])
        new = cur + 1 if kind == "upgrade" else cur - 1
        if not 0 <= new < len(K.PLANS):
            return False
        c = self.cust
        c.at[i, "prev_mrc_aed"] = int(c.at[i, "mrc_aed"])
        c.at[i, "plan_idx"] = new
        c.at[i, "plan_name"] = K.PLANS[new][0]
        c.at[i, "plan_speed_mbps"] = K.PLANS[new][1]
        c.at[i, "mrc_aed"] = int(round(K.PLANS[new][2] * float(c.at[i, "price_ratio"])))
        c.at[i, "last_plan_change_date"] = self._date(days_ago).isoformat()
        c.at[i, "last_plan_change_type"] = kind
        self._shift(i, *((-0.9, -0.5) if kind == "upgrade" else (0.9, -0.9)), days_ago)
        return True

    def renewal(self, i, days_ago, term):
        c = self.cust
        c.at[i, "contract_term_months"] = int(term)
        c.at[i, "contract_months_remaining"] = int(term)
        c.at[i, "last_renewal_date"] = self._date(days_ago).isoformat()
        self._shift(i, -1.2, 0.0, days_ago)

    def addon(self, i, name, days_ago, is_upsell=True) -> bool:
        held = self.held.setdefault(int(i), set())
        if name in held:
            return False
        held.add(name)
        self.addons.append({
            "i": int(i), "addon_name": name, "monthly_price_aed": K.ADDONS[name][0],
            "start_date": self._date(days_ago).isoformat(), "end_date": None,
            "is_upsell": int(is_upsell),
        })
        if is_upsell:
            self._shift(i, -0.5, -0.7, days_ago)
        return True

    def external_signal(self, i, days_ago, kind, index=None):
        pool = K.EXTERNAL_SIGNALS[kind]
        source, text = pool[index] if index is not None else self.pyrng.choice(pool)
        self.external.append({
            "i": int(i), "signal_date": self._date(days_ago).isoformat(), "source": source,
            "signal_kind": kind, "signal_text": text.format(emirate=self.cust.at[i, "emirate"]),
            "polarity": EXTERNAL_POLARITY[kind],
        })
        self._shift(i, *EXTERNAL_SHIFT[kind], days_ago)

    def sept_bill(self, i, status):
        self.sept[i] = status
        self._shift(i, *SEPT_BILL_SHIFT[status], 1)

    def upsell_event(self, i, days_ago) -> str:
        """A 'recent upsell': a plan upgrade when possible (50%), otherwise a new upsell add-on."""
        if self.rng.random() < 0.5 and self.plan_change(i, days_ago, "upgrade"):
            return "plan_upgrade"
        free = [a for a in ADDON_NAMES if a not in self.held.get(int(i), set())]
        if free and self.addon(i, self.pyrng.choice(free), days_ago):
            return "addon_upsell"
        return "plan_upgrade" if self.plan_change(i, days_ago, "upgrade") else "none"


# --- baseline (non-planted) events -------------------------------------------------


def baseline_events(book: EventBook, planted_mask: np.ndarray) -> None:
    """Background noise for everyone. Planted accounts get no fresh-window events from here."""
    rng, pyrng, pop, n = book.rng, book.pyrng, book.pop, book.n
    u_c = pop.latent["u_c"].to_numpy()
    late_pre = pop.latent["late_bills_pre_3m"].to_numpy()
    remaining = pop.cust["contract_months_remaining"].to_numpy()
    quiet = ~planted_mask

    # fresh tickets (complaints only; ~55% already closed)
    cat_p = np.array(K.TICKET_CATEGORY_P[:5])
    cat_p = cat_p / cat_p.sum()
    for i in np.flatnonzero(quiet):
        for _ in range(rng.poisson(0.12 * np.exp(0.5 * u_c[i]))):
            cat = K.COMPLAINT_CATEGORIES[rng.choice(5, p=cat_p)]
            sev = K.SEVERITIES[rng.choice(3, p=[0.50, 0.35, 0.15])]
            days = int(rng.integers(1, 31))
            if days > 1 and rng.random() < 0.55:
                after = min(int(rng.integers(1, 6)), days - 1)
                csat = (int(rng.choice([1, 2, 3, 4, 5], p=[0.07, 0.10, 0.20, 0.33, 0.30]))
                        if rng.random() < 0.8 else None)
                book.ticket(i, days, cat, sev, closed_after=after, csat=csat)
            else:
                book.ticket(i, days, cat, sev)

    # fresh calls
    intents = ("complaint", "enquiry", "acquisition")
    bias = {"complaint": -0.3, "enquiry": 0.0, "acquisition": 0.2}
    has_call = quiet & (rng.random(n) < 0.10)
    for i in np.flatnonzero(has_call):
        intent = intents[rng.choice(3, p=[0.30, 0.45, 0.25])]
        s = float(np.clip(-0.25 * u_c[i] + rng.normal(0, 0.45) + bias[intent], -1, 1))
        book.call(i, int(rng.integers(1, 31)), intent, s)

    # fresh plan changes, renewals, upsell add-ons
    for i in np.flatnonzero(quiet & (rng.random(n) < 0.012)):
        book.plan_change(i, int(rng.integers(1, 31)), "upgrade")
    for i in np.flatnonzero(quiet & (rng.random(n) < 0.008)):
        book.plan_change(i, int(rng.integers(1, 31)), "downgrade")
    near_expiry = remaining <= 2
    for i in np.flatnonzero(quiet & near_expiry & (rng.random(n) < 0.15)):
        book.renewal(i, int(rng.integers(1, 31)), int(rng.choice([12, 24])))
    for i in np.flatnonzero(quiet & (rng.random(n) < 0.035)):
        free = [a for a in ADDON_NAMES if a not in book.held.get(int(i), set())]
        if free:
            book.addon(i, pyrng.choice(free), int(rng.integers(1, 31)), is_upsell=True)

    # September bill status
    p_late = 0.03 + 0.04 * (late_pre > 0)
    draw = rng.random(n)
    for i in np.flatnonzero(quiet):
        d = draw[i]
        if d < p_late[i]:
            book.sept_bill(i, "late")
        elif d < p_late[i] + 0.015:
            book.sept_bill(i, "unpaid")
        elif d < p_late[i] + 0.030:
            book.sept_bill(i, "disputed")

    # older calls (history, no effect on truth); never earlier than the account's start
    tenure_days = (pop.cust["tenure_months"].to_numpy() * 30.44).astype(int) - 1
    for i in np.flatnonzero(rng.random(n) < 0.15):
        intent = intents[rng.choice(3, p=[0.30, 0.45, 0.25])]
        s = float(np.clip(rng.normal(0, 0.5) + bias[intent], -1, 1))
        book.call(i, int(rng.integers(31, min(360, tenure_days[i]) + 1)), intent, s)

    # external enrichment: ~90% coverage; planted accounts only get neutral history
    kinds = [k for k, _ in K.EXTERNAL_KIND_P]
    kind_p = np.array([p for _, p in K.EXTERNAL_KIND_P])
    covered = rng.random(n) < K.EXTERNAL_COVERAGE
    for i in np.flatnonzero(covered):
        for _ in range(1 + rng.poisson(0.6)):
            kind = "neutral" if planted_mask[i] else kinds[rng.choice(len(kinds), p=kind_p)]
            hi = min(180, tenure_days[i])
            days = int(rng.integers(31, hi + 1)) if planted_mask[i] else int(rng.integers(1, hi + 1))
            book.external_signal(i, days, kind)
