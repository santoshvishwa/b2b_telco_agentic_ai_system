"""Customer Outcomes knowledge base: historical cases that Agent 4 retrieves (RAG).

Cases come from a HISTORICAL cohort (ids like H-0001) that is completely separate from the
current accounts, so retrieval can never leak the answer for an eval account. About 10% of
paradox-pattern cases have a contrary outcome on purpose, so the KB is not a perfect oracle.
"""

from __future__ import annotations

import random

import numpy as np

from . import constants as K
from .catalog import OFFER_BY_ID

# (pattern, signal_type, count)
KB_PLAN = (
    ("churn_true_risk", "churn", 90),
    ("churn_stale_recovered", "churn", 45),
    ("churn_stale_renewed", "churn", 25),
    ("churn_stale_mixed", "churn", 40),
    ("acq_opportunity", "acquisition", 70),
    ("acq_blocked_complaint", "acquisition", 25),
    ("acq_blocked_financial", "acquisition", 20),
    ("acq_saturated", "acquisition", 20),
    ("acq_blocked_mixed", "acquisition", 25),
)
CONFIRMED_PATTERNS = {"churn_true_risk", "acq_opportunity"}
NOISE = 0.10

RETENTION_OFFERS = ["RET-01", "RET-02", "RET-03", "RET-06"]
UPSELL_OFFERS = ["UP-01", "UP-02", "UP-03", "UP-06", "UP-07", "UP-09"]
COMPLAINT_TYPES = ["repeated outages", "slow speeds at peak hours", "a billing dispute",
                   "a delayed installation", "an SLA breach"]
TOPICS = ["adding more bandwidth", "the cybersecurity bundle", "renewal options", "a second branch connection"]


def _offer(o: str) -> str:
    return f"{o} ({OFFER_BY_ID[o].name})"


class _G:
    def __init__(self, rng, pyrng):
        self.rng, self.pyrng = rng, pyrng

    def d(self, lo, hi):
        return int(self.rng.integers(lo, hi + 1))

    def p(self, prob):
        return self.rng.random() < prob

    def pick(self, seq):
        return self.pyrng.choice(seq)


def _confirmed_churn(g: _G):
    n = g.d(1, 3)
    ctx = (f"{n} open complaint{'s' if n > 1 else ''} about {g.pick(COMPLAINT_TYPES)} raised in the last "
           f"{g.d(3, 25)} days, rated {g.pick(['High', 'High', 'Medium'])} severity. An inbound call "
           f"{g.d(1, 12)} days ago was negative and the customer asked about contract exit terms. "
           f"The latest bill was {g.pick(['unpaid', 'disputed', 'paid late'])}.")
    if g.p(0.5):
        ctx += " A competitor promotion was running in the area."
    fixed = g.p(0.6)
    offer = g.pick(RETENTION_OFFERS)
    action = (f"Service Recovery fixed the open issue first, then retention offer {_offer(offer)} was made."
              if fixed else f"Retention offer {_offer(offer)} was made without resolving the open complaint.")
    retained = g.p(0.72 if fixed else 0.35)
    outcome = "Retained; still active after 6 months." if retained else "Churned within 3 months."
    lesson = ("Open unresolved complaints plus an exit enquiry are a genuine churn signal. "
              "Retention offers work best when the service problem is fixed first.")
    return ctx, action, outcome, lesson


def _paradox_actions(g: _G, signal: str, stale_ok: str, legacy_outcomes: tuple[str, str], offers: list[str]):
    """Return (action, outcome, validity) for a paradox pattern with ~10% contrary outcomes."""
    if g.p(0.5):   # the account was correctly suppressed
        action = "Campaign suppressed and the account routed to Service Recovery for a courtesy check-in."
        if g.p(NOISE):
            return action, "Account churned within 6 months despite the suppression.", "real"
        return action, stale_ok, "stale_or_blocked"
    offer = g.pick(offers)   # the legacy process acted on the score anyway
    action = f"The legacy process sent {_offer(offer)} anyway, based on the score alone."
    if g.p(NOISE):
        return action, "Offer accepted and the customer stayed engaged.", "real"
    return action, legacy_outcomes[0] if g.p(0.7) else legacy_outcomes[1], "stale_or_blocked"


def _stale_recovered(g: _G):
    ctx = (f"The plan was {g.pick(['upgraded', 'expanded'])} {g.d(8, 24)} days before the flag. "
           f"An inbound call {g.d(1, 12)} days ago was positive: the customer asked about {g.pick(TOPICS)}. "
           f"An earlier complaint about {g.pick(COMPLAINT_TYPES)} was closed {g.d(5, 20)} days ago "
           f"with a CSAT of {g.pick([4, 5])}. The latest bill was paid on time.")
    lesson = ("A recent upgrade and a positive call outweigh an old complaint history; "
              "the score was calculated before these events happened.")
    return ctx, lesson


def _stale_renewed(g: _G):
    ctx = (f"The customer renewed a {g.pick([12, 24])}-month contract {g.d(5, 25)} days before the flag. "
           + (f"An inbound call {g.d(1, 12)} days ago was positive. " if g.p(0.6) else "")
           + "There are no open complaints and the latest bill was paid on time.")
    lesson = "A fresh multi-month renewal removes near-term churn risk, whatever the older history says."
    return ctx, lesson


def _stale_mixed(g: _G):
    ctx = (f"The customer renewed a {g.pick([12, 24])}-month contract {g.d(10, 25)} days ago and upgraded "
           f"{g.d(8, 24)} days ago. A call {g.d(1, 10)} days ago was positive. One minor ticket about slow "
           f"speeds is open and a competitor promotion is running in the area.")
    lesson = "Two strong positives (renewal and upgrade) outweigh a minor open ticket and background competitor noise."
    return ctx, lesson


def _acq_opportunity(g: _G):
    ctx = (f"A call {g.d(1, 14)} days ago was about {g.pick(['a second office', 'adding staff', 'higher bandwidth'])}. "
           f"External signal: {g.pick(['a new branch was announced', 'a hiring surge was posted', 'the trade licence was expanded'])}. "
           f"There are no open complaints, bills are on time and utilisation is {g.d(78, 97)}%.")
    offer = g.pick(UPSELL_OFFERS)
    action = f"Upsell offer {_offer(offer)} was made."
    accepted = g.p(0.68)
    outcome = "Offer accepted within 30 days." if accepted else "Offer declined; the customer said the timing was not right."
    lesson = "An explicit acquisition call plus a growth signal and healthy account standing converts well."
    return ctx, action, outcome, lesson


def _blocked_complaint(g: _G):
    ctx = (f"An open High-severity ticket about {g.pick(['an outage', 'an SLA breach'])} was raised {g.d(2, 18)} days ago, "
           f"and a call {g.d(1, 10)} days ago was negative. Utilisation is high, so the propensity score is high.")
    lesson = "Never pitch an upsell during an open service failure (POL-CARE-04). Fix the problem first."
    return ctx, lesson


def _blocked_financial(g: _G):
    ctx = (f"Two of the last three bills were late and the latest bill is unpaid. "
           f"External signal: {g.pick(['credit bureau flag for delayed supplier payments', 'trade licence renewal overdue'])}.")
    lesson = "Payment stress overrides appetite: avoid new commitments for accounts with repeated late bills (POL-CREDIT-05)."
    return ctx, lesson


def _saturated(g: _G):
    ctx = (f"The plan was upgraded {g.d(3, 15)} days ago and an add-on ({g.pick(['Cloud Backup', 'Static IP', 'Managed Wi-Fi'])}) "
           f"was bought {g.d(5, 20)} days ago. There has been no recent contact about more services.")
    lesson = "Recent purchases mean the appetite is already served; the score does not know about them."
    return ctx, lesson


def _blocked_mixed(g: _G):
    ctx = (f"A call {g.d(1, 12)} days ago was positive about expansion, but an external signal reports "
           f"{g.pick(['downsizing', 'a branch closure'])}, the latest bill is unpaid and a billing dispute is open.")
    lesson = "Expansion talk does not outweigh downsizing, an unpaid bill and an open dispute."
    return ctx, lesson


STALE_BUILDERS = {
    "churn_stale_recovered": (_stale_recovered, "Account still active after 6 months with no intervention.",
                              ("Offer accepted, but the customer had already recovered; the discount was given away unnecessarily.",
                               "Offer ignored; the account stayed anyway."), RETENTION_OFFERS),
    "churn_stale_renewed": (_stale_renewed, "Account still active after 6 months with no intervention.",
                            ("Offer accepted, but the customer was locked in by the renewal anyway; margin was lost for nothing.",
                             "Offer ignored; the account stayed on contract."), RETENTION_OFFERS),
    "churn_stale_mixed": (_stale_mixed, "Account still active after 6 months with no intervention.",
                          ("Offer accepted, but the customer was already secure; discount wasted.",
                           "Offer ignored; the account stayed."), RETENTION_OFFERS),
    "acq_blocked_complaint": (_blocked_complaint, "No offer sent; the ticket closed and a later upsell was accepted.",
                              ("Offer rejected and the customer escalated the open complaint.",
                               "Offer rejected; the customer said they wanted the fault fixed first."), UPSELL_OFFERS),
    "acq_blocked_financial": (_blocked_financial, "No offer sent; the account stayed stable without new commitments.",
                              ("Offer accepted, but the customer defaulted on payment within 2 months.",
                               "Offer rejected."), UPSELL_OFFERS),
    "acq_saturated": (_saturated, "No offer sent; the customer later asked for the next product themselves.",
                      ("Offer declined: the customer said they had just bought something.",
                       "Offer declined: no need at present."), UPSELL_OFFERS),
    "acq_blocked_mixed": (_blocked_mixed, "No offer sent; the account later downsized.",
                          ("Offer declined; the customer downsized shortly after.",
                           "Offer rejected: unpaid bill still unresolved."), UPSELL_OFFERS),
}


def build_outcomes_kb(rng: np.random.Generator, pyrng: random.Random) -> list[dict]:
    g = _G(rng, pyrng)
    plan = [(pat, sig) for pat, sig, count in KB_PLAN for _ in range(count)]
    order = rng.permutation(len(plan))
    cases = []
    for k, idx in enumerate(order, start=1):
        pattern, signal = plan[idx]
        if pattern == "churn_true_risk":
            ctx, action, outcome, lesson = _confirmed_churn(g)
            validity = "real"
        elif pattern == "acq_opportunity":
            ctx, action, outcome, lesson = _acq_opportunity(g)
            validity = "real"
        else:
            builder, stale_ok, legacy, offers = STALE_BUILDERS[pattern]
            ctx, lesson = builder(g)
            action, outcome, validity = _paradox_actions(g, signal, stale_ok, legacy, offers)
        segment = K.SEGMENTS[int(rng.choice(3, p=K.SEGMENT_WEIGHTS))]
        industry = K.INDUSTRIES[int(rng.choice(len(K.INDUSTRIES), p=[x[1] for x in K.INDUSTRIES]))][0]
        tenure = int(np.clip(3 + rng.gamma(2.0, 16.0), 3, 120))
        score = round(float(rng.uniform(0.70, 0.98)), 2)
        text = (f"[{signal} | {segment} | {industry} | tenure {tenure} months | score {score:.2f}] "
                f"{ctx} Action: {action} Outcome: {outcome} Lesson: {lesson}")
        cases.append({
            "case_id": f"CASE-{k:04d}", "hist_account_id": f"H-{k:04d}", "signal_type": signal,
            "segment": segment, "industry": industry, "tenure_months": tenure, "score_at_flag": score,
            "context": ctx, "action": action, "outcome": outcome, "signal_validity": validity,
            "lesson": lesson, "text": text,
        })
    return cases
