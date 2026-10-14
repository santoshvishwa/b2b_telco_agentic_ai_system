"""Offer catalog, machine-checkable policy rules, and the policy handbook text.

The rules in `evaluate_offer` are the ORACLE used to build and score Agent 6's evals.
Agent 6 itself must reason over the handbook text (RAG) plus the deterministic margin
check; it never calls the oracle.

All prices, costs, caps and thresholds are invented.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import MARGIN_FLOOR, meets_margin_floor

# --- Offer catalog -----------------------------------------------------------


@dataclass(frozen=True)
class Offer:
    offer_id: str
    name: str
    offer_type: str                 # "retention" or "upsell"
    monthly_price_aed: float        # what the customer pays per month for this offer
    monthly_cost_aed: float         # our cost to deliver it per month
    discount_pct: float             # discount embedded in the offer (0 if none)
    term_months: int
    eligible_segments: tuple[str, ...]
    min_tenure_months: int
    addon_name: str | None          # add-on the account would hold after accepting
    description: str

    @property
    def gross_margin(self) -> float:
        return (self.monthly_price_aed - self.monthly_cost_aed) / self.monthly_price_aed


ALL = ("Micro", "Small", "Medium")
SM = ("Small", "Medium")

OFFERS: tuple[Offer, ...] = (
    # --- retention (answer a churn signal) ---
    Offer("RET-01", "Loyalty Wi-Fi Bundle", "retention", 260, 70, 20, 12, ALL, 6, "Managed Wi-Fi",
          "Managed Wi-Fi at a loyalty price on a 12-month term."),
    Offer("RET-02", "Speed Step-Up", "retention", 180, 55, 10, 12, ALL, 6, None,
          "One speed tier up on a 12-month commitment."),
    Offer("RET-03", "Price Lock 24", "retention", 120, 30, 0, 24, ALL, 12, None,
          "Locks the current plan price for 24 months."),
    Offer("RET-04", "Premium Care Plus", "retention", 320, 150, 0, 12, SM, 12, None,
          "24/7 priority SLA and free router swap. Expensive to deliver."),
    Offer("RET-05", "Mega Loyalty Rebate", "retention", 400, 140, 30, 12, ALL, 6, None,
          "Deep 30% rebate package. Margin is fine, discount is not."),
    Offer("RET-06", "Security Shield Loyalty Pack", "retention", 200, 55, 15, 12, ALL, 6,
          "Cybersecurity Bundle", "Cybersecurity bundle at a loyalty price."),
    # --- upsell (answer an acquisition signal) ---
    Offer("UP-01", "Extra Bandwidth +200", "upsell", 350, 110, 0, 12, ALL, 3, "Extra Bandwidth",
          "Adds 200 Mbps of bandwidth."),
    Offer("UP-02", "Cybersecurity Bundle", "upsell", 220, 60, 0, 12, ALL, 3, "Cybersecurity Bundle",
          "Managed firewall and endpoint protection."),
    Offer("UP-03", "Cloud Backup 1TB", "upsell", 150, 35, 0, 12, ALL, 3, "Cloud Backup",
          "1 TB managed cloud backup."),
    Offer("UP-04", "VoIP 5 Lines", "upsell", 280, 120, 0, 12, ALL, 3, "VoIP Lines",
          "Five business VoIP lines. Margin below the floor."),
    Offer("UP-05", "SD-WAN Lite", "upsell", 900, 330, 0, 24, ("Medium",), 12, "SD-WAN Lite",
          "Multi-site SD-WAN for medium businesses."),
    Offer("UP-06", "Managed Wi-Fi", "upsell", 200, 50, 0, 12, ALL, 3, "Managed Wi-Fi",
          "Managed Wi-Fi with cloud controller."),
    Offer("UP-07", "Static IP Block", "upsell", 90, 15, 0, 12, ALL, 3, "Static IP",
          "Dedicated static IP address."),
    Offer("UP-08", "Symmetric Premium Upgrade", "upsell", 600, 250, 0, 12, SM, 6, None,
          "Move to a symmetric premium plan. Margin below the floor."),
    Offer("UP-09", "4G Business Continuity", "upsell", 240, 95, 0, 12, ALL, 3, "4G Backup",
          "4G failover link. Margin just above the floor."),
    Offer("UP-10", "Business Analytics Pack", "upsell", 500, 200, 0, 12, ALL, 6, None,
          "Usage analytics pack. Margin is exactly on the floor (boundary case)."),
)
OFFER_BY_ID = {o.offer_id: o for o in OFFERS}

# --- Policy rules (oracle) ---------------------------------------------------

DISCOUNT_CAP = {"Micro": 15, "Small": 20, "Medium": 25}   # percent
RETENTION_MIN_GAP_DAYS = 90
UPSELL_MIN_GAP_DAYS = 30
CREDIT_MAX_LATE = 2          # this many late/unpaid bills among the last 3 blocks big commitments
CREDIT_PRICE_LIMIT = 300     # AED per month
CREDIT_TERM_LIMIT = 24       # months

RULE_IDS = (
    "POL-MARGIN-01", "POL-DISC-02", "POL-FREQ-03", "POL-CARE-04",
    "POL-CREDIT-05", "POL-CONSENT-06", "POL-ELIG-07", "POL-DUP-08",
)


@dataclass(frozen=True)
class AccountFacts:
    """The facts a policy decision depends on, as of the scoring date."""

    segment: str
    tenure_months: int
    marketing_opt_in: bool
    open_high_ticket: bool
    late_or_unpaid_last3: int
    days_since_last_retention_offer: int | None
    days_since_last_upsell_offer: int | None
    active_addons: frozenset[str]


def evaluate_offer(offer: Offer, facts: AccountFacts) -> list[str]:
    """Policy rule IDs this (account, offer) pair violates, in RULE_IDS order."""
    v: list[str] = []
    if not meets_margin_floor(offer.gross_margin):
        v.append("POL-MARGIN-01")
    if offer.discount_pct > DISCOUNT_CAP[facts.segment]:
        v.append("POL-DISC-02")
    gap = (facts.days_since_last_retention_offer if offer.offer_type == "retention"
           else facts.days_since_last_upsell_offer)
    min_gap = RETENTION_MIN_GAP_DAYS if offer.offer_type == "retention" else UPSELL_MIN_GAP_DAYS
    if gap is not None and gap < min_gap:
        v.append("POL-FREQ-03")
    if offer.offer_type == "upsell" and facts.open_high_ticket:
        v.append("POL-CARE-04")
    if facts.late_or_unpaid_last3 >= CREDIT_MAX_LATE and (
        offer.monthly_price_aed > CREDIT_PRICE_LIMIT or offer.term_months >= CREDIT_TERM_LIMIT
    ):
        v.append("POL-CREDIT-05")
    if not facts.marketing_opt_in:
        v.append("POL-CONSENT-06")
    if facts.segment not in offer.eligible_segments or facts.tenure_months < offer.min_tenure_months:
        v.append("POL-ELIG-07")
    if offer.addon_name is not None and offer.addon_name in facts.active_addons:
        v.append("POL-DUP-08")
    return v


# --- Policy handbook (what Agent 6's RAG index is built from) ------------------


def render_handbook() -> str:
    """Markdown handbook. Each `## POL-...` section is one retrieval chunk."""
    floor_pct = int(round(MARGIN_FLOOR * 100))
    caps = ", ".join(f"{seg} {pct}%" for seg, pct in DISCOUNT_CAP.items())
    return f"""# Commercial Offer Policy Handbook (synthetic)

This handbook is invented for the FiberPulse AI learning project. It does not describe any real company's policy.
Each numbered section is a standalone policy. Section IDs are cited in compliance decisions.

## POL-DEF-00 Definitions
- **Segment**: Micro (1-9 employees), Small (10-49), Medium (50-249).
- **Retention offer**: an offer made in response to a churn signal. **Upsell offer**: an offer made in response to an acquisition signal.
- **Gross margin**: (monthly price - monthly cost) / monthly price, calculated per offer.
- **High-severity ticket**: a customer complaint classed High severity that is still open.
- **Late bill**: a bill paid more than {5} days after its due date, or an overdue unpaid bill.
- **Scoring date**: the date on which an account's signal is evaluated.

## POL-MARGIN-01 Gross margin floor
Every offer must have a gross margin of at least {floor_pct}%. An offer below {floor_pct}% must not be presented to a customer
under any circumstance, regardless of the account's value or churn risk. A margin of exactly {floor_pct}.0% meets the floor.
This is a hard rule and cannot be waived by an account manager.

## POL-DISC-02 Discount caps by segment
The discount embedded in an offer may not exceed the segment cap: {caps}. An offer whose discount exceeds the cap for the
customer's segment is not permitted for that customer, even if its margin meets the floor.

## POL-FREQ-03 Contact frequency
A customer may receive at most one retention offer in any {RETENTION_MIN_GAP_DAYS}-day period, and at most one upsell offer in any
{UPSELL_MIN_GAP_DAYS}-day period, counted from the date the previous offer of the same type was made, whether or not it was accepted.
A retention offer and an upsell offer do not count against each other.

## POL-CARE-04 No upsell during an open service failure
No upsell offer may be made to a customer who has an open High-severity ticket. Retention offers are still allowed, but the
account should first be contacted by Service Recovery. The restriction ends when the ticket is closed.

## POL-CREDIT-05 Credit control
If two or more of a customer's last three bills were late, the customer must not be offered anything with a monthly price above
{CREDIT_PRICE_LIMIT} AED or a term of {CREDIT_TERM_LIMIT} months or longer. Smaller, shorter offers remain allowed.

## POL-CONSENT-06 Marketing consent
Offers may only be sent to customers whose marketing consent is recorded as opted in. For customers who have opted out, the offer
must not be sent on any channel.

## POL-ELIG-07 Segment and tenure eligibility
Each offer in the catalog lists the segments that may receive it and a minimum customer tenure in months. A customer outside the
eligible segments, or with less than the minimum tenure, must not be offered it.

## POL-DUP-08 No duplicate add-ons
An offer that provides an add-on the customer already holds as an active service must not be made. Offers that do not add a new
add-on (plan changes, price locks, analytics) are not affected.

## POL-BRAND-09 Tone of customer messages
Messages use plain, respectful language, name the customer's company, avoid urgency tactics, and never mention internal scores.
Messages are sent in English or Arabic according to the account's language setting.

## POL-ESC-10 Escalation to Service Recovery
Accounts with an unresolved High-severity ticket older than 7 days, or with a documented threat to cancel, are routed to the
Service Recovery team. Service Recovery owns contact with the account until the issue is resolved.

## POL-DATA-11 Customer data handling
Personal data in call recordings and transcripts is masked before it reaches any language model. Account data is accessed read-only
by decision agents. Model reasoning, cited sources and approvals are stored for audit.
"""
