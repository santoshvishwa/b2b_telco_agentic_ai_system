"""Shared state that flows through the LangGraph pipeline.

One AccountState per (account, month, signal). Every agent reads what earlier
agents wrote and fills in exactly one section. Keeping each agent's output a
typed Pydantic model is what makes the graph testable: you can validate the
goal of each step without reading prose.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    CHURN = "churn"
    ACQUISITION = "acquisition"


class Verdict(str, Enum):
    CONFIRMED = "confirmed"  # reasoning agrees with the model score
    PARADOX = "paradox"      # reasoning disagrees -> suppress the action


class Status(str, Enum):
    PENDING = "pending"
    ENRICHED = "enriched"                    # after Agent 3
    TRIAGED = "triaged"                      # after Agent 4 (confirmed)
    EXCEPTION_EXIT = "exception_exit"        # Agent 4 paradox -> Service Recovery
    OFFERS_DRAFTED = "offers_drafted"        # after Agent 5
    COMPLIANCE_CHECKED = "compliance_checked"  # after Agent 6
    RANKED = "ranked"                        # after Agent 7, ready for human review
    FAILED = "failed"


# --- Input: what the predictive layer hands to the agents --------------------


class ModelScore(BaseModel):
    signal_type: SignalType
    score: float = Field(ge=0.0, le=1.0)
    model_version: str
    scored_month: str = Field(pattern=r"^\d{4}-\d{2}$")  # e.g. "2026-10"


# --- Agent 3: Context Enrichment ---------------------------------------------


class AccountContext(BaseModel):
    """Full account picture. Built mostly from deterministic SQL tools."""

    tenure_months: int = Field(ge=0)
    plan_name: str
    mrc_aed: float = Field(ge=0)
    addon_count: int = Field(ge=0)
    open_complaints_90d: int = Field(ge=0)
    days_since_last_complaint: int | None = None
    days_since_last_upsell: int | None = None
    bill_trend_pct_3m: float | None = None
    call_intent: str | None = None  # from Voice Intelligence, if any
    external_signals: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)  # what we could NOT find


# --- Agent 4: Paradox Triage -------------------------------------------------


class TriageDecision(BaseModel):
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=10)
    evidence_case_ids: list[str] = Field(default_factory=list)  # Outcomes KB cases cited


# --- Agent 5: Offer Architect ------------------------------------------------


class OfferCandidate(BaseModel):
    offer_id: str
    name: str
    monthly_price_aed: float = Field(gt=0)
    monthly_cost_aed: float = Field(ge=0)
    rationale: str

    @property
    def gross_margin(self) -> float:
        return (self.monthly_price_aed - self.monthly_cost_aed) / self.monthly_price_aed


# --- Agent 6: RAG Compliance -------------------------------------------------


class OfferCheck(BaseModel):
    offer_id: str
    gross_margin: float  # computed in code, never by the LLM
    passed: bool
    policy_refs: list[str] = Field(default_factory=list)  # policy chunks cited
    violations: list[str] = Field(default_factory=list)


class ComplianceResult(BaseModel):
    checks: list[OfferCheck] = Field(default_factory=list)

    @property
    def compliant_offer_ids(self) -> list[str]:
        return [c.offer_id for c in self.checks if c.passed]


# --- Agent 7: Recommendation Ranking -----------------------------------------


class RankedOffer(BaseModel):
    rank: int = Field(ge=1)
    offer_id: str
    expected_revenue_lift_aed: float
    propensity_to_accept: float = Field(ge=0.0, le=1.0)
    reasoning: str


# --- Observability -----------------------------------------------------------


class StepRecord(BaseModel):
    agent: str
    started_at: datetime
    duration_s: float
    provider: str | None = None
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    ok: bool = True
    note: str | None = None


# --- The state object --------------------------------------------------------


class AccountState(BaseModel):
    account_id: str
    score: ModelScore

    context: AccountContext | None = None
    triage: TriageDecision | None = None
    offers: list[OfferCandidate] = Field(default_factory=list)
    compliance: ComplianceResult | None = None
    ranking: list[RankedOffer] = Field(default_factory=list)

    status: Status = Status.PENDING
    exception_reason: str | None = None
    trace: list[StepRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def idempotency_key(self) -> str:
        """Re-running the same account/month/signal must overwrite, never duplicate."""
        return f"{self.account_id}:{self.score.scored_month}:{self.score.signal_type.value}"

    def record_step(self, agent: str, started_at: datetime, **fields) -> None:
        duration = (datetime.now(timezone.utc) - started_at).total_seconds()
        self.trace.append(
            StepRecord(agent=agent, started_at=started_at, duration_s=duration, **fields)
        )

    def exit_with_exception(self, reason: str) -> None:
        """Paradox path: suppress the campaign and bypass Agents 5-7."""
        self.status = Status.EXCEPTION_EXIT
        self.exception_reason = reason
        self.offers = []
        self.compliance = None
        self.ranking = []
