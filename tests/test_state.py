from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from fiberpulse.state import (
    AccountState,
    ComplianceResult,
    ModelScore,
    OfferCandidate,
    OfferCheck,
    RankedOffer,
    SignalType,
    Status,
)


def make_state(**overrides) -> AccountState:
    score = ModelScore(
        signal_type=SignalType.CHURN, score=0.82, model_version="churn-v1", scored_month="2026-10"
    )
    return AccountState(account_id="ACC-000123", score=score, **overrides)


def test_score_must_be_a_probability():
    with pytest.raises(ValidationError):
        ModelScore(signal_type=SignalType.CHURN, score=1.4, model_version="v1", scored_month="2026-10")


def test_scored_month_format_is_enforced():
    with pytest.raises(ValidationError):
        ModelScore(signal_type=SignalType.CHURN, score=0.5, model_version="v1", scored_month="Oct-2026")


def test_idempotency_key_is_stable_per_account_month_signal():
    assert make_state().idempotency_key == "ACC-000123:2026-10:churn"


def test_gross_margin_is_computed_from_price_and_cost():
    offer = OfferCandidate(
        offer_id="O1", name="Speed upgrade", monthly_price_aed=1000, monthly_cost_aed=300,
        rationale="x",
    )
    assert offer.gross_margin == pytest.approx(0.70)


def test_compliant_offer_ids_only_lists_passed_checks():
    result = ComplianceResult(
        checks=[
            OfferCheck(offer_id="O1", gross_margin=0.7, passed=True),
            OfferCheck(offer_id="O2", gross_margin=0.4, passed=False, violations=["margin<60%"]),
        ]
    )
    assert result.compliant_offer_ids == ["O1"]


def test_exception_exit_clears_downstream_work():
    state = make_state(
        offers=[OfferCandidate(offer_id="O1", name="n", monthly_price_aed=10, monthly_cost_aed=1, rationale="r")],
        ranking=[RankedOffer(rank=1, offer_id="O1", expected_revenue_lift_aed=5, propensity_to_accept=0.3, reasoning="r")],
    )
    state.exit_with_exception("Paradox: recent upsell + positive call contradict churn score")
    assert state.status is Status.EXCEPTION_EXIT
    assert state.offers == [] and state.ranking == [] and state.compliance is None
    assert "Paradox" in state.exception_reason


def test_record_step_appends_trace():
    state = make_state()
    state.record_step("context_researcher", datetime.now(timezone.utc), provider="ollama", tokens_in=10)
    assert len(state.trace) == 1 and state.trace[0].ok


def test_state_round_trips_through_json():
    state = make_state()
    again = AccountState.model_validate_json(state.model_dump_json())
    assert again == state
