"""Behavior tests for the full-horizon runtime-review decision mapping."""

from __future__ import annotations

from metrifid._native_upgrade import CaseEvidence, GateEvent, ScalarObservation
from metrifid.runtime_review._decision import evaluate_runtime_case
from metrifid.runtime_review._status import RuntimeReviewReasonCode, RuntimeReviewStatus


def _scalar(
    channel_id: str,
    time: float,
    candidate: float,
) -> ScalarObservation:
    """Build one converged synthetic scalar witness at an exact physical time."""
    return ScalarObservation(
        channel_id,
        time,
        1.0,
        0.000001,
        (0.0, 0.0, 0.0),
        (candidate, candidate, candidate),
    )


def test_complete_within_horizon_has_no_single_decisive_witness() -> None:
    """A complete all-within case is green but no individual row proves full-horizon coverage."""
    decision = evaluate_runtime_case(
        CaseEvidence("synthetic_within", (_scalar("joint", 1.0, 0.0),), ())
    )

    assert decision.status is RuntimeReviewStatus.WITHIN_DECLARED_MIGRATION_ENVELOPE
    assert decision.reason_code is None
    assert decision.admitted_prefix == "1"
    assert decision.first_decisive_witness is None


def test_earliest_outside_witness_rejects_before_later_unqualified_suffix() -> None:
    """A decisive OUTSIDE row remains rejection when a later solver gate truncates evidence."""
    case = CaseEvidence(
        "synthetic_outside_before_gate",
        (
            _scalar("within.first", 0.1, 0.0),
            _scalar("outside.second", 0.2, 0.001),
            _scalar("outside.later", 0.3, 0.002),
        ),
        (),
        (GateEvent(0.75, "SOLVER_NOT_CONVERGED", "solver", "synthetic later gate"),),
    )

    decision = evaluate_runtime_case(case)

    assert decision.status is RuntimeReviewStatus.OUTSIDE_DECLARED_MIGRATION_ENVELOPE
    assert decision.reason_code is None
    assert decision.first_decisive_witness is not None
    assert decision.first_decisive_witness.channel_id == "outside.second"
    assert decision.first_decisive_witness.time == "0.2"


def test_partial_within_prefix_maps_to_named_insufficient_evidence_reason() -> None:
    """A nondecisive partial prefix maps its first named gate to insufficient evidence."""
    case = CaseEvidence(
        "synthetic_partial",
        (_scalar("joint", 0.5, 0.0),),
        (),
        (GateEvent(0.75, "SOLVER_NOT_CONVERGED", "solver", "synthetic later gate"),),
    )

    decision = evaluate_runtime_case(case)

    assert decision.status is RuntimeReviewStatus.INSUFFICIENT_EVIDENCE
    assert decision.reason_code is RuntimeReviewReasonCode.SOLVER_NOT_CONVERGED
    assert decision.first_decisive_witness is None


def test_full_horizon_boundary_overlap_maps_to_unresolved() -> None:
    """A complete witness whose interval straddles tolerance remains unresolved, never green."""
    decision = evaluate_runtime_case(
        CaseEvidence("synthetic_boundary", (_scalar("joint", 1.0, 0.000001),), ())
    )

    assert decision.status is RuntimeReviewStatus.UNRESOLVED_NEAR_BOUNDARY
    assert decision.reason_code is None
    assert decision.first_decisive_witness is not None
    assert decision.first_decisive_witness.classification == "UNRESOLVED_NEAR_BOUNDARY"
