"""Completed receipt and operational-failure wire lifecycle attacks."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

import metrifid
from metrifid import (
    ComparisonReceipt,
    OperationalFailure,
    OperationalReasonCode,
    OperationalToolObservation,
    finalize_receipt,
)
from metrifid.operational import OperationalReason


def _digest(label: str) -> str:
    """Compute the canonical digest value used by completed artifact lifecycle fixtures.

    Content addressing keeps the mutation boundary explicit for completed artifact lifecycle.
    """
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _operational_candidate() -> OperationalFailure:
    """Construct the operational candidate fixture used by completed artifact lifecycle scenarios.

    Deterministic setup isolates completed artifact lifecycle without bypassing the contract
    boundary under assertion.
    """
    code = OperationalReasonCode.TOLERANCE_MISSING
    return OperationalFailure(
        schema="metrifid.operational_failure",
        schema_version=1,
        failure_rule_schema="metrifid.operational_failure_rules",
        failure_rule_schema_version=1,
        tool=OperationalToolObservation(
            "0.1.0a3", "VERIFIED_INSTALLED_DISTRIBUTION", _digest("distribution")
        ),
        operation="compare",
        stage=code.stage,
        reason=OperationalReason(code, "comparison", "joint_tolerances", "elbow", {}),
        available_inputs=(),
        environment=None,
        exit_code=code.exit_code,
        failure_sha256=None,
    )


def test_receipt_draft_cannot_serialize_and_completed_round_trip(
    green_candidate: ComparisonReceipt,
) -> None:
    """Protect the completed artifact lifecycle assurance boundary from behavioral drift.

    This scenario exercises receipt draft cannot serialize and completed round trip; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    with pytest.raises(ValueError, match="receipt_sha256"):
        green_candidate.to_primitive()

    completed = finalize_receipt(green_candidate)
    primitive = completed.to_primitive()
    assert primitive["receipt_sha256"] == completed.receipt_sha256
    assert ComparisonReceipt.from_primitive(primitive) == completed


def test_receipt_strict_parser_rejects_null_hash(
    green_candidate: ComparisonReceipt,
) -> None:
    """Protect the completed artifact lifecycle assurance boundary from behavioral drift.

    This scenario exercises receipt strict parser rejects null hash; the assertions pin the
    user-visible result and the evidence needed to explain that result.
    """
    primitive = finalize_receipt(green_candidate).to_primitive()
    primitive["receipt_sha256"] = None
    with pytest.raises((TypeError, ValueError), match="receipt_sha256"):
        ComparisonReceipt.from_primitive(primitive)


def test_operational_draft_cannot_serialize_and_completed_round_trip() -> None:
    """Protect the completed artifact lifecycle assurance boundary from behavioral drift.

    This scenario exercises operational draft cannot serialize and completed round trip; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    candidate = _operational_candidate()
    with pytest.raises(ValueError, match="failure_sha256"):
        candidate.to_primitive()

    completed = candidate.finalized()
    primitive = completed.to_primitive()
    assert primitive["failure_sha256"] == completed.failure_sha256
    assert OperationalFailure.from_primitive(primitive) == completed


def test_operational_strict_parser_rejects_null_hash() -> None:
    """Protect the completed artifact lifecycle assurance boundary from behavioral drift.

    This scenario exercises operational strict parser rejects null hash; the assertions pin the
    user-visible result and the evidence needed to explain that result.
    """
    primitive = _operational_candidate().finalized().to_primitive()
    primitive["failure_sha256"] = None
    with pytest.raises((TypeError, ValueError), match="failure_sha256"):
        OperationalFailure.from_primitive(primitive)


def test_finalizers_do_not_mutate_drafts(green_candidate: ComparisonReceipt) -> None:
    """Protect the completed artifact lifecycle assurance boundary from behavioral drift.

    This scenario exercises finalizers do not mutate drafts; the assertions pin the user-visible
    result and the evidence needed to explain that result.
    """
    receipt_completed = finalize_receipt(green_candidate)
    assert green_candidate.receipt_sha256 is None
    assert receipt_completed.receipt_sha256 is not None

    operational_candidate = _operational_candidate()
    operational_completed = operational_candidate.finalized()
    assert operational_candidate.failure_sha256 is None
    assert operational_completed.failure_sha256 is not None
    assert replace(operational_completed, failure_sha256=None) == operational_candidate


def test_public_namespace_exposes_no_draft_type_or_unhashed_helper() -> None:
    """Protect the completed artifact lifecycle assurance boundary from behavioral drift.

    This scenario exercises public namespace exposes no draft type or unhashed helper; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    forbidden = {
        "DraftComparisonReceipt",
        "DraftOperationalFailure",
        "_receipt_unhashed_primitive",
        "_operational_failure_unhashed_primitive",
    }
    assert forbidden.isdisjoint(metrifid.__all__)
    assert forbidden.isdisjoint(dir(metrifid))
