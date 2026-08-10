"""Exact comparison-reason role and joint-metric semantic bindings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import pytest

from metrifid import (
    ComparisonReceipt,
    EngineThreadpoolState,
    ReasonCode,
    ReasonRecord,
    finalize_receipt,
)
from metrifid.errors import ReasonRole, derive_comparison_status
from metrifid.json_values import CanonicalValue, FrozenCanonicalObject, freeze_canonical

_ALLOWED: dict[ReasonCode, frozenset[ReasonRole]] = {
    ReasonCode.ENGINE_THREADPOOL_ACTIVE: frozenset({"comparison"}),
    ReasonCode.ENGINE_THREADPOOL_STATE_UNKNOWN: frozenset({"comparison"}),
    ReasonCode.INITIAL_STATE_NOT_PRESERVED: frozenset({"baseline", "candidate"}),
    ReasonCode.INTERNAL_STEP_BUDGET_EXCEEDED: frozenset({"comparison"}),
    ReasonCode.TRACE_MEMORY_BUDGET_EXCEEDED: frozenset({"comparison"}),
    ReasonCode.COMPARISON_TIMEOUT: frozenset({"comparison"}),
    ReasonCode.BASELINE_NONDETERMINISTIC: frozenset({"baseline"}),
    ReasonCode.CANDIDATE_NONDETERMINISTIC: frozenset({"candidate"}),
    ReasonCode.BASELINE_NONFINITE_STATE: frozenset({"baseline"}),
    ReasonCode.BASELINE_INVALID_QUATERNION: frozenset({"baseline"}),
    ReasonCode.BASELINE_MUJOCO_WARNING: frozenset({"baseline"}),
    ReasonCode.BASELINE_NUMERICAL_ERROR_LOG: frozenset({"baseline"}),
    ReasonCode.BASELINE_EARLY_TERMINATION: frozenset({"baseline"}),
    ReasonCode.CANDIDATE_NONFINITE_STATE: frozenset({"candidate"}),
    ReasonCode.CANDIDATE_INVALID_QUATERNION: frozenset({"candidate"}),
    ReasonCode.CANDIDATE_MUJOCO_WARNING: frozenset({"candidate"}),
    ReasonCode.CANDIDATE_NUMERICAL_ERROR_LOG: frozenset({"candidate"}),
    ReasonCode.CANDIDATE_EARLY_TERMINATION: frozenset({"candidate"}),
    ReasonCode.TRACE_SAMPLE_COUNT_MISMATCH: frozenset({"comparison"}),
    ReasonCode.TRACE_BOUNDARY_INDEX_MISMATCH: frozenset({"comparison"}),
    ReasonCode.TRACE_TIME_RECURRENCE_MISMATCH: frozenset({"comparison"}),
    ReasonCode.TRACE_CHANNEL_LAYOUT_MISMATCH: frozenset({"comparison"}),
    ReasonCode.TRACE_MALFORMED: frozenset({"comparison"}),
    ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED: frozenset({"comparison"}),
}
_ALL_ROLES: tuple[ReasonRole, ...] = (None, "baseline", "candidate", "comparison")


def _evidence() -> FrozenCanonicalObject:
    """Construct the evidence fixture used by reason role binding scenarios.

    Deterministic setup isolates reason role binding without bypassing the contract boundary
    under assertion.
    """
    value = freeze_canonical(cast(CanonicalValue, {}))
    assert isinstance(value, Mapping)
    return cast(FrozenCanonicalObject, value)


def _record(code: ReasonCode, role: ReasonRole) -> ReasonRecord:
    """Construct the record fixture used by reason role binding scenarios.

    Deterministic setup isolates reason role binding without bypassing the contract boundary
    under assertion.
    """
    joint_metric = code is ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED
    return ReasonRecord(
        code=code,
        role=role,
        object_type="joint" if joint_metric else None,
        object_name="elbow" if joint_metric else None,
        metric="angle_rad" if joint_metric else None,
        boundary_index=0 if joint_metric else None,
        evidence=_evidence(),
    )


@pytest.mark.parametrize("code", tuple(ReasonCode))
@pytest.mark.parametrize("role", _ALL_ROLES)
def test_every_reason_code_has_exact_role_domain(code: ReasonCode, role: ReasonRole) -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises every reason code has exact role domain; each refusal must retain
    its exact stage, exit code, and structured evidence contract.
    """
    if role in _ALLOWED[code]:
        record = _record(code, role)
        assert ReasonRecord.from_primitive(record.to_primitive()) == record
    else:
        with pytest.raises(ValueError, match="role"):
            _record(code, role)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("object_type", "body"),
        ("object_name", None),
        ("metric", None),
        ("boundary_index", None),
    ],
)
def test_joint_metric_required_fields_refuse(field: str, value: object) -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises joint metric required fields refuse; each refusal must retain its
    exact stage, exit code, and structured evidence contract.
    """
    values: dict[str, object] = {
        "code": ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED,
        "role": "comparison",
        "object_type": "joint",
        "object_name": "elbow",
        "metric": "angle_rad",
        "boundary_index": 0,
        "evidence": _evidence(),
    }
    values[field] = value
    with pytest.raises(ValueError, match="JOINT_METRIC_TOLERANCE_EXCEEDED"):
        ReasonRecord(**values)  # type: ignore[arg-type]


def test_joint_metric_unmonitored_name_refuses_at_receipt_boundary(
    green_candidate: ComparisonReceipt,
) -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises joint metric unmonitored name refuses at receipt boundary; each
    refusal must retain its exact stage, exit code, and structured evidence contract.
    """
    reason = replace(
        _record(ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED, "comparison"), object_name="wrist"
    )
    candidate = replace(
        green_candidate,
        status=derive_comparison_status((reason,)),
        reasons=(reason,),
        reason_codes=(),
    )
    with pytest.raises(ValueError, match="monitored joint"):
        finalize_receipt(candidate)


def test_joint_metric_wrong_metric_for_joint_type_refuses_at_receipt_boundary(
    green_candidate: ComparisonReceipt,
) -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises joint metric wrong metric for joint type refuses at receipt
    boundary; each refusal must retain its exact stage, exit code, and structured evidence
    contract.
    """
    reason = replace(
        _record(ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED, "comparison"),
        metric="translation_m",
    )
    candidate = replace(
        green_candidate,
        status=derive_comparison_status((reason,)),
        reasons=(reason,),
        reason_codes=(),
    )
    with pytest.raises(ValueError, match="invalid for the monitored joint type"):
        finalize_receipt(candidate)


def test_valid_joint_metric_and_both_initial_state_roles_finalize(
    green_candidate: ComparisonReceipt,
) -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises valid joint metric and both initial state roles finalize; each
    refusal must retain its exact stage, exit code, and structured evidence contract.
    """
    for reason in (
        _record(ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED, "comparison"),
        _record(ReasonCode.INITIAL_STATE_NOT_PRESERVED, "baseline"),
        _record(ReasonCode.INITIAL_STATE_NOT_PRESERVED, "candidate"),
    ):
        threadpool = green_candidate.environment
        candidate = replace(
            green_candidate,
            status=derive_comparison_status((reason,)),
            reasons=(reason,),
            reason_codes=(),
            environment=replace(threadpool, engine_threadpool_state=EngineThreadpoolState.DISABLED),
        )
        assert finalize_receipt(candidate).receipt_sha256 is not None
