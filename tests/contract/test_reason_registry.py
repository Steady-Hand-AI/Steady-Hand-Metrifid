"""Frozen comparison-reason, precedence, ordering, and exit tests."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any, cast

import pytest

from metrifid import (
    REASON_REGISTRY,
    STATUS_PRECEDENCE,
    ComparisonStatus,
    OperationalExitCode,
    ReasonCode,
    ReasonRecord,
)
from metrifid.errors import (
    ReasonRole,
    derive_comparison_status,
    ordered_reasons,
    projected_reason_codes,
    reason_order_key,
    status_exit_code,
)
from metrifid.json_values import CanonicalValue, FrozenCanonicalObject, freeze_canonical

EXPECTED_CODES = [
    "ENGINE_THREADPOOL_ACTIVE",
    "ENGINE_THREADPOOL_STATE_UNKNOWN",
    "INITIAL_STATE_NOT_PRESERVED",
    "INTERNAL_STEP_BUDGET_EXCEEDED",
    "TRACE_MEMORY_BUDGET_EXCEEDED",
    "COMPARISON_TIMEOUT",
    "BASELINE_NONDETERMINISTIC",
    "CANDIDATE_NONDETERMINISTIC",
    "BASELINE_NONFINITE_STATE",
    "BASELINE_INVALID_QUATERNION",
    "BASELINE_MUJOCO_WARNING",
    "BASELINE_NUMERICAL_ERROR_LOG",
    "BASELINE_EARLY_TERMINATION",
    "CANDIDATE_NONFINITE_STATE",
    "CANDIDATE_INVALID_QUATERNION",
    "CANDIDATE_MUJOCO_WARNING",
    "CANDIDATE_NUMERICAL_ERROR_LOG",
    "CANDIDATE_EARLY_TERMINATION",
    "TRACE_SAMPLE_COUNT_MISMATCH",
    "TRACE_BOUNDARY_INDEX_MISMATCH",
    "TRACE_TIME_RECURRENCE_MISMATCH",
    "TRACE_CHANNEL_LAYOUT_MISMATCH",
    "TRACE_MALFORMED",
    "JOINT_METRIC_TOLERANCE_EXCEEDED",
]


_DEFAULT_ROLE = object()
_DEFAULT_ROLES: dict[ReasonCode, ReasonRole] = {
    ReasonCode.ENGINE_THREADPOOL_ACTIVE: "comparison",
    ReasonCode.ENGINE_THREADPOOL_STATE_UNKNOWN: "comparison",
    ReasonCode.INITIAL_STATE_NOT_PRESERVED: "baseline",
    ReasonCode.INTERNAL_STEP_BUDGET_EXCEEDED: "comparison",
    ReasonCode.TRACE_MEMORY_BUDGET_EXCEEDED: "comparison",
    ReasonCode.COMPARISON_TIMEOUT: "comparison",
    ReasonCode.BASELINE_NONDETERMINISTIC: "baseline",
    ReasonCode.CANDIDATE_NONDETERMINISTIC: "candidate",
    ReasonCode.BASELINE_NONFINITE_STATE: "baseline",
    ReasonCode.BASELINE_INVALID_QUATERNION: "baseline",
    ReasonCode.BASELINE_MUJOCO_WARNING: "baseline",
    ReasonCode.BASELINE_NUMERICAL_ERROR_LOG: "baseline",
    ReasonCode.BASELINE_EARLY_TERMINATION: "baseline",
    ReasonCode.CANDIDATE_NONFINITE_STATE: "candidate",
    ReasonCode.CANDIDATE_INVALID_QUATERNION: "candidate",
    ReasonCode.CANDIDATE_MUJOCO_WARNING: "candidate",
    ReasonCode.CANDIDATE_NUMERICAL_ERROR_LOG: "candidate",
    ReasonCode.CANDIDATE_EARLY_TERMINATION: "candidate",
    ReasonCode.TRACE_SAMPLE_COUNT_MISMATCH: "comparison",
    ReasonCode.TRACE_BOUNDARY_INDEX_MISMATCH: "comparison",
    ReasonCode.TRACE_TIME_RECURRENCE_MISMATCH: "comparison",
    ReasonCode.TRACE_CHANNEL_LAYOUT_MISMATCH: "comparison",
    ReasonCode.TRACE_MALFORMED: "comparison",
    ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED: "comparison",
}


def evidence(value: dict[str, CanonicalValue]) -> FrozenCanonicalObject:
    """Construct the evidence fixture used by reason registry scenarios.

    Deterministic setup isolates reason registry without bypassing the contract boundary under
    assertion.
    """
    frozen = freeze_canonical(value)
    assert isinstance(frozen, Mapping)
    return frozen


def record(
    code: ReasonCode,
    *,
    role: ReasonRole | object = _DEFAULT_ROLE,
    object_type: str | None = None,
    object_name: str | None = None,
    metric: str | None = None,
    boundary_index: int | None = None,
    evidence_value: dict[str, CanonicalValue] | None = None,
) -> ReasonRecord:
    """Construct the record fixture used by reason registry scenarios.

    Deterministic setup isolates reason registry without bypassing the contract boundary under
    assertion.
    """
    selected_role = _DEFAULT_ROLES[code] if role is _DEFAULT_ROLE else cast(ReasonRole, role)
    if code is ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED:
        object_type = "joint" if object_type is None else object_type
        object_name = "elbow" if object_name is None else object_name
        metric = "angle_rad" if metric is None else metric
        boundary_index = 0 if boundary_index is None else boundary_index
    return ReasonRecord(
        code,
        selected_role,
        object_type,
        object_name,
        metric,
        boundary_index,
        evidence(evidence_value or {}),
    )


def test_registry_is_exact_total_and_unique() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises registry is exact total and unique; each refusal must retain its
    exact stage, exit code, and structured evidence contract.
    """
    assert [code.value for code in ReasonCode] == EXPECTED_CODES
    assert set(REASON_REGISTRY) == set(ReasonCode)
    assert len({code.value for code in ReasonCode}) == 24
    assert [rule.rank for rule in STATUS_PRECEDENCE] == list(range(1, 8))
    assert len({rule.rule_id for rule in STATUS_PRECEDENCE}) == 7
    assert STATUS_PRECEDENCE[-1].rule_id == "GREEN_DECLARED_WORKLOAD"
    assert all(
        binding.status_rule_id != "GREEN_DECLARED_WORKLOAD" for binding in REASON_REGISTRY.values()
    )


def test_exact_group_boundaries() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises exact group boundaries; each refusal must retain its exact stage,
    exit code, and structured evidence contract.
    """
    expected_groups = {
        "COVERAGE_PREEXECUTION": EXPECTED_CODES[0:6],
        "NONDETERMINISTIC_ROLE": EXPECTED_CODES[6:8],
        "BASELINE_NUMERICALLY_INVALID": EXPECTED_CODES[8:13],
        "CANDIDATE_NUMERICAL_REGRESSION": EXPECTED_CODES[13:18],
        "TRACE_INTEGRITY_FAILURE": EXPECTED_CODES[18:23],
        "JOINT_METRIC_CROSSING": EXPECTED_CODES[23:24],
    }
    for group, tokens in expected_groups.items():
        assert [
            code.value for code, rule in REASON_REGISTRY.items() if rule.status_rule_id == group
        ] == tokens


def test_public_status_exit_mapping_is_total() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises public status exit mapping is total; each refusal must retain its
    exact stage, exit code, and structured evidence contract.
    """
    expected = {
        ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD: OperationalExitCode.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD,
        ComparisonStatus.MATERIAL_BEHAVIOR_CHANGE: OperationalExitCode.MATERIAL_BEHAVIOR_CHANGE,
        ComparisonStatus.COVERAGE_INSUFFICIENT: OperationalExitCode.COVERAGE_INSUFFICIENT,
        ComparisonStatus.NONDETERMINISTIC_REPLAY: OperationalExitCode.NONDETERMINISTIC_REPLAY,
    }
    assert {status: status_exit_code(status) for status in ComparisonStatus} == expected
    with pytest.raises(TypeError):
        status_exit_code(cast(Any, "COVERAGE_INSUFFICIENT"))


def test_reason_record_round_trip_and_frozen_evidence() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises reason record round trip and frozen evidence; each refusal must
    retain its exact stage, exit code, and structured evidence contract.
    """
    primitive = {
        "code": "JOINT_METRIC_TOLERANCE_EXCEEDED",
        "role": "comparison",
        "object_type": "joint",
        "object_name": "elbow",
        "metric": "angle_rad",
        "boundary_index": 84,
        "evidence": {"metric": {"kind": "ieee754_binary64", "bits": "3ff0000000000000"}},
    }
    parsed = ReasonRecord.from_primitive(primitive)
    assert parsed.to_primitive() == primitive
    assert parsed.evidence is not primitive["evidence"]


@pytest.mark.parametrize("field", ["object_type", "object_name", "metric"])
def test_optional_reason_string_domain(field: str) -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises optional reason string domain; each refusal must retain its exact
    stage, exit code, and structured evidence contract.
    """
    base: dict[str, object] = {
        "code": "TRACE_MALFORMED",
        "role": "comparison",
        "object_type": None,
        "object_name": None,
        "metric": None,
        "boundary_index": None,
        "evidence": {},
    }
    assert ReasonRecord.from_primitive(base).to_primitive()[field] is None
    nonempty = {**base, field: "x"}
    assert ReasonRecord.from_primitive(nonempty).to_primitive()[field] == "x"
    with pytest.raises(ValueError, match=field):
        ReasonRecord.from_primitive({**base, field: ""})


@pytest.mark.parametrize(
    "primitive",
    [
        None,
        {},
        {
            "code": "UNKNOWN",
            "role": None,
            "object_type": None,
            "object_name": None,
            "metric": None,
            "boundary_index": None,
            "evidence": {},
        },
        {
            "code": 1,
            "role": None,
            "object_type": None,
            "object_name": None,
            "metric": None,
            "boundary_index": None,
            "evidence": {},
        },
        {
            "code": "TRACE_MALFORMED",
            "role": "other",
            "object_type": None,
            "object_name": None,
            "metric": None,
            "boundary_index": None,
            "evidence": {},
        },
        {
            "code": "TRACE_MALFORMED",
            "role": None,
            "object_type": 1,
            "object_name": None,
            "metric": None,
            "boundary_index": None,
            "evidence": {},
        },
        {
            "code": "TRACE_MALFORMED",
            "role": None,
            "object_type": None,
            "object_name": None,
            "metric": None,
            "boundary_index": -1,
            "evidence": {},
        },
        {
            "code": "TRACE_MALFORMED",
            "role": None,
            "object_type": None,
            "object_name": None,
            "metric": None,
            "boundary_index": True,
            "evidence": {},
        },
        {
            "code": "TRACE_MALFORMED",
            "role": None,
            "object_type": None,
            "object_name": None,
            "metric": None,
            "boundary_index": None,
            "evidence": [],
        },
        {
            "code": "TRACE_MALFORMED",
            "role": None,
            "object_type": None,
            "object_name": None,
            "metric": None,
            "boundary_index": None,
            "evidence": {},
            "extra": 1,
        },
    ],
)
def test_reason_record_invalid_primitive_refuses(primitive: object) -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises reason record invalid primitive refuses; each refusal must retain
    its exact stage, exit code, and structured evidence contract.
    """
    with pytest.raises((TypeError, ValueError)):
        ReasonRecord.from_primitive(primitive)


def test_reason_record_direct_constructor_validation() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises reason record direct constructor validation; each refusal must
    retain its exact stage, exit code, and structured evidence contract.
    """
    valid_evidence = evidence({})
    with pytest.raises(TypeError):
        ReasonRecord(
            cast(Any, "TRACE_MALFORMED"), "comparison", None, None, None, None, valid_evidence
        )
    with pytest.raises(ValueError):
        ReasonRecord(
            ReasonCode.TRACE_MALFORMED, cast(Any, "other"), None, None, None, None, valid_evidence
        )
    with pytest.raises(TypeError):
        ReasonRecord(
            ReasonCode.TRACE_MALFORMED, "comparison", cast(Any, 1), None, None, None, valid_evidence
        )
    with pytest.raises(TypeError):
        ReasonRecord(
            ReasonCode.TRACE_MALFORMED, "comparison", None, None, None, True, valid_evidence
        )
    with pytest.raises(ValueError):
        ReasonRecord(ReasonCode.TRACE_MALFORMED, "comparison", None, None, None, -1, valid_evidence)
    with pytest.raises(TypeError):
        ReasonRecord(
            ReasonCode.TRACE_MALFORMED, "comparison", None, None, None, None, cast(Any, [])
        )
    with pytest.raises(TypeError):
        ReasonRecord(
            ReasonCode.TRACE_MALFORMED,
            "comparison",
            None,
            None,
            None,
            None,
            cast(Any, {"x": object()}),
        )


def test_total_order_and_projection_across_permutations() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises total order and projection across permutations; each refusal must
    retain its exact stage, exit code, and structured evidence contract.
    """
    reasons = [
        record(ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED, object_name="z", boundary_index=2),
        record(ReasonCode.ENGINE_THREADPOOL_ACTIVE),
        record(ReasonCode.TRACE_MALFORMED, object_name="a", boundary_index=None),
        record(ReasonCode.TRACE_MALFORMED, object_name="a", boundary_index=1),
        record(
            ReasonCode.TRACE_MALFORMED, object_name="a", boundary_index=1, evidence_value={"x": 2}
        ),
        record(ReasonCode.TRACE_MALFORMED, object_type="joint", object_name=None, metric="m"),
    ]
    expected = ordered_reasons(reasons)
    expected_codes = (
        ReasonCode.ENGINE_THREADPOOL_ACTIVE,
        ReasonCode.TRACE_MALFORMED,
        ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED,
    )
    for permutation in itertools.permutations(reasons):
        assert ordered_reasons(permutation) == expected
        assert projected_reason_codes(permutation) == expected_codes
    assert len({reason_order_key(reason) for reason in reasons}) == len(reasons)


def test_duplicate_full_reason_records_refuse() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises duplicate full reason records refuse; each refusal must retain its
    exact stage, exit code, and structured evidence contract.
    """
    item = record(ReasonCode.TRACE_MALFORMED)
    with pytest.raises(ValueError, match="duplicate"):
        ordered_reasons((item, item))
    with pytest.raises(ValueError, match="duplicate"):
        projected_reason_codes((item, item))
    with pytest.raises(ValueError, match="duplicate"):
        derive_comparison_status((item, item))


def test_status_derivation_covers_every_rule() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises status derivation covers every rule; each refusal must retain its
    exact stage, exit code, and structured evidence contract.
    """
    cases = {
        ReasonCode.ENGINE_THREADPOOL_ACTIVE: ComparisonStatus.COVERAGE_INSUFFICIENT,
        ReasonCode.BASELINE_NONDETERMINISTIC: ComparisonStatus.NONDETERMINISTIC_REPLAY,
        ReasonCode.BASELINE_NONFINITE_STATE: ComparisonStatus.COVERAGE_INSUFFICIENT,
        ReasonCode.CANDIDATE_NONFINITE_STATE: ComparisonStatus.MATERIAL_BEHAVIOR_CHANGE,
        ReasonCode.TRACE_MALFORMED: ComparisonStatus.COVERAGE_INSUFFICIENT,
        ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED: ComparisonStatus.MATERIAL_BEHAVIOR_CHANGE,
    }
    assert (
        derive_comparison_status(()) is ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD
    )
    for code, expected in cases.items():
        assert derive_comparison_status((record(code),)) is expected
    assert (
        derive_comparison_status(
            (
                record(ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED),
                record(ReasonCode.ENGINE_THREADPOOL_ACTIVE),
            )
        )
        is ComparisonStatus.COVERAGE_INSUFFICIENT
    )


def test_order_helpers_reject_wrong_types() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises order helpers reject wrong types; each refusal must retain its exact
    stage, exit code, and structured evidence contract.
    """
    with pytest.raises(TypeError):
        reason_order_key(cast(Any, "not-a-reason"))
    for invalid in (cast(Any, "x"), cast(Any, [object()])):
        with pytest.raises(TypeError):
            ordered_reasons(invalid)
    with pytest.raises(TypeError):
        projected_reason_codes(cast(Any, [object()]))
    with pytest.raises(TypeError):
        derive_comparison_status(cast(Any, [object()]))


def test_reason_defensive_canonical_object_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise defensive adapters that remain deliberate public-schema guards."""
    import metrifid.errors as errors_module

    frozen = evidence({})
    monkeypatch.setattr(errors_module, "thaw_canonical", lambda value: [])
    with pytest.raises(TypeError, match="canonical object"):
        ReasonRecord(ReasonCode.TRACE_MALFORMED, "comparison", None, None, None, None, frozen)

    monkeypatch.setattr(errors_module, "thaw_canonical", lambda value: {})
    monkeypatch.setattr(errors_module, "freeze_canonical", lambda value: ())
    with pytest.raises(TypeError, match="canonical object"):
        ReasonRecord(ReasonCode.TRACE_MALFORMED, "comparison", None, None, None, None, frozen)

    primitive = {
        "code": "TRACE_MALFORMED",
        "role": "comparison",
        "object_type": None,
        "object_name": None,
        "metric": None,
        "boundary_index": None,
        "evidence": {},
    }
    with pytest.raises(TypeError, match="evidence must be an object"):
        ReasonRecord.from_primitive(primitive)
