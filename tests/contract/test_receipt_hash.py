"""Strict completed receipt lifecycle, hash, and semantic tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, cast

import pytest

from metrifid import (
    ComparisonReceipt,
    ComparisonStatus,
    EngineThreadpoolState,
    LimitationCode,
    ReasonCode,
    ReasonRecord,
    finalize_receipt,
    validate_receipt,
)
from metrifid.schemas import (
    CanonicalSummary,
    ComparisonInputsIdentity,
    MetricEvidenceSummary,
)


def digest(label: str) -> str:
    """Compute the canonical digest value used by receipt hash fixtures.

    Content addressing keeps the mutation boundary explicit for receipt hash.
    """
    return hashlib.sha256(label.encode()).hexdigest()


def reason(code: ReasonCode) -> ReasonRecord:
    """Construct the reason fixture used by receipt hash scenarios.

    Deterministic setup isolates receipt hash without bypassing the contract boundary under
    assertion.
    """
    baseline_codes = {
        ReasonCode.INITIAL_STATE_NOT_PRESERVED,
        ReasonCode.BASELINE_NONDETERMINISTIC,
        ReasonCode.BASELINE_NONFINITE_STATE,
        ReasonCode.BASELINE_INVALID_QUATERNION,
        ReasonCode.BASELINE_MUJOCO_WARNING,
        ReasonCode.BASELINE_NUMERICAL_ERROR_LOG,
        ReasonCode.BASELINE_EARLY_TERMINATION,
    }
    candidate_codes = {
        ReasonCode.CANDIDATE_NONDETERMINISTIC,
        ReasonCode.CANDIDATE_NONFINITE_STATE,
        ReasonCode.CANDIDATE_INVALID_QUATERNION,
        ReasonCode.CANDIDATE_MUJOCO_WARNING,
        ReasonCode.CANDIDATE_NUMERICAL_ERROR_LOG,
        ReasonCode.CANDIDATE_EARLY_TERMINATION,
    }
    role = (
        "baseline"
        if code in baseline_codes
        else "candidate"
        if code in candidate_codes
        else "comparison"
    )
    is_joint_metric = code is ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED
    return ReasonRecord.from_primitive(
        {
            "code": code.value,
            "role": role,
            "object_type": "joint" if is_joint_metric else None,
            "object_name": "elbow" if is_joint_metric else None,
            "metric": "angle_rad" if is_joint_metric else None,
            "boundary_index": 0 if is_joint_metric else None,
            "evidence": {},
        }
    )


def candidate(
    base: ComparisonReceipt,
    status: ComparisonStatus,
    reasons: tuple[ReasonRecord, ...] = (),
    *,
    threadpool: EngineThreadpoolState = EngineThreadpoolState.DISABLED,
    tool_version: str | None = None,
) -> ComparisonReceipt:
    """Construct the candidate fixture used by receipt hash scenarios.

    Deterministic setup isolates receipt hash without bypassing the contract boundary under
    assertion.
    """
    tool = base.tool if tool_version is None else replace(base.tool, version=tool_version)
    environment = replace(base.environment, engine_threadpool_state=threadpool)
    return replace(
        base,
        tool=tool,
        status=status,
        reasons=reasons,
        reason_codes=(),
        environment=environment,
        receipt_sha256=None,
    )


@pytest.mark.parametrize(
    ("status", "reason_code", "threadpool"),
    [
        (
            ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD,
            None,
            EngineThreadpoolState.DISABLED,
        ),
        (
            ComparisonStatus.COVERAGE_INSUFFICIENT,
            ReasonCode.INITIAL_STATE_NOT_PRESERVED,
            EngineThreadpoolState.DISABLED,
        ),
        (
            ComparisonStatus.NONDETERMINISTIC_REPLAY,
            ReasonCode.BASELINE_NONDETERMINISTIC,
            EngineThreadpoolState.DISABLED,
        ),
        (
            ComparisonStatus.MATERIAL_BEHAVIOR_CHANGE,
            ReasonCode.CANDIDATE_NONFINITE_STATE,
            EngineThreadpoolState.DISABLED,
        ),
        (
            ComparisonStatus.COVERAGE_INSUFFICIENT,
            ReasonCode.ENGINE_THREADPOOL_ACTIVE,
            EngineThreadpoolState.ACTIVE,
        ),
        (
            ComparisonStatus.COVERAGE_INSUFFICIENT,
            ReasonCode.ENGINE_THREADPOOL_STATE_UNKNOWN,
            EngineThreadpoolState.UNKNOWN,
        ),
    ],
)
def test_valid_minimum_receipt_for_each_status_and_threadpool(
    green_candidate: ComparisonReceipt,
    status: ComparisonStatus,
    reason_code: ReasonCode | None,
    threadpool: EngineThreadpoolState,
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises valid minimum receipt for each status and threadpool; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    reasons = () if reason_code is None else (reason(reason_code),)
    draft = candidate(green_candidate, status, reasons, threadpool=threadpool)
    assert draft.receipt_sha256 is None
    with pytest.raises(ValueError, match="receipt_sha256"):
        draft.to_primitive()
    finalized = finalize_receipt(draft)
    assert draft.receipt_sha256 is None
    assert finalized.status is status
    assert finalized.receipt_sha256 is not None
    assert finalized.environment.environment_sha256 is not None
    assert finalized.alignment.alignment_sha256 is not None
    assert finalized.limitations == tuple(LimitationCode)
    assert validate_receipt(finalized) is finalized
    assert ComparisonReceipt.from_primitive(finalized.to_primitive()) == finalized


def test_historical_producer_version_parses(green_candidate: ComparisonReceipt) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises historical producer version parses; accepting a contradictory or
    noncanonical value would make the signed decision evidence ambiguous.
    """
    finalized = finalize_receipt(
        candidate(
            green_candidate,
            ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD,
            tool_version="0.0.7-historical",
        )
    )
    parsed = ComparisonReceipt.from_primitive(finalized.to_primitive())
    assert parsed.tool.version == "0.0.7-historical"


def test_reason_ordering_and_projection_are_recomputed(
    green_candidate: ComparisonReceipt,
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises reason ordering and projection are recomputed; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    reasons = (
        reason(ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED),
        reason(ReasonCode.TRACE_MALFORMED),
        reason(ReasonCode.INITIAL_STATE_NOT_PRESERVED),
    )
    finalized = finalize_receipt(
        candidate(green_candidate, ComparisonStatus.COVERAGE_INSUFFICIENT, reasons)
    )
    assert finalized.reason_codes == (
        ReasonCode.INITIAL_STATE_NOT_PRESERVED,
        ReasonCode.TRACE_MALFORMED,
        ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED,
    )


def test_duplicate_reason_records_refuse_before_finalization(
    green_candidate: ComparisonReceipt,
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises duplicate reason records refuse before finalization; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    item = reason(ReasonCode.TRACE_MALFORMED)
    with pytest.raises(ValueError, match="duplicate"):
        candidate(green_candidate, ComparisonStatus.COVERAGE_INSUFFICIENT, (item, item))


def test_one_decision_bearing_mutation_changes_receipt_hash(
    green_candidate: ComparisonReceipt,
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises one decision bearing mutation changes receipt hash; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    base = candidate(
        green_candidate,
        ComparisonStatus.MATERIAL_BEHAVIOR_CHANGE,
        (reason(ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED),),
    )
    first = finalize_receipt(base)
    second = finalize_receipt(
        replace(base, metrics=MetricEvidenceSummary.from_primitive({"decision": "changed"}))
    )
    assert first.receipt_sha256 != second.receipt_sha256


def test_finalize_refuses_already_hashed_and_wrong_type(
    green_candidate: ComparisonReceipt,
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises finalize refuses already hashed and wrong type; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    finalized = finalize_receipt(green_candidate)
    with pytest.raises(ValueError, match="unhashed"):
        finalize_receipt(finalized)
    with pytest.raises(TypeError):
        finalize_receipt(cast(Any, {}))
    with pytest.raises(TypeError):
        validate_receipt(cast(Any, {}))


def test_validate_never_repairs_reason_order_or_projection(
    green_candidate: ComparisonReceipt,
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises validate never repairs reason order or projection; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    finalized = finalize_receipt(
        candidate(
            green_candidate,
            ComparisonStatus.COVERAGE_INSUFFICIENT,
            (reason(ReasonCode.TRACE_MALFORMED), reason(ReasonCode.INITIAL_STATE_NOT_PRESERVED)),
        )
    )
    with pytest.raises(ValueError, match="order"):
        replace(finalized, reasons=tuple(reversed(finalized.reasons)))
    with pytest.raises(ValueError, match="projection"):
        replace(finalized, reason_codes=())


def test_status_and_reason_inconsistency_refuses(
    green_candidate: ComparisonReceipt,
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises status and reason inconsistency refuses; accepting a contradictory
    or noncanonical value would make the signed decision evidence ambiguous.
    """
    with pytest.raises(ValueError, match="status"):
        finalize_receipt(
            candidate(
                green_candidate,
                ComparisonStatus.MATERIAL_BEHAVIOR_CHANGE,
                (reason(ReasonCode.INITIAL_STATE_NOT_PRESERVED),),
            )
        )
    with pytest.raises(ValueError, match="green"):
        finalize_receipt(
            candidate(
                green_candidate,
                ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD,
                (reason(ReasonCode.INITIAL_STATE_NOT_PRESERVED),),
            )
        )
    with pytest.raises(ValueError, match="non-green"):
        finalize_receipt(candidate(green_candidate, ComparisonStatus.COVERAGE_INSUFFICIENT))


def test_strict_parser_rejects_bad_outer_and_nested_hashes(
    green_candidate: ComparisonReceipt,
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises strict parser rejects bad outer and nested hashes; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    finalized = finalize_receipt(green_candidate)
    primitive = finalized.to_primitive()
    with pytest.raises(ValueError):
        ComparisonReceipt.from_primitive({**primitive, "receipt_sha256": digest("wrong")})

    bad_env = dict(primitive)
    bad_env["environment"] = {
        **cast(dict[str, object], primitive["environment"]),
        "environment_sha256": digest("wrong-env"),
    }
    with pytest.raises(ValueError):
        ComparisonReceipt.from_primitive(bad_env)

    bad_alignment = dict(primitive)
    bad_alignment["alignment"] = {
        **cast(dict[str, object], primitive["alignment"]),
        "alignment_sha256": digest("wrong-alignment"),
    }
    with pytest.raises(ValueError):
        ComparisonReceipt.from_primitive(bad_alignment)

    bad_closure = dict(primitive)
    bad_inputs = dict(cast(dict[str, object], primitive["inputs"]))
    bad_inputs["baseline_model_closure_sha256"] = digest("wrong-closure")
    bad_closure["inputs"] = bad_inputs
    with pytest.raises(ValueError, match="baseline model closure"):
        ComparisonReceipt.from_primitive(bad_closure)

    bad_time = dict(primitive)
    time = dict(cast(dict[str, object], primitive["time"]))
    time["state_samples"] = 2
    bad_time["time"] = time
    with pytest.raises(ValueError, match="state_samples"):
        ComparisonReceipt.from_primitive(bad_time)


def test_strict_parser_rejects_semantic_inconsistencies(
    green_candidate: ComparisonReceipt,
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises strict parser rejects semantic inconsistencies; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    finalized = finalize_receipt(
        candidate(
            green_candidate,
            ComparisonStatus.COVERAGE_INSUFFICIENT,
            (reason(ReasonCode.INITIAL_STATE_NOT_PRESERVED),),
        )
    )
    primitive = finalized.to_primitive()
    with pytest.raises(ValueError, match="status"):
        ComparisonReceipt.from_primitive({**primitive, "status": "MATERIAL_BEHAVIOR_CHANGE"})
    with pytest.raises(ValueError, match="projection"):
        ComparisonReceipt.from_primitive({**primitive, "reason_codes": []})
    with pytest.raises(ValueError):
        ComparisonReceipt.from_primitive({**primitive, "receipt_sha256": None})


def test_closure_tolerance_and_alias_binding_refuse(
    green_candidate: ComparisonReceipt,
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises closure tolerance and alias binding refuse; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    finalized = finalize_receipt(green_candidate)
    wrong_candidate_inputs = replace(
        finalized.inputs,
        candidate_model_closure_sha256=digest("wrong-candidate-closure"),
    )
    with pytest.raises(ValueError, match="candidate model closure"):
        replace(finalized, inputs=wrong_candidate_inputs)
    with pytest.raises(ValueError, match="tolerances"):
        replace(finalized, tolerances=CanonicalSummary.from_primitive({}))

    inputs = finalized.inputs.to_primitive()
    inputs["aliases_raw_sha256"] = digest("raw")
    inputs["aliases_semantic_sha256"] = None
    with pytest.raises(ValueError):
        ComparisonInputsIdentity.from_primitive(inputs)


def test_empty_monitored_set_refuses_under_every_status(
    green_candidate: ComparisonReceipt,
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises empty monitored set refuses under every status; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    status_reasons = {
        ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD: (),
        ComparisonStatus.COVERAGE_INSUFFICIENT: (reason(ReasonCode.INITIAL_STATE_NOT_PRESERVED),),
        ComparisonStatus.NONDETERMINISTIC_REPLAY: (reason(ReasonCode.BASELINE_NONDETERMINISTIC),),
        ComparisonStatus.MATERIAL_BEHAVIOR_CHANGE: (
            reason(ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED),
        ),
    }
    for status, reasons in status_reasons.items():
        with pytest.raises(ValueError, match="monitored_joints"):
            replace(candidate(green_candidate, status, reasons), monitored_joints=())


def test_threadpool_state_reason_binding_is_honest(
    green_candidate: ComparisonReceipt,
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises threadpool state reason binding is honest; accepting a contradictory
    or noncanonical value would make the signed decision evidence ambiguous.
    """
    active = candidate(
        green_candidate,
        ComparisonStatus.COVERAGE_INSUFFICIENT,
        (reason(ReasonCode.ENGINE_THREADPOOL_ACTIVE),),
        threadpool=EngineThreadpoolState.ACTIVE,
    )
    assert (
        finalize_receipt(active).environment.engine_threadpool_state is EngineThreadpoolState.ACTIVE
    )

    unknown = candidate(
        green_candidate,
        ComparisonStatus.COVERAGE_INSUFFICIENT,
        (reason(ReasonCode.ENGINE_THREADPOOL_STATE_UNKNOWN),),
        threadpool=EngineThreadpoolState.UNKNOWN,
    )
    assert (
        finalize_receipt(unknown).environment.engine_threadpool_state
        is EngineThreadpoolState.UNKNOWN
    )

    invalid_cases = [
        candidate(
            green_candidate,
            ComparisonStatus.COVERAGE_INSUFFICIENT,
            (reason(ReasonCode.ENGINE_THREADPOOL_ACTIVE),),
        ),
        candidate(
            green_candidate,
            ComparisonStatus.COVERAGE_INSUFFICIENT,
            (reason(ReasonCode.ENGINE_THREADPOOL_STATE_UNKNOWN),),
            threadpool=EngineThreadpoolState.ACTIVE,
        ),
        candidate(
            green_candidate,
            ComparisonStatus.COVERAGE_INSUFFICIENT,
            (reason(ReasonCode.INITIAL_STATE_NOT_PRESERVED),),
            threadpool=EngineThreadpoolState.UNKNOWN,
        ),
    ]
    for invalid in invalid_cases:
        with pytest.raises(ValueError, match="threadpool"):
            finalize_receipt(invalid)


def test_receipt_from_primitive_unknown_missing_and_enum_fields_refuse(
    green_candidate: ComparisonReceipt,
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises receipt from primitive unknown missing and enum fields refuse;
    accepting a contradictory or noncanonical value would make the signed decision evidence
    ambiguous.
    """
    primitive = finalize_receipt(green_candidate).to_primitive()
    with pytest.raises(ValueError, match="unknown fields"):
        ComparisonReceipt.from_primitive({**primitive, "extra": 1})
    missing = dict(primitive)
    del missing["metrics"]
    with pytest.raises(ValueError, match="missing fields"):
        ComparisonReceipt.from_primitive(missing)
    with pytest.raises(ValueError, match="unknown comparison status"):
        ComparisonReceipt.from_primitive({**primitive, "status": "UNKNOWN"})
    with pytest.raises(ValueError, match="unknown reason code"):
        ComparisonReceipt.from_primitive({**primitive, "reason_codes": ["UNKNOWN"]})
    with pytest.raises(ValueError):
        ComparisonReceipt.from_primitive({**primitive, "limitations": ["UNKNOWN"] * 4})


def test_receipt_direct_constructor_nested_type_invariants(
    green_candidate: ComparisonReceipt,
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises receipt direct constructor nested type invariants; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    mutations: list[tuple[str, object]] = [
        ("tool", object()),
        ("status", "wrong"),
        ("reason_codes", []),
        ("reasons", []),
        ("environment", object()),
        ("inputs", object()),
        ("model_closures", object()),
        ("time", object()),
        ("alignment", object()),
        ("monitored_joints", []),
        ("tolerances", object()),
        ("repeatability", object()),
        ("numerical_evidence", object()),
        ("metrics", object()),
        ("first_crossing", object()),
        ("limitations", []),
    ]
    for field, value in mutations:
        with pytest.raises((TypeError, ValueError)) as exc_info:
            replace(green_candidate, **{field: value})
        assert not isinstance(exc_info.value, AttributeError)


def test_receipt_direct_schema_invariants(green_candidate: ComparisonReceipt) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises receipt direct schema invariants; accepting a contradictory or
    noncanonical value would make the signed decision evidence ambiguous.
    """
    with pytest.raises(ValueError, match="schema_version"):
        replace(green_candidate, schema_version="wrong")
    with pytest.raises(ValueError, match="status_rule_schema"):
        replace(green_candidate, status_rule_schema="wrong")
    with pytest.raises(ValueError, match="status_rule_schema_version"):
        replace(green_candidate, status_rule_schema_version=2)
    joint = green_candidate.monitored_joints[0]
    with pytest.raises(ValueError, match="monitored_joints"):
        replace(green_candidate, monitored_joints=(joint, joint))
    with pytest.raises(ValueError, match="duplicates"):
        replace(
            green_candidate,
            limitations=(green_candidate.limitations[0], green_candidate.limitations[0]),
        )
