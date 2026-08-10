"""comparison hinge/slide metric evaluation, repeatability qualification, and reason production."""

# Keep the baseline import statement order for rename-normalized AST identity.
from __future__ import annotations  # noqa: I001

from dataclasses import dataclass
from typing import cast

from .._model_closure import AlignedJoint
from .._timegrid import TimeGrid
from ..errors import ReasonCode, ReasonRecord, ordered_reasons
from ..json_values import (
    Binary64,
    CanonicalValue,
)
from ..schemas import MonitoredJoint
from ._metric_math import (
    _absolute_errors,
    _canonical_offsets,
    _metric_specs,
    _build_metric_reason_record,
    _summarize_metric,
)
from ._trace import RoleRepeatSet, RoleTrace


@dataclass(frozen=True, slots=True)
class MetricEvaluation:
    """Carry complete reason, metric, witness, and first-crossing evidence for two roles."""

    reasons: tuple[ReasonRecord, ...]
    repeatability: dict[str, CanonicalValue]
    numerical_evidence: dict[str, CanonicalValue]
    metrics: dict[str, CanonicalValue]
    first_crossing: dict[str, CanonicalValue] | None


def evaluate_role_pair(
    *,
    baseline: RoleRepeatSet,
    candidate: RoleRepeatSet,
    joints: tuple[AlignedJoint, ...],
    monitored_joints: tuple[MonitoredJoint, ...],
    time_grid: TimeGrid,
) -> MetricEvaluation:
    """Produce complete comparison reasons and bounded metric summaries."""
    reasons = _stability_reasons(baseline, candidate)
    baseline_trace = baseline.representative
    candidate_trace = candidate.representative
    reasons.extend(_numerical_reasons(baseline_trace))
    reasons.extend(_numerical_reasons(candidate_trace))
    reasons.extend(_trace_integrity_reasons(baseline_trace, candidate_trace, time_grid))
    metrics, first_crossing, metric_reasons = _metric_evidence(
        baseline, candidate, joints, monitored_joints, time_grid
    )
    reasons.extend(metric_reasons)
    return MetricEvaluation(
        reasons=ordered_reasons(reasons),
        repeatability={
            "schema": "metrifid.repeatability_evidence",
            "schema_version": 1,
            "baseline": _repeat_summary(baseline),
            "candidate": _repeat_summary(candidate),
        },
        numerical_evidence={
            "schema": "metrifid.numerical_evidence",
            "schema_version": 1,
            "baseline": _trace_numerical_summary(baseline_trace),
            "candidate": _trace_numerical_summary(candidate_trace),
        },
        metrics=metrics,
        first_crossing=first_crossing,
    )


def _stability_reasons(baseline: RoleRepeatSet, candidate: RoleRepeatSet) -> list[ReasonRecord]:
    """Return deterministic role-local repeatability reasons in role order."""
    reasons: list[ReasonRecord] = []
    if not baseline.stable:
        reasons.append(
            _build_metric_reason_record(
                ReasonCode.BASELINE_NONDETERMINISTIC,
                "baseline",
                evidence={"signatures": list(baseline.signatures)},
            )
        )
    if not candidate.stable:
        reasons.append(
            _build_metric_reason_record(
                ReasonCode.CANDIDATE_NONDETERMINISTIC,
                "candidate",
                evidence={"signatures": list(candidate.signatures)},
            )
        )
    return reasons


def _metric_evidence(
    baseline: RoleRepeatSet,
    candidate: RoleRepeatSet,
    joints: tuple[AlignedJoint, ...],
    monitored: tuple[MonitoredJoint, ...],
    time_grid: TimeGrid,
) -> tuple[dict[str, CanonicalValue], dict[str, CanonicalValue] | None, list[ReasonRecord]]:
    """Evaluate joint metrics only when repeatability and complete-trace requirements hold."""
    baseline_trace = baseline.representative
    candidate_trace = candidate.representative
    eligible = (
        baseline.stable
        and candidate.stable
        and baseline_trace.complete
        and candidate_trace.complete
        and baseline_trace.boundary_indices == candidate_trace.boundary_indices
    )
    if eligible:
        return _joint_metrics(baseline_trace, candidate_trace, joints, monitored, time_grid)
    return (
        {
            "schema": "metrifid.metric_evidence",
            "schema_version": 1,
            "evaluated": False,
            "reason": "repeatability_or_complete_trace_requirement_not_met",
            "joints": [],
        },
        None,
        [],
    )


def _repeat_summary(value: RoleRepeatSet) -> dict[str, CanonicalValue]:
    """Summarize repeat signatures and stability for one role."""
    return {
        "repeat_count": len(value.traces),
        "stable": value.stable,
        "signatures": list(value.signatures),
        "complete_repeats": sum(1 for trace in value.traces if trace.complete),
        "captured_boundary_counts": [len(trace.boundary_indices) for trace in value.traces],
    }


def _trace_numerical_summary(trace: RoleTrace) -> dict[str, CanonicalValue]:
    """Summarize completion, warnings, invalid state, and captured boundary count."""
    first_warning: dict[str, CanonicalValue] | None = None
    for boundary, snapshot in zip(trace.boundary_indices, trace.warning_snapshots, strict=True):
        if snapshot:
            first_warning = {
                "boundary_index": boundary,
                "warnings": [
                    {"warning_index": item[0], "number": item[1], "lastinfo": item[2]}
                    for item in snapshot
                ],
            }
            break
    return {
        "complete": trace.complete,
        "initial_state_preserved": trace.initial_state_preserved,
        "captured_boundary_count": len(trace.boundary_indices),
        "expected_boundary_count": trace.expected_boundary_count,
        "invalid_kind": trace.invalid_kind,
        "invalid_boundary_index": trace.invalid_boundary_index,
        "first_warning": first_warning,
        "error_logs": list(trace.error_logs),
    }


def _numerical_reasons(trace: RoleTrace) -> list[ReasonRecord]:
    """Build role-local warning, invalid-state, and early-termination reasons."""
    role = trace.role
    reasons: list[ReasonRecord] = []
    if not trace.initial_state_preserved:
        reasons.append(
            _build_metric_reason_record(
                ReasonCode.INITIAL_STATE_NOT_PRESERVED,
                role,
                boundary_index=0,
                evidence={"captured_boundary_count": len(trace.boundary_indices)},
            )
        )
    invalid = _invalid_state_reason(trace)
    if invalid is not None:
        reasons.append(invalid)
    warning = _warning_reason(trace)
    if warning is not None:
        reasons.append(warning)
    termination = _early_termination_reason(trace)
    if termination is not None:
        reasons.append(termination)
    return reasons


def _invalid_state_reason(trace: RoleTrace) -> ReasonRecord | None:
    """Return the role-specific invalid-state reason, when the trace records one."""
    code = {
        ("baseline", "NONFINITE_STATE"): ReasonCode.BASELINE_NONFINITE_STATE,
        ("baseline", "INVALID_QUATERNION"): ReasonCode.BASELINE_INVALID_QUATERNION,
        ("baseline", "NUMERICAL_ERROR_LOG"): ReasonCode.BASELINE_NUMERICAL_ERROR_LOG,
        ("candidate", "NONFINITE_STATE"): ReasonCode.CANDIDATE_NONFINITE_STATE,
        ("candidate", "INVALID_QUATERNION"): ReasonCode.CANDIDATE_INVALID_QUATERNION,
        ("candidate", "NUMERICAL_ERROR_LOG"): ReasonCode.CANDIDATE_NUMERICAL_ERROR_LOG,
    }.get(cast("tuple[str, str]", (trace.role, trace.invalid_kind)))
    if code is None:
        return None
    return _build_metric_reason_record(
        code,
        trace.role,
        boundary_index=trace.invalid_boundary_index,
        evidence={"invalid_kind": cast(str, trace.invalid_kind)},
    )


def _warning_reason(trace: RoleTrace) -> ReasonRecord | None:
    """Return the first role-specific MuJoCo warning reason, when present."""
    code = (
        ReasonCode.BASELINE_MUJOCO_WARNING
        if trace.role == "baseline"
        else ReasonCode.CANDIDATE_MUJOCO_WARNING
    )
    for boundary, snapshot in zip(trace.boundary_indices, trace.warning_snapshots, strict=True):
        if snapshot:
            return _build_metric_reason_record(
                code,
                trace.role,
                boundary_index=boundary,
                evidence={
                    "warnings": [
                        {"warning_index": item[0], "number": item[1], "lastinfo": item[2]}
                        for item in snapshot
                    ]
                },
            )
    return None


def _early_termination_reason(trace: RoleTrace) -> ReasonRecord | None:
    """Return the role-specific early-termination reason for an incomplete trace."""
    if trace.complete:
        return None
    code = (
        ReasonCode.BASELINE_EARLY_TERMINATION
        if trace.role == "baseline"
        else ReasonCode.CANDIDATE_EARLY_TERMINATION
    )
    return _build_metric_reason_record(
        code,
        trace.role,
        boundary_index=trace.invalid_boundary_index,
        evidence={
            "captured_boundary_count": len(trace.boundary_indices),
            "expected_boundary_count": trace.expected_boundary_count,
        },
    )


def _trace_integrity_reasons(
    baseline: RoleTrace,
    candidate: RoleTrace,
    time_grid: TimeGrid,
) -> list[ReasonRecord]:
    """Cross-check retained boundary counts, times, and channel layouts across roles."""
    result: list[ReasonRecord] = []
    result.extend(_trace_count_reasons(baseline, candidate, time_grid))
    result.extend(_trace_time_reasons(baseline, candidate, time_grid))
    layout = _trace_layout_reason(baseline, candidate)
    if layout is not None:
        result.append(layout)
    return result


def _trace_count_reasons(
    baseline: RoleTrace, candidate: RoleTrace, time_grid: TimeGrid
) -> list[ReasonRecord]:
    """Return sample-count and boundary-index trace integrity reasons."""
    result: list[ReasonRecord] = []
    counts = (len(baseline.boundary_indices), len(candidate.boundary_indices))
    if counts != (time_grid.boundary_count, time_grid.boundary_count):
        result.append(
            _build_metric_reason_record(
                ReasonCode.TRACE_SAMPLE_COUNT_MISMATCH,
                "comparison",
                evidence={
                    "baseline_count": counts[0],
                    "candidate_count": counts[1],
                    "expected_count": time_grid.boundary_count,
                },
            )
        )
    if baseline.boundary_indices != candidate.boundary_indices:
        result.append(
            _build_metric_reason_record(
                ReasonCode.TRACE_BOUNDARY_INDEX_MISMATCH,
                "comparison",
                evidence={
                    "baseline": list(baseline.boundary_indices),
                    "candidate": list(candidate.boundary_indices),
                },
            )
        )
    return result


def _trace_time_reasons(
    baseline: RoleTrace, candidate: RoleTrace, time_grid: TimeGrid
) -> list[ReasonRecord]:
    """Return exact time-recurrence reasons in baseline then candidate order."""
    result: list[ReasonRecord] = []
    expected_baseline = tuple(time_grid.iter_role_boundary_time_bits("baseline"))
    expected_candidate = tuple(time_grid.iter_role_boundary_time_bits("candidate"))
    mismatch = _first_time_mismatch(baseline.observed_time_bits, expected_baseline)
    if mismatch is not None:
        result.append(
            _build_metric_reason_record(
                ReasonCode.TRACE_TIME_RECURRENCE_MISMATCH,
                "comparison",
                boundary_index=mismatch,
                evidence={"role": "baseline"},
            )
        )
    mismatch = _first_time_mismatch(candidate.observed_time_bits, expected_candidate)
    if mismatch is not None:
        result.append(
            _build_metric_reason_record(
                ReasonCode.TRACE_TIME_RECURRENCE_MISMATCH,
                "comparison",
                boundary_index=mismatch,
                evidence={"role": "candidate"},
            )
        )
    return result


def _trace_layout_reason(baseline: RoleTrace, candidate: RoleTrace) -> ReasonRecord | None:
    """Return the channel-layout mismatch reason when retained trace widths differ."""
    if (
        baseline.qpos.shape[1:] != candidate.qpos.shape[1:]
        or baseline.qvel.shape[1:] != candidate.qvel.shape[1:]
    ):
        return _build_metric_reason_record(
            ReasonCode.TRACE_CHANNEL_LAYOUT_MISMATCH,
            "comparison",
            evidence={
                "baseline_qpos_width": baseline.qpos.shape[1],
                "candidate_qpos_width": candidate.qpos.shape[1],
                "baseline_qvel_width": baseline.qvel.shape[1],
                "candidate_qvel_width": candidate.qvel.shape[1],
            },
        )
    return None


def _first_time_mismatch(
    actual: tuple[Binary64, ...], expected: tuple[Binary64, ...]
) -> int | None:
    """Return the first boundary whose exact time bits differ from the schedule."""
    for index, (left, right) in enumerate(zip(actual, expected, strict=False)):
        if left.bits != right.bits:
            return index
    return None


def _joint_metrics(
    baseline: RoleTrace,
    candidate: RoleTrace,
    joints: tuple[AlignedJoint, ...],
    monitored: tuple[MonitoredJoint, ...],
    time_grid: TimeGrid,
) -> tuple[dict[str, CanonicalValue], dict[str, CanonicalValue] | None, list[ReasonRecord]]:
    """Evaluate every configured metric over aligned retained joint traces."""
    offsets = _canonical_offsets(joints)
    by_name = {item.canonical_name: item for item in joints}
    joint_rows: list[CanonicalValue] = []
    reasons: list[ReasonRecord] = []
    crossing_candidates: list[dict[str, CanonicalValue]] = []
    for monitored_joint in monitored:
        row, crossings, joint_reasons = _monitored_joint_metrics(
            baseline,
            candidate,
            by_name[monitored_joint.canonical_name],
            monitored_joint,
            offsets,
            time_grid,
        )
        joint_rows.append(row)
        crossing_candidates.extend(crossings)
        reasons.extend(joint_reasons)
    first = min(
        crossing_candidates,
        key=lambda item: (
            cast(int, item["boundary_index"]),
            cast(str, item["joint_name"]),
            cast(str, item["metric"]),
        ),
        default=None,
    )
    return (
        {
            "schema": "metrifid.metric_evidence",
            "schema_version": 1,
            "evaluated": True,
            "compared_boundary_count": time_grid.boundary_count,
            "joints": joint_rows,
        },
        first,
        reasons,
    )


def _monitored_joint_metrics(
    baseline: RoleTrace,
    candidate: RoleTrace,
    aligned: AlignedJoint,
    monitored: MonitoredJoint,
    offsets: dict[str, tuple[int, int]],
    time_grid: TimeGrid,
) -> tuple[CanonicalValue, list[dict[str, CanonicalValue]], list[ReasonRecord]]:
    """Evaluate every declared metric for one aligned monitored joint."""
    qpos_offset, qvel_offset = offsets[aligned.canonical_name]
    metric_rows: dict[str, CanonicalValue] = {}
    crossings: list[dict[str, CanonicalValue]] = []
    reasons: list[ReasonRecord] = []
    for metric_name, source, column, wrapped in _metric_specs(
        aligned.joint_type, qpos_offset, qvel_offset
    ):
        left = baseline.qpos[:, column] if source == "qpos" else baseline.qvel[:, column]
        right = candidate.qpos[:, column] if source == "qpos" else candidate.qvel[:, column]
        values = _absolute_errors(left, right, wrapped=wrapped)
        row, crossing = _summarize_metric(
            values,
            monitored.tolerances[metric_name],
            time_grid,
            monitored.canonical_name,
            metric_name,
        )
        metric_rows[metric_name] = row
        if crossing is not None:
            crossings.append(crossing)
            reasons.append(_metric_crossing_reason(monitored.canonical_name, metric_name, crossing))
    return (
        {
            "canonical_name": monitored.canonical_name,
            "joint_type": monitored.joint_type,
            "metrics": metric_rows,
        },
        crossings,
        reasons,
    )


def _metric_crossing_reason(
    joint_name: str, metric_name: str, crossing: dict[str, CanonicalValue]
) -> ReasonRecord:
    """Build the exact tolerance-exceeded reason for one first metric crossing."""
    return _build_metric_reason_record(
        ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED,
        "comparison",
        object_type="joint",
        object_name=joint_name,
        metric=metric_name,
        boundary_index=cast(int, crossing["boundary_index"]),
        evidence={
            "error": crossing["error"],
            "tolerance": crossing["tolerance"],
            "ratio": crossing["ratio"],
            "time": crossing["time"],
        },
    )
