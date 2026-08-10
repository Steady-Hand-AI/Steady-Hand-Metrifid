"""Frozen comparison statuses, reasons, limitations, and canonical ordering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

from .json_values import (
    CanonicalValue,
    FrozenCanonicalObject,
    canonical_json_bytes,
    freeze_canonical,
    thaw_canonical,
)

ReasonRole: TypeAlias = Literal["baseline", "candidate", "comparison"] | None


class ComparisonStatus(StrEnum):
    """The complete comparison-status registry."""

    MATERIAL_BEHAVIOR_CHANGE = "MATERIAL_BEHAVIOR_CHANGE"
    NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD = "NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD"
    COVERAGE_INSUFFICIENT = "COVERAGE_INSUFFICIENT"
    NONDETERMINISTIC_REPLAY = "NONDETERMINISTIC_REPLAY"


class OperationalExitCode(IntEnum):
    """The complete process exit-code registry."""

    NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD = 0
    MATERIAL_BEHAVIOR_CHANGE = 10
    COVERAGE_INSUFFICIENT = 20
    NONDETERMINISTIC_REPLAY = 30
    INVALID_INVOCATION_INPUT_OUTPUT = 64
    INTERNAL_PROJECT_FAILURE = 70


class EngineThreadpoolState(StrEnum):
    """Observed MuJoCo data-threadpool state."""

    DISABLED = "DISABLED"
    ACTIVE = "ACTIVE"
    UNKNOWN = "UNKNOWN"


class LimitationCode(StrEnum):
    """The comparison-claim limitation registry, in canonical order."""

    DECLARED_WORKLOAD_ONLY = "DECLARED_WORKLOAD_ONLY"
    MONITORED_JOINT_COORDINATES_ONLY = "MONITORED_JOINT_COORDINATES_ONLY"
    NO_BODY_CONTACT_SENSOR_REWARD_OR_TASK_CLAIM = "NO_BODY_CONTACT_SENSOR_REWARD_OR_TASK_CLAIM"
    NO_GLOBAL_EQUIVALENCE_CLAIM = "NO_GLOBAL_EQUIVALENCE_CLAIM"


class ReasonCode(StrEnum):
    """Comparison-only reason codes in their frozen registry order."""

    ENGINE_THREADPOOL_ACTIVE = "ENGINE_THREADPOOL_ACTIVE"
    ENGINE_THREADPOOL_STATE_UNKNOWN = "ENGINE_THREADPOOL_STATE_UNKNOWN"
    INITIAL_STATE_NOT_PRESERVED = "INITIAL_STATE_NOT_PRESERVED"
    INTERNAL_STEP_BUDGET_EXCEEDED = "INTERNAL_STEP_BUDGET_EXCEEDED"
    TRACE_MEMORY_BUDGET_EXCEEDED = "TRACE_MEMORY_BUDGET_EXCEEDED"
    COMPARISON_TIMEOUT = "COMPARISON_TIMEOUT"
    BASELINE_NONDETERMINISTIC = "BASELINE_NONDETERMINISTIC"
    CANDIDATE_NONDETERMINISTIC = "CANDIDATE_NONDETERMINISTIC"
    BASELINE_NONFINITE_STATE = "BASELINE_NONFINITE_STATE"
    BASELINE_INVALID_QUATERNION = "BASELINE_INVALID_QUATERNION"
    BASELINE_MUJOCO_WARNING = "BASELINE_MUJOCO_WARNING"
    BASELINE_NUMERICAL_ERROR_LOG = "BASELINE_NUMERICAL_ERROR_LOG"
    BASELINE_EARLY_TERMINATION = "BASELINE_EARLY_TERMINATION"
    CANDIDATE_NONFINITE_STATE = "CANDIDATE_NONFINITE_STATE"
    CANDIDATE_INVALID_QUATERNION = "CANDIDATE_INVALID_QUATERNION"
    CANDIDATE_MUJOCO_WARNING = "CANDIDATE_MUJOCO_WARNING"
    CANDIDATE_NUMERICAL_ERROR_LOG = "CANDIDATE_NUMERICAL_ERROR_LOG"
    CANDIDATE_EARLY_TERMINATION = "CANDIDATE_EARLY_TERMINATION"
    TRACE_SAMPLE_COUNT_MISMATCH = "TRACE_SAMPLE_COUNT_MISMATCH"
    TRACE_BOUNDARY_INDEX_MISMATCH = "TRACE_BOUNDARY_INDEX_MISMATCH"
    TRACE_TIME_RECURRENCE_MISMATCH = "TRACE_TIME_RECURRENCE_MISMATCH"
    TRACE_CHANNEL_LAYOUT_MISMATCH = "TRACE_CHANNEL_LAYOUT_MISMATCH"
    TRACE_MALFORMED = "TRACE_MALFORMED"
    JOINT_METRIC_TOLERANCE_EXCEEDED = "JOINT_METRIC_TOLERANCE_EXCEEDED"


_REASON_ALLOWED_ROLES: Mapping[ReasonCode, frozenset[ReasonRole]] = MappingProxyType(
    {
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
)


@dataclass(frozen=True, slots=True)
class StatusRule:
    """One immutable row in the frozen status-precedence table."""

    rule_id: str
    rank: int
    status: ComparisonStatus
    meaning: str


@dataclass(frozen=True, slots=True)
class ReasonRule:
    """One immutable comparison-reason to status-rule binding."""

    code: ReasonCode
    status_rule_id: str


def _validate_metric_reason_subject(reason: ReasonRecord) -> None:
    """Validate fields required by a joint metric tolerance reason."""
    if reason.code is not ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED:
        return
    if reason.object_type != "joint":
        raise ValueError("JOINT_METRIC_TOLERANCE_EXCEEDED requires object_type joint")
    if reason.object_name is None:
        raise ValueError("JOINT_METRIC_TOLERANCE_EXCEEDED requires object_name")
    if reason.metric is None:
        raise ValueError("JOINT_METRIC_TOLERANCE_EXCEEDED requires metric")
    if reason.boundary_index is None:
        raise ValueError("JOINT_METRIC_TOLERANCE_EXCEEDED requires boundary_index")


def _strict_reason_evidence(evidence: FrozenCanonicalObject) -> FrozenCanonicalObject:
    """Return evidence after strict canonical round-trip validation."""
    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be a canonical object")
    try:
        thawed = thaw_canonical(evidence)
        if type(thawed) is not dict:
            raise TypeError("evidence must be a canonical object")
        frozen = freeze_canonical(cast(CanonicalValue, thawed))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TypeError("evidence must be a strict canonical object") from exc
    if not isinstance(frozen, Mapping):
        raise TypeError("evidence must be a canonical object")
    return frozen


@dataclass(frozen=True, slots=True)
class ReasonRecord:
    """One immutable, strictly typed comparison reason record."""

    code: ReasonCode
    role: ReasonRole
    object_type: str | None
    object_name: str | None
    metric: str | None
    boundary_index: int | None
    evidence: FrozenCanonicalObject

    def __post_init__(self) -> None:
        """Validate registry-bound subjects and immutably canonicalize reason evidence."""
        if not isinstance(self.code, ReasonCode):
            raise TypeError("code must be a ReasonCode")
        if self.role not in {None, "baseline", "candidate", "comparison"}:
            raise ValueError("role is outside the frozen role registry")
        if self.role not in _REASON_ALLOWED_ROLES[self.code]:
            raise ValueError(f"role is not allowed for {self.code.value}")
        for field_name in ("object_type", "object_name", "metric"):
            _optional_nonempty_string(getattr(self, field_name), field_name)
        if self.boundary_index is not None:
            if type(self.boundary_index) is not int:
                raise TypeError("boundary_index must be an integer or null")
            if self.boundary_index < 0:
                raise ValueError("boundary_index must be nonnegative")
        _validate_metric_reason_subject(self)
        object.__setattr__(self, "evidence", _strict_reason_evidence(self.evidence))

    @classmethod
    def from_primitive(cls, value: object) -> ReasonRecord:
        """Parse the exact seven-field comparison reason object."""
        obj = _require_exact_object_fields(
            value,
            {
                "code",
                "role",
                "object_type",
                "object_name",
                "metric",
                "boundary_index",
                "evidence",
            },
            "ReasonRecord",
        )
        token = _nonempty_string(obj["code"], "code")
        try:
            code = ReasonCode(token)
        except ValueError as exc:
            raise ValueError(f"unknown comparison reason code: {token}") from exc
        role_raw = obj["role"]
        if role_raw not in {None, "baseline", "candidate", "comparison"}:
            raise ValueError("role is outside the frozen role registry")
        role = cast(ReasonRole, role_raw)
        boundary_raw = obj["boundary_index"]
        if boundary_raw is None:
            boundary_index = None
        elif type(boundary_raw) is int and boundary_raw >= 0:
            boundary_index = boundary_raw
        else:
            raise TypeError("boundary_index must be a nonnegative integer or null")
        evidence_raw = obj["evidence"]
        if type(evidence_raw) is not dict:
            raise TypeError("evidence must be an object")
        frozen = freeze_canonical(cast(CanonicalValue, evidence_raw))
        if not isinstance(frozen, Mapping):
            raise TypeError("evidence must be an object")
        return cls(
            code=code,
            role=role,
            object_type=_optional_nonempty_string(obj["object_type"], "object_type"),
            object_name=_optional_nonempty_string(obj["object_name"], "object_name"),
            metric=_optional_nonempty_string(obj["metric"], "metric"),
            boundary_index=boundary_index,
            evidence=frozen,
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return the exact canonical comparison-reason object."""
        return {
            "code": self.code.value,
            "role": self.role,
            "object_type": self.object_type,
            "object_name": self.object_name,
            "metric": self.metric,
            "boundary_index": self.boundary_index,
            "evidence": thaw_canonical(self.evidence),
        }


STATUS_PRECEDENCE: tuple[StatusRule, ...] = (
    StatusRule(
        "COVERAGE_PREEXECUTION",
        1,
        ComparisonStatus.COVERAGE_INSUFFICIENT,
        "unsupported or underspecified after a complete comparison contract",
    ),
    StatusRule(
        "NONDETERMINISTIC_ROLE",
        2,
        ComparisonStatus.NONDETERMINISTIC_REPLAY,
        "one or both roles lack one stable repeat signature",
    ),
    StatusRule(
        "BASELINE_NUMERICALLY_INVALID",
        3,
        ComparisonStatus.COVERAGE_INSUFFICIENT,
        "repeatable baseline cannot support a materiality comparison",
    ),
    StatusRule(
        "CANDIDATE_NUMERICAL_REGRESSION",
        4,
        ComparisonStatus.MATERIAL_BEHAVIOR_CHANGE,
        "clean baseline and repeatably numerically invalid candidate",
    ),
    StatusRule(
        "TRACE_INTEGRITY_FAILURE",
        5,
        ComparisonStatus.COVERAGE_INSUFFICIENT,
        "trace cannot be bound to the exact declared schedule or layout",
    ),
    StatusRule(
        "JOINT_METRIC_CROSSING",
        6,
        ComparisonStatus.MATERIAL_BEHAVIOR_CHANGE,
        "supported monitored metric strictly exceeds tolerance",
    ),
    StatusRule(
        "GREEN_DECLARED_WORKLOAD",
        7,
        ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD,
        "all earlier rules cleared",
    ),
)

_REASON_GROUPS: Mapping[str, tuple[ReasonCode, ...]] = MappingProxyType(
    {
        "COVERAGE_PREEXECUTION": tuple(ReasonCode)[0:6],
        "NONDETERMINISTIC_ROLE": tuple(ReasonCode)[6:8],
        "BASELINE_NUMERICALLY_INVALID": tuple(ReasonCode)[8:13],
        "CANDIDATE_NUMERICAL_REGRESSION": tuple(ReasonCode)[13:18],
        "TRACE_INTEGRITY_FAILURE": tuple(ReasonCode)[18:23],
        "JOINT_METRIC_CROSSING": tuple(ReasonCode)[23:24],
    }
)

REASON_REGISTRY: Mapping[ReasonCode, ReasonRule] = MappingProxyType(
    {code: ReasonRule(code, rule_id) for rule_id, codes in _REASON_GROUPS.items() for code in codes}
)

_STATUS_BY_ID: Mapping[str, StatusRule] = MappingProxyType(
    {rule.rule_id: rule for rule in STATUS_PRECEDENCE}
)
_REASON_RANK: Mapping[ReasonCode, int] = MappingProxyType(
    {code: rank for rank, code in enumerate(ReasonCode)}
)
_ROLE_ORDER: Mapping[ReasonRole, int] = MappingProxyType(
    {"baseline": 0, "candidate": 1, "comparison": 2, None: 3}
)
_STATUS_EXIT_CODES: Mapping[ComparisonStatus, OperationalExitCode] = MappingProxyType(
    {
        ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD: (
            OperationalExitCode.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD
        ),
        ComparisonStatus.MATERIAL_BEHAVIOR_CHANGE: OperationalExitCode.MATERIAL_BEHAVIOR_CHANGE,
        ComparisonStatus.COVERAGE_INSUFFICIENT: OperationalExitCode.COVERAGE_INSUFFICIENT,
        ComparisonStatus.NONDETERMINISTIC_REPLAY: OperationalExitCode.NONDETERMINISTIC_REPLAY,
    }
)


def status_exit_code(status: ComparisonStatus) -> OperationalExitCode:
    """Map every public comparison status to its frozen process exit code."""
    if not isinstance(status, ComparisonStatus):
        raise TypeError("status must be a ComparisonStatus")
    return _STATUS_EXIT_CODES[status]


def reason_order_key(
    reason: ReasonRecord,
) -> tuple[int, int, int, int, str, int, str, int, str, int, int, bytes]:
    """Return the exact total canonical comparison-reason ordering key."""
    if not isinstance(reason, ReasonRecord):
        raise TypeError("reason must be a ReasonRecord")
    rule = REASON_REGISTRY[reason.code]
    status_rank = _STATUS_BY_ID[rule.status_rule_id].rank
    return (
        status_rank,
        _REASON_RANK[reason.code],
        _ROLE_ORDER[reason.role],
        1 if reason.object_type is None else 0,
        reason.object_type or "",
        1 if reason.object_name is None else 0,
        reason.object_name or "",
        1 if reason.metric is None else 0,
        reason.metric or "",
        1 if reason.boundary_index is None else 0,
        0 if reason.boundary_index is None else reason.boundary_index,
        canonical_json_bytes(thaw_canonical(reason.evidence)),
    )


def ordered_reasons(reasons: Sequence[ReasonRecord]) -> tuple[ReasonRecord, ...]:
    """Validate uniqueness and return reasons in the frozen total order."""
    raw: object = reasons
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("reasons must be a sequence of ReasonRecord values")
    if any(not isinstance(reason, ReasonRecord) for reason in reasons):
        raise TypeError("all reasons must be ReasonRecord values")
    identities = [canonical_json_bytes(reason.to_primitive()) for reason in reasons]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate reason records are forbidden")
    return tuple(sorted(reasons, key=reason_order_key))


def projected_reason_codes(reasons: Sequence[ReasonRecord]) -> tuple[ReasonCode, ...]:
    """Return the stable first-occurrence code projection of canonical reasons."""
    result: list[ReasonCode] = []
    seen: set[ReasonCode] = set()
    for reason in ordered_reasons(reasons):
        if reason.code not in seen:
            seen.add(reason.code)
            result.append(reason.code)
    return tuple(result)


def derive_comparison_status(reasons: Sequence[ReasonRecord]) -> ComparisonStatus:
    """Derive the frozen public status from a complete unique reason set."""
    ordered = ordered_reasons(reasons)
    if not ordered:
        return ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD
    rule_id = REASON_REGISTRY[ordered[0].code].status_rule_id
    return _STATUS_BY_ID[rule_id].status


def canonical_limitations(values: Sequence[LimitationCode]) -> tuple[LimitationCode, ...]:
    """Validate the exact limitation set and emit registry order."""
    raw: object = values
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("limitations must be a sequence of LimitationCode values")
    if any(not isinstance(value, LimitationCode) for value in values):
        raise TypeError("limitations must contain only LimitationCode values")
    if len(values) != len(set(values)):
        raise ValueError("limitations must not contain duplicates")
    required = tuple(LimitationCode)
    if set(values) != set(required):
        raise ValueError("limitations must contain the exact limitation registry")
    return required


def _require_exact_object_fields(
    value: object, fields: set[str], context: str
) -> dict[str, object]:
    """Require a reason-record mapping with every frozen field and no unknown members."""
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    obj = cast(dict[str, object], value)
    actual = set(obj)
    missing = fields - actual
    unknown = actual - fields
    if missing:
        raise ValueError(f"{context} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{context} unknown fields: {sorted(unknown)}")
    return obj


def _nonempty_string(value: object, field: str) -> str:
    """Admit a nonempty UTF-8 comparison-reason subject string."""
    if type(value) is not str:
        raise TypeError(f"{field} must be a string")
    value.encode("utf-8", errors="strict")
    if not value:
        raise ValueError(f"{field} must be nonempty")
    return value


def _optional_nonempty_string(value: object, field: str) -> str | None:
    """Admit either ``None`` or a nonempty UTF-8 reason-subject string."""
    if value is None:
        return None
    return _nonempty_string(value, field)


__all__ = [
    "ComparisonStatus",
    "EngineThreadpoolState",
    "LimitationCode",
    "OperationalExitCode",
    "ReasonCode",
    "ReasonRecord",
    "STATUS_PRECEDENCE",
    "REASON_REGISTRY",
]
