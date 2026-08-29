"""Private data-only decision core for the selected native-upgrade method.

This module implements only the preregistered ``CONDITIONAL_TAIL_ENVELOPE`` winner.  It accepts
already-admitted semantic evidence and performs no model execution, file access, environment
inspection, networking, or subprocess work.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, cast

DecisionStatus: TypeAlias = Literal[
    "REFUSED",
    "REPEATABILITY_FAILED",
    "SOLVER_NOT_CONVERGED",
    "CONTACT_EVENT_TOPOLOGY_CHANGED",
    "NON_ASYMPTOTIC_REGIME",
    "PREFIX_TOO_SHORT",
    "WITHIN_DECLARED_MIGRATION_ENVELOPE",
    "OUTSIDE_DECLARED_MIGRATION_ENVELOPE",
    "UNRESOLVED_NEAR_BOUNDARY",
]
GateStatus: TypeAlias = Literal[
    "REFUSED",
    "REPEATABILITY_FAILED",
    "SOLVER_NOT_CONVERGED",
    "CONTACT_EVENT_TOPOLOGY_CHANGED",
    "NON_ASYMPTOTIC_REGIME",
]
Quaternion: TypeAlias = tuple[float, float, float, float]

_METHOD: Final = "CONDITIONAL_TAIL_ENVELOPE"
_EPSILON_NUM: Final = 0.000000000001
_RHO_MAX: Final = 0.75
_MINIMUM_PREFIX: Final = 0.50
_HORIZON: Final = 1.00

_REFUSED: Final[Literal["REFUSED"]] = "REFUSED"
_REPEATABILITY_FAILED: Final[Literal["REPEATABILITY_FAILED"]] = "REPEATABILITY_FAILED"
_SOLVER_NOT_CONVERGED: Final[Literal["SOLVER_NOT_CONVERGED"]] = "SOLVER_NOT_CONVERGED"
_CONTACT_EVENT_TOPOLOGY_CHANGED: Final[Literal["CONTACT_EVENT_TOPOLOGY_CHANGED"]] = (
    "CONTACT_EVENT_TOPOLOGY_CHANGED"
)
_NON_ASYMPTOTIC_REGIME: Final[Literal["NON_ASYMPTOTIC_REGIME"]] = "NON_ASYMPTOTIC_REGIME"
_PREFIX_TOO_SHORT: Final[Literal["PREFIX_TOO_SHORT"]] = "PREFIX_TOO_SHORT"
_WITHIN: Final[Literal["WITHIN_DECLARED_MIGRATION_ENVELOPE"]] = "WITHIN_DECLARED_MIGRATION_ENVELOPE"
_OUTSIDE: Final[Literal["OUTSIDE_DECLARED_MIGRATION_ENVELOPE"]] = (
    "OUTSIDE_DECLARED_MIGRATION_ENVELOPE"
)
_UNRESOLVED: Final[Literal["UNRESOLVED_NEAR_BOUNDARY"]] = "UNRESOLVED_NEAR_BOUNDARY"

_GATE_ORDER: Final[tuple[GateStatus, ...]] = (
    _REFUSED,
    _REPEATABILITY_FAILED,
    _SOLVER_NOT_CONVERGED,
    _CONTACT_EVENT_TOPOLOGY_CHANGED,
    _NON_ASYMPTOTIC_REGIME,
)
_GATE_RANK: Final = {status: rank for rank, status in enumerate(_GATE_ORDER)}


@dataclass(frozen=True, slots=True)
class Interval:
    """Represent one ordered finite closed binary64 interval."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        """Reject a non-finite or reversed interval."""
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise ValueError("interval endpoints must be finite")
        if self.lower > self.upper:
            raise ValueError("interval lower endpoint exceeds its upper endpoint")

    def primitive(self) -> list[float]:
        """Return the deterministic JSON-compatible representation."""
        return [self.lower, self.upper]


@dataclass(frozen=True, slots=True)
class GateEvent:
    """Describe one named gate failure beginning at a physical-time boundary."""

    time: float
    status: GateStatus
    channel_id: str
    detail: str

    def __post_init__(self) -> None:
        """Reject malformed gate evidence."""
        if not math.isfinite(self.time) or self.time < 0.0:
            raise ValueError("gate time must be finite and nonnegative")
        if self.status not in _GATE_RANK:
            raise ValueError(f"unknown gate status: {self.status}")
        if not self.channel_id:
            raise ValueError("gate channel_id must be nonempty")

    def primitive(self) -> dict[str, object]:
        """Return the deterministic JSON-compatible representation."""
        return {
            "channel_id": self.channel_id,
            "detail": self.detail,
            "status": self.status,
            "time": self.time,
        }


@dataclass(frozen=True, slots=True)
class ScalarObservation:
    """Hold two profiles' three-grid evidence for one scaled scalar witness."""

    channel_id: str
    time: float
    scale: float
    tolerance: float
    profile_a: tuple[float, float, float]
    profile_b: tuple[float, float, float]
    semantic_type: str = "CONTINUOUS_SCALAR"


@dataclass(frozen=True, slots=True)
class OrientationObservation:
    """Hold two profiles' three-grid orientation evidence in ``wxyz`` order."""

    channel_id: str
    time: float
    tolerance: float
    profile_a: tuple[Quaternion, Quaternion, Quaternion]
    profile_b: tuple[Quaternion, Quaternion, Quaternion]
    semantic_type: str = "UNIT_QUATERNION_WXYZ"


@dataclass(frozen=True, slots=True)
class CaseEvidence:
    """Contain the admitted data needed for one selected-method decision."""

    case_id: str
    scalar_observations: tuple[ScalarObservation, ...]
    orientation_observations: tuple[OrientationObservation, ...]
    gate_events: tuple[GateEvent, ...] = ()
    minimum_prefix: float = _MINIMUM_PREFIX
    horizon: float = _HORIZON
    campaign_role: str = "ORACLE"
    baseline_fixture_id: str | None = None


def _outward(lower: float, upper: float) -> Interval:
    """Expand finite ordered endpoints by one binary64 step toward infinity."""
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("outward interval requires ordered finite endpoints")
    return Interval(math.nextafter(lower, -math.inf), math.nextafter(upper, math.inf))


def _difference(profile_b: Interval, profile_a: Interval) -> Interval:
    """Return the outward interval difference ``profile_b - profile_a``."""
    return _outward(profile_b.lower - profile_a.upper, profile_b.upper - profile_a.lower)


def _absolute_bounds(interval: Interval) -> tuple[float, float]:
    """Return the least and greatest possible magnitudes in an interval."""
    lower = (
        0.0
        if interval.lower <= 0.0 <= interval.upper
        else min(abs(interval.lower), abs(interval.upper))
    )
    return lower, max(abs(interval.lower), abs(interval.upper))


def _classification(interval: Interval, tolerance: float) -> DecisionStatus:
    """Apply the frozen full-enclosure decision rule."""
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and greater than zero")
    lower, upper = _absolute_bounds(interval)
    if upper <= tolerance:
        return _WITHIN
    if lower > tolerance:
        return _OUTSIDE
    return _UNRESOLVED


def _tail_interval(values: tuple[float, float, float]) -> Interval | None:
    """Build one profile's conditional-tail interval after its convergence gate."""
    if not all(math.isfinite(value) for value in values):
        raise ValueError("conditional-tail inputs must be finite")
    coarse, fine, finest = values
    d0 = abs(fine - coarse)
    d1 = abs(finest - fine)
    stable = d0 <= _EPSILON_NUM and d1 <= _EPSILON_NUM
    if not stable and d1 > _RHO_MAX * d0:
        return None
    uncertainty = _EPSILON_NUM + d1 * _RHO_MAX / (1.0 - _RHO_MAX)
    return _outward(finest - uncertainty, finest + uncertainty)


def _normalized(quaternion: Sequence[float]) -> Quaternion:
    """Normalize one finite nonzero quaternion without choosing its sign."""
    if len(quaternion) != 4:
        raise ValueError("a quaternion must contain exactly four components")
    values = tuple(float(value) for value in quaternion)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("quaternion components must be finite")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("quaternion norm must be finite and nonzero")
    return cast(Quaternion, tuple(value / norm for value in values))


def _rotation_angle(left: Sequence[float], right: Sequence[float]) -> float:
    """Measure the sign-invariant SO(3) geodesic angle ``right * conjugate(left)``."""
    lw, lx, ly, lz = _normalized(left)
    rw, rx, ry, rz = _normalized(right)
    scalar = rw * lw + rx * lx + ry * ly + rz * lz
    vector_x = -rw * lx + rx * lw - ry * lz + rz * ly
    vector_y = -rw * ly + rx * lz + ry * lw - rz * lx
    vector_z = -rw * lz - rx * ly + ry * lx + rz * lw
    if scalar < 0.0:
        scalar, vector_x, vector_y, vector_z = (
            -scalar,
            -vector_x,
            -vector_y,
            -vector_z,
        )
    vector_norm = math.sqrt(vector_x * vector_x + vector_y * vector_y + vector_z * vector_z)
    return 2.0 * math.atan2(vector_norm, abs(scalar))


def _orientation_radius(values: tuple[Quaternion, Quaternion, Quaternion]) -> float | None:
    """Build one profile's conditional SO(3) refinement radius about its finest value."""
    coarse, fine, finest = values
    d0 = _rotation_angle(fine, coarse)
    d1 = _rotation_angle(finest, fine)
    stable = d0 <= _EPSILON_NUM and d1 <= _EPSILON_NUM
    if not stable and d1 > _RHO_MAX * d0:
        return None
    return _EPSILON_NUM + d1 * _RHO_MAX / (1.0 - _RHO_MAX)


def _orientation_interval(observation: OrientationObservation) -> Interval | None:
    """Combine separately admitted profile radii using the SO(3) triangle inequality."""
    radius_a = _orientation_radius(observation.profile_a)
    radius_b = _orientation_radius(observation.profile_b)
    if radius_a is None or radius_b is None:
        return None
    center = _rotation_angle(observation.profile_a[2], observation.profile_b[2])
    radius = radius_a + radius_b
    outward = _outward(max(0.0, center - radius), min(math.pi, center + radius))
    return Interval(max(0.0, outward.lower), min(math.pi, outward.upper))


def _validate_observation(
    time: float,
    channel_id: str,
    tolerance: float,
    horizon: float,
    seen: set[tuple[float, str]],
) -> None:
    """Validate one unique bounded time/channel identity and positive tolerance."""
    identity = (time, channel_id)
    if identity in seen:
        raise ValueError("duplicate time/channel observation")
    seen.add(identity)
    if not channel_id:
        raise ValueError("channel_id must be nonempty")
    if not math.isfinite(time) or not 0.0 <= time <= horizon:
        raise ValueError("observation time lies outside the declared horizon")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("every tolerance must be finite and greater than zero")


def _validate_case(case: CaseEvidence) -> None:
    """Reject malformed case metadata before any scientific decision."""
    if not case.case_id:
        raise ValueError("case_id must be nonempty")
    if not math.isfinite(case.minimum_prefix) or case.minimum_prefix <= 0.0:
        raise ValueError("minimum_prefix must be finite and greater than zero")
    if not math.isfinite(case.horizon) or case.horizon < case.minimum_prefix:
        raise ValueError("horizon must be finite and no shorter than minimum_prefix")
    if any(event.time > case.horizon for event in case.gate_events):
        raise ValueError("gate time lies outside the declared horizon")
    seen: set[tuple[float, str]] = set()
    for scalar in case.scalar_observations:
        _validate_observation(
            scalar.time,
            scalar.channel_id,
            scalar.tolerance,
            case.horizon,
            seen,
        )
        if not math.isfinite(scalar.scale) or scalar.scale <= 0.0:
            raise ValueError("every scalar scale must be finite and greater than zero")
    for orientation in case.orientation_observations:
        _validate_observation(
            orientation.time,
            orientation.channel_id,
            orientation.tolerance,
            case.horizon,
            seen,
        )


def _nonfinite_events(case: CaseEvidence) -> list[GateEvent]:
    """Map non-finite required evidence to named solver gates."""
    events: list[GateEvent] = []
    for scalar in case.scalar_observations:
        if not all(math.isfinite(value) for value in (*scalar.profile_a, *scalar.profile_b)):
            events.append(
                GateEvent(
                    scalar.time,
                    _SOLVER_NOT_CONVERGED,
                    scalar.channel_id,
                    "required scalar evidence is non-finite",
                )
            )
    for orientation in case.orientation_observations:
        values = (
            value
            for quaternion in (*orientation.profile_a, *orientation.profile_b)
            for value in quaternion
        )
        if not all(math.isfinite(value) for value in values):
            events.append(
                GateEvent(
                    orientation.time,
                    _SOLVER_NOT_CONVERGED,
                    orientation.channel_id,
                    "required quaternion evidence is non-finite",
                )
            )
    return events


def _convergence_events(case: CaseEvidence) -> list[GateEvent]:
    """Apply the selected method's convergence gate separately to both profiles."""
    events: list[GateEvent] = []
    for scalar in case.scalar_observations:
        values = (*scalar.profile_a, *scalar.profile_b)
        if not all(math.isfinite(value) for value in values):
            continue
        profile_a = cast(
            tuple[float, float, float],
            tuple(value / scalar.scale for value in scalar.profile_a),
        )
        profile_b = cast(
            tuple[float, float, float],
            tuple(value / scalar.scale for value in scalar.profile_b),
        )
        if _tail_interval(profile_a) is None or _tail_interval(profile_b) is None:
            events.append(
                GateEvent(
                    scalar.time,
                    _NON_ASYMPTOTIC_REGIME,
                    scalar.channel_id,
                    "at least one scalar profile does not satisfy the frozen contraction gate",
                )
            )
    for orientation in case.orientation_observations:
        finite = all(
            math.isfinite(value)
            for quaternion in (*orientation.profile_a, *orientation.profile_b)
            for value in quaternion
        )
        if finite and _orientation_interval(orientation) is None:
            events.append(
                GateEvent(
                    orientation.time,
                    _NON_ASYMPTOTIC_REGIME,
                    orientation.channel_id,
                    "at least one orientation profile does not satisfy the frozen contraction gate",
                )
            )
    return events


def _first_gate(events: Iterable[GateEvent]) -> GateEvent | None:
    """Select the earliest gate using fixed label and lexical-channel tie breaks."""
    ordered = sorted(
        events, key=lambda event: (event.time, _GATE_RANK[event.status], event.channel_id)
    )
    return ordered[0] if ordered else None


def _scalar_witness(observation: ScalarObservation) -> dict[str, object]:
    """Evaluate one nondimensional scalar witness with the selected tail method."""
    profile_a_values = cast(
        tuple[float, float, float],
        tuple(value / observation.scale for value in observation.profile_a),
    )
    profile_b_values = cast(
        tuple[float, float, float],
        tuple(value / observation.scale for value in observation.profile_b),
    )
    profile_a = _tail_interval(profile_a_values)
    profile_b = _tail_interval(profile_b_values)
    if profile_a is None or profile_b is None:
        raise ValueError("non-asymptotic scalar reached interval evaluation")
    difference = _difference(profile_b, profile_a)
    lower, upper = _absolute_bounds(difference)
    return {
        "channel_id": observation.channel_id,
        "classification": _classification(difference, observation.tolerance),
        "difference_interval": difference.primitive(),
        "kind": "SCALAR",
        "lower_abs": lower,
        "profile_a_interval": profile_a.primitive(),
        "profile_b_interval": profile_b.primitive(),
        "ratio": upper / observation.tolerance,
        "semantic_type": observation.semantic_type,
        "time": observation.time,
        "tolerance": observation.tolerance,
        "upper_abs": upper,
    }


def _orientation_witness(observation: OrientationObservation) -> dict[str, object]:
    """Evaluate one direct SO(3) witness with separate profile refinement radii."""
    interval = _orientation_interval(observation)
    if interval is None:
        raise ValueError("non-asymptotic orientation reached interval evaluation")
    lower, upper = _absolute_bounds(interval)
    return {
        "channel_id": observation.channel_id,
        "classification": _classification(interval, observation.tolerance),
        "difference_interval": interval.primitive(),
        "kind": "ORIENTATION",
        "lower_abs": lower,
        "profile_a_interval": None,
        "profile_b_interval": None,
        "ratio": upper / observation.tolerance,
        "semantic_type": observation.semantic_type,
        "time": observation.time,
        "tolerance": observation.tolerance,
        "upper_abs": upper,
    }


def _decision_status(rows: Sequence[dict[str, object]]) -> DecisionStatus:
    """Apply OUTSIDE, then UNRESOLVED, then WITHIN completed-decision precedence."""
    statuses = {cast(DecisionStatus, row["classification"]) for row in rows}
    if _OUTSIDE in statuses:
        return _OUTSIDE
    if _UNRESOLVED in statuses:
        return _UNRESOLVED
    return _WITHIN


def _first_witness(rows: Sequence[dict[str, object]]) -> dict[str, object] | None:
    """Return the earliest witness using lexical channel order for a time tie."""
    if not rows:
        return None
    return min(rows, key=lambda row: (cast(float, row["time"]), cast(str, row["channel_id"])))


def _worst_witness(rows: Sequence[dict[str, object]]) -> dict[str, object] | None:
    """Return the greatest upper/tolerance ratio with the frozen tie breaks."""
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            -cast(float, row["ratio"]),
            cast(float, row["time"]),
            cast(str, row["channel_id"]),
        ),
    )


def _refused(case: CaseEvidence, detail: str) -> dict[str, object]:
    """Build one fail-closed refusal over the case's declared finite horizon."""
    event = GateEvent(0.0, _REFUSED, "input", detail)
    horizon = case.horizon if math.isfinite(case.horizon) and case.horizon >= 0.0 else _HORIZON
    return {
        "admitted_prefix": 0.0,
        "case_id": case.case_id,
        "first_failing_gate": event.primitive(),
        "first_witness": None,
        "method": _METHOD,
        "status": _REFUSED,
        "unqualified_suffix": [0.0, horizon],
        "witness_count": 0,
        "witnesses": [],
        "worst_witness": None,
    }


def evaluate_case(case: CaseEvidence) -> dict[str, object]:
    """Evaluate one admitted case using only the selected conditional-tail method."""
    try:
        _validate_case(case)
        events = [*case.gate_events, *_nonfinite_events(case), *_convergence_events(case)]
        first_gate = _first_gate(events)
        admitted_prefix = case.horizon if first_gate is None else first_gate.time
        if first_gate is not None and first_gate.time == 0.0:
            return {
                "admitted_prefix": 0.0,
                "case_id": case.case_id,
                "first_failing_gate": first_gate.primitive(),
                "first_witness": None,
                "method": _METHOD,
                "status": first_gate.status,
                "unqualified_suffix": [0.0, case.horizon],
                "witness_count": 0,
                "witnesses": [],
                "worst_witness": None,
            }
        if first_gate is not None and first_gate.time < case.minimum_prefix:
            return {
                "admitted_prefix": first_gate.time,
                "case_id": case.case_id,
                "first_failing_gate": first_gate.primitive(),
                "first_witness": None,
                "method": _METHOD,
                "status": _PREFIX_TOO_SHORT,
                "unqualified_suffix": [first_gate.time, case.horizon],
                "witness_count": 0,
                "witnesses": [],
                "worst_witness": None,
            }
        rows = [
            _scalar_witness(observation)
            for observation in case.scalar_observations
            if first_gate is None or observation.time < first_gate.time
        ]
        rows.extend(
            _orientation_witness(observation)
            for observation in case.orientation_observations
            if first_gate is None or observation.time < first_gate.time
        )
        rows.sort(key=lambda row: (cast(float, row["time"]), cast(str, row["channel_id"])))
        if not rows:
            return _refused(case, "no observation lies in the admitted prefix")
        return {
            "admitted_prefix": admitted_prefix,
            "case_id": case.case_id,
            "first_failing_gate": first_gate.primitive() if first_gate is not None else None,
            "first_witness": _first_witness(rows),
            "method": _METHOD,
            "status": _decision_status(rows),
            "unqualified_suffix": (
                [first_gate.time, case.horizon] if first_gate is not None else None
            ),
            "witness_count": len(rows),
            "witnesses": rows,
            "worst_witness": _worst_witness(rows),
        }
    except (TypeError, ValueError, OverflowError) as exc:
        return _refused(case, str(exc))


__all__ = [
    "Interval",
    "GateEvent",
    "ScalarObservation",
    "OrientationObservation",
    "CaseEvidence",
    "evaluate_case",
]
