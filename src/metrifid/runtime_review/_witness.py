"""Stable, platform-neutral witness identities for Native Runtime Review."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from decimal import Decimal
from typing import Final, cast

import numpy as np
import numpy.typing as npt

from .._native_upgrade import CaseEvidence, OrientationObservation, ScalarObservation
from ..json_values import CanonicalValue, canonical_json_bytes
from ._status import RuntimeReviewStatus

_WITHIN: Final = RuntimeReviewStatus.WITHIN_DECLARED_MIGRATION_ENVELOPE.value
_UNRESOLVED: Final = RuntimeReviewStatus.UNRESOLVED_NEAR_BOUNDARY.value
_OUTSIDE: Final = RuntimeReviewStatus.OUTSIDE_DECLARED_MIGRATION_ENVELOPE.value
_CLASSIFICATIONS: Final = frozenset({_WITHIN, _UNRESOLVED, _OUTSIDE})


@dataclass(frozen=True, slots=True)
class StableWitness:
    """Decision-bearing witness fields that are stable across conforming platforms."""

    channel_id: str
    classification: str
    kind: str
    semantic_type: str
    time: str
    tolerance: str
    decision_input_sha256: str

    def __post_init__(self) -> None:
        """Reject values outside the frozen public witness vocabulary."""
        if not self.channel_id:
            raise ValueError("stable witness channel_id must be nonempty")
        if self.classification not in _CLASSIFICATIONS:
            raise ValueError("stable witness classification is unsupported")
        if self.kind not in {"SCALAR", "ORIENTATION"}:
            raise ValueError("stable witness kind is unsupported")
        if not self.semantic_type:
            raise ValueError("stable witness semantic_type must be nonempty")
        if len(self.decision_input_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.decision_input_sha256
        ):
            raise ValueError("decision_input_sha256 must be lowercase SHA-256 hexadecimal")

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return exactly the canonical decision fields required by the receipt."""
        return {
            "channel_id": self.channel_id,
            "classification": self.classification,
            "kind": self.kind,
            "semantic_type": self.semantic_type,
            "time": self.time,
            "tolerance": self.tolerance,
            "decision_input_sha256": self.decision_input_sha256,
        }


def stable_witnesses(
    case: CaseEvidence,
    private_rows: list[dict[str, object]],
) -> tuple[StableWitness, ...]:
    """Bind every private evaluator row to the exact six raw three-grid inputs."""
    observations: dict[tuple[float, str], ScalarObservation | OrientationObservation] = {}
    for scalar_observation in case.scalar_observations:
        key = (scalar_observation.time, scalar_observation.channel_id)
        if key in observations:
            raise ValueError("case contains duplicate time/channel witness inputs")
        observations[key] = scalar_observation
    for orientation in case.orientation_observations:
        key = (orientation.time, orientation.channel_id)
        if key in observations:
            raise ValueError("case contains duplicate time/channel witness inputs")
        observations[key] = orientation

    stable: list[StableWitness] = []
    seen: set[tuple[float, str]] = set()
    for row in private_rows:
        time = _row_float(row, "time")
        channel_id = _row_string(row, "channel_id")
        key = (time, channel_id)
        if key in seen:
            raise ValueError("private result contains a duplicate witness row")
        seen.add(key)
        try:
            bound_observation = observations[key]
        except KeyError as exc:
            raise ValueError("private witness is not bound to an admitted observation") from exc
        kind = _row_string(row, "kind")
        expected_kind = (
            "ORIENTATION" if isinstance(bound_observation, OrientationObservation) else "SCALAR"
        )
        if kind != expected_kind:
            raise ValueError("private witness kind differs from its admitted observation")
        classification = _row_string(row, "classification")
        semantic_type = _row_string(row, "semantic_type")
        if semantic_type != bound_observation.semantic_type:
            raise ValueError("private witness semantic type differs from admitted evidence")
        stable.append(
            StableWitness(
                channel_id=channel_id,
                classification=classification,
                kind=kind,
                semantic_type=semantic_type,
                time=_decimal_token(time),
                tolerance=_decimal_token(bound_observation.tolerance),
                decision_input_sha256=witness_input_sha256(bound_observation),
            )
        )
    stable.sort(key=lambda row: (Decimal(row.time), row.channel_id))
    return tuple(stable)


def witness_input_sha256(observation: ScalarObservation | OrientationObservation) -> str:
    """Hash canonical metadata and exact little-endian raw three-grid input bytes."""
    if isinstance(observation, OrientationObservation):
        kind = "ORIENTATION"
        scale: str | None = None
        shape: list[CanonicalValue] = [4]
        values = np.asarray(
            (*observation.profile_a, *observation.profile_b),
            dtype=np.dtype("<f8"),
            order="C",
        )
    elif isinstance(observation, ScalarObservation):
        kind = "SCALAR"
        scale = _decimal_token(observation.scale)
        shape = []
        values = np.asarray(
            (*observation.profile_a, *observation.profile_b),
            dtype=np.dtype("<f8"),
            order="C",
        )
    else:  # pragma: no cover - the type boundary is retained at runtime
        raise TypeError("observation must be a scalar or orientation observation")
    contiguous = cast(npt.NDArray[np.float64], np.ascontiguousarray(values, dtype="<f8"))
    metadata: dict[str, CanonicalValue] = {
        "schema": "metrifid.runtime_review_witness_input",
        "schema_version": 1,
        "channel_id": observation.channel_id,
        "time": _decimal_token(observation.time),
        "kind": kind,
        "semantic_type": observation.semantic_type,
        "scale": scale,
        "tolerance": _decimal_token(observation.tolerance),
        "value_shape": shape,
    }
    encoded = canonical_json_bytes(metadata)
    digest = hashlib.sha256()
    digest.update(struct.pack(">Q", len(encoded)))
    digest.update(encoded)
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def first_decisive_witness(
    status: RuntimeReviewStatus,
    witnesses: tuple[StableWitness, ...],
) -> StableWitness | None:
    """Select the earliest witness that proves the final OUTSIDE or UNRESOLVED result."""
    classification = {
        RuntimeReviewStatus.OUTSIDE_DECLARED_MIGRATION_ENVELOPE: _OUTSIDE,
        RuntimeReviewStatus.UNRESOLVED_NEAR_BOUNDARY: _UNRESOLVED,
    }.get(status)
    if classification is None:
        return None
    candidates = [item for item in witnesses if item.classification == classification]
    return (
        min(candidates, key=lambda row: (Decimal(row.time), row.channel_id)) if candidates else None
    )


def stable_worst_witness(
    private_worst: object,
    witnesses: tuple[StableWitness, ...],
) -> StableWitness | None:
    """Map the private ratio-ranked worst row onto its stable public identity."""
    if private_worst is None:
        return None
    if type(private_worst) is not dict:
        raise ValueError("private worst witness must be an object or null")
    row = cast(dict[str, object], private_worst)
    key = (_decimal_token(_row_float(row, "time")), _row_string(row, "channel_id"))
    matches = [item for item in witnesses if (item.time, item.channel_id) == key]
    if len(matches) != 1:
        raise ValueError("private worst witness is not bound to one admitted stable witness")
    return matches[0]


def _decimal_token(value: float) -> str:
    """Render one finite binary64 value as a canonical ordinary decimal token."""
    decimal = Decimal(str(value))
    if not decimal.is_finite():
        raise ValueError("witness decimal token must be finite")
    if decimal == 0:
        return "0"
    token = format(decimal, "f")
    return token.rstrip("0").rstrip(".") if "." in token else token


def _row_string(row: dict[str, object], field: str) -> str:
    """Read one required nonempty private-row string."""
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"private witness {field} must be a nonempty string")
    return value


def _row_float(row: dict[str, object], field: str) -> float:
    """Read one required private-row binary64 value."""
    value = row.get(field)
    if type(value) is not float:
        raise ValueError(f"private witness {field} must be a binary64 value")
    return value


__all__ = [
    "StableWitness",
    "first_decisive_witness",
    "stable_witnesses",
    "stable_worst_witness",
    "witness_input_sha256",
]
