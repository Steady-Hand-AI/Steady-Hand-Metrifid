"""Strict workload-qualification configuration schema.

Every field is declared by the user. Nothing is inferred: not units, not probes, not tolerances, not
model roots, not workload identities. Unknown fields are refused rather than ignored, because a
silently dropped field is a claim the user believes they made and the receipt does not carry.

The bounds here are the ones the evidence supports. Schema version 1 freezes the workload budget at
three and the candidate count at sixteen, so the subset search is at most 560 combinations and needs
no host-timed bound.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Self

from .._configuration_schemas import JointToleranceConfig, ModelRoleConfig
from .._schema_primitives import (
    _bounded_int,
    _fields,
    _name,
    _nonempty_string,
    _object,
    _positive_decimal,
    _sequence,
    _string_keyed_mapping,
)
from ..json_values import CanonicalValue, ExactRational
from ._paths import admit_relative_path

SCHEMA_VERSION: Final[int] = 1
MIN_PROBE_GROUPS: Final[int] = 1
MAX_PROBE_GROUPS: Final[int] = 16
MIN_VARIANTS: Final[int] = 2
MAX_VARIANTS: Final[int] = 8
MIN_WORKLOADS: Final[int] = 3
MAX_WORKLOADS: Final[int] = 16
REQUIRED_BUDGET: Final[int] = 3
DIRECTIONS: Final[frozenset[str]] = frozenset({"increase", "decrease"})
# The declared magnitude is an exact decimal; this string says what that decimal means. It is a user
# declaration that Metrifid preserves and never parses, converts, or verifies.
MAGNITUDE_SEMANTICS_FIELD: Final[str] = "magnitude_semantics"


def _magnitude_semantics(value: object) -> str:
    """Admit the user's declaration of what the exact magnitude means.

    Preserved byte for byte. It is never trimmed, normalized, parsed as a unit, or treated as an
    established fact about the source edit; it exists so a reviewer can read a receipt and know what
    the declaring user believed the number meant.
    """
    text = _nonempty_string(value, MAGNITUDE_SEMANTICS_FIELD)
    if not text.strip():
        raise ValueError(f"{MAGNITUDE_SEMANTICS_FIELD} must contain a non-whitespace character")
    return text


def _model_role(value: object, context: str) -> ModelRoleConfig:
    """Decode one model role and admit its paths at their declared bases.

    ``model_root`` is relative to the qualification file's directory; ``entrypoint`` is relative to
    that model root. Resolving an entrypoint against the configuration directory would read a
    different file than the one the user named.
    """
    role = ModelRoleConfig.from_primitive(value)
    admit_relative_path(role.model_root, f"{context}.model_root")
    admit_relative_path(role.entrypoint, f"{context}.entrypoint")
    return role


def _direction(value: object) -> str:
    """Admit exactly one declared perturbation direction."""
    text = _nonempty_string(value, "direction")
    if text not in DIRECTIONS:
        raise ValueError("direction must be increase or decrease")
    return text


@dataclass(frozen=True, slots=True)
class ProbeVariant:
    """One rung of a probe ladder: a declared magnitude and the model that realizes it."""

    magnitude: ExactRational
    candidate: ModelRoleConfig

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode one ladder rung."""
        obj = _object(value, "ProbeVariant")
        _fields(obj, {"magnitude", "candidate"}, "ProbeVariant")
        return cls(
            _positive_decimal(obj["magnitude"], "magnitude"),
            _model_role(obj["candidate"], "probe candidate"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the rung with its magnitude as an ordinary decimal token."""
        return {
            "magnitude": self.magnitude.to_decimal_token(),
            "candidate": self.candidate.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class ProbeGroup:
    """One declared perturbation, one direction, and its strictly increasing ladder."""

    probe_id: str
    parameter: str
    direction: str
    magnitude_semantics: str
    required_detection_magnitude: ExactRational
    variants: tuple[ProbeVariant, ...]

    def __post_init__(self) -> None:
        """Require a bounded, strictly increasing ladder containing the required magnitude."""
        _name(self.probe_id, "probe_id")
        _nonempty_string(self.parameter, "parameter")
        if self.direction not in DIRECTIONS:
            raise ValueError("direction must be increase or decrease")
        _magnitude_semantics(self.magnitude_semantics)
        count = len(self.variants)
        if not MIN_VARIANTS <= count <= MAX_VARIANTS:
            raise ValueError(
                f"variants must contain between {MIN_VARIANTS} and {MAX_VARIANTS} entries"
            )
        previous: ExactRational | None = None
        declared_models: set[tuple[str, str]] = set()
        for variant in self.variants:
            if previous is not None and not _strictly_greater(variant.magnitude, previous):
                raise ValueError("variant magnitudes must be strictly increasing and unique")
            previous = variant.magnitude
            declared_model = (variant.candidate.model_root, variant.candidate.entrypoint)
            if declared_model in declared_models:
                raise ValueError(
                    "probe variants in one group must declare distinct model_root + entrypoint "
                    "pairs"
                )
            declared_models.add(declared_model)
        if not any(_equal(v.magnitude, self.required_detection_magnitude) for v in self.variants):
            raise ValueError("required_detection_magnitude must equal one listed magnitude")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode one probe group exactly."""
        obj = _object(value, "ProbeGroup")
        _fields(
            obj,
            {
                "probe_id",
                "parameter",
                "direction",
                MAGNITUDE_SEMANTICS_FIELD,
                "required_detection_magnitude",
                "variants",
            },
            "ProbeGroup",
        )
        raw = _sequence(obj["variants"], "variants")
        return cls(
            _name(obj["probe_id"], "probe_id"),
            _nonempty_string(obj["parameter"], "parameter"),
            _direction(obj["direction"]),
            _magnitude_semantics(obj[MAGNITUDE_SEMANTICS_FIELD]),
            _positive_decimal(obj["required_detection_magnitude"], "required_detection_magnitude"),
            tuple(ProbeVariant.from_primitive(item) for item in raw),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the declared group without reordering its ladder."""
        return {
            "probe_id": self.probe_id,
            "parameter": self.parameter,
            "direction": self.direction,
            MAGNITUDE_SEMANTICS_FIELD: self.magnitude_semantics,
            "required_detection_magnitude": self.required_detection_magnitude.to_decimal_token(),
            "variants": [variant.to_primitive() for variant in self.variants],
        }

    def required_index(self) -> int:
        """Return the ladder index of the required detection magnitude."""
        for index, variant in enumerate(self.variants):
            if _equal(variant.magnitude, self.required_detection_magnitude):
                return index
        raise ValueError("required_detection_magnitude is not present in the ladder")


@dataclass(frozen=True, slots=True)
class WorkloadCandidate:
    """One declared workload artifact pair and its exact control period."""

    workload_id: str
    initial_state: str
    actions: str
    control_dt: ExactRational

    def __post_init__(self) -> None:
        """Validate the workload identity, artifact paths, and exact control period."""
        _name(self.workload_id, "workload_id")
        _nonempty_string(self.initial_state, "initial_state")
        _nonempty_string(self.actions, "actions")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode one workload candidate exactly."""
        obj = _object(value, "WorkloadCandidate")
        _fields(obj, {"workload_id", "initial_state", "actions", "control_dt"}, "WorkloadCandidate")
        return cls(
            _name(obj["workload_id"], "workload_id"),
            admit_relative_path(obj["initial_state"], "initial_state"),
            admit_relative_path(obj["actions"], "actions"),
            _positive_decimal(obj["control_dt"], "control_dt"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the declared workload candidate."""
        return {
            "workload_id": self.workload_id,
            "initial_state": self.initial_state,
            "actions": self.actions,
            "control_dt": self.control_dt.to_decimal_token(),
        }


@dataclass(frozen=True, slots=True)
class QualificationConfig:
    """The complete strict qualification configuration."""

    schema_version: int
    baseline: ModelRoleConfig
    probe_groups: tuple[ProbeGroup, ...]
    workloads: tuple[WorkloadCandidate, ...]
    repeats: int
    joint_tolerances: Mapping[str, JointToleranceConfig]
    aliases: str | None
    budget: int
    output_dir: str

    def __post_init__(self) -> None:
        """Enforce every bound schema version 1 freezes."""
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        groups = len(self.probe_groups)
        if not MIN_PROBE_GROUPS <= groups <= MAX_PROBE_GROUPS:
            raise ValueError(
                f"probe_groups must contain between {MIN_PROBE_GROUPS} and "
                f"{MAX_PROBE_GROUPS} entries"
            )
        _unique(tuple(group.probe_id for group in self.probe_groups), "probe_id")
        workloads = len(self.workloads)
        if not MIN_WORKLOADS <= workloads <= MAX_WORKLOADS:
            raise ValueError(
                f"workloads must contain between {MIN_WORKLOADS} and {MAX_WORKLOADS} entries"
            )
        _unique(tuple(item.workload_id for item in self.workloads), "workload_id")
        _bounded_int(self.repeats, "repeats", 2, 5)
        if self.budget != REQUIRED_BUDGET:
            raise ValueError(f"budget must be exactly {REQUIRED_BUDGET} in schema version 1")
        tolerances = _string_keyed_mapping(self.joint_tolerances, "joint_tolerances")
        if not tolerances:
            raise ValueError("joint_tolerances must be nonempty")
        for tolerance in tolerances.values():
            if not isinstance(tolerance, JointToleranceConfig):
                raise TypeError("joint_tolerances values must be JointToleranceConfig")
        normalized = {
            _name(name, "joint_tolerances key"): tolerances[name] for name in sorted(tolerances)
        }
        object.__setattr__(self, "joint_tolerances", MappingProxyType(normalized))
        if self.aliases is not None:
            _nonempty_string(self.aliases, "aliases")
        _nonempty_string(self.output_dir, "output_dir")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode the complete configuration, refusing any unknown top-level field."""
        obj = _object(value, "QualificationConfig")
        _fields(
            obj,
            {
                "schema_version",
                "baseline",
                "probe_groups",
                "workloads",
                "repeats",
                "joint_tolerances",
                "aliases",
                "budget",
                "output_dir",
            },
            "QualificationConfig",
        )
        tolerances_raw = _object(obj["joint_tolerances"], "joint_tolerances")
        aliases = obj["aliases"]
        return cls(
            _bounded_int(obj["schema_version"], "schema_version", 1, 1),
            _model_role(obj["baseline"], "baseline"),
            tuple(
                ProbeGroup.from_primitive(item)
                for item in _sequence(obj["probe_groups"], "probe_groups")
            ),
            tuple(
                WorkloadCandidate.from_primitive(item)
                for item in _sequence(obj["workloads"], "workloads")
            ),
            _bounded_int(obj["repeats"], "repeats", 2, 5),
            {
                _name(name, "joint_tolerances key"): JointToleranceConfig.from_primitive(item)
                for name, item in tolerances_raw.items()
            },
            None if aliases is None else admit_relative_path(aliases, "aliases"),
            _bounded_int(obj["budget"], "budget", REQUIRED_BUDGET, REQUIRED_BUDGET),
            admit_relative_path(obj["output_dir"], "output_dir"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the configuration exactly as declared."""
        return {
            "schema_version": self.schema_version,
            "baseline": self.baseline.to_primitive(),
            "probe_groups": [group.to_primitive() for group in self.probe_groups],
            "workloads": [item.to_primitive() for item in self.workloads],
            "repeats": self.repeats,
            "joint_tolerances": {
                name: tolerance.to_primitive() for name, tolerance in self.joint_tolerances.items()
            },
            "aliases": self.aliases,
            "budget": self.budget,
            "output_dir": self.output_dir,
        }


def _unique(values: Sequence[str], field: str) -> None:
    """Require every declared identity in one collection to be distinct."""
    if len(set(values)) != len(values):
        raise ValueError(f"{field} values must be unique")


def _equal(left: ExactRational, right: ExactRational) -> bool:
    """Compare two exact rationals without floating-point arithmetic."""
    return left.numerator * right.denominator == right.numerator * left.denominator


def _strictly_greater(left: ExactRational, right: ExactRational) -> bool:
    """Return whether left is strictly greater than right, exactly."""
    return left.numerator * right.denominator > right.numerator * left.denominator
