"""Strict workload and alias artifact schemas."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Self, cast

from ._schema_constants import (
    _ACTIONS_SCHEMA,
    _ACTIONS_SCHEMA_VERSION,
    _ALIASES_SCHEMA,
    _ALIASES_SCHEMA_VERSION,
    _STATE_SCHEMA,
    _STATE_SCHEMA_VERSION,
)
from ._schema_primitives import (
    _bounded_int,
    _exact_int,
    _fields,
    _int_sequence,
    _name,
    _name_sequence,
    _nonempty_string,
    _nonnegative_int,
    _object,
    _require_instance,
    _require_int_tuple,
    _require_string_tuple,
    _require_typed_tuple,
    _sequence,
    _sorted_unique_names,
    _string,
    _unique_names,
    _validate_offsets,
)
from .json_values import CanonicalValue, require_sha256


@dataclass(frozen=True, slots=True)
class StateArtifactMetadata:
    """Strict semantic metadata for a state artifact."""

    schema: str
    schema_version: int
    joint_names: tuple[str, ...]
    qpos_offsets: tuple[int, ...]
    qpos_count: int
    qvel_offsets: tuple[int, ...]
    qvel_count: int
    actuator_names: tuple[str, ...]
    act_offsets: tuple[int, ...]
    act_count: int

    def __post_init__(self) -> None:
        """Validate state schema identity, unique names, counts, and segmented-array offsets."""
        if self.schema != _STATE_SCHEMA:
            raise ValueError("invalid state artifact schema")
        if type(self.schema_version) is not int or self.schema_version != _STATE_SCHEMA_VERSION:
            raise ValueError("invalid state artifact schema_version")
        _require_string_tuple(self.joint_names, "joint_names", names=True)
        _require_int_tuple(self.qpos_offsets, "qpos_offsets")
        _nonnegative_int(self.qpos_count, "qpos_count")
        _require_int_tuple(self.qvel_offsets, "qvel_offsets")
        _nonnegative_int(self.qvel_count, "qvel_count")
        _require_string_tuple(self.actuator_names, "actuator_names", names=True)
        _require_int_tuple(self.act_offsets, "act_offsets")
        _nonnegative_int(self.act_count, "act_count")
        _unique_names(self.joint_names, "joint_names")
        _unique_names(self.actuator_names, "actuator_names")
        _validate_offsets(self.qpos_offsets, len(self.joint_names), self.qpos_count, "qpos_offsets")
        _validate_offsets(self.qvel_offsets, len(self.joint_names), self.qvel_count, "qvel_offsets")
        _validate_offsets(self.act_offsets, len(self.actuator_names), self.act_count, "act_offsets")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode the exact state-artifact metadata object into validated typed fields."""
        obj = _object(value, "StateArtifactMetadata")
        expected = {
            "schema",
            "schema_version",
            "joint_names",
            "qpos_offsets",
            "qpos_count",
            "qvel_offsets",
            "qvel_count",
            "actuator_names",
            "act_offsets",
            "act_count",
        }
        _fields(obj, expected, "StateArtifactMetadata")
        return cls(
            _string(obj["schema"], "schema"),
            _exact_int(obj["schema_version"], "schema_version"),
            _name_sequence(obj["joint_names"], "joint_names"),
            _int_sequence(obj["qpos_offsets"], "qpos_offsets"),
            _nonnegative_int(obj["qpos_count"], "qpos_count"),
            _int_sequence(obj["qvel_offsets"], "qvel_offsets"),
            _nonnegative_int(obj["qvel_count"], "qvel_count"),
            _name_sequence(obj["actuator_names"], "actuator_names"),
            _int_sequence(obj["act_offsets"], "act_offsets"),
            _nonnegative_int(obj["act_count"], "act_count"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit state schema identity, names, offsets, and counts as canonical JSON values."""
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "joint_names": list(self.joint_names),
            "qpos_offsets": list(self.qpos_offsets),
            "qpos_count": self.qpos_count,
            "qvel_offsets": list(self.qvel_offsets),
            "qvel_count": self.qvel_count,
            "actuator_names": list(self.actuator_names),
            "act_offsets": list(self.act_offsets),
            "act_count": self.act_count,
        }


@dataclass(frozen=True, slots=True)
class ActionsArtifactMetadata:
    """Strict semantic metadata for an actions artifact."""

    schema: str
    schema_version: int
    actuator_names: tuple[str, ...]
    control_intervals: int
    actuator_count: int

    def __post_init__(self) -> None:
        """Validate actions schema identity, canonical actuator names, and matrix dimensions."""
        if self.schema != _ACTIONS_SCHEMA:
            raise ValueError("invalid actions artifact schema")
        if type(self.schema_version) is not int or self.schema_version != _ACTIONS_SCHEMA_VERSION:
            raise ValueError("invalid actions artifact schema_version")
        _require_string_tuple(self.actuator_names, "actuator_names", names=True)
        _unique_names(self.actuator_names, "actuator_names")
        _bounded_int(self.control_intervals, "control_intervals", 1, 100_000)
        _nonnegative_int(self.actuator_count, "actuator_count")
        if self.actuator_count != len(self.actuator_names):
            raise ValueError("actuator_count must equal len(actuator_names)")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode the exact actions-artifact metadata object and enforce its bounds."""
        obj = _object(value, "ActionsArtifactMetadata")
        expected = {
            "schema",
            "schema_version",
            "actuator_names",
            "control_intervals",
            "actuator_count",
        }
        _fields(obj, expected, "ActionsArtifactMetadata")
        return cls(
            _string(obj["schema"], "schema"),
            _exact_int(obj["schema_version"], "schema_version"),
            _name_sequence(obj["actuator_names"], "actuator_names"),
            _bounded_int(obj["control_intervals"], "control_intervals", 1, 100_000),
            _nonnegative_int(obj["actuator_count"], "actuator_count"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit actions schema identity, actuator order, and matrix dimensions as JSON values."""
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "actuator_names": list(self.actuator_names),
            "control_intervals": self.control_intervals,
            "actuator_count": self.actuator_count,
        }


@dataclass(frozen=True, slots=True)
class TargetReference:
    """One named transmission target in slot order."""

    object_type: str
    name: str

    def __post_init__(self) -> None:
        """Require a nonempty MuJoCo object kind and a valid semantic target name."""
        _nonempty_string(self.object_type, "object_type")
        _name(self.name, "name")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode an exact object-kind/name transmission target."""
        obj = _object(value, "TargetReference")
        _fields(obj, {"object_type", "name"}, "TargetReference")
        return cls(
            _nonempty_string(obj["object_type"], "object_type"),
            _name(obj["name"], "name"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the transmission target kind and name as a two-field JSON object."""
        return {"object_type": self.object_type, "name": self.name}


@dataclass(frozen=True, slots=True)
class ActuatorAliasEndpoint:
    """A named actuator endpoint or exact unnamed semantic selector."""

    kind: Literal["NAMED", "UNNAMED_SELECTOR"]
    name: str | None
    transmission_type: str | None
    targets: tuple[TargetReference, ...]
    activation_family: str | None
    activation_width: int | None

    def __post_init__(self) -> None:
        """Enforce the disjoint named and unnamed-selector endpoint shapes."""
        _require_typed_tuple(self.targets, TargetReference, "targets")
        if self.kind == "NAMED":
            if self.name is None:
                raise ValueError("NAMED endpoint requires name")
            _name(self.name, "name")
            if any(
                value is not None and value != ()
                for value in (
                    self.transmission_type,
                    self.targets,
                    self.activation_family,
                    self.activation_width,
                )
            ):
                raise ValueError("NAMED endpoint cannot contain selector fields")
        elif self.kind == "UNNAMED_SELECTOR":
            if self.name is not None:
                raise ValueError("UNNAMED_SELECTOR cannot contain name")
            _nonempty_string(self.transmission_type, "transmission_type")
            if not self.targets:
                raise ValueError("UNNAMED_SELECTOR requires at least one target")
            _nonempty_string(self.activation_family, "activation_family")
            _nonnegative_int(self.activation_width, "activation_width")
        else:
            raise ValueError("invalid actuator alias endpoint kind")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode either a named actuator or its exact unnamed semantic selector."""
        obj = _object(value, "ActuatorAliasEndpoint")
        kind = obj.get("kind")
        if kind == "NAMED":
            _fields(obj, {"kind", "name"}, "ActuatorAliasEndpoint")
            return cls("NAMED", _name(obj["name"], "name"), None, (), None, None)
        if kind == "UNNAMED_SELECTOR":
            expected = {
                "kind",
                "transmission_type",
                "targets",
                "activation_family",
                "activation_width",
            }
            _fields(obj, expected, "ActuatorAliasEndpoint")
            targets = tuple(
                TargetReference.from_primitive(item)
                for item in _sequence(obj["targets"], "targets")
            )
            return cls(
                "UNNAMED_SELECTOR",
                None,
                _nonempty_string(obj["transmission_type"], "transmission_type"),
                targets,
                _nonempty_string(obj["activation_family"], "activation_family"),
                _nonnegative_int(obj["activation_width"], "activation_width"),
            )
        raise ValueError("invalid actuator alias endpoint kind")

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit only the fields permitted by this endpoint's discriminated kind."""
        if self.kind == "NAMED":
            return {"kind": self.kind, "name": cast(str, self.name)}
        return {
            "kind": self.kind,
            "transmission_type": cast(str, self.transmission_type),
            "targets": [target.to_primitive() for target in self.targets],
            "activation_family": cast(str, self.activation_family),
            "activation_width": cast(int, self.activation_width),
        }


@dataclass(frozen=True, slots=True)
class JointAliasPair:
    """Bind one canonical joint name to its baseline and candidate names."""

    canonical_name: str
    baseline_name: str
    candidate_name: str

    def __post_init__(self) -> None:
        """Validate all three names in a cross-role joint alias binding."""
        _name(self.canonical_name, "canonical_name")
        _name(self.baseline_name, "baseline_name")
        _name(self.candidate_name, "candidate_name")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode an exact canonical/baseline/candidate joint-name binding."""
        obj = _object(value, "JointAliasPair")
        _fields(obj, {"canonical_name", "baseline_name", "candidate_name"}, "JointAliasPair")
        return cls(
            _name(obj["canonical_name"], "canonical_name"),
            _name(obj["baseline_name"], "baseline_name"),
            _name(obj["candidate_name"], "candidate_name"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the three names that define this joint alias binding."""
        return {
            "canonical_name": self.canonical_name,
            "baseline_name": self.baseline_name,
            "candidate_name": self.candidate_name,
        }


@dataclass(frozen=True, slots=True)
class ActuatorAliasPair:
    """Bind one canonical actuator name to role-local endpoint selectors."""

    canonical_name: str
    baseline: ActuatorAliasEndpoint
    candidate: ActuatorAliasEndpoint

    def __post_init__(self) -> None:
        """Validate the canonical name and both typed actuator endpoints."""
        _name(self.canonical_name, "canonical_name")
        _require_instance(self.baseline, ActuatorAliasEndpoint, "baseline")
        _require_instance(self.candidate, ActuatorAliasEndpoint, "candidate")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode a canonical actuator name with its two role-local endpoints."""
        obj = _object(value, "ActuatorAliasPair")
        _fields(obj, {"canonical_name", "baseline", "candidate"}, "ActuatorAliasPair")
        return cls(
            _name(obj["canonical_name"], "canonical_name"),
            ActuatorAliasEndpoint.from_primitive(obj["baseline"]),
            ActuatorAliasEndpoint.from_primitive(obj["candidate"]),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the canonical actuator name and nested role endpoint objects."""
        return {
            "canonical_name": self.canonical_name,
            "baseline": self.baseline.to_primitive(),
            "candidate": self.candidate.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class AliasArtifact:
    """Bind canonical joint and actuator aliases to two measured model closures."""

    schema: str
    schema_version: int
    baseline_model_closure_sha256: str
    candidate_model_closure_sha256: str
    joint_pairs: tuple[JointAliasPair, ...]
    actuator_pairs: tuple[ActuatorAliasPair, ...]

    def __post_init__(self) -> None:
        """Validate alias schema identity, closure hashes, and ordered unique bindings."""
        if self.schema != _ALIASES_SCHEMA:
            raise ValueError("invalid alias artifact schema")
        if type(self.schema_version) is not int or self.schema_version != _ALIASES_SCHEMA_VERSION:
            raise ValueError("invalid alias artifact schema_version")
        require_sha256(self.baseline_model_closure_sha256, "baseline_model_closure_sha256")
        require_sha256(self.candidate_model_closure_sha256, "candidate_model_closure_sha256")
        _require_typed_tuple(self.joint_pairs, JointAliasPair, "joint_pairs")
        _require_typed_tuple(self.actuator_pairs, ActuatorAliasPair, "actuator_pairs")
        _sorted_unique_pairs(self.joint_pairs, "joint_pairs")
        _sorted_unique_pairs(self.actuator_pairs, "actuator_pairs")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode an exact alias artifact with typed joint and actuator bindings."""
        obj = _object(value, "AliasArtifact")
        expected = {
            "schema",
            "schema_version",
            "baseline_model_closure_sha256",
            "candidate_model_closure_sha256",
            "joint_pairs",
            "actuator_pairs",
        }
        _fields(obj, expected, "AliasArtifact")
        return cls(
            _string(obj["schema"], "schema"),
            _exact_int(obj["schema_version"], "schema_version"),
            require_sha256(obj["baseline_model_closure_sha256"], "baseline_model_closure_sha256"),
            require_sha256(obj["candidate_model_closure_sha256"], "candidate_model_closure_sha256"),
            tuple(
                JointAliasPair.from_primitive(item)
                for item in _sequence(obj["joint_pairs"], "joint_pairs")
            ),
            tuple(
                ActuatorAliasPair.from_primitive(item)
                for item in _sequence(obj["actuator_pairs"], "actuator_pairs")
            ),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit closure bindings and ordered alias pairs as canonical JSON values."""
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "baseline_model_closure_sha256": self.baseline_model_closure_sha256,
            "candidate_model_closure_sha256": self.candidate_model_closure_sha256,
            "joint_pairs": [pair.to_primitive() for pair in self.joint_pairs],
            "actuator_pairs": [pair.to_primitive() for pair in self.actuator_pairs],
        }


def _sorted_unique_pairs(
    pairs: Sequence[JointAliasPair] | Sequence[ActuatorAliasPair],
    field: str,
) -> None:
    """Require alias pairs to have unique canonical names in code-point order."""
    names = tuple(pair.canonical_name for pair in pairs)
    _sorted_unique_names(names, field)
