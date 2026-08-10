"""Strict monitored-joint and comparison-contract schemas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self, cast

from ._configuration_schemas import JointToleranceConfig
from ._schema_constants import (
    _CONTRACT_SCHEMA,
    _CONTRACT_SCHEMA_VERSION,
    _METRICS_BY_JOINT_TYPE,
    JointType,
)
from ._schema_primitives import (
    _bounded_int,
    _exact_int,
    _fields,
    _name,
    _object,
    _optional_hash,
    _positive_rational,
    _positive_rational_primitive,
    _require_typed_tuple,
    _sequence,
    _string,
    _string_keyed_mapping,
)
from .json_values import CanonicalValue, ExactRational, canonical_sha256, require_sha256


@dataclass(frozen=True, slots=True)
class MonitoredJoint:
    """Bind one canonical joint and type to its exact comparison tolerances."""

    canonical_name: str
    joint_type: JointType
    tolerances: Mapping[str, ExactRational]

    def __post_init__(self) -> None:
        """Validate the canonical name and normalize joint-specific metric tolerances."""
        _name(self.canonical_name, "canonical_name")
        tolerances = _string_keyed_mapping(self.tolerances, "tolerances")
        JointToleranceConfig(self.joint_type, cast(Mapping[str, ExactRational], tolerances))
        normalized = {
            metric: cast(ExactRational, tolerances[metric])
            for metric in _METRICS_BY_JOINT_TYPE[self.joint_type]
        }
        object.__setattr__(self, "tolerances", MappingProxyType(normalized))

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode a monitored joint with exactly the metrics required by its type."""
        obj = _object(value, "MonitoredJoint")
        joint_type_raw = obj.get("joint_type")
        if joint_type_raw not in _METRICS_BY_JOINT_TYPE:
            raise ValueError("unsupported joint_type")
        joint_type = cast(JointType, joint_type_raw)
        expected = {"canonical_name", "joint_type", *_METRICS_BY_JOINT_TYPE[joint_type]}
        _fields(obj, expected, "MonitoredJoint")
        tolerances = {
            metric: _positive_rational_primitive(obj[metric], metric)
            for metric in _METRICS_BY_JOINT_TYPE[joint_type]
        }
        return cls(_name(obj["canonical_name"], "canonical_name"), joint_type, tolerances)

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit canonical joint identity and exact rational tolerance primitives."""
        result: dict[str, CanonicalValue] = {
            "canonical_name": self.canonical_name,
            "joint_type": self.joint_type,
        }
        result.update(
            {
                metric: self.tolerances[metric].to_primitive()
                for metric in _METRICS_BY_JOINT_TYPE[self.joint_type]
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class ComparisonContractIdentity:
    """Canonical decision-bearing identity of one declared comparison."""

    schema: str
    schema_version: int
    baseline_model_closure_sha256: str
    candidate_model_closure_sha256: str
    initial_state_semantic_sha256: str
    actions_semantic_sha256: str
    aliases_semantic_sha256: str | None
    baseline_step_dt: ExactRational
    candidate_step_dt: ExactRational
    control_dt: ExactRational
    repeats: int
    monitored_joints: tuple[MonitoredJoint, ...]

    def __post_init__(self) -> None:
        """Validate all input digests, exact time values, repeats, and monitored-joint order."""
        if self.schema != _CONTRACT_SCHEMA:
            raise ValueError("invalid comparison contract schema")
        if type(self.schema_version) is not int or self.schema_version != _CONTRACT_SCHEMA_VERSION:
            raise ValueError("invalid comparison contract schema_version")
        for field_name in (
            "baseline_model_closure_sha256",
            "candidate_model_closure_sha256",
            "initial_state_semantic_sha256",
            "actions_semantic_sha256",
        ):
            require_sha256(getattr(self, field_name), field_name)
        if self.aliases_semantic_sha256 is not None:
            require_sha256(self.aliases_semantic_sha256, "aliases_semantic_sha256")
        for field_name in ("baseline_step_dt", "candidate_step_dt", "control_dt"):
            _positive_rational(getattr(self, field_name), field_name)
        _bounded_int(self.repeats, "repeats", 2, 5)
        _require_typed_tuple(self.monitored_joints, MonitoredJoint, "monitored_joints")
        if not self.monitored_joints:
            raise ValueError("monitored_joints must be nonempty")
        names = tuple(joint.canonical_name for joint in self.monitored_joints)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("monitored_joints must be uniquely ordered by canonical_name")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode the complete decision-bearing comparison identity from canonical values."""
        obj = _object(value, "ComparisonContractIdentity")
        expected = {
            "schema",
            "schema_version",
            "baseline_model_closure_sha256",
            "candidate_model_closure_sha256",
            "initial_state_semantic_sha256",
            "actions_semantic_sha256",
            "aliases_semantic_sha256",
            "baseline_step_dt",
            "candidate_step_dt",
            "control_dt",
            "repeats",
            "monitored_joints",
        }
        _fields(obj, expected, "ComparisonContractIdentity")
        return cls(
            _string(obj["schema"], "schema"),
            _exact_int(obj["schema_version"], "schema_version"),
            require_sha256(obj["baseline_model_closure_sha256"], "baseline_model_closure_sha256"),
            require_sha256(obj["candidate_model_closure_sha256"], "candidate_model_closure_sha256"),
            require_sha256(obj["initial_state_semantic_sha256"], "initial_state_semantic_sha256"),
            require_sha256(obj["actions_semantic_sha256"], "actions_semantic_sha256"),
            _optional_hash(obj["aliases_semantic_sha256"], "aliases_semantic_sha256"),
            _positive_rational_primitive(obj["baseline_step_dt"], "baseline_step_dt"),
            _positive_rational_primitive(obj["candidate_step_dt"], "candidate_step_dt"),
            _positive_rational_primitive(obj["control_dt"], "control_dt"),
            _bounded_int(obj["repeats"], "repeats", 2, 5),
            tuple(
                MonitoredJoint.from_primitive(item)
                for item in _sequence(obj["monitored_joints"], "monitored_joints")
            ),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit model, workload, timing, repeat, and tolerance identities canonically."""
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "baseline_model_closure_sha256": self.baseline_model_closure_sha256,
            "candidate_model_closure_sha256": self.candidate_model_closure_sha256,
            "initial_state_semantic_sha256": self.initial_state_semantic_sha256,
            "actions_semantic_sha256": self.actions_semantic_sha256,
            "aliases_semantic_sha256": self.aliases_semantic_sha256,
            "baseline_step_dt": self.baseline_step_dt.to_primitive(),
            "candidate_step_dt": self.candidate_step_dt.to_primitive(),
            "control_dt": self.control_dt.to_primitive(),
            "repeats": self.repeats,
            "monitored_joints": [joint.to_primitive() for joint in self.monitored_joints],
        }

    def sha256(self) -> str:
        """Hash the complete canonical decision-bearing comparison identity."""
        return canonical_sha256(self.to_primitive())
