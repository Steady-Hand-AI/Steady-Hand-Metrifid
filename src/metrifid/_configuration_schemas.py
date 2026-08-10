"""Strict configuration schemas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self, cast

from ._schema_constants import _METRICS_BY_JOINT_TYPE, JointType
from ._schema_primitives import (
    _bounded_int,
    _exact_int,
    _fields,
    _name,
    _nonempty_string,
    _object,
    _positive_decimal,
    _positive_rational,
    _require_instance,
    _string_keyed_mapping,
)
from .json_values import CanonicalValue, ExactRational


@dataclass(frozen=True, slots=True)
class ModelRoleConfig:
    """One role's source-location and declared exact step period."""

    model_root: str
    entrypoint: str
    declared_step_dt: ExactRational

    def __post_init__(self) -> None:
        """Validate one role's model paths and positive exact declared timestep."""
        _nonempty_string(self.model_root, "model_root")
        _nonempty_string(self.entrypoint, "entrypoint")
        _positive_rational(self.declared_step_dt, "declared_step_dt")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode model root, entrypoint, and decimal timestep for one role."""
        obj = _object(value, "ModelRoleConfig")
        _fields(obj, {"model_root", "entrypoint", "declared_step_dt"}, "ModelRoleConfig")
        return cls(
            _nonempty_string(obj["model_root"], "model_root"),
            _nonempty_string(obj["entrypoint"], "entrypoint"),
            _positive_decimal(obj["declared_step_dt"], "declared_step_dt"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit role paths and the exact timestep as its ordinary-decimal token."""
        return {
            "model_root": self.model_root,
            "entrypoint": self.entrypoint,
            "declared_step_dt": self.declared_step_dt.to_decimal_token(),
        }


@dataclass(frozen=True, slots=True)
class JointToleranceConfig:
    """Exact typed tolerance declarations for one monitored joint."""

    joint_type: JointType
    tolerances: Mapping[str, ExactRational]

    def __post_init__(self) -> None:
        """Require exactly the positive metrics defined for the declared joint type."""
        if self.joint_type not in _METRICS_BY_JOINT_TYPE:
            raise ValueError("unsupported joint_type")
        tolerances = _string_keyed_mapping(self.tolerances, "tolerances")
        for value in tolerances.values():
            if not isinstance(value, ExactRational):
                raise TypeError("tolerances values must be ExactRational")
        expected = set(_METRICS_BY_JOINT_TYPE[self.joint_type])
        actual = set(tolerances)
        if actual != expected:
            raise ValueError("tolerance fields do not match the declared joint type")
        normalized: dict[str, ExactRational] = {}
        for metric in sorted(expected):
            value = cast(ExactRational, tolerances[metric])
            _positive_rational(value, metric)
            normalized[metric] = value
        object.__setattr__(self, "tolerances", MappingProxyType(normalized))

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode joint-specific decimal tolerance fields into exact rationals."""
        obj = _object(value, "JointToleranceConfig")
        joint_type_raw = obj.get("joint_type")
        if joint_type_raw not in _METRICS_BY_JOINT_TYPE:
            raise ValueError("unsupported joint_type")
        joint_type = cast(JointType, joint_type_raw)
        expected = {"joint_type", *_METRICS_BY_JOINT_TYPE[joint_type]}
        _fields(obj, expected, "JointToleranceConfig")
        tolerances = {
            metric: _positive_decimal(obj[metric], metric)
            for metric in _METRICS_BY_JOINT_TYPE[joint_type]
        }
        return cls(joint_type, tolerances)

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit joint type and tolerances using the frozen decimal-token surface."""
        result: dict[str, CanonicalValue] = {"joint_type": self.joint_type}
        result.update(
            {
                metric: self.tolerances[metric].to_decimal_token()
                for metric in _METRICS_BY_JOINT_TYPE[self.joint_type]
            }
        )
        return result

    def semantic_primitive(self) -> dict[str, CanonicalValue]:
        """Emit joint tolerances as normalized numerator/denominator semantic values."""
        result: dict[str, CanonicalValue] = {"joint_type": self.joint_type}
        result.update(
            {
                metric: self.tolerances[metric].to_primitive()
                for metric in _METRICS_BY_JOINT_TYPE[self.joint_type]
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    """Strict shape of the comparison configuration."""

    schema_version: int
    baseline: ModelRoleConfig
    candidate: ModelRoleConfig
    initial_state: str
    actions: str
    control_dt: ExactRational
    repeats: int
    joint_tolerances: Mapping[str, JointToleranceConfig]
    aliases: str | None
    output_dir: str

    def __post_init__(self) -> None:
        """Validate paths, exact timing, repeat bounds, and canonical joint tolerances."""
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        _require_instance(self.baseline, ModelRoleConfig, "baseline")
        _require_instance(self.candidate, ModelRoleConfig, "candidate")
        _nonempty_string(self.initial_state, "initial_state")
        _nonempty_string(self.actions, "actions")
        _positive_rational(self.control_dt, "control_dt")
        _bounded_int(self.repeats, "repeats", 2, 5)
        joint_tolerances = _string_keyed_mapping(self.joint_tolerances, "joint_tolerances")
        for tolerance in joint_tolerances.values():
            if not isinstance(tolerance, JointToleranceConfig):
                raise TypeError("joint_tolerances values must be JointToleranceConfig")
        if not joint_tolerances:
            raise ValueError("joint_tolerances must be nonempty")
        canonical_names = {name: _name(name, "joint_tolerances key") for name in joint_tolerances}
        normalized: dict[str, JointToleranceConfig] = {}
        for name in sorted(canonical_names):
            normalized[canonical_names[name]] = cast(JointToleranceConfig, joint_tolerances[name])
        object.__setattr__(self, "joint_tolerances", MappingProxyType(normalized))
        if self.aliases is not None:
            _nonempty_string(self.aliases, "aliases")
        _nonempty_string(self.output_dir, "output_dir")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode the exact comparison configuration and normalize tolerance ordering."""
        obj = _object(value, "ComparisonConfig")
        expected = {
            "schema_version",
            "baseline",
            "candidate",
            "initial_state",
            "actions",
            "control_dt",
            "repeats",
            "joint_tolerances",
            "aliases",
            "output_dir",
        }
        _fields(obj, expected, "ComparisonConfig")
        tolerances_raw = _object(obj["joint_tolerances"], "joint_tolerances")
        tolerances = {
            _name(name, "joint_tolerances key"): JointToleranceConfig.from_primitive(item)
            for name, item in tolerances_raw.items()
        }
        aliases_raw = obj["aliases"]
        aliases = None if aliases_raw is None else _nonempty_string(aliases_raw, "aliases")
        return cls(
            _exact_int(obj["schema_version"], "schema_version"),
            ModelRoleConfig.from_primitive(obj["baseline"]),
            ModelRoleConfig.from_primitive(obj["candidate"]),
            _nonempty_string(obj["initial_state"], "initial_state"),
            _nonempty_string(obj["actions"], "actions"),
            _positive_decimal(obj["control_dt"], "control_dt"),
            _bounded_int(obj["repeats"], "repeats", 2, 5),
            tolerances,
            aliases,
            _nonempty_string(obj["output_dir"], "output_dir"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the complete comparison configuration using exact decimal time tokens."""
        return {
            "schema_version": self.schema_version,
            "baseline": self.baseline.to_primitive(),
            "candidate": self.candidate.to_primitive(),
            "initial_state": self.initial_state,
            "actions": self.actions,
            "control_dt": self.control_dt.to_decimal_token(),
            "repeats": self.repeats,
            "joint_tolerances": {
                name: tolerance.to_primitive() for name, tolerance in self.joint_tolerances.items()
            },
            "aliases": self.aliases,
            "output_dir": self.output_dir,
        }
