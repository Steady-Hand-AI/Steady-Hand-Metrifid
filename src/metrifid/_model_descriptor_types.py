"""Joint and actuator descriptors and the compiled-model semantic identity.

These describe what a compiled model *is* - its joints, actuators, transmission targets and
activation layout - independently of how it was compiled. ``_model_admission`` re-exports them.
"""

# Keep the baseline import statement order for rename-normalized AST identity.
from __future__ import annotations  # noqa: I001

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import ClassVar, Self, cast

from ._model_closure import (
    _JOINT_WIDTHS,
    ActivationFamily,
    JointType,
    _completed_hash,
    _require_exact_object_fields,
    _nonnegative_int,
    _strict_array,
    _strict_name,
    _valid_target_shape,
    _validate_activation_layout,
)
from ._model_identity_validation import (
    validate_represented_joint_targets,
    validate_unique_named_actuators,
)
from .json_values import (
    CanonicalValue,
    canonical_json_bytes,
    compute_self_hash,
    require_sha256,
    validate_self_hash,
)
from .schemas import TargetReference


def _covers(total: int, slices: Sequence[tuple[int, int]]) -> bool:
    """Return whether sorted nonempty slices exactly partition ``range(total)``."""
    cursor = 0
    for start, width in sorted(slices):
        if start != cursor or width <= 0:
            return False
        cursor += width
    return cursor == total


@dataclass(frozen=True, slots=True)
class JointDescriptor:
    """Describe one compiled joint's semantic kind and qpos/qvel layout."""

    name: str
    joint_type: JointType
    qpos_width: int
    qvel_width: int
    qpos_address: int
    qvel_address: int

    def __post_init__(self) -> None:
        """Match joint widths to type and validate both compiled-state addresses."""
        _strict_name(self.name, "name")
        if self.joint_type not in _JOINT_WIDTHS:
            raise ValueError("unsupported joint type")
        if (self.qpos_width, self.qvel_width) != _JOINT_WIDTHS[self.joint_type]:
            raise ValueError("joint widths do not match joint type")
        _nonnegative_int(self.qpos_address, "qpos_address")
        _nonnegative_int(self.qvel_address, "qvel_address")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode an exact compiled-joint name, type, widths, and addresses object."""
        fields = {"name", "joint_type", "qpos_width", "qvel_width", "qpos_address", "qvel_address"}
        obj = _require_exact_object_fields(value, fields, "JointDescriptor")
        return cls(
            cast(str, _strict_name(obj["name"], "name")),
            cast(JointType, _strict_name(obj["joint_type"], "joint_type")),
            cast(int, _nonnegative_int(obj["qpos_width"], "qpos_width")),
            cast(int, _nonnegative_int(obj["qvel_width"], "qvel_width")),
            cast(int, _nonnegative_int(obj["qpos_address"], "qpos_address")),
            cast(int, _nonnegative_int(obj["qvel_address"], "qvel_address")),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the joint's semantic identity and compiled-state layout."""
        return {
            "name": self.name,
            "joint_type": self.joint_type,
            "qpos_width": self.qpos_width,
            "qvel_width": self.qvel_width,
            "qpos_address": self.qpos_address,
            "qvel_address": self.qvel_address,
        }


@dataclass(frozen=True, slots=True)
class ActuatorDescriptor:
    """Describe one compiled actuator's transmission, targets, and state addresses."""

    name: str | None
    transmission_type: str
    targets: tuple[TargetReference, ...]
    activation_family: ActivationFamily
    activation_width: int
    control_address: int
    activation_address: int | None

    def __post_init__(self) -> None:
        """Validate transmission targets, control address, and activation layout."""
        _strict_name(self.name, "name", optional=True)
        _strict_name(self.transmission_type, "transmission_type")
        if not self.targets or not all(isinstance(x, TargetReference) for x in self.targets):
            raise ValueError("actuator targets must be a nonempty TargetReference tuple")
        if not _valid_target_shape(self.transmission_type, self.targets):
            raise ValueError("actuator targets do not match transmission type")
        _nonnegative_int(self.control_address, "control_address")
        _validate_activation_layout(
            self.activation_family,
            self.activation_width,
            (self.activation_address,),
            "activation address and width are inconsistent",
        )

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode the exact semantic and compiled-layout fields for one actuator."""
        fields = {
            "name",
            "transmission_type",
            "targets",
            "activation_family",
            "activation_width",
            "control_address",
            "activation_address",
        }
        obj = _require_exact_object_fields(value, fields, "ActuatorDescriptor")
        return cls(
            _strict_name(obj["name"], "name", optional=True),
            cast(str, _strict_name(obj["transmission_type"], "transmission_type")),
            tuple(
                TargetReference.from_primitive(item)
                for item in _strict_array(obj["targets"], "targets")
            ),
            cast(ActivationFamily, _strict_name(obj["activation_family"], "activation_family")),
            cast(int, _nonnegative_int(obj["activation_width"], "activation_width")),
            cast(int, _nonnegative_int(obj["control_address"], "control_address")),
            _nonnegative_int(obj["activation_address"], "activation_address", optional=True),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit actuator name, transmission semantics, and compiled addresses."""
        return {
            "name": self.name,
            "transmission_type": self.transmission_type,
            "targets": [target.to_primitive() for target in self.targets],
            "activation_family": self.activation_family,
            "activation_width": self.activation_width,
            "control_address": self.control_address,
            "activation_address": self.activation_address,
        }


def _actuator_sort_key(item: ActuatorDescriptor) -> tuple[bool, str, bytes, int]:
    """Order named actuators first, then unnamed actuators by canonical semantics and address."""
    semantic: dict[str, CanonicalValue] = {
        "transmission_type": item.transmission_type,
        "targets": [target.to_primitive() for target in item.targets],
        "activation_family": item.activation_family,
        "activation_width": item.activation_width,
    }
    return item.name is None, item.name or "", canonical_json_bytes(semantic), item.control_address


def _validate_compiled_members(identity: CompiledModelIdentity) -> None:
    """Validate descriptor types, order, targets, and actuator coverage."""
    if not all(isinstance(item, JointDescriptor) for item in identity.joints):
        raise TypeError("joints must contain JointDescriptor values")
    if not all(isinstance(item, ActuatorDescriptor) for item in identity.actuators):
        raise TypeError("actuators must contain ActuatorDescriptor values")
    joint_names = tuple(item.name for item in identity.joints)
    if joint_names != tuple(sorted(joint_names)) or len(joint_names) != len(set(joint_names)):
        raise ValueError("joints must be uniquely sorted by semantic name")
    if tuple(identity.actuators) != tuple(sorted(identity.actuators, key=_actuator_sort_key)):
        raise ValueError("actuators must be deterministically sorted")
    validate_represented_joint_targets(identity.joints, identity.actuators)
    validate_unique_named_actuators(identity.actuators)
    if identity.nu != len(identity.actuators):
        raise ValueError("nu must equal the actuator descriptor count")
    if {item.control_address for item in identity.actuators} != set(range(identity.nu)):
        raise ValueError("actuator control addresses must cover nu exactly")


def _validate_compiled_layout(identity: CompiledModelIdentity) -> None:
    """Validate exact qpos, qvel, and activation slice coverage."""
    qpos = [(item.qpos_address, item.qpos_width) for item in identity.joints]
    qvel = [(item.qvel_address, item.qvel_width) for item in identity.joints]
    act = [
        (item.activation_address, item.activation_width)
        for item in identity.actuators
        if item.activation_address is not None
    ]
    if not (
        _covers(identity.nq, qpos) and _covers(identity.nv, qvel) and _covers(identity.na, act)
    ):
        raise ValueError("compiled identity slices must cover nq, nv, and na exactly")


@dataclass(frozen=True, slots=True)
class CompiledModelIdentity:
    """Bind a measured model closure to its complete compiled joint and actuator layout."""

    schema: str
    schema_version: int
    compiled_identity_sha256: str | None
    model_closure_sha256: str
    nq: int
    nv: int
    nu: int
    na: int
    joints: tuple[JointDescriptor, ...]
    actuators: tuple[ActuatorDescriptor, ...]
    _SCHEMA: ClassVar[str] = "metrifid.compiled_model_identity"
    _SCHEMA_VERSION: ClassVar[int] = 1
    _HASH_FIELD: ClassVar[str] = "compiled_identity_sha256"

    def __post_init__(self) -> None:
        """Validate schema, closure binding, descriptor ordering, layout coverage, and self-hash."""
        if self.schema != self._SCHEMA:
            raise ValueError("invalid compiled model identity schema")
        if type(self.schema_version) is not int or self.schema_version != self._SCHEMA_VERSION:
            raise ValueError("invalid compiled model identity schema_version")
        require_sha256(self.model_closure_sha256, "model_closure_sha256")
        for field in ("nq", "nv", "nu", "na"):
            _nonnegative_int(getattr(self, field), field)
        _validate_compiled_members(self)
        _validate_compiled_layout(self)
        if self.compiled_identity_sha256 is not None:
            validate_self_hash(self._primitive(), self._HASH_FIELD)

    def _primitive(self) -> dict[str, CanonicalValue]:
        """Assemble the canonical compiled-model object, including an optional self-hash."""
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "compiled_identity_sha256": self.compiled_identity_sha256,
            "model_closure_sha256": self.model_closure_sha256,
            "nq": self.nq,
            "nv": self.nv,
            "nu": self.nu,
            "na": self.na,
            "joints": [item.to_primitive() for item in self.joints],
            "actuators": [item.to_primitive() for item in self.actuators],
        }

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit a completed compiled-model identity; reject an unhashed draft."""
        if self.compiled_identity_sha256 is None:
            raise ValueError("compiled model identity is not completed")
        return self._primitive()

    def finalized(self) -> Self:
        """Return this identity with its canonical self-hash populated."""
        if self.compiled_identity_sha256 is not None:
            return self
        return replace(
            self,
            compiled_identity_sha256=compute_self_hash(self._primitive(), self._HASH_FIELD),
        )

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode and self-hash-validate a complete compiled-model identity."""
        fields = {
            "schema",
            "schema_version",
            "compiled_identity_sha256",
            "model_closure_sha256",
            "nq",
            "nv",
            "nu",
            "na",
            "joints",
            "actuators",
        }
        obj = _require_exact_object_fields(value, fields, "CompiledModelIdentity")
        result = cls(
            cast(str, _strict_name(obj["schema"], "schema")),
            cast(int, _nonnegative_int(obj["schema_version"], "schema_version")),
            _completed_hash(obj["compiled_identity_sha256"], "compiled_identity_sha256"),
            require_sha256(obj["model_closure_sha256"], "model_closure_sha256"),
            cast(int, _nonnegative_int(obj["nq"], "nq")),
            cast(int, _nonnegative_int(obj["nv"], "nv")),
            cast(int, _nonnegative_int(obj["nu"], "nu")),
            cast(int, _nonnegative_int(obj["na"], "na")),
            tuple(
                JointDescriptor.from_primitive(item)
                for item in _strict_array(obj["joints"], "joints")
            ),
            tuple(
                ActuatorDescriptor.from_primitive(item)
                for item in _strict_array(obj["actuators"], "actuators")
            ),
        )
        validate_self_hash(result.to_primitive(), result._HASH_FIELD)
        return result
