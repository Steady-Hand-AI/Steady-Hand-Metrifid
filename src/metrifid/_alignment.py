"""Strict aligned model value objects and their primitive codecs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Self, TypeAlias, cast

from .json_values import CanonicalValue, require_sha256
from .schemas import TargetReference

JointType: TypeAlias = Literal["FREE", "BALL", "SLIDE", "HINGE"]
ActivationFamily: TypeAlias = Literal[
    "NONE", "INTEGRATOR", "FILTER", "FILTEREXACT", "MUSCLE", "DCMOTOR"
]
_JOINT_WIDTHS: dict[JointType, tuple[int, int]] = {
    "FREE": (7, 6),
    "BALL": (4, 3),
    "SLIDE": (1, 1),
    "HINGE": (1, 1),
}
_ACTIVATION_LAYOUT_WIDTHS: Final[Mapping[ActivationFamily, tuple[int, ...]]] = MappingProxyType(
    {
        "NONE": (0,),
        **dict.fromkeys(("INTEGRATOR", "FILTER", "FILTEREXACT", "MUSCLE"), (1,)),
        "DCMOTOR": (0, 1),
    }
)


def _require_exact_object_fields(
    value: object, fields: set[str], context: str
) -> dict[str, object]:
    """Require an alignment mapping with exactly the fields frozen by its serialized schema."""
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    obj = cast(dict[str, object], value)
    if set(obj) != fields:
        raise ValueError(f"{context} fields do not match the frozen schema")
    return obj


def _strict_name(value: object, field: str, *, optional: bool = False) -> str | None:
    """Admit a nonempty UTF-8 alignment name, or ``None`` for an optional field."""
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value or "\x00" in value:
        raise ValueError(f"{field} must be a nonempty name")
    value.encode("utf-8", errors="strict")
    return value


def _nonnegative_int(value: object, field: str, *, optional: bool = False) -> int | None:
    """Admit a nonnegative alignment index while excluding booleans and invalid nulls."""
    if value is None and optional:
        return None
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer and not a boolean")
    if value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return value


def _validate_activation_layout(
    family: ActivationFamily,
    width: int,
    addresses: Sequence[int | None],
    address_error: str,
) -> None:
    """Match an actuator family to its allowed activation width and role addresses."""
    allowed_widths = _ACTIVATION_LAYOUT_WIDTHS.get(family)
    if allowed_widths is None:
        raise ValueError("unsupported activation family")
    _nonnegative_int(width, "activation_width")
    validated = tuple(
        _nonnegative_int(value, "activation_address", optional=True) for value in addresses
    )
    if width not in allowed_widths:
        raise ValueError("activation family and width are inconsistent")
    invalid = (
        any(value is not None for value in validated)
        if width == 0
        else any(value is None for value in validated)
    )
    if invalid:
        raise ValueError(address_error)


def _strict_array(value: object, field: str) -> list[object]:
    """Admit only a JSON-array value for an alignment field."""
    if type(value) is not list:
        raise TypeError(f"{field} must be an array")
    return cast(list[object], value)


def _completed_hash(value: object, field: str) -> str:
    """Require a completed alignment field to carry a valid lowercase SHA-256 digest."""
    if value is None:
        raise ValueError(f"{field} must be present on a completed object")
    return require_sha256(value, field)


def _slice(value: object, field: str) -> tuple[int, int]:
    """Parse a JSON ``[start, width]`` pair into a nonempty model-array slice."""
    values = _strict_array(value, field)
    if len(values) != 2:
        raise ValueError(f"{field} must have two elements")
    start = cast(int, _nonnegative_int(values[0], field))
    width = cast(int, _nonnegative_int(values[1], field))
    if width == 0:
        raise ValueError(f"{field} width must be positive")
    return start, width


def _valid_target_shape(transmission: str, targets: tuple[TargetReference, ...]) -> bool:
    """Return whether target object kinds match the MuJoCo transmission family."""
    kinds = tuple(item.object_type for item in targets)
    expected = {
        "JOINT": (("JOINT",),),
        "JOINTINPARENT": (("JOINT",),),
        "TENDON": (("TENDON",),),
        "SLIDERCRANK": (("SITE", "SITE"),),
        "SITE": (("SITE",), ("SITE", "SITE")),
        "BODY": (("BODY",),),
    }
    return kinds in expected.get(transmission, ())


def _validate_slice(value: tuple[int, int], field: str, expected_width: int | None = None) -> None:
    """Validate a model-array slice and, when known, its joint-specific width."""
    if type(value) is not tuple or len(value) != 2:
        raise TypeError(f"{field} must be a two-integer tuple")
    _nonnegative_int(value[0], field)
    if type(value[1]) is not int or value[1] <= 0:
        raise ValueError(f"{field} width must be positive")
    if expected_width is not None and value[1] != expected_width:
        raise ValueError(f"{field} width does not match the joint type")


@dataclass(frozen=True, slots=True)
class AlignedJoint:
    """Bind one canonical joint to its baseline and candidate state slices."""

    canonical_name: str
    joint_type: JointType
    baseline_qpos: tuple[int, int]
    candidate_qpos: tuple[int, int]
    baseline_qvel: tuple[int, int]
    candidate_qvel: tuple[int, int]

    def __post_init__(self) -> None:
        """Validate the joint kind and its role-local qpos and qvel slice widths."""
        _strict_name(self.canonical_name, "canonical_name")
        if self.joint_type not in _JOINT_WIDTHS:
            raise ValueError("unsupported joint type")
        qpos_width, qvel_width = _JOINT_WIDTHS[self.joint_type]
        _validate_slice(self.baseline_qpos, "baseline_qpos", qpos_width)
        _validate_slice(self.candidate_qpos, "candidate_qpos", qpos_width)
        _validate_slice(self.baseline_qvel, "baseline_qvel", qvel_width)
        _validate_slice(self.candidate_qvel, "candidate_qvel", qvel_width)

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode the exact six-field aligned-joint object and validate every slice."""
        fields = {
            "canonical_name",
            "joint_type",
            "baseline_qpos",
            "candidate_qpos",
            "baseline_qvel",
            "candidate_qvel",
        }
        obj = _require_exact_object_fields(value, fields, "AlignedJoint")
        return cls(
            cast(str, _strict_name(obj["canonical_name"], "canonical_name")),
            cast(JointType, _strict_name(obj["joint_type"], "joint_type")),
            _slice(obj["baseline_qpos"], "baseline_qpos"),
            _slice(obj["candidate_qpos"], "candidate_qpos"),
            _slice(obj["baseline_qvel"], "baseline_qvel"),
            _slice(obj["candidate_qvel"], "candidate_qvel"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit canonical joint identity and role-local state slices as JSON values."""
        return {
            "canonical_name": self.canonical_name,
            "joint_type": self.joint_type,
            "baseline_qpos": list(self.baseline_qpos),
            "candidate_qpos": list(self.candidate_qpos),
            "baseline_qvel": list(self.baseline_qvel),
            "candidate_qvel": list(self.candidate_qvel),
        }


@dataclass(frozen=True, slots=True)
class AlignedActuator:
    """Bind one canonical actuator to equivalent role-local controls and activation state."""

    canonical_name: str
    transmission_type: str
    targets: tuple[TargetReference, ...]
    activation_family: ActivationFamily
    activation_width: int
    baseline_control_address: int
    candidate_control_address: int
    baseline_activation_address: int | None
    candidate_activation_address: int | None

    def __post_init__(self) -> None:
        """Validate transmission targets, control addresses, and activation layout."""
        _strict_name(self.canonical_name, "canonical_name")
        _strict_name(self.transmission_type, "transmission_type")
        if not self.targets or not all(isinstance(item, TargetReference) for item in self.targets):
            raise ValueError("aligned actuator requires named targets")
        if not _valid_target_shape(self.transmission_type, self.targets):
            raise ValueError("aligned actuator targets do not match transmission type")
        _nonnegative_int(self.baseline_control_address, "baseline_control_address")
        _nonnegative_int(self.candidate_control_address, "candidate_control_address")
        _validate_activation_layout(
            self.activation_family,
            self.activation_width,
            (self.baseline_activation_address, self.candidate_activation_address),
            "aligned activation addresses and width are inconsistent",
        )

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode and validate the exact aligned-actuator evidence object."""
        fields = {
            "canonical_name",
            "transmission_type",
            "targets",
            "activation_family",
            "activation_width",
            "baseline_control_address",
            "candidate_control_address",
            "baseline_activation_address",
            "candidate_activation_address",
        }
        obj = _require_exact_object_fields(value, fields, "AlignedActuator")
        return cls(
            cast(str, _strict_name(obj["canonical_name"], "canonical_name")),
            cast(str, _strict_name(obj["transmission_type"], "transmission_type")),
            tuple(
                TargetReference.from_primitive(item)
                for item in _strict_array(obj["targets"], "targets")
            ),
            cast(ActivationFamily, _strict_name(obj["activation_family"], "activation_family")),
            cast(int, _nonnegative_int(obj["activation_width"], "activation_width")),
            cast(
                int, _nonnegative_int(obj["baseline_control_address"], "baseline_control_address")
            ),
            cast(
                int, _nonnegative_int(obj["candidate_control_address"], "candidate_control_address")
            ),
            _nonnegative_int(
                obj["baseline_activation_address"], "baseline_activation_address", optional=True
            ),
            _nonnegative_int(
                obj["candidate_activation_address"], "candidate_activation_address", optional=True
            ),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit actuator targets, layouts, and role-local addresses as canonical JSON values."""
        return {
            "canonical_name": self.canonical_name,
            "transmission_type": self.transmission_type,
            "targets": [item.to_primitive() for item in self.targets],
            "activation_family": self.activation_family,
            "activation_width": self.activation_width,
            "baseline_control_address": self.baseline_control_address,
            "candidate_control_address": self.candidate_control_address,
            "baseline_activation_address": self.baseline_activation_address,
            "candidate_activation_address": self.candidate_activation_address,
        }


def _unique_sorted_names(values: Sequence[AlignedJoint] | Sequence[AlignedActuator]) -> bool:
    """Return whether canonical alignment names are unique and code-point ordered."""
    names = tuple(item.canonical_name for item in values)
    return names == tuple(sorted(names)) and len(names) == len(set(names))


def _canonical_targets(
    targets: tuple[TargetReference, ...], joint_names: Mapping[str, str]
) -> tuple[TargetReference, ...]:
    """Replace joint target names with their aliases while preserving non-joint targets."""
    return tuple(
        TargetReference(
            item.object_type,
            joint_names[item.name] if item.object_type == "JOINT" else item.name,
        )
        for item in targets
    )
