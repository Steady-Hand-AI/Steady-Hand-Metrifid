"""Rolling MuJoCo runtime and claim-surface admission.

This module deliberately resolves MuJoCo capabilities at the admission boundary.  Optional or
future-sensitive accessors are therefore never bound while the module is imported, and a changed
runtime produces a typed operational refusal instead of an ``AttributeError``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from operator import index as integer_index
from types import MappingProxyType
from typing import Final, Literal, TypeAlias, cast

import mujoco  # type: ignore[import-untyped]

from ._model_refusal import ModelRole, refuse
from .json_values import CanonicalValue
from .operational import OperationalReasonCode

VersionTriplet: TypeAlias = tuple[int, int, int]
CapabilityKind: TypeAlias = Literal["ATTRIBUTE", "CALLABLE"]

_MINIMUM_BASE_VERSION: Final[VersionTriplet] = (3, 9, 0)
_STABLE_PACKAGE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\A(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)"
    r"(?:\.post(?:0|[1-9][0-9]*))?"
    r"(?:\+[0-9A-Za-z]+(?:[._-][0-9A-Za-z]+)*)?\Z",
)
_STABLE_NATIVE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\A(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)\Z"
)
_MODEL_FEATURE_MEMBERS: Final[tuple[str, ...]] = (
    "nactuator",
    "nout",
    "actuator_ctrlnum",
    "actuator_outnum",
    "actuator_ctrlspec",
    "geom_surfacevel",
    "pair_surfacevel",
    "geom_adhesion",
    "pair_adhesion",
)
_MISSING: Final = object()


class MujocoClaimSurface(StrEnum):
    """The three native claim surfaces with independently measured call graphs."""

    COMPILED_ARTIFACT = "COMPILED_ARTIFACT"
    STATIC_MODEL_REVIEW = "STATIC_MODEL_REVIEW"
    DYNAMIC_REPLAY = "DYNAMIC_REPLAY"


class MujocoSupportTier(StrEnum):
    """The two successful rolling-runtime support classifications."""

    VALIDATED_EXACT_PROFILE = "VALIDATED_EXACT_PROFILE"
    ADMITTED_CAPABILITY_COMPATIBLE_PROFILE = "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE"


@dataclass(frozen=True, slots=True)
class MujocoCapabilityRequirement:
    """One lazily resolved MuJoCo module or class capability."""

    name: str
    kind: CapabilityKind


@dataclass(frozen=True, slots=True)
class MujocoRuntimeAdmission:
    """One coherent runtime identity admitted for a specific claim surface."""

    package_version: str
    package_base_version: str
    native_version_string: str
    native_version_integer: int
    support_tier: MujocoSupportTier
    operation: MujocoClaimSurface
    measured_capabilities: tuple[str, ...]
    measured_feature_facts: tuple[tuple[str, bool], ...]

    def to_evidence(self) -> dict[str, CanonicalValue]:
        """Return canonical evidence shared by capability and feature decisions."""
        return {
            "operation": self.operation.value,
            "detected_package_version": self.package_version,
            "detected_native_version_string": self.native_version_string,
            "detected_native_version_integer": self.native_version_integer,
            "package_base_version": self.package_base_version,
            "support_tier": self.support_tier.value,
            "measured_capabilities": list(self.measured_capabilities),
            "measured_feature_facts": dict(self.measured_feature_facts),
        }


@dataclass(frozen=True, slots=True)
class MujocoActuatorSignature:
    """The measured input/output signature of one compiled actuator."""

    actuator_index: int
    control_inputs: int
    force_outputs: int
    control_spec: int | None

    def to_evidence(self) -> dict[str, CanonicalValue]:
        """Return a canonical signature object suitable for refusal evidence."""
        return {
            "actuator_index": self.actuator_index,
            "control_inputs": self.control_inputs,
            "force_outputs": self.force_outputs,
            "control_spec": self.control_spec,
        }


@dataclass(frozen=True, slots=True)
class MujocoModelFeatureFacts:
    """Measured claim-relevant facts from one already compiled model."""

    actuator_count: int
    control_width: int
    output_width: int
    legacy_implicit_signature: bool
    actuator_signatures: tuple[MujocoActuatorSignature, ...]
    signature_measurement_issues: tuple[str, ...]
    surface_velocity_fields: tuple[str, ...]
    active_surface_velocity_fields: tuple[str, ...]
    adhesion_fields: tuple[str, ...]
    active_adhesion_fields: tuple[str, ...]

    def unsupported_actuator_signatures(self) -> tuple[MujocoActuatorSignature, ...]:
        """Return every signature the current one-actuator projection cannot represent."""
        return tuple(
            signature
            for signature in self.actuator_signatures
            if signature.control_inputs != 1 or signature.force_outputs != 1
        )

    def to_evidence(self) -> dict[str, CanonicalValue]:
        """Return complete canonical model-feature observations."""
        return {
            "actuator_count": self.actuator_count,
            "control_width": self.control_width,
            "output_width": self.output_width,
            "legacy_implicit_signature": self.legacy_implicit_signature,
            "actuator_signatures": [item.to_evidence() for item in self.actuator_signatures],
            "signature_measurement_issues": list(self.signature_measurement_issues),
            "surface_velocity_fields": list(self.surface_velocity_fields),
            "active_surface_velocity_fields": list(self.active_surface_velocity_fields),
            "adhesion_fields": list(self.adhesion_fields),
            "active_adhesion_fields": list(self.active_adhesion_fields),
        }


def _requirement(name: str, kind: CapabilityKind = "CALLABLE") -> MujocoCapabilityRequirement:
    """Build one compact immutable capability requirement."""
    return MujocoCapabilityRequirement(name, kind)


_COMPILE_REQUIREMENTS: Final[tuple[MujocoCapabilityRequirement, ...]] = (
    _requirement("__version__", "ATTRIBUTE"),
    _requirement("__file__", "ATTRIBUTE"),
    _requirement("mj_versionString"),
    _requirement("mj_version"),
    _requirement("MjModel", "ATTRIBUTE"),
    _requirement("MjModel.from_xml_path"),
    _requirement("FatalError"),
    _requirement("mju_getXMLDependencies"),
    _requirement("get_mjcb_control"),
    _requirement("get_mjcb_sensor"),
    _requirement("get_mjcb_passive"),
    _requirement("get_mjcb_act_dyn"),
    _requirement("get_mjcb_act_gain"),
    _requirement("get_mjcb_act_bias"),
    _requirement("get_mjcb_contactfilter"),
    _requirement("get_mjcb_time"),
    _requirement("get_mju_user_warning"),
    _requirement("get_mju_user_malloc"),
    _requirement("get_mju_user_free"),
    _requirement("set_mju_user_warning"),
    _requirement("mjtDyn.mjDYN_USER", "ATTRIBUTE"),
    _requirement("mjtGain.mjGAIN_USER", "ATTRIBUTE"),
    _requirement("mjtBias.mjBIAS_USER", "ATTRIBUTE"),
    _requirement("mjtSensor.mjSENS_USER", "ATTRIBUTE"),
    _requirement("mjtSensor.mjSENS_PLUGIN", "ATTRIBUTE"),
)
_SERIALIZATION_REQUIREMENTS: Final[tuple[MujocoCapabilityRequirement, ...]] = (
    _requirement("MjModel.from_binary_path"),
    _requirement("mj_sizeModel"),
    _requirement("mj_saveModel"),
)
_DYNAMIC_DESCRIPTOR_REQUIREMENTS: Final[tuple[MujocoCapabilityRequirement, ...]] = (
    _requirement("mjtJoint.mjJNT_FREE", "ATTRIBUTE"),
    _requirement("mjtJoint.mjJNT_BALL", "ATTRIBUTE"),
    _requirement("mjtJoint.mjJNT_SLIDE", "ATTRIBUTE"),
    _requirement("mjtJoint.mjJNT_HINGE", "ATTRIBUTE"),
    _requirement("mjtDyn.mjDYN_NONE", "ATTRIBUTE"),
    _requirement("mjtDyn.mjDYN_INTEGRATOR", "ATTRIBUTE"),
    _requirement("mjtDyn.mjDYN_FILTER", "ATTRIBUTE"),
    _requirement("mjtDyn.mjDYN_FILTEREXACT", "ATTRIBUTE"),
    _requirement("mjtDyn.mjDYN_MUSCLE", "ATTRIBUTE"),
    _requirement("mjtDyn.mjDYN_DCMOTOR", "ATTRIBUTE"),
)
_NAMED_REFERENCE_REQUIREMENTS: Final[tuple[MujocoCapabilityRequirement, ...]] = (
    _requirement("mj_id2name"),
    _requirement("mjtTrn.mjTRN_JOINT", "ATTRIBUTE"),
    _requirement("mjtTrn.mjTRN_JOINTINPARENT", "ATTRIBUTE"),
    _requirement("mjtTrn.mjTRN_SLIDERCRANK", "ATTRIBUTE"),
    _requirement("mjtTrn.mjTRN_TENDON", "ATTRIBUTE"),
    _requirement("mjtTrn.mjTRN_SITE", "ATTRIBUTE"),
    _requirement("mjtTrn.mjTRN_BODY", "ATTRIBUTE"),
    _requirement("mjtObj.mjOBJ_JOINT", "ATTRIBUTE"),
    _requirement("mjtObj.mjOBJ_ACTUATOR", "ATTRIBUTE"),
    _requirement("mjtObj.mjOBJ_TENDON", "ATTRIBUTE"),
    _requirement("mjtObj.mjOBJ_SITE", "ATTRIBUTE"),
    _requirement("mjtObj.mjOBJ_BODY", "ATTRIBUTE"),
)
_STATIC_REVIEW_REQUIREMENTS: Final[tuple[MujocoCapabilityRequirement, ...]] = (
    _requirement("mjtObj.mjOBJ_GEOM", "ATTRIBUTE"),
    _requirement("mjtObj.mjOBJ_MESH", "ATTRIBUTE"),
)
_DYNAMIC_REQUIREMENTS: Final[tuple[MujocoCapabilityRequirement, ...]] = (
    _requirement("MjData"),
    _requirement("mj_forward"),
    _requirement("mj_step"),
)

MUJOCO_CAPABILITY_INVENTORY: Final[
    Mapping[MujocoClaimSurface, tuple[MujocoCapabilityRequirement, ...]]
] = MappingProxyType(
    {
        MujocoClaimSurface.COMPILED_ARTIFACT: (
            *_COMPILE_REQUIREMENTS,
            *_SERIALIZATION_REQUIREMENTS,
        ),
        MujocoClaimSurface.STATIC_MODEL_REVIEW: (
            *_COMPILE_REQUIREMENTS,
            *_SERIALIZATION_REQUIREMENTS,
            *_NAMED_REFERENCE_REQUIREMENTS,
            *_STATIC_REVIEW_REQUIREMENTS,
        ),
        MujocoClaimSurface.DYNAMIC_REPLAY: (
            *_COMPILE_REQUIREMENTS,
            *_DYNAMIC_DESCRIPTOR_REQUIREMENTS,
            *_NAMED_REFERENCE_REQUIREMENTS,
            *_DYNAMIC_REQUIREMENTS,
        ),
    }
)


def _parse_stable_version(value: object, *, package: bool) -> VersionTriplet | None:
    """Parse an unambiguous stable package or exact native version into an integer triplet."""
    if not isinstance(value, str):
        return None
    pattern = _STABLE_PACKAGE_PATTERN if package else _STABLE_NATIVE_PATTERN
    match = pattern.match(value)
    if match is None:
        return None
    triplet = cast(
        VersionTriplet,
        tuple(int(match.group(component)) for component in ("major", "minor", "patch")),
    )
    return triplet if all(component < 1000 for component in triplet) else None


def _base_version(triplet: VersionTriplet) -> str:
    """Render one parsed version triplet in its canonical stable form."""
    return ".".join(str(component) for component in triplet)


def _native_integer(triplet: VersionTriplet) -> int:
    """Encode one bounded native version triplet without a release lookup table."""
    if any(component < 0 or component >= 1000 for component in triplet):
        raise ValueError("MuJoCo native version components must be between zero and 999")
    major, minor, patch = triplet
    return major * 1_000_000 + minor * 1_000 + patch


def _resolve_capability(runtime_module: object, name: str) -> object | None:
    """Resolve one dotted capability lazily, returning null for any absent segment."""
    value = runtime_module
    for segment in name.split("."):
        try:
            value = getattr(value, segment)
        except (AttributeError, RuntimeError):
            return None
    return value


def _missing_capabilities(runtime_module: object, operation: MujocoClaimSurface) -> tuple[str, ...]:
    """Return every absent or wrong-kind capability required by one operation."""
    missing: list[str] = []
    for requirement in MUJOCO_CAPABILITY_INVENTORY[operation]:
        resolved = _resolve_capability(runtime_module, requirement.name)
        if resolved is None or (requirement.kind == "CALLABLE" and not callable(resolved)):
            missing.append(requirement.name)
    return tuple(missing)


def _runtime_feature_facts(runtime_module: object) -> tuple[tuple[str, bool], ...]:
    """Measure the optional model feature descriptors exposed by this binding."""
    model_class = _resolve_capability(runtime_module, "MjModel")
    return tuple(
        (member, model_class is not None and _resolve_capability(model_class, member) is not None)
        for member in _MODEL_FEATURE_MEMBERS
    )


def _capability_refusal(
    operation: MujocoClaimSurface,
    package_version: object,
    native_string: object,
    native_integer: object,
    missing: Sequence[str],
    support_tier: MujocoSupportTier | None,
) -> Exception:
    """Build one teachable typed refusal for a missing operation capability."""
    evidence: dict[str, CanonicalValue] = {
        "operation": operation.value,
        "detected_package_version": package_version if isinstance(package_version, str) else None,
        "detected_native_version_string": native_string if isinstance(native_string, str) else None,
        "detected_native_version_integer": (
            native_integer if type(native_integer) is int else None
        ),
        "missing_capabilities": list(missing),
        "claim_risk": "the requested operation would otherwise execute through an unmeasured MuJoCo API surface",
        "remediation": "install a stable capability-complete MuJoCo release at or above 3.9 with matching Python and native components",
    }
    if support_tier is not None:
        evidence["support_tier"] = support_tier.value
    return refuse(OperationalReasonCode.MUJOCO_RUNTIME_CAPABILITY_MISSING, "comparison", **evidence)


def admit_mujoco_runtime(
    operation: MujocoClaimSurface = MujocoClaimSurface.COMPILED_ARTIFACT,
    *,
    runtime_module: object = mujoco,
) -> MujocoRuntimeAdmission:
    """Admit a coherent stable MuJoCo runtime for one measured operation call graph."""
    if not isinstance(operation, MujocoClaimSurface):
        raise TypeError("operation must be a MujocoClaimSurface")
    package_version = getattr(runtime_module, "__version__", None)
    package_triplet = _parse_stable_version(package_version, package=True)
    if package_triplet is None or package_triplet < _MINIMUM_BASE_VERSION:
        raise refuse(
            OperationalReasonCode.UNSUPPORTED_MUJOCO_VERSION,
            "comparison",
            operation=operation.value,
            mujoco_python_version=package_version if isinstance(package_version, str) else None,
            detected_package_version=(
                package_version if isinstance(package_version, str) else None
            ),
            minimum_base_version=_base_version(_MINIMUM_BASE_VERSION),
            claim_risk="an unstable, malformed, or below-floor engine has no admitted compatibility contract",
            remediation="use a stable MuJoCo release at or above 3.9",
        )
    package_token = cast(str, package_version)
    version_accessors = tuple(
        name
        for name in ("mj_versionString", "mj_version")
        if not callable(_resolve_capability(runtime_module, name))
    )
    if version_accessors:
        raise _capability_refusal(operation, package_version, None, None, version_accessors, None)
    native_string_getter = _resolve_capability(runtime_module, "mj_versionString")
    native_integer_getter = _resolve_capability(runtime_module, "mj_version")
    assert callable(native_string_getter)
    assert callable(native_integer_getter)
    native_string = native_string_getter()
    native_integer = native_integer_getter()
    native_triplet = _parse_stable_version(native_string, package=False)
    if (
        native_triplet is None
        or native_triplet != package_triplet
        or type(native_integer) is not int
        or native_integer != _native_integer(native_triplet)
    ):
        raise refuse(
            OperationalReasonCode.MUJOCO_PYTHON_NATIVE_VERSION_MISMATCH,
            "comparison",
            operation=operation.value,
            mujoco_python_version=package_version,
            detected_package_version=package_version,
            native_version_string=native_string if isinstance(native_string, str) else None,
            native_version_integer=native_integer if type(native_integer) is int else None,
            detected_native_version_string=(
                native_string if isinstance(native_string, str) else None
            ),
            detected_native_version_integer=(
                native_integer if type(native_integer) is int else None
            ),
            claim_risk="the Python package and loaded native library do not identify the same engine",
            remediation="install a matching Python/native MuJoCo pair",
        )
    base_version = _base_version(package_triplet)
    missing = _missing_capabilities(runtime_module, operation)
    if missing:
        raise _capability_refusal(
            operation, package_version, native_string, native_integer, missing, None
        )
    support_tier = MujocoSupportTier.ADMITTED_CAPABILITY_COMPATIBLE_PROFILE
    return MujocoRuntimeAdmission(
        package_token,
        base_version,
        cast(str, native_string),
        native_integer,
        support_tier,
        operation,
        tuple(requirement.name for requirement in MUJOCO_CAPABILITY_INVENTORY[operation]),
        _runtime_feature_facts(runtime_module),
    )


def _integer_sequence(model: object, name: str, count: int) -> tuple[int, ...] | None:
    """Read one exact-width model integer array without importing a numerical dependency."""
    values = getattr(model, name, None)
    if values is None:
        return None
    try:
        result = tuple(integer_index(values[index]) for index in range(count))
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None
    return result


def _integer_attribute(model: object, name: str) -> int | None:
    """Read one model integer attribute, returning null when it is absent or malformed."""
    try:
        return integer_index(getattr(model, name))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def _has_nonzero_value(value: object) -> bool:
    """Report whether one scalar or nested array-like value contains a nonzero number."""
    any_method = getattr(value, "any", None)
    if callable(any_method):
        try:
            return bool(any_method())
        except (TypeError, ValueError, OverflowError):
            return False
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        try:
            return any(_has_nonzero_value(item) for item in value)
        except (TypeError, ValueError, OverflowError):
            return False
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def _feature_fields(model: object, names: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return exposed and active optional model feature-field names."""
    exposed: list[str] = []
    active: list[str] = []
    for name in names:
        value = getattr(model, name, None)
        if value is None:
            continue
        exposed.append(name)
        if _has_nonzero_value(value):
            active.append(name)
    return tuple(exposed), tuple(active)


def measure_model_feature_facts(model: object) -> MujocoModelFeatureFacts:
    """Measure actuator signatures and newer contact facts before descriptor indexing."""
    issues: list[str] = []
    measured_control_width = _integer_attribute(model, "nu")
    if measured_control_width is None:
        control_width = -1
        issues.append("control_width_unreadable")
    else:
        control_width = measured_control_width
    modern_count = getattr(model, "nactuator", _MISSING)
    if modern_count is _MISSING:
        legacy_implicit_signature = True
        actuator_count = control_width
        output_width = control_width
        control_counts = (1,) * max(actuator_count, 0)
        output_counts = (1,) * max(actuator_count, 0)
        control_specs: tuple[int | None, ...] = (None,) * max(actuator_count, 0)
    else:
        legacy_implicit_signature = False
        measured_actuator_count = _integer_attribute(model, "nactuator")
        if measured_actuator_count is None:
            actuator_count = -1
            issues.append("actuator_count_unreadable")
        else:
            actuator_count = measured_actuator_count
        measured_output_width = _integer_attribute(model, "nout")
        if measured_output_width is None:
            output_width = -1
            issues.append("output_width_unreadable")
        else:
            output_width = measured_output_width
        width = max(actuator_count, 0)
        control_counts = _integer_sequence(model, "actuator_ctrlnum", width) or ()
        output_counts = _integer_sequence(model, "actuator_outnum", width) or ()
        raw_specs = _integer_sequence(model, "actuator_ctrlspec", width)
        control_specs = (
            cast(tuple[int | None, ...], raw_specs) if raw_specs is not None else (None,) * width
        )
        if len(control_counts) != width:
            issues.append("actuator_control_counts_unreadable")
        if len(output_counts) != width:
            issues.append("actuator_output_counts_unreadable")
        if hasattr(model, "actuator_ctrlspec") and raw_specs is None:
            issues.append("actuator_control_specs_unreadable")
    signatures = tuple(
        MujocoActuatorSignature(
            index, control_counts[index], output_counts[index], control_specs[index]
        )
        for index in range(max(actuator_count, 0))
        if index < len(control_counts) and index < len(output_counts) and index < len(control_specs)
    )
    if control_width < 0 or actuator_count < 0 or output_width < 0:
        issues.append("actuator_dimensions_invalid")
    if len(signatures) != max(actuator_count, 0):
        issues.append("actuator_signatures_incomplete")
    if len(control_counts) == max(actuator_count, 0) and sum(control_counts) != control_width:
        issues.append("actuator_control_width_incoherent")
    if len(output_counts) == max(actuator_count, 0) and sum(output_counts) != output_width:
        issues.append("actuator_output_width_incoherent")
    surface_fields, active_surface_fields = _feature_fields(
        model, ("geom_surfacevel", "pair_surfacevel")
    )
    adhesion_fields, active_adhesion_fields = _feature_fields(
        model, ("geom_adhesion", "pair_adhesion")
    )
    return MujocoModelFeatureFacts(
        actuator_count,
        control_width,
        output_width,
        legacy_implicit_signature,
        signatures,
        tuple(dict.fromkeys(issues)),
        surface_fields,
        active_surface_fields,
        adhesion_fields,
        active_adhesion_fields,
    )


def admit_model_feature_coverage(
    model: object,
    admission: MujocoRuntimeAdmission,
    role: ModelRole,
) -> MujocoModelFeatureFacts:
    """Admit model features representable by the selected operation before evidence construction."""
    if not isinstance(admission, MujocoRuntimeAdmission):
        raise TypeError("admission must be a MujocoRuntimeAdmission")
    if role not in {"baseline", "candidate", "comparison"}:
        raise ValueError("feature admission requires a baseline, candidate, or comparison role")
    facts = measure_model_feature_facts(model)
    if admission.operation is MujocoClaimSurface.COMPILED_ARTIFACT:
        return facts
    runtime_triplet = _parse_stable_version(admission.package_base_version, package=False)
    if facts.legacy_implicit_signature and (
        runtime_triplet is None or runtime_triplet >= (3, 11, 0)
    ):
        facts = replace(
            facts,
            signature_measurement_issues=(
                *facts.signature_measurement_issues,
                "modern_actuator_signature_fields_absent",
            ),
        )
    unsupported = facts.unsupported_actuator_signatures()
    if facts.signature_measurement_issues or unsupported:
        evidence = admission.to_evidence()
        evidence.update(
            {
                "unsupported_features": {
                    "actuator_signatures": [item.to_evidence() for item in unsupported],
                    "measurement_issues": list(facts.signature_measurement_issues),
                },
                "measured_model_feature_facts": facts.to_evidence(),
                "claim_risk": (
                    "the current typed actuator identity maps one semantic actuator to exactly one control input and one force output"
                ),
                "remediation": (
                    "reduce each actuator to an admitted one-input/one-output contract, or use a later Metrifid version that supports the measured signature"
                ),
            }
        )
        raise refuse(
            OperationalReasonCode.MUJOCO_FEATURE_COVERAGE_INCOMPLETE,
            role,
            **evidence,
        )
    return facts


__all__ = [
    "MUJOCO_CAPABILITY_INVENTORY",
    "MujocoActuatorSignature",
    "MujocoCapabilityRequirement",
    "MujocoClaimSurface",
    "MujocoModelFeatureFacts",
    "MujocoRuntimeAdmission",
    "MujocoSupportTier",
    "admit_model_feature_coverage",
    "admit_mujoco_runtime",
    "measure_model_feature_facts",
]
