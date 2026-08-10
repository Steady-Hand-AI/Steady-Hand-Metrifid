"""Align compiled model descriptors under strict alias semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from ._alignment import AlignedActuator, AlignedJoint, _canonical_targets
from ._model_closure import ModelRole, refuse
from ._model_descriptors import ActuatorDescriptor, CompiledModelIdentity, JointDescriptor
from ._model_identity_types import SemanticAlignment
from .json_values import (
    CanonicalValue,
    FrozenCanonicalObject,
    canonical_json_bytes,
    freeze_canonical,
)
from .operational import OperationalReasonCode
from .schemas import (
    ActuatorAliasEndpoint,
    ActuatorAliasPair,
    AliasArtifact,
    JointAliasPair,
    TargetReference,
)


def _align_joints(
    baseline: CompiledModelIdentity,
    candidate: CompiledModelIdentity,
    aliases: Sequence[JointAliasPair],
) -> tuple[
    tuple[AlignedJoint, ...], dict[str, str], dict[str, str], list[dict[str, CanonicalValue]]
]:
    """Pair compiled joints by explicit aliases and remaining identical names."""
    left = {item.name: item for item in baseline.joints}
    right = {item.name: item for item in candidate.joints}
    pairs, bindings, consumed_left, consumed_right = _aliased_joint_pairs(left, right, aliases)
    automatic = sorted((set(left) - consumed_left) & (set(right) - consumed_right))
    alias_names = {item.canonical_name for item in aliases}
    if collision := alias_names & set(automatic):
        raise refuse(
            OperationalReasonCode.ALIAS_BINDING_DUPLICATE,
            "comparison",
            canonical_names=cast(CanonicalValue, sorted(collision)),
        )
    for name in automatic:
        pairs.append((name, left[name], right[name]))
        consumed_left.add(name)
        consumed_right.add(name)
    if missing := sorted((set(left) - consumed_left) | (set(right) - consumed_right)):
        raise refuse(
            OperationalReasonCode.JOINT_IDENTITY_MISSING,
            "comparison",
            names=cast(CanonicalValue, missing),
        )
    return _finish_joint_pairs(pairs, bindings)


def _aliased_joint_pairs(
    left: Mapping[str, JointDescriptor],
    right: Mapping[str, JointDescriptor],
    aliases: Sequence[JointAliasPair],
) -> tuple[
    list[tuple[str, JointDescriptor, JointDescriptor]],
    list[dict[str, CanonicalValue]],
    set[str],
    set[str],
]:
    """Resolve explicit joint aliases and track consumed role-local names."""
    consumed_left: set[str] = set()
    consumed_right: set[str] = set()
    pairs: list[tuple[str, JointDescriptor, JointDescriptor]] = []
    bindings: list[dict[str, CanonicalValue]] = []
    for alias in aliases:
        if alias.baseline_name not in left or alias.candidate_name not in right:
            raise refuse(
                OperationalReasonCode.JOINT_IDENTITY_MISSING,
                "comparison",
                canonical_name=alias.canonical_name,
                baseline_name=alias.baseline_name,
                candidate_name=alias.candidate_name,
            )
        if alias.baseline_name in consumed_left or alias.candidate_name in consumed_right:
            raise refuse(
                OperationalReasonCode.ALIAS_BINDING_DUPLICATE,
                "comparison",
                canonical_name=alias.canonical_name,
            )
        consumed_left.add(alias.baseline_name)
        consumed_right.add(alias.candidate_name)
        pairs.append((alias.canonical_name, left[alias.baseline_name], right[alias.candidate_name]))
        bindings.append({"kind": "JOINT", **alias.to_primitive()})
    return pairs, bindings, consumed_left, consumed_right


def _finish_joint_pairs(
    pairs: Sequence[tuple[str, JointDescriptor, JointDescriptor]],
    bindings: list[dict[str, CanonicalValue]],
) -> tuple[
    tuple[AlignedJoint, ...], dict[str, str], dict[str, str], list[dict[str, CanonicalValue]]
]:
    """Validate joint-pair compatibility and build canonical aligned-joint records."""
    aligned: list[AlignedJoint] = []
    left_map: dict[str, str] = {}
    right_map: dict[str, str] = {}
    for canonical, first, second in sorted(pairs):
        if first.joint_type != second.joint_type:
            raise refuse(
                OperationalReasonCode.JOINT_TYPE_MISMATCH,
                "comparison",
                canonical_name=canonical,
                baseline=first.joint_type,
                candidate=second.joint_type,
            )
        if first.qpos_width != second.qpos_width:
            raise refuse(
                OperationalReasonCode.JOINT_QPOS_WIDTH_MISMATCH,
                "comparison",
                canonical_name=canonical,
            )
        if first.qvel_width != second.qvel_width:
            raise refuse(
                OperationalReasonCode.JOINT_QVEL_WIDTH_MISMATCH,
                "comparison",
                canonical_name=canonical,
            )
        left_map[first.name] = canonical
        right_map[second.name] = canonical
        aligned.append(
            AlignedJoint(
                canonical,
                first.joint_type,
                (first.qpos_address, first.qpos_width),
                (second.qpos_address, second.qpos_width),
                (first.qvel_address, first.qvel_width),
                (second.qvel_address, second.qvel_width),
            )
        )
    return tuple(aligned), left_map, right_map, bindings


def _selector_matches(endpoint: ActuatorAliasEndpoint, item: ActuatorDescriptor) -> bool:
    """Return whether an unnamed actuator matches an alias endpoint's full semantics."""
    return (
        item.name is None
        and endpoint.transmission_type == item.transmission_type
        and endpoint.targets == item.targets
        and endpoint.activation_family == item.activation_family
        and endpoint.activation_width == item.activation_width
    )


def _resolve_endpoint(
    endpoint: ActuatorAliasEndpoint,
    values: Sequence[ActuatorDescriptor],
    role: ModelRole,
) -> ActuatorDescriptor:
    """Resolve a named or semantic actuator endpoint to exactly one role-local actuator."""
    if endpoint.kind == "NAMED":
        matches = [item for item in values if item.name == endpoint.name]
        if not matches:
            raise refuse(OperationalReasonCode.ACTUATOR_IDENTITY_MISSING, role, name=endpoint.name)
    else:
        matches = [item for item in values if _selector_matches(endpoint, item)]
        if not matches:
            raise refuse(
                OperationalReasonCode.ALIAS_SELECTOR_NO_MATCH,
                role,
                selector=endpoint.to_primitive(),
            )
        if len(matches) > 1:
            raise refuse(
                OperationalReasonCode.ALIAS_SELECTOR_AMBIGUOUS,
                role,
                selector=endpoint.to_primitive(),
                match_count=len(matches),
            )
    return matches[0]


def _actuator_pairs(
    baseline: CompiledModelIdentity,
    candidate: CompiledModelIdentity,
    aliases: Sequence[ActuatorAliasPair],
) -> tuple[
    list[tuple[str, ActuatorDescriptor, ActuatorDescriptor]], list[dict[str, CanonicalValue]]
]:
    """Pair actuators through aliases, shared names, and unambiguous unnamed semantics."""
    pairs, bindings, used_left, used_right = _aliased_actuator_pairs(baseline, candidate, aliases)
    left_named = _unconsumed_named_actuators(baseline.actuators, used_left)
    right_named = _unconsumed_named_actuators(candidate.actuators, used_right)
    automatic = sorted(set(left_named) & set(right_named))
    if collision := {item.canonical_name for item in aliases} & set(automatic):
        raise refuse(
            OperationalReasonCode.ALIAS_BINDING_DUPLICATE,
            "comparison",
            canonical_names=cast(CanonicalValue, sorted(collision)),
        )
    for name in automatic:
        first, second = left_named[name], right_named[name]
        used_left.add(first.control_address)
        used_right.add(second.control_address)
        pairs.append((name, first, second))
    _refuse_unmatched_actuators(baseline, candidate, used_left, used_right)
    return pairs, bindings


def _aliased_actuator_pairs(
    baseline: CompiledModelIdentity,
    candidate: CompiledModelIdentity,
    aliases: Sequence[ActuatorAliasPair],
) -> tuple[
    list[tuple[str, ActuatorDescriptor, ActuatorDescriptor]],
    list[dict[str, CanonicalValue]],
    set[int],
    set[int],
]:
    """Resolve explicit actuator aliases and track consumed control addresses."""
    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs: list[tuple[str, ActuatorDescriptor, ActuatorDescriptor]] = []
    bindings: list[dict[str, CanonicalValue]] = []
    for alias in aliases:
        first = _resolve_endpoint(alias.baseline, baseline.actuators, "baseline")
        second = _resolve_endpoint(alias.candidate, candidate.actuators, "candidate")
        if first.control_address in used_left or second.control_address in used_right:
            raise refuse(
                OperationalReasonCode.ALIAS_BINDING_DUPLICATE,
                "comparison",
                canonical_name=alias.canonical_name,
            )
        used_left.add(first.control_address)
        used_right.add(second.control_address)
        pairs.append((alias.canonical_name, first, second))
        bindings.append({"kind": "ACTUATOR", **alias.to_primitive()})
    return pairs, bindings, used_left, used_right


def _unconsumed_named_actuators(
    actuators: Sequence[ActuatorDescriptor], consumed: set[int]
) -> dict[str, ActuatorDescriptor]:
    """Index named actuators whose control addresses remain unconsumed."""
    return cast(
        dict[str, ActuatorDescriptor],
        {
            item.name: item
            for item in actuators
            if item.name is not None and item.control_address not in consumed
        },
    )


def _refuse_unmatched_actuators(
    baseline: CompiledModelIdentity,
    candidate: CompiledModelIdentity,
    used_left: set[int],
    used_right: set[int],
) -> None:
    """Refuse when either role retains an actuator outside the complete alignment."""
    leftovers = [item for item in baseline.actuators if item.control_address not in used_left]
    leftovers += [item for item in candidate.actuators if item.control_address not in used_right]
    if leftovers:
        raise refuse(
            OperationalReasonCode.ACTUATOR_IDENTITY_MISSING,
            "comparison",
            unmatched=[
                item.name if item.name is not None else item.to_primitive() for item in leftovers
            ],
        )


def _finish_actuator_pairs(
    pairs: Sequence[tuple[str, ActuatorDescriptor, ActuatorDescriptor]],
    left_joints: Mapping[str, str],
    right_joints: Mapping[str, str],
) -> tuple[AlignedActuator, ...]:
    """Canonicalize paired actuator targets and build aligned actuator records."""
    return tuple(
        _aligned_actuator(canonical, first, second, left_joints, right_joints)
        for canonical, first, second in sorted(pairs)
    )


def _aligned_actuator(
    canonical: str,
    first: ActuatorDescriptor,
    second: ActuatorDescriptor,
    left_joints: Mapping[str, str],
    right_joints: Mapping[str, str],
) -> AlignedActuator:
    """Validate one actuator pair and return its canonical alignment descriptor."""
    if first.transmission_type != second.transmission_type:
        raise refuse(
            OperationalReasonCode.ACTUATOR_TRANSMISSION_MISMATCH,
            "comparison",
            canonical_name=canonical,
            baseline=first.transmission_type,
            candidate=second.transmission_type,
        )
    first_targets = _canonical_targets(first.targets, left_joints)
    second_targets = _canonical_targets(second.targets, right_joints)
    if first_targets != second_targets:
        raise refuse(
            OperationalReasonCode.ACTUATOR_TARGET_MISMATCH,
            "comparison",
            canonical_name=canonical,
            baseline=[item.to_primitive() for item in first_targets],
            candidate=[item.to_primitive() for item in second_targets],
        )
    if first.activation_family != second.activation_family:
        raise refuse(
            OperationalReasonCode.ACTUATOR_ACTIVATION_FAMILY_MISMATCH,
            "comparison",
            canonical_name=canonical,
            baseline=first.activation_family,
            candidate=second.activation_family,
        )
    if first.activation_width != second.activation_width:
        raise refuse(
            OperationalReasonCode.ACTUATOR_ACTIVATION_WIDTH_MISMATCH,
            "comparison",
            canonical_name=canonical,
            baseline=first.activation_width,
            candidate=second.activation_width,
        )
    return _aligned_actuator_record(canonical, first, second, first_targets)


def _aligned_actuator_record(
    canonical: str,
    first: ActuatorDescriptor,
    second: ActuatorDescriptor,
    targets: tuple[TargetReference, ...],
) -> AlignedActuator:
    """Construct an aligned actuator after pair compatibility is established."""
    return AlignedActuator(
        canonical,
        first.transmission_type,
        targets,
        first.activation_family,
        first.activation_width,
        first.control_address,
        second.control_address,
        first.activation_address,
        second.activation_address,
    )


def align_compiled_models(
    baseline: CompiledModelIdentity,
    candidate: CompiledModelIdentity,
    aliases: AliasArtifact | None = None,
    *,
    aliases_raw_sha256: str | None = None,
    aliases_semantic_sha256: str | None = None,
) -> SemanticAlignment:
    """Produce one complete alignment or raise one typed refusal with no partial result."""
    joint_aliases = aliases.joint_pairs if aliases is not None else ()
    actuator_aliases = aliases.actuator_pairs if aliases is not None else ()
    joints, left_joints, right_joints, joint_bindings = _align_joints(
        baseline, candidate, joint_aliases
    )
    pairs, actuator_bindings = _actuator_pairs(baseline, candidate, actuator_aliases)
    actuators = _finish_actuator_pairs(pairs, left_joints, right_joints)
    bindings = sorted(joint_bindings + actuator_bindings, key=canonical_json_bytes)
    frozen = tuple(
        cast(FrozenCanonicalObject, freeze_canonical(cast(CanonicalValue, item)))
        for item in bindings
    )
    return SemanticAlignment(
        SemanticAlignment._SCHEMA,
        SemanticAlignment._SCHEMA_VERSION,
        None,
        aliases_raw_sha256,
        aliases_semantic_sha256,
        joints,
        actuators,
        frozen,
    ).finalized()
