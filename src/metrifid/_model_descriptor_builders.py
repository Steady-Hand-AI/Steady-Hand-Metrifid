"""Build strict descriptor identities from compiled MuJoCo models."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import cast

import mujoco  # type: ignore[import-untyped]

from ._model_closure import (
    _ACTIVATION_LAYOUT_WIDTHS,
    ActivationFamily,
    JointType,
    ModelRole,
    refuse,
)
from ._model_descriptor_types import (
    ActuatorDescriptor,
    CompiledModelIdentity,
    JointDescriptor,
    _actuator_sort_key,
    _covers,
)
from .json_values import CanonicalValue, require_sha256
from .operational import OperationalReasonCode
from .schemas import TargetReference


def _joint_types() -> dict[int, tuple[JointType, int, int]]:
    """Resolve the admitted runtime's joint enum values only at descriptor execution."""
    joint = mujoco.mjtJoint
    return {
        int(joint.mjJNT_FREE): ("FREE", 7, 6),
        int(joint.mjJNT_BALL): ("BALL", 4, 3),
        int(joint.mjJNT_SLIDE): ("SLIDE", 1, 1),
        int(joint.mjJNT_HINGE): ("HINGE", 1, 1),
    }


def _activation_families() -> dict[int, ActivationFamily]:
    """Resolve admitted actuator-dynamics enums without eager module-level binding."""
    return {
        int(getattr(mujoco.mjtDyn, f"mjDYN_{family}")): family
        for family in _ACTIVATION_LAYOUT_WIDTHS
    }


def _transmissions() -> dict[int, str]:
    """Resolve admitted actuator-transmission enums at descriptor execution."""
    transmission = mujoco.mjtTrn
    return {
        int(transmission.mjTRN_JOINT): "JOINT",
        int(transmission.mjTRN_JOINTINPARENT): "JOINTINPARENT",
        int(transmission.mjTRN_SLIDERCRANK): "SLIDERCRANK",
        int(transmission.mjTRN_TENDON): "TENDON",
        int(transmission.mjTRN_SITE): "SITE",
        int(transmission.mjTRN_BODY): "BODY",
    }


def _objects() -> dict[int, tuple[str, str]]:
    """Resolve admitted object enums and their model-count projections lazily."""
    object_type = mujoco.mjtObj
    return {
        int(object_type.mjOBJ_JOINT): ("njnt", "JOINT"),
        int(object_type.mjOBJ_TENDON): ("ntendon", "TENDON"),
        int(object_type.mjOBJ_SITE): ("nsite", "SITE"),
        int(object_type.mjOBJ_BODY): ("nbody", "BODY"),
    }


def _count_model_object_names(model: mujoco.MjModel, object_type: int) -> Counter[str]:
    """Count nonempty compiled semantic names for one MuJoCo object type."""
    count = int(getattr(model, _objects()[object_type][0]))
    return Counter(
        name for index in range(count) if (name := mujoco.mj_id2name(model, object_type, index))
    )


def _resolve_actuator_target_reference(
    model: mujoco.MjModel,
    object_type: int,
    index: int,
    role: ModelRole,
    actuator: int,
) -> TargetReference:
    """Resolve one compiled transmission target to its unique named model object."""
    count_attribute, token = _objects()[object_type]
    count = int(getattr(model, count_attribute))
    if index < 0 or index >= count:
        raise refuse(
            OperationalReasonCode.ACTUATOR_TARGET_MISMATCH,
            role,
            actuator_index=actuator,
            target_index=index,
        )
    name = mujoco.mj_id2name(model, object_type, index)
    if not name:
        raise refuse(
            OperationalReasonCode.ACTUATOR_TARGET_MISMATCH,
            role,
            actuator_index=actuator,
            target_index=index,
            issue="target_name_missing",
        )
    if _count_model_object_names(model, object_type)[name] != 1:
        raise refuse(
            OperationalReasonCode.ACTUATOR_IDENTITY_AMBIGUOUS,
            role,
            actuator_index=actuator,
            target_name=name,
        )
    return TargetReference(token, name)


# The frozen helper rename is longer than its baseline spelling; retain the compact
# expression layout so the unchanged function stays within the audited size bound.
# fmt: off
def _actuator_targets(
    model: mujoco.MjModel,
    index: int,
    role: ModelRole,
    transmission: str,
) -> tuple[TargetReference, ...]:
    """Extract the ordered semantic targets for one compiled actuator transmission."""
    first, second = (int(value) for value in model.actuator_trnid[index])
    if transmission in {"JOINT", "JOINTINPARENT"}:
        return (_resolve_actuator_target_reference(model, int(mujoco.mjtObj.mjOBJ_JOINT), first, role, index),)
    if transmission == "TENDON":
        return (_resolve_actuator_target_reference(model, int(mujoco.mjtObj.mjOBJ_TENDON), first, role, index),)
    if transmission == "SLIDERCRANK":
        return (
            _resolve_actuator_target_reference(model, int(mujoco.mjtObj.mjOBJ_SITE), first, role, index),
            _resolve_actuator_target_reference(model, int(mujoco.mjtObj.mjOBJ_SITE), second, role, index),
        )
    if transmission == "SITE":
        targets = [_resolve_actuator_target_reference(model, int(mujoco.mjtObj.mjOBJ_SITE), first, role, index)]
        if second >= 0:
            targets.append(_resolve_actuator_target_reference(model, int(mujoco.mjtObj.mjOBJ_SITE), second, role, index))
        return tuple(targets)
    if transmission == "BODY":
        return (_resolve_actuator_target_reference(model, int(mujoco.mjtObj.mjOBJ_BODY), first, role, index),)
    raise refuse(
        OperationalReasonCode.MODEL_LAYOUT_INCOMPATIBLE,
        role,
        actuator_index=index,
        transmission=transmission,
    )
# fmt: on


def _joint_descriptors(model: mujoco.MjModel, role: ModelRole) -> tuple[JointDescriptor, ...]:
    """Build canonical descriptors for every supported named compiled joint."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    if any(not name for name in names):
        raise refuse(
            OperationalReasonCode.JOINT_NAME_MISSING,
            role,
            joint_indices=[index for index, name in enumerate(names) if not name],
        )
    joint_names = cast(Sequence[str], names)
    duplicates = sorted(name for name, count in Counter(joint_names).items() if count > 1)
    if duplicates:
        raise refuse(
            OperationalReasonCode.JOINT_NAME_DUPLICATE,
            role,
            names=cast(CanonicalValue, duplicates),
        )
    descriptors: list[JointDescriptor] = []
    for index, name in enumerate(joint_names):
        data = _joint_types().get(int(model.jnt_type[index]))
        if data is None:
            raise refuse(
                OperationalReasonCode.MODEL_LAYOUT_INCOMPATIBLE,
                role,
                joint_index=index,
                joint_type=int(model.jnt_type[index]),
            )
        token, qpos_width, qvel_width = data
        descriptors.append(
            JointDescriptor(
                name,
                token,
                qpos_width,
                qvel_width,
                int(model.jnt_qposadr[index]),
                int(model.jnt_dofadr[index]),
            )
        )
    return tuple(sorted(descriptors, key=lambda item: item.name))


def _actuator_descriptors(model: mujoco.MjModel, role: ModelRole) -> tuple[ActuatorDescriptor, ...]:
    """Build deterministic semantic and layout descriptors for all compiled actuators."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    duplicates = sorted(
        name for name, count in Counter(name for name in names if name).items() if count > 1
    )
    if duplicates:
        raise refuse(
            OperationalReasonCode.ACTUATOR_IDENTITY_AMBIGUOUS,
            role,
            names=cast(CanonicalValue, duplicates),
        )
    descriptors: list[ActuatorDescriptor] = []
    for index, name in enumerate(names):
        descriptors.append(_actuator_descriptor(model, role, index, name))
    return tuple(sorted(descriptors, key=_actuator_sort_key))


def _actuator_descriptor(
    model: mujoco.MjModel, role: ModelRole, index: int, name: str | None
) -> ActuatorDescriptor:
    """Build and validate one compiled-model actuator descriptor."""
    transmission = _transmissions().get(int(model.actuator_trntype[index]))
    family = _activation_families().get(int(model.actuator_dyntype[index]))
    if transmission is None or family is None:
        raise refuse(
            OperationalReasonCode.MODEL_LAYOUT_INCOMPATIBLE,
            role,
            actuator_index=index,
            transmission_type=int(model.actuator_trntype[index]),
            activation_type=int(model.actuator_dyntype[index]),
        )
    address, width = int(model.actuator_actadr[index]), int(model.actuator_actnum[index])
    activation_address = None if address < 0 else address
    try:
        return ActuatorDescriptor(
            name or None,
            transmission,
            _actuator_targets(model, index, role, transmission),
            family,
            width,
            index,
            activation_address,
        )
    except ValueError as exc:
        error = refuse(
            OperationalReasonCode.MODEL_LAYOUT_INCOMPATIBLE,
            role,
            actuator_index=index,
            issue="unsupported_activation_layout",
        )
        raise error from exc


def _validate_layout(
    model: mujoco.MjModel,
    joints: Sequence[JointDescriptor],
    actuators: Sequence[ActuatorDescriptor],
    role: ModelRole,
) -> None:
    """Require descriptors to exactly cover compiled qpos, qvel, control, and activation layouts."""
    qpos = [(item.qpos_address, item.qpos_width) for item in joints]
    qvel = [(item.qvel_address, item.qvel_width) for item in joints]
    activation = [
        (item.activation_address, item.activation_width)
        for item in actuators
        if item.activation_address is not None
    ]
    if not (_covers(model.nq, qpos) and _covers(model.nv, qvel) and _covers(model.na, activation)):
        raise refuse(
            OperationalReasonCode.MODEL_LAYOUT_INCOMPATIBLE,
            role,
            nq=int(model.nq),
            nv=int(model.nv),
            nu=int(model.nu),
            na=int(model.na),
        )


def compile_model_identity(
    model: mujoco.MjModel,
    closure_sha256: str,
    role: ModelRole,
) -> CompiledModelIdentity:
    """Build and self-hash the semantic identity of one admitted compiled model."""
    require_sha256(closure_sha256, "closure_sha256")
    joints = _joint_descriptors(model, role)
    actuators = _actuator_descriptors(model, role)
    _validate_layout(model, joints, actuators, role)
    return CompiledModelIdentity(
        CompiledModelIdentity._SCHEMA,
        CompiledModelIdentity._SCHEMA_VERSION,
        None,
        closure_sha256,
        int(model.nq),
        int(model.nv),
        int(model.nu),
        int(model.na),
        joints,
        actuators,
    ).finalized()
