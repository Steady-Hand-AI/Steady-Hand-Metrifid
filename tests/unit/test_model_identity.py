"""Collect model identity scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from metrifid import _model_admission as admission
from metrifid import _model_closure as closure
from metrifid import _model_descriptors as model_descriptors
from metrifid import _model_identity as identity
from metrifid.json_values import canonical_sha256
from metrifid.operational import OperationalReasonCode
from metrifid.schemas import (
    ActuatorAliasEndpoint,
    ActuatorAliasPair,
    AliasArtifact,
    JointAliasPair,
    TargetReference,
)

SHA_A = "a" * 64


SHA_B = "b" * 64


def _joint(
    name: str = "joint",
    joint_type: closure.JointType = "HINGE",
    qpos_address: int = 0,
    qvel_address: int = 0,
) -> admission.JointDescriptor:
    """Construct the joint fixture used by model identity scenarios.

    Deterministic setup isolates model identity without bypassing the contract boundary under
    assertion.
    """
    widths = {"FREE": (7, 6), "BALL": (4, 3), "SLIDE": (1, 1), "HINGE": (1, 1)}
    qpos, qvel = widths[joint_type]
    return admission.JointDescriptor(name, joint_type, qpos, qvel, qpos_address, qvel_address)


def _actuator(
    name: str | None = "act",
    *,
    transmission: str = "JOINT",
    targets: tuple[TargetReference, ...] | None = None,
    family: closure.ActivationFamily = "NONE",
    width: int = 0,
    control: int = 0,
    activation: int | None = None,
) -> admission.ActuatorDescriptor:
    """Construct the actuator fixture used by model identity scenarios.

    Deterministic setup isolates model identity without bypassing the contract boundary under
    assertion.
    """
    return admission.ActuatorDescriptor(
        name,
        transmission,
        targets or (TargetReference("JOINT", "joint"),),
        family,
        width,
        control,
        activation,
    )


def _compiled(
    joints: tuple[admission.JointDescriptor, ...] = (_joint(),),
    actuators: tuple[admission.ActuatorDescriptor, ...] = (_actuator(),),
    *,
    closure_hash: str = SHA_A,
) -> admission.CompiledModelIdentity:
    """Construct the compiled fixture used by model identity scenarios.

    Deterministic setup isolates model identity without bypassing the contract boundary under
    assertion.
    """
    nq = sum(item.qpos_width for item in joints)
    nv = sum(item.qvel_width for item in joints)
    na = sum(item.activation_width for item in actuators)
    return admission.CompiledModelIdentity(
        admission.CompiledModelIdentity._SCHEMA,
        admission.CompiledModelIdentity._SCHEMA_VERSION,
        None,
        closure_hash,
        nq,
        nv,
        len(actuators),
        na,
        tuple(sorted(joints, key=lambda item: item.name)),
        tuple(sorted(actuators, key=model_descriptors._actuator_sort_key)),
    ).finalized()


def _alias(
    *,
    joints: tuple[JointAliasPair, ...] = (),
    actuators: tuple[ActuatorAliasPair, ...] = (),
    baseline_hash: str = SHA_A,
    candidate_hash: str = SHA_A,
) -> AliasArtifact:
    """Construct the alias fixture used by model identity scenarios.

    Deterministic setup isolates model identity without bypassing the contract boundary under
    assertion.
    """
    return AliasArtifact(
        "metrifid.aliases",
        1,
        baseline_hash,
        candidate_hash,
        tuple(sorted(joints, key=lambda item: item.canonical_name)),
        tuple(sorted(actuators, key=lambda item: item.canonical_name)),
    )


def _refusal_reason(
    exc: pytest.ExceptionInfo[closure.ModelAdmissionRefusal],
) -> OperationalReasonCode:
    """Extract the closure refusal code from identity construction."""
    return exc.value.reason


def _unsafe_joint(
    *,
    qpos_width: int = 1,
    qvel_width: int = 1,
    joint_type: str = "HINGE",
) -> admission.JointDescriptor:
    """Construct the unsafe joint fixture used by model identity scenarios.

    Deterministic setup isolates model identity without bypassing the contract boundary under
    assertion.
    """
    value = object.__new__(admission.JointDescriptor)
    for name, item in {
        "name": "joint",
        "joint_type": joint_type,
        "qpos_width": qpos_width,
        "qvel_width": qvel_width,
        "qpos_address": 0,
        "qvel_address": 0,
    }.items():
        object.__setattr__(value, name, item)
    return value


def test_descriptor_roundtrips_and_validation() -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises descriptor roundtrips and validation; incomplete, ambiguous, or
    unbound model components cannot support certification.
    """
    joint = _joint()
    assert admission.JointDescriptor.from_primitive(joint.to_primitive()) == joint
    with pytest.raises(ValueError):
        admission.JointDescriptor("j", "BAD", 1, 1, 0, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        admission.JointDescriptor("j", "HINGE", 2, 1, 0, 0)

    actuator = _actuator()
    assert admission.ActuatorDescriptor.from_primitive(actuator.to_primitive()) == actuator
    with pytest.raises(ValueError):
        admission.ActuatorDescriptor("a", "JOINT", (), "NONE", 0, 0, None)
    with pytest.raises(ValueError):
        admission.ActuatorDescriptor(
            "a", "JOINT", (TargetReference("SITE", "s"),), "NONE", 0, 0, None
        )
    with pytest.raises(ValueError):
        admission.ActuatorDescriptor(
            "a",
            "JOINT",
            (TargetReference("JOINT", "j"),),
            "BAD",  # type: ignore[arg-type]
            0,
            0,
            None,
        )
    with pytest.raises(ValueError):
        admission.ActuatorDescriptor(
            "a", "JOINT", (TargetReference("JOINT", "j"),), "NONE", 1, 0, None
        )


def test_compiled_identity_completion_roundtrip_and_mutations() -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises compiled identity completion roundtrip and mutations; incomplete,
    ambiguous, or unbound model components cannot support certification.
    """
    value = _compiled()
    assert value.finalized() is value
    assert admission.CompiledModelIdentity.from_primitive(value.to_primitive()) == value
    unfinished = replace(value, compiled_identity_sha256=None)
    with pytest.raises(ValueError):
        unfinished.to_primitive()
    assert unfinished.finalized().compiled_identity_sha256 is not None

    primitive = value.to_primitive()
    for mutation in (
        lambda obj: obj.update(schema_version="bad"),
        lambda obj: obj.update(compiled_identity_sha256=None),
        lambda obj: obj.update(compiled_identity_sha256="0" * 64),
        lambda obj: obj.update(model_closure_sha256="bad"),
        lambda obj: obj.update(nu=2),
        lambda obj: obj.update(extra=True),
        lambda obj: obj.update(joints="bad"),
        lambda obj: obj.update(actuators="bad"),
    ):
        changed = copy.deepcopy(primitive)
        mutation(changed)
        with pytest.raises((TypeError, ValueError)):
            admission.CompiledModelIdentity.from_primitive(changed)


def test_compiled_identity_structural_invariants() -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises compiled identity structural invariants; incomplete, ambiguous, or
    unbound model components cannot support certification.
    """
    _assert_compiled_identity_member_type_refusals()
    _assert_compiled_identity_layout_refusals()


def _assert_compiled_identity_member_type_refusals() -> None:
    """Assert compiled identities reject untyped joint and actuator members."""
    with pytest.raises(TypeError):
        admission.CompiledModelIdentity(
            admission.CompiledModelIdentity._SCHEMA,
            admission.CompiledModelIdentity._SCHEMA_VERSION,
            None,
            SHA_A,
            0,
            0,
            0,
            0,
            (object(),),  # type: ignore[arg-type]
            (),
        )
    with pytest.raises(TypeError):
        admission.CompiledModelIdentity(
            admission.CompiledModelIdentity._SCHEMA,
            admission.CompiledModelIdentity._SCHEMA_VERSION,
            None,
            SHA_A,
            0,
            0,
            1,
            0,
            (),
            (object(),),  # type: ignore[arg-type]
        )


def _assert_compiled_identity_layout_refusals() -> None:
    """Assert compiled identities reject order, control, and coverage violations."""
    with pytest.raises(ValueError):
        admission.CompiledModelIdentity(
            admission.CompiledModelIdentity._SCHEMA,
            admission.CompiledModelIdentity._SCHEMA_VERSION,
            None,
            SHA_A,
            2,
            2,
            0,
            0,
            (
                _joint("z", qpos_address=0, qvel_address=0),
                _joint("a", qpos_address=1, qvel_address=1),
            ),
            (),
        )
    with pytest.raises(ValueError):
        admission.CompiledModelIdentity(
            admission.CompiledModelIdentity._SCHEMA,
            admission.CompiledModelIdentity._SCHEMA_VERSION,
            None,
            SHA_A,
            1,
            1,
            1,
            0,
            (_joint(),),
            (_actuator(control=1),),
        )
    with pytest.raises(ValueError):
        admission.CompiledModelIdentity(
            admission.CompiledModelIdentity._SCHEMA,
            admission.CompiledModelIdentity._SCHEMA_VERSION,
            None,
            SHA_A,
            2,
            1,
            0,
            0,
            (_joint(),),
            (),
        )


def test_automatic_alignment_and_summary_are_canonical() -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises automatic alignment and summary are canonical; incomplete,
    ambiguous, or unbound model components cannot support certification.
    """
    baseline = _compiled()
    candidate = _compiled(closure_hash=SHA_B)
    result = identity.align_compiled_models(baseline, candidate)
    assert [item.canonical_name for item in result.joints] == ["joint"]
    assert [item.canonical_name for item in result.actuators] == ["act"]
    assert result.summary().joint_order == ("joint",)
    assert result.summary().actuator_order == ("act",)
    assert identity.SemanticAlignment.from_primitive(result.to_primitive()) == result
    assert result.finalized() is result


def test_joint_alias_and_joint_target_canonicalization() -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises joint alias and joint target canonicalization; incomplete,
    ambiguous, or unbound model components cannot support certification.
    """
    left = _compiled((_joint("left"),), (_actuator(targets=(TargetReference("JOINT", "left"),)),))
    right = _compiled(
        (_joint("right"),),
        (_actuator(targets=(TargetReference("JOINT", "right"),)),),
        closure_hash=SHA_B,
    )
    aliases = _alias(
        joints=(JointAliasPair("canonical", "left", "right"),),
        baseline_hash=SHA_A,
        candidate_hash=SHA_B,
    )
    result = identity.align_compiled_models(
        left,
        right,
        aliases,
        aliases_raw_sha256=SHA_A,
        aliases_semantic_sha256=canonical_sha256(aliases.to_primitive()),
    )
    assert result.joints[0].canonical_name == "canonical"
    assert result.actuators[0].targets == (TargetReference("JOINT", "canonical"),)
    assert result.alias_bindings


def test_joint_alias_failures_and_no_partial_result() -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises joint alias failures and no partial result; incomplete, ambiguous,
    or unbound model components cannot support certification.
    """
    base = _compiled()
    aliases = _alias(joints=(JointAliasPair("c", "missing", "joint"),))
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity.align_compiled_models(base, base, aliases)
    assert _refusal_reason(exc) is OperationalReasonCode.JOINT_IDENTITY_MISSING

    duplicate = _alias(
        joints=(
            JointAliasPair("a", "joint", "joint"),
            JointAliasPair("b", "joint", "joint"),
        )
    )
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity.align_compiled_models(base, base, duplicate)
    assert _refusal_reason(exc) is OperationalReasonCode.ALIAS_BINDING_DUPLICATE

    collision = _alias(joints=(JointAliasPair("other", "joint", "joint"),))
    extra_left = _compiled((_joint("joint"), _joint("other", qpos_address=1, qvel_address=1)), ())
    extra_right = _compiled((_joint("joint"), _joint("other", qpos_address=1, qvel_address=1)), ())
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity.align_compiled_models(extra_left, extra_right, collision)
    assert _refusal_reason(exc) is OperationalReasonCode.ALIAS_BINDING_DUPLICATE

    missing = _compiled((_joint("other"),), ())
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity.align_compiled_models(base, missing)
    assert _refusal_reason(exc) is OperationalReasonCode.JOINT_IDENTITY_MISSING


def test_joint_mismatch_reasons() -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises joint mismatch reasons; incomplete, ambiguous, or unbound model
    components cannot support certification.
    """
    left = _compiled((_joint(),), ())
    right_type = _compiled((_joint(joint_type="SLIDE"),), (), closure_hash=SHA_B)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity.align_compiled_models(left, right_type)
    assert _refusal_reason(exc) is OperationalReasonCode.JOINT_TYPE_MISMATCH

    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity._finish_joint_pairs((("joint", _unsafe_joint(), _unsafe_joint(qpos_width=2)),), [])
    assert _refusal_reason(exc) is OperationalReasonCode.JOINT_QPOS_WIDTH_MISMATCH
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity._finish_joint_pairs((("joint", _unsafe_joint(), _unsafe_joint(qvel_width=2)),), [])
    assert _refusal_reason(exc) is OperationalReasonCode.JOINT_QVEL_WIDTH_MISMATCH


def test_named_actuator_alias_and_failures() -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises named actuator alias and failures; incomplete, ambiguous, or unbound
    model components cannot support certification.
    """
    base = _compiled()
    renamed = _compiled(actuators=(_actuator("renamed"),), closure_hash=SHA_B)
    aliases = _alias(
        actuators=(
            ActuatorAliasPair(
                "canonical",
                ActuatorAliasEndpoint("NAMED", "act", None, (), None, None),
                ActuatorAliasEndpoint("NAMED", "renamed", None, (), None, None),
            ),
        ),
        candidate_hash=SHA_B,
    )
    result = identity.align_compiled_models(
        base,
        renamed,
        aliases,
        aliases_raw_sha256=SHA_A,
        aliases_semantic_sha256=canonical_sha256(aliases.to_primitive()),
    )
    assert result.actuators[0].canonical_name == "canonical"

    missing = _alias(
        actuators=(
            ActuatorAliasPair(
                "canonical",
                ActuatorAliasEndpoint("NAMED", "missing", None, (), None, None),
                ActuatorAliasEndpoint("NAMED", "renamed", None, (), None, None),
            ),
        ),
        candidate_hash=SHA_B,
    )
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity.align_compiled_models(base, renamed, missing)
    assert _refusal_reason(exc) is OperationalReasonCode.ACTUATOR_IDENTITY_MISSING
