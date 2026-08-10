"""Collect model alignment scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from metrifid import _model_admission as admission
from metrifid import _model_closure as closure
from metrifid import _model_descriptors as model_descriptors
from metrifid import _model_identity as identity
from metrifid import _model_identity_validation as validation
from metrifid.json_values import canonical_json_bytes, canonical_sha256, freeze_canonical
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
    """Construct the joint fixture used by model alignment scenarios.

    Deterministic setup isolates model alignment without bypassing the contract boundary under
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
    """Construct the actuator fixture used by model alignment scenarios.

    Deterministic setup isolates model alignment without bypassing the contract boundary under
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
    """Construct the compiled fixture used by model alignment scenarios.

    Deterministic setup isolates model alignment without bypassing the contract boundary under
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
    """Construct the alias fixture used by model alignment scenarios.

    Deterministic setup isolates model alignment without bypassing the contract boundary under
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
    """Extract the model-admission code from an alignment refusal."""
    return exc.value.reason


def _unsafe_joint(
    *,
    qpos_width: int = 1,
    qvel_width: int = 1,
    joint_type: str = "HINGE",
) -> admission.JointDescriptor:
    """Construct the unsafe joint fixture used by model alignment scenarios.

    Deterministic setup isolates model alignment without bypassing the contract boundary under
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


def test_unique_unnamed_selector_zero_and_ambiguous() -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises unique unnamed selector zero and ambiguous; incomplete, ambiguous,
    or unbound model components cannot support certification.
    """
    unnamed = _actuator(None)
    left = _compiled(actuators=(unnamed,))
    right = _compiled(actuators=(unnamed,), closure_hash=SHA_B)
    selector = ActuatorAliasEndpoint(
        "UNNAMED_SELECTOR",
        None,
        "JOINT",
        (TargetReference("JOINT", "joint"),),
        "NONE",
        0,
    )
    aliases = _alias(
        actuators=(ActuatorAliasPair("canonical", selector, selector),),
        candidate_hash=SHA_B,
    )
    result = identity.align_compiled_models(
        left,
        right,
        aliases,
        aliases_raw_sha256=SHA_A,
        aliases_semantic_sha256=canonical_sha256(aliases.to_primitive()),
    )
    assert result.actuators[0].canonical_name == "canonical"

    no_match_selector = replace(selector, activation_family="FILTER")
    no_match = _alias(
        actuators=(ActuatorAliasPair("canonical", no_match_selector, selector),),
        candidate_hash=SHA_B,
    )
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity.align_compiled_models(left, right, no_match)
    assert _refusal_reason(exc) is OperationalReasonCode.ALIAS_SELECTOR_NO_MATCH

    first = _actuator(None, control=0)
    second = _actuator(None, control=1)
    ambiguous_left = _compiled(actuators=(first, second))
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity.align_compiled_models(ambiguous_left, right, aliases)
    assert _refusal_reason(exc) is OperationalReasonCode.ALIAS_SELECTOR_AMBIGUOUS


def test_actuator_pairing_collision_leftover_and_mismatch_reasons() -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises actuator pairing collision leftover and mismatch reasons;
    incomplete, ambiguous, or unbound model components cannot support certification.
    """
    base = _compiled()
    two = _compiled(
        actuators=(
            _actuator("act", control=0),
            _actuator("other", control=1),
        )
    )
    collision_alias = _alias(
        actuators=(
            ActuatorAliasPair(
                "other",
                ActuatorAliasEndpoint("NAMED", "act", None, (), None, None),
                ActuatorAliasEndpoint("NAMED", "act", None, (), None, None),
            ),
        )
    )
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity.align_compiled_models(two, two, collision_alias)
    assert _refusal_reason(exc) is OperationalReasonCode.ALIAS_BINDING_DUPLICATE

    extra = _compiled(actuators=())
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity.align_compiled_models(base, extra)
    assert _refusal_reason(exc) is OperationalReasonCode.ACTUATOR_IDENTITY_MISSING

    transmission = _compiled(
        actuators=(_actuator(transmission="TENDON", targets=(TargetReference("TENDON", "t"),)),),
        closure_hash=SHA_B,
    )
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity.align_compiled_models(base, transmission)
    assert _refusal_reason(exc) is OperationalReasonCode.ACTUATOR_TRANSMISSION_MISMATCH

    site_base = _compiled(
        actuators=(_actuator(transmission="SITE", targets=(TargetReference("SITE", "s1"),)),)
    )
    target = _compiled(
        actuators=(_actuator(transmission="SITE", targets=(TargetReference("SITE", "s2"),)),),
        closure_hash=SHA_B,
    )
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity.align_compiled_models(site_base, target)
    assert _refusal_reason(exc) is OperationalReasonCode.ACTUATOR_TARGET_MISMATCH

    family = _compiled(
        actuators=(_actuator(family="INTEGRATOR", width=1, activation=0),), closure_hash=SHA_B
    )
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity.align_compiled_models(base, family)
    assert _refusal_reason(exc) is OperationalReasonCode.ACTUATOR_ACTIVATION_FAMILY_MISMATCH

    width_left = _compiled(actuators=(_actuator(family="DCMOTOR", width=0),))
    width_right = _compiled(
        actuators=(_actuator(family="DCMOTOR", width=1, activation=0),), closure_hash=SHA_B
    )
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity.align_compiled_models(width_left, width_right)
    assert _refusal_reason(exc) is OperationalReasonCode.ACTUATOR_ACTIVATION_WIDTH_MISMATCH


def test_alias_parser_none_str_bytes_and_refusals() -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises alias parser none str bytes and refusals; incomplete, ambiguous, or
    unbound model components cannot support certification.
    """
    assert identity._parse_aliases(None, SHA_A, SHA_B) == (None, None, None)
    artifact = _alias(baseline_hash=SHA_A, candidate_hash=SHA_B)
    raw = canonical_json_bytes(artifact.to_primitive())
    parsed, raw_hash, semantic_hash = identity._parse_aliases(raw, SHA_A, SHA_B)
    assert parsed == artifact
    assert raw_hash is not None
    assert semantic_hash is not None
    assert identity._parse_aliases(raw.decode(), SHA_A, SHA_B)[0] == artifact

    for invalid in (3, b"{", b'{"a":1,"a":2}', "\ud800"):
        with pytest.raises(closure.ModelAdmissionRefusal) as exc:
            identity._parse_aliases(invalid, SHA_A, SHA_B)  # type: ignore[arg-type]
        assert _refusal_reason(exc) is OperationalReasonCode.ALIAS_SCHEMA_INVALID

    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity._parse_aliases(raw, SHA_B, SHA_A)
    assert _refusal_reason(exc) is OperationalReasonCode.ALIAS_CLOSURE_HASH_MISMATCH


def test_semantic_alignment_strict_parser_mutations() -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises semantic alignment strict parser mutations; incomplete, ambiguous,
    or unbound model components cannot support certification.
    """
    value = identity.align_compiled_models(_compiled(), _compiled(closure_hash=SHA_B))
    primitive = value.to_primitive()
    assert identity.SemanticAlignment.from_primitive(primitive) == value
    unfinished = replace(value, semantic_alignment_sha256=None)
    with pytest.raises(ValueError):
        unfinished.to_primitive()
    assert unfinished.finalized().semantic_alignment_sha256 is not None

    mutations = [
        lambda obj: obj.update(schema_version="bad"),
        lambda obj: obj.update(semantic_alignment_sha256=None),
        lambda obj: obj.update(semantic_alignment_sha256="0" * 64),
        lambda obj: obj.update(aliases_raw_sha256=SHA_A),
        lambda obj: obj.update(extra=True),
        lambda obj: obj.update(alias_bindings=[1]),
        lambda obj: obj.update(joints="bad"),
        lambda obj: obj.update(actuators="bad"),
    ]
    for mutation in mutations:
        changed = copy.deepcopy(primitive)
        mutation(changed)
        with pytest.raises((TypeError, ValueError)):
            identity.SemanticAlignment.from_primitive(changed)

    duplicated = copy.deepcopy(primitive)
    duplicated["alias_bindings"] = [{"a": 1}, {"a": 1}]
    duplicated["semantic_alignment_sha256"] = None
    with pytest.raises(ValueError):
        identity.SemanticAlignment(
            identity.SemanticAlignment._SCHEMA,
            identity.SemanticAlignment._SCHEMA_VERSION,
            None,
            None,
            None,
            value.joints,
            value.actuators,
            (freeze_canonical({"a": 1}), freeze_canonical({"a": 1})),  # type: ignore[arg-type]
        )


def test_model_pair_completion_roundtrip_and_binding_attacks() -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises model pair completion roundtrip and binding attacks; incomplete,
    ambiguous, or unbound model components cannot support certification.
    """
    from metrifid.schemas import ModelClosureIdentity, ModelClosureMember

    member = ModelClosureMember("model.xml", 1, SHA_A)
    baseline_closure = ModelClosureIdentity("model.xml", 1, (member,))
    candidate_closure = ModelClosureIdentity("model.xml", 1, (member,))
    closure_hash = baseline_closure.sha256()
    baseline = _compiled(closure_hash=closure_hash)
    candidate = _compiled(closure_hash=closure_hash)
    alignment = identity.align_compiled_models(baseline, candidate)
    pair = identity.ModelPairIdentity(
        identity.ModelPairIdentity._SCHEMA,
        identity.ModelPairIdentity._SCHEMA_VERSION,
        None,
        baseline_closure,
        candidate_closure,
        baseline,
        candidate,
        alignment,
        alignment.summary(),
    ).finalized()
    assert pair.finalized() is pair
    assert identity.ModelPairIdentity.from_primitive(pair.to_primitive()) == pair
    unfinished = replace(pair, model_pair_identity_sha256=None)
    with pytest.raises(ValueError):
        unfinished.to_primitive()

    primitive = pair.to_primitive()
    for mutation in (
        lambda obj: obj.update(schema_version="bad"),
        lambda obj: obj.update(model_pair_identity_sha256=None),
        lambda obj: obj.update(model_pair_identity_sha256="0" * 64),
        lambda obj: obj.update(extra=True),
    ):
        changed = copy.deepcopy(primitive)
        mutation(changed)
        with pytest.raises((TypeError, ValueError)):
            identity.ModelPairIdentity.from_primitive(changed)

    with pytest.raises(ValueError):
        replace(
            pair, baseline_compiled=_compiled(closure_hash=SHA_B), model_pair_identity_sha256=None
        )
    with pytest.raises(ValueError):
        replace(
            pair, candidate_compiled=_compiled(closure_hash=SHA_B), model_pair_identity_sha256=None
        )
    wrong_summary = replace(
        pair.alignment_summary, alignment_sha256=None, joint_order=("wrong",)
    ).finalized()
    with pytest.raises(ValueError):
        replace(pair, alignment_summary=wrong_summary, model_pair_identity_sha256=None)


def test_pair_binding_rejects_missing_slices_targets_and_coverage() -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises pair binding rejects missing slices targets and coverage;
    incomplete, ambiguous, or unbound model components cannot support certification.
    """
    baseline = _compiled()
    candidate = _compiled(closure_hash=SHA_B)
    alignment = identity.align_compiled_models(baseline, candidate)
    bad_joint = replace(alignment.joints[0], baseline_qpos=(9, 1))
    _assert_pair_semantics_refused(
        baseline, candidate, replace(alignment, joints=(bad_joint,), semantic_alignment_sha256=None)
    )
    bad_type = replace(alignment.joints[0], joint_type="SLIDE")
    _assert_pair_semantics_refused(
        baseline, candidate, replace(alignment, joints=(bad_type,), semantic_alignment_sha256=None)
    )
    _assert_pair_semantics_refused(
        baseline, candidate, replace(alignment, joints=(), semantic_alignment_sha256=None)
    )
    bad_control = replace(alignment.actuators[0], baseline_control_address=9)
    _assert_pair_semantics_refused(
        baseline,
        candidate,
        replace(alignment, actuators=(bad_control,), semantic_alignment_sha256=None),
    )
    bad_target = replace(alignment.actuators[0], targets=(TargetReference("JOINT", "wrong"),))
    _assert_pair_semantics_refused(
        baseline,
        candidate,
        replace(alignment, actuators=(bad_target,), semantic_alignment_sha256=None),
    )
    _assert_pair_semantics_refused(
        baseline, candidate, replace(alignment, actuators=(), semantic_alignment_sha256=None)
    )


def _assert_pair_semantics_refused(
    baseline: admission.CompiledModelIdentity,
    candidate: admission.CompiledModelIdentity,
    alignment: identity.SemanticAlignment,
) -> None:
    """Assert one malformed alignment fails complete pair semantic validation."""
    with pytest.raises(ValueError):
        validation.validate_model_pair_semantics(
            SHA_A,
            SHA_B,
            baseline,
            candidate,
            alignment,
        )
