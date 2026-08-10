"""Collect model identity validation scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import cast

import pytest

from metrifid import _model_admission as admission
from metrifid import _model_closure as closure
from metrifid import _model_descriptors as model_descriptors
from metrifid import _model_identity as identity
from metrifid import _model_identity_validation as validation
from metrifid.json_values import (
    CanonicalValue,
    canonical_sha256,
    compute_self_hash,
    freeze_canonical,
)
from metrifid.operational import OperationalReasonCode
from metrifid.schemas import (
    ActuatorAliasEndpoint,
    ActuatorAliasPair,
    AliasArtifact,
    JointAliasPair,
    ModelClosureIdentity,
    ModelClosureMember,
    TargetReference,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
RAW_SHA = "c" * 64


def _closure(content_sha: str) -> ModelClosureIdentity:
    """Build a one-member closure bound to the selected content digest."""
    return ModelClosureIdentity(
        "model.xml",
        1,
        (ModelClosureMember("model.xml", 1, content_sha),),
    )


def _joint(
    name: str = "joint",
    *,
    qpos_address: int = 0,
    qvel_address: int = 0,
) -> admission.JointDescriptor:
    """Build a named hinge descriptor with selected state addresses."""
    return admission.JointDescriptor(name, "HINGE", 1, 1, qpos_address, qvel_address)


def _actuator(
    name: str | None = "act",
    *,
    target: str = "joint",
    control: int = 0,
    activation: int | None = None,
    width: int = 0,
) -> admission.ActuatorDescriptor:
    """Build a joint actuator descriptor with selected control and activation layout."""
    return admission.ActuatorDescriptor(
        name,
        "JOINT",
        (TargetReference("JOINT", target),),
        "NONE" if width == 0 else "INTEGRATOR",
        width,
        control,
        activation,
    )


def _compiled(
    closure_sha: str,
    *,
    joints: tuple[admission.JointDescriptor, ...] = (_joint(),),
    actuators: tuple[admission.ActuatorDescriptor, ...] = (_actuator(),),
) -> admission.CompiledModelIdentity:
    """Finalize compiled identity evidence for supplied joints and actuators."""
    return admission.CompiledModelIdentity(
        admission.CompiledModelIdentity._SCHEMA,
        admission.CompiledModelIdentity._SCHEMA_VERSION,
        None,
        closure_sha,
        sum(item.qpos_width for item in joints),
        sum(item.qvel_width for item in joints),
        len(actuators),
        sum(item.activation_width for item in actuators),
        tuple(sorted(joints, key=lambda item: item.name)),
        tuple(sorted(actuators, key=model_descriptors._actuator_sort_key)),
    ).finalized()


def _pair_no_alias(
    *,
    joints: tuple[admission.JointDescriptor, ...] = (_joint(),),
    actuators: tuple[admission.ActuatorDescriptor, ...] = (_actuator(),),
) -> identity.ModelPairIdentity:
    """Finalize an aligned model pair that needs no explicit aliases."""
    baseline_closure = _closure(SHA_A)
    candidate_closure = _closure(SHA_A)
    baseline = _compiled(baseline_closure.sha256(), joints=joints, actuators=actuators)
    candidate = _compiled(candidate_closure.sha256(), joints=joints, actuators=actuators)
    alignment = identity.align_compiled_models(baseline, candidate)
    return identity.ModelPairIdentity(
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


def _rehash_pair(primitive: dict[str, CanonicalValue]) -> None:
    """Recompute nested alignment and pair hashes after a fixture mutation."""
    alignment = cast(dict[str, CanonicalValue], primitive["alignment"])
    alignment["semantic_alignment_sha256"] = compute_self_hash(
        alignment, "semantic_alignment_sha256"
    )
    summary = cast(dict[str, CanonicalValue], primitive["alignment_summary"])
    summary["alignment_sha256"] = compute_self_hash(summary, "alignment_sha256")
    primitive["model_pair_identity_sha256"] = compute_self_hash(
        primitive, "model_pair_identity_sha256"
    )


def test_duplicate_named_actuators_are_refused() -> None:
    """Refuse compiled identities whose nonnull actuator names are ambiguous."""
    first = _actuator("same", control=0)
    second = _actuator("same", control=1)
    with pytest.raises(ValueError, match="nonnull actuator names"):
        _compiled(_closure(SHA_A).sha256(), actuators=(first, second))
    validation.validate_unique_named_actuators((_actuator(),))


def test_local_joint_control_and_activation_slice_uniqueness() -> None:
    """Refuse overlapping joint, control, and activation address slices."""
    pair = _pair_no_alias(actuators=())
    aligned_joint = pair.alignment.joints[0]
    with pytest.raises(ValueError, match="baseline qpos slices"):
        replace(
            pair.alignment,
            semantic_alignment_sha256=None,
            joints=(aligned_joint, replace(aligned_joint, canonical_name="second")),
        )

    pair_two = _pair_no_alias(
        joints=(
            _joint("j1", qpos_address=0, qvel_address=0),
            _joint("j2", qpos_address=1, qvel_address=1),
        ),
        actuators=(
            _actuator("a1", target="j1", control=0),
            _actuator("a2", target="j2", control=1),
        ),
    )
    first, second = pair_two.alignment.actuators
    with pytest.raises(ValueError, match="control addresses"):
        replace(
            pair_two.alignment,
            semantic_alignment_sha256=None,
            actuators=(first, replace(second, baseline_control_address=0)),
        )

    activated = (
        closure.AlignedActuator(
            "a1", "JOINT", (TargetReference("JOINT", "j1"),), "INTEGRATOR", 1, 0, 0, 0, 0
        ),
        closure.AlignedActuator(
            "a2", "JOINT", (TargetReference("JOINT", "j2"),), "INTEGRATOR", 1, 1, 1, 0, 0
        ),
    )
    with pytest.raises(ValueError, match="activation slices must be nonoverlapping"):
        identity.SemanticAlignment(
            identity.SemanticAlignment._SCHEMA,
            identity.SemanticAlignment._SCHEMA_VERSION,
            None,
            None,
            None,
            pair_two.alignment.joints,
            activated,
            (),
        )


def test_aligned_actuator_enforces_supported_transmission_target_shape() -> None:
    """Refuse actuator targets that contradict the supported transmission type."""
    with pytest.raises(ValueError, match="targets do not match"):
        closure.AlignedActuator(
            "a", "BOGUS", (TargetReference("JOINT", "j"),), "NONE", 0, 0, 0, None, None
        )
    with pytest.raises(ValueError, match="targets do not match"):
        closure.AlignedActuator(
            "a", "JOINT", (TargetReference("SITE", "s"),), "NONE", 0, 0, 0, None, None
        )


def test_exact_alias_binding_schema_and_hash_presence() -> None:
    """Require exact alias fields plus paired raw and semantic digests."""
    valid_joint: dict[str, CanonicalValue] = {
        "kind": "JOINT",
        "canonical_name": "canonical",
        "baseline_name": "left",
        "candidate_name": "right",
    }
    valid_endpoint = ActuatorAliasEndpoint("NAMED", "act", None, (), None, None)
    valid_actuator: dict[str, CanonicalValue] = {
        "kind": "ACTUATOR",
        **ActuatorAliasPair("act", valid_endpoint, valid_endpoint).to_primitive(),
    }
    parsed = validation.strict_alias_bindings_from_primitive([valid_joint, valid_actuator])
    joints, actuators = validation.parse_alias_bindings(parsed)
    assert joints == (JointAliasPair("canonical", "left", "right"),)
    assert actuators[0].canonical_name == "act"

    for value in (
        "bad",
        [1],
        [{"kind": "UNKNOWN"}],
        [{"kind": "JOINT", "canonical_name": "x"}],
        [{**valid_joint, "extra": "x"}],
        [{"kind": "ACTUATOR", "canonical_name": "a"}],
        [{**valid_actuator, "extra": "x"}],
    ):
        with pytest.raises((TypeError, ValueError)):
            validation.strict_alias_bindings_from_primitive(value)

    frozen = tuple(cast(object, freeze_canonical(item)) for item in [valid_joint])
    with pytest.raises(ValueError, match="empty binding set"):
        validation.validate_semantic_alignment_local((), (), None, None, frozen)  # type: ignore[arg-type]


def test_internal_binding_object_and_unreachable_defense_branches() -> None:
    """Reject nonobject bindings and inconsistent internal alias evidence."""
    array_binding = cast(object, freeze_canonical([]))
    with pytest.raises(TypeError, match="must be an object"):
        validation._binding_primitive(array_binding)  # type: ignore[arg-type]

    pair = _pair_no_alias()
    unsafe = object.__new__(identity.SemanticAlignment)
    for name, value in {
        "schema": identity.SemanticAlignment._SCHEMA,
        "schema_version": identity.SemanticAlignment._SCHEMA_VERSION,
        "semantic_alignment_sha256": pair.alignment.semantic_alignment_sha256,
        "aliases_raw_sha256": None,
        "aliases_semantic_sha256": None,
        "joints": pair.alignment.joints,
        "actuators": pair.alignment.actuators,
        "alias_bindings": (
            freeze_canonical(
                {
                    "kind": "JOINT",
                    "canonical_name": "joint",
                    "baseline_name": "joint",
                    "candidate_name": "joint",
                }
            ),
        ),
    }.items():
        object.__setattr__(unsafe, name, value)
    with pytest.raises(ValueError, match="require alias hashes"):
        validation._reconstructed_aliases(
            pair.baseline_closure.sha256(), pair.candidate_closure.sha256(), unsafe
        )

    with pytest.raises(ValueError, match="both be present or absent"):
        validation.validate_semantic_alignment_local((), (), SHA_A, None, ())


def test_alias_binding_endpoint_and_canonical_uniqueness() -> None:
    """Require unique canonical names and role-specific alias endpoints."""
    joint_a = cast(
        object,
        freeze_canonical(
            {"kind": "JOINT", "canonical_name": "x", "baseline_name": "l", "candidate_name": "r"}
        ),
    )
    joint_b = cast(
        object,
        freeze_canonical(
            {"kind": "JOINT", "canonical_name": "x", "baseline_name": "l2", "candidate_name": "r2"}
        ),
    )
    with pytest.raises(ValueError, match="canonical names"):
        validation.parse_alias_bindings((joint_a, joint_b))  # type: ignore[arg-type]

    joint_c = cast(
        object,
        freeze_canonical(
            {"kind": "JOINT", "canonical_name": "y", "baseline_name": "l", "candidate_name": "r2"}
        ),
    )
    with pytest.raises(ValueError, match="baseline joint alias endpoints"):
        validation.parse_alias_bindings((joint_a, joint_c))  # type: ignore[arg-type]

    endpoint = ActuatorAliasEndpoint("NAMED", "a", None, (), None, None)
    act_a = cast(
        object,
        freeze_canonical(
            {"kind": "ACTUATOR", **ActuatorAliasPair("x", endpoint, endpoint).to_primitive()}
        ),
    )
    act_b = cast(
        object,
        freeze_canonical(
            {"kind": "ACTUATOR", **ActuatorAliasPair("y", endpoint, endpoint).to_primitive()}
        ),
    )
    with pytest.raises(ValueError, match="baseline actuator alias endpoints"):
        validation.parse_alias_bindings((act_a, act_b))  # type: ignore[arg-type]


def test_valid_explicit_joint_and_actuator_aliases_roundtrip() -> None:
    """Round-trip finalized explicit joint and actuator alias evidence."""
    baseline_closure = _closure(SHA_A)
    candidate_closure = _closure(SHA_B)
    baseline = _compiled(
        baseline_closure.sha256(),
        joints=(_joint("left_joint"),),
        actuators=(_actuator("left_act", target="left_joint"),),
    )
    candidate = _compiled(
        candidate_closure.sha256(),
        joints=(_joint("right_joint"),),
        actuators=(_actuator("right_act", target="right_joint"),),
    )
    endpoint_left = ActuatorAliasEndpoint("NAMED", "left_act", None, (), None, None)
    endpoint_right = ActuatorAliasEndpoint("NAMED", "right_act", None, (), None, None)
    artifact = AliasArtifact(
        "metrifid.aliases",
        1,
        baseline_closure.sha256(),
        candidate_closure.sha256(),
        (JointAliasPair("joint", "left_joint", "right_joint"),),
        (ActuatorAliasPair("act", endpoint_left, endpoint_right),),
    )
    semantic_hash = canonical_sha256(artifact.to_primitive())
    alignment = identity.align_compiled_models(
        baseline,
        candidate,
        artifact,
        aliases_raw_sha256=RAW_SHA,
        aliases_semantic_sha256=semantic_hash,
    )
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
    assert identity.ModelPairIdentity.from_primitive(pair.to_primitive()) == pair


def test_reconstructed_alias_hash_and_phantom_binding_refuse() -> None:
    """Refuse forged alias hashes and bindings to absent endpoints."""
    pair = _pair_no_alias()
    primitive = pair.to_primitive()
    alignment = cast(dict[str, CanonicalValue], primitive["alignment"])
    alignment["aliases_raw_sha256"] = RAW_SHA
    alignment["aliases_semantic_sha256"] = SHA_B
    _rehash_pair(primitive)
    with pytest.raises(ValueError, match="reconstructed aliases"):
        identity.ModelPairIdentity.from_primitive(primitive)

    phantom = pair.to_primitive()
    alignment = cast(dict[str, CanonicalValue], phantom["alignment"])
    binding: dict[str, CanonicalValue] = {
        "kind": "JOINT",
        "canonical_name": "joint",
        "baseline_name": "missing",
        "candidate_name": "joint",
    }
    artifact = AliasArtifact(
        "metrifid.aliases",
        1,
        pair.baseline_closure.sha256(),
        pair.candidate_closure.sha256(),
        (JointAliasPair("joint", "missing", "joint"),),
        (),
    )
    alignment["aliases_raw_sha256"] = RAW_SHA
    alignment["aliases_semantic_sha256"] = canonical_sha256(artifact.to_primitive())
    alignment["alias_bindings"] = [binding]
    _rehash_pair(phantom)
    with pytest.raises(ValueError, match="cannot be regenerated"):
        identity.ModelPairIdentity.from_primitive(phantom)


def test_forged_canonical_names_and_duplicate_bijections_refuse() -> None:
    """Refuse forged canonical names and duplicate alignment slices."""
    pair = _pair_no_alias()
    forged = pair.to_primitive()
    alignment = cast(dict[str, CanonicalValue], forged["alignment"])
    joints = cast(list[dict[str, CanonicalValue]], alignment["joints"])
    actuators = cast(list[dict[str, CanonicalValue]], alignment["actuators"])
    joints[0]["canonical_name"] = "forged_joint"
    actuators[0]["canonical_name"] = "forged_actuator"
    cast(list[dict[str, CanonicalValue]], actuators[0]["targets"])[0]["name"] = "forged_joint"
    summary = cast(dict[str, CanonicalValue], forged["alignment_summary"])
    summary["joint_order"] = ["forged_joint"]
    summary["actuator_order"] = ["forged_actuator"]
    _rehash_pair(forged)
    with pytest.raises(ValueError, match="differs from generated"):
        identity.ModelPairIdentity.from_primitive(forged)

    no_act = _pair_no_alias(actuators=())
    duplicate = no_act.to_primitive()
    alignment = cast(dict[str, CanonicalValue], duplicate["alignment"])
    joints = cast(list[dict[str, CanonicalValue]], alignment["joints"])
    second = copy.deepcopy(joints[0])
    second["canonical_name"] = "second"
    joints.append(second)
    joints.sort(key=lambda item: cast(str, item["canonical_name"]))
    summary = cast(dict[str, CanonicalValue], duplicate["alignment_summary"])
    summary["joint_order"] = [cast(str, item["canonical_name"]) for item in joints]
    _rehash_pair(duplicate)
    with pytest.raises(ValueError, match="qpos slices must be unique"):
        identity.ModelPairIdentity.from_primitive(duplicate)


def test_full_pair_exact_coverage_and_expected_alignment_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require exact role coverage and reproducible regenerated alignment."""
    pair = _pair_no_alias()
    with pytest.raises(ValueError, match="exact one-to-one cover"):
        validation._exact_coverage([1], [1, 1], "values")
    validation._exact_coverage([1, 2], [2, 1], "values")

    original = identity.align_compiled_models

    def refused(*args: object, **kwargs: object) -> identity.SemanticAlignment:
        """Inject the deterministic refused branch required by this scenario.

        The model identity validation test can assert failure delivery for full pair exact
        coverage and expected alignment branches without depending on incidental runtime errors.
        """
        del args, kwargs
        raise closure.ModelAdmissionRefusal(
            # The exact token is immaterial: completed objects convert this to ValueError.
            OperationalReasonCode.JOINT_IDENTITY_MISSING,
            "comparison",
        )

    monkeypatch.setattr(identity, "align_compiled_models", refused)
    with pytest.raises(ValueError, match="cannot be regenerated"):
        validation.validate_model_pair_semantics(
            pair.baseline_closure.sha256(),
            pair.candidate_closure.sha256(),
            pair.baseline_compiled,
            pair.candidate_compiled,
            pair.alignment,
        )
    monkeypatch.setattr(identity, "align_compiled_models", original)


def test_canonical_binding_comparison_detects_noncanonical_internal_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises canonical binding comparison detects noncanonical internal value;
    incomplete, ambiguous, or unbound model components cannot support certification.
    """
    binding = cast(
        object,
        freeze_canonical(
            {"kind": "JOINT", "canonical_name": "j", "baseline_name": "j", "candidate_name": "j"}
        ),
    )
    real = validation._parse_binding

    def changed(value: object) -> tuple[str, JointAliasPair, object]:
        """Apply the targeted changed mutation to otherwise valid evidence.

        Resealing the result isolates the semantic contradiction exercised by canonical binding
        comparison detects noncanonical internal value.
        """
        kind, pair, _ = real(value)  # type: ignore[arg-type]
        normalized = freeze_canonical(
            {
                "kind": "JOINT",
                "canonical_name": "different",
                "baseline_name": "j",
                "candidate_name": "j",
            }
        )
        return kind, cast(JointAliasPair, pair), normalized

    monkeypatch.setattr(validation, "_parse_binding", changed)
    with pytest.raises(ValueError, match="exact canonical schema form"):
        validation.parse_alias_bindings((binding,))  # type: ignore[arg-type]


def test_represented_joint_target_validation_is_deliberate_and_bounded() -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises represented joint target validation is deliberate and bounded;
    incomplete, ambiguous, or unbound model components cannot support certification.
    """
    valid = _actuator(target="joint")
    validation.validate_represented_joint_targets((_joint("joint"),), (valid,))

    missing = _actuator(target="ghost")
    with pytest.raises(ValueError, match=r"^actuator joint target must name a compiled joint$"):
        validation.validate_represented_joint_targets((_joint("joint"),), (missing,))

    for transmission, object_type in (
        ("TENDON", "TENDON"),
        ("SITE", "SITE"),
        ("BODY", "BODY"),
    ):
        detached = admission.ActuatorDescriptor(
            "act",
            transmission,
            (TargetReference(object_type, "not_represented"),),
            "NONE",
            0,
            0,
            None,
        )
        validation.validate_represented_joint_targets((), (detached,))


def test_compiled_identity_rejects_missing_joint_target_for_joint_and_parent() -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises compiled identity rejects missing joint target for joint and parent;
    incomplete, ambiguous, or unbound model components cannot support certification.
    """
    for transmission in ("JOINT", "JOINTINPARENT"):
        actuator = admission.ActuatorDescriptor(
            "act",
            transmission,
            (TargetReference("JOINT", "ghost"),),
            "NONE",
            0,
            0,
            None,
        )
        with pytest.raises(ValueError, match=r"^actuator joint target must name a compiled joint$"):
            _compiled(_closure(SHA_A).sha256(), actuators=(actuator,))

    parent = admission.ActuatorDescriptor(
        "act",
        "JOINTINPARENT",
        (TargetReference("JOINT", "joint"),),
        "NONE",
        0,
        0,
        None,
    )
    assert _compiled(_closure(SHA_A).sha256(), actuators=(parent,)).actuators == (parent,)


def test_nonjoint_target_existence_is_not_overclaimed_by_detached_identity() -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises nonjoint target existence is not overclaimed by detached identity;
    incomplete, ambiguous, or unbound model components cannot support certification.
    """
    for transmission, object_type in (
        ("TENDON", "TENDON"),
        ("SITE", "SITE"),
        ("BODY", "BODY"),
    ):
        actuator = admission.ActuatorDescriptor(
            "act",
            transmission,
            (TargetReference(object_type, "not_represented"),),
            "NONE",
            0,
            0,
            None,
        )
        completed = _compiled(
            _closure(SHA_A).sha256(),
            joints=(),
            actuators=(actuator,),
        )
        assert completed.actuators == (actuator,)


def test_fully_rehashed_pair_rejects_forged_missing_joint_target_deliberately() -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises fully rehashed pair rejects forged missing joint target
    deliberately; incomplete, ambiguous, or unbound model components cannot support
    certification.
    """
    pair = _pair_no_alias()
    primitive = pair.to_primitive()
    for role in ("baseline_compiled", "candidate_compiled"):
        compiled = cast(dict[str, CanonicalValue], primitive[role])
        actuators = cast(list[dict[str, CanonicalValue]], compiled["actuators"])
        targets = cast(list[dict[str, CanonicalValue]], actuators[0]["targets"])
        targets[0]["name"] = "ghost"
        compiled["compiled_identity_sha256"] = compute_self_hash(
            compiled,
            "compiled_identity_sha256",
        )
    alignment = cast(dict[str, CanonicalValue], primitive["alignment"])
    actuators = cast(list[dict[str, CanonicalValue]], alignment["actuators"])
    targets = cast(list[dict[str, CanonicalValue]], actuators[0]["targets"])
    targets[0]["name"] = "ghost"
    _rehash_pair(primitive)

    with pytest.raises(ValueError, match=r"^actuator joint target must name a compiled joint$"):
        identity.ModelPairIdentity.from_primitive(primitive)
