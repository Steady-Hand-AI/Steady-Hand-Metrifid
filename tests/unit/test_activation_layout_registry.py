"""Collect activation layout registry scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

import copy
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import cast

import mujoco  # type: ignore[import-untyped]
import pytest

from metrifid import _model_admission as admission
from metrifid import _model_closure as closure
from metrifid import _model_compile as model_compile
from metrifid import _model_descriptor_builders as descriptor_builders
from metrifid import _model_descriptors as model_descriptors
from metrifid import _model_identity as identity
from metrifid import _model_identity_validation as validation
from metrifid.json_values import CanonicalValue, compute_self_hash
from metrifid.operational import OperationalReasonCode
from metrifid.schemas import TargetReference
from tests._support.model_identity import build_model_pair_identity

SHA_A = "a" * 64
SHA_B = "b" * 64
TARGET = (TargetReference("JOINT", "j"),)
VALID_LAYOUTS = (
    ("NONE", 0, None),
    ("INTEGRATOR", 1, 0),
    ("FILTER", 1, 0),
    ("FILTEREXACT", 1, 0),
    ("MUSCLE", 1, 0),
    ("DCMOTOR", 0, None),
    ("DCMOTOR", 1, 0),
)
INVALID_LAYOUTS = (
    ("NONE", 1, 0),
    ("INTEGRATOR", 0, None),
    ("INTEGRATOR", 2, 0),
    ("DCMOTOR", 2, 0),
    ("USER", 0, None),
)


def _descriptor(family: str, width: int, address: int | None) -> admission.ActuatorDescriptor:
    """Construct the descriptor fixture used by activation layout registry scenarios.

    Deterministic setup isolates activation layout registry without bypassing the contract
    boundary under assertion.
    """
    return admission.ActuatorDescriptor(
        "a",
        "JOINT",
        TARGET,
        cast(closure.ActivationFamily, family),
        width,
        0,
        address,
    )


def _aligned(family: str, width: int, address: int | None) -> closure.AlignedActuator:
    """Construct the aligned fixture used by activation layout registry scenarios.

    Deterministic setup isolates activation layout registry without bypassing the contract
    boundary under assertion.
    """
    return closure.AlignedActuator(
        "a",
        "JOINT",
        TARGET,
        cast(closure.ActivationFamily, family),
        width,
        0,
        0,
        address,
        address,
    )


def _compiled(family: str, width: int, address: int | None) -> admission.CompiledModelIdentity:
    """Construct the compiled fixture used by activation layout registry scenarios.

    Deterministic setup isolates activation layout registry without bypassing the contract
    boundary under assertion.
    """
    joint = admission.JointDescriptor("j", "HINGE", 1, 1, 0, 0)
    actuator = _descriptor(family, width, address)
    return admission.CompiledModelIdentity(
        admission.CompiledModelIdentity._SCHEMA,
        admission.CompiledModelIdentity._SCHEMA_VERSION,
        None,
        SHA_A,
        1,
        1,
        1,
        width,
        (joint,),
        (actuator,),
    ).finalized()


def _rehash(value: dict[str, object], field: str) -> None:
    """Compute the canonical rehash value used by activation layout registry fixtures.

    Content addressing keeps the mutation boundary explicit for activation layout registry.
    """
    value[field] = None
    value[field] = compute_self_hash(cast(dict[str, CanonicalValue], value), field)


def _motor_pair(tmp_path: Path) -> identity.ModelPairIdentity:
    """Construct the motor pair fixture used by activation layout registry scenarios.

    Deterministic setup isolates activation layout registry without bypassing the contract
    boundary under assertion.
    """
    xml = (
        '<mujoco><worldbody><body><joint name="j"/>'
        '<geom size=".1" mass="1"/></body></worldbody>'
        '<actuator><motor name="a" joint="j"/></actuator></mujoco>'
    )
    (tmp_path / "model.xml").write_text(xml, encoding="utf-8")
    return build_model_pair_identity(tmp_path, "model.xml", tmp_path, "model.xml")


def test_registry_is_one_exact_immutable_source_of_truth() -> None:
    """Protect the activation layout registry assurance boundary from behavioral drift.

    This scenario exercises registry is one exact immutable source of truth; the assertions pin
    the user-visible result and the evidence needed to explain that result.
    """
    assert isinstance(closure._ACTIVATION_LAYOUT_WIDTHS, MappingProxyType)
    assert dict(closure._ACTIVATION_LAYOUT_WIDTHS) == {
        "NONE": (0,),
        "INTEGRATOR": (1,),
        "FILTER": (1,),
        "FILTEREXACT": (1,),
        "MUSCLE": (1,),
        "DCMOTOR": (0, 1),
    }
    with pytest.raises(TypeError):
        closure._ACTIVATION_LAYOUT_WIDTHS["NONE"] = (1,)  # type: ignore[index]


@pytest.mark.parametrize(("family", "width", "address"), VALID_LAYOUTS)
def test_valid_layouts_survive_direct_and_strict_roundtrips(
    family: str, width: int, address: int | None
) -> None:
    """Protect the activation layout registry assurance boundary from behavioral drift.

    This scenario exercises valid layouts survive direct and strict roundtrips; the assertions
    pin the user-visible result and the evidence needed to explain that result.
    """
    descriptor = _descriptor(family, width, address)
    assert admission.ActuatorDescriptor.from_primitive(descriptor.to_primitive()) == descriptor
    aligned = _aligned(family, width, address)
    assert closure.AlignedActuator.from_primitive(aligned.to_primitive()) == aligned
    completed = _compiled(family, width, address)
    assert admission.CompiledModelIdentity.from_primitive(completed.to_primitive()) == completed


@pytest.mark.parametrize(("family", "width", "address"), INVALID_LAYOUTS)
def test_invalid_layouts_refuse_direct_and_primitive_construction(
    family: str, width: int, address: int | None
) -> None:
    """Protect the activation layout registry assurance boundary from behavioral drift.

    This scenario exercises invalid layouts refuse direct and primitive construction; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    with pytest.raises(ValueError):
        _descriptor(family, width, address)
    primitive = _descriptor("NONE", 0, None).to_primitive()
    primitive.update(
        activation_family=family,
        activation_width=width,
        activation_address=address,
    )
    with pytest.raises(ValueError):
        admission.ActuatorDescriptor.from_primitive(primitive)
    with pytest.raises(ValueError):
        _aligned(family, width, address)
    aligned_primitive = _aligned("NONE", 0, None).to_primitive()
    aligned_primitive.update(
        activation_family=family,
        activation_width=width,
        baseline_activation_address=address,
        candidate_activation_address=address,
    )
    with pytest.raises(ValueError):
        closure.AlignedActuator.from_primitive(aligned_primitive)


@pytest.mark.parametrize(("family", "width", "address"), INVALID_LAYOUTS)
def test_fully_rehashed_impossible_completed_identity_and_pair_refuse(
    tmp_path: Path, family: str, width: int, address: int | None
) -> None:
    """Protect the activation layout registry assurance boundary from behavioral drift.

    This scenario exercises fully rehashed impossible completed identity and pair refuse; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    pair_value = copy.deepcopy(_motor_pair(tmp_path).to_primitive())
    for side in ("baseline_compiled", "candidate_compiled"):
        compiled = cast(dict[str, object], pair_value[side])
        compiled["na"] = width
        actuator = cast(list[dict[str, object]], compiled["actuators"])[0]
        actuator.update(
            activation_family=family,
            activation_width=width,
            activation_address=address,
        )
        _rehash(compiled, "compiled_identity_sha256")
        with pytest.raises(ValueError):
            admission.CompiledModelIdentity.from_primitive(copy.deepcopy(compiled))
    alignment = cast(dict[str, object], pair_value["alignment"])
    aligned = cast(list[dict[str, object]], alignment["actuators"])[0]
    aligned.update(
        activation_family=family,
        activation_width=width,
        baseline_activation_address=address,
        candidate_activation_address=address,
    )
    _rehash(alignment, "semantic_alignment_sha256")
    _rehash(pair_value, "model_pair_identity_sha256")
    with pytest.raises(ValueError):
        identity.ModelPairIdentity.from_primitive(pair_value)


@pytest.mark.parametrize(("family", "width", "_address"), VALID_LAYOUTS)
def test_normal_mujoco_extraction_emits_only_registered_layouts(
    family: str, width: int, _address: int | None
) -> None:
    """Protect the activation layout registry assurance boundary from behavioral drift.

    This scenario exercises normal MuJoCo extraction emits only registered layouts; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    token = family.lower()
    early = ' actearly="true"' if family == "DCMOTOR" else ""
    xml = (
        '<mujoco><worldbody><body><joint name="j"/>'
        '<geom size=".1" mass="1"/></body></worldbody>'
        f'<actuator><general name="a" joint="j" dyntype="{token}" '
        f'actdim="{width}"{early}/></actuator></mujoco>'
    )
    model = mujoco.MjModel.from_xml_string(xml)
    descriptor = admission.compile_model_identity(model, SHA_A, "baseline").actuators[0]
    assert (
        descriptor.activation_width
        in closure._ACTIVATION_LAYOUT_WIDTHS[descriptor.activation_family]
    )


def test_generated_descriptor_sanity_refuses_an_impossible_compiled_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect the activation layout registry assurance boundary from behavioral drift.

    This scenario exercises generated descriptor sanity refuses an impossible compiled layout;
    the assertions pin the user-visible result and the evidence needed to explain that result.
    """
    model = SimpleNamespace(
        nu=1,
        actuator_trntype=[int(mujoco.mjtTrn.mjTRN_JOINT)],
        actuator_dyntype=[int(mujoco.mjtDyn.mjDYN_NONE)],
        actuator_actadr=[0],
        actuator_actnum=[1],
    )
    monkeypatch.setattr(model_compile.mujoco, "mj_id2name", lambda *_: "a")
    monkeypatch.setattr(descriptor_builders, "_actuator_targets", lambda *_: TARGET)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        model_descriptors._actuator_descriptors(cast(mujoco.MjModel, model), "baseline")
    assert exc.value.reason is OperationalReasonCode.MODEL_LAYOUT_INCOMPATIBLE
    assert exc.value.evidence["issue"] == "unsupported_activation_layout"


def test_activation_nonoverlap_precedes_uniqueness_and_disjoint_slices_pass() -> None:
    """Protect the activation layout registry assurance boundary from behavioral drift.

    This scenario exercises activation nonoverlap precedes uniqueness and disjoint slices pass;
    the assertions pin the user-visible result and the evidence needed to explain that result.
    """
    duplicate = (
        closure.AlignedActuator("a", "JOINT", TARGET, "INTEGRATOR", 1, 0, 0, 0, 0),
        closure.AlignedActuator("b", "JOINT", TARGET, "INTEGRATOR", 1, 1, 1, 0, 0),
    )
    with pytest.raises(ValueError, match="activation slices must be nonoverlapping"):
        validation.validate_semantic_alignment_local((), duplicate, None, None, ())
    disjoint = (
        duplicate[0],
        closure.AlignedActuator("b", "JOINT", TARGET, "INTEGRATOR", 1, 1, 1, 1, 1),
    )
    validation.validate_semantic_alignment_local((), disjoint, None, None, ())
