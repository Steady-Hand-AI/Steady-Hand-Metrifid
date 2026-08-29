"""Collect MuJoCo model identity scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

import hashlib as _hashlib
import os
import shutil as _shutil
from pathlib import Path

import mujoco
import pytest

from metrifid import _model_admission as admission
from metrifid import _model_closure as closure
from metrifid.json_values import canonical_json_bytes
from metrifid.operational import OperationalReasonCode
from metrifid.schemas import (
    ActuatorAliasEndpoint,
    ActuatorAliasPair,
    AliasArtifact,
    JointAliasPair,
    TargetReference,
)
from tests._support.model_identity import build_model_pair_identity


def _write(root: Path, xml: str, *, name: str = "model.xml") -> Path:
    """Write write data into the isolated test workspace.

    The MuJoCo model identity scenario observes real bytes and filesystem effects for MuJoCo
    model identity.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")
    return root.resolve()


def _hinge_model(
    *,
    joint_name: str = "joint",
    actuator_name: str | None = "act",
    joint_type: str = "hinge",
    body_name: str = "body",
    actuator: str | None = None,
) -> str:
    """Construct the hinge model fixture used by MuJoCo model identity scenarios.

    Deterministic setup isolates MuJoCo model identity without bypassing the contract boundary
    under assertion.
    """
    name_attribute = "" if actuator_name is None else f' name="{actuator_name}"'
    joint_name_attribute = "" if not joint_name else f' name="{joint_name}"'
    actuator_xml = (
        actuator if actuator is not None else f'<motor{name_attribute} joint="{joint_name}"/>'
    )
    return f"""<mujoco model="m">
  <worldbody>
    <body name="{body_name}">
      <joint{joint_name_attribute} type="{joint_type}"/>
      <geom name="geom" type="sphere" size="0.1" mass="1"/>
      <site name="site" pos="0 0 0" size="0.01"/>
    </body>
  </worldbody>
  <actuator>{actuator_xml}</actuator>
</mujoco>"""


def _pair(tmp_path: Path, baseline_xml: str, candidate_xml: str | None = None) -> tuple[Path, Path]:
    """Construct the pair fixture used by MuJoCo model identity scenarios.

    Deterministic setup isolates MuJoCo model identity without bypassing the contract boundary
    under assertion.
    """
    baseline = _write(tmp_path / "baseline", baseline_xml)
    candidate = _write(tmp_path / "candidate", candidate_xml or baseline_xml)
    return baseline, candidate


def _refusal_reason(
    exc: pytest.ExceptionInfo[closure.ModelAdmissionRefusal],
) -> OperationalReasonCode:
    """Extract the stable admission reason from a captured identity refusal."""
    return exc.value.reason


def _compile(root: Path, role: closure.ModelRole) -> mujoco.MjModel:
    """Construct the compile fixture used by MuJoCo model identity scenarios.

    Deterministic setup isolates MuJoCo model identity without bypassing the contract boundary
    under assertion.
    """
    with closure.create_model_closure_snapshot(root, "model.xml", role) as snapshot:
        return admission.compile_snapshot_model(snapshot, role)


def _alias_json(
    baseline: Path,
    candidate: Path,
    *,
    joint_pairs: tuple[JointAliasPair, ...] = (),
    actuator_pairs: tuple[ActuatorAliasPair, ...] = (),
) -> bytes:
    """Construct the alias json fixture used by MuJoCo model identity scenarios.

    Deterministic setup isolates MuJoCo model identity without bypassing the contract boundary
    under assertion.
    """
    baseline_hash = closure.measure_model_closure(baseline, "model.xml", "baseline").sha256()
    candidate_hash = closure.measure_model_closure(candidate, "model.xml", "candidate").sha256()
    artifact = AliasArtifact(
        "metrifid.aliases",
        1,
        baseline_hash,
        candidate_hash,
        joint_pairs,
        actuator_pairs,
    )
    return canonical_json_bytes(artifact.to_primitive())


_UNITREE_G1_ENV = "METRIFID_UNITREE_G1_ROOT"


_UNITREE_PINNED_COMMIT = "ae6a8403e272733e9996ef59990880330496177f"


def _g1_root() -> Path:
    """The official pinned unitree_mujoco g1 model directory."""
    raw = os.environ.get(_UNITREE_G1_ENV)
    if not raw:
        pytest.skip(f"{_UNITREE_G1_ENV} is not set; the pinned Unitree checkout is unavailable")
    root = Path(raw)
    if not (root / "scene.xml").is_file():
        pytest.skip(f"{_UNITREE_G1_ENV}={raw} does not contain scene.xml")
    return root


def _tree_digest(root: Path) -> dict[str, str]:
    """Compute the canonical tree digest value used by MuJoCo model identity fixtures.

    Content addressing keeps the mutation boundary explicit for MuJoCo model identity.
    """
    digest: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            digest[path.relative_to(root).as_posix()] = _hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return digest


@pytest.fixture
def g1_copy(tmp_path: Path) -> Path:
    """A writable copy of the pinned model directory; the official checkout stays untouched."""
    source = _g1_root()
    work = tmp_path / "g1"
    _shutil.copytree(source, work, symlinks=True)
    return work


def test_identity_pair_and_reordered_declarations(tmp_path: Path) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises identity pair and reordered declarations; incomplete, ambiguous, or
    unbound model components cannot support certification.
    """
    baseline = """<mujoco><worldbody>
      <body name="b1"><joint name="j1"/><geom size=".1" mass="1"/></body>
      <body name="b2"><joint name="j2"/><geom size=".1" mass="1"/></body>
    </worldbody><actuator><motor name="a1" joint="j1"/><motor name="a2" joint="j2"/></actuator></mujoco>"""
    candidate = """<mujoco><worldbody>
      <body name="b2"><joint name="j2"/><geom size=".1" mass="1"/></body>
      <body name="b1"><joint name="j1"/><geom size=".1" mass="1"/></body>
    </worldbody><actuator><motor name="a2" joint="j2"/><motor name="a1" joint="j1"/></actuator></mujoco>"""
    left, right = _pair(tmp_path, baseline, candidate)
    result = build_model_pair_identity(left, "model.xml", right, "model.xml")
    assert result.alignment_summary.joint_order == ("j1", "j2")
    assert result.alignment_summary.actuator_order == ("a1", "a2")
    assert result.alignment.joints[0].baseline_qpos != result.alignment.joints[0].candidate_qpos
    assert result.alignment.actuators[0].baseline_control_address != (
        result.alignment.actuators[0].candidate_control_address
    )


def test_all_joint_widths_and_slices(tmp_path: Path) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises all joint widths and slices; incomplete, ambiguous, or unbound model
    components cannot support certification.
    """
    xml = """<mujoco><worldbody>
      <body name="free"><freejoint name="free"/><geom size=".1" mass="1"/></body>
      <body name="ball" pos="1 0 0"><joint name="ball" type="ball"/><geom size=".1" mass="1"/></body>
      <body name="slide" pos="2 0 0"><joint name="slide" type="slide"/><geom size=".1" mass="1"/></body>
      <body name="hinge" pos="3 0 0"><joint name="hinge" type="hinge"/><geom size=".1" mass="1"/></body>
    </worldbody></mujoco>"""
    left, right = _pair(tmp_path, xml)
    result = build_model_pair_identity(left, "model.xml", right, "model.xml")
    measured = {
        item.canonical_name: (
            item.joint_type,
            item.baseline_qpos[1],
            item.baseline_qvel[1],
        )
        for item in result.alignment.joints
    }
    assert measured == {
        "ball": ("BALL", 4, 3),
        "free": ("FREE", 7, 6),
        "hinge": ("HINGE", 1, 1),
        "slide": ("SLIDE", 1, 1),
    }


def test_joint_rename_alias_and_target_rewrite(tmp_path: Path) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises joint rename alias and target rewrite; incomplete, ambiguous, or
    unbound model components cannot support certification.
    """
    left, right = _pair(
        tmp_path,
        _hinge_model(joint_name="old"),
        _hinge_model(joint_name="new"),
    )
    aliases = _alias_json(
        left,
        right,
        joint_pairs=(JointAliasPair("canonical", "old", "new"),),
    )
    result = build_model_pair_identity(left, "model.xml", right, "model.xml", aliases)
    assert result.alignment_summary.joint_order == ("canonical",)
    assert result.alignment.actuators[0].targets[0].name == "canonical"
    assert result.alignment.aliases_raw_sha256 is not None
    assert result.alignment.aliases_semantic_sha256 is not None


def test_unnamed_duplicate_missing_extra_and_type_mismatched_joints(tmp_path: Path) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises unnamed duplicate missing extra and type mismatched joints;
    incomplete, ambiguous, or unbound model components cannot support certification.
    """
    unnamed = _hinge_model(joint_name="", actuator="")
    left, right = _pair(tmp_path / "unnamed", unnamed)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        build_model_pair_identity(left, "model.xml", right, "model.xml")
    assert _refusal_reason(exc) is OperationalReasonCode.JOINT_NAME_MISSING

    duplicate = """<mujoco><worldbody>
      <body><joint name="same"/><geom size=".1" mass="1"/></body>
      <body><joint name="same"/><geom size=".1" mass="1"/></body>
    </worldbody></mujoco>"""
    left, right = _pair(tmp_path / "duplicate", duplicate)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        build_model_pair_identity(left, "model.xml", right, "model.xml")
    assert _refusal_reason(exc) is OperationalReasonCode.BASELINE_MODEL_COMPILE_ERROR

    left, right = _pair(
        tmp_path / "missing",
        _hinge_model(joint_name="left", actuator_name=None, actuator=""),
        _hinge_model(joint_name="right", actuator_name=None, actuator=""),
    )
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        build_model_pair_identity(left, "model.xml", right, "model.xml")
    assert _refusal_reason(exc) is OperationalReasonCode.JOINT_IDENTITY_MISSING

    extra_xml = """<mujoco><worldbody>
      <body><joint name="joint"/><geom size=".1" mass="1"/></body>
      <body><joint name="extra"/><geom size=".1" mass="1"/></body>
    </worldbody></mujoco>"""
    left, right = _pair(
        tmp_path / "extra",
        _hinge_model(actuator_name=None, actuator=""),
        extra_xml,
    )
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        build_model_pair_identity(left, "model.xml", right, "model.xml")
    assert _refusal_reason(exc) is OperationalReasonCode.JOINT_IDENTITY_MISSING

    left, right = _pair(
        tmp_path / "type",
        _hinge_model(actuator_name=None, actuator=""),
        _hinge_model(joint_type="slide", actuator_name=None, actuator=""),
    )
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        build_model_pair_identity(left, "model.xml", right, "model.xml")
    assert _refusal_reason(exc) is OperationalReasonCode.JOINT_TYPE_MISMATCH


def test_named_actuator_reorder_rename_and_unique_unnamed_selector(tmp_path: Path) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises named actuator reorder rename and unique unnamed selector;
    incomplete, ambiguous, or unbound model components cannot support certification.
    """
    left_xml = """<mujoco><worldbody>
      <body><joint name="j1"/><geom size=".1" mass="1"/></body>
      <body><joint name="j2"/><geom size=".1" mass="1"/></body>
    </worldbody><actuator><motor name="a1" joint="j1"/><motor name="a2" joint="j2"/></actuator></mujoco>"""
    right_xml = """<mujoco><worldbody>
      <body><joint name="j1"/><geom size=".1" mass="1"/></body>
      <body><joint name="j2"/><geom size=".1" mass="1"/></body>
    </worldbody><actuator><motor name="a2" joint="j2"/><motor name="renamed" joint="j1"/></actuator></mujoco>"""
    left, right = _pair(tmp_path / "named", left_xml, right_xml)
    aliases = _alias_json(
        left,
        right,
        actuator_pairs=(
            ActuatorAliasPair(
                "a1",
                ActuatorAliasEndpoint("NAMED", "a1", None, (), None, None),
                ActuatorAliasEndpoint("NAMED", "renamed", None, (), None, None),
            ),
        ),
    )
    result = build_model_pair_identity(left, "model.xml", right, "model.xml", aliases)
    assert result.alignment_summary.actuator_order == ("a1", "a2")

    unnamed_xml = _hinge_model(actuator_name=None)
    left, right = _pair(tmp_path / "unnamed-actuator", unnamed_xml)
    selector = ActuatorAliasEndpoint(
        "UNNAMED_SELECTOR",
        None,
        "JOINT",
        (TargetReference("JOINT", "joint"),),
        "NONE",
        0,
    )
    aliases = _alias_json(
        left,
        right,
        actuator_pairs=(ActuatorAliasPair("canonical", selector, selector),),
    )
    result = build_model_pair_identity(left, "model.xml", right, "model.xml", aliases)
    assert result.alignment_summary.actuator_order == ("canonical",)


def test_all_six_transmission_target_shapes(tmp_path: Path) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises all six transmission target shapes; incomplete, ambiguous, or
    unbound model components cannot support certification.
    """
    xml = """<mujoco><worldbody>
      <body name="body_target"><joint name="joint"/><geom size=".1" mass="1"/>
        <site name="crank" pos="0 0 0"/><site name="slider" pos="0.2 0 0"/>
        <site name="site" pos="0 0 0"/><site name="refsite" pos="0 0.1 0"/>
      </body>
    </worldbody>
    <tendon><fixed name="tendon"><joint joint="joint" coef="1"/></fixed></tendon>
    <actuator>
      <motor name="joint_act" joint="joint"/>
      <motor name="joint_parent_act" jointinparent="joint"/>
      <motor name="tendon_act" tendon="tendon"/>
      <motor name="slider_act" cranksite="crank" slidersite="slider" cranklength="0.2"/>
      <general name="site_act" site="site"/>
      <general name="site_ref_act" site="site" refsite="refsite"/>
      <general name="body_act" body="body_target"/>
    </actuator></mujoco>"""
    root = _write(tmp_path / "transmissions", xml)
    with closure.create_model_closure_snapshot(root, "model.xml", "baseline") as snap:
        model = admission.compile_snapshot_model(snap, "baseline")
        admission.admit_compiled_model(model, "baseline")
        compiled = admission.compile_model_identity(model, snap.identity.sha256(), "baseline")
    by_name = {item.name: item for item in compiled.actuators}
    assert by_name["joint_act"].transmission_type == "JOINT"
    assert by_name["joint_parent_act"].transmission_type == "JOINTINPARENT"
    assert by_name["tendon_act"].transmission_type == "TENDON"
    assert [item.object_type for item in by_name["slider_act"].targets] == ["SITE", "SITE"]
    assert len(by_name["site_act"].targets) == 1
    assert len(by_name["site_ref_act"].targets) == 2
    assert by_name["body_act"].targets[0].object_type == "BODY"


def test_builtin_activation_families(tmp_path: Path) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises builtin activation families; incomplete, ambiguous, or unbound model
    components cannot support certification.
    """
    xml = """<mujoco><worldbody><body><joint name="joint"/><geom size=".1" mass="1"/></body></worldbody>
    <actuator>
      <motor name="none" joint="joint"/>
      <general name="integrator" joint="joint" dyntype="integrator"/>
      <general name="filter" joint="joint" dyntype="filter" dynprm="0.1"/>
      <general name="filterexact" joint="joint" dyntype="filterexact" dynprm="0.1"/>
      <muscle name="muscle" joint="joint" lengthrange="0.01 1"/>
      <general name="dcmotor" joint="joint" dyntype="dcmotor" actearly="true"/>
    </actuator></mujoco>"""
    root = _write(tmp_path / "activation", xml)
    with closure.create_model_closure_snapshot(root, "model.xml", "baseline") as snap:
        model = admission.compile_snapshot_model(snap, "baseline")
        admission.admit_compiled_model(model, "baseline")
        compiled = admission.compile_model_identity(model, snap.identity.sha256(), "baseline")
    by_name = {item.name: item for item in compiled.actuators}
    assert {item.activation_family for item in compiled.actuators} == {
        "NONE",
        "INTEGRATOR",
        "FILTER",
        "FILTEREXACT",
        "MUSCLE",
        "DCMOTOR",
    }
    assert by_name["none"].activation_width == 0
    assert by_name["integrator"].activation_width == 1


def test_nested_include_and_file_backed_asset_compile_from_snapshot(tmp_path: Path) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises nested include and file backed asset compile from snapshot;
    incomplete, ambiguous, or unbound model components cannot support certification.
    """
    root = tmp_path / "asset-model"
    (root / "parts").mkdir(parents=True)
    (root / "model.xml").write_text(
        """<mujoco><asset><mesh name="tetra" file="parts/tetra.obj"/></asset>
        <include file="parts/body.xml"/></mujoco>""",
        encoding="utf-8",
    )
    (root / "parts" / "body.xml").write_text(
        """<mujoco><worldbody><body><joint name="joint"/>
        <geom type="mesh" mesh="tetra" mass="1"/></body></worldbody></mujoco>""",
        encoding="utf-8",
    )
    (root / "parts" / "tetra.obj").write_text(
        """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 2 3
f 1 2 4
f 1 3 4
f 2 3 4
""",
        encoding="ascii",
    )
    root = root.resolve()
    result = build_model_pair_identity(root, "model.xml", root, "model.xml")
    assert result.baseline_closure.member_count == 3
    assert [item.path for item in result.baseline_closure.members] == [
        "model.xml",
        "parts/body.xml",
        "parts/tetra.obj",
    ]
