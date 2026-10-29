"""Collect MuJoCo model identity scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

import hashlib as _hashlib
import json
import os
import shutil as _shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import mujoco
import numpy as np
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


def _file_backed_asset_root(root: Path) -> Path:
    """Write the existing nested-include tetrahedral OBJ fixture under one owned root."""
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
    return root.resolve()


def test_nested_include_and_file_backed_asset_compile_from_snapshot(tmp_path: Path) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises nested include and file backed asset compile from snapshot;
    incomplete, ambiguous, or unbound model components cannot support certification.
    """
    root = _file_backed_asset_root(tmp_path / "asset-model")
    result = build_model_pair_identity(root, "model.xml", root, "model.xml")
    assert result.baseline_closure.member_count == 3
    assert [item.path for item in result.baseline_closure.members] == [
        "model.xml",
        "parts/body.xml",
        "parts/tetra.obj",
    ]


def _complete_mjb(model: mujoco.MjModel) -> bytes:
    """Serialize every byte of a live model without field filtering."""
    buffer = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return buffer.tobytes()


def test_file_backed_snapshots_preserve_exact_bytes_across_roles_and_relocation(
    tmp_path: Path,
) -> None:
    """Temporary snapshot paths must not turn the same measured inputs into different MJBs."""
    root = _file_backed_asset_root(tmp_path / "source")
    relocated = tmp_path / "deeper" / "copy"
    _shutil.copytree(root, relocated)
    models = [
        _compile(root, "baseline"),
        _compile(root, "candidate"),
        _compile(relocated, "candidate"),
    ]
    original = _complete_mjb(models[0])
    assert all(_complete_mjb(model) == original for model in models)
    assert all(model.paths == b"parts/tetra.obj\x00" for model in models)
    body = root / "parts" / "body.xml"
    body.write_text(body.read_text().replace('mass="1"', 'mass="1.01"'), encoding="utf-8")
    assert _complete_mjb(_compile(root, "candidate")) != original
    retained = tmp_path / "retained.mjb"
    retained.write_bytes(original)
    root.rename(tmp_path / "source-unavailable")
    relocated.rename(tmp_path / "copy-unavailable")
    assert _complete_mjb(mujoco.MjModel.from_binary_path(str(retained))) == original


def test_nested_entrypoint_preserves_distinct_same_basename_assets(tmp_path: Path) -> None:
    """Compile exact closure-relative names without flattening or basename substitution."""
    root = _file_backed_asset_root(tmp_path / "source")
    for name, apex in (("left", "1"), ("right", "1.4")):
        asset = root / "parts" / name / "tetra.obj"
        asset.parent.mkdir()
        asset.write_text(
            (root / "parts" / "tetra.obj").read_text().replace("v 0 0 1\n", f"v 0 0 {apex}\n"),
            encoding="ascii",
        )
    (root / "parts" / "assets.xml").write_text(
        '<mujocoinclude><asset><mesh name="left" file="left/tetra.obj"/>'
        '<mesh name="right" file="right/tetra.obj"/></asset></mujocoinclude>',
        encoding="utf-8",
    )
    entrypoint = root / "models" / "model.xml"
    entrypoint.parent.mkdir()
    entrypoint.write_text(
        '<mujoco><compiler meshdir="../parts"/><include file="../parts/assets.xml"/>'
        '<worldbody><body><joint name="joint" armature=".01"/>'
        '<geom type="mesh" mesh="left" mass="1"/>'
        '<geom type="mesh" mesh="right" mass="1" pos="0 0 2"/>'
        "</body></worldbody></mujoco>",
        encoding="utf-8",
    )
    native = mujoco.MjModel.from_xml_path(str(entrypoint))
    with closure.create_model_closure_snapshot(root, "models/model.xml", "baseline") as snapshot:
        model = admission.compile_snapshot_model(snapshot, "baseline")
    assert model.nmesh == 2
    assert model.mesh_vertnum.tolist() == [4, 4]
    np.testing.assert_array_equal(model.mesh_vert, native.mesh_vert)
    assert b"left/tetra.obj\x00" in model.paths
    assert b"right/tetra.obj\x00" in model.paths
    assert str(tmp_path).encode() not in model.paths


def _texture_model_root(root: Path, *, explicit: bool) -> Path:
    """Write a valid two-by-two RGB texture and a by-reference texture declaration."""
    root.mkdir()

    def chunk(kind: bytes, content: bytes) -> bytes:
        return (
            struct.pack(">I", len(content))
            + kind
            + content
            + struct.pack(">I", zlib.crc32(kind + content))
        )

    rows = b"\x00" + bytes((255, 0, 0, 0, 255, 0))
    rows += b"\x00" + bytes((0, 0, 255, 255, 255, 255))
    (root / "skin.png").write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    compiler = "" if explicit else '<compiler passthrough="textures"/>'
    embed = ' embed="false"' if explicit else ""
    (root / "model.xml").write_text(
        f'<mujoco>{compiler}<asset><texture name="t" type="2d" file="skin.png"{embed}/>'
        '<material name="mat" texture="t"/></asset><worldbody>'
        '<geom type="sphere" size=".1" material="mat"/></worldbody></mujoco>',
        encoding="utf-8",
    )
    return root


def _assert_texture_syntax_refused(root: Path, attribute: str) -> None:
    """Check native syntax rejection and its typed product refusal on retained runtimes."""
    message = f"unrecognized attribute: '{attribute}'"
    with pytest.raises(ValueError, match=message):
        mujoco.MjModel.from_xml_path(str(root / "model.xml"))
    with pytest.raises(closure.ModelAdmissionRefusal) as caught:
        _compile(root, "baseline")
    assert caught.value.reason is OperationalReasonCode.BASELINE_MODEL_COMPILE_ERROR
    assert message in caught.value.evidence["message"]


@pytest.mark.parametrize("explicit", [False, True])
def test_by_reference_texture_cannot_escape_complete_artifact_admission(
    tmp_path: Path, explicit: bool
) -> None:
    """A real external texture is refused; compiling alone cannot prove a self-contained MJB."""
    root = _texture_model_root(tmp_path / "source", explicit=explicit)
    if mujoco.mj_version() < 3_017_000:
        _assert_texture_syntax_refused(root, "embed" if explicit else "passthrough")
        return
    native = mujoco.MjModel.from_xml_path(str(root / "model.xml"))
    assert native.tex_adr.tolist() == [-1]
    assert native.ntexdata == 0
    with pytest.raises(closure.ModelAdmissionRefusal) as caught:
        admission.admit_external_implementation_free_model(native, "baseline")
    assert caught.value.reason is OperationalReasonCode.MUJOCO_FEATURE_COVERAGE_INCOMPLETE
    assert caught.value.evidence["feature"] == "external_texture_assets"
    assert caught.value.evidence["external_texture_indices"] == (0,)


def test_explicit_by_reference_texture_refuses_without_publication(tmp_path: Path) -> None:
    """Explicit external storage cannot become a successful complete-artifact certificate."""
    from metrifid.certify import CertifyOperationError, certify_models

    root = _texture_model_root(tmp_path / "source", explicit=True)
    output = tmp_path / "output"
    with pytest.raises(CertifyOperationError) as caught:
        certify_models(str(root / "model.xml"), str(root / "model.xml"), str(output))
    assert caught.value.failure.reason.code is OperationalReasonCode.BASELINE_MODEL_COMPILE_ERROR
    expected_message = (
        "unrecognized attribute: 'embed'"
        if mujoco.mj_version() < 3_017_000
        else "texture cannot be by reference"
    )
    assert expected_message in caught.value.failure.reason.evidence["message"]
    assert not (output / "certification.json").exists()


def test_texture_passthrough_retains_pixels_after_source_disappears(tmp_path: Path) -> None:
    """A VFS-served texture remains fully embedded through serialization and source removal."""
    root = _texture_model_root(tmp_path / "source", explicit=False)
    if mujoco.mj_version() < 3_017_000:
        _assert_texture_syntax_refused(root, "passthrough")
        return
    model = _compile(root, "baseline")
    admission.admit_external_implementation_free_model(model, "baseline")
    assert model.tex_adr.tolist() == [0]
    assert model.ntexdata == 12
    pixels = model.tex_data.tobytes()
    complete = _complete_mjb(model)
    retained = tmp_path / "retained.mjb"
    retained.write_bytes(complete)
    root.rename(tmp_path / "source-unavailable")
    reloaded = mujoco.MjModel.from_binary_path(str(retained))
    assert reloaded.tex_data.tobytes() == pixels
    assert _complete_mjb(reloaded) == complete


@pytest.mark.parametrize("operation", ["certify", "diff"])
def test_file_backed_installed_commands_distinguish_relocation_from_physical_change(
    tmp_path: Path, operation: str
) -> None:
    """Exercise actual command dispatch, retained results, stable repeats and a physical control."""
    source = _file_backed_asset_root(tmp_path / "source")
    relocated = tmp_path / "deeper" / "relocated"
    _shutil.copytree(source, relocated)
    physical = tmp_path / "physical"
    _shutil.copytree(source, physical)
    body = physical / "parts" / "body.xml"
    body.write_text(body.read_text().replace('mass="1"', 'mass="1.01"'), encoding="utf-8")
    receipts = []
    filename = "certification.json" if operation == "certify" else "model_release.json"
    for label, candidate, expected in (
        ("same", source, 0),
        ("repeat", source, 0),
        ("relocated", relocated, 0),
        ("physical", physical, 40),
    ):
        output = tmp_path / ("out-" + label)
        command = [
            sys.executable,
            "-m",
            "metrifid.cli",
            operation,
            str(source / "model.xml"),
            str(candidate / "model.xml"),
            "--output",
            str(output),
        ]
        if operation == "diff":
            command.append("--json")
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        assert completed.returncode == expected, completed.stderr
        document = json.loads(completed.stdout)
        assert isinstance(document, dict)
        receipt = (output / filename).read_bytes()
        assert str(tmp_path).encode() not in receipt
        receipts.append(receipt)
    assert receipts[0] == receipts[1]
