"""Collect MuJoCo model identity native scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

import hashlib as _hashlib
import json
import os
import shutil as _shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import mujoco
import pytest

from metrifid import _model_admission as admission
from metrifid import _model_closure as closure
from metrifid import _model_compile as model_compile
from metrifid import _model_identity as identity
from metrifid._model_dependencies import discover_snapshot_dependencies as _discover
from metrifid.json_values import canonical_json_bytes
from metrifid.operational import OperationalReasonCode
from metrifid.schemas import (
    ActuatorAliasPair,
    AliasArtifact,
    JointAliasPair,
)


def _write(root: Path, xml: str, *, name: str = "model.xml") -> Path:
    """Write write data into the isolated test workspace.

    The MuJoCo model identity native scenario observes real bytes and filesystem effects for
    MuJoCo model identity native.
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
    """Construct the hinge model fixture used by MuJoCo model identity native scenarios.

    Deterministic setup isolates MuJoCo model identity native without bypassing the contract
    boundary under assertion.
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
    """Construct the pair fixture used by MuJoCo model identity native scenarios.

    Deterministic setup isolates MuJoCo model identity native without bypassing the contract
    boundary under assertion.
    """
    baseline = _write(tmp_path / "baseline", baseline_xml)
    candidate = _write(tmp_path / "candidate", candidate_xml or baseline_xml)
    return baseline, candidate


def _refusal_reason(
    exc: pytest.ExceptionInfo[closure.ModelAdmissionRefusal],
) -> OperationalReasonCode:
    """Extract the reason code from a native-model admission refusal."""
    return exc.value.reason


def _compile(root: Path, role: closure.ModelRole) -> mujoco.MjModel:
    """Construct the compile fixture used by MuJoCo model identity native scenarios.

    Deterministic setup isolates MuJoCo model identity native without bypassing the contract
    boundary under assertion.
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
    """Construct the alias json fixture used by MuJoCo model identity native scenarios.

    Deterministic setup isolates MuJoCo model identity native without bypassing the contract
    boundary under assertion.
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
    """Compute the canonical tree digest value used by MuJoCo model identity native fixtures.

    Content addressing keeps the mutation boundary explicit for MuJoCo model identity native.
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


def test_source_mutation_during_compilation_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises source mutation during compilation refuses; incomplete, ambiguous,
    or unbound model components cannot support certification.
    """
    left, right = _pair(tmp_path, _hinge_model())
    original = identity.compile_snapshot_model
    calls = 0

    def mutate(snapshot: closure.ModelClosureSnapshot, role: closure.ModelRole) -> mujoco.MjModel:
        """Apply the targeted mutate mutation to otherwise valid evidence.

        Resealing the result isolates the semantic contradiction exercised by source mutation
        during compilation refuses.
        """
        nonlocal calls
        model = original(snapshot, role)
        calls += 1
        if calls == 1:
            (left / "late.txt").write_text("mutation", encoding="utf-8")
        return model

    monkeypatch.setattr(identity, "compile_snapshot_model", mutate)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity.build_model_pair_identity(left, "model.xml", right, "model.xml")
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_CLOSURE_MUTATED


def test_compile_error_warning_and_callback_restoration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises compile error warning and callback restoration; incomplete,
    ambiguous, or unbound model components cannot support certification.
    """
    invalid = _write(tmp_path / "invalid", "<not-mujoco/>")
    assert mujoco.get_mju_user_warning() is None
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        _compile(invalid, "baseline")
    assert _refusal_reason(exc) is OperationalReasonCode.BASELINE_MODEL_COMPILE_ERROR
    assert mujoco.get_mju_user_warning() is None

    valid = _write(tmp_path / "valid", _hinge_model())
    real_class = model_compile.mujoco.MjModel
    real_from_path = real_class.from_xml_path

    def warned(path: str) -> mujoco.MjModel:
        """Construct the warned fixture used by MuJoCo model identity native scenarios.

        Deterministic setup isolates compile error warning and callback restoration without
        bypassing the contract boundary under assertion.
        """
        callback = mujoco.get_mju_user_warning()
        assert callback is not None
        callback("captured warning")
        return real_from_path(path)

    monkeypatch.setattr(model_compile.mujoco, "MjModel", SimpleNamespace(from_xml_path=warned))
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        _compile(valid, "candidate")
    assert _refusal_reason(exc) is OperationalReasonCode.CANDIDATE_MODEL_COMPILE_WARNING
    assert mujoco.get_mju_user_warning() is None

    def failed(path: str) -> mujoco.MjModel:
        """Inject the deterministic failed branch required by this scenario.

        The MuJoCo model identity native test can assert failure delivery for compile error
        warning and callback restoration without depending on incidental runtime errors.
        """
        raise ValueError(f"failed {path}")

    monkeypatch.setattr(model_compile.mujoco, "MjModel", SimpleNamespace(from_xml_path=failed))
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        _compile(valid, "candidate")
    assert _refusal_reason(exc) is OperationalReasonCode.CANDIDATE_MODEL_COMPILE_ERROR
    assert mujoco.get_mju_user_warning() is None
    monkeypatch.setattr(model_compile.mujoco, "MjModel", real_class)
    assert _compile(valid, "baseline").nq == 1
    assert mujoco.get_mju_user_warning() is None


def test_preexisting_callback_refuses_and_is_not_overwritten(tmp_path: Path) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises preexisting callback refuses and is not overwritten; incomplete,
    ambiguous, or unbound model components cannot support certification.
    """
    root = _write(tmp_path / "callbacks", _hinge_model())

    def callback(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Construct the callback fixture used by MuJoCo model identity native scenarios.

        Deterministic setup isolates preexisting callback refuses and is not overwritten without
        bypassing the contract boundary under assertion.
        """
        del model, data

    mujoco.set_mjcb_control(callback)
    try:
        with pytest.raises(closure.ModelAdmissionRefusal) as exc:
            _compile(root, "baseline")
        assert _refusal_reason(exc) is OperationalReasonCode.UNSUPPORTED_USER_CALLBACK
        assert mujoco.get_mjcb_control() is callback
    finally:
        mujoco.set_mjcb_control(None)


@pytest.mark.parametrize(
    ("case", "xml", "reason"),
    [
        (
            "history",
            """<mujoco><worldbody><body><joint name="joint"/><geom size=".1" mass="1"/></body></worldbody>
            <sensor><jointpos name="s" joint="joint" nsample="2" delay="1"/></sensor></mujoco>""",
            OperationalReasonCode.UNSUPPORTED_HISTORY_STATE,
        ),
        (
            "mocap",
            """<mujoco><worldbody><body name="m" mocap="true"><geom size=".1"/></body></worldbody></mujoco>""",
            OperationalReasonCode.UNSUPPORTED_MOCAP_STATE,
        ),
        (
            "user-actuator",
            """<mujoco><worldbody><body><joint name="joint"/><geom size=".1" mass="1"/></body></worldbody>
            <actuator><general name="a" joint="joint" dyntype="user"/></actuator></mujoco>""",
            OperationalReasonCode.UNSUPPORTED_USER_CALLBACK,
        ),
        (
            "user-sensor",
            """<mujoco><worldbody><body><joint name="joint"/><geom size=".1" mass="1"/></body></worldbody>
            <sensor><user name="s" dim="1" needstage="pos"/></sensor></mujoco>""",
            OperationalReasonCode.UNSUPPORTED_USER_CALLBACK,
        ),
        (
            "plugin-sensor",
            """<mujoco><extension><plugin plugin="mujoco.sensor.touch_grid"/></extension>
            <worldbody><body><geom name="g" size=".1"/><site name="s" size=".1"/></body></worldbody>
            <sensor><plugin name="p" plugin="mujoco.sensor.touch_grid" objtype="site" objname="s">
            <config key="size" value="4 4"/><config key="fov" value="90 90"/>
            </plugin></sensor></mujoco>""",
            OperationalReasonCode.UNSUPPORTED_PLUGIN_STATE,
        ),
    ],
)
def test_unsupported_compiled_state_refusals(
    tmp_path: Path,
    case: str,
    xml: str,
    reason: OperationalReasonCode,
) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises unsupported compiled state refusals; incomplete, ambiguous, or
    unbound model components cannot support certification.
    """
    root = _write(tmp_path / case, xml)
    model = _compile(root, "baseline")
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        admission.admit_compiled_model(model, "baseline")
    assert _refusal_reason(exc) is reason


def test_plugin_reference_refuses(tmp_path: Path) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises plugin reference refuses; incomplete, ambiguous, or unbound model
    components cannot support certification.
    """
    xml = """<mujoco><extension><plugin plugin="mujoco.elasticity.cable"/></extension>
    <worldbody><body name="cable"><composite type="cable" count="4 1 1" curve="s">
    <plugin plugin="mujoco.elasticity.cable"/></composite></body></worldbody></mujoco>"""
    root = _write(tmp_path / "plugin", xml)
    try:
        model = _compile(root, "baseline")
    except closure.ModelAdmissionRefusal as error:
        # Plugin availability is installation-dependent; a faithful compile error remains explicit.
        if error.reason is not OperationalReasonCode.BASELINE_MODEL_COMPILE_ERROR:
            pytest.fail(f"unexpected plugin compile refusal: {error.reason.value}")
        return
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        admission.admit_compiled_model(model, "baseline")
    assert _refusal_reason(exc) is OperationalReasonCode.UNSUPPORTED_PLUGIN_STATE


def test_canonical_hash_is_stable_across_fresh_processes(tmp_path: Path) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises canonical hash is stable across fresh processes; incomplete,
    ambiguous, or unbound model components cannot support certification.
    """
    left, right = _pair(tmp_path, _hinge_model())
    code = """from pathlib import Path
from metrifid._model_identity import build_model_pair_identity
import sys
value = build_model_pair_identity(Path(sys.argv[1]), 'model.xml', Path(sys.argv[2]), 'model.xml')
print(value.model_pair_identity_sha256)
"""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    hashes = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", code, str(left), str(right)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        hashes.append(result.stdout.strip())
    assert hashes[0] == hashes[1]
    assert len(hashes[0]) == 64


def test_alias_json_is_strict_duplicate_key_refusal(tmp_path: Path) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises alias json is strict duplicate key refusal; incomplete, ambiguous,
    or unbound model components cannot support certification.
    """
    left, right = _pair(tmp_path, _hinge_model())
    raw = json.dumps(
        {
            "schema": "metrifid.aliases",
            "schema_version": 1,
            "baseline_model_closure_sha256": "a" * 64,
            "candidate_model_closure_sha256": "b" * 64,
            "joint_pairs": [],
            "actuator_pairs": [],
        }
    )
    duplicated = raw.replace('"schema_version":', '"schema_version":"x","schema_version":', 1)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity.build_model_pair_identity(left, "model.xml", right, "model.xml", duplicated)
    assert _refusal_reason(exc) is OperationalReasonCode.ALIAS_SCHEMA_INVALID


def test_mujoco_compiles_official_scene(g1_copy: Path) -> None:
    """18. MuJoCo 3.10.0 compiles the official scene.xml."""
    assert mujoco.__version__ == "3.10.0"
    model = mujoco.MjModel.from_xml_path(str(g1_copy / "scene.xml"))
    assert int(model.nu) == 29
    assert int(model.nhfield) >= 1


def test_same_role_model_pair_identity_for_scene(g1_copy: Path) -> None:
    """19. metrifid builds a same-role model pair identity for the official scene.xml."""
    result = identity.build_model_pair_identity(g1_copy, "scene.xml", g1_copy, "scene.xml")
    assert len(result.alignment.actuators) == 29
    for actuator in result.alignment.actuators:
        assert actuator.baseline_control_address == actuator.candidate_control_address


def test_dependencies_contain_both_hfield_pngs(g1_copy: Path) -> None:
    """20. discovered dependencies contain height_field.png and unitree_hfield.png."""
    with closure.create_model_closure_snapshot(g1_copy, "scene.xml", "baseline") as snapshot:
        dependencies = _discover(snapshot, "baseline")
    names = {PurePosixPath(dependency).name for dependency in dependencies}
    assert "height_field.png" in names
    assert "unitree_hfield.png" in names


def test_every_dependency_is_a_contained_measured_member(g1_copy: Path) -> None:
    """21. every discovered dependency is a contained measured closure member."""
    with closure.create_model_closure_snapshot(g1_copy, "scene.xml", "baseline") as snapshot:
        dependencies = _discover(snapshot, "baseline")
        members = {member.path for member in snapshot.identity.members}
        root = snapshot.snapshot_root.resolve()
        assert dependencies
        for dependency in dependencies:
            assert dependency in members
            resolved = snapshot.snapshot_root.joinpath(*PurePosixPath(dependency).parts).resolve()
            assert root in resolved.parents
            assert resolved.is_file()
            assert not resolved.is_symlink()


def test_source_model_bytes_unchanged(g1_copy: Path) -> None:
    """22. source model bytes are unchanged before and after admission and comparison."""
    before = _tree_digest(g1_copy)
    with closure.create_model_closure_snapshot(g1_copy, "scene.xml", "baseline") as snapshot:
        _discover(snapshot, "baseline")
        admission.compile_snapshot_model(snapshot, "baseline")
    identity.build_model_pair_identity(g1_copy, "scene.xml", g1_copy, "scene.xml")
    assert _tree_digest(g1_copy) == before


def test_existing_scene_29dof_path_still_works(g1_copy: Path) -> None:
    """23. the existing scene_29dof.xml path still works."""
    before = _tree_digest(g1_copy)
    with closure.create_model_closure_snapshot(g1_copy, "scene_29dof.xml", "baseline") as snapshot:
        model = admission.compile_snapshot_model(snapshot, "baseline")
        dependencies = _discover(snapshot, "baseline")
        members = {member.path for member in snapshot.identity.members}
    assert int(model.nu) == 29
    assert dependencies
    for dependency in dependencies:
        assert dependency in members
    result = identity.build_model_pair_identity(
        g1_copy, "scene_29dof.xml", g1_copy, "scene_29dof.xml"
    )
    assert len(result.alignment.actuators) == 29
    assert _tree_digest(g1_copy) == before


def test_official_scene_compiles_natively_and_is_admitted(g1_copy: Path) -> None:
    """MuJoCo 3.10.0 compiles the official scene.xml and metrifid admits the same bytes."""
    assert mujoco.__version__ == "3.10.0"
    assert mujoco.mj_versionString() == "3.10.0"
    native = mujoco.MjModel.from_xml_path(str(g1_copy / "scene.xml"))
    with closure.create_model_closure_snapshot(g1_copy, "scene.xml", "baseline") as snapshot:
        admitted = admission.compile_snapshot_model(snapshot, "baseline")
    assert int(native.nu) == int(admitted.nu) == 29
    assert int(native.nq) == int(admitted.nq)
    assert int(native.nhfield) == int(admitted.nhfield) >= 2


def test_unrelated_checkout_xml_does_not_affect_the_dependency_result(
    g1_copy: Path,
) -> None:
    """An unrelated XML added to the model root must not change the dependency set."""
    with closure.create_model_closure_snapshot(g1_copy, "scene.xml", "baseline") as snapshot:
        before = _discover(snapshot, "baseline")

    (g1_copy / "unrelated_dependency discovery.xml").write_text(
        '<mujoco><compiler meshdir="nowhere" texturedir="nowhere" strippath="true"/>'
        '<asset><mesh name="ghost" file="ghost.STL"/>'
        '<hfield name="ghost_h" size="1 1 1 1" file="ghost.png"/></asset></mujoco>',
        encoding="utf-8",
    )
    with closure.create_model_closure_snapshot(g1_copy, "scene.xml", "baseline") as snapshot:
        after = _discover(snapshot, "baseline")
        members = {member.path for member in snapshot.identity.members}

    # The unrelated file is measured as part of the full-root snapshot, and ignored by resolution.
    assert "unrelated_dependency discovery.xml" in members
    assert after == before


def test_scene_29dof_remains_admitted_and_unaffected(g1_copy: Path) -> None:
    """scene_29dof.xml keeps its accepted behaviour."""
    with closure.create_model_closure_snapshot(g1_copy, "scene_29dof.xml", "baseline") as snapshot:
        model = admission.compile_snapshot_model(snapshot, "baseline")
        dependencies = _discover(snapshot, "baseline")
        members = {member.path for member in snapshot.identity.members}
    assert int(model.nu) == 29
    assert dependencies
    assert all(dependency in members for dependency in dependencies)
