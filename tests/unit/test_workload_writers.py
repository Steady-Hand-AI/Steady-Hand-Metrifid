"""Canonical comparison workload-writer tests using the accepted strict NPZ loaders."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import metrifid._owned_artifacts as owned
from metrifid import workload_writers as writers
from metrifid import write_actions_artifact, write_state_artifact
from metrifid._model_closure import AlignedActuator, AlignedJoint
from metrifid.schemas import TargetReference


def _isolate_modules(monkeypatch: pytest.MonkeyPatch, names: Sequence[str]) -> None:
    """Remove each module and its parent-package attribute, restoring both exactly at teardown.

    Two things have to be undone, not one. ``importlib.import_module`` installs the module in
    ``sys.modules`` *and* sets it as an attribute on its parent package, so restoring
    ``sys.modules`` alone leaves the parent attribute pointing at the stub-loaded module for the
    rest of the session. Both are recorded here through ``monkeypatch``.

    Each pair is deliberately ``setitem``/``setattr`` followed by ``delitem``/``delattr``. The
    first call records the prior value *or its absence*; the second performs the removal. A bare
    ``delitem(..., raising=False)`` records nothing when the key was already absent, so a module
    created later by the re-import would survive teardown. Nothing here removes an entry
    without first recording what was there.
    """
    for name in names:
        parent_name, _, attribute = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            monkeypatch.setattr(parent, attribute, getattr(parent, attribute, None), raising=False)
            monkeypatch.delattr(parent, attribute, raising=False)
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
        monkeypatch.delitem(sys.modules, name, raising=False)


def _fake_mujoco_module() -> types.ModuleType:
    """Construct the fake MuJoCo module fixture used by workload writers scenarios.

    Deterministic setup isolates workload writers without bypassing the contract boundary under
    assertion.
    """
    module = types.ModuleType("mujoco")
    module.MjModel = type("MjModel", (), {})
    module.MjData = type("MjData", (), {})
    enum_type = type("EnumValues", (), {})
    module.mjtJoint = enum_type()
    for index, name in enumerate(("FREE", "BALL", "SLIDE", "HINGE")):
        setattr(module.mjtJoint, f"mjJNT_{name}", index)
    module.mjtDyn = enum_type()
    for index, name in enumerate(
        ("NONE", "INTEGRATOR", "FILTER", "FILTEREXACT", "MUSCLE", "DCMOTOR")
    ):
        setattr(module.mjtDyn, f"mjDYN_{name}", index)
    module.mjtTrn = enum_type()
    for index, name in enumerate(
        ("JOINT", "JOINTINPARENT", "SLIDERCRANK", "TENDON", "SITE", "BODY")
    ):
        setattr(module.mjtTrn, f"mjTRN_{name}", index)
    module.mjtObj = enum_type()
    for index, name in enumerate(("JOINT", "TENDON", "SITE", "BODY")):
        setattr(module.mjtObj, f"mjOBJ_{name}", index)
    for name in (
        "get_mjcb_control",
        "get_mjcb_sensor",
        "get_mjcb_passive",
        "get_mjcb_act_dyn",
        "get_mjcb_act_gain",
        "get_mjcb_act_bias",
        "get_mjcb_contactfilter",
        "get_mjcb_time",
        "get_mju_user_warning",
        "get_mju_user_malloc",
        "get_mju_user_free",
    ):
        setattr(module, name, lambda: None)
    return module


def _strict_workload_module(monkeypatch: pytest.MonkeyPatch):
    # Import once against the real dependency first. That leaves every transitively imported
    # project module already in sys.modules, so the fake-bound import below re-executes only the
    # three modules recorded here and cannot create a new module bound to the fake. Without this,
    # a module first imported during the fake-bound import - metrifid._model_dependencies, for
    # instance - would survive teardown holding the incomplete fake.
    """Construct the strict workload module fixture used by workload writers scenarios.

    Deterministic setup isolates workload writers without bypassing the contract boundary under
    assertion.
    """
    importlib.import_module("metrifid._workload")
    _isolate_modules(
        monkeypatch,
        (
            "metrifid._workload",
            "metrifid._model_identity",
            "metrifid._model_admission",
            "metrifid._model_compile",
            "metrifid._model_descriptors",
        ),
    )
    # The fake dependency is installed only for the import below. Every project module the import
    # binds to it was recorded above, so none survives teardown still bound to it.
    monkeypatch.setitem(sys.modules, "mujoco", _fake_mujoco_module())
    return importlib.import_module("metrifid._workload")


def _alignment() -> SimpleNamespace:
    """Construct the alignment fixture used by workload writers scenarios.

    Deterministic setup isolates workload writers without bypassing the contract boundary under
    assertion.
    """
    joints = (
        AlignedJoint("hinge", "HINGE", (0, 1), (0, 1), (0, 1), (0, 1)),
        AlignedJoint("slide", "SLIDE", (1, 1), (1, 1), (1, 1), (1, 1)),
    )
    actuators = (
        AlignedActuator(
            "hinge_motor",
            "JOINT",
            (TargetReference("JOINT", "hinge"),),
            "NONE",
            0,
            0,
            0,
            None,
            None,
        ),
        AlignedActuator(
            "slide_motor",
            "JOINT",
            (TargetReference("JOINT", "slide"),),
            "NONE",
            0,
            1,
            1,
            None,
            None,
        ),
    )
    return SimpleNamespace(joints=joints, actuators=actuators)


def _write_semantically_equal_pair(root: Path, *, reverse: bool) -> tuple[Path, Path]:
    """Write write semantically equal pair data into the isolated test workspace.

    The workload writers scenario observes real bytes and filesystem effects for workload
    writers.
    """
    state_path = root / ("state_reverse.npz" if reverse else "state.npz")
    actions_path = root / ("actions_reverse.npz" if reverse else "actions.npz")
    if reverse:
        write_state_artifact(
            state_path,
            joint_names=("slide", "hinge"),
            qpos_offsets=(0, 1, 2),
            qpos=np.asarray([2.0, 1.0]),
            qvel_offsets=(0, 1, 2),
            qvel=np.asarray([4.0, 3.0]),
            actuator_names=("slide_motor", "hinge_motor"),
            act_offsets=(0, 0, 0),
            act=np.empty((0,)),
        )
        write_actions_artifact(
            actions_path,
            actuator_names=("slide_motor", "hinge_motor"),
            values=np.asarray([[20.0, 10.0], [40.0, 30.0]]),
        )
    else:
        write_state_artifact(
            state_path,
            joint_names=("hinge", "slide"),
            qpos_offsets=(0, 1, 2),
            qpos=np.asarray([1.0, 2.0]),
            qvel_offsets=(0, 1, 2),
            qvel=np.asarray([3.0, 4.0]),
            actuator_names=("hinge_motor", "slide_motor"),
            act_offsets=(0, 0, 0),
            act=np.empty((0,)),
        )
        write_actions_artifact(
            actions_path,
            actuator_names=("hinge_motor", "slide_motor"),
            values=np.asarray([[10.0, 20.0], [30.0, 40.0]]),
        )
    return state_path, actions_path


def test_writers_are_byte_deterministic_and_round_trip_through_strict_loaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises writers are byte deterministic and round trip through strict
    loaders; malformed arrays, names, or dimensions must fail before comparison evidence is
    produced.
    """
    workload = _strict_workload_module(monkeypatch)
    alignment = _alignment()
    monkeypatch.setattr(workload, "_completed_alignment", lambda _pair: alignment)
    normal_state, normal_actions = _write_semantically_equal_pair(tmp_path, reverse=False)
    reverse_state, reverse_actions = _write_semantically_equal_pair(tmp_path, reverse=True)

    assert normal_state.read_bytes() == reverse_state.read_bytes()
    assert normal_actions.read_bytes() == reverse_actions.read_bytes()
    assert hashlib.sha256(normal_state.read_bytes()).hexdigest() == (
        "c68f7c47ecbeee9e2c5e968d946435d3d545ccdf4fe8b86b9a9b4f0ca82df3d5"
    )
    assert hashlib.sha256(normal_actions.read_bytes()).hexdigest() == (
        "50fa33cab29cae6dedca0c04551ab18dab60773ee8e31bf276b33a1304266003"
    )

    loaded_state = workload.load_state_artifact(normal_state, object())
    loaded_actions = workload.load_actions_artifact(normal_actions, object())
    assert loaded_state.metadata.joint_names == ("hinge", "slide")
    assert loaded_state.metadata.actuator_names == ("hinge_motor", "slide_motor")
    assert loaded_state.qpos.tolist() == [1.0, 2.0]
    assert loaded_state.qvel.tolist() == [3.0, 4.0]
    assert loaded_actions.metadata.actuator_names == ("hinge_motor", "slide_motor")
    assert loaded_actions.values.tolist() == [[10.0, 20.0], [30.0, 40.0]]
    assert not loaded_state.qpos.flags.writeable
    assert not loaded_actions.values.flags.writeable


def test_writers_refuse_duplicates_nonfinite_values_and_symlink_targets(tmp_path: Path) -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises writers refuse duplicates nonfinite values and symlink targets;
    malformed arrays, names, or dimensions must fail before comparison evidence is produced.
    """
    with pytest.raises(ValueError, match="duplicate"):
        write_actions_artifact(
            tmp_path / "duplicate.npz",
            actuator_names=("motor", "motor"),
            values=np.zeros((1, 2)),
        )
    with pytest.raises(ValueError, match="finite"):
        write_actions_artifact(
            tmp_path / "nonfinite.npz",
            actuator_names=("motor",),
            values=np.asarray([[np.nan]]),
        )
    target = tmp_path / "real.npz"
    target.write_bytes(b"existing")
    link = tmp_path / "link.npz"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="absent"):
        write_actions_artifact(
            link,
            actuator_names=("motor",),
            values=np.asarray([[0.0]]),
        )


def test_failed_writer_link_preserves_injected_hardlink_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve an injected hardlink destination when the writer gets EEXIST."""
    target = tmp_path / "actions.npz"
    real_link = owned.os.link
    injected = False

    def inject_hardlink(source: object, name: object, **kwargs: object) -> None:
        """Inject the destination immediately before the writer's no-clobber link."""
        nonlocal injected
        if not injected:
            injected = True
            real_link(source, name, **kwargs)  # type: ignore[arg-type]
        real_link(source, name, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(owned.os, "link", inject_hardlink)
    with pytest.raises(ValueError, match="absent"):
        write_actions_artifact(target, actuator_names=("motor",), values=np.zeros((1, 1)))
    assert target.exists()
    assert target.stat().st_size > 0


def test_workload_writer_does_not_follow_replaced_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep an artifact commit confined to the parent bound before serialization."""
    public = tmp_path / "artifacts"
    admitted = tmp_path / "admitted"
    outside = tmp_path / "outside"
    public.mkdir()
    outside.mkdir()
    real_open_temp = writers._open_writer_temp

    def replace_parent(parent_fd: int, target_name: str) -> writers.OwnedArtifact:
        """Replace the public parent immediately before descriptor-relative temp creation."""
        public.rename(admitted)
        public.symlink_to(outside, target_is_directory=True)
        return real_open_temp(parent_fd, target_name)

    monkeypatch.setattr(writers, "_open_writer_temp", replace_parent)
    with pytest.raises(ValueError, match="parent changed"):
        write_actions_artifact(
            public / "actions.npz",
            actuator_names=("motor",),
            values=np.asarray([[0.0]]),
        )
    assert list(admitted.iterdir()) == []
    assert list(outside.iterdir()) == []


# Isolation mechanism.
#
# The fake mujoco module here is deliberately incomplete. If any metrifid module survives the
# fixture still bound to it, every later test that compiles or certifies a model fails with an
# unrelated AttributeError or an UNSUPPORTED_MUJOCO_VERSION refusal. These prove it does not.


def test_isolation_restores_pre_existing_project_modules_and_attributes() -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises isolation restores pre existing project modules and attributes;
    malformed arrays, names, or dimensions must fail before comparison evidence is produced.
    """
    import metrifid as package
    import metrifid._model_compile as model_compile

    original_module = sys.modules["metrifid._model_compile"]
    original_attribute = package._model_compile
    with pytest.MonkeyPatch.context() as patch:
        _isolate_modules(patch, ("metrifid._model_compile",))
        assert "metrifid._model_compile" not in sys.modules
        assert not hasattr(package, "_model_compile")
        sys.modules["metrifid._model_compile"] = types.ModuleType("stub")
        package._model_compile = types.ModuleType("stub")
    assert sys.modules["metrifid._model_compile"] is original_module is model_compile
    assert package._model_compile is original_attribute


def test_isolation_leaves_an_originally_absent_project_module_absent() -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises isolation leaves an originally absent project module absent;
    malformed arrays, names, or dimensions must fail before comparison evidence is produced.
    """
    name = "metrifid._absent_for_this_test"
    assert name not in sys.modules
    with pytest.MonkeyPatch.context() as patch:
        _isolate_modules(patch, (name,))
        sys.modules[name] = types.ModuleType(name)
    assert name not in sys.modules


def test_isolation_leaves_an_originally_absent_project_attribute_absent() -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises isolation leaves an originally absent project attribute absent;
    malformed arrays, names, or dimensions must fail before comparison evidence is produced.
    """
    import metrifid as package

    assert not hasattr(package, "_absent_for_this_test")
    with pytest.MonkeyPatch.context() as patch:
        _isolate_modules(patch, ("metrifid._absent_for_this_test",))
        package._absent_for_this_test = types.ModuleType("stub")
    assert not hasattr(package, "_absent_for_this_test")


def test_no_project_module_survives_the_fixture_bound_to_the_fake_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises no project module survives the fixture bound to the fake dependency;
    malformed arrays, names, or dimensions must fail before comparison evidence is produced.
    """
    real = sys.modules["mujoco"]
    before = {name: module for name, module in sys.modules.items() if name.startswith("metrifid")}

    workload = _strict_workload_module(monkeypatch)
    fake = sys.modules["mujoco"]
    assert fake is not real
    assert workload is sys.modules["metrifid._workload"]
    monkeypatch.undo()

    assert sys.modules["mujoco"] is real
    after = {name: module for name, module in sys.modules.items() if name.startswith("metrifid")}
    assert not (set(after) - set(before)), "a new project module survived teardown"
    replaced = [name for name in set(before) & set(after) if before[name] is not after[name]]
    assert not replaced, "a project module was replaced and not restored"
    assert not [
        name for name, module in after.items() if getattr(module, "mujoco", None) is fake
    ], "a project module is still bound to the fake dependency"


def test_a_real_compile_path_works_after_the_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """The concrete symptom the leak produced: the real MuJoCo binding must still be reachable."""
    _strict_workload_module(monkeypatch)
    monkeypatch.undo()
    import mujoco

    assert mujoco.mj_versionString() == "3.10.0"
    assert mujoco.mj_version() == 3010000
    assert hasattr(mujoco, "FatalError")


def test_the_writer_module_never_imports_the_native_runtime_gate() -> None:
    """Keep the pure workload writers off the shared native-runtime gate entirely."""
    assert not hasattr(writers, "require_supported_runtime")
    source = Path(writers.__file__ or "").read_text(encoding="utf-8")
    assert "require_supported_runtime" not in source


def test_writers_publish_without_any_native_runtime_admission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Write both artifacts while the shared native-runtime gate would refuse everything.

    The gate is forced into unconditional refusal for the duration of the call. The writers are
    pure, so they must still produce their exact artifacts: they never consult it.
    """
    from metrifid import _model_compile as model_compile

    def _always_refuse() -> None:
        """Stand in for a runtime the shared native gate would reject outright."""
        raise AssertionError("pure workload writers must not invoke the native runtime gate")

    monkeypatch.setattr(model_compile, "require_supported_runtime", _always_refuse)
    state_path = tmp_path / "state.npz"
    actions_path = tmp_path / "actions.npz"
    write_state_artifact(
        state_path,
        joint_names=("hinge", "slide"),
        qpos_offsets=(0, 1, 2),
        qpos=np.asarray([1.0, 2.0]),
        qvel_offsets=(0, 1, 2),
        qvel=np.asarray([3.0, 4.0]),
        actuator_names=("hinge_motor",),
        act_offsets=(0, 0),
        act=np.empty((0,)),
    )
    write_actions_artifact(
        actions_path, actuator_names=("hinge_motor",), values=np.asarray([[1.0], [2.0]])
    )
    assert state_path.is_file()
    assert actions_path.is_file()


def test_writers_publish_when_the_native_binding_is_absent(tmp_path: Path) -> None:
    """Write a state artifact in a fresh process where importing MuJoCo always fails.

    A subprocess is used so the blocked import cannot leak into the rest of the session. This is
    the strongest form of the claim: the pure writers do not need the native binding at all.
    """
    script = tmp_path / "writer_without_mujoco.py"
    target = tmp_path / "state.npz"
    script.write_text(
        "import sys\n"
        "class _Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self if name == 'mujoco' or name.startswith('mujoco.') else None\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'mujoco' or name.startswith('mujoco.'):\n"
        "            raise ImportError('mujoco is unavailable in this environment')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        "import numpy as np\n"
        "from metrifid import write_state_artifact\n"
        "write_state_artifact(\n"
        "    sys.argv[1],\n"
        "    joint_names=('hinge',),\n"
        "    qpos_offsets=(0, 1),\n"
        "    qpos=np.asarray([1.5]),\n"
        "    qvel_offsets=(0, 1),\n"
        "    qvel=np.asarray([-0.25]),\n"
        "    actuator_names=(),\n"
        "    act_offsets=(0,),\n"
        "    act=np.empty((0,)),\n"
        ")\n"
        "assert 'mujoco' not in sys.modules, 'the writers imported the native binding'\n"
        "print('WROTE')\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script), str(target)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "WROTE" in completed.stdout
    assert target.is_file()
    with np.load(target) as loaded:
        assert loaded["qvel"].tolist() == [-0.25]
