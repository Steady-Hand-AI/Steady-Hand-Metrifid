"""Collect workload identity binding scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from metrifid import ExactRational, canonical_json_bytes
from metrifid import _model_identity as identity
from metrifid import _timegrid as timegrid
from metrifid import _workload as workload
from metrifid._npz import ArtifactAdmissionRefusal
from metrifid.operational import OperationalReasonCode
from tests._support.model_identity import build_model_pair_identity


def _write_model(root: Path, *, candidate: bool = False, other_name: bool = False) -> None:
    """Write write model data into the isolated test workspace.

    The workload identity binding scenario observes real bytes and filesystem effects for
    workload identity binding.
    """
    root.mkdir()
    hinge = "other" if other_name else "hinge"
    bodies = (
        f'<body name="h"><joint name="{hinge}" type="hinge"/><geom size=".1" mass="1"/></body>',
        '<body name="s" pos="0 0 1"><joint name="slide" type="slide"/>'
        '<geom size=".1" mass="1"/></body>',
    )
    if candidate:
        bodies = tuple(reversed(bodies))
    actuators = (
        f'<motor name="motor" joint="{hinge}"/>',
        '<general name="integrator" joint="slide" dyntype="integrator" actdim="1"/>',
    )
    if candidate:
        actuators = tuple(reversed(actuators))
    timestep = "0.005" if candidate else "0.002"
    xml = (
        f'<mujoco><option timestep="{timestep}"/><worldbody>{"".join(bodies)}</worldbody>'
        f"<actuator>{''.join(actuators)}</actuator></mujoco>"
    )
    (root / "model.xml").write_text(xml, encoding="utf-8")


def _pair(tmp_path: Path) -> identity.ModelPairIdentity:
    """Construct the pair fixture used by workload identity binding scenarios.

    Deterministic setup isolates workload identity binding without bypassing the contract
    boundary under assertion.
    """
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_model(baseline)
    _write_model(candidate, candidate=True)
    return build_model_pair_identity(baseline, "model.xml", candidate, "model.xml")


def _write_artifacts(root: Path, pair: identity.ModelPairIdentity) -> tuple[Path, Path]:
    """Write write artifacts data into the isolated test workspace.

    The workload identity binding scenario observes real bytes and filesystem effects for
    workload identity binding.
    """
    joints = pair.alignment.joints
    actuators = pair.alignment.actuators

    def offsets(widths: list[int]) -> np.ndarray[tuple[int, ...], np.dtype[np.int64]]:
        """Construct the offsets fixture used by workload identity binding scenarios.

        Deterministic setup isolates write artifacts without bypassing the contract boundary
        under assertion.
        """
        values = [0]
        for width in widths:
            values.append(values[-1] + width)
        return np.array(values, dtype="<i8")

    state = root / "state.npz"
    actions = root / "actions.npz"
    qpos_widths = [item.baseline_qpos[1] for item in joints]
    qvel_widths = [item.baseline_qvel[1] for item in joints]
    act_widths = [item.activation_width for item in actuators]
    np.savez(
        state,
        schema=np.array("metrifid.state", dtype="<U32"),
        schema_version=np.array(1, dtype="<i8"),
        joint_names=np.array([item.canonical_name for item in joints], dtype="<U16"),
        qpos_offsets=offsets(qpos_widths),
        qpos=np.arange(sum(qpos_widths), dtype="<f8") / 10,
        qvel_offsets=offsets(qvel_widths),
        qvel=np.arange(sum(qvel_widths), dtype="<f8") / 20,
        actuator_names=np.array([item.canonical_name for item in actuators], dtype="<U16"),
        act_offsets=offsets(act_widths),
        act=np.array([0.25], dtype="<f8"),
    )
    np.savez(
        actions,
        schema=np.array("metrifid.actions", dtype="<U32"),
        schema_version=np.array(1, dtype="<i8"),
        actuator_names=np.array([item.canonical_name for item in actuators], dtype="<U16"),
        values=np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]], dtype="<f8"),
    )
    return state, actions


def test_actual_mujoco_alignment_binds_workload_and_exact_time_without_stepping(
    tmp_path: Path,
) -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises actual MuJoCo alignment binds workload and exact time without
    stepping; malformed arrays, names, or dimensions must fail before comparison evidence is
    produced.
    """
    pair = _pair(tmp_path)
    state_path, actions_path = _write_artifacts(tmp_path, pair)
    artifacts = workload.load_workload_artifacts(state_path, actions_path, pair)

    baseline_model = mujoco.MjModel.from_xml_path(str(tmp_path / "baseline" / "model.xml"))
    candidate_model = mujoco.MjModel.from_xml_path(str(tmp_path / "candidate" / "model.xml"))
    grid = timegrid.build_time_grid(
        baseline_step_dt=ExactRational.from_decimal_token("0.002"),
        candidate_step_dt=ExactRational.from_decimal_token("0.005"),
        control_dt=ExactRational.from_decimal_token("0.01"),
        baseline_compiled_timestep=float(baseline_model.opt.timestep),
        candidate_compiled_timestep=float(candidate_model.opt.timestep),
        control_intervals=artifacts.actions.metadata.control_intervals,
    )
    assert artifacts.state.metadata.joint_names == ("hinge", "slide")
    assert artifacts.state.metadata.actuator_names == ("integrator", "motor")
    assert grid.baseline_substeps_per_control == 5
    assert grid.candidate_substeps_per_control == 2
    assert grid.boundary_count == artifacts.actions.values.shape[0] + 1
    assert not hasattr(workload, "rollout")


def test_artifacts_cannot_be_rebound_to_a_different_alignment(tmp_path: Path) -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises artifacts cannot be rebound to a different alignment; malformed
    arrays, names, or dimensions must fail before comparison evidence is produced.
    """
    pair = _pair(tmp_path)
    state_path, _ = _write_artifacts(tmp_path, pair)
    other_baseline = tmp_path / "other-baseline"
    other_candidate = tmp_path / "other-candidate"
    _write_model(other_baseline, other_name=True)
    _write_model(other_candidate, candidate=True, other_name=True)
    other = build_model_pair_identity(other_baseline, "model.xml", other_candidate, "model.xml")
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        workload.load_state_artifact(state_path, other)
    assert exc.value.reason is OperationalReasonCode.STATE_NAME_SET_MISMATCH


def test_semantic_hashes_are_identical_across_fresh_processes(tmp_path: Path) -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises semantic hashes are identical across fresh processes; malformed
    arrays, names, or dimensions must fail before comparison evidence is produced.
    """
    pair = _pair(tmp_path)
    state_path, actions_path = _write_artifacts(tmp_path, pair)
    pair_path = tmp_path / "pair.json"
    pair_path.write_bytes(canonical_json_bytes(pair.to_primitive()))
    script = tmp_path / "hashes.py"
    script.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "from metrifid import strict_json_loads\n"
        "from metrifid._model_identity import ModelPairIdentity\n"
        "from metrifid._workload import load_state_artifact, load_actions_artifact\n"
        "pair=ModelPairIdentity.from_primitive(strict_json_loads(Path(sys.argv[1]).read_bytes()))\n"
        "state=load_state_artifact(Path(sys.argv[2]), pair)\n"
        "actions=load_actions_artifact(Path(sys.argv[3]), pair)\n"
        "print(json.dumps([state.semantic_sha256, actions.semantic_sha256], separators=(',',':')))\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    command = [
        sys.executable,
        "-B",
        str(script),
        str(pair_path),
        str(state_path),
        str(actions_path),
    ]
    outputs = [
        subprocess.run(
            command,
            cwd=tmp_path,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for _ in range(2)
    ]
    assert outputs[0] == outputs[1]
    assert len(json.loads(outputs[0])) == 2
