"""Controlled backend checks; none of these tests invokes native simulation."""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from metrifid.repair._controller import StrokeController, decode_schedule
from metrifid.repair._physical_types import snapshot_well_formed, validate_snapshot
from metrifid.repair._physics import _Episode, _Scene, _settle, _State, target_errors
from metrifid.repair._profile import generate_expected, minimum_step_budget, pair_steps, setup_steps
from metrifid.repair._request import RepairController, RepairRequest, RepairTask


def _request(rail=-0.06, target=(0.08, 0.06), yaw=math.pi / 2):
    return RepairRequest(RepairTask(rail, target, yaw), RepairController((0.5,) * 15))


def _scene(request=None):
    profile, _, _, identities = generate_expected(request or _request())
    data = SimpleNamespace(
        qpos=np.array([-0.06, -0.04, 0.01, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        qvel=np.zeros(8),
        act=np.zeros(0),
        mocap_pos=np.zeros((0, 3)),
        mocap_quat=np.zeros((0, 4)),
        ctrl=np.zeros(2),
        qfrc_applied=np.zeros(8),
        xfrc_applied=np.zeros((3, 6)),
        qacc_warmstart=np.zeros(8),
        warning=[],
        time=0.0,
    )
    state = _State(
        data.qpos[:3], 0.0, 0.0, np.zeros(3), np.zeros(3), np.array([-0.14, -0.04]), np.zeros(2)
    )
    calls = []

    def advance(model, native_data):
        calls.append(native_data.time)
        native_data.time += 0.002

    scene = SimpleNamespace(
        profile=profile,
        scene=profile["scene"],
        data=data,
        dt=0.002,
        state=lambda: state,
        contacts=lambda **kwargs: (set(), 0.0001, {(0, 1): 0.981}),
        floor_pair=(0, 1),
        model=object(),
        mj=SimpleNamespace(mj_step=advance),
    )
    return scene, calls, identities


def test_task_inputs_reach_model_controller_and_goal_consumers():
    """A variation changes joints, reference rails and measured target errors."""
    first, first_xml, _, _ = generate_expected(_request())
    second, second_xml, _, _ = generate_expected(_request(-0.08, (0.07, 0.05), 0.0))
    for profile, xml, rail in ((first, first_xml, -0.06), (second, second_xml, -0.08)):
        joint = ET.fromstring(xml).find(".//joint[@name='finger_y']")
        assert joint is not None
        assert [float(x) for x in joint.attrib["range"].split()] == [rail + 0.04, 0.22]
        assert profile["controller"]["reference_bounds_xy_m"][1] == [rail, 0.18]
        assert profile["scene"]["finger"]["rail_limits_xy_m"][1] == [rail, 0.18]
        controller = StrokeController(profile, (0.5,) * 15)
        assert controller.protocol["controller"]["reference_bounds_xy_m"][1][0] == rail
    position = np.array([0.08, 0.06, 0.01])
    velocity = np.zeros(3)
    assert target_errors(first["scene"], position, math.pi / 2, velocity, velocity) == (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    changed = target_errors(second["scene"], position, math.pi / 2, velocity, velocity)
    assert changed[0] == pytest.approx(math.sqrt(0.0002))
    assert changed[1] == math.pi / 2
    assert first_xml != second_xml


def test_timestep_scene_identity_and_derived_work_limits():
    """Both scenes expose the actual requested timestep and fixed topology."""
    _, nominal, finer, identities = generate_expected(_request())
    for name, xml, dt in (("nominal", nominal, 0.002), ("finer", finer, 0.001)):
        root = ET.fromstring(xml)
        option = root.find("option")
        assert option is not None
        assert float(option.attrib["timestep"]) == dt
        assert identities[name + "_scene_sha256"] == hashlib.sha256(xml.encode()).hexdigest()
        assert {g.attrib["name"] for g in root.findall(".//geom") if "name" in g.attrib} == {
            "floor",
            "block",
            "finger",
        }
    assert setup_steps() == 200
    assert pair_steps() == 6000
    assert minimum_step_budget() == 12200


def test_settling_counts_every_completed_call_and_guard_interrupts():
    """The actual settling loop delegates every integration to the counter."""
    scene, calls, identities = _scene()
    counted = []

    def step(operation):
        operation()
        counted.append(1)

    snapshot, measured = _settle(scene, identities["nominal_scene_sha256"], step, lambda: None)
    assert len(calls) == len(counted) == measured.completed_steps == 200
    assert measured.classification == "COMPLETE"
    assert snapshot_well_formed(snapshot, identities["nominal_scene_sha256"])
    scene, calls, identities = _scene()

    def check():
        if len(calls) == 3:
            raise TimeoutError("controlled cutoff")

    with pytest.raises(TimeoutError, match="controlled cutoff"):
        _settle(scene, identities["nominal_scene_sha256"], lambda operation: operation(), check)
    assert len(calls) == 3


def test_independent_reset_equality_and_solver_history_clearing():
    """Recomputed digests cannot hide a changed independently settled state."""
    scene, _, identities = _scene()
    snapshot, _ = _settle(
        scene, identities["nominal_scene_sha256"], lambda operation: operation(), lambda: None
    )
    changed = list(snapshot.arrays)
    name, shape, values = changed[0]
    changed[0] = (name, shape, (values[0] + 0.001, *values[1:]))
    payload = b"".join(
        np.asarray(values, dtype="<f8").tobytes() for name, shape, values in sorted(changed)
    )
    forged = replace(
        snapshot, arrays=tuple(changed), state_sha256=hashlib.sha256(payload).hexdigest()
    )
    assert snapshot_well_formed(forged, snapshot.nominal_scene_sha256)
    assert not validate_snapshot(forged, snapshot)
    assert not validate_snapshot(replace(snapshot, settled_block_yaw=0.1), snapshot)
    assert not snapshot_well_formed(replace(snapshot, arrays=()), snapshot.nominal_scene_sha256)
    malformed = tuple(
        (name, (False, 3) if name == "mocap_pos" else shape, values)
        for name, shape, values in snapshot.arrays
    )
    assert not snapshot_well_formed(
        replace(snapshot, arrays=malformed), snapshot.nominal_scene_sha256
    )
    assert validate_snapshot(snapshot, snapshot)
    scene.data.time = 8.0
    scene.data.ctrl[:] = 1
    scene.data.qfrc_applied[:] = 2
    scene.data.xfrc_applied[:] = 3
    scene.data.qacc_warmstart[:] = 4
    scene.mj.mj_resetData = lambda model, data: None

    def forward(model, data):
        data.qacc_warmstart[:] = 7

    scene.mj.mj_forward = forward
    _Scene.restore(scene, snapshot, snapshot.nominal_scene_sha256, lambda: None)
    assert scene.data.time == 0
    for field in ("ctrl", "qfrc_applied", "xfrc_applied", "qacc_warmstart"):
        assert np.count_nonzero(getattr(scene.data, field)) == 0
    for name, shape, values in snapshot.arrays:
        assert np.array_equal(getattr(scene.data, name), np.asarray(values).reshape(shape))


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("PENETRATION_LIMIT", "HARD_REJECTION"),
        ("NONFINITE_STATE", "INCOMPLETE"),
        ("MUJOCO_WARNING_REQUIRES_ADJUDICATION", "INCOMPLETE"),
    ],
)
def test_known_rejection_and_unknown_cannot_be_complete(reason, expected):
    """A hard rejection is distinct from missing or adjudication-required data."""
    scene, _, identities = _scene()
    snapshot, _ = _settle(
        scene, identities["nominal_scene_sha256"], lambda operation: operation(), lambda: None
    )
    episode = _Episode(scene, snapshot, StrokeController(scene.profile, (0.5,) * 15))
    episode.hard.add(reason)
    measured = episode.measurement()
    assert measured.classification == expected
    assert not measured.qualified
    assert measured.hold_maxima is None


def test_full_hold_requires_every_sample_and_preserves_actual_maximum():
    """An endpoint pass cannot replace the full final hold."""
    scene, _, identities = _scene()
    snapshot, _ = _settle(
        scene, identities["nominal_scene_sha256"], lambda operation: operation(), lambda: None
    )
    position = np.array([0.08, 0.06, 0.01])
    state = _State(
        position, math.pi / 2, 0.0, np.zeros(3), np.zeros(3), np.array([-0.14, -0.04]), np.zeros(2)
    )
    scene.state = lambda: state
    episode = _Episode(scene, snapshot, StrokeController(scene.profile, (0.5,) * 15))
    episode.completed = 2000
    episode.sample(4.0)
    assert episode.measurement().classification == "INCOMPLETE"
    episode.hold_samples = 0
    for index in range(101):
        position[0] = 0.086 if index == 0 else 0.08
        episode.sample(3.8 + index * 0.002)
    measured = episode.measurement()
    assert measured.classification == "COMPLETE"
    assert measured.hold_samples == measured.expected_hold_samples == 101
    assert measured.hold_maxima is not None
    assert measured.hold_maxima[0] == pytest.approx(0.006)
    assert not measured.qualified


def test_controller_reset_and_timing_parameters_are_live():
    """Timing suffix changes the schedule and reset discards prior reference."""
    profile, _, _, _ = generate_expected(_request())
    parameters = [0.5] * 15
    first = decode_schedule(parameters)
    parameters[12] = 0.6
    assert decode_schedule(parameters)["boundaries_s"] != first["boundaries_s"]
    policy = StrokeController(profile, parameters)
    policy.reference[:] = 99
    policy.reference_velocity[:] = 99
    policy.reset([-0.14, -0.04])
    assert policy.reference.tolist() == [-0.14, -0.04]
    assert policy.reference_velocity.tolist() == [0.0, 0.0]
