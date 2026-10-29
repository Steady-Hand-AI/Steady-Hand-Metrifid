"""Fresh-reset planar dynamics with per-step gates and complete final holds.

The fixed planar family uses independently generated scenes and one admitted
physical reset. Native model creation is restricted to execution.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._controller import Array, StrokeController
from ._physical_types import Classification, Measurement, Snapshot, snapshot_well_formed
from ._profile import (
    EPISODE_SECONDS,
    FINER_DT_S,
    HOLD_LIMITS,
    HOLD_SECONDS,
    NOMINAL_DT_S,
    SETTLE_SECONDS,
    generate_expected,
    setup_steps,
)
from ._request import RepairRequest

Step = Callable[[Callable[[], None]], None]
Check = Callable[[], None]
_ARRAY_NAMES = ("qpos", "qvel", "act", "mocap_pos", "mocap_quat")


def _wrapped(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _unknown(failures: set[str]) -> bool:
    return any(
        reason.startswith("NONFINITE_") or reason == "MUJOCO_WARNING_REQUIRES_ADJUDICATION"
        for reason in failures
    )


def target_errors(
    scene: dict[str, Any],
    position: Array,
    yaw: float,
    linear_velocity: Array,
    angular_velocity: Array,
) -> tuple[float, float, float, float]:
    """Measure the caller's resolved goal with the fixed four physical units."""
    target = scene["target"]
    return (
        math.hypot(position[0] - target["xy_m"][0], position[1] - target["xy_m"][1]),
        abs(_wrapped(yaw - target["yaw_rad"])),
        math.sqrt(float(np.dot(linear_velocity, linear_velocity))),
        math.sqrt(float(np.dot(angular_velocity, angular_velocity))),
    )


@dataclass
class _State:
    position: Array
    yaw: float
    tilt: float
    linear: Array
    angular: Array
    finger: Array
    finger_velocity: Array


class _Scene:
    def __init__(
        self, profile: dict[str, Any], nominal_xml: str, dt: float, mj: Any, check: Check
    ) -> None:
        self.profile, self.scene, self.mj, self.dt = profile, profile["scene"], mj, dt
        check()
        self.model = mj.MjModel.from_xml_string(nominal_xml)
        check()
        self.data = mj.MjData(self.model)
        check()
        # Compile the same physical template, then select integration spacing;
        # no timestep-dependent solver state is transferred between cases.
        self.model.opt.timestep = dt
        self.block_body = self._id("BODY", "block_body")
        block_joint = self._id("JOINT", "block_free")
        self.block_q = int(self.model.jnt_qposadr[block_joint])
        joints = [self._id("JOINT", "finger_" + axis) for axis in "xy"]
        self.finger_q = [int(self.model.jnt_qposadr[j]) for j in joints]
        self.finger_v = [int(self.model.jnt_dofadr[j]) for j in joints]
        self.motors = [self._id("ACTUATOR", "motor_" + axis) for axis in "xy"]
        self.geoms = {name: self._id("GEOM", name) for name in ("block", "finger", "floor")}
        floor_ids = sorted((self.geoms["block"], self.geoms["floor"]))
        self.floor_pair = (floor_ids[0], floor_ids[1])
        self.allowed_pairs = {
            self.floor_pair,
            tuple(sorted((self.geoms["block"], self.geoms["finger"]))),
        }
        check()

    def _id(self, kind: str, name: str) -> int:
        value = int(self.mj.mj_name2id(self.model, getattr(self.mj.mjtObj, "mjOBJ_" + kind), name))
        if value < 0:
            raise ValueError("Generated model is missing " + name)
        return value

    def state(self) -> _State:
        self.mj.mj_forward(self.model, self.data)
        velocity = np.zeros(6)
        self.mj.mj_objectVelocity(
            self.model, self.data, self.mj.mjtObj.mjOBJ_BODY, self.block_body, velocity, 0
        )
        rotation = self.data.xmat[self.block_body]
        return _State(
            self.data.xipos[self.block_body],
            math.atan2(rotation[3], rotation[0]),
            math.acos(max(-1.0, min(1.0, float(rotation[8])))),
            velocity[3:],
            velocity[:3],
            np.asarray(self.scene["finger"]["initial_xy_m"]) + self.data.qpos[self.finger_q],
            self.data.qvel[self.finger_v].copy(),
        )

    def contacts(
        self, *, forces: bool = False
    ) -> tuple[set[str], float, dict[tuple[int, int], float]]:
        failures: set[str] = set()
        penetration = 0.0
        loads: dict[tuple[int, int], float] = {}
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
            key = (pair[0], pair[1])
            penetration = max(penetration, -float(contact.dist))
            if key not in self.allowed_pairs:
                failures.add("UNEXPECTED_CONTACT_PAIR")
            if forces and int(contact.efc_address) >= 0:
                wrench = np.zeros(6)
                self.mj.mj_contactForce(self.model, self.data, contact_index, wrench)
                loads[key] = loads.get(key, 0.0) + max(0.0, float(wrench[0]))
        return failures, penetration, loads

    def restore(self, snapshot: Snapshot, nominal_hash: str, check: Check) -> None:
        if not snapshot_well_formed(snapshot, nominal_hash):
            raise ValueError("Invalid admitted physical reset")
        arrays = {
            name: np.asarray(values, dtype=float).reshape(shape)
            for name, shape, values in snapshot.arrays
        }
        for name, array in arrays.items():
            if array.shape != getattr(self.data, name).shape:
                raise ValueError("Reset state dimensions disagree with generated model")
        self.mj.mj_resetData(self.model, self.data)
        for name, array in arrays.items():
            getattr(self.data, name)[:] = array
        self.data.time = 0.0
        self.data.ctrl[:] = 0
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        self.data.qacc_warmstart[:] = 0
        self.mj.mj_forward(self.model, self.data)
        self.data.qacc_warmstart[:] = 0
        check()


def _path_failures(scene: _Scene, state: _State, settled_z: float, penetration: float) -> set[str]:
    gates = scene.scene["whole_episode_gates"]
    failures: set[str] = set()
    if not all(
        np.isfinite(a).all()
        for a in (scene.data.qpos, scene.data.qvel, state.position, state.linear, state.angular)
    ):
        failures.add("NONFINITE_STATE")
    checks = (
        (state.position[2] - settled_z > gates["lift_above_settled_com_m"], "BLOCK_LIFT"),
        (state.tilt > gates["tilt_max_rad"], "BLOCK_TILT"),
        (
            math.sqrt(float(np.dot(state.linear, state.linear))) > gates["block_speed_max_m_s"],
            "BLOCK_SPEED",
        ),
        (math.hypot(*state.finger_velocity) > gates["finger_speed_max_m_s"], "FINGER_SPEED"),
        (penetration > scene.scene["contact"]["penetration_gate_m"], "PENETRATION_LIMIT"),
    )
    failures.update(name for failed, name in checks if failed)
    bounds = gates["block_xy_com_bounds_m"]
    rails = scene.scene["finger"]["rail_limits_xy_m"]
    if any(state.position[k] < bounds[k][0] or state.position[k] > bounds[k][1] for k in (0, 1)):
        failures.add("BLOCK_WORKSPACE")
    if any(
        state.finger[k] < rails[k][0] - 1e-9 or state.finger[k] > rails[k][1] + 1e-9 for k in (0, 1)
    ):
        failures.add("FINGER_RAIL_LIMIT")
    if any(int(warning.number) > 0 for warning in scene.data.warning):
        failures.add("MUJOCO_WARNING_REQUIRES_ADJUDICATION")
    return failures


def _capture(scene: _Scene, state: _State, nominal_hash: str, failures: set[str]) -> Snapshot:
    arrays = tuple(
        (
            name,
            tuple(getattr(scene.data, name).shape),
            tuple(float(value) for value in getattr(scene.data, name).reshape(-1)),
        )
        for name in _ARRAY_NAMES
    )
    payload = b"".join(
        np.asarray(getattr(scene.data, name), dtype="<f8").tobytes()
        for name in sorted(_ARRAY_NAMES)
    )
    position = tuple(float(value) for value in state.position)
    return Snapshot(
        arrays,
        (position[0], position[1], position[2]),
        float(state.yaw),
        nominal_hash,
        hashlib.sha256(payload).hexdigest(),
        tuple(sorted(failures)),
    )


def _settle(
    scene: _Scene, nominal_hash: str, step: Step, check: Check
) -> tuple[Snapshot, Measurement]:
    failures: set[str] = set()
    completed = 0
    state: _State
    for index in range(setup_steps() + 1):
        check()
        state = scene.state()
        contacts, penetration, loads = scene.contacts(forces=True)
        if index * NOMINAL_DT_S >= SETTLE_SECONDS - 0.1 - 1e-12:
            failures.update(contacts)
            failures.update(
                _path_failures(
                    scene, state, scene.scene["block"]["initial_center_m"][2], penetration
                )
            )
            if np.linalg.norm(state.linear) > 0.005:
                failures.add("UNSETTLED_LINEAR_VELOCITY")
            if np.linalg.norm(state.angular) > 0.05:
                failures.add("UNSETTLED_ANGULAR_VELOCITY")
            if loads.get(scene.floor_pair, 0.0) <= 0.05:
                failures.add("UNSETTLED_FLOOR_SUPPORT")
            if any(pair != scene.floor_pair and force > 0.025 for pair, force in loads.items()):
                failures.add("UNEXPECTED_SETTLING_SUPPORT")
        if index < setup_steps():
            scene.data.ctrl[:] = 0

            def advance() -> None:
                nonlocal completed
                scene.mj.mj_step(scene.model, scene.data)
                completed += 1

            step(advance)
    check()
    snapshot = _capture(scene, state, nominal_hash, failures)
    classification: Classification = (
        "INCOMPLETE" if _unknown(failures) else "HARD_REJECTION" if failures else "COMPLETE"
    )
    measurement = Measurement(
        classification,
        not failures,
        "INITIAL_STATE_NOT_ADMISSIBLE" if failures else "ADMITTED",
        metrics={
            "settling_failures": sorted(failures),
            "physical_state_sha256": snapshot.state_sha256,
        },
        hard_failures=tuple(sorted(failures)),
        completed_steps=completed,
        expected_steps=setup_steps(),
        dt_s=NOMINAL_DT_S,
    )
    return snapshot, measurement


@dataclass
class _Episode:
    scene: _Scene
    snapshot: Snapshot
    policy: StrokeController
    completed: int = 0
    hold_samples: int = 0
    hard: set[str] = field(default_factory=set)
    hold: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    maxima: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(
            (
                "maximum_penetration_m",
                "max_command_norm_N",
                "max_finger_speed_m_s",
                "max_block_speed_m_s",
                "maximum_lift_m",
                "maximum_tilt_rad",
                "max_reference_speed_m_s",
            ),
            0.0,
        )
    )
    previous_reference: Array = field(default_factory=lambda: np.zeros(2))
    endpoint: tuple[float, float, float, float] | None = None

    def control(self, time_s: float) -> None:
        data = self.scene.data
        finger = (
            np.asarray(self.scene.scene["finger"]["initial_xy_m"]) + data.qpos[self.scene.finger_q]
        )
        finger_velocity = data.qvel[self.scene.finger_v]
        q = self.scene.block_q
        qw, qx, qy, qz = data.qpos[q + 3 : q + 7]
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        reference, velocity = self.policy.update(time_s, finger, data.qpos[q : q + 2], yaw)
        if not np.isfinite(reference).all() or not np.isfinite(velocity).all():
            self.hard.add("NONFINITE_REFERENCE")
            return
        speed = math.hypot(*velocity)
        self.maxima["max_reference_speed_m_s"] = max(self.maxima["max_reference_speed_m_s"], speed)
        if speed > 0.12 + 1e-9:
            self.hard.add("REFERENCE_SPEED_LIMIT")
        if (
            time_s
            and float(np.linalg.norm(reference - self.previous_reference)) > 0.12 * 0.02 + 1e-9
        ):
            self.hard.add("REFERENCE_DISCONTINUITY")
        self.previous_reference[:] = reference
        bounds = self.scene.profile["controller"]["reference_bounds_xy_m"]
        if any(
            reference[k] < bounds[k][0] - 1e-9 or reference[k] > bounds[k][1] + 1e-9 for k in (0, 1)
        ):
            self.hard.add("REFERENCE_BOUNDS")
        force = 30.0 * (reference - finger) + 1.0 * (velocity - finger_velocity)
        norm = float(np.linalg.norm(force))
        if norm > 2:
            force = force * (2 / norm)
        if not np.isfinite(force).all():
            self.hard.add("NONFINITE_FORCE")
            return
        self.maxima["max_command_norm_N"] = max(
            self.maxima["max_command_norm_N"], math.hypot(*force)
        )
        if math.hypot(*force) > 2.0 + 1e-9:
            self.hard.add("FORCE_NORM_LIMIT")
        data.ctrl[self.scene.motors] = force

    def sample(self, time_s: float) -> None:
        state = self.scene.state()
        contact_failures, penetration, _ = self.scene.contacts()
        self.hard.update(contact_failures)
        self.hard.update(
            _path_failures(self.scene, state, self.snapshot.settled_block_com[2], penetration)
        )
        if _unknown(self.hard):
            return
        errors = target_errors(
            self.scene.scene, state.position, state.yaw, state.linear, state.angular
        )
        maxima = {
            "maximum_penetration_m": penetration,
            "max_finger_speed_m_s": math.hypot(*state.finger_velocity),
            "max_block_speed_m_s": errors[2],
            "maximum_lift_m": float(state.position[2] - self.snapshot.settled_block_com[2]),
            "maximum_tilt_rad": state.tilt,
        }
        for key, value in maxima.items():
            self.maxima[key] = max(self.maxima[key], value)
        if time_s >= EPISODE_SECONDS - HOLD_SECONDS - 1e-12:
            self.hold = [max(old, new) for old, new in zip(self.hold, errors, strict=True)]
            self.hold_samples += 1
        self.endpoint = errors

    def measurement(self, *, interrupted: bool = False) -> Measurement:
        expected = round(EPISODE_SECONDS / self.scene.dt)
        expected_hold = round(HOLD_SECONDS / self.scene.dt) + 1
        complete = (
            self.completed == expected
            and self.hold_samples == expected_hold
            and self.endpoint is not None
        )
        classification: Classification = (
            "INCOMPLETE"
            if interrupted or _unknown(self.hard)
            else "HARD_REJECTION"
            if self.hard
            else "COMPLETE"
            if complete
            else "INCOMPLETE"
        )
        maxima = (
            (self.hold[0], self.hold[1], self.hold[2], self.hold[3]) if self.hold_samples else None
        )
        qualified = bool(
            classification == "COMPLETE"
            and maxima is not None
            and all(value <= limit for value, limit in zip(maxima, HOLD_LIMITS, strict=True))
        )
        reason = (
            ("COMPLETE" if qualified else "TARGET_HOLD_NOT_QUALIFIED")
            if classification == "COMPLETE"
            else (",".join(sorted(self.hard)) if self.hard else "INCOMPLETE_EPISODE")
        )
        metrics: dict[str, object] = {
            **self.maxima,
            "dt_s": self.scene.dt,
            "episode_time_s": float(self.scene.data.time),
            "physics_steps": self.completed,
            "requested_physics_steps": expected,
            "hold_samples": self.hold_samples,
            "expected_hold_samples": expected_hold,
            "settled_state_sha256": self.snapshot.state_sha256,
            "endpoint_errors": self.endpoint,
            "schedule": self.policy.schedule,
            "warnings": [
                {"index": i, "number": int(w.number), "lastinfo": int(w.lastinfo)}
                for i, w in enumerate(self.scene.data.warning)
                if w.number
            ],
        }
        for key, value in zip(
            (
                "hold_max_position_error_m",
                "hold_max_yaw_error_rad",
                "hold_max_linear_speed_m_s",
                "hold_max_angular_speed_rad_s",
            ),
            maxima or (None, None, None, None),
            strict=True,
        ):
            metrics[key] = value
        return Measurement(
            classification,
            qualified,
            reason,
            maxima,
            metrics,
            tuple(sorted(self.hard)),
            self.completed,
            expected,
            self.scene.dt,
            self.hold_samples,
            expected_hold,
        )


class Backend:
    """Two generated timestep models sharing one independently settled reset."""

    def __init__(self, request: RepairRequest, *, step: Step, check: Check) -> None:
        self.profile, self.nominal_xml, self.finer_xml, self.identities = generate_expected(request)
        check()
        mj = importlib.import_module("mujoco")
        check()
        self.nominal = _Scene(self.profile, self.nominal_xml, NOMINAL_DT_S, mj, check)
        nominal_hash = str(self.identities["nominal_scene_sha256"])
        self.snapshot, self.initialization = _settle(self.nominal, nominal_hash, step, check)
        self.finer = _Scene(self.profile, self.nominal_xml, FINER_DT_S, mj, check)
        check()

    def evaluate(
        self, parameters: tuple[float, ...], *, finer: bool, step: Step, check: Check
    ) -> Measurement:
        """Evaluate once from the exact admitted physical and fresh controller state."""
        if len(parameters) != 15 or not all(math.isfinite(v) and 0 <= v <= 1 for v in parameters):
            raise ValueError("Evaluation requires a closed-box selected controller")
        scene = self.finer if finer else self.nominal
        check()
        scene.restore(self.snapshot, str(self.identities["nominal_scene_sha256"]), check)
        policy = StrokeController(self.profile, parameters)
        initial_finger = (
            np.asarray(scene.scene["finger"]["initial_xy_m"]) + scene.data.qpos[scene.finger_q]
        )
        policy.reset(initial_finger)
        episode = _Episode(scene, self.snapshot, policy, previous_reference=initial_finger.copy())
        episode.hard.update(self.snapshot.settling_failures)
        total, control = round(EPISODE_SECONDS / scene.dt), round(0.02 / scene.dt)
        try:
            for index in range(total + 1):
                check()
                time_s = index * scene.dt
                if index % control == 0 and index < total:
                    episode.control(time_s)
                if _unknown(episode.hard):
                    break
                episode.sample(time_s)
                if episode.hard:
                    break
                if index < total:

                    def advance() -> None:
                        scene.mj.mj_step(scene.model, scene.data)
                        episode.completed += 1

                    step(advance)
            check()
        except Exception as exc:
            exc.measurement = episode.measurement(interrupted=True)  # type: ignore[attr-defined]
            raise
        return episode.measurement()


def create_backend(request: RepairRequest, *, step: Step, check: Check) -> Backend:
    """Build the fixed family and charge nominal settling to the invocation."""
    return Backend(request, step=step, check=check)
