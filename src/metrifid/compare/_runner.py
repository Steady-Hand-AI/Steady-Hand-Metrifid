"""Exact left-boundary replay over a concrete MuJoCo-compatible stepping backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from .._model_closure import AlignedActuator, AlignedJoint
from .._timegrid import TimeGrid
from ..json_values import Binary64, ExactRational
from ._trace import Role, RoleRepeatSet, RoleTrace, _BoundaryRows, build_role_trace


class StateMetadata(Protocol):
    """Describe canonical joint and actuator segment layouts in an initial state."""

    qpos_offsets: tuple[int, ...]
    qvel_offsets: tuple[int, ...]
    act_offsets: tuple[int, ...]


class StateArtifact(Protocol):
    """Expose immutable initial qpos, qvel, and activation vectors."""

    metadata: StateMetadata
    qpos: npt.NDArray[np.float64]
    qvel: npt.NDArray[np.float64]
    act: npt.NDArray[np.float64]


class ActionsMetadata(Protocol):
    """Declare the number of control intervals in an action workload."""

    control_intervals: int


class ActionsArtifact(Protocol):
    """Expose the immutable canonical control-interval matrix."""

    metadata: ActionsMetadata
    values: npt.NDArray[np.float64]


class SimulationData(Protocol):
    """Expose mutable simulator arrays required by the replay boundary."""

    time: float
    qpos: npt.NDArray[np.float64]
    qvel: npt.NDArray[np.float64]
    act: npt.NDArray[np.float64]
    ctrl: npt.NDArray[np.float64]


class StepBackend(Protocol):
    """Small runtime boundary implemented by the real MuJoCo backend."""

    nq: int
    nv: int
    na: int
    nu: int

    def new_data(self) -> SimulationData:
        """Allocate a fresh simulator data object for one replay."""
        ...

    def forward(self, data: SimulationData) -> None:
        """Propagate derived simulator state without advancing time."""
        ...

    def step(self, data: SimulationData) -> None:
        """Advance the simulator data by one native integration step."""
        ...

    def warning_snapshot(self, data: SimulationData) -> tuple[tuple[int, int, int], ...]:
        """Capture warning counters that contribute to numerical evidence."""
        ...

    def clear_auxiliary_inputs(self, data: SimulationData) -> None:
        """Reset non-workload inputs before replaying the next control interval."""
        ...


def run_role_repeats(
    *,
    backend: StepBackend,
    role: Role,
    joints: Sequence[AlignedJoint],
    actuators: Sequence[AlignedActuator],
    state: StateArtifact,
    actions: ActionsArtifact,
    time_grid: TimeGrid,
    repeats: int,
) -> RoleRepeatSet:
    """Execute independent fresh-data repeats for one role."""
    if type(repeats) is not int or not 2 <= repeats <= 5:
        raise ValueError("repeats must be between two and five")
    traces = tuple(
        run_role_once(
            backend=backend,
            role=role,
            joints=joints,
            actuators=actuators,
            state=state,
            actions=actions,
            time_grid=time_grid,
        )
        for _ in range(repeats)
    )
    return RoleRepeatSet(role, traces)


def _prepare_role_data(
    *,
    backend: StepBackend,
    role: Role,
    joints: Sequence[AlignedJoint],
    actuators: Sequence[AlignedActuator],
    state: StateArtifact,
) -> tuple[SimulationData, bool, list[str]]:
    """Restore the declared initial state and report whether the engine preserved it."""
    data = backend.new_data()
    if (len(data.qpos), len(data.qvel), len(data.act), len(data.ctrl)) != (
        backend.nq,
        backend.nv,
        backend.na,
        backend.nu,
    ):
        raise ValueError("backend data layout does not match its declared model dimensions")
    backend.clear_auxiliary_inputs(data)
    data.time = 0.0
    data.ctrl[:] = 0.0
    _restore_state(data, role, joints, actuators, state)
    requested_qpos = np.array(data.qpos, dtype="<f8", copy=True)
    requested_qvel = np.array(data.qvel, dtype="<f8", copy=True)
    requested_act = np.array(data.act, dtype="<f8", copy=True)
    error_logs: list[str] = []
    try:
        backend.forward(data)
    except Exception as exc:  # real backend exceptions become bounded numerical evidence
        error_logs.append(_exception_text(exc))
    preserved = (
        _same_bits(data.qpos, requested_qpos)
        and _same_bits(data.qvel, requested_qvel)
        and _same_bits(data.act, requested_act)
    )
    return data, preserved, error_logs


def _capture_into(
    rows: _BoundaryRows,
    *,
    data: SimulationData,
    role: Role,
    boundary_index: int,
    joints: Sequence[AlignedJoint],
    actuators: Sequence[AlignedActuator],
    time_grid: TimeGrid,
    backend: StepBackend,
) -> str | None:
    """Append one complete role-local state boundary to the trace accumulator."""
    return _capture_boundary(
        data=data,
        role=role,
        boundary_index=boundary_index,
        joints=joints,
        actuators=actuators,
        time_grid=time_grid,
        backend=backend,
        qpos_rows=rows.qpos_rows,
        qvel_rows=rows.qvel_rows,
        act_rows=rows.act_rows,
        boundaries=rows.boundaries,
        canonical_times=rows.canonical_times,
        observed_times=rows.observed_times,
        warnings=rows.warnings,
    )


def _replay_control_intervals(
    rows: _BoundaryRows,
    *,
    data: SimulationData,
    backend: StepBackend,
    role: Role,
    joints: Sequence[AlignedJoint],
    actuators: Sequence[AlignedActuator],
    actions: ActionsArtifact,
    time_grid: TimeGrid,
    error_logs: list[str],
) -> tuple[str | None, int | None]:
    """Step every control interval, stopping at the first invalid boundary."""
    substeps = (
        time_grid.baseline_substeps_per_control
        if role == "baseline"
        else time_grid.candidate_substeps_per_control
    )
    for action_index in range(actions.metadata.control_intervals):
        _assign_control(data, role, actuators, actions.values[action_index])
        try:
            for _ in range(substeps):
                backend.step(data)
        except Exception as exc:
            error_logs.append(_exception_text(exc))
            return "NUMERICAL_ERROR_LOG", action_index + 1
        invalid_kind = _capture_into(
            rows,
            data=data,
            role=role,
            boundary_index=action_index + 1,
            joints=joints,
            actuators=actuators,
            time_grid=time_grid,
            backend=backend,
        )
        if invalid_kind is not None:
            return invalid_kind, action_index + 1
    return None, None


def run_role_once(
    *,
    backend: StepBackend,
    role: Role,
    joints: Sequence[AlignedJoint],
    actuators: Sequence[AlignedActuator],
    state: StateArtifact,
    actions: ActionsArtifact,
    time_grid: TimeGrid,
) -> RoleTrace:
    """Execute one exact schedule, retaining no fabricated post-failure boundary."""
    data, initial_preserved, error_logs = _prepare_role_data(
        backend=backend, role=role, joints=joints, actuators=actuators, state=state
    )
    rows = _BoundaryRows.empty()
    invalid_kind, invalid_boundary = _initial_capture(
        rows, data, backend, role, joints, actuators, time_grid, error_logs
    )
    if invalid_kind is None and initial_preserved:
        invalid_kind, invalid_boundary = _replay_control_intervals(
            rows,
            data=data,
            backend=backend,
            role=role,
            joints=joints,
            actuators=actuators,
            actions=actions,
            time_grid=time_grid,
            error_logs=error_logs,
        )
    elif invalid_kind is None:
        invalid_kind, invalid_boundary = "INITIAL_STATE_NOT_PRESERVED", 0
    widths = (
        sum(item.baseline_qpos[1] for item in joints),
        sum(item.baseline_qvel[1] for item in joints),
        sum(item.activation_width for item in actuators),
    )
    return build_role_trace(
        rows,
        role,
        time_grid.boundary_count,
        widths,
        error_logs,
        invalid_kind,
        invalid_boundary,
        initial_preserved,
    )


def _initial_capture(
    rows: _BoundaryRows,
    data: SimulationData,
    backend: StepBackend,
    role: Role,
    joints: Sequence[AlignedJoint],
    actuators: Sequence[AlignedActuator],
    time_grid: TimeGrid,
    error_logs: list[str],
) -> tuple[str | None, int | None]:
    """Capture the initial boundary or report the first preparation failure."""
    if error_logs:
        return "NUMERICAL_ERROR_LOG", 0
    invalid = _capture_into(
        rows,
        data=data,
        role=role,
        boundary_index=0,
        joints=joints,
        actuators=actuators,
        time_grid=time_grid,
        backend=backend,
    )
    return invalid, 0 if invalid is not None else None


def _restore_state(
    data: SimulationData,
    role: Role,
    joints: Sequence[AlignedJoint],
    actuators: Sequence[AlignedActuator],
    state: StateArtifact,
) -> None:
    """Restore canonical initial joint and activation segments into simulator data."""
    for index, joint in enumerate(joints):
        qpos_left, qpos_right = state.metadata.qpos_offsets[index : index + 2]
        qvel_left, qvel_right = state.metadata.qvel_offsets[index : index + 2]
        qpos_target = joint.baseline_qpos if role == "baseline" else joint.candidate_qpos
        qvel_target = joint.baseline_qvel if role == "baseline" else joint.candidate_qvel
        data.qpos[qpos_target[0] : qpos_target[0] + qpos_target[1]] = state.qpos[
            qpos_left:qpos_right
        ]
        data.qvel[qvel_target[0] : qvel_target[0] + qvel_target[1]] = state.qvel[
            qvel_left:qvel_right
        ]
    for index, actuator in enumerate(actuators):
        left, right = state.metadata.act_offsets[index : index + 2]
        if actuator.activation_width == 0:
            continue
        address = (
            actuator.baseline_activation_address
            if role == "baseline"
            else actuator.candidate_activation_address
        )
        if address is None:
            raise ValueError("stateful aligned actuator has no role-local activation address")
        data.act[address : address + actuator.activation_width] = state.act[left:right]


def _assign_control(
    data: SimulationData,
    role: Role,
    actuators: Sequence[AlignedActuator],
    values: npt.NDArray[np.float64],
) -> None:
    """Map one canonical action row into role-local actuator control addresses."""
    if values.shape != (len(actuators),):
        raise ValueError("one action row must match the canonical actuator count")
    for index, actuator in enumerate(actuators):
        address = (
            actuator.baseline_control_address
            if role == "baseline"
            else actuator.candidate_control_address
        )
        data.ctrl[address] = values[index]


def _capture_boundary(
    *,
    data: SimulationData,
    role: Role,
    boundary_index: int,
    joints: Sequence[AlignedJoint],
    actuators: Sequence[AlignedActuator],
    time_grid: TimeGrid,
    backend: StepBackend,
    qpos_rows: list[npt.NDArray[np.float64]],
    qvel_rows: list[npt.NDArray[np.float64]],
    act_rows: list[npt.NDArray[np.float64]],
    boundaries: list[int],
    canonical_times: list[ExactRational],
    observed_times: list[Binary64],
    warnings: list[tuple[tuple[int, int, int], ...]],
) -> str | None:
    """Capture one boundary or return typed invalid-state evidence without partial rows."""
    qpos = _canonical_joint_row(data.qpos, role, joints, position=True)
    qvel = _canonical_joint_row(data.qvel, role, joints, position=False)
    act = _canonical_activation_row(data.act, role, actuators)
    invalid = _invalid_state_kind(qpos, qvel, act, joints)
    if invalid is not None:
        return invalid
    qpos_rows.append(qpos)
    qvel_rows.append(qvel)
    act_rows.append(act)
    boundaries.append(boundary_index)
    canonical_times.append(time_grid.control_dt.multiplied_by_int(boundary_index))
    observed_times.append(Binary64.from_float(float(data.time)))
    warnings.append(backend.warning_snapshot(data))
    return None


def _canonical_joint_row(
    source: npt.NDArray[np.float64],
    role: Role,
    joints: Sequence[AlignedJoint],
    *,
    position: bool,
) -> npt.NDArray[np.float64]:
    """Concatenate role-local qpos or qvel slices in canonical joint order."""
    pieces: list[npt.NDArray[np.float64]] = []
    for joint in joints:
        if position:
            start, width = joint.baseline_qpos if role == "baseline" else joint.candidate_qpos
        else:
            start, width = joint.baseline_qvel if role == "baseline" else joint.candidate_qvel
        pieces.append(np.asarray(source[start : start + width], dtype="<f8"))
    result = np.ascontiguousarray(np.concatenate(pieces) if pieces else np.empty(0), dtype="<f8")
    return cast(npt.NDArray[np.float64], result)


def _canonical_activation_row(
    source: npt.NDArray[np.float64],
    role: Role,
    actuators: Sequence[AlignedActuator],
) -> npt.NDArray[np.float64]:
    """Concatenate role-local activation slices in canonical actuator order."""
    pieces: list[npt.NDArray[np.float64]] = []
    for actuator in actuators:
        if actuator.activation_width == 0:
            continue
        address = (
            actuator.baseline_activation_address
            if role == "baseline"
            else actuator.candidate_activation_address
        )
        if address is None:
            raise ValueError("stateful aligned actuator has no role-local activation address")
        pieces.append(
            np.asarray(source[address : address + actuator.activation_width], dtype="<f8")
        )
    result = np.ascontiguousarray(np.concatenate(pieces) if pieces else np.empty(0), dtype="<f8")
    return cast(npt.NDArray[np.float64], result)


def _invalid_state_kind(
    qpos: npt.NDArray[np.float64],
    qvel: npt.NDArray[np.float64],
    act: npt.NDArray[np.float64],
    joints: Sequence[AlignedJoint],
) -> str | None:
    """Classify the first state channel containing a nonfinite value."""
    if not (
        bool(np.isfinite(qpos).all())
        and bool(np.isfinite(qvel).all())
        and bool(np.isfinite(act).all())
    ):
        return "NONFINITE_STATE"
    offset = 0
    for joint in joints:
        width = joint.baseline_qpos[1]
        if joint.joint_type == "BALL":
            quaternion = qpos[offset : offset + 4]
            if float(np.dot(quaternion, quaternion)) == 0.0:
                return "INVALID_QUATERNION"
        elif joint.joint_type == "FREE":
            quaternion = qpos[offset + 3 : offset + 7]
            if float(np.dot(quaternion, quaternion)) == 0.0:
                return "INVALID_QUATERNION"
        offset += width
    return None


def _same_bits(left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]) -> bool:
    """Return whether two binary64 arrays have identical shapes and exact bits."""
    if left.shape != right.shape:
        return False
    return bool(np.array_equal(left.view(np.uint64), right.view(np.uint64)))


def _exception_text(exc: Exception) -> str:
    """Render a bounded one-line exception description for numerical evidence."""
    text = str(exc).strip()
    return type(exc).__name__ if not text else f"{type(exc).__name__}: {text}"


__all__ = [
    "ActionsArtifact",
    "ActionsMetadata",
    "SimulationData",
    "StateArtifact",
    "StateMetadata",
    "StepBackend",
    "run_role_once",
    "run_role_repeats",
]
