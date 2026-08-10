"""Constructed values for comparison formula, boundary, and failure-path tests.

These helpers do not emulate MuJoCo and are not evidence for the real-runtime gates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from metrifid import Binary64, ExactRational
from metrifid._model_closure import AlignedActuator, AlignedJoint
from metrifid._timegrid import TimeGrid
from metrifid.compare._trace import Role, RoleTrace, readonly_matrix
from metrifid.schemas import TargetReference


@dataclass(frozen=True, slots=True)
class StateMetadata:
    """Represent state metadata."""

    qpos_offsets: tuple[int, ...]
    qvel_offsets: tuple[int, ...]
    act_offsets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StateArtifact:
    """Represent state artifact."""

    metadata: StateMetadata
    qpos: npt.NDArray[np.float64]
    qvel: npt.NDArray[np.float64]
    act: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ActionsMetadata:
    """Represent actions metadata."""

    control_intervals: int


@dataclass(frozen=True, slots=True)
class ActionsArtifact:
    """Represent actions artifact."""

    metadata: ActionsMetadata
    values: npt.NDArray[np.float64]


def aligned_joints() -> tuple[AlignedJoint, ...]:
    """Return canonical hinge/slide joints with deliberately different native orders."""
    return (
        AlignedJoint("hinge", "HINGE", (1, 1), (0, 1), (1, 1), (0, 1)),
        AlignedJoint("slide", "SLIDE", (0, 1), (1, 1), (0, 1), (1, 1)),
    )


def aligned_actuators() -> tuple[AlignedActuator, ...]:
    """Return canonical stateless actuators with reversed role-local control addresses."""
    return (
        AlignedActuator(
            "hinge_motor",
            "JOINT",
            (TargetReference("JOINT", "hinge"),),
            "NONE",
            0,
            1,
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
            0,
            1,
            None,
            None,
        ),
    )


def time_grid(control_intervals: int = 2) -> TimeGrid:
    """Construct the time grid fixture used by constructed models scenarios.

    Deterministic setup isolates constructed models without bypassing the contract boundary
    under assertion.
    """
    step = ExactRational(1, 200)
    control = ExactRational(1, 100)
    return TimeGrid(
        step,
        step,
        control,
        Binary64.from_float(float(1 / 200)),
        Binary64.from_float(float(1 / 200)),
        2,
        2,
        control_intervals,
        control_intervals + 1,
        control.multiplied_by_int(control_intervals),
    )


def state_artifact() -> StateArtifact:
    """Construct the state artifact fixture used by constructed models scenarios.

    Deterministic setup isolates constructed models without bypassing the contract boundary
    under assertion.
    """
    return StateArtifact(
        StateMetadata((0, 1, 2), (0, 1, 2), (0, 0, 0)),
        np.asarray([0.25, 1.5], dtype="<f8"),
        np.asarray([0.5, -0.25], dtype="<f8"),
        np.empty((0,), dtype="<f8"),
    )


def actions_artifact() -> ActionsArtifact:
    """Construct the actions artifact fixture used by constructed models scenarios.

    Deterministic setup isolates constructed models without bypassing the contract boundary
    under assertion.
    """
    values = np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype="<f8")
    return ActionsArtifact(ActionsMetadata(2), values)


def role_trace(
    role: Role,
    *,
    qpos_rows: list[list[float]],
    qvel_rows: list[list[float]],
    expected_boundary_count: int | None = None,
    invalid_kind: str | None = None,
    invalid_boundary_index: int | None = None,
    warning_boundary: int | None = None,
) -> RoleTrace:
    """Construct the role trace fixture used by constructed models scenarios.

    Deterministic setup isolates constructed models without bypassing the contract boundary
    under assertion.
    """
    count = len(qpos_rows)
    expected = count if expected_boundary_count is None else expected_boundary_count
    grid = time_grid(max(expected - 1, 1))
    warning_snapshots = tuple(
        ((0, 1, 7),) if index == warning_boundary else () for index in range(count)
    )
    qpos_arrays = [np.asarray(row, dtype="<f8") for row in qpos_rows]
    qvel_arrays = [np.asarray(row, dtype="<f8") for row in qvel_rows]
    act_arrays = [np.empty((0,), dtype="<f8") for _ in range(count)]
    return RoleTrace(
        role=role,
        expected_boundary_count=expected,
        boundary_indices=tuple(range(count)),
        canonical_times=tuple(grid.control_dt.multiplied_by_int(index) for index in range(count)),
        observed_time_bits=tuple(grid.iter_role_boundary_time_bits(role))[:count],
        qpos=readonly_matrix(qpos_arrays, len(qpos_rows[0]) if qpos_rows else 0),
        qvel=readonly_matrix(qvel_arrays, len(qvel_rows[0]) if qvel_rows else 0),
        act=readonly_matrix(act_arrays, 0),
        warning_snapshots=warning_snapshots,
        error_logs=(),
        invalid_kind=invalid_kind,
        invalid_boundary_index=invalid_boundary_index,
        initial_state_preserved=True,
    )
