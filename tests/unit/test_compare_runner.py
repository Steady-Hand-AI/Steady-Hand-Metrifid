"""Constructed boundary tests for the explicit comparison replay loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from metrifid.compare._runner import run_role_once, run_role_repeats
from tests._support.constructed_models import (
    actions_artifact,
    aligned_actuators,
    aligned_joints,
    state_artifact,
    time_grid,
)


@dataclass(slots=True)
class FakeData:
    """Represent fake data."""

    time: float
    qpos: npt.NDArray[np.float64]
    qvel: npt.NDArray[np.float64]
    act: npt.NDArray[np.float64]
    ctrl: npt.NDArray[np.float64]


class ConstructedBackend:
    """Deterministic arithmetic backend; it is not a MuJoCo emulator."""

    nq = 2
    nv = 2
    na = 0
    nu = 2

    def __init__(self, *, inject_nonfinite_step: int | None = None) -> None:
        """Construct the init fixture used by compare runner scenarios.

        Deterministic setup isolates ConstructedBackend without bypassing the contract boundary
        under assertion.
        """
        self.instances: list[FakeData] = []
        self.controls_seen: list[tuple[float, float]] = []
        self.step_count = 0
        self.inject_nonfinite_step = inject_nonfinite_step

    def new_data(self) -> FakeData:
        """Construct the new data fixture used by compare runner scenarios.

        Deterministic setup isolates ConstructedBackend without bypassing the contract boundary
        under assertion.
        """
        data = FakeData(
            0.0,
            np.zeros((2,), dtype="<f8"),
            np.zeros((2,), dtype="<f8"),
            np.empty((0,), dtype="<f8"),
            np.zeros((2,), dtype="<f8"),
        )
        self.instances.append(data)
        return data

    def forward(self, data: FakeData) -> None:
        """Construct the forward fixture used by compare runner scenarios.

        Deterministic setup isolates ConstructedBackend without bypassing the contract boundary
        under assertion.
        """
        del data

    def step(self, data: FakeData) -> None:
        """Construct the step fixture used by compare runner scenarios.

        Deterministic setup isolates ConstructedBackend without bypassing the contract boundary
        under assertion.
        """
        self.step_count += 1
        self.controls_seen.append((float(data.ctrl[0]), float(data.ctrl[1])))
        data.time += 0.005
        data.qvel[:] = data.ctrl
        data.qpos[:] += data.ctrl * 0.005
        if self.inject_nonfinite_step == self.step_count:
            data.qpos[0] = np.nan

    def warning_snapshot(self, data: FakeData) -> tuple[tuple[int, int, int], ...]:
        """Construct the warning snapshot fixture used by compare runner scenarios.

        Deterministic setup isolates ConstructedBackend without bypassing the contract boundary
        under assertion.
        """
        del data
        return ()

    def clear_auxiliary_inputs(self, data: FakeData) -> None:
        """Construct the clear auxiliary inputs fixture used by compare runner scenarios.

        Deterministic setup isolates ConstructedBackend without bypassing the contract boundary
        under assertion.
        """
        del data


def test_left_boundary_control_mapping_terminal_boundary_and_canonical_order() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises left boundary control mapping terminal boundary and canonical order;
    status, numerical evidence, and artifact publication must remain stable for the declared
    workload.
    """
    backend = ConstructedBackend()
    trace = run_role_once(
        backend=backend,
        role="baseline",
        joints=aligned_joints(),
        actuators=aligned_actuators(),
        state=state_artifact(),
        actions=actions_artifact(),
        time_grid=time_grid(),
    )

    assert trace.complete
    assert trace.boundary_indices == (0, 1, 2)
    assert trace.qpos.shape == (3, 2)
    assert trace.qpos[0].tolist() == [0.25, 1.5]
    assert backend.controls_seen == [(3.0, 2.0), (3.0, 2.0), (5.0, 4.0), (5.0, 4.0)]
    assert [item.to_float() for item in trace.observed_time_bits] == [0.0, 0.01, 0.02]


def test_repeats_use_fresh_independent_data_and_output_arrays() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises repeats use fresh independent data and output arrays; status,
    numerical evidence, and artifact publication must remain stable for the declared workload.
    """
    backend = ConstructedBackend()
    repeated = run_role_repeats(
        backend=backend,
        role="candidate",
        joints=aligned_joints(),
        actuators=aligned_actuators(),
        state=state_artifact(),
        actions=actions_artifact(),
        time_grid=time_grid(),
        repeats=2,
    )
    assert len(backend.instances) == 2
    assert backend.instances[0] is not backend.instances[1]
    assert repeated.stable
    assert repeated.traces[0].qpos is not repeated.traces[1].qpos
    assert not repeated.traces[0].qpos.flags.writeable


def test_first_nonfinite_boundary_stops_without_fabricated_tail() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises first nonfinite boundary stops without fabricated tail; status,
    numerical evidence, and artifact publication must remain stable for the declared workload.
    """
    backend = ConstructedBackend(inject_nonfinite_step=3)
    trace = run_role_once(
        backend=backend,
        role="baseline",
        joints=aligned_joints(),
        actuators=aligned_actuators(),
        state=state_artifact(),
        actions=actions_artifact(),
        time_grid=time_grid(),
    )
    assert trace.boundary_indices == (0, 1)
    assert trace.invalid_kind == "NONFINITE_STATE"
    assert trace.invalid_boundary_index == 2
    assert not trace.complete
    assert trace.qpos.shape[0] == 2
