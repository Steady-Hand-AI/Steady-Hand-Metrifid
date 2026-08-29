"""Concrete admitted-MuJoCo stepping adapter for the comparison replay core."""

from __future__ import annotations

from typing import cast

import mujoco  # type: ignore[import-untyped]
import numpy as np

from ..errors import EngineThreadpoolState
from ._runner import SimulationData


class MuJoCoBackend:
    """Thin explicit adapter around one already-admitted compiled model."""

    def __init__(self, model: mujoco.MjModel) -> None:
        """Bind the runtime adapter to one already-admitted compiled model."""
        self.model = model
        self.nq = int(model.nq)
        self.nv = int(model.nv)
        self.na = int(model.na)
        self.nu = int(model.nu)

    def new_data(self) -> SimulationData:
        """Allocate fresh MuJoCo simulation data for an independent replay."""
        return cast(SimulationData, mujoco.MjData(self.model))

    def forward(self, data: SimulationData) -> None:
        """Run MuJoCo forward propagation without advancing simulation time."""
        mujoco.mj_forward(self.model, cast(mujoco.MjData, data))

    def step(self, data: SimulationData) -> None:
        """Advance role-local MuJoCo data by one native integration step."""
        mujoco.mj_step(self.model, cast(mujoco.MjData, data))

    def warning_snapshot(self, data: SimulationData) -> tuple[tuple[int, int, int], ...]:
        """Capture all MuJoCo warning counters in stable registry order."""
        raw = getattr(data, "warning", ())
        result: list[tuple[int, int, int]] = []
        for index, warning in enumerate(raw):
            number = int(getattr(warning, "number", 0))
            lastinfo = int(getattr(warning, "lastinfo", 0))
            if number:
                result.append((index, number, lastinfo))
        return tuple(result)

    def clear_auxiliary_inputs(self, data: SimulationData) -> None:
        """Zero applied forces and mocap inputs before the next control interval."""
        for name in ("qacc_warmstart", "qfrc_applied", "xfrc_applied", "userdata"):
            array = getattr(data, name, None)
            if isinstance(array, np.ndarray):
                array[:] = 0.0


def observed_threadpool_state(data: SimulationData) -> EngineThreadpoolState:
    """Classify the role-local optional MuJoCo data threadpool without enabling it."""
    if not hasattr(data, "threadpool"):
        return EngineThreadpoolState.UNKNOWN
    value = data.threadpool
    if value is None or value is False:
        return EngineThreadpoolState.DISABLED
    try:
        if int(value) == 0:
            return EngineThreadpoolState.DISABLED
    except (TypeError, ValueError, OverflowError):
        pass
    return EngineThreadpoolState.ACTIVE


__all__ = ["MuJoCoBackend", "observed_threadpool_state"]
