"""Create the four declared workload artifact pairs this example's qualification.json references.

Metrifid never invents a workload. Each candidate is one recorded initial state and one recorded
action sequence, written here by the public workload writers so the example is runnable from a
clean checkout.

The four candidates differ in how hard they drive the joint. That is the whole point of the
example: a gentle workload cannot reveal a small damping change, and the qualification is what
tells you so, rather than you guessing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from metrifid import write_actions_artifact, write_state_artifact

_JOINT_NAMES = ("shoulder",)
_QPOS_OFFSETS = (0, 1)
_QVEL_OFFSETS = (0, 1)
_ACTUATOR_NAMES = ("shoulder_motor",)
_ACT_OFFSETS = (0, 0)
_CONTROL_STEPS = 200

# amplitude in motor units, angular frequency in radians per control step
_WORKLOADS: dict[str, tuple[float, float]] = {
    "slow_low": (0.02, 0.05),
    "slow_high": (0.60, 0.05),
    "fast_low": (0.02, 0.60),
    "fast_high": (0.60, 0.60),
}


def _actions(amplitude: float, frequency: float) -> np.ndarray:
    """Return one deterministic open-loop torque sequence with shape (steps, actuators)."""
    steps = np.arange(_CONTROL_STEPS, dtype=np.float64)
    return (amplitude * np.sin(frequency * steps)).reshape(_CONTROL_STEPS, len(_ACTUATOR_NAMES))


def _write(path: Path, write: object, **payload: object) -> None:
    """Write one artifact unless it already exists."""
    if path.exists():
        print(f"kept existing artifact: {path.parent.name}/{path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, **payload)  # type: ignore[operator]
    print(f"wrote {path.parent.name}/{path.name}")


def main() -> int:
    """Create every declared workload artifact beside this script."""
    here = Path(__file__).resolve().parent
    for workload_id, (amplitude, frequency) in _WORKLOADS.items():
        directory = here / "workloads" / workload_id
        _write(
            directory / "state.npz",
            write_state_artifact,
            joint_names=_JOINT_NAMES,
            qpos_offsets=_QPOS_OFFSETS,
            qpos=np.zeros(1, dtype=np.float64),
            qvel_offsets=_QVEL_OFFSETS,
            qvel=np.zeros(1, dtype=np.float64),
            actuator_names=_ACTUATOR_NAMES,
            act_offsets=_ACT_OFFSETS,
            act=np.empty(0, dtype=np.float64),
        )
        _write(
            directory / "actions.npz",
            write_actions_artifact,
            actuator_names=_ACTUATOR_NAMES,
            values=_actions(amplitude, frequency),
        )
    print("workloads ready; now run: metrifid qualify-workload qualification.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
