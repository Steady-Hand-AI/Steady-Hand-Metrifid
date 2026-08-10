"""Create the canonical workload artifacts that `comparison.json` declares.

`metrifid compare` never invents a workload. It reads one initial state and one action sequence
from canonical NPZ artifacts, so those two files have to exist before the comparison runs. This
script writes them with the public workload writers and refuses to overwrite an existing artifact,
because an artifact already on disk may be the one a recorded result was measured against.

Run it once from this directory:

    python prepare_workload.py
    metrifid compare comparison.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from metrifid import write_actions_artifact, write_state_artifact

# These names and offsets describe the single hinge joint and single motor in both models. The
# hinge contributes one qpos and one qvel entry; the motor has no activation state.
_JOINT_NAMES = ("shoulder",)
_QPOS_OFFSETS = (0, 1)
_QVEL_OFFSETS = (0, 1)
_ACTUATOR_NAMES = ("shoulder_motor",)
_ACT_OFFSETS = (0, 0)

# Ten control steps of a smooth, deterministic torque sweep. Nothing here is random: the same
# workload must produce the same comparison every time it runs.
_CONTROL_STEPS = 10


def _actions() -> np.ndarray:
    """Return one deterministic open-loop torque sequence with shape (steps, actuators)."""
    steps = np.arange(_CONTROL_STEPS, dtype=np.float64)
    return (0.05 * np.sin(steps / 3.0)).reshape(_CONTROL_STEPS, len(_ACTUATOR_NAMES))


def _write(path: Path, write: object, **payload: object) -> bool:
    """Write one artifact unless it already exists.

    Args:
        path: Destination artifact path.
        write: The public writer to call.
        **payload: Keyword arguments forwarded to the writer.

    Returns:
        ``True`` when the artifact was written, ``False`` when it already existed.
    """
    if path.exists():
        print(f"kept existing artifact: {path.name}")
        return False
    write(path, **payload)  # type: ignore[operator]
    print(f"wrote {path.name}")
    return True


def main() -> int:
    """Create `state.npz` and `actions.npz` beside this script.

    Returns:
        ``0`` once both artifacts exist.
    """
    here = Path(__file__).resolve().parent
    _write(
        here / "state.npz",
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
        here / "actions.npz",
        write_actions_artifact,
        actuator_names=_ACTUATOR_NAMES,
        values=_actions(),
    )
    print("workload ready; now run: metrifid compare comparison.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
