"""Model text and workload helpers shared by the SDK example scripts.

Each example is runnable on its own from any directory, so everything it needs is built here from
literals rather than read from the repository. Nothing in this module imports a private Metrifid
name or touches ``sys.path``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from metrifid import write_actions_artifact, write_state_artifact

__all__ = [
    "ACTUATOR_NAMES",
    "CHANGED_MJCF",
    "EQUIVALENT_MJCF",
    "JOINT_NAMES",
    "MODEL_MJCF",
    "write_model",
    "write_workload",
]

# One hinge and one motor: the smallest model that still produces a meaningful trace.
MODEL_MJCF = """<mujoco model="sdk_example">
  <option timestep="0.001"/>
  <worldbody>
    <body name="arm" pos="0 0 1">
      <geom name="link" type="capsule" size="0.04 0.2" mass="1.5"/>
      <joint name="shoulder" type="hinge" axis="0 1 0"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_motor" joint="shoulder" gear="1"/>
  </actuator>
</mujoco>
"""

# The same physics written differently: attribute order and number formatting change, the compiled
# artifact does not.
EQUIVALENT_MJCF = """<mujoco model="sdk_example">
  <option timestep="0.001"/>
  <worldbody>
    <body pos="0 0 1" name="arm">
      <geom mass="1.50" name="link" size="0.04 0.2" type="capsule"/>
      <joint axis="0 1 0" name="shoulder" type="hinge"/>
    </body>
  </worldbody>
  <actuator>
    <motor gear="1" joint="shoulder" name="shoulder_motor"/>
  </actuator>
</mujoco>
"""

# One deliberate physical change: a heavier link.
CHANGED_MJCF = MODEL_MJCF.replace('mass="1.5"', 'mass="1.6"')

JOINT_NAMES = ("shoulder",)
ACTUATOR_NAMES = ("shoulder_motor",)

_CONTROL_STEPS = 10


def write_model(root: Path, text: str = MODEL_MJCF) -> Path:
    """Write one MJCF model into its own root directory.

    Args:
        root: Directory to create and use as the model root.
        text: MJCF source to write.

    Returns:
        The path of the written entrypoint.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / "model.xml"
    path.write_text(text, encoding="utf-8")
    return path


def write_workload(directory: Path) -> tuple[Path, Path]:
    """Write one canonical initial state and one canonical action sequence.

    Args:
        directory: Directory that will hold both artifacts.

    Returns:
        The state artifact path and the actions artifact path.
    """
    directory.mkdir(parents=True, exist_ok=True)
    state = directory / "state.npz"
    actions = directory / "actions.npz"
    write_state_artifact(
        state,
        joint_names=JOINT_NAMES,
        qpos_offsets=(0, 1),
        qpos=np.zeros(1, dtype=np.float64),
        qvel_offsets=(0, 1),
        qvel=np.zeros(1, dtype=np.float64),
        actuator_names=ACTUATOR_NAMES,
        act_offsets=(0, 0),
        act=np.empty(0, dtype=np.float64),
    )
    steps = np.arange(_CONTROL_STEPS, dtype=np.float64)
    write_actions_artifact(
        actions,
        actuator_names=ACTUATOR_NAMES,
        values=(0.05 * np.sin(steps / 3.0)).reshape(_CONTROL_STEPS, len(ACTUATOR_NAMES)),
    )
    return state, actions
