"""Example: Record standard Metrifid workload NPZ artifacts from a Gymnasium environment.

This helper extracts the initial state and an open-loop action sequence from a Gymnasium
MuJoCo environment rollout, writing canonical `state.npz` and `actions.npz` artifacts
ready for `metrifid compare`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Require metrifid
try:
    from metrifid import write_actions_artifact, write_state_artifact
except ImportError:
    print("Error: metrifid is not installed. Please install it first.", file=sys.stderr)
    sys.exit(1)

# Require gymnasium
try:
    import gymnasium as gym
except ImportError:
    print("Error: gymnasium is not installed. Please install it first.", file=sys.stderr)
    sys.exit(1)


def export_workload(env_id: str, output_dir: Path, steps: int = 10) -> None:
    """Record an open-loop rollout and write the canonical workload artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id)
    env.reset(seed=42)

    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "model") or not hasattr(unwrapped, "data"):
        raise ValueError(f"Environment '{env_id}' must expose MuJoCo 'model' and 'data'.")

    model = unwrapped.model
    data = unwrapped.data

    # 1. Determine joint properties
    joint_names = []
    qpos_offsets = [0]
    qvel_offsets = [0]

    # JntType: 0=free, 1=ball, 2=slide, 3=hinge
    qpos_widths = {0: 7, 1: 4, 2: 1, 3: 1}
    qvel_widths = {0: 6, 1: 3, 2: 1, 3: 1}

    for i in range(model.njnt):
        name = model.joint(i).name
        if not name:
            # Metrifid requires named joints for deterministic canonical alignment.
            raise ValueError(f"Joint {i} is unnamed. Metrifid requires named joints.")
        joint_names.append(name)

        jtype = model.jnt_type[i]
        qpos_offsets.append(qpos_offsets[-1] + qpos_widths[jtype])
        qvel_offsets.append(qvel_offsets[-1] + qvel_widths[jtype])

    # 2. Determine actuator properties
    actuator_names = []
    act_offsets = [0]

    for i in range(model.nu):
        name = model.actuator(i).name
        if not name:
            raise ValueError(f"Actuator {i} is unnamed. Metrifid requires named actuators.")
        actuator_names.append(name)

        # This example assumes zero activation state width (common for simple hands/grippers).
        # If using muscles or third-order actuators, read model.actuator_actnum[i].
        act_offsets.append(act_offsets[-1] + 0)

    act_array = np.empty(0, dtype=np.float64)
    if data.act is not None and len(data.act) > 0:
        print("Warning: Model has activation state. Ensure act_offsets are mapped correctly.")

    # Capture the initial canonical state for the workload
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    # Record the open-loop action sequence
    actions = []
    for _ in range(steps):
        # We sample a random action. In practice, you might provide a specific policy trace.
        action = env.action_space.sample()
        actions.append(action)
        env.step(action)

    actions_array = np.vstack(actions).astype(np.float64)

    # 3. Write canonical NPZ artifacts
    state_path = output_dir / "state.npz"
    if not state_path.exists():
        write_state_artifact(
            state_path,
            joint_names=joint_names,
            qpos_offsets=qpos_offsets,
            qpos=initial_qpos,
            qvel_offsets=qvel_offsets,
            qvel=initial_qvel,
            actuator_names=actuator_names,
            act_offsets=act_offsets,
            act=act_array,
        )
        print(f"Wrote canonical state: {state_path}")
    else:
        print(f"Skipped existing: {state_path}")

    actions_path = output_dir / "actions.npz"
    if not actions_path.exists():
        write_actions_artifact(
            actions_path,
            actuator_names=actuator_names,
            values=actions_array,
        )
        print(f"Wrote canonical actions: {actions_path}")
    else:
        print(f"Skipped existing: {actions_path}")

    print("\nWorkload successfully generated! You can now reference these in comparison.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a Gymnasium workload for Metrifid.")
    parser.add_argument(
        "--env",
        default="Ant-v4",
        help="Gymnasium MuJoCo environment ID (e.g., Ant-v4)",
    )
    parser.add_argument(
        "--output",
        default="workload_artifacts",
        help="Output directory for state.npz and actions.npz",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of control steps to record",
    )
    args = parser.parse_args()

    try:
        export_workload(args.env, Path(args.output), args.steps)
    except Exception as exc:
        print(f"Failed to record workload: {exc}", file=sys.stderr)
        sys.exit(1)
