"""Example: Record standard Metrifid workload NPZ artifacts from a Gymnasium environment.

This helper extracts the initial state and an open-loop action sequence from a Gymnasium
MuJoCo environment rollout, writing canonical `state.npz` and `actions.npz` artifacts
ready for `metrifid compare`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

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


def _get_joint_properties(model: Any) -> tuple[list[str], list[int], list[int]]:
    joint_names = []
    for i in range(model.njnt):
        name = model.joint(i).name
        if not name:
            raise ValueError(f"Joint {i} is unnamed. Metrifid requires named joints.")
        joint_names.append(name)

    qpos_offsets = [int(v) for v in model.jnt_qposadr] + [int(model.nq)]
    qvel_offsets = [int(v) for v in model.jnt_dofadr] + [int(model.nv)]
    return joint_names, qpos_offsets, qvel_offsets


def _get_actuator_properties(model: Any, data: Any) -> tuple[list[str], list[int], np.ndarray]:
    actuator_names = []
    act_offsets = [0]

    for i in range(model.nu):
        name = model.actuator(i).name
        if not name:
            raise ValueError(f"Actuator {i} is unnamed. Metrifid requires named actuators.")
        actuator_names.append(name)
        act_offsets.append(act_offsets[-1] + int(model.actuator_actnum[i]))

    if data.act is not None and len(data.act) > 0:
        act_array = data.act.copy()
        if act_offsets[-1] != len(act_array):
            raise ValueError(
                f"Activation array length ({len(act_array)}) does not match calculated total actuator_actnum ({act_offsets[-1]})."
            )
    else:
        act_array = np.empty(0, dtype=np.float64)
        if act_offsets[-1] != 0:
            raise ValueError(
                f"Model specifies {act_offsets[-1]} activation variables, but data.act is empty."
            )

    return actuator_names, act_offsets, act_array


def export_workload(env_id: str, output_dir: Path, steps: int = 10, seed: int = 42) -> None:
    """Record an open-loop rollout and write the canonical workload artifacts."""
    if not (1 <= steps <= 100000):
        raise ValueError(f"--steps must be between 1 and 100000. Got: {steps}")

    if output_dir.exists():
        raise FileExistsError(f"--output must not already exist: {output_dir}")

    staging_dir = Path(
        tempfile.mkdtemp(dir=output_dir.parent, prefix=output_dir.name + "_staging_")
    )
    env = gym.make(env_id)

    try:
        env.action_space.seed(seed)
        env.reset(seed=seed)

        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "model") or not hasattr(unwrapped, "data"):
            raise ValueError(f"Environment '{env_id}' must expose MuJoCo 'model' and 'data'.")

        model = unwrapped.model
        data = unwrapped.data

        print(f"Environment: {env_id}")
        print(f"Gymnasium: {gym.__version__}")
        try:
            import mujoco

            print(f"MuJoCo: {mujoco.__version__}")
        except ImportError:
            pass
        print(f"Seed: {seed}")
        print(
            f"Control period (dt): {model.opt.timestep * getattr(unwrapped, 'frame_skip', 1):.5f}s"
        )
        print(f"Requested steps: {steps}")

        joint_names, qpos_offsets, qvel_offsets = _get_joint_properties(model)
        actuator_names, act_offsets, act_array = _get_actuator_properties(model, data)

        # Capture the initial canonical state for the workload
        initial_qpos = data.qpos.copy()
        initial_qvel = data.qvel.copy()

        # Record the open-loop action sequence
        actions = []
        for _ in range(steps):
            # We sample a random action. In practice, you might provide a specific policy trace.
            action = env.action_space.sample()
            actions.append(action)
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                print(
                    "Warning: Environment terminated or truncated before requested steps completed. Continuing open-loop sequence intentionally."
                )

        actions_array = np.vstack(actions).astype(np.float64)

        # 3. Write canonical NPZ artifacts to staging directory
        state_path = staging_dir / "state.npz"
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

        actions_path = staging_dir / "actions.npz"
        write_actions_artifact(
            actions_path,
            actuator_names=actuator_names,
            values=actions_array,
        )

        # Atomically rename to final destination
        staging_dir.rename(output_dir)
        print(f"\nWorkload successfully generated in: {output_dir}")
        print("You can now reference these artifacts in comparison.json.")

    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a Gymnasium workload for Metrifid.")
    parser.add_argument(
        "--env",
        default="HalfCheetah-v5",
        help="Gymnasium MuJoCo environment ID (e.g., HalfCheetah-v5)",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the environment and action space",
    )
    args = parser.parse_args()

    try:
        export_workload(args.env, Path(args.output), args.steps, args.seed)
    except Exception as exc:
        print(f"Failed to record workload: {exc}", file=sys.stderr)
        sys.exit(1)
