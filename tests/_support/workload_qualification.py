"""Compact builders for pure qualification-decision tests.

These tests decide from a detection matrix directly. No MuJoCo, no comparison, no filesystem: the
point is that the adjudication and the subset search are exactly specified and independently
checkable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from metrifid._configuration_schemas import JointToleranceConfig, ModelRoleConfig
from metrifid.json_values import ExactRational
from metrifid.workload_qualification._config import (
    ProbeGroup,
    ProbeVariant,
    QualificationConfig,
    WorkloadCandidate,
)
from metrifid.workload_qualification._status import CellOutcome

D = CellOutcome.DETECTED
N = CellOutcome.NOT_DETECTED
U = CellOutcome.UNRESOLVED
_LETTERS = {"D": D, "N": N, "U": U}


def rational(token: str) -> ExactRational:
    """Parse one ordinary decimal token exactly."""
    return ExactRational.from_decimal_token(token)


def role(model_root: str) -> ModelRoleConfig:
    """Build one model role declaration."""
    return ModelRoleConfig(model_root, "model.xml", rational("0.001"))


SEMANTICS = "absolute increase in the declared parameter, in the source model's native units"


def probe_group(
    probe_id: str,
    magnitudes: Sequence[str],
    required: str,
    *,
    parameter: str = "hinge.damping",
    semantics: str = SEMANTICS,
) -> ProbeGroup:
    """Build one probe ladder from decimal magnitude tokens."""
    return ProbeGroup(
        probe_id=probe_id,
        parameter=parameter,
        direction="increase",
        magnitude_semantics=semantics,
        required_detection_magnitude=rational(required),
        variants=tuple(
            ProbeVariant(rational(token), role(f"probes/{probe_id}_{index}"))
            for index, token in enumerate(magnitudes)
        ),
    )


def workload(workload_id: str) -> WorkloadCandidate:
    """Build one workload candidate declaration."""
    return WorkloadCandidate(
        workload_id,
        f"workloads/{workload_id}/state.npz",
        f"workloads/{workload_id}/actions.npz",
        rational("0.01"),
    )


def config(
    groups: Sequence[ProbeGroup], workload_ids: Sequence[str], *, budget: int = 3
) -> QualificationConfig:
    """Build one complete configuration around the supplied groups and workloads."""
    return QualificationConfig(
        schema_version=1,
        baseline=role("baseline"),
        probe_groups=tuple(groups),
        workloads=tuple(workload(item) for item in workload_ids),
        repeats=2,
        joint_tolerances={
            "shoulder": JointToleranceConfig(
                "hinge",
                {"angle_rad": rational("0.001"), "angular_velocity_rad_s": rational("0.01")},
            )
        },
        aliases=None,
        budget=budget,
        output_dir="qualification_out",
    )


def matrix(
    signatures: Mapping[str, Mapping[str, str]],
) -> dict[tuple[str, str, int], CellOutcome]:
    """Build a detection matrix from per-workload, per-probe letter strings.

    ``{"w1": {"p1": "NND"}}`` means workload ``w1`` did not detect the first two rungs of ``p1``
    and detected the third.
    """
    cells: dict[tuple[str, str, int], CellOutcome] = {}
    for workload_id, groups in signatures.items():
        for probe_id, letters in groups.items():
            for index, letter in enumerate(letters):
                cells[(workload_id, probe_id, index)] = _LETTERS[letter]
    return cells


# --- runnable case construction -------------------------------------------------------------
#
# The campaign, tamper, and confinement suites all need one real published qualification. Building
# it from a two-rung ladder over a single-hinge model keeps a complete campaign under a few seconds,
# so those suites can each start from a genuine receipt instead of a hand-written stand-in.

_MODEL = """<mujoco model="qualification_case">
  <option timestep="0.001"/>
  <worldbody>
    <body name="arm" pos="0 0 1">
      <geom name="link" type="capsule" size="0.04 0.2" mass="1.5"/>
      <joint name="shoulder" type="hinge" axis="0 1 0" damping="{damping}"/>
    </body>
  </worldbody>
  <actuator><motor name="shoulder_motor" joint="shoulder" gear="1"/></actuator>
</mujoco>
"""

CASE_WORKLOADS: tuple[tuple[str, float], ...] = (
    ("gentle", 0.05),
    ("medium", 0.30),
    ("strong", 0.60),
)
CASE_RUNGS: tuple[tuple[str, str], ...] = (("0.03", "0.05"), ("0.08", "0.10"))
CASE_REQUIRED = "0.03"


def write_case(root: Path, *, output_dir: str = "qualification_out", **overrides: object) -> Path:
    """Materialize one small runnable qualification case and return its configuration path."""
    import json

    import numpy as np

    from metrifid import write_actions_artifact, write_state_artifact

    root.mkdir(parents=True, exist_ok=True)
    # Resolved because the workload writers admit only real directories, and a scratch root reached
    # through a symlinked temporary prefix would be refused before anything is written.
    root = root.resolve()
    (root / "baseline").mkdir(exist_ok=True)
    (root / "baseline" / "model.xml").write_text(_MODEL.format(damping="0.02"), encoding="utf-8")
    variants = []
    for index, (magnitude, damping) in enumerate(CASE_RUNGS, start=1):
        rung = root / "probes" / "damping_increase" / f"rung_{index}"
        rung.mkdir(parents=True, exist_ok=True)
        (rung / "model.xml").write_text(_MODEL.format(damping=damping), encoding="utf-8")
        variants.append(
            {
                "magnitude": magnitude,
                "candidate": {
                    "model_root": f"probes/damping_increase/rung_{index}",
                    "entrypoint": "model.xml",
                    "declared_step_dt": "0.001",
                },
            }
        )
    for workload_id, amplitude in CASE_WORKLOADS:
        directory = root / "workloads" / workload_id
        directory.mkdir(parents=True, exist_ok=True)
        write_state_artifact(
            directory / "state.npz",
            joint_names=("shoulder",),
            qpos_offsets=(0, 1),
            qpos=np.zeros(1),
            qvel_offsets=(0, 1),
            qvel=np.zeros(1),
            actuator_names=("shoulder_motor",),
            act_offsets=(0, 0),
            act=np.empty(0),
        )
        steps = np.arange(30, dtype=np.float64)
        write_actions_artifact(
            directory / "actions.npz",
            actuator_names=("shoulder_motor",),
            values=(amplitude * np.sin(steps / 3.0)).reshape(30, 1),
        )

    configuration: dict[str, object] = {
        "schema_version": 1,
        "baseline": {
            "model_root": "baseline",
            "entrypoint": "model.xml",
            "declared_step_dt": "0.001",
        },
        "probe_groups": [
            {
                "probe_id": "hinge_damping_increase",
                "parameter": "shoulder.damping",
                "direction": "increase",
                "magnitude_semantics": SEMANTICS,
                "required_detection_magnitude": CASE_REQUIRED,
                "variants": variants,
            }
        ],
        "workloads": [
            {
                "workload_id": workload_id,
                "initial_state": f"workloads/{workload_id}/state.npz",
                "actions": f"workloads/{workload_id}/actions.npz",
                "control_dt": "0.01",
            }
            for workload_id, _amplitude in CASE_WORKLOADS
        ],
        "repeats": 2,
        "joint_tolerances": {
            "shoulder": {
                "joint_type": "hinge",
                "angle_rad": "0.01",
                "angular_velocity_rad_s": "0.05",
            }
        },
        "aliases": None,
        "budget": 3,
        "output_dir": output_dir,
    }
    configuration.update(overrides)
    path = root / "qualification.json"
    path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    return path


# --- the wide-model excluded-zero-change case ------------------------------------------------
#
# Sized so one workload's own zero-change control exceeds compare's retained-trace memory budget
# through the ordinary pre-execution path, with no injection and no patched comparison:
#
#     boundaries x repeats x roles x (qpos + qvel + activation) x 8 bytes
#     65536      x 2       x 2     x (64   + 64   + 1)          x 8  = 270,532,608
#     budget                                                          268,435,456
#
# Sixty-four monitored hinges alone land exactly on the budget, so the single activation unit from
# the integrator actuator is what carries it over. The refusal happens before any stepping, which is
# what keeps this case fast.

WIDE_JOINTS = 64
WIDE_OVERSIZED_INTERVALS = 65_535
WIDE_NORMAL_INTERVALS = 4


def _wide_model(damping: str) -> str:
    """Return one model with 64 monitored hinges and one actuator carrying activation state."""
    bodies = "\n".join(
        f'    <body name="link_{index:02d}" pos="{index * 0.1} 0 1">\n'
        f'      <geom name="geom_{index:02d}" type="capsule" size="0.02 0.05" mass="0.5"/>\n'
        f'      <joint name="hinge_{index:02d}" type="hinge" axis="0 1 0" damping="{damping}"/>\n'
        f"    </body>"
        for index in range(WIDE_JOINTS)
    )
    return (
        '<mujoco model="wide_case">\n'
        '  <option timestep="0.001"/>\n'
        "  <worldbody>\n"
        f"{bodies}\n"
        "  </worldbody>\n"
        "  <actuator>\n"
        '    <general name="driver" joint="hinge_00" dyntype="integrator" gear="1"/>\n'
        "  </actuator>\n"
        "</mujoco>\n"
    )


def write_wide_case(root: Path) -> Path:
    """Materialize the four-workload wide case and return its configuration path."""
    import json

    import numpy as np

    from metrifid import write_actions_artifact, write_state_artifact

    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    (root / "baseline").mkdir(exist_ok=True)
    (root / "baseline" / "model.xml").write_text(_wide_model("0.02"), encoding="utf-8")
    variants = []
    for index, (magnitude, damping) in enumerate((("0.05", "0.07"), ("0.20", "0.22")), start=1):
        rung = root / "probes" / "damping_increase" / f"rung_{index}"
        rung.mkdir(parents=True, exist_ok=True)
        (rung / "model.xml").write_text(_wide_model(damping), encoding="utf-8")
        variants.append(
            {
                "magnitude": magnitude,
                "candidate": {
                    "model_root": f"probes/damping_increase/rung_{index}",
                    "entrypoint": "model.xml",
                    "declared_step_dt": "0.001",
                },
            }
        )

    joint_names = tuple(f"hinge_{index:02d}" for index in range(WIDE_JOINTS))
    qpos_offsets = tuple(range(WIDE_JOINTS + 1))
    workloads = []
    for workload_id, intervals, amplitude in (
        ("normal_low", WIDE_NORMAL_INTERVALS, 0.05),
        ("normal_mid", WIDE_NORMAL_INTERVALS, 0.30),
        ("normal_high", WIDE_NORMAL_INTERVALS, 0.60),
        ("oversized", WIDE_OVERSIZED_INTERVALS, 0.10),
    ):
        directory = root / "workloads" / workload_id
        directory.mkdir(parents=True, exist_ok=True)
        write_state_artifact(
            directory / "state.npz",
            joint_names=joint_names,
            qpos_offsets=qpos_offsets,
            qpos=np.zeros(WIDE_JOINTS),
            qvel_offsets=qpos_offsets,
            qvel=np.zeros(WIDE_JOINTS),
            actuator_names=("driver",),
            act_offsets=(0, 1),
            act=np.zeros(1),
        )
        steps = np.arange(intervals, dtype=np.float64)
        write_actions_artifact(
            directory / "actions.npz",
            actuator_names=("driver",),
            values=(amplitude * np.sin(steps / 3.0)).reshape(intervals, 1),
        )
        workloads.append(
            {
                "workload_id": workload_id,
                "initial_state": f"workloads/{workload_id}/state.npz",
                "actions": f"workloads/{workload_id}/actions.npz",
                "control_dt": "0.001",
            }
        )

    configuration = {
        "schema_version": 1,
        "baseline": {
            "model_root": "baseline",
            "entrypoint": "model.xml",
            "declared_step_dt": "0.001",
        },
        "probe_groups": [
            {
                "probe_id": "wide_damping_increase",
                "parameter": "hinge_00.damping",
                "direction": "increase",
                "magnitude_semantics": SEMANTICS,
                "required_detection_magnitude": "0.05",
                "variants": variants,
            }
        ],
        "workloads": workloads,
        "repeats": 2,
        "joint_tolerances": {
            name: {
                "joint_type": "hinge",
                "angle_rad": "0.000001",
                "angular_velocity_rad_s": "0.0001",
            }
            for name in joint_names
        },
        "aliases": None,
        "budget": 3,
        "output_dir": "qualification_out",
    }
    path = root / "qualification.json"
    path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    return path
