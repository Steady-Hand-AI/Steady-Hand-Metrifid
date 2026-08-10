"""Frozen behaviour of the timestep fidelity audit surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from metrifid import write_actions_artifact, write_state_artifact
from metrifid.json_values import ExactRational
from metrifid.operational import OperationalReasonCode, OperationalToolObservation
from metrifid.timestep_audit import (
    AUDIT_SCHEMA,
    INCONCLUSIVE,
    OUTSIDE,
    RECOMMENDATION_POLICY,
    REFUSED,
    WITHIN,
    AuditAbort,
    _candidate_row,
    _parse_config,
    _recommendation,
    _steps_per_control,
    candidate_token,
)

NONINTEGRAL = OperationalReasonCode.CONTROL_GRID_NONINTEGRAL.value


TOOL = OperationalToolObservation("0.1.0a10", "VERIFIED_INSTALLED_DISTRIBUTION", "e" * 64)


VALID: dict[str, Any] = {
    "schema_version": 1,
    "model_root": "model",
    "entrypoint": "robot.xml",
    "initial_state": "state.npz",
    "actions": "actions.npz",
    "control_dt": "0.01",
    "repeats": 3,
    "joint_tolerances": {
        "hinge": {"joint_type": "hinge", "angle_rad": "0.005", "angular_velocity_rad_s": "0.25"}
    },
    "candidate_step_dts": ["0.005", "0.002"],
    "workload_kind": "SCREENING",
    "workload_label": "generated screening excitation",
    "output_dir": "audit_out",
}


def parse(**overrides: Any):
    """Decode serialized timestep audit output into typed primitives.

    Exact assertions for timestep audit can inspect canonical evidence rather than formatting
    details.
    """
    payload = {**VALID, **overrides}
    return _parse_config(json.dumps(payload).encode("utf-8"), TOOL)


def row(
    token: str,
    classification: str,
    steps: int | None = 2,
    operational_reason: str | None = None,
) -> dict[str, Any]:
    """Construct coherent row evidence for timestep audit scenarios.

    The fixture preserves hashes and classifications needed to isolate timestep audit.
    """
    return _candidate_row(
        token,
        ExactRational.from_decimal_token("0.005"),
        steps,
        10,
        classification=classification,
        operational_reason=operational_reason,
    )


def skippable_refusal(token: str) -> dict[str, Any]:
    """The one refusal shape the recommendation may skip."""
    return row(token, REFUSED, steps=None, operational_reason=NONINTEGRAL)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Construct coherent aggregate evidence for timestep audit scenarios.

    The fixture preserves hashes and classifications needed to isolate timestep audit.
    """
    from metrifid.json_values import compute_self_hash

    value: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "tool": {
            "version": "0.1.0a10",
            "execution_identity_state": "UNBOUND",
            "distribution_sha256": None,
        },
        "inputs": {
            "configuration_raw_sha256": "a" * 64,
            "source_model_tree_sha256": "b" * 64,
            "entrypoint": "robot.xml",
            "initial_state_raw_sha256": None,
            "actions_raw_sha256": None,
            "initial_state_semantic_sha256": None,
            "actions_semantic_sha256": None,
            "control_dt": {"numerator": 1, "denominator": 100},
            "repeats": 3,
            "workload_kind": "SCREENING",
            "workload_label": "generated screening excitation",
        },
        "reference": {
            "step_dt": {"numerator": 1, "denominator": 1000},
            "steps_per_control_interval": 10,
        },
        "candidates": rows,
        "recommendation": _recommendation(rows),
        "monotonicity_assumed": False,
        "claim_boundary": "bounded",
        "audit_sha256": None,
    }
    value["audit_sha256"] = compute_self_hash(value, "audit_sha256")
    return value


MODEL_XML = """<mujoco model="unit_fixture">
  <option timestep="0.001"/>
  <worldbody>
    <body name="link">
      <joint name="hinge_a" type="hinge" axis="0 0 1"/>
      <geom name="ball" type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def model_root(tmp_path: Path) -> Path:
    """Construct the model root fixture used by timestep audit scenarios.

    Deterministic setup isolates timestep audit without bypassing the contract boundary under
    assertion.
    """
    root = tmp_path / "model"
    root.mkdir()
    (root / "robot.xml").write_text(MODEL_XML, encoding="utf-8")
    return root


SERIALIZATION_TOKENS = ("MjSpec", "to_xml", "copytree", "_write_variant")


def audit_config(root: Path, **overrides: Any) -> Path:
    """Construct the audit config fixture used by timestep audit scenarios.

    Deterministic setup isolates timestep audit without bypassing the contract boundary under
    assertion.
    """
    payload = {
        **VALID,
        "entrypoint": "robot.xml",
        "joint_tolerances": {
            "hinge_a": {
                "joint_type": "hinge",
                "angle_rad": "0.005",
                "angular_velocity_rad_s": "0.25",
            }
        },
        **overrides,
    }
    path = root / "timestep_audit.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


ACTUATED_MODEL_XML = """<mujoco model="confinement_fixture">
  <option timestep="0.001"/>
  <worldbody>
    <body name="link" pos="0 0 0.5">
      <joint name="hinge_a" type="hinge" axis="0 1 0" damping="0.01"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor_a" joint="hinge_a" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


def confinement_case(root: Path, entrypoint: str, output_dir: str) -> Path:
    """One complete audit case whose entrypoint and output directory are caller-chosen."""
    model = root / "model"
    model.mkdir(parents=True, exist_ok=True)
    (model / "robot.xml").write_text(ACTUATED_MODEL_XML, encoding="utf-8")
    write_state_artifact(
        root / "state.npz",
        joint_names=["hinge_a"],
        qpos_offsets=[0, 1],
        qpos=np.array([0.2], dtype=np.float64),
        qvel_offsets=[0, 1],
        qvel=np.array([0.0], dtype=np.float64),
        actuator_names=["motor_a"],
        act_offsets=[0, 0],
        act=np.zeros(0, dtype=np.float64),
    )
    write_actions_artifact(
        root / "actions.npz",
        actuator_names=["motor_a"],
        values=np.zeros((4, 1), dtype=np.float64),
    )
    payload = {
        **VALID,
        "entrypoint": entrypoint,
        "control_dt": "0.01",
        "repeats": 2,
        "joint_tolerances": {
            "hinge_a": {
                "joint_type": "hinge",
                "angle_rad": "0.005",
                "angular_velocity_rad_s": "0.25",
            }
        },
        "candidate_step_dts": ["0.002"],
        "output_dir": output_dir,
    }
    path = root / "timestep_audit.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def product_residue(root: Path) -> list[str]:
    """Every artifact the audit is forbidden to create when it refuses a path."""
    names = {".audit_workspace", "candidates", "timestep_audit.json", "timestep_audit.md"}
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.name in names and p != root / "timestep_audit.json"
    )


CROSSING_RECEIPT: dict[str, Any] = {
    "receipt_sha256": "c" * 64,
    "status": "MATERIAL_BEHAVIOR_CHANGE",
    "first_crossing": {
        "boundary_index": 3,
        "error": {"kind": "ieee754_binary64", "bits": "3ec79f5eaf290000"},
        "joint_name": "hinge",
        "metric": "angle_rad",
        "ratio": {"kind": "ieee754_binary64", "bits": "40068738a843b5c4"},
        "time": {"numerator": 3, "denominator": 25},
        "tolerance": {"numerator": 1, "denominator": 1000000},
    },
    "metrics": {
        "joints": [
            {
                "canonical_name": "hinge",
                "metrics": {
                    "angle_rad": {
                        "maximum_ratio": {"kind": "ieee754_binary64", "bits": "41037c9341addfc9"},
                        "maximum_error": {"kind": "ieee754_binary64", "bits": "3fc46ee676d7408a"},
                        "tolerance": {"numerator": 1, "denominator": 1000000},
                        "worst_boundary_index": 64,
                        "worst_time": {"numerator": 64, "denominator": 25},
                    }
                },
            }
        ]
    },
}


def evidence_rows() -> list[dict[str, Any]]:
    """One refused row and one completed row carrying a real first crossing."""
    refused = _candidate_row(
        "dt_3_over_100",
        ExactRational.from_decimal_token("0.03"),
        None,
        10,
        classification=REFUSED,
        operational_reason="CONTROL_GRID_NONINTEGRAL",
        failure_sha256="d" * 64,
    )
    crossed = _candidate_row(
        "dt_1_over_25",
        ExactRational.from_decimal_token("0.04"),
        1,
        10,
        classification=OUTSIDE,
        comparison_status="MATERIAL_BEHAVIOR_CHANGE",
        receipt=CROSSING_RECEIPT,
    )
    return [refused, crossed]


def candidate_table(text: str) -> list[str]:
    """The candidate table body rows, in order."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Candidate timestep |"))
    body = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        body.append(line)
    return body


def candidate_header(text: str) -> list[str]:
    """The candidate table column names, in rendered order."""
    lines = text.splitlines()
    header = next(line for line in lines if line.startswith("| Candidate timestep |"))
    return [cell.strip() for cell in header.strip().strip("|").split("|")]


REQUIRED_COLUMNS = [
    "Candidate timestep",
    "Steps/interval",
    "Step-count factor",
    "Classification",
    "Operational reason",
    "Maximum tolerance ratio",
    "Worst witness",
    "First crossing",
]


DISTRIBUTION_DIGEST = "d" * 64


@pytest.fixture(autouse=True)
def bound_distribution(monkeypatch: pytest.MonkeyPatch) -> str:
    """Bind a fixed installed-distribution identity for in-process product calls.

    Execution-identity binding is environment plumbing, not audited behaviour. The development
    lane installs this package editable, and an editable install is refused by design, so an
    in-process test cannot reach the audit or comparison logic without binding an identity
    first. Nothing about the audited behaviour is relaxed: no classification, refusal, path
    rule, timestep, receipt field, or hash is patched. The unpatched identity path is exercised
    by the packaged native lane, which runs the real console script from a normal non-editable
    wheel install.
    """
    from metrifid import _audit_config as audit_module
    from metrifid.compare import _orchestrator

    monkeypatch.setattr(_orchestrator, "installed_distribution_sha256", lambda: DISTRIBUTION_DIGEST)
    monkeypatch.setattr(audit_module, "installed_distribution_sha256", lambda: DISTRIBUTION_DIGEST)
    return DISTRIBUTION_DIGEST


def comparison_case(
    root: Path, baseline_dt: str, candidate_dt: str, output_dir: str = "out"
) -> Path:
    """One complete comparison configuration whose two roles are the same source model."""
    model = root / "model"
    model.mkdir(parents=True, exist_ok=True)
    (model / "robot.xml").write_text(ACTUATED_MODEL_XML, encoding="utf-8")
    write_state_artifact(
        root / "state.npz",
        joint_names=["hinge_a"],
        qpos_offsets=[0, 1],
        qpos=np.array([0.2], dtype=np.float64),
        qvel_offsets=[0, 1],
        qvel=np.array([0.0], dtype=np.float64),
        actuator_names=["motor_a"],
        act_offsets=[0, 0],
        act=np.zeros(0, dtype=np.float64),
    )
    write_actions_artifact(
        root / "actions.npz",
        actuator_names=["motor_a"],
        values=np.zeros((4, 1), dtype=np.float64),
    )
    payload = {
        "schema_version": 1,
        "baseline": {
            "model_root": str(model),
            "entrypoint": "robot.xml",
            "declared_step_dt": baseline_dt,
        },
        "candidate": {
            "model_root": str(model),
            "entrypoint": "robot.xml",
            "declared_step_dt": candidate_dt,
        },
        "initial_state": str(root / "state.npz"),
        "actions": str(root / "actions.npz"),
        "control_dt": "0.01",
        "repeats": 2,
        "joint_tolerances": {
            "hinge_a": {
                "joint_type": "hinge",
                "angle_rad": "0.005",
                "angular_velocity_rad_s": "0.25",
            }
        },
        "aliases": None,
        "output_dir": str(root / output_dir),
    }
    path = root / "comparison.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_frozen_config_shape_is_accepted() -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises frozen config shape is accepted; exact-grid evidence and completed-
    prefix rules must never promote an unqualified candidate.
    """
    config = parse()
    assert config.control_dt == ExactRational(1, 100)
    assert config.workload_kind == "SCREENING"
    assert config.repeats == 3


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("model_root"),
        lambda p: p.update(extra=1),
        lambda p: p.update(schema_version=2),
        lambda p: p.update(workload_kind="OTHER"),
        lambda p: p.update(candidate_step_dts=[]),
        lambda p: p.update(candidate_step_dts=["0.001"] * 13),
        lambda p: p.update(candidate_step_dts=["-0.001"]),
        lambda p: p.update(candidate_step_dts=["nope"]),
        lambda p: p.update(control_dt="nope"),
        lambda p: p.update(joint_tolerances={}),
        lambda p: p.update(repeats="3"),
        lambda p: p.update(entrypoint=""),
    ],
)
def test_unknown_missing_or_invalid_keys_refuse(mutate: Any) -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises unknown missing or invalid keys refuse; exact-grid evidence and
    completed-prefix rules must never promote an unqualified candidate.
    """
    payload = json.loads(json.dumps(VALID))
    mutate(payload)
    with pytest.raises(AuditAbort) as exc:
        _parse_config(json.dumps(payload).encode("utf-8"), TOOL)
    assert exc.value.error.failure.operation == "audit-timestep"


def test_numeric_duplicate_candidates_are_rejected_not_deduplicated() -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises numeric duplicate candidates are rejected not deduplicated; exact-
    grid evidence and completed-prefix rules must never promote an unqualified candidate.
    """
    with pytest.raises(AuditAbort):
        parse(candidate_step_dts=["0.002", "0.0020"])


def test_candidates_are_ordered_by_exact_rational_value() -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises candidates are ordered by exact rational value; exact-grid evidence
    and completed-prefix rules must never promote an unqualified candidate.
    """
    config = parse(candidate_step_dts=["0.01", "0.002", "0.04", "0.03", "0.0025"])
    assert [candidate_token(c) for c in config.candidate_step_dts] == [
        "dt_1_over_500",
        "dt_1_over_400",
        "dt_1_over_100",
        "dt_3_over_100",
        "dt_1_over_25",
    ]


def test_candidate_tokens_are_deterministic_normalized_rationals() -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises candidate tokens are deterministic normalized rationals; exact-grid
    evidence and completed-prefix rules must never promote an unqualified candidate.
    """
    assert candidate_token(ExactRational.from_decimal_token("0.0025")) == "dt_1_over_400"
    assert candidate_token(ExactRational.from_decimal_token("0.0020")) == "dt_1_over_500"


@pytest.mark.parametrize(
    ("control", "step", "expected"),
    [("0.01", "0.001", 10), ("0.01", "0.005", 2), ("0.01", "0.004", None), ("0.04", "0.03", None)],
)
def test_steps_per_control_is_exact_and_refuses_non_divisors(
    control: str, step: str, expected: int | None
) -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises steps per control is exact and refuses non divisors; exact-grid
    evidence and completed-prefix rules must never promote an unqualified candidate.
    """
    got = _steps_per_control(
        ExactRational.from_decimal_token(control), ExactRational.from_decimal_token(step)
    )
    assert got == expected


def test_step_count_factor_uses_the_frozen_formula_and_is_null_when_refused() -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises step count factor uses the frozen formula and is null when refused;
    exact-grid evidence and completed-prefix rules must never promote an unqualified candidate.
    """
    integral = row("dt_1_over_200", WITHIN, steps=2)
    assert integral["reference_to_candidate_step_count_factor"] == {
        "numerator": 5,
        "denominator": 1,
    }
    refused = row("dt_1_over_250", REFUSED, steps=None)
    assert refused["reference_to_candidate_step_count_factor"] is None
    assert refused["steps_per_control_interval"] is None


def test_recommendation_stops_permanently_at_the_first_non_within() -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises recommendation stops permanently at the first non within; exact-grid
    evidence and completed-prefix rules must never promote an unqualified candidate.
    """
    rows = [
        row("a", WITHIN),
        row("b", WITHIN),
        row("c", OUTSIDE),
        row("d", WITHIN),  # non-monotonic recovery must never be recommended
    ]
    result = _recommendation(rows)
    assert result["candidate_token"] == "b"
    assert result["blocked_by_prior_non_within"] is True
    assert result["policy"] == RECOMMENDATION_POLICY


def test_exact_nonintegral_grid_refusal_does_not_break_the_prefix() -> None:
    """The one skippable refusal: the candidate grid is nonintegral, so nothing was executed."""
    rows = [row("a", WITHIN), skippable_refusal("b"), row("c", WITHIN)]
    assert _recommendation(rows)["candidate_token"] == "c"
    assert _recommendation(rows)["blocked_by_prior_non_within"] is False


@pytest.mark.parametrize(
    ("label", "refused_row"),
    [
        ("missing reason", row("b", REFUSED, steps=None, operational_reason=None)),
        (
            "wrong reason",
            row("b", REFUSED, steps=None, operational_reason="INTERNAL_INVARIANT_FAILED"),
        ),
        (
            "nonintegral reason but a non-null step count",
            row("b", REFUSED, steps=4, operational_reason=NONINTEGRAL),
        ),
        (
            "output failure reason",
            row("b", REFUSED, steps=None, operational_reason="OUTPUT_WRITE_FAILED"),
        ),
    ],
)
def test_any_other_refused_row_blocks_the_prefix_defensively(
    label: str, refused_row: dict[str, Any]
) -> None:
    """Defense in depth: only the exact predicate is skippable.

    A mapping regression that produced some other `REFUSED` row must not be able to recreate
    false acceptance through the recommendation scan.
    """
    rows = [row("a", WITHIN), refused_row, row("c", WITHIN)]
    result = _recommendation(rows)
    assert result["candidate_token"] == "a", label
    assert result["blocked_by_prior_non_within"] is True, label


def test_inconclusive_candidate_failure_blocks_every_larger_candidate() -> None:
    """An operational failure that produced no comparison evidence blocks the prefix."""
    rows = [
        row("a", WITHIN),
        row("b", INCONCLUSIVE, steps=4, operational_reason="INTERNAL_INVARIANT_FAILED"),
        row("c", WITHIN),
    ]
    result = _recommendation(rows)
    assert result["candidate_token"] == "a"
    assert result["blocked_by_prior_non_within"] is True


def test_first_candidate_failing_inconclusively_yields_no_recommendation() -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises first candidate failing inconclusively yields no recommendation;
    exact-grid evidence and completed-prefix rules must never promote an unqualified candidate.
    """
    rows = [
        row("a", INCONCLUSIVE, steps=5, operational_reason="MODEL_CLOSURE_MUTATED"),
        row("b", WITHIN),
    ]
    result = _recommendation(rows)
    assert result["candidate_token"] is None
    assert result["step_dt"] is None
    assert result["blocked_by_prior_non_within"] is True


def test_inconclusive_blocks_exactly_like_outside() -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises inconclusive blocks exactly like outside; exact-grid evidence and
    completed-prefix rules must never promote an unqualified candidate.
    """
    rows = [row("a", WITHIN), row("b", INCONCLUSIVE), row("c", WITHIN)]
    assert _recommendation(rows)["candidate_token"] == "a"
    assert _recommendation(rows)["blocked_by_prior_non_within"] is True


def test_no_recommendation_when_the_first_completed_candidate_is_not_within() -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises no recommendation when the first completed candidate is not within;
    exact-grid evidence and completed-prefix rules must never promote an unqualified candidate.
    """
    result = _recommendation([row("a", OUTSIDE), row("b", WITHIN)])
    assert result["candidate_token"] is None
    assert result["step_dt"] is None
