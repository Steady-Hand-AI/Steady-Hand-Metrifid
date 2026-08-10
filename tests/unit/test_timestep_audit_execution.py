"""Frozen behaviour of the timestep fidelity audit surface."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from metrifid import write_actions_artifact, write_state_artifact
from metrifid._atomic_output import (
    PairedOutputNames,
    prepare_paired_output_directory,
)
from metrifid._audit_execution import _publish_audit
from metrifid.errors import ComparisonStatus
from metrifid.json_values import ExactRational, canonical_json_bytes, validate_self_hash
from metrifid.operational import OperationalReasonCode, OperationalToolObservation
from metrifid.timestep_audit import (
    _STATUS_CLASSIFICATION,
    AUDIT_SCHEMA,
    CLAIM_BOUNDARY,
    INCONCLUSIVE,
    OUTSIDE,
    REFUSED,
    WITHIN,
    AuditAbort,
    _candidate_row,
    _parse_config,
    _recommendation,
    _render_markdown,
    _tree_digest,
    audit_configuration_file,
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
    """Decode serialized timestep audit execution output into typed primitives.

    Exact assertions for timestep audit execution can inspect canonical evidence rather than
    formatting details.
    """
    payload = {**VALID, **overrides}
    return _parse_config(json.dumps(payload).encode("utf-8"), TOOL)


def row(
    token: str,
    classification: str,
    steps: int | None = 2,
    operational_reason: str | None = None,
) -> dict[str, Any]:
    """Construct coherent row evidence for timestep audit execution scenarios.

    The fixture preserves hashes and classifications needed to isolate timestep audit execution.
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
    """Construct coherent aggregate evidence for timestep audit execution scenarios.

    The fixture preserves hashes and classifications needed to isolate timestep audit execution.
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
    """Construct the model root fixture used by timestep audit execution scenarios.

    Deterministic setup isolates timestep audit execution without bypassing the contract
    boundary under assertion.
    """
    root = tmp_path / "model"
    root.mkdir()
    (root / "robot.xml").write_text(MODEL_XML, encoding="utf-8")
    return root


SERIALIZATION_TOKENS = ("MjSpec", "to_xml", "copytree", "_write_variant")


def audit_config(root: Path, **overrides: Any) -> Path:
    """Construct the audit config fixture used by timestep audit execution scenarios.

    Deterministic setup isolates timestep audit execution without bypassing the contract
    boundary under assertion.
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


def test_aggregate_self_hash_validates_and_rendering_is_byte_identical() -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises aggregate self hash validates and rendering is byte identical;
    exact-grid evidence and completed-prefix rules must never promote an unqualified candidate.
    """
    value = aggregate([row("dt_1_over_200", WITHIN)])
    validate_self_hash(value, "audit_sha256")
    assert canonical_json_bytes(value) == canonical_json_bytes(
        aggregate([row("dt_1_over_200", WITHIN)])
    )
    assert _render_markdown(value) == _render_markdown(value)


def test_successful_audit_outputs_match_the_held_candidate_bytes(tmp_path: Path) -> None:
    """Keep aggregate JSON and Markdown bytes unchanged by the ownership correction."""
    value = aggregate([row("dt_1_over_200", WITHIN)])
    output = prepare_paired_output_directory(
        tmp_path / "audit",
        PairedOutputNames("timestep_audit.json", "timestep_audit.md"),
    )
    json_path, markdown_path, retained = _publish_audit(output, value)
    try:
        assert hashlib.sha256(json_path.read_bytes()).hexdigest() == (
            "87d50bb0ce83b1d8791950cfc179bb687354b0a67668a61f789bb20680e262c0"
        )
        assert hashlib.sha256(markdown_path.read_bytes()).hexdigest() == (
            "d7e3dca865cd5eeb000a569b430f0790c62cf538ef443117e06267bc04716ee2"
        )
    finally:
        retained.close()
        output.close()


def test_every_comparison_status_maps_to_exactly_one_classification() -> None:
    """A new comparison status must not be able to fall through silently."""
    assert set(_STATUS_CLASSIFICATION) == {status.value for status in ComparisonStatus}
    assert (
        _STATUS_CLASSIFICATION[ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD.value]
        == WITHIN
    )
    assert _STATUS_CLASSIFICATION[ComparisonStatus.MATERIAL_BEHAVIOR_CHANGE.value] == OUTSIDE
    assert _STATUS_CLASSIFICATION[ComparisonStatus.COVERAGE_INSUFFICIENT.value] == INCONCLUSIVE
    assert _STATUS_CLASSIFICATION[ComparisonStatus.NONDETERMINISTIC_REPLAY.value] == INCONCLUSIVE


def test_runtime_timestep_override_changes_only_that_compiled_field(model_root: Path) -> None:
    """`mjOption` is runtime state: assigning opt.timestep must move nothing else."""
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(model_root / "robot.xml"))

    def snapshot(compiled: object) -> dict[str, bytes]:
        """Construct the snapshot fixture used by timestep audit execution scenarios.

        Deterministic setup isolates runtime timestep override changes only that compiled field
        without bypassing the contract boundary under assertion.
        """
        captured: dict[str, bytes] = {}
        for name in dir(compiled):
            if name.startswith("_"):
                continue
            try:
                value = getattr(compiled, name)
            except Exception:  # pragma: no cover - defensive on unreadable members
                continue
            if isinstance(value, np.ndarray):
                captured[name] = value.tobytes()
            elif isinstance(value, (int, float, bool)):
                captured[name] = repr(value).encode("utf-8")
        return captured

    model_before = snapshot(model)
    option_before = snapshot(model.opt)
    assert float(model.opt.timestep) == 0.001

    model.opt.timestep = float(Fraction(1, 200))

    assert float(model.opt.timestep) == 0.005
    assert snapshot(model) == model_before
    option_after = snapshot(model.opt)
    assert set(option_after) == set(option_before)
    differing = {k for k in option_after if option_after[k] != option_before[k]}
    assert differing == {"timestep"}


def test_top_level_abort_leaves_the_output_directory_empty_and_the_source_unchanged(
    model_root: Path, tmp_path: Path
) -> None:
    """A candidate at or below the reference aborts after the output directory is admitted."""
    before = _tree_digest(model_root)
    config = audit_config(tmp_path, candidate_step_dts=["0.001", "0.005"])
    with pytest.raises(AuditAbort) as exc:
        audit_configuration_file(config)
    failure = exc.value.error.failure
    assert failure.operation == "audit-timestep"
    assert failure.reason.field == "candidate_step_dts"

    output = tmp_path / "audit_out"
    assert output.is_dir()
    assert list(output.rglob("*")) == []
    assert _tree_digest(model_root) == before


def test_audit_invocation_and_internal_failures_use_the_audit_operation(
    capsysbinary: pytest.CaptureFixture[bytes], tmp_path: Path
) -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises audit invocation and internal failures use the audit operation;
    exact-grid evidence and completed-prefix rules must never promote an unqualified candidate.
    """
    from metrifid.cli import main

    assert main(["audit-timestep"]) == 64
    emitted = json.loads(capsysbinary.readouterr().err.decode("utf-8"))
    assert emitted["operation"] == "audit-timestep"
    assert emitted["reason"]["code"] == "INVALID_CLI_INVOCATION"

    missing = tmp_path / "absent.json"
    assert main(["audit-timestep", str(missing)]) == 64
    emitted = json.loads(capsysbinary.readouterr().err.decode("utf-8"))
    assert emitted["operation"] == "audit-timestep"
    assert emitted["reason"]["code"] == "CONFIGURATION_IO_FAILED"


def test_claim_boundary_states_the_reference_is_not_ground_truth() -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises claim boundary states the reference is not ground truth; exact-grid
    evidence and completed-prefix rules must never promote an unqualified candidate.
    """
    assert "not asserted to be physically correct or a ground truth" in CLAIM_BOUNDARY
    for forbidden in ("safe timestep", "global equivalence", "speedup", "wall-clock"):
        assert forbidden not in CLAIM_BOUNDARY.lower()


def test_markdown_has_no_raw_python_primitives_and_states_the_claim_boundary() -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises markdown has no raw python primitives and states the claim boundary;
    exact-grid evidence and completed-prefix rules must never promote an unqualified candidate.
    """
    value = aggregate([row("dt_1_over_200", WITHIN), row("dt_1_over_250", REFUSED, steps=None)])
    value["claim_boundary"] = (
        "The reference timestep is the comparison reference; it is not asserted to be "
        "physically correct or a ground truth."
    )
    value["audit_sha256"] = None
    from metrifid.json_values import compute_self_hash

    value["audit_sha256"] = compute_self_hash(value, "audit_sha256")
    text = _render_markdown(value)
    for leaked in ("{'", '"kind":', "ieee754_binary64", "numerator", "denominator"):
        assert leaked not in text
    assert "not asserted to be" in text
    assert "Monotonicity assumed: no" in text
    for forbidden in ("safe timestep", "global equivalence", "wall-clock", "speedup"):
        assert forbidden not in text.lower()


def test_absolute_entrypoint_refuses_before_output_and_preserves_external_file(
    tmp_path: Path,
) -> None:
    """An absolute entrypoint escaped the model root and rewrote the target file."""
    external = tmp_path / "external.xml"
    external.write_text('<mujoco model="sentinel"/>\n', encoding="utf-8")
    external_before = external.read_bytes()
    config = confinement_case(tmp_path, str(external), "audit_out")
    model_before = _tree_digest(tmp_path / "model")

    with pytest.raises(AuditAbort) as exc:
        audit_configuration_file(config)

    failure = exc.value.error.failure
    assert failure.operation == "audit-timestep"
    assert failure.reason.code is OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE
    assert failure.reason.role == "comparison"
    assert int(failure.exit_code) == 64
    assert external.read_bytes() == external_before
    assert not (tmp_path / "audit_out").exists()
    assert _tree_digest(tmp_path / "model") == model_before
    assert product_residue(tmp_path) == []


def test_parent_traversal_entrypoint_refuses_before_output(tmp_path: Path) -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises parent traversal entrypoint refuses before output; exact-grid
    evidence and completed-prefix rules must never promote an unqualified candidate.
    """
    external = tmp_path / "external.xml"
    external.write_text('<mujoco model="sentinel"/>\n', encoding="utf-8")
    external_before = external.read_bytes()
    config = confinement_case(tmp_path, "../external.xml", "audit_out")
    model_before = _tree_digest(tmp_path / "model")

    with pytest.raises(AuditAbort) as exc:
        audit_configuration_file(config)

    failure = exc.value.error.failure
    assert failure.operation == "audit-timestep"
    assert failure.reason.code is OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE
    assert external.read_bytes() == external_before
    assert not (tmp_path / "audit_out").exists()
    assert _tree_digest(tmp_path / "model") == model_before
    assert product_residue(tmp_path) == []


def test_output_equal_to_model_root_refuses_before_copy(tmp_path: Path) -> None:
    """`output_dir == model_root` made `copytree()` copy the audit output into itself."""
    config = confinement_case(tmp_path, "robot.xml", "model")
    model_before = _tree_digest(tmp_path / "model")

    with pytest.raises(AuditAbort) as exc:
        audit_configuration_file(config)

    failure = exc.value.error.failure
    assert failure.operation == "audit-timestep"
    assert failure.reason.code is OperationalReasonCode.OUTPUT_PATH_INVALID
    assert failure.reason.field == "output_dir"
    assert int(failure.exit_code) == 64
    assert failure.reason.evidence["issue"] == "output_inside_model_root"
    # The model root itself must survive: it existed before the audit and is not the audit's.
    assert (tmp_path / "model").is_dir()
    assert _tree_digest(tmp_path / "model") == model_before
    assert product_residue(tmp_path) == []
