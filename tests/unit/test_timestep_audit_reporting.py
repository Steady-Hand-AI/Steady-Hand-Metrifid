"""Frozen behaviour of the timestep fidelity audit surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from metrifid import write_actions_artifact, write_state_artifact
from metrifid.compare._failure import ComparisonOperationError
from metrifid.json_values import ExactRational, canonical_json_bytes, validate_self_hash
from metrifid.operational import OperationalReasonCode, OperationalToolObservation
from metrifid.timestep_audit import (
    AUDIT_SCHEMA,
    OUTSIDE,
    REFUSED,
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
    """Decode serialized timestep audit reporting output into typed primitives.

    Exact assertions for timestep audit reporting can inspect canonical evidence rather than
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
    """Construct coherent row evidence for timestep audit reporting scenarios.

    The fixture preserves hashes and classifications needed to isolate timestep audit reporting.
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
    """Construct coherent aggregate evidence for timestep audit reporting scenarios.

    The fixture preserves hashes and classifications needed to isolate timestep audit reporting.
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
    """Construct the model root fixture used by timestep audit reporting scenarios.

    Deterministic setup isolates timestep audit reporting without bypassing the contract
    boundary under assertion.
    """
    root = tmp_path / "model"
    root.mkdir()
    (root / "robot.xml").write_text(MODEL_XML, encoding="utf-8")
    return root


SERIALIZATION_TOKENS = ("MjSpec", "to_xml", "copytree", "_write_variant")


def audit_config(root: Path, **overrides: Any) -> Path:
    """Construct the audit config fixture used by timestep audit reporting scenarios.

    Deterministic setup isolates timestep audit reporting without bypassing the contract
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


def test_output_below_model_root_refuses_before_copy(tmp_path: Path) -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises output below model root refuses before copy; exact-grid evidence and
    completed-prefix rules must never promote an unqualified candidate.
    """
    config = confinement_case(tmp_path, "robot.xml", "model/audit_out")
    model_before = _tree_digest(tmp_path / "model")

    with pytest.raises(AuditAbort) as exc:
        audit_configuration_file(config)

    failure = exc.value.error.failure
    assert failure.operation == "audit-timestep"
    assert failure.reason.code is OperationalReasonCode.OUTPUT_PATH_INVALID
    assert failure.reason.field == "output_dir"
    assert int(failure.exit_code) == 64
    assert failure.reason.evidence["issue"] == "output_inside_model_root"
    assert not (tmp_path / "model" / "audit_out").exists()
    assert _tree_digest(tmp_path / "model") == model_before
    assert product_residue(tmp_path) == []


def test_valid_output_outside_configuration_directory_is_allowed(tmp_path: Path) -> None:
    """Confinement is to the model root, not to the configuration directory."""
    case = tmp_path / "case"
    case.mkdir()
    external_parent = tmp_path / "outside"
    external_parent.mkdir()
    config = confinement_case(case, "robot.xml", "../outside/audit_out")
    model_before = _tree_digest(case / "model")

    result = audit_configuration_file(config)

    # `prepare_output_directory()` admits an absolute path without collapsing `..`, so the
    # published location is compared after resolution.
    published = (external_parent / "audit_out").resolve()
    assert result.audit_json.resolve() == published / "timestep_audit.json"
    assert result.audit_json.is_file()
    assert result.audit_markdown.is_file()
    validate_self_hash(result.aggregate, "audit_sha256")
    assert not (case / "audit_out").exists()
    assert _tree_digest(case / "model") == model_before


def test_symlink_model_root_refuses_before_output_and_preserves_target(tmp_path: Path) -> None:
    """A symlink supplied as `model_root` must be refused by the accepted model admission root contract.

    Resolving the declared root before admission replaces the symlink with its real target, so
    `_validate_root()` sees a real directory and its frozen `root_not_real_directory` refusal can
    never fire. The declared root is therefore admitted before it is resolved.
    """
    case = tmp_path / "case"
    case.mkdir()
    config = confinement_case(case, "robot.xml", "audit_out")

    declared = case / "model"
    external = tmp_path / "external_model"
    declared.rename(external)
    declared.symlink_to(external, target_is_directory=True)
    assert declared.is_symlink()
    target_before = _tree_digest(external)

    with pytest.raises(AuditAbort) as exc:
        audit_configuration_file(config)

    failure = exc.value.error.failure
    assert failure.operation == "audit-timestep"
    assert failure.reason.code is OperationalReasonCode.MODEL_ROOT_INVALID
    assert failure.reason.role == "comparison"
    assert failure.reason.evidence["issue"] == "root_not_real_directory"
    assert int(failure.exit_code) == 64

    # Nothing the product creates may exist: no output directory, workspace, candidates
    # directory, copied model, JSON, or Markdown.
    assert not (case / "audit_out").exists()
    assert product_residue(case) == []
    assert not (external / "audit_out").exists()

    # The declared root is still the symlink, and the target tree is byte-identical. A digest
    # over every file under the target also proves no artifact was written inside it.
    assert declared.is_symlink()
    assert _tree_digest(external) == target_before


def test_audit_markdown_exposes_refusal_and_first_crossing() -> None:
    """The refusal reason and the first crossing exist in the JSON; the report must show them."""
    value = aggregate(evidence_rows())
    text = _render_markdown(value)

    assert candidate_header(text) == REQUIRED_COLUMNS

    refused_row, crossed_row = candidate_table(text)
    assert "CONTROL_GRID_NONINTEGRAL" in refused_row
    # A refused candidate has neither a witness nor a crossing; both cells are the em dash.
    assert refused_row.endswith("| — | — |")

    # The completed row shows no operational reason and the full crossing evidence.
    assert "CONTROL_GRID_NONINTEGRAL" not in crossed_row
    assert "`hinge`" in crossed_row
    assert "`angle_rad`" in crossed_row
    assert "boundary 3" in crossed_row
    assert "0.12 (3/25)" in crossed_row
    assert "vs tolerance" in crossed_row

    assert (
        "A non-within result prevented recommendation of larger completed candidates: yes." in text
    )
    for leaked in ("{'", "[{'", '"kind":', "ieee754_binary64", "numerator", "denominator"):
        assert leaked not in text


def test_audit_markdown_escapes_user_controlled_text() -> None:
    """A pipe or newline in model-controlled text must not forge table structure."""
    hostile_joint = "hinge|a\nrow"
    receipt = json.loads(json.dumps(CROSSING_RECEIPT))
    receipt["first_crossing"]["joint_name"] = hostile_joint
    receipt["metrics"]["joints"][0]["canonical_name"] = hostile_joint
    rows = [
        _candidate_row(
            "dt_1_over_25",
            ExactRational.from_decimal_token("0.04"),
            1,
            10,
            classification=OUTSIDE,
            receipt=receipt,
        )
    ]
    value = aggregate(rows)
    inputs = value["inputs"]
    inputs["workload_label"] = "screening | run\\path\r\nsecond line"
    value["audit_sha256"] = None
    from metrifid.json_values import compute_self_hash

    value["audit_sha256"] = compute_self_hash(value, "audit_sha256")
    text = _render_markdown(value)

    assert "screening \\| run\\\\path\\r\\nsecond line" in text
    assert "hinge\\|a\\nrow" in text
    assert "hinge|a" not in text
    # Every escaped string stays on one line, so the table keeps exactly its declared rows.
    assert len(candidate_table(text)) == 1
    assert len([line for line in text.splitlines() if "Workload label" in line]) == 1


def test_audit_markdown_is_byte_deterministic() -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises audit markdown is byte deterministic; exact-grid evidence and
    completed-prefix rules must never promote an unqualified candidate.
    """
    value = aggregate(evidence_rows())
    first = _render_markdown(value)
    second = _render_markdown(value)
    assert first == second
    assert first.encode("utf-8") == second.encode("utf-8")
    assert "\r" not in first
    # Rendering must not mutate the aggregate it was given.
    assert canonical_json_bytes(value) == canonical_json_bytes(aggregate(evidence_rows()))


def test_private_seam_applies_the_candidate_timestep_to_the_same_source_model(
    tmp_path: Path,
) -> None:
    """Both roles are the identical admitted model; only the candidate timestep differs."""
    from metrifid.compare._orchestrator import (
        _compare_configuration_file_with_candidate_timestep,
    )

    config = comparison_case(tmp_path, "0.001", "0.005")
    result = _compare_configuration_file_with_candidate_timestep(config, ExactRational(1, 200))

    primitive = result.receipt.to_primitive()
    inputs = primitive["inputs"]
    assert inputs["baseline_model_closure_sha256"] == inputs["candidate_model_closure_sha256"]
    assert primitive["model_closures"]["baseline"] == primitive["model_closures"]["candidate"]
    assert primitive["time"]["baseline_step_dt"] == {"numerator": 1, "denominator": 1000}
    assert primitive["time"]["candidate_step_dt"] == {"numerator": 1, "denominator": 200}
    # No model file was generated anywhere under the admitted output directory.
    output_root = Path(primitive["time"] and str(tmp_path / "out"))
    assert [p.name for p in output_root.rglob("*") if p.suffix in {".xml", ".mjb"}] == []


def test_private_seam_refuses_a_timestep_the_configuration_did_not_declare(tmp_path: Path) -> None:
    """The override may only restate the declared candidate timestep, and fails before output."""
    from metrifid.compare._orchestrator import (
        _compare_configuration_file_with_candidate_timestep,
    )

    config = comparison_case(tmp_path, "0.001", "0.005")
    with pytest.raises(ComparisonOperationError) as exc:
        _compare_configuration_file_with_candidate_timestep(config, ExactRational(1, 25))

    failure = exc.value.failure
    assert failure.reason.code is OperationalReasonCode.INTERNAL_INVARIANT_FAILED
    assert failure.operation == "compare"
    # It failed closed before the output directory or any replay object existed.
    assert not (tmp_path / "out").exists()


def test_public_compare_applies_no_override_and_still_refuses_a_declared_mismatch(
    tmp_path: Path,
) -> None:
    """The public comparator must never silently retime a model to match its configuration."""
    from metrifid.compare import compare_configuration_file

    config = comparison_case(tmp_path, "0.001", "0.005")
    with pytest.raises(ComparisonOperationError) as exc:
        compare_configuration_file(config)

    failure = exc.value.failure
    assert failure.reason.code is OperationalReasonCode.DECLARED_STEP_DT_MISMATCH
    assert failure.operation == "compare"
