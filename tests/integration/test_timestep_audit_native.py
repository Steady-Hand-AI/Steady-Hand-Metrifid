"""End-to-end timestep audit against a real compiled MuJoCo model.

The model is a small deliberately-authored fixture, not a task policy, and no claim is
made about its physical realism. The candidate list mixes an integral candidate, a
non-integral candidate, and a further integral candidate so the refusal seam and the
"later candidates still execute" rule are exercised on the real path.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from metrifid import write_actions_artifact, write_state_artifact
from metrifid.errors import ReasonRole
from metrifid.json_values import ExactRational, validate_self_hash
from metrifid.operational import OperationalReasonCode, OperationalToolObservation
from metrifid.timestep_audit import (
    INCONCLUSIVE,
    OUTSIDE,
    REFUSED,
    WITHIN,
    AuditAbort,
    _tree_digest,
    audit_configuration_file,
    candidate_token,
)
from metrifid.version import __version__ as CURRENT_VERSION

MODEL_XML = """<mujoco model="native_fixture">
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

CONTROLS = 12
CONTROL_DT = "0.01"
# 0.01 / 0.004 = 2.5, so the middle candidate cannot land on the control grid.
CANDIDATES = ["0.002", "0.004", "0.005"]

DISTRIBUTION_DIGEST = "d" * 64


@pytest.fixture(autouse=True)
def bound_distribution(monkeypatch: pytest.MonkeyPatch) -> str:
    """Bind a fixed installed-distribution identity for in-process product calls.

    The development lane installs this package editable and an editable install is refused by
    design, so an in-process test cannot reach the audit without binding an identity first.
    No audited behaviour is relaxed. The unpatched identity path is exercised by the packaged
    native lane, which runs the real console script from a normal non-editable wheel install.
    """
    from metrifid import _audit_config as audit_module
    from metrifid.compare import _orchestrator

    monkeypatch.setattr(_orchestrator, "installed_distribution_sha256", lambda: DISTRIBUTION_DIGEST)
    monkeypatch.setattr(audit_module, "installed_distribution_sha256", lambda: DISTRIBUTION_DIGEST)
    return DISTRIBUTION_DIGEST


def build_case(root: Path, candidates: list[str] | None = None) -> Path:
    """Construct the build case fixture used by timestep audit native scenarios.

    Deterministic setup isolates timestep audit native without bypassing the contract boundary
    under assertion.
    """
    model_root = root / "model"
    model_root.mkdir(parents=True)
    (model_root / "robot.xml").write_text(MODEL_XML, encoding="utf-8")

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
    values = 0.5 * np.sin(2.0 * np.pi * np.arange(CONTROLS, dtype=np.float64) / CONTROLS)
    write_actions_artifact(
        root / "actions.npz",
        actuator_names=["motor_a"],
        values=values.reshape(CONTROLS, 1),
    )

    config = {
        "schema_version": 1,
        "model_root": "model",
        "entrypoint": "robot.xml",
        "initial_state": "state.npz",
        "actions": "actions.npz",
        "control_dt": CONTROL_DT,
        "repeats": 3,
        "joint_tolerances": {
            "hinge_a": {
                "joint_type": "hinge",
                "angle_rad": "0.005",
                "angular_velocity_rad_s": "0.25",
            }
        },
        "candidate_step_dts": CANDIDATES if candidates is None else candidates,
        "workload_kind": "SCREENING",
        "workload_label": "authored deterministic screening excitation, not a task policy",
        "output_dir": "audit_out",
    }
    path = root / "timestep_audit.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def case(tmp_path: Path) -> tuple[Path, str]:
    """A fresh tree per test: the audit refuses a non-empty admitted output directory."""
    config = build_case(tmp_path)
    return config, _tree_digest(tmp_path / "model")


def test_native_audit_refuses_the_non_integral_candidate_and_continues(
    case: tuple[Path, str],
) -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises native audit refuses the non integral candidate and continues;
    exact-grid evidence and completed-prefix rules must never promote an unqualified candidate.
    """
    config, tree_before = case
    result = audit_configuration_file(config)
    rows = {row["token"]: row for row in result.aggregate["candidates"]}
    assert list(rows) == ["dt_1_over_500", "dt_1_over_250", "dt_1_over_200"]

    refused = rows["dt_1_over_250"]
    assert refused["classification"] == REFUSED
    assert refused["operational_reason"] == "CONTROL_GRID_NONINTEGRAL"
    assert refused["steps_per_control_interval"] is None
    assert refused["reference_to_candidate_step_count_factor"] is None
    assert refused["receipt_sha256"] is None

    # The candidate after the refusal still executed: the refusal is per-candidate.
    completed = [rows["dt_1_over_500"], rows["dt_1_over_200"]]
    for row in completed:
        assert row["classification"] in {WITHIN, OUTSIDE, INCONCLUSIVE}
        assert row["receipt_sha256"] is not None
        assert row["failure_sha256"] is None
    assert rows["dt_1_over_500"]["steps_per_control_interval"] == 5
    assert rows["dt_1_over_200"]["steps_per_control_interval"] == 2

    assert _tree_digest(config.parent / "model") == tree_before


def test_preserved_refusal_failure_keeps_the_compare_operation(case: tuple[Path, str]) -> None:
    """The audit widened the operation registry; a comparison refusal is still `compare`."""
    config, _ = case
    result = audit_configuration_file(config)
    row = next(r for r in result.aggregate["candidates"] if r["classification"] == REFUSED)
    path = config.parent / "audit_out" / str(row["operational_failure_json"])
    failure = json.loads(path.read_text(encoding="utf-8"))
    assert failure["operation"] == "compare"
    assert failure["reason"]["code"] == "CONTROL_GRID_NONINTEGRAL"
    assert failure["exit_code"] == 64
    assert failure["failure_sha256"] == row["failure_sha256"]
    validate_self_hash(failure, "failure_sha256")


def test_repeated_native_runs_are_byte_identical(case: tuple[Path, str]) -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises repeated native runs are byte identical; exact-grid evidence and
    completed-prefix rules must never promote an unqualified candidate.
    """
    config, tree_before = case
    first = audit_configuration_file(config)
    first_json = first.audit_json.read_bytes()
    first_md = first.audit_markdown.read_bytes()
    # Same configuration bytes, so the identities must match exactly; the admitted output
    # directory has to be empty again before the second run.
    shutil.rmtree(config.parent / "audit_out")
    second = audit_configuration_file(config)
    assert second.audit_json.read_bytes() == first_json
    assert second.audit_markdown.read_bytes() == first_md
    assert second.aggregate["audit_sha256"] == first.aggregate["audit_sha256"]
    validate_self_hash(second.aggregate, "audit_sha256")
    assert _tree_digest(config.parent / "model") == tree_before


def test_installed_cli_publishes_the_audit_surface(
    case: tuple[Path, str], capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    """The CLI publishes the audit surface and exits 0.

    Driven in process so the bound identity applies; the same command is run unpatched from a
    normal non-editable wheel install by the packaged native lane.
    """
    from metrifid.cli import main

    config, tree_before = case
    assert main(["audit-timestep", str(config)]) == 0
    captured = capsysbinary.readouterr()
    emitted = json.loads(captured.out.decode("utf-8"))
    assert Path(emitted["timestep_audit_json"]).is_file()
    assert Path(emitted["timestep_audit_markdown"]).is_file()
    aggregate = json.loads(Path(emitted["timestep_audit_json"]).read_text(encoding="utf-8"))
    assert aggregate["audit_sha256"] == emitted["audit_sha256"]
    validate_self_hash(aggregate, "audit_sha256")

    markdown = Path(emitted["timestep_audit_markdown"]).read_text(encoding="utf-8")
    assert "not asserted to be physically correct or a ground truth" in markdown
    for forbidden in ("safe timestep", "global equivalence", "speedup", "wall-clock"):
        assert forbidden not in markdown.lower()
    assert _tree_digest(config.parent / "model") == tree_before


def test_non_empty_output_directory_refuses_with_its_own_reason(
    case: tuple[Path, str], capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    """An expected refusal must not degrade to INTERNAL_INVARIANT_FAILED / exit 70."""
    from metrifid.cli import main

    config, _ = case
    audit_configuration_file(config)
    assert main(["audit-timestep", str(config)]) == 64
    failure = json.loads(capsysbinary.readouterr().err.decode("utf-8"))
    assert failure["operation"] == "audit-timestep"
    assert failure["reason"]["code"] == "OUTPUT_DIRECTORY_NOT_EMPTY"
    assert failure["reason"]["evidence"]["entry_count"] >= 1
    validate_self_hash(failure, "failure_sha256")


def test_source_tree_is_unchanged_after_a_top_level_failure(case: tuple[Path, str]) -> None:
    """A failure after the model root is read must still leave the user's bytes alone."""
    config, _ = case
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["candidate_step_dts"] = ["0.0005", "0.002"]  # below the reference timestep
    broken = config.parent / "timestep_audit_broken.json"
    broken.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    before = _tree_digest(config.parent / "model")
    with pytest.raises(AuditAbort) as exc:
        audit_configuration_file(broken)
    assert exc.value.error.failure.operation == "audit-timestep"
    assert _tree_digest(config.parent / "model") == before
    broken.unlink()


# --- bundled real-policy regression after finalization ----------------------------------------

BUNDLED_CASE_ENV = "METRIFID_ACCEPTANCE_CASE_ROOT"


def bundled_case_root() -> Path:
    """The packaged real-policy case supplied by the validation environment.

    The case is an external input, not a repository fixture, so it is located by environment
    variable. The audited behaviour itself is never relaxed when it is present.
    """
    raw = os.environ.get(BUNDLED_CASE_ENV)
    if not raw:
        pytest.skip(f"{BUNDLED_CASE_ENV} is not set; the bundled real-policy case is unavailable")
    root = Path(raw)
    if not (root / "timestep_audit.json").is_file():
        pytest.skip(f"{BUNDLED_CASE_ENV}={raw} does not contain timestep_audit.json")
    return root


@pytest.fixture
def bundled_case(tmp_path: Path) -> Path:
    """A writable copy of the packaged case; the packaged inputs stay untouched."""
    root = bundled_case_root()
    work = tmp_path / "real_policy"
    shutil.copytree(root, work)
    shutil.rmtree(work / "audit_out", ignore_errors=True)
    return work / "timestep_audit.json"


def test_bundled_real_policy_audit_reproduces_after_finalization(bundled_case: Path) -> None:
    """The finalization must not move the accepted real-policy classifications or evidence."""
    model_before = _tree_digest(bundled_case.parent / "model")
    result = audit_configuration_file(bundled_case)

    validate_self_hash(result.aggregate, "audit_sha256")
    rows = {row["token"]: row for row in result.aggregate["candidates"]}
    assert rows["dt_3_over_100"]["classification"] == REFUSED
    assert rows["dt_3_over_100"]["operational_reason"] == "CONTROL_GRID_NONINTEGRAL"
    assert rows["dt_3_over_100"]["steps_per_control_interval"] is None
    assert rows["dt_1_over_25"]["classification"] == OUTSIDE
    assert rows["dt_1_over_25"]["steps_per_control_interval"] == 1
    assert rows["dt_1_over_25"]["first_crossing"] is not None
    assert result.aggregate["recommendation"]["candidate_token"] is None

    markdown = result.audit_markdown.read_text(encoding="utf-8")
    assert "Operational reason" in markdown
    assert "First crossing" in markdown
    assert "CONTROL_GRID_NONINTEGRAL" in markdown
    for leaked in ("{'", "[{'", '"kind":', "ieee754_binary64"):
        assert leaked not in markdown
    assert result.audit_markdown.read_bytes() == markdown.encode("utf-8")
    assert b"\r" not in result.audit_markdown.read_bytes()

    assert _tree_digest(bundled_case.parent / "model") == model_before


# --- source-faithful reference and runtime candidate timestep ---------------------------------


def test_audit_binds_both_roles_to_the_exact_admitted_source_model(
    case: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every internal comparison must name the user's own model root for both roles.

    The seam is observed, not replaced: the spy records the arguments and then calls the real
    private function, so the audit still executes its normal product path.
    """
    from metrifid.compare import _orchestrator

    original = _orchestrator._compare_frozen_campaign_candidate
    recorded: list[tuple[dict[str, object], ExactRational]] = []

    def spy(**kwargs):  # type: ignore[no-untyped-def]
        # The internal configuration lives in a workspace the audit removes before it returns,
        # so it is read here, while the real product call is still in flight.
        """Construct the spy fixture used by timestep audit native scenarios.

        Deterministic setup isolates audit binds both roles to the exact admitted source model
        without bypassing the contract boundary under assertion.
        """
        payload = json.loads(kwargs["config_raw"])
        candidate_step_dt = kwargs["candidate_step_dt"]
        recorded.append((payload, candidate_step_dt))
        return original(**kwargs)

    monkeypatch.setattr(_orchestrator, "_compare_frozen_campaign_candidate", spy)

    config, tree_before = case
    source_root = (config.parent / "model").resolve()
    result = audit_configuration_file(config)

    # Exactly the declared candidates, in ascending order, each as its exact rational.
    assert [candidate_token(dt) for _, dt in recorded] == [
        "dt_1_over_500",
        "dt_1_over_250",
        "dt_1_over_200",
    ]
    assert [(dt.numerator, dt.denominator) for _, dt in recorded] == [(1, 500), (1, 250), (1, 200)]

    for payload, candidate_dt in recorded:
        assert Path(payload["baseline"]["model_root"]).resolve() == source_root
        assert Path(payload["candidate"]["model_root"]).resolve() == source_root
        assert (
            payload["baseline"]["entrypoint"] == payload["candidate"]["entrypoint"] == "robot.xml"
        )
        assert payload["candidate"]["declared_step_dt"] == candidate_dt.to_decimal_token()

    assert result.aggregate["inputs"]["source_model_tree_sha256"] == _tree_digest(source_root)
    assert _tree_digest(source_root) == tree_before


def test_audit_leaves_no_generated_model_or_copied_tree(case: tuple[Path, str]) -> None:
    """No rewritten XML, compiled MJB, model copy, or workspace may survive the audit."""
    config, tree_before = case
    audit_configuration_file(config)

    output_root = config.parent / "audit_out"
    generated = sorted(
        p.relative_to(output_root).as_posix()
        for p in output_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".xml", ".mjb"}
    )
    assert generated == []
    copied = sorted(
        p.relative_to(output_root).as_posix()
        for p in output_root.rglob("*")
        if p.is_dir() and p.name in {"reference", "candidate", ".audit_workspace"}
    )
    assert copied == []
    assert _tree_digest(config.parent / "model") == tree_before


def test_completed_receipts_share_one_closure_and_separate_only_the_timestep(
    case: tuple[Path, str],
) -> None:
    """Both roles are the same bytes; the only declared difference is the timestep."""
    config, _ = case
    result = audit_configuration_file(config)
    output_root = config.parent / "audit_out"

    completed = [
        row for row in result.aggregate["candidates"] if row["comparison_json"] is not None
    ]
    assert completed
    for row in completed:
        primitive = json.loads(
            (output_root / str(row["comparison_json"])).read_text(encoding="utf-8")
        )
        inputs = primitive["inputs"]
        assert inputs["baseline_model_closure_sha256"] == inputs["candidate_model_closure_sha256"]
        contract = primitive["comparison_contract"]
        assert (
            contract["baseline_model_closure_sha256"]
            == contract["candidate_model_closure_sha256"]
            == inputs["baseline_model_closure_sha256"]
        )
        assert primitive["model_closures"]["baseline"] == primitive["model_closures"]["candidate"]
        time_contract = primitive["time"]
        assert time_contract["baseline_step_dt"] == {"numerator": 1, "denominator": 1000}
        assert time_contract["baseline_step_dt"] != time_contract["candidate_step_dt"]
        assert time_contract["candidate_step_dt"] == row["step_dt"]


def test_audit_uses_one_immutable_model_and_workload_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every candidate reuses one live pair and workload even while live inputs differ."""
    from metrifid.compare import _orchestrator

    config = build_case(tmp_path, ["0.002", "0.005"])
    model_path = tmp_path / "model" / "robot.xml"
    state_path = tmp_path / "state.npz"
    actions_path = tmp_path / "actions.npz"
    originals = {path: path.read_bytes() for path in (model_path, state_path, actions_path)}
    original = _orchestrator._compare_frozen_campaign_candidate
    object_ids: list[tuple[int, int]] = []
    receipt_inputs: list[dict[str, object]] = []
    calls = 0

    def mutate_between_candidates(**kwargs):  # type: ignore[no-untyped-def]
        """Mutate live paths while the real candidate consumes frozen campaign objects."""
        nonlocal calls
        calls += 1
        object_ids.append((id(kwargs["live"]), id(kwargs["workload"])))
        if calls == 1:
            model_path.write_bytes(b"not live campaign XML")
            state_path.write_bytes(b"changed live state")
            actions_path.write_bytes(b"changed live actions")
        result = original(**kwargs)
        receipt_inputs.append(cast("dict[str, object]", result.receipt.to_primitive()["inputs"]))
        if calls == 2:
            for path, payload in originals.items():
                path.write_bytes(payload)
        return result

    monkeypatch.setattr(
        _orchestrator, "_compare_frozen_campaign_candidate", mutate_between_candidates
    )
    result = audit_configuration_file(config)

    assert calls == 2
    assert len(set(object_ids)) == 1
    identity_keys = (
        "baseline_model_closure_sha256",
        "candidate_model_closure_sha256",
        "initial_state_raw_sha256",
        "initial_state_semantic_sha256",
        "actions_raw_sha256",
        "actions_semantic_sha256",
    )
    assert len({tuple(item[key] for key in identity_keys) for item in receipt_inputs}) == 1
    aggregate_inputs = cast("dict[str, object]", result.aggregate["inputs"])
    for key in identity_keys[2:]:
        assert aggregate_inputs[key] == receipt_inputs[0][key]
    assert all(path.read_bytes() == payload for path, payload in originals.items())


def test_audit_refuses_final_live_input_mismatch_and_preserves_public_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a final workload mismatch while preserving completed candidate evidence.

    The mismatch occurs after both real candidates finish but before aggregate publication. The
    private workspace must be removed, the aggregate must remain absent, and each completed
    candidate's exact JSON/Markdown pair must remain available as public failure evidence.
    """
    from metrifid.compare import _orchestrator

    config = build_case(tmp_path, ["0.002", "0.005"])
    actions_path = tmp_path / "actions.npz"
    original = _orchestrator._compare_frozen_campaign_candidate
    calls = 0

    def mutate_after_last_candidate(**kwargs):  # type: ignore[no-untyped-def]
        """Change the live actions only after the last real candidate has completed."""
        nonlocal calls
        calls += 1
        result = original(**kwargs)
        if calls == 2:
            actions_path.write_bytes(b"final live mismatch")
        return result

    monkeypatch.setattr(
        _orchestrator, "_compare_frozen_campaign_candidate", mutate_after_last_candidate
    )
    with pytest.raises(AuditAbort) as caught:
        audit_configuration_file(config)

    assert caught.value.error.failure.reason.code is OperationalReasonCode.ACTIONS_ARTIFACT_INVALID
    output = tmp_path / "audit_out"
    assert output.is_dir()
    assert not (output / ".audit_workspace").exists()
    assert not (output / "timestep_audit.json").exists()
    assert not (output / "timestep_audit.md").exists()

    candidates = output / "candidates"
    expected_tokens = {
        candidate_token(ExactRational.from_decimal_token("0.002")),
        candidate_token(ExactRational.from_decimal_token("0.005")),
    }
    assert candidates.is_dir()
    assert {path.name for path in candidates.iterdir()} == expected_tokens
    for token in expected_tokens:
        candidate_dir = candidates / token
        assert {path.name for path in candidate_dir.iterdir()} == {
            "comparison.json",
            "comparison.md",
        }


def test_audit_output_does_not_follow_replaced_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aggregate publication and cleanup stay inside the retained audit directory."""
    from metrifid import _audit_execution

    config = build_case(tmp_path, ["0.002"])
    outside = tmp_path / "outside"
    outside.mkdir()
    displaced = tmp_path / "admitted-output"
    original = _audit_execution._publish_paired_results

    def replace_public_root(output, **kwargs):  # type: ignore[no-untyped-def]
        """Replace the public pathname at the final aggregate commit boundary."""
        output.path.rename(displaced)
        output.path.symlink_to(outside, target_is_directory=True)
        return original(output, **kwargs)

    monkeypatch.setattr(_audit_execution, "_publish_paired_results", replace_public_root)
    with pytest.raises(AuditAbort):
        audit_configuration_file(config)

    assert list(outside.iterdir()) == []
    assert displaced.is_dir()
    assert not (displaced / ".audit_workspace").exists()
    assert not (displaced / "timestep_audit.json").exists()
    assert not (displaced / "timestep_audit.md").exists()
    candidates = displaced / "candidates"
    token = candidate_token(ExactRational.from_decimal_token("0.002"))
    assert {path.name for path in displaced.iterdir()} == {"candidates"}
    assert {path.name for path in candidates.iterdir()} == {token}
    assert {path.name for path in (candidates / token).iterdir()} == {
        "comparison.json",
        "comparison.md",
    }


def test_exact_original_reference_repeats_are_complete_and_bit_identical(
    case: tuple[Path, str],
) -> None:
    """Independent fresh loads of the exact original model must replay identically."""
    config, _ = case
    result = audit_configuration_file(config)
    output_root = config.parent / "audit_out"

    completed = [
        row for row in result.aggregate["candidates"] if row["comparison_json"] is not None
    ]
    assert completed
    for row in completed:
        primitive = json.loads(
            (output_root / str(row["comparison_json"])).read_text(encoding="utf-8")
        )
        for role in ("baseline", "candidate"):
            repeats = primitive["repeatability"][role]
            assert repeats["repeat_count"] == 3
            assert repeats["complete_repeats"] == 3
            assert repeats["stable"] is True
            counts = repeats["captured_boundary_counts"]
            assert len(set(counts)) == 1
            numerical = primitive["numerical_evidence"][role]
            assert numerical["complete"] is True
            assert numerical["initial_state_preserved"] is True
            assert numerical["captured_boundary_count"] == numerical["expected_boundary_count"]


# --- conservative recommendation across candidate operational failures -------------------------

# All three land on the 0.01 control grid: 8, 5, and 4 substeps, and all three complete within
# tolerance on this fixture. Nothing here is refused by the grid, so any refusal in this case
# must come from the injected operational failure, and any block must come from it as well.
INTEGRAL_CANDIDATES = ["0.00125", "0.002", "0.0025"]


def failing_seam(
    monkeypatch: pytest.MonkeyPatch, token: str, code: OperationalReasonCode, role: str | None
):
    """Make exactly one candidate raise a real operational failure from the comparison engine.

    Only the engine call is replaced, and only for the named candidate. Every other candidate
    still runs the real product path, so the surrounding audit behaviour is genuinely exercised.
    """
    from metrifid.compare import _failure, _orchestrator

    original = _orchestrator._compare_frozen_campaign_candidate
    tool = OperationalToolObservation(
        CURRENT_VERSION, "VERIFIED_INSTALLED_DISTRIBUTION", DISTRIBUTION_DIGEST
    )

    def seam(**kwargs):  # type: ignore[no-untyped-def]
        """Construct the seam fixture used by timestep audit native scenarios.

        Deterministic setup isolates failing seam without bypassing the contract boundary under
        assertion.
        """
        if candidate_token(kwargs["candidate_step_dt"]) == token:
            raise _failure.operational_error(
                tool=tool,
                code=code,
                role=cast("ReasonRole", role),
                evidence={"message": "injected candidate failure"},
            )
        return original(**kwargs)

    monkeypatch.setattr(_orchestrator, "_compare_frozen_campaign_candidate", seam)


def test_internal_candidate_failure_blocks_recommendation_of_larger_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the sequence WITHIN, internal failure, WITHIN.

    The third candidate must still execute and stay visible, but the recommendation may not
    cross a candidate that produced no trustworthy comparison evidence.
    """
    config = build_case(tmp_path, INTEGRAL_CANDIDATES)
    failing_seam(
        monkeypatch, "dt_1_over_500", OperationalReasonCode.INTERNAL_INVARIANT_FAILED, None
    )

    result = audit_configuration_file(config)
    rows = {r["token"]: r for r in result.aggregate["candidates"]}
    assert list(rows) == ["dt_1_over_800", "dt_1_over_500", "dt_1_over_400"]

    failed = rows["dt_1_over_500"]
    assert failed["classification"] == INCONCLUSIVE
    assert failed["operational_reason"] == "INTERNAL_INVARIANT_FAILED"
    # The precomputed integral step count and factor are exact schedule facts and survive.
    assert failed["steps_per_control_interval"] == 5
    assert failed["reference_to_candidate_step_count_factor"] == {"numerator": 2, "denominator": 1}
    assert failed["failure_sha256"] is not None

    # The later candidate still ran, completed within tolerance, and is reported.
    assert rows["dt_1_over_400"]["classification"] == WITHIN
    assert rows["dt_1_over_400"]["receipt_sha256"] is not None

    # Without the correction this run recommends dt_1_over_400 with blocked false.
    recommendation = result.aggregate["recommendation"]
    assert recommendation["candidate_token"] == "dt_1_over_800"
    assert recommendation["blocked_by_prior_non_within"] is True
    validate_self_hash(result.aggregate, "audit_sha256")

    # The preserved failure artifact stays strict and self-consistent.
    failure_path = config.parent / "audit_out" / str(failed["operational_failure_json"])
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["operation"] == "compare"
    assert failure["reason"]["code"] == "INTERNAL_INVARIANT_FAILED"
    validate_self_hash(failure, "failure_sha256")


def test_first_candidate_failing_inconclusively_prevents_any_recommendation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises first candidate failing inconclusively prevents any recommendation;
    exact-grid evidence and completed-prefix rules must never promote an unqualified candidate.
    """
    config = build_case(tmp_path, INTEGRAL_CANDIDATES)
    failing_seam(
        monkeypatch,
        "dt_1_over_800",
        OperationalReasonCode.CANDIDATE_MODEL_COMPILE_ERROR,
        "candidate",
    )

    result = audit_configuration_file(config)
    rows = {r["token"]: r for r in result.aggregate["candidates"]}
    assert rows["dt_1_over_800"]["classification"] == INCONCLUSIVE
    assert rows["dt_1_over_400"]["classification"] == WITHIN
    recommendation = result.aggregate["recommendation"]
    assert recommendation["candidate_token"] is None
    assert recommendation["step_dt"] is None
    assert recommendation["blocked_by_prior_non_within"] is True


@pytest.mark.parametrize(
    ("label", "code", "role"),
    [
        ("internal invariant", OperationalReasonCode.INTERNAL_INVARIANT_FAILED, None),
        ("model compile", OperationalReasonCode.CANDIDATE_MODEL_COMPILE_ERROR, "candidate"),
        ("model closure refusal", OperationalReasonCode.MODEL_CLOSURE_MUTATED, "candidate"),
        ("workload artifact", OperationalReasonCode.ACTIONS_ARTIFACT_INVALID, None),
        ("identity refusal", OperationalReasonCode.JOINT_IDENTITY_MISSING, "candidate"),
        ("environment refusal", OperationalReasonCode.UNSUPPORTED_MUJOCO_VERSION, None),
        ("output failure", OperationalReasonCode.OUTPUT_WRITE_FAILED, None),
        # The nonintegral code alone is not enough: the role must be the candidate role.
        (
            "nonintegral with baseline role",
            OperationalReasonCode.CONTROL_GRID_NONINTEGRAL,
            "baseline",
        ),
    ],
)
def test_every_non_grid_candidate_failure_is_inconclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    code: OperationalReasonCode,
    role: str | None,
) -> None:
    """Only the exact candidate-grid refusal is skippable; everything else is inconclusive."""
    config = build_case(tmp_path, INTEGRAL_CANDIDATES)
    failing_seam(monkeypatch, "dt_1_over_500", code, role)

    result = audit_configuration_file(config)
    rows = {r["token"]: r for r in result.aggregate["candidates"]}
    failed = rows["dt_1_over_500"]
    assert failed["classification"] == INCONCLUSIVE, label
    assert failed["operational_reason"] == code.value, label
    assert failed["steps_per_control_interval"] == 5, label
    assert result.aggregate["recommendation"]["candidate_token"] == "dt_1_over_800", label
    assert result.aggregate["recommendation"]["blocked_by_prior_non_within"] is True, label


def test_candidate_role_nonintegral_grid_stays_a_skippable_refusal(
    case: tuple[Path, str],
) -> None:
    """The real, uninjected nonintegral candidate keeps its skippable REFUSED classification."""
    config, _ = case
    result = audit_configuration_file(config)
    rows = {r["token"]: r for r in result.aggregate["candidates"]}
    refused = rows["dt_1_over_250"]
    assert refused["classification"] == REFUSED
    assert refused["operational_reason"] == "CONTROL_GRID_NONINTEGRAL"
    assert refused["steps_per_control_interval"] is None
    assert refused["reference_to_candidate_step_count_factor"] is None

    # It did not break the prefix. On this fixture the largest candidate completes outside
    # tolerance, so the recommendation is compared against the same rows with the refusal
    # removed: an identical result proves the refusal itself contributed nothing.
    from metrifid.timestep_audit import _recommendation

    without_refusal = [r for r in result.aggregate["candidates"] if r["token"] != "dt_1_over_250"]
    assert result.aggregate["recommendation"] == _recommendation(without_refusal)
    assert result.aggregate["recommendation"]["candidate_token"] == "dt_1_over_500"
