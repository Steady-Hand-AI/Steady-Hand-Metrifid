"""No-clobber, partial-run, and completed-record tests for execution output ownership."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from metrifid._json_admission import (
    RECEIPT_JSON_LIMITS,
    JsonAdmissionError,
    bounded_strict_json_loads,
)
from metrifid.json_values import (
    CanonicalValue,
    canonical_json_bytes,
    canonical_sha256,
    compute_self_hash,
    strict_json_loads,
    validate_self_hash,
)
from metrifid.runtime_review import _execution_output as execution_output
from metrifid.runtime_review._execution_config import (
    RUN_CONFIG_SCHEMA,
    AdmittedRuntimeReviewRunConfiguration,
    load_runtime_review_run_configuration,
)
from metrifid.runtime_review._execution_output import (
    COMPLETED_RUN_RECORD_LOCATOR,
    RUN_RECORD_SCHEMA,
    RUNTIME_REVIEW_RUN_CLAIM_LIMITATIONS,
    OwnedRuntimeReviewRunOutput,
    RuntimeReviewRunOutputError,
    prepare_runtime_review_run_output,
    verify_completed_run_record,
    verify_owned_runtime_review_run_output,
)

_ROLES = ("baseline", "candidate")
_STEPS = ("0.004", "0.002", "0.001")
_REPEATS = (0, 1)


def _make_executable(path: Path, marker: str) -> Path:
    """Create one distinct test-local executable resource."""
    path.write_text(f"#!/bin/sh\n# {marker}\nexit 0\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def _admitted_configuration(tmp_path: Path) -> AdmittedRuntimeReviewRunConfiguration:
    """Create and admit one complete test-local run configuration."""
    baseline = _make_executable(tmp_path / "baseline-python", "baseline")
    candidate = _make_executable(tmp_path / "candidate-python", "candidate")
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    primitive = {
        "schema": RUN_CONFIG_SCHEMA,
        "schema_version": 1,
        "baseline_python": baseline.as_posix(),
        "candidate_python": candidate.as_posix(),
        "manifest": "manifest.json",
        "fixture_id": "smooth_pendulum",
        "output_dir": "run-output",
    }
    (tmp_path / "run.json").write_text(
        json.dumps(primitive, indent=2) + "\n",
        encoding="utf-8",
    )
    return load_runtime_review_run_configuration(tmp_path / "run.json")


def _process_command(resource_sha256: str) -> dict[str, CanonicalValue]:
    """Return one canonical process-command observation."""
    return {
        "argv": ["/explicit/python", "/installed/resource"],
        "environment": {"LANG": "C", "LC_ALL": "C"},
        "shell": False,
        "timeout_seconds": 300,
        "resource_sha256": resource_sha256,
    }


def _write_identity(path: Path, role: str, *, schema_version: int = 1) -> tuple[str, str]:
    """Write one compact legacy or role-based identity for output-boundary tests."""
    if schema_version == 1:
        identity: dict[str, CanonicalValue] = {
            "profile_id": "A_3.10.0" if role == "baseline" else "B_3.11.0",
            "profile_identity_sha256": None,
        }
    else:
        identity = {
            "profile_role": role,
            "package_version": "3.12.0",
            "native_version": "3.12.0",
            "native_version_integer": 3_012_000,
            "support_tier": "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE",
            "sentinel": {
                "status": "PASS",
                "sentinel_identity_sha256": "c" * 64,
            },
            "profile_identity_sha256": None,
        }
    identity["profile_identity_sha256"] = compute_self_hash(identity, "profile_identity_sha256")
    raw = canonical_json_bytes(identity) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), cast(str, identity["profile_identity_sha256"])


def _write_cell(
    path: Path,
    role: str,
    step: str,
    repeat: int,
    *,
    schema_version: int = 1,
    profile_identity_sha256: str | None = None,
    include_finite_manifest_decimal: bool = False,
) -> tuple[str, str]:
    """Write one exact six-member captured cell and return its result/manifest hashes."""
    path.mkdir()
    result: dict[str, CanonicalValue] = {
        "profile_role": role,
        "step_dt": step,
        "repeat_id": repeat,
    }
    if schema_version == 2:
        assert profile_identity_sha256 is not None
        result.update(
            {
                "schema": "metrifid.native_upgrade_worker_result",
                "schema_version": 2,
                "status": "COMPLETED",
                "package_version": "3.12.0",
                "native_version": "3.12.0",
                "native_version_integer": 3_012_000,
                "profile_identity_sha256": profile_identity_sha256,
                "runtime_identity_sha256": "b" * 64,
                "sentinel_identity_sha256": "c" * 64,
            }
        )
        if include_finite_manifest_decimal:
            result["manifest_echo"] = {"fixtures": [{"scale": "__FINITE_DECIMAL_TOKEN__"}]}
    result_raw = canonical_json_bytes(result)
    if include_finite_manifest_decimal:
        result_raw = result_raw.replace(b'"__FINITE_DECIMAL_TOKEN__"', b"3.141592653589793")
    payloads = {
        "fixture.xml": b"<mujoco/>\n",
        "input_manifest.json": b"{}\n",
        "model.mjb": f"model:{role}:{step}:{repeat}".encode("ascii"),
        "result.json": result_raw + b"\n",
        "trace.npz": f"trace:{role}:{step}:{repeat}".encode("ascii"),
    }
    for name, payload in payloads.items():
        (path / name).write_bytes(payload)
    checksums = "".join(
        f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n" for name in sorted(payloads)
    ).encode("ascii")
    (path / "CHECKSUMS.sha256").write_bytes(checksums)
    return (
        hashlib.sha256(payloads["result.json"]).hexdigest(),
        hashlib.sha256(checksums).hexdigest(),
    )


def _install_test_replayers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace scientific schema replay only; output byte and locator replay remains real."""

    def load_identity(
        path: Path, *, expected_profile_id: str, expected_worker_sha256: str
    ) -> dict[str, Any]:
        """Load the compact test identity while checking role and worker inputs."""
        assert expected_worker_sha256 == execution_output.LEGACY_FROZEN_EVIDENCE_WORKER_SHA256
        value = strict_json_loads(path.read_bytes())
        assert isinstance(value, dict)
        assert value["profile_id"] == expected_profile_id
        validate_self_hash(value, "profile_identity_sha256")
        return value

    def load_receipt(path: Path) -> dict[str, CanonicalValue]:
        """Load the compact test receipt used to exercise operational reference replay."""
        value = strict_json_loads(path.read_bytes())
        assert isinstance(value, dict)
        validate_self_hash(value, "receipt_sha256")
        return value

    def load_role_identity(
        path: Path,
        *,
        expected_profile_role: str,
        expected_profile_identity_sha256: str,
        expected_worker_sha256: str,
    ) -> dict[str, Any]:
        """Load the compact role identity while checking all replay inputs."""
        assert expected_worker_sha256 == execution_output.FROZEN_EVIDENCE_WORKER_SHA256
        value = strict_json_loads(path.read_bytes())
        assert isinstance(value, dict)
        assert value["profile_role"] == expected_profile_role
        assert value["profile_identity_sha256"] == expected_profile_identity_sha256
        validate_self_hash(value, "profile_identity_sha256")
        return value

    monkeypatch.setattr(execution_output, "load_native_profile_identity", load_identity)
    monkeypatch.setattr(execution_output, "load_native_profile_identity_v2", load_role_identity)
    monkeypatch.setattr(execution_output, "load_and_validate_runtime_review_receipt", load_receipt)


def _populate_completed_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    schema_version: int = 1,
    include_finite_manifest_decimal: bool = False,
) -> tuple[OwnedRuntimeReviewRunOutput, dict[str, CanonicalValue]]:
    """Populate one legacy or role-based completed-run tree before final publication."""
    _install_test_replayers(monkeypatch)
    admitted = _admitted_configuration(tmp_path)
    output = prepare_runtime_review_run_output(admitted)
    worker = tmp_path / "installed-worker.txt"
    worker.write_bytes(b"frozen worker resource")
    worker_sha256 = hashlib.sha256(worker.read_bytes()).hexdigest()
    worker_constant = (
        "LEGACY_FROZEN_EVIDENCE_WORKER_SHA256"
        if schema_version == 1
        else "FROZEN_EVIDENCE_WORKER_SHA256"
    )
    monkeypatch.setattr(execution_output, worker_constant, worker_sha256)
    collector = tmp_path / "profile-collector.py"
    collector.write_bytes(b"collector resource")
    collector_sha256 = hashlib.sha256(collector.read_bytes()).hexdigest()

    preflights: list[CanonicalValue] = []
    attempts: list[CanonicalValue] = []
    for role in _ROLES:
        interpreter = (
            admitted.baseline_interpreter if role == "baseline" else admitted.candidate_interpreter
        )
        identity_path = output.new_profile_identity_path(role)
        identity_file_sha256, profile_identity_sha256 = _write_identity(
            identity_path, role, schema_version=schema_version
        )
        retained = output.write_profile_preflight(
            role,
            command=_process_command(collector_sha256),
            stdout=b"profile ready\n",
            stderr=b"",
            exit_code=0,
        )
        preflight: dict[str, CanonicalValue] = {
            "role": role,
            "lexical_interpreter": interpreter.lexical_path.as_posix(),
            "resolved_interpreter": interpreter.resolved_path.as_posix(),
            "resolved_executable_sha256": interpreter.resolved_sha256,
            "identity_locator": f"profile_identities/{role}.json",
            "identity_file_sha256": identity_file_sha256,
            "profile_identity_sha256": profile_identity_sha256,
            **retained.to_primitive(),
        }
        if schema_version == 1:
            preflight["profile_id"] = "A_3.10.0" if role == "baseline" else "B_3.11.0"
        else:
            preflight.update(
                {
                    "package_version": "3.12.0",
                    "native_version": "3.12.0",
                    "native_version_integer": 3_012_000,
                    "support_tier": "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE",
                    "sentinel_identity_sha256": "c" * 64,
                }
            )
        preflights.append(preflight)

        for step in _STEPS:
            for repeat in _REPEATS:
                cell_path = output.new_evidence_cell_path(role, step, repeat)
                result_sha256, checksum_sha256 = _write_cell(
                    cell_path,
                    role,
                    step,
                    repeat,
                    schema_version=schema_version,
                    profile_identity_sha256=profile_identity_sha256,
                    include_finite_manifest_decimal=include_finite_manifest_decimal,
                )
                process = output.write_evidence_attempt(
                    role,
                    step,
                    repeat,
                    command=_process_command(worker_sha256),
                    stdout=b"cell ready\n",
                    stderr=b"",
                    exit_code=0,
                )
                attempt: dict[str, CanonicalValue] = {
                    "role": role,
                    "step_dt": step,
                    "repeat_id": repeat,
                    "lexical_interpreter": interpreter.lexical_path.as_posix(),
                    "resolved_interpreter": interpreter.resolved_path.as_posix(),
                    "resolved_executable_sha256": interpreter.resolved_sha256,
                    "cell_locator": (
                        f"captured_evidence/{role}/{step.replace('.', 'p')}/repeat_{repeat}"
                    ),
                    "result_sha256": result_sha256,
                    "checksum_manifest_sha256": checksum_sha256,
                    **process.to_primitive(),
                }
                if schema_version == 1:
                    attempt.update(
                        {
                            "profile_id": ("A_3.10.0" if role == "baseline" else "B_3.11.0"),
                            "mujoco_version": "3.10.0" if role == "baseline" else "3.11.0",
                        }
                    )
                else:
                    attempt.update(
                        {
                            "package_version": "3.12.0",
                            "native_version": "3.12.0",
                            "native_version_integer": 3_012_000,
                            "profile_identity_sha256": profile_identity_sha256,
                            "runtime_identity_sha256": "b" * 64,
                            "sentinel_identity_sha256": "c" * 64,
                        }
                    )
                attempts.append(attempt)

    generated_primitive: dict[str, CanonicalValue] = {
        "schema": "metrifid.runtime_review_config",
        "schema_version": schema_version,
    }
    generated_raw = canonical_json_bytes(generated_primitive) + b"\n"
    generated = output.write_generated_runtime_review_configuration(generated_raw)
    receipt: dict[str, CanonicalValue] = {
        "status": "WITHIN_DECLARED_MIGRATION_ENVELOPE",
        "reason_code": None,
        "configuration": {
            "raw_sha256": hashlib.sha256(generated_raw).hexdigest(),
            "semantic_sha256": canonical_sha256(generated_primitive),
        },
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = compute_self_hash(receipt, "receipt_sha256")
    receipt_raw = canonical_json_bytes(receipt) + b"\n"
    receipt_path = output.root / "decision" / "runtime_review" / "runtime_review.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(receipt_raw)

    record: dict[str, CanonicalValue] = {
        "schema": RUN_RECORD_SCHEMA,
        "schema_version": schema_version,
        "status": receipt["status"],
        "reason_code": receipt["reason_code"],
        "exit_code": 0,
        "input_configuration": {
            "locator": "admitted_runtime_review_run_config.json",
            "raw_sha256": admitted.raw_sha256,
            "semantic_sha256": admitted.semantic_sha256,
        },
        "packaged_worker": {"locator": worker.as_posix(), "sha256": worker_sha256},
        "profile_identity_collector": {
            "locator": collector.as_posix(),
            "sha256": collector_sha256,
        },
        "profile_preflights": preflights,
        "evidence_attempts": attempts,
        "generated_runtime_review_config": generated.to_primitive(),
        "runtime_review_receipt": {
            "locator": "decision/runtime_review/runtime_review.json",
            "file_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "receipt_sha256": receipt["receipt_sha256"],
        },
        "claim_limitations": list(RUNTIME_REVIEW_RUN_CLAIM_LIMITATIONS),
        "run_sha256": None,
    }
    generated_record = cast(dict[str, CanonicalValue], record["generated_runtime_review_config"])
    generated_record.pop("size_bytes")
    return output, record


def test_partial_run_never_publishes_completed_record(tmp_path: Path) -> None:
    """A retained bounded refusal remains diagnostic and never gains a completion filename."""
    admitted = _admitted_configuration(tmp_path)
    with prepare_runtime_review_run_output(admitted) as output:
        retained = output.write_profile_preflight(
            "baseline",
            command=_process_command("a" * 64),
            stdout=b"",
            stderr=b"bounded refusal\n",
            exit_code=2,
        )
        assert retained.exit_code == 2
        with pytest.raises(RuntimeReviewRunOutputError, match="partial run"):
            verify_owned_runtime_review_run_output(output)
    assert output.root.is_dir()
    assert not (output.root / COMPLETED_RUN_RECORD_LOCATOR).exists()


def test_completed_run_record_is_self_hashed_and_replayable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete tree publishes last and independently replays every retained byte identity."""
    output, record = _populate_completed_output(tmp_path, monkeypatch)
    with output:
        published = output.publish_completed_run_record(record)
        validate_self_hash(published.document, "run_sha256")
        replayed = verify_owned_runtime_review_run_output(output)
        assert replayed.run_sha256 == published.run_sha256
        assert replayed.file_sha256 == published.file_sha256
        assert not (output.root / ".runtime_review_run.json.staging").exists()
    standalone = verify_completed_run_record(output.completed_run_record, output_root=output.root)
    assert standalone.run_sha256 == published.run_sha256


def test_role_based_completed_run_record_replays_every_identity_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay one v2 run through exact profile, runtime, sentinel, and cell bindings."""
    output, record = _populate_completed_output(tmp_path, monkeypatch, schema_version=2)

    with output:
        published = output.publish_completed_run_record(record)
        replayed = verify_owned_runtime_review_run_output(output)

    assert published.document["schema_version"] == 2
    assert replayed.run_sha256 == published.run_sha256


def test_role_based_completed_run_replays_finite_manifest_echo_decimals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finite manifest decimals remain admissible at the role-based result replay boundary."""
    output, record = _populate_completed_output(
        tmp_path,
        monkeypatch,
        schema_version=2,
        include_finite_manifest_decimal=True,
    )
    first_result = (
        output.root / "captured_evidence" / "baseline" / "0p004" / "repeat_0" / "result.json"
    )
    result_raw = first_result.read_bytes()
    assert b'"scale":3.141592653589793' in result_raw
    parsed_result = json.loads(result_raw)
    assert parsed_result["manifest_echo"]["fixtures"][0]["scale"] == 3.141592653589793
    with pytest.raises(JsonAdmissionError, match="raw JSON floating-point token"):
        bounded_strict_json_loads(
            b'{"schema":"metrifid.runtime_review_run","schema_version":2,'
            b'"scale":3.141592653589793}',
            RECEIPT_JSON_LIMITS,
        )

    with output:
        published = output.publish_completed_run_record(record)
        replayed = verify_owned_runtime_review_run_output(output)
    standalone = verify_completed_run_record(output.completed_run_record, output_root=output.root)

    assert replayed.run_sha256 == published.run_sha256
    assert standalone.run_sha256 == published.run_sha256


def test_role_based_completed_run_rejects_a_resealed_runtime_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a run-record edit that substitutes one cell runtime identity."""
    output, record = _populate_completed_output(tmp_path, monkeypatch, schema_version=2)
    attempts = cast(list[CanonicalValue], record["evidence_attempts"])
    first = cast(dict[str, CanonicalValue], attempts[0])
    first["runtime_identity_sha256"] = "d" * 64

    with output, pytest.raises(RuntimeReviewRunOutputError, match="v2 result binding"):
        output.publish_completed_run_record(record)
    assert not output.completed_run_record.exists()


def test_bad_completed_reference_never_publishes_completed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reference replay happens before publication, so an invalid worker binding stays partial."""
    output, record = _populate_completed_output(tmp_path, monkeypatch)
    worker = cast(dict[str, CanonicalValue], record["packaged_worker"])
    worker["sha256"] = "0" * 64

    with output, pytest.raises(RuntimeReviewRunOutputError, match="frozen worker"):
        output.publish_completed_run_record(record)
    assert not output.completed_run_record.exists()


def test_changed_generated_configuration_hash_prevents_completed_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed generated-configuration digest is rejected before completion publishes."""
    output, record = _populate_completed_output(tmp_path, monkeypatch)
    generated = cast(dict[str, CanonicalValue], record["generated_runtime_review_config"])
    generated["sha256"] = hashlib.sha256(b"changed generated configuration reference").hexdigest()

    with (
        output,
        pytest.raises(RuntimeReviewRunOutputError, match="referenced output bytes changed"),
    ):
        output.publish_completed_run_record(record)
    assert not output.completed_run_record.exists()


def test_changed_final_receipt_hash_prevents_completed_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed final-receipt self-hash reference is rejected before completion publishes."""
    output, record = _populate_completed_output(tmp_path, monkeypatch)
    receipt = cast(dict[str, CanonicalValue], record["runtime_review_receipt"])
    receipt["receipt_sha256"] = hashlib.sha256(b"changed final receipt reference").hexdigest()

    with output, pytest.raises(RuntimeReviewRunOutputError, match="receipt self-hash changed"):
        output.publish_completed_run_record(record)
    assert not output.completed_run_record.exists()


def test_completed_record_detects_retained_attempt_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing an attempt stream after publication is detected before a result is returned."""
    output, record = _populate_completed_output(tmp_path, monkeypatch)
    with output:
        output.publish_completed_run_record(record)
        stream = output.root / "attempts" / "baseline" / "0p004" / "repeat_0" / "stdout.txt"
        stream.write_bytes(b"substituted\n")
        with pytest.raises(RuntimeReviewRunOutputError, match="referenced bytes changed"):
            output.verify_completed_output()


def test_cell_member_mutation_prevents_completed_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All five checksum-bound cell members are rehashed before completion becomes visible."""
    output, record = _populate_completed_output(tmp_path, monkeypatch)
    trace = output.root / "captured_evidence" / "baseline" / "0p004" / "repeat_0" / "trace.npz"
    trace.write_bytes(b"changed trace")

    with output, pytest.raises(RuntimeReviewRunOutputError, match="checksum mismatch"):
        output.publish_completed_run_record(record)
    assert not output.completed_run_record.exists()


def test_input_mutation_prevents_completed_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final gate rechecks admitted input bytes immediately before atomic publication."""
    output, record = _populate_completed_output(tmp_path, monkeypatch)
    (tmp_path / "manifest.json").write_text('{"changed":true}\n', encoding="utf-8")

    with output, pytest.raises(RuntimeReviewRunOutputError, match="inputs changed"):
        output.publish_completed_run_record(record)
    assert not output.completed_run_record.exists()


def test_output_root_and_process_attempts_never_clobber(tmp_path: Path) -> None:
    """Root collisions and duplicate semantic process slots preserve the first caller bytes."""
    admitted = _admitted_configuration(tmp_path)
    output = prepare_runtime_review_run_output(admitted)
    with output:
        first = output.write_evidence_attempt(
            "baseline",
            "0.004",
            0,
            command=_process_command("b" * 64),
            stdout=b"first",
            stderr=b"",
            exit_code=0,
        )
        with pytest.raises(RuntimeReviewRunOutputError, match="already exists"):
            output.write_evidence_attempt(
                "baseline",
                "0.004",
                0,
                command=_process_command("b" * 64),
                stdout=b"second",
                stderr=b"",
                exit_code=0,
            )
        assert first.stdout.path.read_bytes() == b"first"
    with pytest.raises(RuntimeReviewRunOutputError, match="already exists"):
        prepare_runtime_review_run_output(admitted)


def test_timeout_is_retained_without_fabricating_a_process_exit(tmp_path: Path) -> None:
    """A timeout records its no-exit state literally and remains an incomplete partial run."""
    admitted = _admitted_configuration(tmp_path)
    with prepare_runtime_review_run_output(admitted) as output:
        retained = output.write_profile_preflight(
            "baseline",
            command=_process_command("c" * 64),
            stdout=b"",
            stderr=b"",
            exit_code=None,
            no_exit_status="TIMEOUT",
        )
        assert retained.exit_code is None
        assert retained.exit_code_file.path.read_bytes() == b"TIMEOUT\n"
        assert not output.completed_run_record.exists()


def test_generated_configuration_and_external_paths_are_no_clobber(tmp_path: Path) -> None:
    """Generated config, profile identity, and evidence cell locators can be allocated only once."""
    admitted = _admitted_configuration(tmp_path)
    with prepare_runtime_review_run_output(admitted) as output:
        raw = canonical_json_bytes({"schema": "generated"}) + b"\n"
        output.write_generated_runtime_review_configuration(raw)
        with pytest.raises(RuntimeReviewRunOutputError, match="could not be created"):
            output.write_generated_runtime_review_configuration(raw)

        identity = output.new_profile_identity_path("baseline")
        identity.write_bytes(b"caller-created")
        with pytest.raises(RuntimeReviewRunOutputError, match="already exists"):
            output.new_profile_identity_path("baseline")

        cell = output.new_evidence_cell_path("candidate", "0.001", 1)
        cell.mkdir()
        with pytest.raises(RuntimeReviewRunOutputError, match="already exists"):
            output.new_evidence_cell_path("candidate", "0.001", 1)


def test_impossible_status_reason_reseal_is_refused_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resealed operational record cannot attach an evidence reason to a non-evidence status."""
    output, record = _populate_completed_output(tmp_path, monkeypatch)
    record["reason_code"] = "REPEATABILITY_FAILED"
    record["run_sha256"] = None

    with output, pytest.raises(RuntimeReviewRunOutputError, match="present exactly"):
        output.publish_completed_run_record(record)
    assert not output.completed_run_record.exists()


def test_unknown_attempt_record_field_is_refused_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resealed attempt cannot smuggle a hidden retry or any other undeclared fact."""
    output, record = _populate_completed_output(tmp_path, monkeypatch)
    attempts = cast(list[dict[str, CanonicalValue]], record["evidence_attempts"])
    attempts[0]["hidden_retry_count"] = 7
    record["run_sha256"] = None

    with output, pytest.raises(RuntimeReviewRunOutputError, match="field set is not closed"):
        output.publish_completed_run_record(record)
    assert not output.completed_run_record.exists()


def test_successful_attempt_cannot_claim_a_timeout_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed attempt cannot pair an exit-zero observation with a no-exit claim."""
    output, record = _populate_completed_output(tmp_path, monkeypatch)
    attempts = cast(list[dict[str, CanonicalValue]], record["evidence_attempts"])
    attempts[0]["no_exit_status"] = "TIMEOUT"
    record["run_sha256"] = None

    with output, pytest.raises(RuntimeReviewRunOutputError, match="must be null"):
        output.publish_completed_run_record(record)
    assert not output.completed_run_record.exists()


def test_frozen_worker_digest_is_exact() -> None:
    """The verifier separately pins immutable legacy and current worker resources."""
    assert execution_output.LEGACY_FROZEN_EVIDENCE_WORKER_SHA256 == (
        "941cc0cba66632901e89ee0a5be63575a2a5635dc98595e10271d9bed003dd6f"
    )
    assert execution_output.FROZEN_EVIDENCE_WORKER_SHA256 == (
        "b00e509a344593806c088c4e49783ed71bacd815466d74bce9e27c931535b4ff"
    )
