"""Filesystem-bound tests for strict six-member runtime evidence admission."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest

from metrifid.json_values import canonical_sha256
from metrifid.runtime_review import _evidence as evidence_module
from metrifid.runtime_review._config import (
    AdmittedRuntimeReviewConfiguration,
    AdmittedRuntimeReviewConfigurationV2,
    RuntimeReviewCellConfig,
)
from metrifid.runtime_review._evidence import (
    _EXPECTED_OBSERVATION_TIME_TOKENS,
    _LIMITATIONS,
    AdmittedEvidenceCell,
    RuntimeEvidenceAdmissionError,
    _admit_contact,
    _admit_diagnostics,
    _admit_runtime_v2,
    _bind_cross_cell_evidence,
    _canonical_trace_sha256,
    _common_runtime_projection,
    _read_verified_members,
    _select_manifest_fixture,
    _strict_json_bytes,
    _strict_manifest_echo_json_bytes,
    _strict_manifest_json_bytes,
    _validate_fixture_contract,
    _validate_observation_clock,
    _validate_result_profile_identity,
    _validate_result_root,
    _validate_runtime_numpy,
    _validate_self_contained_fixture_xml,
)
from metrifid.runtime_review._native_profile_identity import _FROZEN_WORKER_SHA256

_PAYLOADS = {
    "fixture.xml": b"<mujoco/>\n",
    "input_manifest.json": b"{}\n",
    "model.mjb": b"model",
    "result.json": b"{}\n",
    "trace.npz": b"trace",
}
_FIXTURE_MANIFEST = Path(__file__).parents[2] / "fixtures/native_upgrade/manifest.json"


def _smooth_fixture() -> dict[str, object]:
    """Return a defensive exact-decimal projection of the retained smooth fixture."""
    manifest = _strict_manifest_json_bytes(_FIXTURE_MANIFEST.read_bytes())
    return copy.deepcopy(_select_manifest_fixture(manifest, "smooth_pendulum"))


def _valid_diagnostics() -> dict[str, object]:
    """Build the worker's cumulative 51-sample successful diagnostic projection."""
    samples: list[object] = [
        {
            "time": token,
            "finite_values": True,
            "warnings_passed": True,
            "solver_converged": True,
            "max_solver_iterations": 1,
            "max_solver_residual": "0",
        }
        for token in _EXPECTED_OBSERVATION_TIME_TOKENS
    ]
    return {
        "finite_values": True,
        "warnings_passed": True,
        "solver_converged": True,
        "warning_records": [],
        "max_solver_iterations": 1,
        "solver_iteration_limit": 100,
        "max_solver_residual": "0",
        "raw_contact_order_sha256": "0" * 64,
        "samples": samples,
    }


def _cross_cell_campaign() -> tuple[AdmittedRuntimeReviewConfiguration, list[SimpleNamespace]]:
    """Build twelve duck-typed cells for isolated cross-cell identity binding tests."""
    configuration = cast(
        AdmittedRuntimeReviewConfiguration,
        SimpleNamespace(
            config=SimpleNamespace(expected_subject=SimpleNamespace(fixture_id="subject"))
        ),
    )
    runtime: dict[str, object] = {
        "python": {
            "cache_tag": "cpython-test",
            "compiler": "test-compiler",
            "implementation": "CPython",
            "implementation_name": "cpython",
            "resolved_executable": "/exact/python",
            "resolved_executable_sha256": "1" * 64,
            "version": "3.12.0",
            "version_full": "3.12.0 test",
        },
        "numpy": {
            "distribution": {
                "name": "numpy",
                "version": "2.3.5",
                "record_bound_member_count": 1,
                "record_bound_payload_sha256": "2" * 64,
            }
        },
        "host": {"system": "test-host"},
        "thread_environment": {"OMP_NUM_THREADS": "1"},
    }
    cells: list[SimpleNamespace] = []
    for role in ("baseline", "candidate"):
        for step_dt in ("0.004", "0.002", "0.001"):
            for repeat_id in (0, 1):
                cells.append(
                    SimpleNamespace(
                        profile_role=role,
                        step_dt=step_dt,
                        repeat_id=repeat_id,
                        fixture_id="subject",
                        workload={"semantic_sha256": "3" * 64},
                        subject={
                            "source_closure": {"closure_sha256": "4" * 64},
                            "compiled_mjb_sha256": "5" * 64,
                            "compiled_mjb_size_bytes": 1,
                        },
                        channels=(
                            SimpleNamespace(
                                to_primitive=Mock(
                                    return_value={
                                        "channel_id": "joint.position",
                                        "tolerance": "0.000001",
                                    }
                                )
                            ),
                        ),
                        observation_time_tokens=_EXPECTED_OBSERVATION_TIME_TOKENS,
                        contact_event_time_tolerance="0.02",
                        runtime=copy.deepcopy(runtime),
                    )
                )
    return configuration, cells


def _strict_manifest_echo_with_label(payload: bytes, _label: str) -> dict[str, object]:
    """Adapt the binary64 manifest reading to the common labeled-loader test signature."""
    return _strict_manifest_echo_json_bytes(payload)


def _strict_manifest_with_label(payload: bytes, _label: str) -> dict[str, object]:
    """Adapt exact manifest admission to the common labeled-loader test signature."""
    return _strict_manifest_json_bytes(payload)


def _write_checksum_bound_cell(directory: Path) -> None:
    """Create one minimal exact-member cell for checksum-layer isolation."""
    directory.mkdir()
    for name, payload in _PAYLOADS.items():
        (directory / name).write_bytes(payload)
    checksums = "".join(
        f"{hashlib.sha256(_PAYLOADS[name]).hexdigest()}  {name}\n" for name in sorted(_PAYLOADS)
    )
    (directory / "CHECKSUMS.sha256").write_text(checksums, encoding="ascii")


def _full_distribution(name: str, version: str, digest: str) -> dict[str, object]:
    """Build one full worker distribution with independently reproducible hashes."""
    member = {
        "logical_path": f"{name}/module.py",
        "sha256": digest,
        "size_bytes": 10,
        "record_hash_mode": "sha256",
        "record_hash_value": "declared-record-value",
    }
    payload = [{key: member[key] for key in ("logical_path", "sha256", "size_bytes")}]
    return {
        "member_count": 1,
        "members": [member],
        "name": name,
        "payload_identity_algorithm": "sha256(canonical-json(payload))",
        "payload_sha256": canonical_sha256(cast(object, payload)),
        "record_bound_identity_algorithm": "sha256(canonical-json(record-bound))",
        "record_bound_member_count": 1,
        "record_bound_payload_sha256": canonical_sha256(cast(object, payload)),
        "record_declared_sha256_member_count": 1,
        "record_unhashed_member_count": 0,
        "version": version,
    }


def _compact_distribution(distribution: dict[str, object]) -> dict[str, object]:
    """Project one full test distribution to the preflight identity shape."""
    return {key: value for key, value in distribution.items() if key != "members"}


def _role_runtime_fixture() -> tuple[object, dict[str, object], dict[str, object]]:
    """Build a configured role, exact runtime, and matching preflight identity."""
    profile_hash = "a" * 64
    sentinel_hash = "b" * 64
    raw_hash = "c" * 64
    package_version = "3.12.0+vendor.1"
    native_version = "3.12.0"
    native_integer = 3_012_000
    mujoco_distribution = _full_distribution("mujoco", package_version, "d" * 64)
    numpy_distribution = _full_distribution("numpy", "2.4.1", "e" * 64)
    python = {
        "executable": "/profiles/baseline/bin/python",
        "resolved_executable": "/opt/python/bin/python",
        "resolved_executable_sha256": "f" * 64,
        "version": "3.12.13",
        "version_full": "3.12.13 test-build",
        "implementation": "CPython",
        "implementation_name": "cpython",
        "compiler": "test-compiler",
        "cache_tag": "cpython-312",
    }
    host = {
        "system": "TestOS",
        "release": "test-release",
        "version": "test-kernel",
        "platform": "test-platform",
        "machine": "test-machine",
        "architecture": ["64bit", ""],
        "libc": ["glibc", "2.39"],
        "logical_cpu_count": 4,
        "cpu_model": "test-cpu",
        "cpu_model_source": "test-source",
        "hardware_model": "test-hardware",
        "hardware_profile": {},
        "physical_cpu_count": 2,
        "hyper_threading_technology": None,
    }
    loaded_library = {
        "filename": "libmujoco.so.3.12.0",
        "loaded_path": "/profiles/baseline/libmujoco.so.3.12.0",
        "resolved_path": "/profiles/baseline/libmujoco.so.3.12.0",
        "size_bytes": 4096,
        "sha256": "1" * 64,
    }
    installation = {"available": False, "distribution": None}
    pip_check = {
        "argv": [python["executable"], "-m", "pip", "check"],
        "exit_code": 0,
        "stdout": "No broken requirements found.\n",
        "stderr": "",
    }
    runtime: dict[str, object] = {
        "schema": "metrifid.native_upgrade_runtime_identity",
        "schema_version": 2,
        "profile_role": "baseline",
        "package_version": package_version,
        "native_version": native_version,
        "native_version_integer": native_integer,
        "support_tier": "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE",
        "python": python,
        "host": host,
        "thread_environment": dict(evidence_module._THREAD_ENVIRONMENT),
        "mujoco": {
            "package_version": package_version,
            "native_version": native_version,
            "native_version_integer": native_integer,
            "distribution": mujoco_distribution,
            "loaded_native_library": loaded_library,
        },
        "numpy": {"python_version": "2.4.1", "distribution": numpy_distribution},
        "installation": installation,
        "pip_check": pip_check,
        "external_profile_identity": {
            "available": True,
            "raw_sha256": raw_hash,
            "profile_identity_sha256": profile_hash,
        },
        "worker_sha256": _FROZEN_WORKER_SHA256,
        "profile_identity_sha256": profile_hash,
        "sentinel_identity_sha256": sentinel_hash,
    }
    runtime["runtime_identity_sha256"] = canonical_sha256(cast(object, runtime))
    identity: dict[str, object] = {
        "profile_role": "baseline",
        "package_version": package_version,
        "native_version": native_version,
        "native_version_integer": native_integer,
        "support_tier": "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE",
        "profile_identity_sha256": profile_hash,
        "profile_contract": {"worker_sha256": _FROZEN_WORKER_SHA256},
        "python": {**python, "build": ["main", "build-date"]},
        "host": host,
        "environment": dict(evidence_module._THREAD_ENVIRONMENT),
        "mujoco": {
            "distribution": _compact_distribution(mujoco_distribution),
            "loaded_native_library": loaded_library,
        },
        "numpy": {
            "python_version": "2.4.1",
            "distribution": _compact_distribution(numpy_distribution),
        },
        "installation": installation,
        "pip_check": pip_check,
        "sentinel": {"status": "PASS", "sentinel_identity_sha256": sentinel_hash},
    }
    declaration = SimpleNamespace(
        profile_role="baseline",
        package_version=package_version,
        native_version=native_version,
        native_version_integer=native_integer,
        profile_identity_sha256=profile_hash,
    )
    configuration = SimpleNamespace(
        config=SimpleNamespace(baseline_profile=declaration, candidate_profile=declaration),
        profile_identity_path=lambda _role: Path("/profiles/baseline.json"),
        profile_identity_file_hash=lambda _role: raw_hash,
    )
    return configuration, runtime, identity


def test_checksum_layer_admits_exact_six_regular_members(tmp_path: Path) -> None:
    """All six regular files are measured only after the five-member manifest validates."""
    cell = tmp_path / "cell"
    _write_checksum_bound_cell(cell)

    members, raw = _read_verified_members(cell)

    assert [member.name for member in members] == sorted({"CHECKSUMS.sha256", *_PAYLOADS})
    assert raw["fixture.xml"] == _PAYLOADS["fixture.xml"]


def test_checksum_layer_refuses_an_extra_member(tmp_path: Path) -> None:
    """An undeclared file cannot enter a decision-bearing evidence directory."""
    cell = tmp_path / "cell"
    _write_checksum_bound_cell(cell)
    (cell / "extra.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeEvidenceAdmissionError, match="member mismatch"):
        _read_verified_members(cell)


def test_checksum_layer_refuses_a_symlinked_member(tmp_path: Path) -> None:
    """A checksum-matching symbolic link is still not an admissible evidence member."""
    cell = tmp_path / "cell"
    _write_checksum_bound_cell(cell)
    target = tmp_path / "fixture-target.xml"
    target.write_bytes(_PAYLOADS["fixture.xml"])
    (cell / "fixture.xml").unlink()
    (cell / "fixture.xml").symlink_to(target)

    with pytest.raises(RuntimeEvidenceAdmissionError, match="unsafe"):
        _read_verified_members(cell)


def test_retained_worker_contract_admits_the_exact_fifty_one_sample_clock() -> None:
    """The retained one-second/0.02-second campaign remains a valid positive control."""
    fixture = _smooth_fixture()

    _validate_fixture_contract(fixture)
    _validate_observation_clock(_EXPECTED_OBSERVATION_TIME_TOKENS, 51, "1")

    assert _EXPECTED_OBSERVATION_TIME_TOKENS[:3] == ("0", "0.02", "0.04")
    assert _EXPECTED_OBSERVATION_TIME_TOKENS[-1] == "1"


def test_resealed_two_point_trace_cannot_replace_the_frozen_observation_clock() -> None:
    """A recomputed canonical trace digest cannot authorize a two-point structural forgery."""
    times = np.asarray([0.0, 1.0], dtype="<f8")
    descriptors: list[object] = [
        {
            "array_key": "channel_0000",
            "channel_id": "pendulum.theta.position",
            "kind": "JOINT_POSITION",
            "semantic_type": "CONTINUOUS_SCALAR",
            "object_name": "theta",
            "component": None,
            "shape": [2],
            "dtype": "<f8",
            "scale": "3.141592653589793",
            "tolerance": "0.000001",
        }
    ]
    arrays = {"channel_0000": np.asarray([0.4, 0.5], dtype="<f8")}

    resealed = _canonical_trace_sha256(descriptors, [], times, arrays)

    assert len(resealed) == 64
    with pytest.raises(RuntimeEvidenceAdmissionError, match=r"exactly t=0\.\.1"):
        _validate_observation_clock(("0", "1"), 2, "1")


def test_lexically_near_control_dt_is_not_rounded_into_the_frozen_contract() -> None:
    """Decimal admission preserves a near-0.02 lexical mutation through semantic checking."""
    mutated = _FIXTURE_MANIFEST.read_bytes().replace(
        b'"control_dt": 0.02', b'"control_dt": 0.020000000000000001', 1
    )
    fixture = _select_manifest_fixture(_strict_manifest_json_bytes(mutated), "smooth_pendulum")

    with pytest.raises(RuntimeEvidenceAdmissionError, match="control_dt"):
        _validate_fixture_contract(fixture)


@pytest.mark.parametrize(
    "loader",
    [_strict_json_bytes, _strict_manifest_echo_with_label, _strict_manifest_with_label],
)
def test_deeply_nested_json_is_a_bounded_evidence_refusal(
    loader: Callable[[bytes, str], dict[str, object]],
) -> None:
    """Refuse caller-controlled nesting identically on every supported CPython.

    This document is far beyond both declared depth profiles and beyond what some interpreters will
    parse at all: CPython 3.14 parses this nesting natively and is stopped by the declared bound,
    while CPython 3.12's parser exhausts its own stack first. Which of the two intervenes is an
    interpreter detail, so only the stable typed refusal is contracted here. The bound itself is
    proved directly, at the exact declared boundary, in
    ``tests/unit/runtime_compatibility/test_runtime_evidence_json_admission.py``.
    """
    nested = b'{"x":' + (b"[" * 10_000) + b"0" + (b"]" * 10_000) + b"}"

    with pytest.raises(RuntimeEvidenceAdmissionError, match="strict JSON admission"):
        loader(nested, "result.json")


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("horizon", 2),
        ("step_dts", [0.004, 0.002]),
        ("continuous_tolerance", 0.000002),
        ("so3_tolerance", 0.0000002),
        ("contact_event_time_tolerance", 0.04),
        ("minimum_common_prefix", 0.40),
    ],
)
def test_resealed_selected_fixture_contract_mutations_are_rejected(
    field: str, replacement: object
) -> None:
    """Outer identity resealing cannot change any fixed decision-bearing physical field."""
    fixture = _smooth_fixture()
    fixture[field] = replacement

    with pytest.raises(RuntimeEvidenceAdmissionError, match=field):
        _validate_fixture_contract(fixture)


def test_selected_fixture_requires_contiguous_full_horizon_controls() -> None:
    """A resealed action program cannot leave a timing gap between control segments."""
    fixture = _smooth_fixture()
    controls = cast(list[object], fixture["controls"])
    second = cast(dict[str, object], controls[1])
    second["start"] = 0.32

    with pytest.raises(RuntimeEvidenceAdmissionError, match="contiguous"):
        _validate_fixture_contract(fixture)


@pytest.mark.parametrize(
    ("role", "baseline_id"),
    [
        ("UNSUPPORTED", "smooth_pendulum"),
        ("MIGRATION_SUBJECT", "planar_arm"),
        ("CONTROLLED_MUTATION", "smooth_pendulum"),
    ],
)
def test_selected_fixture_rejects_unsupported_or_unbound_campaign_roles(
    role: str, baseline_id: str
) -> None:
    """Only the worker's closed roles and baseline relationships can reach CaseEvidence."""
    fixture = _smooth_fixture()
    fixture["campaign_role"] = role
    fixture["baseline_fixture_id"] = baseline_id

    with pytest.raises(
        RuntimeEvidenceAdmissionError, match=r"campaign_role|baseline|controlled mutation"
    ):
        _validate_fixture_contract(fixture)


def test_manifest_fixture_and_projected_channel_ids_must_be_unique() -> None:
    """Duplicate semantic identities remain invalid after an attacker rewrites outer hashes."""
    manifest = _strict_manifest_json_bytes(_FIXTURE_MANIFEST.read_bytes())
    fixtures = cast(list[object], manifest["fixtures"])
    cast(dict[str, object], fixtures[1])["fixture_id"] = "smooth_pendulum"
    with pytest.raises(RuntimeEvidenceAdmissionError, match="fixture IDs"):
        _select_manifest_fixture(manifest, "smooth_pendulum")

    fixture = _smooth_fixture()
    channels = cast(list[object], fixture["channels"])
    cast(dict[str, object], channels[1])["channel_id"] = cast(dict[str, object], channels[0])[
        "channel_id"
    ]
    with pytest.raises(RuntimeEvidenceAdmissionError, match="channel IDs"):
        _validate_fixture_contract(fixture)


@pytest.mark.parametrize(
    ("identity", "message"),
    [
        ("subject", "configured subject identity"),
        ("source", "source-closure identity"),
        ("workload", "workload identity"),
    ],
    ids=("subject", "source-closure", "workload"),
)
def test_cross_cell_subject_source_and_workload_substitutions_are_rejected(
    identity: str,
    message: str,
) -> None:
    """No cell may substitute a different configured or campaign-wide identity."""
    configuration, cells = _cross_cell_campaign()
    if identity == "subject":
        cells[-1].fixture_id = "substituted-subject"
    elif identity == "source":
        cells[-1].subject["source_closure"] = {"closure_sha256": "6" * 64}
    else:
        cells[-1].workload["semantic_sha256"] = "7" * 64

    with pytest.raises(RuntimeEvidenceAdmissionError, match=message):
        _bind_cross_cell_evidence(
            configuration,
            cast(tuple[AdmittedEvidenceCell, ...], tuple(cells)),
        )


@pytest.mark.parametrize(
    ("identity", "message"),
    [
        ("channels", "channel layout/tolerances"),
        ("times", "observation times"),
        ("contact-tolerance", "contact event tolerances"),
        ("execution", "runtime identity"),
    ],
    ids=("channel-layout", "observation-clock", "contact-tolerance", "execution-identity"),
)
def test_cross_cell_channel_time_tolerance_and_execution_substitutions_are_rejected(
    identity: str,
    message: str,
) -> None:
    """Every cell must retain one exact layout clock tolerance and execution identity."""
    configuration, cells = _cross_cell_campaign()
    if identity == "channels":
        cells[1].channels[0].to_primitive.return_value = {
            "channel_id": "substituted.channel",
            "tolerance": "0.000001",
        }
    elif identity == "times":
        cells[1].observation_time_tokens = (*_EXPECTED_OBSERVATION_TIME_TOKENS[:-1], "0.99")
    elif identity == "contact-tolerance":
        cells[1].contact_event_time_tolerance = "0.04"
    else:
        cells[1].runtime["python"]["compiler"] = "substituted-compiler"

    with pytest.raises(RuntimeEvidenceAdmissionError, match=message):
        _bind_cross_cell_evidence(
            configuration,
            cast(tuple[AdmittedEvidenceCell, ...], tuple(cells)),
        )


@pytest.mark.parametrize("member", ["result.json", "trace.npz"], ids=("result", "trace"))
def test_checksum_layer_refuses_substituted_result_and_trace_bytes(
    tmp_path: Path,
    member: str,
) -> None:
    """Caller substitutions remain invalid while their checksum manifest stays unchanged."""
    cell = tmp_path / "cell"
    _write_checksum_bound_cell(cell)
    (cell / member).write_bytes(b"substituted")

    with pytest.raises(RuntimeEvidenceAdmissionError, match=f"checksum mismatch for {member}"):
        _read_verified_members(cell)


def test_result_root_profile_claim_must_match_the_admitted_runtime() -> None:
    """A root-level profile claim cannot float independently of live runtime identity."""
    with pytest.raises(RuntimeEvidenceAdmissionError, match="admitted runtime"):
        _validate_result_profile_identity(
            {"profile_id": "B_3.11.0", "profile_version": "3.11.0"},
            {
                "schema_version": 1,
                "profile_id": "A_3.10.0",
                "profile_version": "3.10.0",
            },
        )


def test_role_runtime_binds_preflight_identity_before_evidence_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A role runtime admits only when every live fact matches exact preflight evidence."""
    configuration, runtime, identity = _role_runtime_fixture()
    monkeypatch.setattr(
        evidence_module,
        "load_native_profile_identity_v2",
        lambda *_args, **_kwargs: identity,
    )
    slot = RuntimeReviewCellConfig("baseline", "0.004", 0, "evidence/cell")

    role, package_version, admitted, admitted_identity = _admit_runtime_v2(
        cast(AdmittedRuntimeReviewConfigurationV2, configuration),
        slot,
        runtime,
    )

    assert role == "baseline"
    assert package_version == "3.12.0+vendor.1"
    assert admitted["profile_identity_sha256"] == identity["profile_identity_sha256"]
    assert admitted["sentinel_identity_sha256"] == identity["sentinel"]["sentinel_identity_sha256"]
    assert admitted_identity == identity


def test_cross_role_runtime_projection_uses_wheel_stable_numpy_identity() -> None:
    """Role-specific venv paths and scripts do not replace the exact RECORD-bound wheel proof."""
    _configuration, baseline, _identity = _role_runtime_fixture()
    candidate = copy.deepcopy(baseline)
    candidate_python = cast(dict[str, object], candidate["python"])
    candidate_python["executable"] = "/profiles/candidate/bin/python"
    candidate_python["resolved_executable"] = "/profiles/candidate/bin/python"
    candidate_numpy = cast(dict[str, object], candidate["numpy"])
    candidate_distribution = cast(dict[str, object], candidate_numpy["distribution"])
    candidate_distribution["payload_sha256"] = "9" * 64

    assert cast(dict[str, object], baseline["numpy"])["distribution"] != candidate_distribution
    assert _common_runtime_projection(cast(dict[str, object], baseline)) == (
        _common_runtime_projection(cast(dict[str, object], candidate))
    )

    candidate_distribution["record_bound_payload_sha256"] = "8" * 64
    assert _common_runtime_projection(cast(dict[str, object], baseline)) != (
        _common_runtime_projection(cast(dict[str, object], candidate))
    )


@pytest.mark.parametrize(
    "binding",
    ["profile", "sentinel", "numpy", "worker"],
    ids=("profile", "sentinel", "numpy", "worker"),
)
def test_role_runtime_substitution_is_refused_before_receipt(
    monkeypatch: pytest.MonkeyPatch, binding: str
) -> None:
    """A canonical runtime re-seal cannot hide any preflight or worker substitution."""
    configuration, runtime, identity = _role_runtime_fixture()
    if binding == "profile":
        runtime["profile_identity_sha256"] = "2" * 64
    elif binding == "sentinel":
        runtime["sentinel_identity_sha256"] = "3" * 64
    elif binding == "numpy":
        distribution = cast(
            dict[str, object], cast(dict[str, object], runtime["numpy"])["distribution"]
        )
        member = cast(dict[str, object], cast(list[object], distribution["members"])[0])
        member["sha256"] = "4" * 64
        payload = [{key: member[key] for key in ("logical_path", "sha256", "size_bytes")}]
        distribution["payload_sha256"] = canonical_sha256(cast(object, payload))
        distribution["record_bound_payload_sha256"] = canonical_sha256(cast(object, payload))
    else:
        runtime["worker_sha256"] = "5" * 64
    unhashed_runtime = {
        key: value for key, value in runtime.items() if key != "runtime_identity_sha256"
    }
    runtime["runtime_identity_sha256"] = canonical_sha256(cast(object, unhashed_runtime))
    monkeypatch.setattr(
        evidence_module,
        "load_native_profile_identity_v2",
        lambda *_args, **_kwargs: identity,
    )
    slot = RuntimeReviewCellConfig("baseline", "0.004", 0, "evidence/cell")

    with pytest.raises(RuntimeEvidenceAdmissionError):
        _admit_runtime_v2(
            cast(AdmittedRuntimeReviewConfigurationV2, configuration),
            slot,
            runtime,
        )


def test_role_result_redundant_hashes_must_match_runtime() -> None:
    """The result root cannot substitute a sentinel hash above a self-hashed runtime."""
    _configuration, runtime, _identity = _role_runtime_fixture()
    result = {
        field: runtime[field]
        for field in (
            "profile_role",
            "package_version",
            "native_version",
            "native_version_integer",
            "profile_identity_sha256",
            "runtime_identity_sha256",
            "sentinel_identity_sha256",
        )
    }
    result["sentinel_identity_sha256"] = "6" * 64

    with pytest.raises(RuntimeEvidenceAdmissionError, match="admitted runtime"):
        _validate_result_profile_identity(result, cast(dict[str, object], runtime))


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("schema_version", True, "completed supported result"),
        ("repeat_id", False, "configured slot"),
        ("profile_id", "B_3.11.0", "configured role"),
    ],
    ids=("boolean-document-revision", "boolean-repeat-claim", "cross-role-profile-claim"),
)
def test_result_root_requires_exact_types_and_configured_profile_binding(
    field: str, replacement: object, message: str
) -> None:
    """Boolean integers and cross-role profile claims cannot exploit equality semantics."""
    manifest_raw = (
        b'{"fixtures":[],"schema":"metrifid.native_upgrade_manifest","schema_version":1}\n'
    )
    manifest: dict[str, object] = {
        "fixtures": [],
        "schema": "metrifid.native_upgrade_manifest",
        "schema_version": 1,
    }
    result: dict[str, object] = {
        "schema": "metrifid.native_upgrade_worker_result",
        "schema_version": 1,
        "status": "COMPLETED",
        "manifest_raw_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_echo": manifest,
        "profile_id": "A_3.10.0",
        "profile_version": "3.10.0",
        "step_dt": "0.004",
        "repeat_id": 0,
        "limitations": list(_LIMITATIONS),
    }
    result[field] = replacement
    configured = cast(
        AdmittedRuntimeReviewConfiguration,
        SimpleNamespace(
            config=SimpleNamespace(
                baseline_profile=SimpleNamespace(profile_id="A_3.10.0", mujoco_version="3.10.0"),
                candidate_profile=SimpleNamespace(profile_id="B_3.11.0", mujoco_version="3.11.0"),
            )
        ),
    )
    slot = RuntimeReviewCellConfig("baseline", "0.004", 0, "evidence/cell")

    with pytest.raises(RuntimeEvidenceAdmissionError, match=message):
        _validate_result_root(
            configured, result, manifest, {"input_manifest.json": manifest_raw}, slot
        )


def test_solver_iterations_at_the_limit_cannot_claim_convergence() -> None:
    """The worker's convergence predicate is strict: iterations must be below the limit."""
    diagnostics = _valid_diagnostics()
    diagnostics["max_solver_iterations"] = 100
    samples = cast(list[object], diagnostics["samples"])
    for sample in samples:
        cast(dict[str, object], sample)["max_solver_iterations"] = 100

    with pytest.raises(RuntimeEvidenceAdmissionError, match="convergence predicate"):
        _admit_diagnostics(diagnostics, _EXPECTED_OBSERVATION_TIME_TOKENS)


def test_diagnostic_iteration_maximum_cannot_decrease_between_samples() -> None:
    """Time-local diagnostic maxima are cumulative worker evidence, not independent claims."""
    diagnostics = _valid_diagnostics()
    samples = cast(list[object], diagnostics["samples"])
    cast(dict[str, object], samples[0])["max_solver_iterations"] = 2
    diagnostics["max_solver_iterations"] = 2

    with pytest.raises(RuntimeEvidenceAdmissionError, match="not cumulative"):
        _admit_diagnostics(diagnostics, _EXPECTED_OBSERVATION_TIME_TOKENS)


def test_fixture_xml_must_remain_bounded_and_self_contained() -> None:
    """The product independently mirrors the worker's self-contained XML trust boundary."""
    _validate_self_contained_fixture_xml(b"<mujoco><worldbody/></mujoco>")

    with pytest.raises(RuntimeEvidenceAdmissionError, match="one-MiB"):
        _validate_self_contained_fixture_xml(b" " * 1_048_577)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'<!DOCTYPE mujoco SYSTEM "outside.dtd"><mujoco/>', "DTD or entity"),
        (b'<mujoco><include file="outside.xml"/></mujoco>', "includes and plugins"),
        (b"<mujoco><plugin/></mujoco>", "includes and plugins"),
        (b'<mujoco><mesh file="outside.stl"/></mujoco>', "external files"),
        (b"<mujoco>", "well-formed"),
    ],
)
def test_fixture_xml_rejects_each_worker_forbidden_closure_form(
    payload: bytes, message: str
) -> None:
    """DTD, include, plugin, external-resource, and malformed XML forms stay inadmissible."""
    with pytest.raises(RuntimeEvidenceAdmissionError, match=message):
        _validate_self_contained_fixture_xml(payload)


def test_runtime_numpy_version_is_frozen_to_the_campaign_version() -> None:
    """Internal runtime consistency alone cannot substitute another NumPy version."""
    with pytest.raises(RuntimeEvidenceAdmissionError, match=r"2\.3\.5"):
        _validate_runtime_numpy({"python_version": "2.3.4", "distribution": {}})


def test_contact_topology_times_must_strictly_progress_on_the_slot_grid() -> None:
    """Contact event, duration, and aggregate persistence tokens bind the simulation grid."""
    contact = {
        "channel_id": "peg.tip_wall",
        "geom_names": ["peg_tip", "wall"],
        "events": [
            {"event": "ONSET", "time": "0.003"},
            {"event": "RELEASE", "time": "0.008"},
        ],
        "segments": [{"onset": "0.003", "release": "0.008", "persistence": "0.005"}],
        "persistence": "0.005",
        "aggregate_normal_impulse": "0",
    }

    with pytest.raises(RuntimeEvidenceAdmissionError, match="alternate within horizon"):
        _admit_contact(contact, "1", "0.004")

    contact["events"] = [
        {"event": "ONSET", "time": "0.004"},
        {"event": "RELEASE", "time": "0.004"},
    ]
    contact["segments"] = [{"onset": "0.004", "release": "0.004", "persistence": "0"}]
    contact["persistence"] = "0"
    with pytest.raises(RuntimeEvidenceAdmissionError, match="strictly increase"):
        _admit_contact(contact, "1", "0.004")
