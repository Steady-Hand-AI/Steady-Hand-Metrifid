"""Role-based native profile identity and sentinel admission tests."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import pytest

from metrifid.runtime_review._native_profile_identity import (
    _DETERMINISTIC_ENVIRONMENT,
    _FROZEN_WORKER_SHA256,
    ProfileIdentityRefusal,
    load_native_profile_identity_v2,
    profile_identity_receipt_projection_v2,
    require_compatible_profile_identities_v2,
)

_DIGEST = "a" * 64
_LIMITATIONS = [
    "EXACT_PROFILE_FIXTURE_AND_COARSEST_TIMESTEP_ONLY",
    "PUBLIC_MJSTATE_INTEGRATION_ONLY",
    "EXACT_BINARY64_EQUALITY_NO_TOLERANCE",
    "NO_CROSS_PROFILE_EQUIVALENCE_CLAIM",
    "SENTINEL_PROVES_SAME_PROFILE_RESTORE_PIPELINE_ONLY",
]


def _canonical_sha256(value: Any) -> str:
    """Hash one test value with the production canonical JSON convention."""
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _reseal(value: dict[str, Any], hash_field: str) -> dict[str, Any]:
    """Recompute one self-hash after an intentional semantic test mutation."""
    projection = copy.deepcopy(value)
    projection.pop(hash_field, None)
    value[hash_field] = _canonical_sha256(projection)
    return value


def _distribution(name: str, version: str, digest: str = _DIGEST) -> dict[str, Any]:
    """Build one compact exact distribution identity."""
    return {
        "member_count": 3,
        "name": name,
        "payload_identity_algorithm": "sha256(canonical-json(payload))",
        "payload_sha256": digest,
        "record_bound_identity_algorithm": "sha256(canonical-json(record-bound))",
        "record_bound_member_count": 2,
        "record_bound_payload_sha256": digest,
        "record_declared_sha256_member_count": 2,
        "record_unhashed_member_count": 1,
        "version": version,
    }


def _state(values: list[str]) -> dict[str, Any]:
    """Build exact little-endian state tokens and their byte digest."""
    payload = b"".join(struct.pack("<d", float(value)) for value in values)
    return {
        "dtype": "<f8",
        "shape": [len(values)],
        "values": values,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _projection() -> dict[str, Any]:
    """Build a minimal independently self-hashed deterministic projection."""
    projection: dict[str, Any] = {
        "qacc": {"available": True, "shape": [1], "values": ["0"]},
        "qfrc_actuator": {"available": True, "shape": [1], "values": ["0"]},
        "qfrc_constraint": {"available": True, "shape": [1], "values": ["0"]},
        "sensordata": {"available": True, "shape": [1], "values": ["0"]},
        "warnings": {"available": True, "passed": True, "records": []},
        "solver": {
            "iterations": {"available": True, "values": [0]},
            "residuals": {"available": True, "values": ["0"]},
            "iteration_limit": 100,
            "max_iterations": 0,
            "passed": True,
        },
        "contacts": {"ncon": 0, "contacts": []},
        "constraints": {"available": True, "active_count": 0, "rows": []},
    }
    projection["projection_sha256"] = _canonical_sha256(projection)
    return projection


def _sentinel(role: str, *, state_signature: int = 16_383) -> dict[str, Any]:
    """Build one passing complete-state sentinel identity."""
    before = _state(["0", "1.5"])
    after = _state(["0.25", "1.25"])
    forward_pair = {"left": _projection(), "right": _projection()}
    step_pair = {"left": _projection(), "right": _projection()}
    state_pair = {"left": after, "right": copy.deepcopy(after)}
    document: dict[str, Any] = {
        "schema": "metrifid.runtime_review.integration_state_sentinel",
        "schema_version": 1,
        "profile_role": role,
        "fixture_id": "smooth_pendulum",
        "step_dt": "0.004",
        "state_signature": state_signature,
        "state_size": 2,
        "warmup_step_count": 2,
        "pre_restore_integration_state": before,
        "pre_restore_integration_state_sha256": before["sha256"],
        "post_forward_projection": forward_pair,
        "post_forward_projection_sha256": _canonical_sha256(forward_pair),
        "post_step_integration_state": state_pair,
        "post_step_integration_state_sha256": _canonical_sha256(state_pair),
        "post_step_projection": step_pair,
        "post_step_projection_sha256": _canonical_sha256(step_pair),
        "finite_values": True,
        "warnings_passed": True,
        "solver_converged": True,
        "status": "PASS",
        "failure_reason": None,
        "limitations": list(_LIMITATIONS),
    }
    return _reseal(document, "sentinel_identity_sha256")


def _base_triplet(package_version: str) -> tuple[int, int, int]:
    """Extract the stable package base triplet for test data construction."""
    base = package_version.split("+", 1)[0].split(".post", 1)[0]
    major, minor, patch = (int(item) for item in base.split("."))
    return major, minor, patch


def _identity(
    role: str,
    package_version: str,
    *,
    numpy_digest: str = _DIGEST,
    mujoco_digest: str = _DIGEST,
) -> dict[str, Any]:
    """Build one closed role-based profile identity."""
    triplet = _base_triplet(package_version)
    native_version = ".".join(str(item) for item in triplet)
    native_integer = triplet[0] * 1_000_000 + triplet[1] * 1_000 + triplet[2]
    launcher = "/profiles/runtime/bin/python"
    document: dict[str, Any] = {
        "schema": "metrifid.runtime_review.native_profile_identity",
        "schema_version": 2,
        "profile_role": role,
        "package_version": package_version,
        "native_version": native_version,
        "native_version_integer": native_integer,
        "support_tier": "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE",
        "profile_contract": {
            "headless": True,
            "viewer_or_renderer_used": False,
            "worker_sha256": _FROZEN_WORKER_SHA256,
        },
        "python": {
            "build": ["main", "build-date"],
            "cache_tag": "cpython-312",
            "compiler": "test-compiler",
            "executable": launcher,
            "implementation": "CPython",
            "implementation_name": "cpython",
            "resolved_executable": "/opt/python/bin/python",
            "resolved_executable_sha256": _DIGEST,
            "version": "3.12.13",
            "version_full": "3.12.13 test-build",
        },
        "host": {
            "architecture": ["64bit", ""],
            "cpu_model": "test-cpu",
            "cpu_model_source": "test measurement",
            "hardware_model": "test-hardware",
            "hardware_profile": {},
            "hyper_threading_technology": None,
            "libc": ["glibc", "2.39"],
            "logical_cpu_count": 4,
            "machine": "test-machine",
            "physical_cpu_count": 2,
            "platform": "test-platform",
            "release": "test-release",
            "system": "TestOS",
            "version": "test-kernel",
        },
        "environment": dict(_DETERMINISTIC_ENVIRONMENT),
        "mujoco": {
            "distribution": _distribution("mujoco", package_version, mujoco_digest),
            "loaded_native_library": {
                "filename": f"libmujoco.{native_version}.so",
                "loaded_path": "/profiles/runtime/libmujoco.so",
                "resolved_path": "/profiles/runtime/libmujoco.so",
                "sha256": mujoco_digest,
                "size_bytes": 1024,
            },
            "package_version": package_version,
            "native_version": native_version,
            "native_version_integer": native_integer,
        },
        "numpy": {
            "distribution": _distribution("numpy", "2.4.1", numpy_digest),
            "python_version": "2.4.1",
        },
        "installation": {"available": False, "distribution": None},
        "metrifid_installed": False,
        "pip_check": {
            "argv": [launcher, "-m", "pip", "check"],
            "exit_code": 0,
            "stderr": "",
            "stdout": "No broken requirements found.\n",
        },
        "native_smoke": {
            "compiled_model_nbody": 2,
            "compiled_model_ngeom": 1,
            "compiled_model_nq": 1,
            "compiled_model_nsensordata": 1,
            "compiled_model_nu": 1,
            "compiled_model_nv": 1,
            "compiled_model_size_bytes": 2048,
            "fixture_id": "smooth_pendulum",
            "manifest_raw_sha256": _DIGEST,
            "passed": True,
            "step_dt": "0.004",
        },
        "sentinel": _sentinel(role),
    }
    return _reseal(document, "profile_identity_sha256")


def _write_identity(path: Path, identity: dict[str, Any]) -> None:
    """Write canonical identity bytes for the public path loader."""
    path.write_text(
        json.dumps(identity, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "package_version",
    ["3.9.0", "3.12.0.post2", "3.12.0+vendor.1", "4.2.1"],
    ids=("minimum-stable", "post-release", "vendor-build", "future-stable"),
)
def test_role_identity_admits_coherent_stable_package_tokens(
    tmp_path: Path, package_version: str
) -> None:
    """Stable capability-compatible profiles preserve the exact package token."""
    identity = _identity("baseline", package_version)
    path = tmp_path / "baseline.json"
    _write_identity(path, identity)

    admitted = load_native_profile_identity_v2(path, expected_profile_role="baseline")

    assert admitted["package_version"] == package_version
    assert admitted["support_tier"] == "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE"


def test_same_package_profile_pair_is_compatible() -> None:
    """Semantic roles may use the same coherent package and exact NumPy identity."""
    baseline = _identity("baseline", "3.12.0")
    candidate = _identity("candidate", "3.12.0")

    require_compatible_profile_identities_v2(baseline, candidate)


def test_install_local_paths_and_payloads_preserve_cross_role_compatibility() -> None:
    """Distinct venv paths may vary literal scripts while retaining one exact wheel identity."""
    baseline = _identity("baseline", "3.12.0")
    candidate = _identity("candidate", "3.12.0")
    candidate["python"]["executable"] = "/profiles/candidate/bin/python"
    candidate["python"]["resolved_executable"] = "/profiles/candidate/bin/python"
    candidate["pip_check"]["argv"][0] = "/profiles/candidate/bin/python"
    candidate["numpy"]["distribution"]["payload_sha256"] = "b" * 64
    _reseal(candidate, "profile_identity_sha256")

    require_compatible_profile_identities_v2(baseline, candidate)

    baseline_projection = profile_identity_receipt_projection_v2(baseline)
    candidate_projection = profile_identity_receipt_projection_v2(candidate)
    assert (
        baseline_projection["numpy"]["distribution"]["payload_sha256"]
        != candidate_projection["numpy"]["distribution"]["payload_sha256"]
    )
    assert (
        baseline_projection["numpy"]["distribution"]["record_bound_payload_sha256"]
        == candidate_projection["numpy"]["distribution"]["record_bound_payload_sha256"]
    )


def test_distribution_payload_difference_remains_exactly_bound() -> None:
    """Equal base versions with different package bytes retain different self-hashes."""
    baseline = _identity("baseline", "3.12.0", mujoco_digest="b" * 64)
    candidate = _identity("candidate", "3.12.0", mujoco_digest="c" * 64)

    assert baseline["profile_identity_sha256"] != candidate["profile_identity_sha256"]
    assert baseline["support_tier"] == candidate["support_tier"]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("native_version", "3.11.0"),
        ("native_version_integer", 3_012_001),
        ("support_tier", "VALIDATED_EXACT_PROFILE"),
    ],
    ids=("native-string", "native-integer", "exact-tier"),
)
def test_incoherent_live_profile_claim_is_refused(
    tmp_path: Path, field: str, replacement: object
) -> None:
    """A canonical reseal cannot admit incoherent native or exact-tier claims."""
    identity = _identity("baseline", "3.12.0")
    identity[field] = replacement
    _reseal(identity, "profile_identity_sha256")
    path = tmp_path / "changed.json"
    _write_identity(path, identity)

    with pytest.raises(ProfileIdentityRefusal):
        load_native_profile_identity_v2(path, expected_profile_role="baseline")


def test_swapped_role_and_wrong_outer_hash_are_refused(tmp_path: Path) -> None:
    """A candidate identity cannot satisfy either baseline role or hash binding."""
    identity = _identity("candidate", "3.12.0")
    path = tmp_path / "candidate.json"
    _write_identity(path, identity)

    with pytest.raises(ProfileIdentityRefusal, match="role"):
        load_native_profile_identity_v2(path, expected_profile_role="baseline")
    with pytest.raises(ProfileIdentityRefusal, match="expected self-hash"):
        load_native_profile_identity_v2(
            path,
            expected_profile_role="candidate",
            expected_profile_identity_sha256="b" * 64,
        )


def test_wrong_worker_binding_is_refused(tmp_path: Path) -> None:
    """A resealed identity cannot substitute different worker bytes."""
    identity = _identity("baseline", "3.12.0")
    identity["profile_contract"]["worker_sha256"] = "b" * 64
    _reseal(identity, "profile_identity_sha256")
    path = tmp_path / "substituted.json"
    _write_identity(path, identity)

    with pytest.raises(ProfileIdentityRefusal, match="production worker"):
        load_native_profile_identity_v2(path, expected_profile_role="baseline")


@pytest.mark.parametrize(
    "mutation",
    ["python", "host", "environment", "numpy", "numpy-version"],
    ids=(
        "interpreter",
        "host",
        "thread-environment",
        "numpy-record-binding",
        "numpy-version",
    ),
)
def test_cross_profile_environment_substitution_is_refused(mutation: str) -> None:
    """Both roles must share exact execution and NumPy identity."""
    baseline = _identity("baseline", "3.11.0")
    candidate = _identity("candidate", "3.12.0")
    if mutation == "python":
        candidate["python"]["resolved_executable_sha256"] = "b" * 64
    elif mutation == "host":
        candidate["host"]["machine"] = "other-machine"
    elif mutation == "environment":
        candidate["environment"]["OMP_NUM_THREADS"] = "2"
    elif mutation == "numpy":
        candidate["numpy"]["distribution"]["record_bound_payload_sha256"] = "b" * 64
    else:
        candidate["numpy"]["python_version"] = "2.4.2"
        candidate["numpy"]["distribution"]["version"] = "2.4.2"
    _reseal(candidate, "profile_identity_sha256")

    with pytest.raises(ProfileIdentityRefusal):
        require_compatible_profile_identities_v2(baseline, candidate)


@pytest.mark.parametrize(
    "state_signature",
    [1, 32_767],
    ids=("minimal-positive", "expanded-public-mask"),
)
def test_public_state_signature_is_measured_not_release_frozen(
    tmp_path: Path, state_signature: int
) -> None:
    """Any positive measured public integration-state mask remains admissible."""
    identity = _identity("baseline", "3.12.0")
    identity["sentinel"] = _sentinel("baseline", state_signature=state_signature)
    _reseal(identity, "profile_identity_sha256")
    path = tmp_path / "baseline.json"
    _write_identity(path, identity)

    admitted = load_native_profile_identity_v2(path, expected_profile_role="baseline")

    assert admitted["sentinel"]["state_signature"] == state_signature


@pytest.mark.parametrize("state_signature", [0, True], ids=("zero", "boolean"))
def test_invalid_public_state_signature_is_refused(tmp_path: Path, state_signature: object) -> None:
    """Zero and boolean state masks cannot impersonate a measured public mask."""
    identity = _identity("baseline", "3.12.0")
    identity["sentinel"] = _sentinel("baseline")
    identity["sentinel"]["state_signature"] = state_signature
    _reseal(identity["sentinel"], "sentinel_identity_sha256")
    _reseal(identity, "profile_identity_sha256")
    path = tmp_path / "invalid.json"
    _write_identity(path, identity)

    with pytest.raises(ProfileIdentityRefusal, match="state_signature"):
        load_native_profile_identity_v2(path, expected_profile_role="baseline")


@pytest.mark.parametrize(
    "mutation",
    ["empty-projection", "malformed-array", "unavailable-required", "contact-width"],
    ids=("empty-projection", "malformed-array", "required-absence", "contact-width"),
)
def test_resealed_malformed_sentinel_projection_is_refused(tmp_path: Path, mutation: str) -> None:
    """Nested re-seals cannot hide an empty or internally inconsistent projection."""
    identity = _identity("baseline", "3.12.0")
    sentinel = identity["sentinel"]
    pair = sentinel["post_forward_projection"]
    if mutation == "empty-projection":
        projection: dict[str, Any] = {}
        projection["projection_sha256"] = _canonical_sha256(projection)
        pair["left"] = projection
        pair["right"] = copy.deepcopy(projection)
    elif mutation == "malformed-array":
        for side in ("left", "right"):
            projection = pair[side]
            projection["qacc"] = {
                "available": True,
                "shape": [2],
                "values": ["0"],
            }
            _reseal(projection, "projection_sha256")
    elif mutation == "unavailable-required":
        for side in ("left", "right"):
            projection = pair[side]
            projection["qacc"] = {
                "available": False,
                "shape": None,
                "values": None,
            }
            _reseal(projection, "projection_sha256")
    else:
        malformed_contact = {
            "geom_names": ["alpha", "beta"],
            "dim": 3,
            "distance": "0",
            "position": ["0", "0"],
            "frame": ["0"] * 9,
            "friction": ["0"] * 5,
            "force": ["0"] * 6,
        }
        for side in ("left", "right"):
            projection = pair[side]
            projection["contacts"] = {"ncon": 1, "contacts": [malformed_contact]}
            _reseal(projection, "projection_sha256")
    sentinel["post_forward_projection_sha256"] = _canonical_sha256(pair)
    _reseal(sentinel, "sentinel_identity_sha256")
    _reseal(identity, "profile_identity_sha256")
    path = tmp_path / "malformed.json"
    _write_identity(path, identity)

    with pytest.raises(ProfileIdentityRefusal):
        load_native_profile_identity_v2(path, expected_profile_role="baseline")


def test_metrifid_installation_facts_must_agree(tmp_path: Path) -> None:
    """A reseal cannot claim Metrifid is installed while retaining no distribution."""
    identity = _identity("baseline", "3.12.0")
    identity["metrifid_installed"] = True
    _reseal(identity, "profile_identity_sha256")
    path = tmp_path / "installation.json"
    _write_identity(path, identity)

    with pytest.raises(ProfileIdentityRefusal, match="installation facts"):
        load_native_profile_identity_v2(path, expected_profile_role="baseline")


def test_sentinel_fixture_must_equal_compiled_smoke_fixture(tmp_path: Path) -> None:
    """A nested re-seal cannot substitute another fixture's sentinel result."""
    identity = _identity("baseline", "3.12.0")
    identity["sentinel"]["fixture_id"] = "different_fixture"
    _reseal(identity["sentinel"], "sentinel_identity_sha256")
    _reseal(identity, "profile_identity_sha256")
    path = tmp_path / "fixture-substitution.json"
    _write_identity(path, identity)

    with pytest.raises(ProfileIdentityRefusal, match="compiled smoke fixture"):
        load_native_profile_identity_v2(path, expected_profile_role="baseline")


@pytest.mark.parametrize(
    "mutation",
    ["missing-sentinel", "substituted-sentinel-role"],
    ids=("missing", "role-substitution"),
)
def test_sentinel_identity_is_mandatory_and_role_bound(tmp_path: Path, mutation: str) -> None:
    """A profile cannot omit its sentinel or substitute the other semantic role."""
    identity = _identity("baseline", "3.12.0")
    if mutation == "missing-sentinel":
        identity.pop("sentinel")
    else:
        identity["sentinel"]["profile_role"] = "candidate"
        _reseal(identity["sentinel"], "sentinel_identity_sha256")
    _reseal(identity, "profile_identity_sha256")
    path = tmp_path / "sentinel.json"
    _write_identity(path, identity)

    with pytest.raises(ProfileIdentityRefusal):
        load_native_profile_identity_v2(path, expected_profile_role="baseline")


def test_signed_zero_state_is_admitted_and_sign_substitution_is_refused(
    tmp_path: Path,
) -> None:
    """Exact state evidence preserves zero sign and rejects a resealed unequal state pair."""
    identity = _identity("baseline", "3.12.0")
    sentinel = identity["sentinel"]
    before = _state(["-0.0", "1.5"])
    after = _state(["-0.0", "1.25"])
    sentinel["pre_restore_integration_state"] = before
    sentinel["pre_restore_integration_state_sha256"] = before["sha256"]
    sentinel["post_step_integration_state"] = {
        "left": after,
        "right": copy.deepcopy(after),
    }
    sentinel["post_step_integration_state_sha256"] = _canonical_sha256(
        sentinel["post_step_integration_state"]
    )
    _reseal(sentinel, "sentinel_identity_sha256")
    _reseal(identity, "profile_identity_sha256")
    path = tmp_path / "signed-zero.json"
    _write_identity(path, identity)

    admitted = load_native_profile_identity_v2(path, expected_profile_role="baseline")

    assert admitted["sentinel"]["pre_restore_integration_state"]["values"][0] == "-0.0"

    substituted = copy.deepcopy(identity)
    right = substituted["sentinel"]["post_step_integration_state"]["right"]
    right["values"][0] = "0"
    right["sha256"] = _state(right["values"])["sha256"]
    substituted["sentinel"]["post_step_integration_state_sha256"] = _canonical_sha256(
        substituted["sentinel"]["post_step_integration_state"]
    )
    _reseal(substituted["sentinel"], "sentinel_identity_sha256")
    _reseal(substituted, "profile_identity_sha256")
    substituted_path = tmp_path / "substituted-zero-sign.json"
    _write_identity(substituted_path, substituted)

    with pytest.raises(ProfileIdentityRefusal, match="post-step integration-state"):
        load_native_profile_identity_v2(substituted_path, expected_profile_role="baseline")


@pytest.mark.parametrize(
    "token", ["-0", "0.0", "+0.0"], ids=("short-negative", "positive-decimal", "explicit-plus")
)
def test_noncanonical_zero_tokens_are_refused(tmp_path: Path, token: str) -> None:
    """Only the frozen positive- and negative-zero tokens may represent binary64 zero."""
    identity = _identity("baseline", "3.12.0")
    sentinel = identity["sentinel"]
    before = sentinel["pre_restore_integration_state"]
    before["values"][0] = token
    before["sha256"] = _state([token, "1.5"])["sha256"]
    sentinel["pre_restore_integration_state_sha256"] = before["sha256"]
    _reseal(sentinel, "sentinel_identity_sha256")
    _reseal(identity, "profile_identity_sha256")
    path = tmp_path / "noncanonical-zero.json"
    _write_identity(path, identity)

    with pytest.raises(ProfileIdentityRefusal, match="canonical binary64 token"):
        load_native_profile_identity_v2(path, expected_profile_role="baseline")


def test_receipt_projection_is_closed_and_sentinel_bound() -> None:
    """The receipt projection exposes only frozen role and exact identity fields."""
    identity = _identity("baseline", "3.12.0.post2+vendor.1")

    projection = profile_identity_receipt_projection_v2(identity)

    assert set(projection) == {
        "profile_role",
        "package_version",
        "native_version",
        "native_version_integer",
        "profile_identity_sha256",
        "mujoco_distribution",
        "loaded_native_library",
        "numpy",
        "sentinel",
        "support_tier",
    }
    assert projection["sentinel"] == {
        "status": "PASS",
        "sentinel_identity_sha256": identity["sentinel"]["sentinel_identity_sha256"],
    }
