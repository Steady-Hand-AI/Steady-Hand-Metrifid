"""Strict native profile-identity collector admission tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import py_compile
from pathlib import Path
from typing import Any, cast

import pytest

from metrifid.runtime_review import _native_profile_identity
from metrifid.runtime_review._native_profile_identity import (
    _DETERMINISTIC_ENVIRONMENT,
    _LEGACY_FROZEN_WORKER_SHA256,
    _PROFILE_IDENTITY_SCHEMA,
    ProfileIdentityRefusal,
    load_native_profile_identity,
    main,
)

_DIGEST = "a" * 64


def test_verified_worker_bytes_bypass_substituted_bytecode_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute measured worker source bytes even when a matching stale cache is hostile."""
    worker = tmp_path / "native_evidence_worker.py.txt"
    malicious = b"VALUE = 'evil'\n"
    admitted = b"VALUE = 'safe'\n"
    assert len(malicious) == len(admitted)
    timestamp = 1_700_000_000
    worker.write_bytes(malicious)
    os.utime(worker, (timestamp, timestamp))
    cache = Path(importlib.util.cache_from_source(str(worker)))
    cache.parent.mkdir()
    py_compile.compile(str(worker), cfile=str(cache), doraise=True)
    worker.write_bytes(admitted)
    os.utime(worker, (timestamp, timestamp))
    monkeypatch.setattr(
        _native_profile_identity,
        "_FROZEN_WORKER_SHA256",
        hashlib.sha256(admitted).hexdigest(),
    )

    loaded = _native_profile_identity._load_frozen_worker(worker)

    assert cast(Any, loaded).VALUE == "safe"


def _distribution(name: str, version: str) -> dict[str, Any]:
    """Build one internally consistent compact distribution projection."""
    return {
        "member_count": 3,
        "name": name,
        "payload_identity_algorithm": "sha256(canonical-json(payload))",
        "payload_sha256": _DIGEST,
        "record_bound_identity_algorithm": "sha256(canonical-json(record-bound))",
        "record_bound_member_count": 2,
        "record_bound_payload_sha256": _DIGEST,
        "record_declared_sha256_member_count": 2,
        "record_unhashed_member_count": 1,
        "version": version,
    }


def _profile_identity(profile_id: str, runtime_version: str, launcher: str) -> dict[str, Any]:
    """Build one canonical collector identity for a declared runtime role."""
    version_integer = 3_010_000 if runtime_version == "3.10.0" else 3_011_000
    document: dict[str, Any] = {
        "schema": _PROFILE_IDENTITY_SCHEMA,
        "schema_version": 1,
        "profile_id": profile_id,
        "profile_contract": {
            "headless": True,
            "mujoco_version": runtime_version,
            "numpy_version": "2.3.5",
            "viewer_or_renderer_used": False,
            "worker_sha256": _LEGACY_FROZEN_WORKER_SHA256,
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
            "distribution": _distribution("mujoco", runtime_version),
            "loaded_native_library": {
                "filename": f"libmujoco.{runtime_version}.dylib",
                "loaded_path": f"/profiles/{profile_id}/libmujoco.dylib",
                "resolved_path": f"/profiles/{profile_id}/libmujoco.dylib",
                "sha256": _DIGEST,
                "size_bytes": 1024,
            },
            "native_version": runtime_version,
            "python_version": runtime_version,
            "version_integer": version_integer,
        },
        "numpy": {
            "distribution": _distribution("numpy", "2.3.5"),
            "python_version": "2.3.5",
        },
        "installation": {"artifacts": [], "available": False, "install_command": None},
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
            "compiled_model_nu": 1,
            "compiled_model_nv": 1,
            "compiled_model_size_bytes": 2048,
            "fixture_id": "smooth_pendulum",
            "manifest_raw_sha256": _DIGEST,
            "passed": True,
            "step_dt": "0.004",
        },
    }
    return _reseal(document)


def _reseal(document: dict[str, Any]) -> dict[str, Any]:
    """Recompute one test identity's canonical self-hash after a semantic mutation."""
    projection = dict(document)
    projection.pop("profile_identity_sha256", None)
    canonical = json.dumps(
        projection,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    document["profile_identity_sha256"] = hashlib.sha256(canonical).hexdigest()
    return document


def _write_identity(path: Path, document: dict[str, Any]) -> None:
    """Write one canonical test identity with the collector's line termination."""
    path.write_text(
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_profile_identity_binds_runtime_and_worker(tmp_path: Path) -> None:
    """A canonical identity binds both the declared native role and frozen worker bytes."""
    identity = _profile_identity("A_3.10.0", "3.10.0", "/profiles/baseline/bin/python")
    identity_path = tmp_path / "baseline.json"
    _write_identity(identity_path, identity)

    admitted = load_native_profile_identity(
        identity_path,
        expected_profile_id="A_3.10.0",
    )

    assert admitted == identity
    assert admitted["profile_contract"]["worker_sha256"] == _LEGACY_FROZEN_WORKER_SHA256
    assert admitted["mujoco"]["native_version"] == "3.10.0"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("worker_sha256", "b" * 64), ("mujoco_version", "3.11.0")],
    ids=("worker-substitution", "runtime-substitution"),
)
def test_resealed_profile_contract_substitution_is_refused(
    tmp_path: Path, field: str, replacement: str
) -> None:
    """A canonical reseal cannot replace the frozen worker or native role contract."""
    identity = _profile_identity("A_3.10.0", "3.10.0", "/profiles/baseline/bin/python")
    identity["profile_contract"][field] = replacement
    _reseal(identity)
    identity_path = tmp_path / "substituted.json"
    _write_identity(identity_path, identity)

    with pytest.raises(ProfileIdentityRefusal):
        load_native_profile_identity(identity_path, expected_profile_id="A_3.10.0")


def test_profile_identity_requires_exact_self_hash(tmp_path: Path) -> None:
    """Changing an identity without a canonical reseal is refused."""
    identity = _profile_identity("A_3.10.0", "3.10.0", "/profiles/baseline/bin/python")
    identity["native_smoke"]["compiled_model_size_bytes"] = 4096
    identity_path = tmp_path / "changed.json"
    _write_identity(identity_path, identity)

    with pytest.raises(ProfileIdentityRefusal, match="self-hash"):
        load_native_profile_identity(identity_path, expected_profile_id="A_3.10.0")


def test_profile_identity_document_revision_requires_an_integer(tmp_path: Path) -> None:
    """A boolean cannot exploit JSON equality to impersonate the integer schema version."""
    identity = _profile_identity("A_3.10.0", "3.10.0", "/profiles/baseline/bin/python")
    identity["schema_version"] = True
    _reseal(identity)
    identity_path = tmp_path / "boolean-version.json"
    _write_identity(identity_path, identity)

    with pytest.raises(ProfileIdentityRefusal, match="schema_version"):
        load_native_profile_identity(identity_path, expected_profile_id="A_3.10.0")


def test_swapped_profile_role_is_refused(tmp_path: Path) -> None:
    """A correctly hashed candidate identity cannot satisfy the baseline role."""
    identity = _profile_identity("B_3.11.0", "3.11.0", "/profiles/candidate/bin/python")
    identity_path = tmp_path / "candidate.json"
    _write_identity(identity_path, identity)

    with pytest.raises(ProfileIdentityRefusal, match="wrong runtime role"):
        load_native_profile_identity(identity_path, expected_profile_id="A_3.10.0")


def test_failed_pip_check_is_refused_even_after_reseal(tmp_path: Path) -> None:
    """A collector identity cannot admit an unsuccessful dependency consistency gate."""
    identity = _profile_identity("A_3.10.0", "3.10.0", "/profiles/baseline/bin/python")
    identity["pip_check"]["exit_code"] = 1
    _reseal(identity)
    identity_path = tmp_path / "pip-refusal.json"
    _write_identity(identity_path, identity)

    with pytest.raises(ProfileIdentityRefusal, match="pip check did not pass"):
        load_native_profile_identity(identity_path, expected_profile_id="A_3.10.0")


def test_failed_compile_smoke_is_refused_even_after_reseal(tmp_path: Path) -> None:
    """A collector identity cannot admit a failed headless native compile smoke."""
    identity = _profile_identity("A_3.10.0", "3.10.0", "/profiles/baseline/bin/python")
    identity["native_smoke"]["passed"] = False
    _reseal(identity)
    identity_path = tmp_path / "smoke-refusal.json"
    _write_identity(identity_path, identity)

    with pytest.raises(ProfileIdentityRefusal, match="native smoke"):
        load_native_profile_identity(identity_path, expected_profile_id="A_3.10.0")


def test_nonempty_zero_coordinate_compile_smoke_is_admitted(tmp_path: Path) -> None:
    """Treat nonempty compiled bytes, rather than coordinate count, as the smoke success gate."""
    identity = _profile_identity("A_3.10.0", "3.10.0", "/profiles/baseline/bin/python")
    identity["native_smoke"]["compiled_model_nq"] = 0
    _reseal(identity)
    identity_path = tmp_path / "static-smoke.json"
    _write_identity(identity_path, identity)

    admitted = load_native_profile_identity(identity_path, expected_profile_id="A_3.10.0")

    assert admitted["native_smoke"]["compiled_model_nq"] == 0


def test_unknown_identity_field_is_refused(tmp_path: Path) -> None:
    """The collector identity schema remains closed after canonical resealing."""
    identity = _profile_identity("A_3.10.0", "3.10.0", "/profiles/baseline/bin/python")
    identity["unexpected"] = None
    _reseal(identity)
    identity_path = tmp_path / "unknown.json"
    _write_identity(identity_path, identity)

    with pytest.raises(ProfileIdentityRefusal, match="fields differ"):
        load_native_profile_identity(identity_path, expected_profile_id="A_3.10.0")


def test_collector_refusal_is_bounded_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An invalid worker path exits two, writes no identity, and exposes no traceback."""
    output = tmp_path / "identity.json"

    exit_code = main(
        [
            "--worker",
            "relative-worker.py",
            "--manifest",
            str((tmp_path / "manifest.json").resolve()),
            "--fixture-id",
            "smooth_pendulum",
            "--profile-id",
            "A_3.10.0",
            "--output",
            str(output.resolve()),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("REFUSED: ")
    assert "Traceback" not in captured.err
    assert not output.exists()
