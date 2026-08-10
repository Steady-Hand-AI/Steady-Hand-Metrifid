"""Hostile installed-distribution identity and return-pair binding tests."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from metrifid.distribution import installed_distribution_sha256

_REFUSAL_CODE = r"""
import json
try:
    from metrifid.distribution import DistributionIdentityError, installed_distribution_sha256
    digest = installed_distribution_sha256()
except DistributionIdentityError as exc:
    failure = exc.to_operational_failure("compare")
    print(json.dumps({
        "outcome": "REFUSED",
        "reason": exc.reason_code.value,
        "identity_state": failure.tool.execution_identity_state,
        "distribution_sha256": failure.tool.distribution_sha256,
    }, sort_keys=True))
else:
    print(json.dumps({"outcome": "TRUSTED", "distribution_sha256": digest}, sort_keys=True))
"""


def _source_root() -> Path:
    """Construct the source root fixture used by execution identity attacks scenarios.

    Deterministic setup isolates execution identity attacks without bypassing the contract
    boundary under assertion.
    """
    configured = os.environ.get("METRIFID_CANDIDATE_SOURCE_ROOT")
    if configured:
        return Path(configured).resolve()
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file():
        return candidate
    raise RuntimeError("METRIFID_CANDIDATE_SOURCE_ROOT is required for external tests")


def _run_json(
    executable: Path,
    code: str,
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Construct the run json fixture used by execution identity attacks scenarios.

    Deterministic setup isolates execution identity attacks without bypassing the contract
    boundary under assertion.
    """
    result = subprocess.run(
        [str(executable), "-c", code],
        cwd=cwd,
        env=env,
        capture_output=True,
        check=True,
        text=True,
    )
    return json.loads(result.stdout)


def _assert_refusal(result: dict[str, Any], expected: set[str]) -> None:
    """Construct the assert refusal fixture used by execution identity attacks scenarios.

    Deterministic setup isolates execution identity attacks without bypassing the contract
    boundary under assertion.
    """
    assert result["outcome"] == "REFUSED"
    assert result["reason"] in expected
    assert result["identity_state"] in {"UNBOUND", "MISMATCH"}
    assert result["distribution_sha256"] is None


def test_normal_installed_wheel_returns_trusted_hash(tmp_path: Path) -> None:
    """Protect the execution identity attacks assurance boundary from behavioral drift.

    This scenario exercises normal installed wheel returns trusted hash; the assertions pin the
    user-visible result and the evidence needed to explain that result.
    """
    result = _run_json(Path(sys.executable), _REFUSAL_CODE, tmp_path, env=os.environ.copy())
    assert result == {
        "distribution_sha256": installed_distribution_sha256(),
        "outcome": "TRUSTED",
    }


def test_source_checkout_and_pythonpath_shadow_refuse(tmp_path: Path) -> None:
    """Protect the execution identity attacks assurance boundary from behavioral drift.

    This scenario exercises source checkout and pythonpath shadow refuse; the assertions pin the
    user-visible result and the evidence needed to explain that result.
    """
    source = _source_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source / "src")
    result = _run_json(Path(sys.executable), _REFUSAL_CODE, source, env=env)
    _assert_refusal(result, {"EXECUTING_CODE_NOT_INSTALLED_DISTRIBUTION"})

    shadow = tmp_path / "shadow"
    shutil.copytree(source / "src" / "metrifid", shadow / "metrifid")
    env["PYTHONPATH"] = str(shadow)
    result = _run_json(Path(sys.executable), _REFUSAL_CODE, tmp_path, env=env)
    _assert_refusal(result, {"EXECUTING_CODE_NOT_INSTALLED_DISTRIBUTION"})


def _site_without_metrifid(tmp_path: Path, normal_site: Path) -> Path:
    """Mirror site-packages without metrifid so third-party imports still resolve.

    Dropping site-packages from sys.path outright also dropped NumPy and MuJoCo, so the
    subprocess died on import long before it could reach the refusal under test. Linking every
    unrelated member keeps the environment intact while leaving the editable install as the only
    visible metrifid distribution.
    """
    mirror = tmp_path / "site-without-metrifid"
    mirror.mkdir()
    for entry in sorted(normal_site.iterdir()):
        if entry.name == "metrifid" or entry.name.startswith(("metrifid-", "metrifid-")):
            continue
        (mirror / entry.name).symlink_to(entry)
    return mirror


def test_editable_install_refuses(tmp_path: Path) -> None:
    """Protect the execution identity attacks assurance boundary from behavioral drift.

    This scenario exercises editable install refuses; the assertions pin the user-visible result
    and the evidence needed to explain that result.
    """
    source = _source_root()
    editable_site = tmp_path / "editable-site"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--editable",
            str(source),
            "--target",
            str(editable_site),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    normal_site = Path(__import__("metrifid").__file__).resolve().parents[1]
    mirror = _site_without_metrifid(tmp_path, normal_site)
    code = (
        "import site, sys; "
        f"sys.path = [p for p in sys.path if p != {str(normal_site)!r}]; "
        f"site.addsitedir({str(mirror)!r}); "
        f"site.addsitedir({str(editable_site)!r}); " + _REFUSAL_CODE
    )
    result = _run_json(Path(sys.executable), code, tmp_path, env=os.environ.copy())
    _assert_refusal(result, {"EDITABLE_INSTALL_UNSUPPORTED"})


def test_mixed_loaded_module_roots_refuse(tmp_path: Path) -> None:
    """Protect the execution identity attacks assurance boundary from behavioral drift.

    This scenario exercises mixed loaded module roots refuse; the assertions pin the user-
    visible result and the evidence needed to explain that result.
    """
    outside = tmp_path / "foreign.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")
    code = (
        "import sys; from types import SimpleNamespace; import metrifid; "
        f"sys.modules['metrifid.foreign'] = SimpleNamespace(__file__={str(outside)!r}); "
        + _REFUSAL_CODE
    )
    result = _run_json(Path(sys.executable), code, tmp_path, env=os.environ.copy())
    _assert_refusal(result, {"MIXED_METRIFID_MODULE_ROOTS"})


def test_loaded_module_absent_from_manifest_refuses(tmp_path: Path) -> None:
    """Protect the execution identity attacks assurance boundary from behavioral drift.

    This scenario exercises loaded module absent from manifest refuses; the assertions pin the
    user-visible result and the evidence needed to explain that result.
    """
    distribution = metadata.distribution("metrifid")
    initializer = next(
        path for path in distribution.files or () if str(path) == "metrifid/__init__.py"
    )
    package_root = Path(str(distribution.locate_file(initializer))).resolve().parent
    extra = package_root / "runtime_only_attack.py"
    extra.write_text("VALUE = 1\n", encoding="utf-8")
    code = (
        "import sys; from types import SimpleNamespace; import metrifid; "
        f"sys.modules['metrifid.runtime_only_attack'] = SimpleNamespace(__file__={str(extra)!r}); "
        + _REFUSAL_CODE
    )
    try:
        result = _run_json(Path(sys.executable), code, tmp_path, env=os.environ.copy())
    finally:
        extra.unlink(missing_ok=True)
    _assert_refusal(result, {"DISTRIBUTION_MANIFEST_INVALID"})


def test_loaded_module_bytes_changed_after_install_refuses(tmp_path: Path) -> None:
    """Protect the execution identity attacks assurance boundary from behavioral drift.

    This scenario exercises loaded module bytes changed after install refuses; the assertions
    pin the user-visible result and the evidence needed to explain that result.
    """
    code = r"""
import json
from pathlib import Path
import metrifid.version as version_module
from metrifid.distribution import DistributionIdentityError, installed_distribution_sha256
target = Path(version_module.__file__).resolve()
original = target.read_bytes()
target.write_bytes(b"X" * len(original))
try:
    try:
        digest = installed_distribution_sha256()
    except DistributionIdentityError as exc:
        failure = exc.to_operational_failure("compare")
        result = {
            "outcome": "REFUSED",
            "reason": exc.reason_code.value,
            "identity_state": failure.tool.execution_identity_state,
            "distribution_sha256": failure.tool.distribution_sha256,
        }
    else:
        result = {"outcome": "TRUSTED", "distribution_sha256": digest}
finally:
    target.write_bytes(original)
print(json.dumps(result, sort_keys=True))
"""
    result = _run_json(Path(sys.executable), code, tmp_path, env=os.environ.copy())
    _assert_refusal(result, {"DISTRIBUTION_MANIFEST_INVALID"})
    assert installed_distribution_sha256()


_PAYLOAD_FILENAME = "project_payload.zip"


def _sha256(path: Path) -> str:
    """Compute the canonical sha256 value used by execution identity attacks fixtures.

    Content addressing keeps the mutation boundary explicit for execution identity attacks.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_pair(summary: dict[str, object], payload: Path, manifest: Path) -> None:
    """Construct the verify pair fixture used by execution identity attacks scenarios.

    Deterministic setup isolates execution identity attacks without bypassing the contract
    boundary under assertion.
    """
    required = {
        "payload_filename",
        "payload_zip_sha256",
        "payload_manifest_sha256",
        "candidate_source_sha256",
    }
    if not required.issubset(summary):
        raise ValueError("missing cross-binding field")
    if payload.name != summary["payload_filename"] or payload.name != _PAYLOAD_FILENAME:
        raise ValueError("wrong payload filename")
    if _sha256(payload) != summary["payload_zip_sha256"]:
        raise ValueError("wrong payload ZIP hash")
    if _sha256(manifest) != summary["payload_manifest_sha256"]:
        raise ValueError("wrong payload manifest hash")
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_value["candidate_source_sha256"] != summary["candidate_source_sha256"]:
        raise ValueError("wrong candidate source hash")


def test_pair_cross_binding_rejects_every_frozen_attack(tmp_path: Path) -> None:
    """Protect the execution identity attacks assurance boundary from behavioral drift.

    This scenario exercises pair cross binding rejects every frozen attack; the assertions pin
    the user-visible result and the evidence needed to explain that result.
    """
    payload = tmp_path / _PAYLOAD_FILENAME
    payload.write_bytes(b"payload-run-a")
    manifest = tmp_path / "PAYLOAD_MANIFEST.json"
    manifest.write_text(json.dumps({"candidate_source_sha256": "1" * 64}), encoding="utf-8")
    summary: dict[str, object] = {
        "payload_filename": _PAYLOAD_FILENAME,
        "payload_zip_sha256": _sha256(payload),
        "payload_manifest_sha256": _sha256(manifest),
        "candidate_source_sha256": "1" * 64,
    }
    _verify_pair(summary, payload, manifest)

    payload.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="ZIP hash"):
        _verify_pair(summary, payload, manifest)
    payload.write_bytes(b"payload-run-a")

    other = tmp_path / "other" / _PAYLOAD_FILENAME
    other.parent.mkdir()
    other.write_bytes(b"payload-run-b")
    with pytest.raises(ValueError, match="ZIP hash"):
        _verify_pair(summary, other, manifest)

    wrong_name = tmp_path / "project_payload-copy.zip"
    wrong_name.write_bytes(payload.read_bytes())
    with pytest.raises(ValueError, match="filename"):
        _verify_pair(summary, wrong_name, manifest)

    bad_manifest = dict(summary)
    bad_manifest["payload_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest hash"):
        _verify_pair(bad_manifest, payload, manifest)

    bad_source = dict(summary)
    bad_source["candidate_source_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="source hash"):
        _verify_pair(bad_source, payload, manifest)

    missing = dict(summary)
    del missing["payload_zip_sha256"]
    with pytest.raises(ValueError, match="missing cross-binding"):
        _verify_pair(missing, payload, manifest)
