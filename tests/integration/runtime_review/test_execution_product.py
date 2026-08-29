"""Public integration boundaries for the installed Runtime Review execution journey."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from metrifid.operational import OperationalReasonCode, OperationalToolObservation
from metrifid.runtime_review import (
    RuntimeReviewOperationError,
    _execution,
    run_runtime_review_configuration_file,
)
from metrifid.version import __version__


def _write_executable(path: Path) -> Path:
    """Create one explicit test-local launcher without invoking environment discovery."""
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def _write_run_configuration(tmp_path: Path, *, output_dir: str) -> Path:
    """Write one exact run declaration using distinct explicit launcher paths."""
    baseline = _write_executable(tmp_path / "baseline-python")
    candidate = _write_executable(tmp_path / "candidate-python")
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    document = {
        "schema": "metrifid.runtime_review_run_config",
        "schema_version": 1,
        "baseline_python": baseline.as_posix(),
        "candidate_python": candidate.as_posix(),
        "manifest": "manifest.json",
        "fixture_id": "smooth_pendulum",
        "output_dir": output_dir,
    }
    path = tmp_path / "runtime_review_run.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def _bind_source_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use one deterministic tool observation while exercising the source checkout."""
    observation = OperationalToolObservation(
        __version__, "VERIFIED_INSTALLED_DISTRIBUTION", "1" * 64
    )
    monkeypatch.setattr(_execution, "_tool", lambda: observation)


def test_public_execution_refuses_relative_profile_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Map strict caller-input rejection to the existing bounded operational artifact."""
    _bind_source_tool(monkeypatch)
    config = _write_run_configuration(tmp_path, output_dir="fresh-output")
    document = json.loads(config.read_text(encoding="utf-8"))
    document["baseline_python"] = "relative/bin/python"
    config.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeReviewOperationError) as captured:
        run_runtime_review_configuration_file(config)

    assert captured.value.failure.reason.code is OperationalReasonCode.CONFIGURATION_PARSE_FAILED
    assert captured.value.failure.exit_code == 64
    assert not (tmp_path / "fresh-output").exists()


def test_public_execution_preserves_an_existing_output_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse an occupied output root without changing its caller-owned sentinel bytes."""
    _bind_source_tool(monkeypatch)
    config = _write_run_configuration(tmp_path, output_dir="occupied-output")
    output = tmp_path / "occupied-output"
    output.mkdir()
    sentinel = output / "caller-owned.txt"
    sentinel.write_bytes(b"preserve\n")

    with pytest.raises(RuntimeReviewOperationError) as captured:
        run_runtime_review_configuration_file(config)

    assert captured.value.failure.reason.code is OperationalReasonCode.CONFIGURATION_PARSE_FAILED
    assert captured.value.failure.exit_code == 64
    assert sentinel.read_bytes() == b"preserve\n"
