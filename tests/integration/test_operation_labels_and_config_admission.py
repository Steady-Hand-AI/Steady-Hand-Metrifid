"""Operation labels on refusals, and refusal of hostile configuration files.

A refusal document is evidence. If a ``certify`` run refuses, the artifact must say ``certify``, not
the name of a different command. These tests drive the real installed console script so the label
is observed the way a user observes it, and they confirm that an invalid configuration publishes no
completed result.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_MODEL_XML = """
<mujoco model="labels">
  <worldbody>
    <body name="b" pos="0 0 1">
      <geom name="g" type="sphere" size="0.1"/>
      <joint name="j" type="hinge" axis="0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _console() -> str:
    """Return the installed console script, or skip when it is unavailable."""
    found = shutil.which("metrifid")
    if found is None:  # pragma: no cover - environment guard
        pytest.skip("the installed metrifid console script is required")
    return found


def _run(arguments: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the installed console script and capture its output."""
    return subprocess.run(
        [_console(), *arguments], cwd=cwd, check=False, capture_output=True, text=True
    )


def _failure(completed: subprocess.CompletedProcess[str]) -> dict[str, object]:
    """Parse one operational failure document from a refusing invocation."""
    payload = completed.stdout.strip() or completed.stderr.strip()
    document = json.loads(payload)
    assert isinstance(document, dict)
    return document


def _write_models(root: Path) -> tuple[Path, Path]:
    """Write one identical baseline/candidate model pair."""
    baseline = root / "baseline.xml"
    candidate = root / "candidate.xml"
    baseline.write_text(_MODEL_XML, encoding="utf-8")
    candidate.write_text(_MODEL_XML, encoding="utf-8")
    return baseline, candidate


@pytest.mark.parametrize(
    ("arguments", "expected_operation"),
    [
        (["compare", "missing.json"], "compare"),
        (["audit-timestep", "missing.json"], "audit-timestep"),
        (["certify", "a.xml", "b.xml", "--output", "out"], "certify"),
    ],
)
def test_each_command_reports_its_own_operation_on_refusal(
    tmp_path: Path, arguments: list[str], expected_operation: str
) -> None:
    """Report the real command in the failure artifact for all three operations."""
    _write_models(tmp_path)
    completed = _run(arguments, tmp_path)
    assert completed.returncode != 0
    document = _failure(completed)
    assert document["operation"] == expected_operation


def test_duplicate_keys_in_a_comparison_config_refuse_without_publishing(tmp_path: Path) -> None:
    """Refuse a duplicate member name and leave no completed comparison behind."""
    config = tmp_path / "comparison.json"
    config.write_text('{"schema_version": 1, "schema_version": 2}', encoding="utf-8")
    completed = _run(["compare", str(config)], tmp_path)
    assert completed.returncode != 0
    document = _failure(completed)
    assert document["operation"] == "compare"
    assert document["reason"]["code"] != "INTERNAL_INVARIANT_FAILED"
    assert not list(tmp_path.glob("**/comparison.json.completed"))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version": 1, "schema_version": 2}',
        '{"schema_version": 1.5}',
        '{"schema_version": NaN}',
        '{"schema_version": Infinity}',
    ],
)
def test_hostile_audit_configurations_refuse_with_a_typed_reason(
    tmp_path: Path, payload: str
) -> None:
    """Refuse duplicate names and noncanonical numbers with a typed audit reason."""
    config = tmp_path / "timestep_audit.json"
    config.write_text(payload, encoding="utf-8")
    completed = _run(["audit-timestep", str(config)], tmp_path)
    assert completed.returncode != 0
    document = _failure(completed)
    assert document["operation"] == "audit-timestep"
    assert document["reason"]["code"] != "INTERNAL_INVARIANT_FAILED"


def test_malformed_utf8_audit_configuration_refuses(tmp_path: Path) -> None:
    """Refuse a configuration file that is not strict UTF-8."""
    config = tmp_path / "timestep_audit.json"
    config.write_bytes(b'{"schema_version": 1, "label": "\xff\xfe"}')
    completed = _run(["audit-timestep", str(config)], tmp_path)
    assert completed.returncode != 0
    assert _failure(completed)["reason"]["code"] != "INTERNAL_INVARIANT_FAILED"


def test_a_configuration_symlink_refuses(tmp_path: Path) -> None:
    """Refuse a symbolic link supplied as a configuration path."""
    target = tmp_path / "real.json"
    target.write_text('{"schema_version": 1}', encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    completed = _run(["compare", str(link)], tmp_path)
    assert completed.returncode != 0
    assert _failure(completed)["reason"]["code"] != "INTERNAL_INVARIANT_FAILED"


def test_a_configuration_fifo_refuses_without_blocking(tmp_path: Path) -> None:
    """Refuse a FIFO configuration path instead of blocking on it."""
    fifo = tmp_path / "pipe.json"
    os.mkfifo(fifo)
    completed = subprocess.run(
        [_console(), "compare", str(fifo)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode != 0
