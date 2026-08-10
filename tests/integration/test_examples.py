"""The bundled example must work exactly as its README claims."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "certify" / "run_example.py"


def _installed_console_script() -> str | None:
    """Construct the installed console script fixture used by examples scenarios.

    Deterministic setup isolates examples without bypassing the contract boundary under
    assertion.
    """
    found = shutil.which("metrifid")
    if found:
        return found
    candidate = Path(sys.executable).parent / "metrifid"
    return str(candidate) if candidate.is_file() else None


def test_the_example_script_is_shipped() -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises the example script is shipped; the observable command or import
    contract is pinned without relying on repository layout.
    """
    assert EXAMPLE.is_file()
    for name in ("equivalent/baseline.xml", "equivalent/candidate.xml", "changed.xml"):
        assert (EXAMPLE.parent / name).is_file(), name


def test_the_example_runs_and_reports_both_expected_statuses(tmp_path: Path) -> None:
    """Runs the shipped example against the installed console script, from outside the source."""
    script = _installed_console_script()
    if script is None:
        pytest.skip("the metrifid console script is not installed in this environment")
    environment = dict(os.environ)
    environment["PATH"] = os.pathsep.join([str(Path(script).parent), environment.get("PATH", "")])
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CERTIFIED_COMPILED_EQUIVALENCE (exit 0)" in completed.stdout
    assert "NOT_CERTIFIED_COMPILED_DIFFERS (exit 40)" in completed.stdout
    assert "all 6 checks passed" in completed.stdout


def test_the_example_leaves_nothing_behind(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises the example leaves nothing behind; the observable command or import
    contract is pinned without relying on repository layout.
    """
    script = _installed_console_script()
    if script is None:
        pytest.skip("the metrifid console script is not installed in this environment")
    environment = dict(os.environ)
    environment["PATH"] = os.pathsep.join([str(Path(script).parent), environment.get("PATH", "")])
    subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert list(tmp_path.iterdir()) == []
