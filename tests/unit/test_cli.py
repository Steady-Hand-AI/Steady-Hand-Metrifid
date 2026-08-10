"""Installed-command surface tests that do not require the unavailable MuJoCo runtime."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from metrifid.cli import main
from metrifid.operational import OperationalReasonCode


def test_help_is_the_only_root_success_surface(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises help is the only root success surface; the observable command or
    import contract is pinned without relying on repository layout.
    """
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "metrifid.cli", "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "compare" in completed.stdout
    assert "capture" not in completed.stdout
    assert completed.stderr == ""


def test_invalid_invocation_emits_strict_operational_failure(capsys) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises invalid invocation emits strict operational failure; the observable
    command or import contract is pinned without relying on repository layout.
    """
    assert main([]) == 64
    captured = capsys.readouterr()
    failure = json.loads(captured.err)
    assert failure["reason"]["code"] == OperationalReasonCode.INVALID_CLI_INVOCATION.value
    assert failure["operation"] == "compare"


def test_the_root_help_lists_all_three_installed_commands(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises the root help lists all three installed commands; the observable
    command or import contract is pinned without relying on repository layout.
    """
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "metrifid.cli", "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    for command in ("compare", "audit-timestep", "certify"):
        assert command in completed.stdout


def test_the_certify_parser_freezes_the_public_argument_contract(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises the certify parser freezes the public argument contract; the
    observable command or import contract is pinned without relying on repository layout.
    """
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "metrifid.cli", "certify", "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    text = " ".join(completed.stdout.split())
    assert "baseline_mjcf" in text
    assert "candidate_mjcf" in text
    assert "--output OUTPUT" in text
    assert "[--baseline-root BASELINE_ROOT]" in text
    assert "[--candidate-root CANDIDATE_ROOT]" in text
    assert "configuration" not in text


def test_an_invalid_certify_invocation_is_labelled_with_the_certify_operation(capsys) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises an invalid certify invocation is labelled with the certify
    operation; the observable command or import contract is pinned without relying on repository
    layout.
    """
    assert main(["certify"]) == 64
    failure = json.loads(capsys.readouterr().err)
    assert failure["reason"]["code"] == OperationalReasonCode.INVALID_CLI_INVOCATION.value
    assert failure["operation"] == "certify"


def test_certify_without_an_output_option_is_an_invocation_failure(capsys) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises certify without an output option is an invocation failure; the
    observable command or import contract is pinned without relying on repository layout.
    """
    assert main(["certify", "a.xml", "b.xml"]) == 64
    failure = json.loads(capsys.readouterr().err)
    assert failure["reason"]["code"] == OperationalReasonCode.INVALID_CLI_INVOCATION.value
    assert failure["operation"] == "certify"


def test_an_unknown_command_still_reports_the_compare_operation(capsys) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises an unknown command still reports the compare operation; the
    observable command or import contract is pinned without relying on repository layout.
    """
    assert main(["certifyy", "a.xml"]) == 64
    assert json.loads(capsys.readouterr().err)["operation"] == "compare"
