"""Actual nested CLI dispatch for the shared repair file workflow, without physics."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from metrifid import repair
from metrifid.cli import main
from metrifid.repair import _engine
from metrifid.repair._files import WorkflowResult
from metrifid.repair._types import ExecutionCosts

SUMMARY_KEYS = {
    "schema",
    "version",
    "operation",
    "command_status",
    "command_exit_code",
    "recorded_status",
    "recorded_exit_code",
    "qualified",
    "reason",
    "result_path",
}


def _arguments(operation: str, *, as_json: bool = True) -> list[str]:
    arguments = ["repair", operation, "input.json" if operation != "replay" else "input-run"]
    if operation != "show":
        arguments += ["--output", "new-output"]
    if as_json:
        arguments.append("--json")
    return arguments


def _outcome(operation: str, status: str, code: int, **changes: object) -> WorkflowResult:
    fields = {
        "operation": operation,
        "command_status": status,
        "command_exit_code": code,
        "recorded_status": status if code not in (64, 70) else None,
        "recorded_exit_code": code if code not in (64, 70) else None,
        "qualified": code == 0 and status != "READ",
        "reason": "COMPLETE_PAIRED_CONFIRMATION" if code == 0 else "TEST_OUTCOME",
    }
    fields.update(changes)
    return WorkflowResult(**fields)


def _install(monkeypatch, operation: str, outcome: WorkflowResult) -> list[tuple[str, ...]]:
    calls = []

    def invoke(*arguments):
        calls.append(arguments)
        return outcome

    monkeypatch.setattr(
        repair,
        {"run": "run_request_file", "show": "read_result_file", "replay": "replay_run_directory"}[
            operation
        ],
        invoke,
    )
    return calls


@pytest.mark.parametrize(
    ("operation", "status", "code"),
    [
        ("run", "REPAIRED", 0),
        ("run", "ALREADY_QUALIFIED", 0),
        ("run", "NO_QUALIFIED_POLICY_FOUND", 10),
        ("run", "INCOMPLETE", 20),
        ("run", "REFUSED", 64),
        ("run", "ERROR", 70),
        ("replay", "QUALIFIED", 0),
        ("replay", "NOT_QUALIFIED", 10),
        ("replay", "INCOMPLETE", 20),
        ("replay", "REFUSED", 64),
        ("replay", "ERROR", 70),
        ("show", "READ", 0),
        ("show", "REFUSED", 64),
        ("show", "ERROR", 70),
    ],
)
def test_nested_dispatch_uses_sdk_and_emits_one_exact_summary(
    monkeypatch, capsys, operation, status, code
):
    """Every parsed route reaches its SDK function and preserves its command outcome."""
    outcome = _outcome(operation, status, code, reason="Recorded decimal input: café")
    calls = _install(monkeypatch, operation, outcome)
    assert main(_arguments(operation)) == code
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.endswith("\n")
    assert captured.out.count("\n") == 1
    summary = json.loads(captured.out)
    assert set(summary) == SUMMARY_KEYS
    assert summary["schema"] == "metrifid.repair_summary"
    assert summary["version"] == 1
    assert summary["operation"] == operation
    assert summary["command_status"] == status
    assert summary["command_exit_code"] == code
    assert summary["reason"] == outcome.reason
    assert summary["qualified"] is outcome.qualified
    assert summary["result_path"] is None
    expected = (
        ("input.json",)
        if operation == "show"
        else ("input.json" if operation == "run" else "input-run", "new-output")
    )
    assert calls == [expected]


def test_show_distinguishes_read_exit_from_recorded_nonqualification(monkeypatch, capsys, tmp_path):
    """A successful read cannot be confused with the saved evaluator's exit 10."""
    outcome = _outcome(
        "show",
        "READ",
        0,
        recorded_status="NO_QUALIFIED_POLICY_FOUND",
        recorded_exit_code=10,
        qualified=False,
        result_path=tmp_path / "repair.json",
    )
    _install(monkeypatch, "show", outcome)
    assert main(_arguments("show")) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    summary = json.loads(captured.out)
    assert summary["command_status"] == "READ"
    assert summary["recorded_status"] == "NO_QUALIFIED_POLICY_FOUND"
    assert summary["recorded_exit_code"] == 10
    assert not summary["qualified"]
    assert summary["result_path"] == str(tmp_path / "repair.json")


@pytest.mark.parametrize("operation", ["run", "show", "replay"])
def test_human_output_names_task_work_verified_path_and_simulation_scope(
    monkeypatch, capsys, tmp_path, operation
):
    """Readable output preserves task, actual work and recorded-versus-fresh meaning."""
    outcome = _outcome(
        operation,
        "READ" if operation == "show" else "REPAIRED",
        0,
        recorded_status="REPAIRED",
        recorded_exit_code=0,
        qualified=True,
        result_path=tmp_path / "repair.json",
        document={
            "request": {
                "family": "planar_push_timed15_v1",
                "task": {
                    "rail_y_lower_m": -0.06,
                    "target_xy_m": [0.08, 0.06],
                    "target_yaw_rad": 1.5707963267948966,
                },
            }
        },
        actual_work=ExecutionCosts(
            attempts=6, integration_steps=42200, known_completed_steps=42200, cpu_seconds=1.25
        ),
    )
    _install(monkeypatch, operation, outcome)
    assert main(_arguments(operation, as_json=False)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    text = captured.out
    assert "Partial output may remain" not in text
    assert "Simulation only" in text
    assert "planar_push_timed15_v1" in text
    assert "-0.06" in text
    assert "[0.08, 0.06]" in text
    assert "1.5707963267948966" in text
    assert "A changed controller qualified" in text
    assert "42200 integration steps" in text
    assert "1.25 s engine process CPU" in text
    assert str(tmp_path / "repair.json") in text
    if operation == "show":
        assert "no fresh evaluation" in text
        assert "recorded exit 0" in text


@pytest.mark.parametrize("as_json", [False, True])
def test_publication_failure_has_no_qualification_or_successful_path(monkeypatch, capsys, as_json):
    """A failure after writes explicitly permits partial output without exposing a good path."""
    outcome = _outcome(
        "run",
        "ERROR",
        70,
        partial_output=True,
        diagnostic="final verification failed",
        actual_work=ExecutionCosts(
            integration_steps=None, known_completed_steps=7, unknown_step_calls=1, cpu_seconds=0.125
        ),
    )
    _install(monkeypatch, "run", outcome)
    assert main(_arguments("run", as_json=as_json)) == 70
    captured = capsys.readouterr()
    warning = "Partial output may remain; no verified result path is available."
    if as_json:
        assert captured.err == "final verification failed\n" + warning + "\n"
        assert captured.out.endswith("\n")
        assert captured.out.count("\n") == 1
        summary = json.loads(captured.out)
        assert set(summary) == SUMMARY_KEYS
        assert summary["command_status"] == "ERROR"
        assert summary["command_exit_code"] == 70
        assert summary["result_path"] is None
        assert not summary["qualified"]
    else:
        assert captured.err == "final verification failed\n"
        assert captured.out.count(warning) == 1
        assert "7 known completed" in captured.out
        assert "unknown total" in captured.out
        assert "Verified result:" not in captured.out
        assert "Simulation only" in captured.out


@pytest.mark.parametrize(
    "arguments",
    [
        ["repair"],
        ["repair", "run"],
        ["repair", "run", "input.json", "--json"],
        ["repair", "replay", "input-run", "--json"],
        ["repair", "show", "--json"],
        ["repair", "show", "repair.json", "--json", "--full"],
        ["repair", "run", "input.json", "--output", "new", "--json", "--bogus"],
        ["repair", "unknown", "--json"],
    ],
)
def test_syntax_refusal_leaves_stdout_empty(arguments, capsys):
    """Unparsed repair syntax keeps exit 64 and the existing stderr-only parse ABI."""
    assert main(arguments) == 64
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err


def test_successfully_parsed_unexpected_sdk_error_is_one_summary(monkeypatch, capsys):
    """Unexpected ordinary exceptions cannot bypass parsed JSON error presentation."""

    def fail(*arguments):
        raise RuntimeError("workflow failed")

    monkeypatch.setattr(repair, "run_request_file", fail)
    assert main(_arguments("run")) == 70
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert captured.out.count("\n") == 1
    assert summary["command_status"] == "ERROR"
    assert summary["recorded_status"] is None
    assert summary["result_path"] is None
    assert not summary["qualified"]
    assert "RuntimeError: workflow failed" in captured.err


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_process_control_exceptions_are_not_completed_results(monkeypatch, exception):
    """An interrupt remains an interrupt instead of a false completed summary."""

    def fail(*arguments):
        raise exception()

    monkeypatch.setattr(repair, "run_request_file", fail)
    with pytest.raises(exception):
        main(_arguments("run"))


def _write_request(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "metrifid.repair_request",
                "version": 1,
                "family": "planar_push_timed15_v1",
                "task": {
                    "rail_y_lower_m": -0.06,
                    "target_xy_m": [0.08, 0.06],
                    "target_yaw_rad": 0.0,
                },
                "controller": {"parameters_normalized": [0.5] * 15},
            }
        )
    )


def test_actual_file_workflow_invalid_and_missing_request_outcomes(tmp_path, capsys):
    """Real wrapper admission distinguishes malformed content from an OS read failure."""
    source = tmp_path / "bad.json"
    source.write_text('{"schema": 2}')
    for path, expected in ((source, 64), (tmp_path / "absent.json", 70)):
        assert (
            main(["repair", "run", str(path), "--output", str(tmp_path / "out"), "--json"])
            == expected
        )
        summary = json.loads(capsys.readouterr().out)
        assert summary["command_exit_code"] == expected
        assert summary["result_path"] is None
        assert not summary["qualified"]


def test_actual_dispatch_refuses_unsupported_runtime_without_backend(monkeypatch, tmp_path, capsys):
    """Parsed run goes through file admission and the real engine runtime boundary."""
    source = tmp_path / "request.json"
    _write_request(source)

    def unsupported():
        raise _engine._RuntimeRefused("requires the supported Linux runtime")

    def no_backend(*arguments, **keywords):
        raise AssertionError("native setup must not run")

    monkeypatch.setattr(_engine, "_runtime", unsupported)
    monkeypatch.setattr(_engine, "_create_backend", no_backend)
    assert main(["repair", "run", str(source), "--output", str(tmp_path / "run"), "--json"]) == 64
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["command_status"] == "REFUSED"
    assert summary["reason"] == "UNSUPPORTED_RUNTIME"
    assert not summary["qualified"]


def test_all_repair_help_routes_import_without_numpy_or_mujoco(tmp_path):
    """Root, nested and operation help remain usable with both native libraries forbidden."""
    package = importlib.import_module("metrifid")
    package_parent = Path(package.__file__).resolve().parent.parent
    script = """
import importlib.abc
import sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'mujoco'}:
            raise AssertionError('native import attempted: ' + fullname)
sys.meta_path.insert(0, NoNative())
sys.path.insert(0, sys.argv[1])
from metrifid.cli import main
for arguments in (['--help'], ['repair', '--help'], ['repair', 'run', '--help'],
                  ['repair', 'show', '--help'], ['repair', 'replay', '--help']):
    try:
        main(arguments)
    except SystemExit as exc:
        assert exc.code == 0
assert not any(key.split('.')[0] in {'numpy', 'mujoco'} for key in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(package_parent)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "repair_summary" in completed.stdout
    assert "Simulation only" in completed.stdout
