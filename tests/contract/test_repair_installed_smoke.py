"""Exercise the installed smoke orchestration with a controlled console process."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "repair"
EXPECTED_VERSION = "0.0.0-test"

# These small records exercise orchestration only. The real console's show command
# owns record admission; the fixture neither imports the engine nor runs physics.
CONSOLE_SOURCE = r"""
import hashlib
import json
import os
import sys
from pathlib import Path

mode, version = sys.argv[1:3]
arguments = sys.argv[3:]
assert arguments[0] == "repair"
assert arguments[-1] == "--json"
operation = arguments[1]
source = Path(arguments[2])
cwd = Path.cwd()
observation = {
    "argv": arguments,
    "cwd": str(cwd),
    "pythonpath": os.environ.get("PYTHONPATH"),
    "request_exists": (cwd / "request.json").exists(),
    "old_run_exists": (cwd / "run").exists(),
    "moved_run_exists": (cwd / "relocated_run").exists(),
}
if operation == "run":
    observation["request_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
with (cwd / "calls.jsonl").open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(observation) + "\n")

limits = {
    "max_attempts": 128,
    "max_integration_steps": 96000,
    "max_cpu_seconds": 120.0,
}


def pair():
    value = {"complete": True, "qualified": True, "objective": 0.0}
    for name, steps, samples in (("nominal", 2000, 101), ("finer", 4000, 201)):
        value[name] = {
            "classification": "COMPLETE", "qualified": True,
            "completed_steps": steps, "expected_steps": steps,
            "hold_samples": samples, "expected_hold_samples": samples,
            "hard_failures": [],
        }
    return value


def work(steps, attempts, unknown):
    return {
        "limits": limits, "integration_steps": None if unknown else steps,
        "known_completed_steps": steps, "unknown_step_calls": int(unknown),
        "attempts": attempts, "cpu_seconds": 0.125,
    }


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


if operation == "run":
    destination = Path(arguments[arguments.index("--output") + 1])
    result_path = destination / "repair.json"
    recorded_status = "REPAIRED"
    if mode != "no_op":
        destination.mkdir()
        request = json.loads(source.read_text(encoding="utf-8"))
        original = request["controller"]["parameters_normalized"]
        selected = list(original)
        selected[0] += 0.01
        request["limits"] = limits
        record = {
            "tool": {"version": version}, "request": request,
            "status": "REPAIRED", "qualified": True, "confirmation_complete": True,
            "original_parameters": original, "selected_parameters": selected,
            "confirmation": pair(), "costs": work(18200, 2, mode == "unknown_run"),
        }
        if mode != "missing_record":
            save(result_path, record)
        for name in ("nominal.xml", "finer.xml", "reset.json", "policy.json"):
            if mode != "missing_member" or name != "reset.json":
                (destination / name).write_text("controlled fixture\n", encoding="utf-8")
    command_status = recorded_status
elif operation == "replay":
    destination = Path(arguments[arguments.index("--output") + 1])
    destination.mkdir()
    result_path = destination / "replay.json"
    raw = (source / "repair.json").read_bytes()
    record = json.loads(raw)
    save(result_path, {
        "tool": {"version": version}, "request": record["request"],
        "status": "QUALIFIED", "qualified": True,
        "selected_parameters": record["selected_parameters"],
        "source_repair_sha256": hashlib.sha256(raw).hexdigest(),
        "measurement": pair(), "costs": work(6200, 0, mode == "unknown_replay"),
    })
    if mode == "input_mutation":
        (source / "policy.json").write_text("changed by replay\n", encoding="utf-8")
    command_status = recorded_status = "QUALIFIED"
else:
    assert operation == "show"
    result_path = source
    command_status = "READ"
    recorded_status = "REPAIRED" if source.name == "repair.json" else "QUALIFIED"

print(json.dumps({
    "schema": "metrifid.repair_summary", "version": 1, "operation": operation,
    "command_status": command_status, "command_exit_code": 0,
    "recorded_status": recorded_status, "recorded_exit_code": 0,
    "qualified": True, "result_path": str(result_path),
}))
# A successful-looking summary must never hide the actual process exit.
if mode == "run_failure" and operation == "run":
    print("controlled run failure", file=sys.stderr)
    raise SystemExit(20)
if mode == "replay_failure" and operation == "replay":
    print("controlled replay failure", file=sys.stderr)
    raise SystemExit(70)
"""


@pytest.fixture
def smoke():
    """Load the distributed example, not a test copy of its implementation."""
    spec = importlib.util.spec_from_file_location(
        "repair_installed_smoke", EXAMPLES / "installed_smoke.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exercise(smoke, monkeypatch, tmp_path, mode):
    driver = tmp_path / "console_fixture.py"
    driver.write_text(CONSOLE_SOURCE, encoding="utf-8")
    console = tmp_path / "metrifid"
    arguments = [sys.executable, "-I", "-B", str(driver), mode, EXPECTED_VERSION]
    console.write_text(
        "#!/bin/sh\nexec " + " ".join(shlex.quote(item) for item in arguments) + ' "$@"\n',
        encoding="utf-8",
    )
    console.chmod(0o700)

    def installed_console(expected_version):
        assert expected_version == EXPECTED_VERSION
        return console

    monkeypatch.setattr(smoke, "_installed_console", installed_console)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "must-not-reach-child"))
    output = tmp_path / "smoke-output"
    exit_code = smoke.main(["--output", str(output), "--expected-version", EXPECTED_VERSION])
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    call_log = output / "calls.jsonl"
    calls = (
        [json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()]
        if call_log.exists()
        else []
    )
    return exit_code, output, report, calls


def test_installed_smoke_runs_reads_moves_and_replays_with_clean_environment(
    smoke, monkeypatch, tmp_path, capsys
):
    """The real runner launches four commands and removes the copied request before replay."""
    original = (EXAMPLES / "planar_push.json").read_bytes()
    exit_code, output, report, calls = _exercise(smoke, monkeypatch, tmp_path, "positive")
    assert exit_code == 0
    assert report["status"] == "PASS"
    assert json.loads(capsys.readouterr().out) == report
    assert [entry["returncode"] for entry in report["commands"]] == [0, 0, 0, 0]
    assert [entry["argv"][2] for entry in report["commands"]] == ["run", "show", "replay", "show"]
    assert [entry["argv"] for entry in calls] == [
        ["repair", "run", str(output / "request.json"), "--output", str(output / "run"), "--json"],
        ["repair", "show", str(output / "run" / "repair.json"), "--json"],
        [
            "repair",
            "replay",
            str(output / "relocated_run"),
            "--output",
            str(output / "replay"),
            "--json",
        ],
        ["repair", "show", str(output / "replay" / "replay.json"), "--json"],
    ]
    assert all(entry["pythonpath"] is None for entry in calls)
    assert all(entry["cwd"] == str(output) for entry in calls)
    assert calls[0]["request_sha256"] == hashlib.sha256(original).hexdigest()
    assert calls[1]["old_run_exists"]
    assert not calls[2]["old_run_exists"]
    assert not calls[2]["request_exists"]
    assert calls[2]["moved_run_exists"]
    assert not (output / "request.json").exists()
    assert not (output / "run").exists()
    assert (EXAMPLES / "planar_push.json").read_bytes() == original
    assert report["run_steps"] == 18200
    assert report["replay_steps"] == 6200
    assert report["input_bundle_unchanged"] is True
    assert (
        report["source_repair_sha256"]
        == hashlib.sha256((output / "relocated_run" / "repair.json").read_bytes()).hexdigest()
    )
    for entry in report["commands"]:
        assert (output / entry["stdout"]).is_file()
        assert (output / entry["stderr"]).is_file()


@pytest.mark.parametrize(
    ("mode", "operations", "exits"),
    [
        ("run_failure", ["run"], [20]),
        ("replay_failure", ["run", "show", "replay"], [0, 0, 70]),
    ],
)
def test_native_nonzero_exit_stops_without_retry_despite_qualified_stdout(
    smoke, monkeypatch, tmp_path, mode, operations, exits
):
    """Neither show nor replay can turn a failed predecessor into a passing smoke run."""
    exit_code, output, report, calls = _exercise(smoke, monkeypatch, tmp_path, mode)
    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert [entry["returncode"] for entry in report["commands"]] == exits
    assert [entry["argv"][1] for entry in calls] == operations
    assert f"exited {exits[-1]}" in report["diagnostic"]
    final = report["commands"][-1]
    assert json.loads((output / final["stdout"]).read_text(encoding="utf-8"))["qualified"]
    assert "controlled" in (output / final["stderr"]).read_text(encoding="utf-8")
    assert "input_bundle_unchanged" not in report


@pytest.mark.parametrize("mode", ["no_op", "missing_record", "missing_member", "unknown_run"])
def test_missing_or_unknown_run_evidence_prevents_replay(smoke, monkeypatch, tmp_path, mode):
    """Positive console summaries alone cannot establish a usable run bundle."""
    exit_code, output, report, calls = _exercise(smoke, monkeypatch, tmp_path, mode)
    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert [entry["argv"][1] for entry in calls] == ["run", "show"]
    assert [entry["returncode"] for entry in report["commands"]] == [0, 0]
    assert not (output / "replay").exists()
    assert "input_bundle_unchanged" not in report


@pytest.mark.parametrize(
    ("mode", "diagnostic"),
    [
        ("unknown_replay", "Actual work is missing, unknown or outside admitted limits"),
        ("input_mutation", "Replay changed the input bundle"),
    ],
)
def test_replay_must_retain_known_work_and_unchanged_source(
    smoke, monkeypatch, tmp_path, mode, diagnostic
):
    """Four zero process exits cannot hide incomplete work or source mutation."""
    exit_code, _output, report, calls = _exercise(smoke, monkeypatch, tmp_path, mode)
    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert [entry["argv"][1] for entry in calls] == ["run", "show", "replay", "show"]
    assert [entry["returncode"] for entry in report["commands"]] == [0, 0, 0, 0]
    assert diagnostic in report["diagnostic"]
    assert "input_bundle_unchanged" not in report


@pytest.mark.parametrize(
    "failure", [OSError("launch refused"), subprocess.TimeoutExpired("fixture", 300)]
)
def test_launch_or_timeout_is_recorded_once_without_a_native_exit(
    smoke, monkeypatch, tmp_path, failure
):
    """A command that did not finish retains an attempted command with unknown exit."""
    attempts = []

    def fail(command, **kwargs):
        attempts.append(command)
        raise failure

    monkeypatch.setattr(smoke.subprocess, "run", fail)
    exit_code, _output, report, calls = _exercise(smoke, monkeypatch, tmp_path, "positive")
    assert exit_code == 1
    assert report["status"] == "FAIL"
    assert len(attempts) == 1
    assert calls == []
    assert len(report["commands"]) == 1
    assert report["commands"][0]["argv"] == attempts[0]
    assert report["commands"][0]["returncode"] is None
    assert report["commands"][0]["diagnostic"]
