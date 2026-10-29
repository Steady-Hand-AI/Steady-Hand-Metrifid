"""Check installed repair, native-free reading and independent moved-bundle replay.

Run with the supported repair environment's Python. Output must name a new
folder beneath an existing parent. The adjacent request is copied before use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _installed_console(expected_version: str) -> Path:
    _require(version("metrifid") == expected_version, "Installed package version differs")
    console = Path(sys.executable).parent / "metrifid"
    _require(console.is_file() and os.access(console, os.X_OK), "Installed console script missing")
    return console


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected a JSON object: {path.name}")
    return value


def _save(output: Path, report: dict[str, Any]) -> None:
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _command(
    console: Path, output: Path, report: dict[str, Any], label: str, arguments: list[str]
) -> dict[str, Any]:
    command = [str(console), "repair", *arguments, "--json"]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    try:
        completed = subprocess.run(
            command, cwd=output, env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=300
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["commands"].append({"argv": command, "returncode": None, "diagnostic": str(exc)})
        _save(output, report)
        raise
    stdout, stderr = label + ".stdout.json", label + ".stderr.txt"
    (output / stdout).write_bytes(completed.stdout)
    (output / stderr).write_bytes(completed.stderr)
    report["commands"].append(
        {"argv": command, "returncode": completed.returncode, "stdout": stdout, "stderr": stderr}
    )
    _save(output, report)
    _require(completed.returncode == 0, f"{label} exited {completed.returncode}; see {stderr}")
    summary = _read(output / stdout)
    _require(
        summary.get("schema") == "metrifid.repair_summary"
        and summary.get("version") == 1
        and summary.get("operation") == arguments[0]
        and summary.get("command_exit_code") == 0
        and summary.get("qualified") is True,
        f"{label} did not report a qualified outcome",
    )
    return summary


def _outcome(summary: dict[str, Any], status: str, recorded: str, path: Path) -> None:
    _require(
        summary.get("command_status") == status
        and summary.get("recorded_status") == recorded
        and summary.get("recorded_exit_code") == 0
        and summary.get("result_path") == str(path),
        "Command status, recorded outcome or result path differs",
    )


def _confirmed(pair: dict[str, Any]) -> None:
    _require(
        pair["complete"] is True and pair["qualified"] is True and pair["objective"] == 0.0,
        "Paired qualification is missing",
    )
    for name in ("nominal", "finer"):
        case = pair[name]
        _require(
            case["classification"] == "COMPLETE"
            and case["qualified"] is True
            and case["completed_steps"] == case["expected_steps"] > 0
            and case["hold_samples"] == case["expected_hold_samples"] > 0
            and not case["hard_failures"],
            f"Incomplete {name} confirmation",
        )


def _work(document: dict[str, Any], expected_version: str) -> int:
    _require(document["tool"]["version"] == expected_version, "Record package version differs")
    costs = document["costs"]
    limits = document["request"]["limits"]
    steps = costs["integration_steps"]
    _require(
        costs["limits"] == limits
        and costs["unknown_step_calls"] == 0
        and type(steps) is int
        and steps == costs["known_completed_steps"]
        and 0 < steps <= limits["max_integration_steps"]
        and 0 <= costs["attempts"] <= limits["max_attempts"]
        and math.isfinite(costs["cpu_seconds"])
        and 0 <= costs["cpu_seconds"] < limits["max_cpu_seconds"],
        "Actual work is missing, unknown or outside admitted limits",
    )
    return int(steps)


def _bundle_hashes(directory: Path) -> dict[str, str]:
    expected = {"repair.json", "nominal.xml", "finer.xml", "reset.json", "policy.json"}
    _require({p.name for p in directory.iterdir()} == expected, "Run bundle is incomplete")
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in expected}


def _journey(console: Path, output: Path, report: dict[str, Any], expected_version: str) -> None:
    supplied = output / "request.json"
    shutil.copyfile(Path(__file__).with_name("planar_push.json"), supplied)
    run = output / "run"
    summary = _command(console, output, report, "run", ["run", str(supplied), "--output", str(run)])
    _outcome(summary, "REPAIRED", "REPAIRED", run / "repair.json")
    shown = _command(console, output, report, "show_run", ["show", str(run / "repair.json")])
    _outcome(shown, "READ", "REPAIRED", run / "repair.json")
    record = _read(run / "repair.json")
    _require(
        record["status"] == "REPAIRED"
        and record["qualified"] is True
        and record["confirmation_complete"] is True
        and record["selected_parameters"] is not None
        and record["selected_parameters"] != record["original_parameters"],
        "Run did not confirm a changed controller",
    )
    _confirmed(record["confirmation"])
    report["run_steps"] = _work(record, expected_version)
    moved = output / "relocated_run"
    run.rename(moved)
    supplied.unlink()
    before = _bundle_hashes(moved)
    replay = output / "replay"
    summary = _command(
        console, output, report, "replay", ["replay", str(moved), "--output", str(replay)]
    )
    _outcome(summary, "QUALIFIED", "QUALIFIED", replay / "replay.json")
    shown = _command(console, output, report, "show_replay", ["show", str(replay / "replay.json")])
    _outcome(shown, "READ", "QUALIFIED", replay / "replay.json")
    fresh = _read(replay / "replay.json")
    _require(
        fresh["status"] == "QUALIFIED"
        and fresh["qualified"] is True
        and fresh["selected_parameters"] == record["selected_parameters"]
        and fresh["source_repair_sha256"] == before["repair.json"],
        "Fresh replay qualification or source linkage differs",
    )
    _confirmed(fresh["measurement"])
    report["replay_steps"] = _work(fresh, expected_version)
    _require(_bundle_hashes(moved) == before, "Replay changed the input bundle")
    report["source_repair_sha256"] = before["repair.json"]
    report["input_bundle_unchanged"] = True


def main(argv: list[str] | None = None) -> int:
    """Run the installed example once, retaining command exits and useful failure logs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args(argv)
    output = args.output.absolute()
    report: dict[str, Any] = {"status": "RUNNING", "commands": [], "version": args.expected_version}
    owned = False
    try:
        console = _installed_console(args.expected_version)
        output.mkdir()
        owned = True
        _journey(console, output, report, args.expected_version)
        report["status"] = "PASS"
        _save(output, report)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        ImportError,
        subprocess.TimeoutExpired,
    ) as exc:
        report["status"] = "FAIL"
        report["diagnostic"] = f"{type(exc).__name__}: {exc}"
        if owned:
            _save(output, report)
        print(report["diagnostic"], file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
