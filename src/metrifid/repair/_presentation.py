"""Console presentation for the shared repair file workflow."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ._codec import encode_json

if TYPE_CHECKING:
    from ._files import WorkflowResult

_PARTIAL_OUTPUT_WARNING = "Partial output may remain; no verified result path is available."


def _invoke(operation: str, source: str, output: str | None) -> WorkflowResult:
    from . import read_result_file, replay_run_directory, run_request_file

    if operation == "show":
        return read_result_file(source)
    if output is None:
        raise ValueError("An explicit output directory is required")
    if operation == "run":
        return run_request_file(source, output)
    if operation == "replay":
        return replay_run_directory(source, output)
    raise ValueError("Unsupported repair operation")


def _summary(outcome: WorkflowResult) -> dict[str, object]:
    return {
        "schema": "metrifid.repair_summary",
        "version": 1,
        "operation": outcome.operation,
        "command_status": outcome.command_status,
        "command_exit_code": outcome.command_exit_code,
        "recorded_status": outcome.recorded_status,
        "recorded_exit_code": outcome.recorded_exit_code,
        "qualified": outcome.qualified,
        "reason": outcome.reason,
        "result_path": str(outcome.result_path) if outcome.result_path is not None else None,
    }


def _task_line(document: dict[str, Any] | None) -> str:
    request = None if document is None else document.get("request")
    if not isinstance(request, dict):
        return "Task: unavailable."
    task = request.get("task")
    if not isinstance(task, dict):
        return "Task: unavailable."
    return (
        f"Task: {request['family']}; lower y rail {task['rail_y_lower_m']} m; "
        f"target {task['target_xy_m']} m, yaw {task['target_yaw_rad']} rad."
    )


def _human(outcome: WorkflowResult) -> str:
    lines = [f"{outcome.command_status} (exit {outcome.command_exit_code})."]
    if outcome.command_status == "READ":
        lines.append(
            f"Recorded outcome: {outcome.recorded_status} "
            f"(recorded exit {outcome.recorded_exit_code}); no fresh evaluation."
        )
    lines.append(_task_line(outcome.document))
    status = outcome.recorded_status or outcome.command_status
    prefix = "Recorded: " if outcome.command_status == "READ" else ""
    if outcome.qualified:
        subject = {
            "ALREADY_QUALIFIED": "The starting controller",
            "REPAIRED": "A changed controller",
        }.get(status, "The selected controller")
        lines.append(prefix + subject + " qualified at both timesteps.")
    else:
        lines.append(prefix + "No controller qualification is established by this command.")
    lines.append(f"Reason: {outcome.reason}.")
    costs = outcome.actual_work
    if costs is not None:
        steps = (
            str(costs.integration_steps)
            if costs.integration_steps is not None
            else f"unknown total; {costs.known_completed_steps} known completed, "
            f"{costs.unknown_step_calls} unknown native call(s)"
        )
        lines.append(
            f"Work: {costs.attempts} attempt(s), {steps} integration steps, "
            f"{costs.cpu_seconds:.6g} s engine process CPU."
        )
    if outcome.result_path is not None:
        lines.append(f"Verified result: {outcome.result_path}")
    if outcome.partial_output:
        lines.append(_PARTIAL_OUTPUT_WARNING)
    lines.append("Simulation only; no hardware safety, global-optimum or robustness guarantee.")
    return "\n".join(lines) + "\n"


def execute_cli(
    operation: str,
    source: str,
    output: str | None,
    *,
    as_json: bool,
    write: Callable[[bytes], None],
) -> int:
    """Run the SDK workflow and emit exactly one parsed-command summary."""
    partial_output = False
    try:
        outcome = _invoke(operation, source, output)
        partial_output = outcome.partial_output
        summary = _summary(outcome)
        human = _human(outcome)
        diagnostic = outcome.diagnostic
        exit_code = outcome.command_exit_code
    except Exception as exc:
        exit_code = 70
        diagnostic = f"{type(exc).__name__}: {str(exc)[:240]}"
        summary = {
            "schema": "metrifid.repair_summary",
            "version": 1,
            "operation": operation,
            "command_status": "ERROR",
            "command_exit_code": exit_code,
            "recorded_status": None,
            "recorded_exit_code": None,
            "qualified": False,
            "reason": "UNEXPECTED_INTERNAL_FAILURE",
            "result_path": None,
        }
        human = (
            "ERROR (exit 70): unexpected internal failure.\n"
            "No verified result path is available; partial output may remain.\n"
            "Simulation only; no hardware safety, global-optimum or robustness guarantee.\n"
        )
    if diagnostic:
        sys.stderr.write(diagnostic.rstrip("\n") + "\n")
    if as_json and partial_output:
        sys.stderr.write(_PARTIAL_OUTPUT_WARNING + "\n")
    write(encode_json(summary) if as_json else human.encode("utf-8"))
    return exit_code
