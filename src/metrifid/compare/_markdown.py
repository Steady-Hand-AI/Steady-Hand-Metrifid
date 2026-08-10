"""Deterministic maintainer-readable rendering of one completed comparison comparison receipt.

This module is presentation only. It never mutates, re-finalizes, or recalculates the
receipt, and it never emits Python or wire primitive representations: exact binary64 bits
and full rational primitives remain available in ``comparison.json``.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path

from ..errors import ReasonCode
from ..json_values import Binary64, ExactRational
from ..schemas import ComparisonReceipt

_MISSING = "—"
_DECIMAL_FORMAT = ".12g"

_CLAIM_BOUNDARY = (
    "This decision applies only to the declared initial state, controls, exact horizon, "
    "monitored hinge/slide joint coordinates, execution conditions, metrics, and tolerances."
)
_NOT_A_CLAIM = (
    "It is not a global equivalence, physical-correctness, contact, sensor, reward, task, "
    "or hardware-safety claim."
)
_NO_CROSSING = "No monitored tolerance crossing was recorded."


def render_markdown(receipt: ComparisonReceipt, config_path: Path) -> str:
    """Render decision, repeatability, metrics, witnesses, reasons, and identities."""
    metrics = receipt.metrics.to_primitive()
    repeatability = receipt.repeatability.to_primitive()
    numerical = receipt.numerical_evidence.to_primitive()

    lines: list[str] = [
        "# metrifid comparison",
        "",
        f"**Status:** `{receipt.status.value}`",
        "",
        *_decision_summary(receipt, metrics),
        *_claim_boundary(),
        *_repeatability(repeatability),
        *_metric_summary(metrics),
        *_first_crossing(receipt),
    ]
    incomplete = _incomplete_lines(numerical)
    if incomplete:
        lines.extend(["## Incomplete-trace evidence", "", *incomplete, ""])
    if receipt.reasons:
        lines.extend(_reasons(receipt))
    lines.extend(_identities(receipt))
    lines.extend(_command(config_path))
    return "\n".join(lines)


def _decimal(value: float) -> str:
    """Render one canonical binary64 value as readable decimal text."""
    return format(value, _DECIMAL_FORMAT)


def _binary64(value: object) -> str:
    """Decode one tagged binary64 primitive to a readable decimal."""
    try:
        decoded = Binary64.from_primitive(value)
    except (TypeError, ValueError):
        return _MISSING
    return _decimal(decoded.to_float())


def _rational(value: object) -> str:
    """Render an exact rational as ``<decimal> (<numerator>/<denominator>)``."""
    try:
        exact = ExactRational.from_primitive(value)
    except (TypeError, ValueError):
        return _MISSING
    decimal = _decimal(exact.numerator / exact.denominator)
    return f"{decimal} ({exact.numerator}/{exact.denominator})"


def _rational_object(value: ExactRational) -> str:
    """Require and parse an exact-rational canonical object for rendering."""
    decimal = _decimal(value.numerator / value.denominator)
    return f"{decimal} ({value.numerator}/{value.denominator})"


def _seconds(value: object) -> str:
    """Render an exact rational duration in seconds with fractional identity."""
    rendered = _rational(value)
    return rendered if rendered == _MISSING else f"{rendered} s"


def _boolean(value: object) -> str:
    """Render a strict canonical boolean as lowercase Markdown text."""
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return _MISSING


def _integer(value: object) -> str:
    """Render a strict canonical integer while excluding booleans."""
    return str(value) if isinstance(value, int) and not isinstance(value, bool) else _MISSING


def _escape(text: object) -> str:
    """Escape table text that originates from user-controlled model or metric names."""
    if not isinstance(text, str):
        return _MISSING
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\r", "\\r").replace("\n", "\\n")


def _role_summary(summary: Mapping[str, object], role: str) -> Mapping[str, object]:
    """Render one role's completion, stability, warnings, and boundary count."""
    value = summary.get(role)
    return value if isinstance(value, Mapping) else {}


def _decision_summary(receipt: ComparisonReceipt, metrics: Mapping[str, object]) -> list[str]:
    """Render status, decisive reasons, and worst metric witness."""
    compared = metrics.get("compared_boundary_count")
    return [
        "## Decision summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Status | `{receipt.status.value}` |",
        f"| Control intervals | {receipt.time.control_intervals} |",
        f"| Compared boundaries | {_integer(compared)} |",
        f"| Horizon | {_rational_object(receipt.time.horizon)} s |",
        f"| Monitored joints | {len(receipt.monitored_joints)} |",
        "",
    ]


def _claim_boundary() -> list[str]:
    """Render the frozen workload-specific claim and limitation boundary."""
    return ["## Claim boundary", "", _CLAIM_BOUNDARY, _NOT_A_CLAIM, ""]


def _repeatability(summary: Mapping[str, object]) -> list[str]:
    """Render baseline and candidate repeat signatures and stability."""
    lines = [
        "## Repeatability",
        "",
        "| Role | Stable | Complete repeats | Captured boundaries |",
        "|---|---|---:|---|",
    ]
    for role in ("baseline", "candidate"):
        role_summary = _role_summary(summary, role)
        complete = role_summary.get("complete_repeats")
        total = role_summary.get("repeat_count")
        repeats = (
            f"{complete}/{total}"
            if isinstance(complete, int) and isinstance(total, int)
            else _MISSING
        )
        counts = role_summary.get("captured_boundary_counts")
        captured = (
            ", ".join(str(int(count)) for count in counts)
            if isinstance(counts, Sequence) and not isinstance(counts, (str, bytes)) and counts
            else _MISSING
        )
        lines.append(
            f"| {role} | {_boolean(role_summary.get('stable'))} | {repeats} | {captured} |"
        )
    lines.append("")
    return lines


def _metric_summary(metrics: Mapping[str, object]) -> list[str]:
    """Render the monitored-joint metric table and physical-unit witnesses."""
    lines = [
        "## Metric summary",
        "",
        "| Joint | Type | Metric | Maximum error | Tolerance | Maximum ratio "
        "| Worst boundary | Worst time | First crossing |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    joints = metrics.get("joints")
    rows: list[tuple[str, str, str]] = []
    if isinstance(joints, Sequence) and not isinstance(joints, (str, bytes)):
        for joint in joints:
            if not isinstance(joint, Mapping):
                continue
            name = joint.get("canonical_name")
            joint_metrics = joint.get("metrics")
            if not isinstance(name, str) or not isinstance(joint_metrics, Mapping):
                continue
            joint_type = joint.get("joint_type")
            for metric_name in sorted(joint_metrics):
                entry = joint_metrics[metric_name]
                if not isinstance(entry, Mapping):
                    continue
                rows.append((name, metric_name, _metric_row(name, joint_type, metric_name, entry)))
    lines.extend(row for _, _, row in sorted(rows, key=lambda item: (item[0], item[1])))
    lines.append("")
    return lines


def _metric_row(
    joint_name: str,
    joint_type: object,
    metric_name: str,
    entry: Mapping[str, object],
) -> str:
    """Render one joint metric's tolerance, maximum error, ratio, and crossing."""
    crossing_index = entry.get("first_crossing_boundary_index")
    if isinstance(crossing_index, int) and not isinstance(crossing_index, bool):
        crossing = f"{crossing_index} @ {_seconds(entry.get('first_crossing_time'))}"
    else:
        crossing = _MISSING
    return (
        f"| `{_escape(joint_name)}` "
        f"| {_escape(joint_type)} "
        f"| `{_escape(metric_name)}` "
        f"| {_binary64(entry.get('maximum_error'))} "
        f"| {_rational(entry.get('tolerance'))} "
        f"| {_binary64(entry.get('maximum_ratio'))} "
        f"| {_integer(entry.get('worst_boundary_index'))} "
        f"| {_seconds(entry.get('worst_time'))} "
        f"| {crossing} |"
    )


def _first_crossing(receipt: ComparisonReceipt) -> list[str]:
    """Render the earliest strict tolerance crossing, when one exists."""
    lines = ["## First crossing", ""]
    if receipt.first_crossing is None:
        lines.extend([_NO_CROSSING, ""])
        return lines
    crossing = receipt.first_crossing.to_primitive()
    lines.extend(
        [
            "| Joint | Metric | Boundary | Time | Error | Tolerance | Ratio |",
            "|---|---|---:|---:|---:|---:|---:|",
            f"| `{_escape(crossing.get('joint_name'))}` "
            f"| `{_escape(crossing.get('metric'))}` "
            f"| {_integer(crossing.get('boundary_index'))} "
            f"| {_seconds(crossing.get('time'))} "
            f"| {_binary64(crossing.get('error'))} "
            f"| {_rational(crossing.get('tolerance'))} "
            f"| {_binary64(crossing.get('ratio'))} |",
            "",
        ]
    )
    return lines


def _reasons(receipt: ComparisonReceipt) -> list[str]:
    """Render ordered comparison reasons with bounded canonical evidence."""
    lines = [
        "## Reasons",
        "",
        "| Code | Role | Boundary | Object | Metric |",
        "|---|---|---:|---|---|",
    ]
    details: list[str] = []
    for reason in receipt.reasons:
        lines.append(
            f"| `{reason.code.value}` "
            f"| {_escape(reason.role)} "
            f"| {_integer(reason.boundary_index)} "
            f"| {_escape(reason.object_name) if reason.object_name is not None else _MISSING} "
            f"| {_escape(reason.metric) if reason.metric is not None else _MISSING} |"
        )
        details.extend(_budget_lines(reason.code, reason.to_primitive()["evidence"]))
    lines.append("")
    if details:
        lines.extend([*details, ""])
    return lines


def _budget_lines(code: ReasonCode, evidence: object) -> list[str]:
    """Render requested and allowed counts for a preexecution budget reason."""
    if not isinstance(evidence, Mapping):
        return []
    if code is ReasonCode.INTERNAL_STEP_BUDGET_EXCEEDED:
        return [
            "- Internal steps requested: "
            f"`{evidence.get('requested_total_internal_steps')}`; maximum: "
            f"`{evidence.get('maximum_total_internal_steps')}`."
        ]
    if code is ReasonCode.TRACE_MEMORY_BUDGET_EXCEEDED:
        return [
            "- Trace float64 bytes requested: "
            f"`{evidence.get('requested_trace_float64_bytes')}`; maximum: "
            f"`{evidence.get('maximum_trace_float64_bytes')}`."
        ]
    return []


def _identities(receipt: ComparisonReceipt) -> list[str]:
    """Render receipt, tool, runtime, input, model, and alignment hashes."""
    receipt_sha = receipt.receipt_sha256 if receipt.receipt_sha256 is not None else _MISSING
    return [
        "## Identities",
        "",
        f"- Receipt: `{receipt_sha}`",
        f"- Tool: `{receipt.tool.version}` / `{receipt.tool.distribution_sha256}`",
        f"- MuJoCo: `{receipt.environment.mujoco_version}`",
        f"- Baseline model closure: `{receipt.model_closures.baseline.sha256()}`",
        f"- Candidate model closure: `{receipt.model_closures.candidate.sha256()}`",
        f"- Alignment: `{receipt.alignment.alignment_sha256}`",
        "",
    ]


def _command(config_path: Path) -> list[str]:
    """Render a shell-quoted reproduction command for the configuration path."""
    return [
        "## Command used for this run",
        "",
        "```bash",
        f"metrifid compare {shlex.quote(str(config_path))}",
        "```",
        "",
    ]


def _incomplete_lines(summary: Mapping[str, object]) -> list[str]:
    """Render early-termination and invalid-state details for an incomplete role."""
    result: list[str] = []
    for role in ("baseline", "candidate"):
        value = summary.get(role)
        if isinstance(value, Mapping) and value.get("complete") is not True:
            result.append(
                f"- {role}: captured {_integer(value.get('captured_boundary_count'))} of "
                f"{_integer(value.get('expected_boundary_count'))} boundaries; "
                f"invalid kind: {_escape(value.get('invalid_kind')) if value.get('invalid_kind') is not None else _MISSING}; "
                f"invalid boundary: {_integer(value.get('invalid_boundary_index'))}."
            )
    return result


__all__ = ["render_markdown"]
