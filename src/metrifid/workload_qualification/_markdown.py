"""Render one completed qualification receipt as the human-readable half of the published pair.

Every value is read from the receipt; the Markdown never recomputes a decision, so the two published
files cannot disagree. Every user-controlled label goes through `_safe`, so a semantic label cannot
become a table cell, a heading, a fence terminator, raw HTML, or something that reads like a
filesystem path the report is pointing at.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..json_values import CanonicalValue
from ._safe import markdown_block, markdown_code, markdown_label

_STATUS_SENTENCE: Mapping[str, str] = {
    "QUALIFIED_FOR_DECLARED_PROBES": (
        "Every declared probe group is detected at or above its required magnitude by the "
        "selected workloads."
    ),
    "PARTIALLY_QUALIFIED": (
        "Some declared probe groups are detected at or above their required magnitude and some "
        "are not. No group is unresolved."
    ),
    "INSUFFICIENT_EXCITATION": (
        "No declared probe group is detected at or above its required magnitude. No group is "
        "unresolved."
    ),
    "UNRESOLVED": (
        "At least one probe group could not be adjudicated because a rung at or above its "
        "required magnitude did not complete as a decision."
    ),
}


def _items(value: CanonicalValue) -> list[CanonicalValue]:
    """Read one receipt field that must be a list."""
    if not isinstance(value, list):
        raise TypeError("expected a list field in the receipt")
    return value


def _object(value: CanonicalValue) -> dict[str, CanonicalValue]:
    """Read one receipt field that must be an object."""
    if not isinstance(value, dict):
        raise TypeError("expected an object field in the receipt")
    return value


def _table(rows: Sequence[Sequence[str]], header: Sequence[str]) -> list[str]:
    """Render one Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _probe_declarations(configuration: Mapping[str, CanonicalValue]) -> list[str]:
    """Render what the user declared about each probe ladder, as their declaration."""
    lines = ["## Declared probes", ""]
    rows = []
    for raw in _items(configuration["probe_groups"]):
        group = _object(raw)
        rows.append(
            [
                markdown_code(group["probe_id"]),
                markdown_label(group["parameter"]),
                markdown_label(group["direction"]),
                markdown_label(group["required_detection_magnitude"]),
                markdown_label(group["magnitude_semantics"]),
            ]
        )
    lines.extend(
        _table(
            rows,
            ["probe group", "parameter", "direction", "required magnitude", "magnitude semantics"],
        )
    )
    lines.append("")
    lines.append(
        "Those five columns are the declaring user's own labels, preserved exactly. Metrifid "
        "compares the supplied admitted model closures; it does not establish that these labels "
        "faithfully describe the source edits, or that no other source change exists."
    )
    lines.append("")
    return lines


def _selected_section(receipt: Mapping[str, CanonicalValue]) -> list[str]:
    """Render the selected workloads and any exclusions."""
    lines = ["## Selected workloads", "", "```text"]
    lines.extend(markdown_block(item) for item in _items(receipt["selected_workload_ids"]))
    lines.append("```")
    lines.append("")
    eligible = _items(receipt["eligible_workload_ids"])
    excluded = _items(receipt["excluded_workload_ids"])
    lines.append(
        f"Eligible workloads: {len(eligible)}. "
        f"Excluded by a failed zero-change control: {len(excluded)}."
    )
    if excluded:
        lines.append("")
        for raw in _items(receipt["zero_change_controls"]):
            control = _object(raw)
            if not control["eligible"]:
                lines.append(
                    f"- {markdown_code(control['workload_id'])} — "
                    f"{markdown_label(control['exclusion_reason'])}"
                )
    lines.append("")
    return lines


def _probe_group_section(selection: Mapping[str, CanonicalValue]) -> list[str]:
    """Render the per-group table and the reason behind any missing floor."""
    lines = ["## Probe groups", ""]
    groups = _items(selection["groups"])
    rows = []
    for raw in groups:
        group = _object(raw)
        floor = group["floor_magnitude"]
        rows.append(
            [
                markdown_code(group["probe_id"]),
                markdown_label(group["status"]),
                " → ".join(markdown_label(item) for item in _items(group["detection_signature"])),
                "—" if floor is None else markdown_label(floor),
                markdown_label(group["detected_variants"]),
            ]
        )
    lines.extend(
        _table(rows, ["probe group", "status", "detection signature", "floor", "detected"])
    )
    lines.append("")
    lines.append(
        "The detection signature is ordered by increasing magnitude. A floor exists only when a "
        "rung and every larger rung are detected; a lone detection with a gap above it "
        "establishes nothing."
    )
    lines.append("")
    for raw in groups:
        group = _object(raw)
        if group["no_floor_reason"]:
            lines.append(
                f"- {markdown_code(group['probe_id'])}: {markdown_label(group['no_floor_reason'])}"
            )
    lines.append("")
    return lines


def _witness_section(witnesses: Mapping[str, CanonicalValue]) -> list[str]:
    """Render the status-bearing witness and the separate eligibility warning."""
    lines = ["## Witnesses", ""]
    first = witnesses.get("first_witness")
    if first is None:
        lines.append(
            "No witness explains this status, because every declared probe group qualified."
        )
    else:
        record = _object(first)
        lines.append("```text")
        for key in ("kind", "probe_id", "parameter", "magnitude", "workload_id"):
            value = record.get(key)
            if value is not None:
                lines.append(f"{key:14} {markdown_block(value)}")
        lines.append("```")
        lines.append("")
        lines.append(f"{markdown_label(record['detail'])}.")
    lines.append("")
    alarm = witnesses.get("first_false_alarm_witness")
    if alarm is not None:
        record = _object(alarm)
        lines.append(
            f"Separate eligibility warning: {markdown_code(record['workload_id'])} — "
            f"{markdown_label(record['detail'])}. This does not explain the completed status; the "
            "workload was excluded before selection."
        )
        lines.append("")
    lines.append(f"Witness order: {markdown_label(witnesses['witness_order'])}.")
    lines.append("")
    lines.append(markdown_label(witnesses["status_witness_rule"]))
    lines.append("")
    return lines


def _execution_section(receipt: Mapping[str, CanonicalValue]) -> list[str]:
    """Render planned and actual campaign cost."""
    counts = _object(receipt["execution_counts"])
    lines = ["## What was executed", "", "```text"]
    lines.append(f"{'planned_comparisons':26} {receipt['planned_comparisons']}")
    for key in (
        "zero_change_comparisons",
        "probe_comparisons",
        "total_comparisons",
        "completed_cells",
        "unresolved_cells",
        "failed_cells",
    ):
        lines.append(f"{key:26} {counts[key]}")
    lines.append(f"{'subsets_evaluated':26} {receipt['subsets_evaluated']}")
    lines.append("```")
    lines.append("")
    lines.append(
        "Planned comparisons are one zero-change control per workload plus one comparison per "
        "workload per declared rung, which schema version 1 bounds at 2064. Every cell above is one "
        "completed Metrifid comparison against a user-supplied model. The comparison count is the "
        "real cost of this answer; the three selected workloads are the result, not the cost."
    )
    lines.append("")
    return lines


def render_markdown(receipt: Mapping[str, CanonicalValue]) -> str:
    """Render the receipt without recomputing any decision."""
    status = str(receipt["status"])
    lines: list[str] = [
        "# Workload qualification",
        "",
        f"**{markdown_label(status)}** — exit {receipt['completed_exit_code']}",
        "",
        _STATUS_SENTENCE.get(status, ""),
        "",
        "## Question answered",
        "",
        "> Do the selected workloads detect every declared supplied probe closure at or above its "
        "user-declared required magnitude, under the declared comparison tolerances, and which "
        "declared probes remain blind or unresolved?",
        "",
    ]
    lines.extend(_probe_declarations(_object(receipt["configuration"])))
    lines.extend(_selected_section(receipt))
    lines.extend(_probe_group_section(_object(receipt["selection"])))
    lines.extend(_witness_section(_object(receipt["witnesses"])))
    lines.extend(_execution_section(receipt))
    lines.append("## Not claimed")
    lines.append("")
    lines.extend(f"- {markdown_label(sentence)}" for sentence in _items(receipt["not_claimed"]))
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("```text")
    lines.extend(markdown_block(code) for code in _items(receipt["limitations"]))
    lines.append("```")
    lines.append("")
    lines.append("## Identity")
    lines.append("")
    lines.append("```text")
    lines.append(f"receipt_sha256        {receipt['receipt_sha256']}")
    lines.append(f"configuration_sha256  {receipt['configuration_raw_sha256']}")
    lines.append("```")
    lines.append("")
    lines.append(
        "`receipt_sha256` detects accidental corruption of the canonical receipt content. It is "
        "not a signature, it authenticates no author or machine, and recomputing it cannot make a "
        "contradictory receipt valid."
    )
    lines.append("")
    return "\n".join(lines)
