"""Deterministic Markdown rendering of one compiled-equivalence receipt.

The Markdown restates the receipt and nothing else. It carries no timestamp, no temporary path
and no process identifier, so two identical invocations produce identical text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..json_values import CanonicalValue
from ._receipt import BEHAVIOR_IMPLICATION_PREMISES, NOT_CERTIFIED_GUIDANCE
from ._status import CertifyStatus


def render_markdown(receipt: Mapping[str, CanonicalValue]) -> str:
    """Render the complete receipt as Markdown."""
    status = CertifyStatus(str(receipt["status"]))
    lines: list[str] = [
        "# Compiled artifact certification",
        "",
        f"- Status: `{status.value}`",
        f"- Completed exit code: `{receipt['completed_exit_code']}`",
        f"- Receipt schema: `{receipt['schema_version']}`",
        f"- Receipt SHA-256: `{receipt['receipt_sha256']}`",
        f"- Decision SHA-256: `{receipt['decision_sha256']}`",
        "",
    ]
    lines += _claim_section(receipt)
    lines += _artifact_section(receipt)
    lines += _comparison_section(receipt)
    lines += _runtime_section(receipt)
    lines += _field_section(receipt)
    lines += _limitations_section(receipt)
    return "\n".join(lines) + "\n"


def _claim_section(receipt: Mapping[str, CanonicalValue]) -> list[str]:
    """Render certification status, claim, implication, and receipt identity."""
    claim = _object(receipt["artifact_claim"])
    lines = [
        "## Claim",
        "",
        f"{claim['statement']}",
        "",
        "This claim is unconditional and workload-free. It does not claim "
        + _series([str(item) for item in _array(claim["does_not_claim"])])
        + ".",
        "",
    ]
    guidance = claim.get("guidance")
    if guidance is not None:
        lines += [str(guidance), ""]
    implication = receipt["behavior_implication"]
    if isinstance(implication, dict):
        lines += [
            "## Behavior implication (conditional, not part of the decision)",
            "",
            str(implication["statement"]),
            "",
            "Every premise must hold:",
            "",
        ]
        lines += [
            f"{index}. {premise}" for index, premise in enumerate(BEHAVIOR_IMPLICATION_PREMISES, 1)
        ]
        lines += [
            "",
            "This implication is excluded from `decision_sha256` and never changes the status.",
            "",
        ]
    elif claim.get("guidance") is None:
        lines += [NOT_CERTIFIED_GUIDANCE, ""]
    return lines


def _artifact_section(receipt: Mapping[str, CanonicalValue]) -> list[str]:
    """Render both model closures and compiled MJB artifact identities."""
    lines = [
        "## Compiled artifacts",
        "",
        "| Role | Entrypoint | Closure members | MJB bytes | MJB SHA-256 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for role in ("baseline", "candidate"):
        entry = _object(receipt[role])
        closure = _object(entry["source_closure"])
        artifact = _object(entry["compiled_artifact"])
        lines.append(
            f"| {entry['role']} | `{closure['entrypoint']}` | {closure['member_count']} "
            f"| {artifact['mjb_size_bytes']} | `{artifact['mjb_sha256']}` |"
        )
    lines.append("")
    baseline_artifact = _object(_object(receipt["baseline"])["compiled_artifact"])
    lines += [
        f"- Identity method: `{baseline_artifact['method']}`",
        f"- MJB header words: `{list(_array(baseline_artifact['header_words']))}`",
        f"- Magic: {baseline_artifact['magic_decimal']} (`{baseline_artifact['magic_hex']}`)",
        f"- `sizeof(mjtNum)`: {baseline_artifact['sizeof_mjtnum']}",
        f"- MuJoCo version word: {baseline_artifact['mujoco_version_integer']}",
        "",
        "Header words three and five are recorded exactly as MuJoCo wrote them and are treated "
        "as opaque build and layout words.",
        "",
    ]
    return lines


def _comparison_section(receipt: Mapping[str, CanonicalValue]) -> list[str]:
    """Render byte counts and first differing span from the artifact comparison."""
    comparison = _object(receipt["byte_comparison"])
    return [
        "## Complete byte comparison",
        "",
        f"- Byte-identical: `{str(comparison['equal']).lower()}`",
        f"- Baseline MJB bytes: {comparison['baseline_mjb_size_bytes']}",
        f"- Candidate MJB bytes: {comparison['candidate_mjb_size_bytes']}",
        f"- Compared byte count: {comparison['compared_byte_count']}",
        f"- First differing byte offset: {_optional(comparison['first_differing_byte_offset'])}",
        f"- Differing byte count: {comparison['differing_byte_count']}",
        "",
    ]


def _runtime_section(receipt: Mapping[str, CanonicalValue]) -> list[str]:
    """Render the MuJoCo, Python, NumPy, platform, and CPU runtime identity."""
    runtime = _object(receipt["runtime_identity"])
    tool = _object(receipt["tool"])
    return [
        "## Recorded runtime identity",
        "",
        f"- metrifid {runtime['metrifid_version']} (`{runtime['metrifid_distribution_sha256']}`)",
        f"- Tool execution identity: `{tool['execution_identity_state']}`",
        f"- MuJoCo {runtime['mujoco_version']} / native `{runtime['mujoco_version_string']}` "
        f"/ {runtime['mujoco_version_integer']}",
        f"- MuJoCo Python distribution SHA-256: `{runtime['mujoco_python_distribution_sha256']}`",
        f"- MuJoCo native library SHA-256: `{runtime['mujoco_native_library_sha256']}`",
        f"- {runtime['python_implementation']} {runtime['python_version']}, "
        f"NumPy {runtime['numpy_version']}",
        f"- {runtime['platform_system']} {runtime['platform_machine']} "
        f"{runtime['platform_release']}, libc `{runtime['libc']}`, "
        f"byte order `{runtime['byteorder']}`",
        f"- Execution mode: `{runtime['execution_mode']}`",
        f"- Runtime identity SHA-256: `{runtime['runtime_identity_sha256']}`",
        "",
    ]


def _field_section(receipt: Mapping[str, CanonicalValue]) -> list[str]:
    """Render changed and omitted public-field evidence tables."""
    report = receipt["field_report"]
    if not isinstance(report, dict):
        return []
    lines = [
        "## Descriptive public field report",
        "",
        "This section is descriptive evidence. It cannot change the status.",
        "",
        f"- Field report status: `{report['field_report_status']}`",
        f"- Fields compared: {report['fields_compared_count']}",
        f"- Fields omitted: {report['fields_omitted_count']}",
        f"- Changed fields total: {report['changed_fields_total']}",
        f"- Changed fields returned: {report['changed_fields_returned']}",
        f"- Truncated: `{str(report['truncated']).lower()}`",
        "",
    ]
    changed = _array(report["changed_fields"])
    if changed:
        lines += [
            "| Field | Baseline | Candidate | Changed elements | First witness |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines += [_changed_row(_object(entry)) for entry in changed]
        lines.append("")
    omitted = _array(report["omitted_fields"])
    if omitted:
        lines += ["Omitted members:", ""]
        lines += [f"- `{_object(item)['path']}`: `{_object(item)['reason']}`" for item in omitted]
        lines.append("")
    return lines


def _changed_row(entry: Mapping[str, CanonicalValue]) -> str:
    """Render one changed-field summary row with bounded witness counts."""
    witnesses = _array(entry["witnesses"])
    if witnesses:
        first = _object(witnesses[0])
        witness = (
            f"`{list(_array(first['index']))}` {first['baseline_text']} -> "
            f"{first['candidate_text']}"
        )
    else:
        witness = "none"
    return (
        f"| `{entry['path']}` | {_describe(entry, 'baseline')} | {_describe(entry, 'candidate')} "
        f"| {_optional(entry['changed_element_count'])} | {witness} |"
    )


def _describe(entry: Mapping[str, CanonicalValue], role: str) -> str:
    """Describe one role-local field side by kind, dtype, shape, and digest."""
    shape = entry[f"{role}_shape"]
    dtype = entry[f"{role}_dtype"]
    if shape is None:
        return f"`{entry[f'{role}_type']}`"
    return f"`{dtype}{list(_array(shape))}`"


def _limitations_section(receipt: Mapping[str, CanonicalValue]) -> list[str]:
    """Render every frozen certification limitation in registry order."""
    lines = ["## Limitations", ""]
    for item in _array(receipt["limitations"]):
        entry = _object(item)
        lines.append(f"- `{entry['code']}`: {entry['statement']}")
    lines.append("")
    return lines


def _series(items: list[str]) -> str:
    """Join a closed list so the sentence reads as the exhaustive list it is."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def _optional(value: CanonicalValue) -> str:
    """Render an absent value as `none` rather than leaking a Python repr."""
    return "none" if value is None else str(value)


def _object(value: CanonicalValue) -> Mapping[str, CanonicalValue]:
    """Require a canonical object before rendering nested receipt evidence."""
    if not isinstance(value, dict):
        raise TypeError("receipt member must be an object")
    return value


def _array(value: CanonicalValue) -> Sequence[CanonicalValue]:
    """Require a canonical array before rendering repeated receipt evidence."""
    if not isinstance(value, list):
        raise TypeError("receipt member must be an array")
    return value


__all__ = ["render_markdown"]
