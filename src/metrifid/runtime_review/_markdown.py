"""Render the explanatory, non-decision-bearing Runtime Review Markdown report."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..json_values import CanonicalValue

_STATUS_TEXT: Mapping[str, str] = {
    "WITHIN_DECLARED_MIGRATION_ENVELOPE": (
        "Every admitted witness across the complete declared horizon is enclosed by its declared "
        "migration tolerance."
    ),
    "INSUFFICIENT_EVIDENCE": (
        "The evidence does not qualify the entire declared horizon, so replacement is not green."
    ),
    "UNRESOLVED_NEAR_BOUNDARY": (
        "At least one complete-horizon witness overlaps its tolerance boundary, so replacement "
        "remains unresolved."
    ),
    "OUTSIDE_DECLARED_MIGRATION_ENVELOPE": (
        "At least one decisive admitted witness lies outside its declared migration tolerance."
    ),
}


def render_runtime_review_markdown(receipt: Mapping[str, CanonicalValue]) -> str:
    """Render one canonical receipt without recomputing or changing its decision."""
    status = _text(receipt["status"])
    reason = receipt["reason_code"]
    configuration = _object(receipt["configuration"])
    campaign = _object(receipt["campaign_shape"])
    counts = _object(receipt["witness_counts"])
    lines = [
        "# Native Runtime Review",
        "",
        "## Decision",
        "",
        f"- Status: `{_escape(status)}`",
        f"- Reason: `{_escape(reason) if reason is not None else 'none'}`",
        f"- Method: `{_escape(receipt['method'])}`",
        f"- Required horizon: `{_escape(receipt['required_horizon'])}` seconds",
        f"- Admitted prefix: `{_escape(receipt['admitted_prefix'])}` seconds",
        "",
        _STATUS_TEXT.get(status, "The receipt contains an unrecognized completed status."),
        "",
        "## Exact question answered",
        "",
        "> Can this exact candidate MuJoCo native runtime profile replace this exact baseline "
        "profile for this exact source closure and workload over the entire declared horizon, "
        "under the user-declared channel tolerances, based on complete three-grid/two-repeat "
        "evidence?",
        "",
        "A green decision requires a complete admitted prefix equal to the required horizon, no "
        "failing gate, no unqualified suffix, and no unresolved or outside witness.",
        "",
        "## Campaign and configuration",
        "",
        f"- Admitted configuration: `{_escape(configuration['locator'])}`",
        f"- Configuration raw SHA-256: `{_escape(configuration['raw_sha256'])}`",
        f"- Configuration semantic SHA-256: `{_escape(configuration['semantic_sha256'])}`",
        f"- Step sizes: `{', '.join(_escape(item) for item in _strings(campaign['step_dts']))}`",
        f"- Repeats: `{', '.join(str(item) for item in _integers(campaign['repeat_ids']))}`",
        f"- Owned evidence cells: `{campaign['cell_count']}`",
        "",
        "## Decision witnesses",
        "",
    ]
    lines.extend(_named_witness("First decisive witness", receipt["first_decisive_witness"]))
    lines.extend(_named_witness("Worst witness", receipt["worst_witness"]))
    lines.extend(_gate(receipt["first_failing_gate"]))
    lines.extend(
        [
            "### Witness counts",
            "",
            "| Classification | Count |",
            "| --- | ---: |",
        ]
    )
    lines.extend(f"| `{_escape(name)}` | `{value}` |" for name, value in sorted(counts.items()))
    lines.extend(["", "### Complete stable witnesses", ""])
    witnesses = _array(receipt["witnesses"])
    if witnesses:
        lines.extend(
            [
                "| Time | Channel | Kind | Semantic type | Classification | Tolerance | Input SHA-256 |",
                "| ---: | --- | --- | --- | --- | ---: | --- |",
            ]
        )
        for value in witnesses:
            witness = _object(value)
            lines.append(
                f"| `{_escape(witness['time'])}` | `{_escape(witness['channel_id'])}` | "
                f"`{_escape(witness['kind'])}` | `{_escape(witness['semantic_type'])}` | "
                f"`{_escape(witness['classification'])}` | "
                f"`{_escape(witness['tolerance'])}` | "
                f"`{_escape(witness['decision_input_sha256'])}` |"
            )
    else:
        lines.append("No witness lies inside the admitted decision prefix.")
    lines.extend(
        [
            "",
            "The witness table contains only stable decision fields. Platform-sensitive binary64 "
            "interval endpoints, ratios, and magnitudes are deliberately non-normative and are not "
            "part of the canonical receipt.",
            "",
            "## Portable evidence snapshot",
            "",
            "The twelve evidence cells listed in the canonical JSON are independent byte copies, "
            "not hard links or symbolic links. The independent validator follows that owned "
            "slot-to-cell mapping and does not need the original configuration paths.",
            "",
            "## Limitations and non-claims",
            "",
        ]
    )
    lines.extend(f"- {_escape(value)}" for value in _strings(receipt["limitations"]))
    lines.extend(
        [
            "",
            "## Receipt identity",
            "",
            f"- Schema: `{_escape(receipt['schema'])}` version `{receipt['schema_version']}`",
            f"- Claim scope: `{_escape(receipt['claim_scope'])}`",
            f"- Receipt SHA-256: `{_escape(receipt['receipt_sha256'])}`",
            "",
            "`receipt_sha256` is computed over canonical JSON with the self-field omitted. It "
            "detects accidental corruption but does not authenticate a campaign, author, or "
            "machine.",
            "",
        ]
    )
    return "\n".join(lines)


def _named_witness(title: str, value: CanonicalValue) -> list[str]:
    """Render one optional stable witness as a focused subsection."""
    lines = [f"### {title}", ""]
    if value is None:
        return [*lines, "None.", ""]
    witness = _object(value)
    return [
        *lines,
        f"- Time: `{_escape(witness['time'])}` seconds",
        f"- Channel: `{_escape(witness['channel_id'])}`",
        f"- Classification: `{_escape(witness['classification'])}`",
        f"- Decision-input SHA-256: `{_escape(witness['decision_input_sha256'])}`",
        "",
    ]


def _gate(value: CanonicalValue) -> list[str]:
    """Render one optional first failing gate without treating it as a witness."""
    lines = ["### First failing gate", ""]
    if value is None:
        return [*lines, "None.", ""]
    gate = _object(value)
    return [
        *lines,
        f"- Time: `{_escape(gate['time'])}` seconds",
        f"- Status: `{_escape(gate['status'])}`",
        f"- Channel: `{_escape(gate['channel_id'])}`",
        "",
    ]


def _object(value: CanonicalValue) -> Mapping[str, CanonicalValue]:
    """Require a canonical object before rendering nested fields."""
    if type(value) is not dict:
        raise TypeError("Markdown receipt field must be an object")
    return value


def _array(value: CanonicalValue) -> Sequence[CanonicalValue]:
    """Require a canonical array before rendering repeated fields."""
    if type(value) is not list:
        raise TypeError("Markdown receipt field must be an array")
    return value


def _strings(value: CanonicalValue) -> tuple[str, ...]:
    """Require one canonical array to contain only text tokens."""
    values = _array(value)
    if any(type(item) is not str for item in values):
        raise TypeError("Markdown receipt array must contain strings")
    return tuple(str(item) for item in values)


def _integers(value: CanonicalValue) -> tuple[int, ...]:
    """Require one canonical array to contain strict integers."""
    values = _array(value)
    result: list[int] = []
    for item in values:
        if type(item) is not int:
            raise TypeError("Markdown receipt array must contain integers")
        result.append(item)
    return tuple(result)


def _text(value: CanonicalValue) -> str:
    """Require one receipt scalar to be text."""
    if type(value) is not str:
        raise TypeError("Markdown receipt field must be text")
    return value


def _escape(value: CanonicalValue) -> str:
    """Escape untrusted canonical text for Markdown inline and table contexts."""
    if type(value) is not str:
        raise TypeError("Markdown value must be text")
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("`", "&#96;")
        .replace("!", "&#33;")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace("(", "&#40;")
        .replace(")", "&#41;")
    )


__all__ = ["render_runtime_review_markdown"]
