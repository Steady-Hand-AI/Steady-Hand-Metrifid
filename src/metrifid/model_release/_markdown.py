"""Deterministic, injection-resistant Markdown for model-release receipts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..json_values import CanonicalValue
from ._decision import ChangeClassification
from ._receipt_validation import validate_model_release_receipt


def render_markdown(receipt: Mapping[str, CanonicalValue]) -> str:
    """Render every static compiled change without truncation or ambient metadata."""
    validate_model_release_receipt(receipt)
    certification = _object(receipt["certification_receipt"])
    baseline_artifact = _object(_object(certification["baseline"])["compiled_artifact"])
    candidate_artifact = _object(_object(certification["candidate"])["compiled_artifact"])
    policy = _object(receipt["policy"])
    registry = _object(receipt["public_field_registry"])
    lines = [
        "# Static compiled-model release review",
        "",
        "> [!WARNING]",
        "> **STATIC-ONLY DECISION.** This report does not establish dynamic equivalence, "
        "hardware safety, deployment safety, task suitability, or operational readiness.",
        f"> Dynamic behavior claim: `{receipt['dynamic_behavior_claim']}`.",
        "",
        "## Outcome",
        "",
        f"- Status: `{receipt['status']}`",
        f"- Completed exit code: `{receipt['completed_exit_code']}`",
        f"- Receipt SHA-256: `{receipt['receipt_sha256']}`",
        f"- Decision SHA-256: `{receipt['decision_sha256']}`",
        f"- Changes complete: `{str(receipt['changes_complete']).lower()}`",
        f"- Complete ordered change count: `{receipt['change_count']}`",
        "",
        str(_object(receipt["static_claim"])["statement"]),
        "",
        "## Certified artifact linkage",
        "",
        f"- Embedded Certify receipt SHA-256: `{receipt['certification_receipt_sha256']}`",
        f"- Embedded Certify decision SHA-256: `{receipt['certification_decision_sha256']}`",
        f"- Baseline complete-MJB SHA-256: `{baseline_artifact['mjb_sha256']}`",
        f"- Candidate complete-MJB SHA-256: `{candidate_artifact['mjb_sha256']}`",
        "",
        "## Policy and registry identity",
        "",
        f"- Policy raw SHA-256: `{policy['raw_sha256']}`",
        f"- Policy semantic SHA-256: `{policy['semantic_sha256']}`",
        f"- Policy baseline subject: `{policy['baseline_compiled_sha256']}`",
        f"- Policy candidate subject: `{_optional(policy['candidate_compiled_sha256'])}`",
        f"- Policy rules: `{policy['rule_count']}`",
        f"- Public-field registry schema: `{registry['schema']}` version "
        f"`{registry['schema_version']}`",
        f"- Public-field registry SHA-256: `{registry['sha256']}`",
        f"- Public-field registry fields: `{registry['field_count']}`",
        "",
    ]
    lines += _classification_section(receipt)
    lines += _change_section(receipt)
    lines += _requirement_section(receipt)
    lines += _witness_section(receipt)
    lines += _limitation_section(receipt)
    return "\n".join(lines) + "\n"


def _classification_section(receipt: Mapping[str, CanonicalValue]) -> list[str]:
    """Render all four classification counts in frozen enum order."""
    counts = _object(receipt["classification_counts"])
    lines = ["## Classification counts", ""]
    lines.extend(
        f"- `{classification.value}`: `{counts[classification.value]}`"
        for classification in ChangeClassification
    )
    lines.append("")
    return lines


def _change_section(receipt: Mapping[str, CanonicalValue]) -> list[str]:
    """Render exactly one row for every ordered change, including digest-side identities."""
    changes = _array(receipt["changes"])
    lines = [
        "## Complete ordered static changes (no truncation)",
        "",
        f"All `{len(changes)}` of `{receipt['change_count']}` changes are listed below in "
        "canonical first-witness order.",
        "",
    ]
    if not changes:
        return [*lines, "No compiled change was observed.", ""]
    lines += [
        "| # | Object type | Object name | Field | Kind | Source | Classification | Rule | Before SHA-256 | After SHA-256 |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, raw in enumerate(changes, 1):
        change = _object(raw)
        selector = _object(change["selector"])
        lines.append(
            f"| {index} | {_escape(selector['object_type'])} | "
            f"{_escape(selector['object_name'])} | {_escape(selector['field'])} | "
            f"{_escape(selector['change_kind'])} | {_escape(change['source'])} | "
            f"{_escape(change['classification'])} | {_escape_optional(change['rule_id'])} | "
            f"{_escape_optional(change['before_sha256'])} | "
            f"{_escape_optional(change['after_sha256'])} |"
        )
    lines.append("")
    return lines


def _requirement_section(receipt: Mapping[str, CanonicalValue]) -> list[str]:
    """Render the complete satisfied/missing REQUIRE-rule partition."""
    satisfied = _array(receipt["satisfied_required_rule_ids"])
    missing = _array(receipt["missing_required_rules"])
    lines = ["## REQUIRE rules", ""]
    if satisfied:
        lines += ["Satisfied REQUIRE rule IDs:", ""]
        lines.extend(f"- {_escape(item)}" for item in satisfied)
        lines.append("")
    else:
        lines += ["Satisfied REQUIRE rule IDs: none.", ""]
    if not missing:
        return [*lines, "Missing REQUIRE rules: none.", ""]
    lines += [
        "Missing REQUIRE rules:",
        "",
        "| Rule ID | Object type | Object name | Field | Kind | Before SHA-256 | After SHA-256 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for raw in missing:
        rule = _object(raw)
        selector = _object(rule["selector"])
        lines.append(
            f"| {_escape(rule['id'])} | {_escape(selector['object_type'])} | "
            f"{_escape(selector['object_name'])} | {_escape(selector['field'])} | "
            f"{_escape(selector['change_kind'])} | "
            f"{_escape_optional(rule['before_sha256'])} | "
            f"{_escape_optional(rule['after_sha256'])} |"
        )
    lines.append("")
    return lines


def _witness_section(receipt: Mapping[str, CanonicalValue]) -> list[str]:
    """Render both deterministic first-witness channels without changing their order."""
    return [
        "## Deterministic first witnesses",
        "",
        "- First forbidden or undeclared change: "
        + _change_witness(receipt["first_unexpected_witness"]),
        "- First missing REQUIRE rule: " + _rule_witness(receipt["first_missing_required_witness"]),
        "",
    ]


def _change_witness(value: CanonicalValue) -> str:
    """Render a compact first-change witness or explicit absence."""
    if value is None:
        return "none."
    change = _object(value)
    selector = _object(change["selector"])
    return (
        f"`{_escape(change['classification'])}` at "
        f"`{_escape(selector['object_type'])}/{_escape(selector['object_name'])}/"
        f"{_escape(selector['field'])}/{_escape(selector['change_kind'])}`."
    )


def _rule_witness(value: CanonicalValue) -> str:
    """Render a compact first-missing-rule witness or explicit absence."""
    if value is None:
        return "none."
    rule = _object(value)
    selector = _object(rule["selector"])
    return (
        f"`{_escape(rule['id'])}` at "
        f"`{_escape(selector['object_type'])}/{_escape(selector['object_name'])}/"
        f"{_escape(selector['field'])}/{_escape(selector['change_kind'])}`."
    )


def _limitation_section(receipt: Mapping[str, CanonicalValue]) -> list[str]:
    """Render every frozen limitation in receipt order and repeat the safety boundary."""
    lines = [
        "## Limitations",
        "",
        "This remains a static-only classification. It is not a dynamic-equivalence or safety "
        "certification.",
        "",
    ]
    for raw in _array(receipt["limitations"]):
        entry = _object(raw)
        lines.append(f"- `{entry['code']}`: {entry['statement']}")
    lines.append("")
    return lines


def _escape(value: CanonicalValue) -> str:
    """Escape user/model-controlled text for Markdown table and inline-code placement."""
    if not isinstance(value, str):
        raise TypeError("Markdown text value must be a string")
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


def _optional(value: CanonicalValue) -> str:
    """Render an optional non-user-controlled scalar without a Python representation."""
    return "none" if value is None else str(value)


def _escape_optional(value: CanonicalValue) -> str:
    """Escape one optional receipt string for a Markdown table cell."""
    return "none" if value is None else _escape(value)


def _object(value: CanonicalValue) -> Mapping[str, CanonicalValue]:
    """Require a canonical object before rendering nested evidence."""
    if type(value) is not dict:
        raise TypeError("receipt member must be an object")
    return value


def _array(value: CanonicalValue) -> Sequence[CanonicalValue]:
    """Require a canonical array before rendering complete repeated evidence."""
    if type(value) is not list:
        raise TypeError("receipt member must be an array")
    return value


__all__ = ["render_markdown"]
