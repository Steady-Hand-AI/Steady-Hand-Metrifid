"""Source-shaped findings and limitations read from one validated receipt.

Nothing here reinterprets a decision.  Every finding carries the producer's own canonical row
unchanged in ``data``, an exact JSON Pointer to where that row lives in the receipt, and display
text derived only from values the producer already retained.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from ..json_values import CanonicalValue
from ._display import describe_value, value_text

__all__ = [
    "ATTRIBUTION",
    "CLAIM",
    "POLICY",
    "PRESENTATION",
    "PRODUCER",
    "certification_findings",
    "certification_limitations",
    "json_pointer",
    "model_release_findings",
    "model_release_limitations",
]

CLAIM: Final = "claim"
PRODUCER: Final = "producer"
ATTRIBUTION: Final = "attribution"
POLICY: Final = "policy"
PRESENTATION: Final = "presentation"

_KIND_BY_SOURCE: Final = {
    "SEMANTIC_OBJECT": "named",
    "COMPILED_PUBLIC_FIELD": "compiled_field",
    "OPAQUE_ARTIFACT_RESIDUAL": "residual",
}
_APPEARANCE_TEXT: Final = {"ADD": "added", "REMOVE": "removed"}
_UNEXPLAINED_REASON: Final = "no_public_or_semantic_field_difference_identified"
_POLICY_REASONS: Final = frozenset(
    {"candidate_compiled_subject_unbound", "candidate_compiled_subject_mismatch"}
)
_REASON_STATEMENTS: Final = {
    "candidate_compiled_subject_unbound": (
        "The declared policy bound no candidate compiled artifact, so the candidate was not a "
        "declared subject of this evaluation."
    ),
    "candidate_compiled_subject_mismatch": (
        "The declared policy bound a different candidate compiled artifact than the one compiled."
    ),
    "no_public_or_semantic_field_difference_identified": (
        "The compiled artifacts differ in bytes with no public-field or named difference "
        "identified to explain the difference."
    ),
    "semantic_name_coverage_incomplete": (
        "Some compiled objects carry no name, so their changes cannot be attributed to a named "
        "object."
    ),
    "compiled_name_identity_mapping_changed": (
        "A compiled field that carries the name-to-index mapping changed, so index-based "
        "attribution across the two models is not reliable."
    ),
}


def json_pointer(*tokens: str | int) -> str:
    """Return the RFC 6901 pointer for one exact receipt location.

    Args:
        tokens: Member names and array indices, outermost first.

    Returns:
        The escaped JSON Pointer naming that location.
    """
    escaped = [str(token).replace("~", "~0").replace("/", "~1") for token in tokens]
    return "".join(f"/{token}" for token in escaped)


def _mapping(value: CanonicalValue) -> Mapping[str, CanonicalValue]:
    """Return one canonical object, or an empty mapping when absent."""
    return value if isinstance(value, dict) else {}


def _sequence(value: CanonicalValue) -> Sequence[CanonicalValue]:
    """Return one canonical array, or an empty sequence when absent."""
    return value if isinstance(value, list) else ()


def _text(value: CanonicalValue) -> str:
    """Return one canonical string member, or the empty string when absent."""
    return value if isinstance(value, str) else ""


def model_release_findings(
    receipt: Mapping[str, CanonicalValue],
) -> list[dict[str, CanonicalValue]]:
    """Build one finding for every canonical change row the receipt retained.

    Every row survives, including a row for which no descriptive witness exists.  A compiled-field
    row is joined to the certification report's descriptive row by exact field path, and that row
    is attached whole rather than rebuilt.

    Args:
        receipt: One validated model-release receipt.

    Returns:
        One finding per canonical change, in the receipt's own order.
    """
    descriptive = _descriptive_index(receipt)
    findings: list[dict[str, CanonicalValue]] = []
    for index, row in enumerate(_sequence(receipt.get("changes"))):
        change = _mapping(row)
        source_tag = _text(change.get("source"))
        kind = _KIND_BY_SOURCE.get(source_tag, "residual")
        finding: dict[str, CanonicalValue] = {
            "kind": kind,
            "source": json_pointer("changes", index),
            "data": row,
            "display": _change_display(change, kind),
        }
        matched = descriptive.get(_join_key(change)) if kind == "compiled_field" else None
        if matched is not None:
            witness_row, witness_index = matched
            finding["descriptive"] = witness_row
            finding["descriptive_source"] = json_pointer(
                "certification_receipt", "field_report", "changed_fields", witness_index
            )
            finding["display"] = _witness_display(change, _mapping(witness_row))
        findings.append(finding)
    return findings


def _join_key(change: Mapping[str, CanonicalValue]) -> str | None:
    """Return the compiled field path a public-field row is attributable to."""
    selector = _mapping(change.get("selector"))
    if _text(selector.get("object_type")) != "compiled_field":
        return None
    if _text(selector.get("field")) != "value":
        return None
    return _text(selector.get("object_name")) or None


def _descriptive_index(
    receipt: Mapping[str, CanonicalValue],
) -> dict[str | None, tuple[CanonicalValue, int]]:
    """Index the certification report's descriptive rows by their exact field path."""
    certification = _mapping(receipt.get("certification_receipt"))
    report = _mapping(certification.get("field_report"))
    index: dict[str | None, tuple[CanonicalValue, int]] = {}
    for position, row in enumerate(_sequence(report.get("changed_fields"))):
        path = _text(_mapping(row).get("path"))
        if path:
            index[path] = (row, position)
    return index


def _change_display(change: Mapping[str, CanonicalValue], kind: str) -> str:
    """Render one canonical change row without inventing a value it does not carry."""
    selector = _mapping(change.get("selector"))
    object_type = _text(selector.get("object_type"))
    object_name = _text(selector.get("object_name"))
    field = _text(selector.get("field"))
    change_kind = _text(selector.get("change_kind"))
    label = f"{object_type} {object_name} {field}".strip()
    if kind == "residual":
        reasons = [
            _text(reason) for reason in _sequence(_mapping(change.get("details")).get("reasons"))
        ]
        listed = ", ".join(reason for reason in reasons if reason)
        # Only one recorded reason means the difference itself is unexplained. The others are
        # conditions of the evaluation or limits on attribution, and calling those an unexplained
        # difference would overstate what was observed.
        headline = (
            "compiled difference with no identified explanation"
            if _UNEXPLAINED_REASON in reasons
            else "fail-closed residual recorded"
        )
        return f"{label}: {headline} ({listed})" if listed else f"{label}: {headline}"
    before = change.get("before_value")
    after = change.get("after_value")
    appearance = _APPEARANCE_TEXT.get(change_kind)
    if appearance is not None:
        # An added or removed object is retained on one side only. Saying the other side was not
        # retained would read as missing evidence rather than as an absent object.
        retained = after if change_kind == "ADD" else before
        subject = f"{object_type} {object_name}".strip()
        if retained is None:
            return f"{subject}: {appearance}"
        return f"{subject}: {appearance}, {value_text(retained)}"
    if before is None and after is None:
        return f"{label}: changed, no value retained in this row"
    return f"{label}: {describe_value(before, after)}"


def _witness_display(
    change: Mapping[str, CanonicalValue], witness_row: Mapping[str, CanonicalValue]
) -> str:
    """Render a compiled-field row using the descriptive witnesses joined to it."""
    selector = _mapping(change.get("selector"))
    label = f"compiled field {_text(selector.get('object_name'))}"
    if _text(selector.get("change_kind")) in _APPEARANCE_TEXT:
        return _change_display(change, "compiled_field")
    witnesses = _sequence(witness_row.get("witnesses"))
    if not witnesses:
        return f"{label}: changed, no witness retained"
    first = _mapping(witnesses[0])
    rendered = describe_value(first.get("baseline_value"), first.get("candidate_value"))
    changed = witness_row.get("changed_element_count")
    remaining = len(witnesses) - 1
    suffix = f", {remaining} more witness" + ("es" if remaining > 1 else "") if remaining else ""
    counted = f" of {value_text(changed)} changed element(s)" if changed is not None else ""
    return f"{label}: {rendered}{counted}{suffix}"


def certification_findings(
    receipt: Mapping[str, CanonicalValue],
) -> list[dict[str, CanonicalValue]]:
    """Build one finding for every descriptive changed field a certification retained.

    A certification carries no policy classification and no canonical change total, so nothing is
    invented here: the findings are exactly the retained descriptive rows.

    Args:
        receipt: One validated certification receipt.

    Returns:
        One finding per retained descriptive row, in the receipt's own order.
    """
    report = _mapping(receipt.get("field_report"))
    findings: list[dict[str, CanonicalValue]] = []
    for index, row in enumerate(_sequence(report.get("changed_fields"))):
        entry = _mapping(row)
        findings.append(
            {
                "kind": "compiled_field",
                "source": json_pointer("field_report", "changed_fields", index),
                "data": row,
                "display": _descriptive_display(entry),
            }
        )
    return findings


def _descriptive_display(entry: Mapping[str, CanonicalValue]) -> str:
    """Render one descriptive changed-field row from its own retained witnesses."""
    label = f"compiled field {_text(entry.get('path'))}"
    witnesses = _sequence(entry.get("witnesses"))
    if not witnesses:
        return f"{label}: changed, no witness retained"
    first = _mapping(witnesses[0])
    rendered = describe_value(first.get("baseline_value"), first.get("candidate_value"))
    changed = entry.get("changed_element_count")
    counted = f" of {value_text(changed)} changed element(s)" if changed is not None else ""
    remaining = len(witnesses) - 1
    suffix = f", {remaining} more witness" + ("es" if remaining > 1 else "") if remaining else ""
    return f"{label}: {rendered}{counted}{suffix}"


def _limitation(
    category: str, message: str, source: str, **detail: CanonicalValue
) -> dict[str, CanonicalValue]:
    """Build one limitation entry with its category, message and exact source locator."""
    entry: dict[str, CanonicalValue] = {
        "category": category,
        "message": message,
        "source": source,
    }
    retained = {name: value for name, value in detail.items() if value is not None}
    if retained:
        entry["detail"] = dict(retained)
    return entry


def _claim_limitations(
    receipt: Mapping[str, CanonicalValue], prefix: tuple[str, ...]
) -> list[dict[str, CanonicalValue]]:
    """Carry the producer's own claim limitations through unchanged."""
    entries: list[dict[str, CanonicalValue]] = []
    for index, row in enumerate(_sequence(receipt.get("limitations"))):
        limitation = _mapping(row)
        entries.append(
            _limitation(
                CLAIM,
                _text(limitation.get("statement")),
                json_pointer(*prefix, "limitations", index),
                code=limitation.get("code"),
            )
        )
    return entries


def _report_limitations(
    certification: Mapping[str, CanonicalValue], prefix: tuple[str, ...]
) -> list[dict[str, CanonicalValue]]:
    """Report what the field producer omitted or truncated, with its own counts.

    A pair whose compiled bytes differ with no descriptive report retained is a real gap and is
    reported as one. An identical pair has no per-field report to produce, so nothing is claimed
    about per-field coverage either way.
    """
    entries: list[dict[str, CanonicalValue]] = []
    raw_report = certification.get("field_report")
    if not isinstance(raw_report, dict):
        if _mapping(certification.get("byte_comparison")).get("equal") is False:
            entries.append(
                _limitation(
                    PRODUCER,
                    "The compiled artifacts differ but no descriptive field report was retained, "
                    "so no per-field coverage is recorded for this pair.",
                    json_pointer(*prefix, "field_report"),
                )
            )
        return entries
    report = raw_report
    omitted = report.get("fields_omitted_count")
    if isinstance(omitted, int) and omitted > 0:
        entries.append(
            _limitation(
                PRODUCER,
                f"{omitted} compiled member(s) were omitted from the field-level report, each "
                "with a recorded reason. Some reasons, such as expansion one level below, mean "
                "the member is described by its compared child fields rather than by a row of "
                "its own. This bounds the field-level explanation only; the compiled-byte "
                "comparison is recorded separately.",
                json_pointer(*prefix, "field_report", "omitted_fields"),
                fields_omitted_count=omitted,
                fields_compared_count=report.get("fields_compared_count"),
                # Which members were omitted, and why, is the answer to "what was not covered".
                # The reason on each row is what separates a member described through its children
                # from one the producer could not describe at all, so it is retained verbatim.
                omitted_fields=report.get("omitted_fields"),
            )
        )
    if report.get("truncated") is True:
        entries.append(
            _limitation(
                PRODUCER,
                "The producer truncated its descriptive field report, so some changed fields "
                "carry no retained witness.",
                json_pointer(*prefix, "field_report", "changed_fields"),
                changed_fields_total=report.get("changed_fields_total"),
                changed_fields_returned=report.get("changed_fields_returned"),
            )
        )
    return entries


def certification_limitations(
    receipt: Mapping[str, CanonicalValue],
) -> list[dict[str, CanonicalValue]]:
    """Collect every limitation a standalone certification receipt records.

    Args:
        receipt: One validated certification receipt.

    Returns:
        The claim and producer limitations retained by that receipt.
    """
    entries = _claim_limitations(receipt, ())
    entries.extend(_report_limitations(receipt, ()))
    return entries


def model_release_limitations(
    receipt: Mapping[str, CanonicalValue],
) -> list[dict[str, CanonicalValue]]:
    """Collect every limitation a model-release receipt records, by category.

    Producer omissions, attribution limits and policy or subject conditions stay separate, so a
    condition of the evaluation is never read as an observed physical change.

    Args:
        receipt: One validated model-release receipt.

    Returns:
        The claim, producer, attribution and policy limitations retained by that receipt.
    """
    entries = _claim_limitations(receipt, ())
    certification = _mapping(receipt.get("certification_receipt"))
    # The embedded certification states its own boundaries, and they do not repeat the
    # model-release ones. Dropping them would present a narrower claim than the producer made.
    entries.extend(_claim_limitations(certification, ("certification_receipt",)))
    entries.extend(_report_limitations(certification, ("certification_receipt",)))
    entries.extend(_residual_limitations(receipt))
    entries.extend(_rule_limitations(receipt))
    if receipt.get("changes_complete") is False:
        entries.append(
            _limitation(
                PRODUCER,
                "The producer reported its canonical change list as incomplete.",
                json_pointer("changes_complete"),
                change_count=receipt.get("change_count"),
            )
        )
    return entries


def _residual_limitations(
    receipt: Mapping[str, CanonicalValue],
) -> list[dict[str, CanonicalValue]]:
    """Split each fail-closed residual reason into its own categorized limitation."""
    entries: list[dict[str, CanonicalValue]] = []
    for index, row in enumerate(_sequence(receipt.get("changes"))):
        change = _mapping(row)
        if _text(change.get("source")) != "OPAQUE_ARTIFACT_RESIDUAL":
            continue
        details = _mapping(change.get("details"))
        pointer = json_pointer("changes", index, "details")
        for reason in _sequence(details.get("reasons")):
            code = _text(reason)
            if not code:
                continue
            policy_condition = code in _POLICY_REASONS
            # Naming coverage is evidence for an attribution limit. Attaching it to a policy
            # condition would make a binding decision read as a measurement of the models.
            coverage: dict[str, CanonicalValue] = (
                {}
                if policy_condition
                else {
                    "baseline_coverage_issues": details.get("baseline_coverage_issues") or None,
                    "candidate_coverage_issues": details.get("candidate_coverage_issues") or None,
                }
            )
            entries.append(
                _limitation(
                    POLICY if policy_condition else ATTRIBUTION,
                    _REASON_STATEMENTS.get(
                        code, f"The evaluation recorded the residual reason {code}."
                    ),
                    pointer,
                    reason=code,
                    policy_candidate_compiled_sha256=(
                        details.get("policy_candidate_compiled_sha256")
                        if policy_condition
                        else None
                    ),
                    **coverage,
                )
            )
    return entries


def _rule_limitations(receipt: Mapping[str, CanonicalValue]) -> list[dict[str, CanonicalValue]]:
    """Report declared rules the evaluation did not observe being satisfied."""
    missing = _sequence(receipt.get("missing_required_rules"))
    if not missing:
        return []
    # Each entry is a whole rule object, so the identity a maintainer recognizes is its id.
    identities = [_text(_mapping(rule).get("id")) for rule in missing]
    named = ", ".join(identity for identity in identities if identity)
    listed = f": {named}" if named else ""
    return [
        _limitation(
            POLICY,
            f"The declared policy required {len(missing)} change(s) that were not observed{listed}.",
            json_pointer("missing_required_rules"),
            missing_required_rules=list(missing),
        )
    ]
