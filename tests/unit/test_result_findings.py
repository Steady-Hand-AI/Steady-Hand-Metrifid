"""Source-shaped findings and categorized limitations built over one validated receipt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest

from metrifid.json_values import CanonicalValue
from metrifid.result._findings import (
    ATTRIBUTION,
    CLAIM,
    POLICY,
    PRODUCER,
    certification_findings,
    certification_limitations,
    json_pointer,
    model_release_findings,
    model_release_limitations,
)

_EVIDENCE_ROOT: Final = Path(__file__).parents[1] / "fixtures" / "result"
_CHANGED_RELEASE: Final = _EVIDENCE_ROOT / "changed_model_release.json"
_EQUAL_RELEASE: Final = _EVIDENCE_ROOT / "equal_model_release.json"
_CERTIFICATION: Final = _EVIDENCE_ROOT / "changed_certification.json"

_BEFORE_SHA: Final = "0" * 64
_AFTER_SHA: Final = "1" * 64
_ONE_POINT_FIVE: Final = {"kind": "ieee754_binary64", "bits": "3ff8000000000000"}
_TWO: Final = {"kind": "ieee754_binary64", "bits": "4000000000000000"}

_KNOWN_RESIDUAL_REASONS: Final = (
    "candidate_compiled_subject_unbound",
    "candidate_compiled_subject_mismatch",
    "semantic_name_coverage_incomplete",
    "compiled_name_identity_mapping_changed",
    "no_public_or_semantic_field_difference_identified",
)
_UNKNOWN_RESIDUAL_REASON: Final = "a_reason_code_added_after_this_release"
_FALLBACK_STATEMENT_PREFIX: Final = "The evaluation recorded the residual reason"


def _resolve_pointer(document: CanonicalValue, pointer: str) -> CanonicalValue:
    """Resolve one RFC 6901 JSON Pointer against a receipt and return that exact node."""
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise AssertionError(f"not a JSON Pointer: {pointer!r}")
    node = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            node = node[int(token)]
        elif isinstance(node, dict):
            node = node[token]
        else:
            raise AssertionError(f"pointer {pointer!r} leaves the document at {token!r}")
    return node


def _real_receipt(path: Path) -> dict[str, CanonicalValue]:
    """Read one receipt produced by a real run, skipping when that evidence is absent."""
    if not path.is_file():
        pytest.skip(f"recorded receipt evidence is not present at {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _change_row(
    source: str,
    object_type: str,
    object_name: str,
    field: str,
    *,
    before: CanonicalValue = None,
    after: CanonicalValue = None,
    details: CanonicalValue = None,
) -> dict[str, CanonicalValue]:
    """Build one canonical model-release change row in the frozen receipt shape."""
    return {
        "source": source,
        "selector": {
            "object_type": object_type,
            "object_name": object_name,
            "field": field,
            "change_kind": "MODIFY",
        },
        "classification": "UNDECLARED",
        "rule_id": None,
        "before_value": before,
        "after_value": after,
        "before_sha256": _BEFORE_SHA,
        "after_sha256": _AFTER_SHA,
        "details": {} if details is None else details,
    }


def _witness(index: list[int]) -> dict[str, CanonicalValue]:
    """Build one descriptive witness carrying the retained 1.5 to 2.0 transition."""
    return {
        "index": index,
        "baseline_value": _ONE_POINT_FIVE,
        "baseline_text": "1.5",
        "candidate_value": _TWO,
        "candidate_text": "2.0",
    }


def _descriptive_row(
    path: str,
    *,
    witnesses: list[CanonicalValue] | None = None,
    changed_element_count: CanonicalValue = 1,
) -> dict[str, CanonicalValue]:
    """Build one certification changed-field row in the frozen descriptive shape."""
    return {
        "path": path,
        "baseline_type": "ndarray",
        "baseline_dtype": "float64",
        "baseline_shape": [1],
        "baseline_sha256": _BEFORE_SHA,
        "candidate_type": "ndarray",
        "candidate_dtype": "float64",
        "candidate_shape": [1],
        "candidate_sha256": _AFTER_SHA,
        "changed_element_count": changed_element_count,
        "witnesses": [_witness([0])] if witnesses is None else witnesses,
    }


def _field_report(
    changed_fields: list[CanonicalValue],
    *,
    truncated: bool = False,
    fields_omitted_count: int = 0,
    changed_fields_total: int | None = None,
    changed_fields_returned: int | None = None,
) -> dict[str, CanonicalValue]:
    """Build one descriptive compiled-field report in the frozen producer shape."""
    total = len(changed_fields) if changed_fields_total is None else changed_fields_total
    returned = len(changed_fields) if changed_fields_returned is None else changed_fields_returned
    return {
        "schema": "metrifid.compiled_field_report",
        "schema_version": 1,
        "field_report_status": "PUBLIC_FIELD_DIFFERENCES_IDENTIFIED",
        "changed_fields": changed_fields,
        "changed_fields_total": total,
        "changed_fields_returned": returned,
        "fields_compared_count": 598,
        "fields_omitted_count": fields_omitted_count,
        "omitted_fields": [],
        "truncated": truncated,
    }


def _certification(
    field_report: CanonicalValue,
    *,
    byte_equal: bool = False,
    limitations: list[CanonicalValue] | None = None,
) -> dict[str, CanonicalValue]:
    """Build one embedded certification receipt around a descriptive report."""
    return {
        "schema": "metrifid.certification_receipt",
        "byte_comparison": {"equal": byte_equal, "differing_byte_count": 0 if byte_equal else 71},
        "field_report": field_report,
        "limitations": [] if limitations is None else limitations,
    }


def _release_receipt(
    changes: list[CanonicalValue],
    *,
    certification: CanonicalValue = None,
    limitations: list[CanonicalValue] | None = None,
    missing_required_rules: list[CanonicalValue] | None = None,
) -> dict[str, CanonicalValue]:
    """Build one model-release receipt around a canonical change list."""
    return {
        "schema": "metrifid.model_release_receipt",
        "status": "REVIEW_REQUIRED",
        "changes": changes,
        "change_count": len(changes),
        "changes_complete": True,
        "limitations": [] if limitations is None else limitations,
        "missing_required_rules": [] if missing_required_rules is None else missing_required_rules,
        "certification_receipt": certification,
    }


def _compiled_change(object_name: str, *, field: str = "value") -> dict[str, CanonicalValue]:
    """Build one compiled public-field change row naming a compiled field path."""
    return _change_row("COMPILED_PUBLIC_FIELD", "compiled_field", object_name, field)


def _residual_change(reasons: list[CanonicalValue]) -> dict[str, CanonicalValue]:
    """Build one fail-closed opaque residual row carrying the given reason codes."""
    return _change_row(
        "OPAQUE_ARTIFACT_RESIDUAL",
        "opaque",
        "complete_mjb",
        "compiled_artifact",
        details={
            "reasons": reasons,
            "baseline_coverage_issues": [],
            "candidate_coverage_issues": [],
            "policy_candidate_compiled_sha256": None,
        },
    )


def test_every_real_change_row_becomes_one_finding_in_receipt_order() -> None:
    """Prove the real twelve-change receipt yields twelve findings in its own order."""
    receipt = _real_receipt(_CHANGED_RELEASE)
    findings = model_release_findings(receipt)
    assert len(receipt["changes"]) == 12
    assert len(findings) == 12
    expected_kinds = ["named", "named", *["compiled_field"] * 9, "residual"]
    assert [finding["kind"] for finding in findings] == expected_kinds
    assert [finding["source"] for finding in findings] == [f"/changes/{i}" for i in range(12)]


def test_each_canonical_source_tag_maps_to_its_own_finding_kind() -> None:
    """Prove the three canonical source tags map to named, compiled_field and residual."""
    receipt = _release_receipt(
        [
            _change_row(
                "SEMANTIC_OBJECT", "body", "link", "mass", before=_ONE_POINT_FIVE, after=_TWO
            ),
            _compiled_change("body_mass"),
            _residual_change(["no_public_or_semantic_field_difference_identified"]),
            _change_row("A_SOURCE_TAG_ADDED_LATER", "opaque", "future", "field"),
        ]
    )
    findings = model_release_findings(receipt)
    assert [finding["kind"] for finding in findings] == [
        "named",
        "compiled_field",
        "residual",
        "residual",
    ]
    assert findings[0]["display"] == "body link mass: 1.5 -> 2.0"


def test_every_finding_source_is_a_json_pointer_that_resolves_to_the_identical_change_row() -> None:
    """Prove each finding source pointer resolves to the very row it was built from."""
    receipt = _real_receipt(_CHANGED_RELEASE)
    changes = receipt["changes"]
    assert isinstance(changes, list)
    for index, finding in enumerate(model_release_findings(receipt)):
        source = finding["source"]
        assert isinstance(source, str)
        assert source == json_pointer("changes", index)
        assert _resolve_pointer(receipt, source) is changes[index]
        assert _resolve_pointer(receipt, source) is finding["data"]


def test_finding_data_carries_the_receipt_row_object_itself_and_never_a_rebuilt_copy() -> None:
    """Prove the data member is the identical canonical row object from the receipt."""
    receipt = _real_receipt(_CHANGED_RELEASE)
    changes = receipt["changes"]
    assert isinstance(changes, list)
    findings = model_release_findings(receipt)
    for index, finding in enumerate(findings):
        assert finding["data"] is changes[index]


def test_a_compiled_field_row_joins_the_descriptive_row_at_the_exact_same_field_path() -> None:
    """Prove an exact path match attaches the descriptive row whole and points at it."""
    descriptive = _descriptive_row("body_mass")
    receipt = _release_receipt(
        [_compiled_change("body_mass")],
        certification=_certification(_field_report([descriptive])),
    )
    finding = model_release_findings(receipt)[0]
    assert finding["descriptive"] is descriptive
    assert finding["descriptive_source"] == "/certification_receipt/field_report/changed_fields/0"
    source = finding["descriptive_source"]
    assert isinstance(source, str)
    assert _resolve_pointer(receipt, source) is descriptive
    assert finding["display"] == "compiled field body_mass: 1.5 -> 2.0 of 1 changed element(s)"


def test_exactly_nine_of_the_real_receipts_twelve_findings_carry_a_descriptive_row() -> None:
    """Prove the real receipt joins nine descriptive rows and leaves three unjoined."""
    receipt = _real_receipt(_CHANGED_RELEASE)
    findings = model_release_findings(receipt)
    joined = [finding for finding in findings if "descriptive" in finding]
    unjoined = [finding for finding in findings if "descriptive" not in finding]
    assert len(joined) == 9
    assert [finding["kind"] for finding in unjoined] == ["named", "named", "residual"]
    for finding in joined:
        pointer = finding["descriptive_source"]
        assert isinstance(pointer, str)
        assert pointer.startswith("/certification_receipt/field_report/changed_fields/")
        row = _resolve_pointer(receipt, pointer)
        assert row is finding["descriptive"]
        assert isinstance(row, dict)
        data = finding["data"]
        assert isinstance(data, dict)
        selector = data["selector"]
        assert isinstance(selector, dict)
        assert row["path"] == selector["object_name"]


def test_a_compiled_field_row_without_a_descriptive_row_survives_and_retains_no_value() -> None:
    """Prove an absent descriptive row never deletes or invents a canonical change."""
    described = _compiled_change("body_mass")
    undescribed = _compiled_change("dof_M0")
    receipt = _release_receipt(
        [described, undescribed],
        certification=_certification(_field_report([_descriptive_row("body_mass")])),
    )
    findings = model_release_findings(receipt)
    assert len(findings) == 2
    assert findings[1]["data"] is undescribed
    assert findings[1]["kind"] == "compiled_field"
    assert "descriptive" not in findings[1]
    assert "descriptive_source" not in findings[1]
    assert (
        findings[1]["display"]
        == "compiled_field dof_M0 value: changed, no value retained in this row"
    )


def test_a_compiled_field_row_whose_field_is_not_value_does_not_join_a_descriptive_row() -> None:
    """Prove the join key requires both the compiled_field object type and the value field."""
    receipt = _release_receipt(
        [
            _compiled_change("body_mass", field="shape"),
            _change_row("COMPILED_PUBLIC_FIELD", "semantic_field", "body_mass", "value"),
        ],
        certification=_certification(_field_report([_descriptive_row("body_mass")])),
    )
    findings = model_release_findings(receipt)
    assert len(findings) == 2
    for finding in findings:
        assert finding["kind"] == "compiled_field"
        assert "descriptive" not in finding
        assert "descriptive_source" not in finding


def test_json_pointer_escapes_the_tilde_before_the_slash_so_every_token_round_trips() -> None:
    """Prove pointer escaping is RFC 6901 exact and survives resolution back to the value."""
    assert json_pointer("changes", 3) == "/changes/3"
    assert json_pointer("a/b~c") == "/a~1b~0c"
    assert json_pointer("~1") == "/~01"
    assert json_pointer("field_report", "changed_fields", 0) == "/field_report/changed_fields/0"
    document: dict[str, CanonicalValue] = {"a/b~c": "slash-then-tilde", "~1": "literal-tilde-one"}
    assert _resolve_pointer(document, json_pointer("a/b~c")) == "slash-then-tilde"
    assert _resolve_pointer(document, json_pointer("~1")) == "literal-tilde-one"


def test_the_real_receipt_keeps_claim_producer_attribution_and_policy_separate() -> None:
    """Prove a condition of the evaluation is never categorized as a physics finding."""
    receipt = _real_receipt(_CHANGED_RELEASE)
    entries = model_release_limitations(receipt)
    by_category: dict[str, list[dict[str, CanonicalValue]]] = {}
    for entry in entries:
        category = entry["category"]
        assert isinstance(category, str)
        by_category.setdefault(category, []).append(entry)
    assert sorted(by_category) == sorted({CLAIM, PRODUCER, ATTRIBUTION, POLICY})
    assert len(by_category[CLAIM]) == 10
    assert [entry["source"] for entry in by_category[CLAIM]] == [
        *(f"/limitations/{index}" for index in range(5)),
        *(f"/certification_receipt/limitations/{index}" for index in range(5)),
    ]
    codes = [(entry.get("detail") or {})["code"] for entry in by_category[CLAIM]]  # type: ignore[index]
    assert len(set(codes)) == 10, "the release and certification boundaries do not repeat"
    assert "EXACT_RECORDED_RUNTIME_ONLY" in codes
    assert "NO_CROSS_MUJOCO_VERSION_CLAIM" in codes
    producer = by_category[PRODUCER]
    assert len(producer) == 1
    assert producer[0]["source"] == "/certification_receipt/field_report/omitted_fields"
    detail = producer[0]["detail"]
    assert isinstance(detail, dict)
    assert detail["fields_omitted_count"] == 46
    assert detail["fields_compared_count"] == 598
    assert len(list(detail["omitted_fields"])) == 46
    assert "46" in str(producer[0]["message"])
    omitted = _resolve_pointer(receipt, str(producer[0]["source"]))
    assert isinstance(omitted, list)
    assert len(omitted) == 46
    attribution = by_category[ATTRIBUTION]
    assert len(attribution) == 1
    assert attribution[0]["detail"] == {
        "reason": "semantic_name_coverage_incomplete",
        "baseline_coverage_issues": ["actuator:0:unnamed"],
        "candidate_coverage_issues": ["actuator:0:unnamed"],
    }
    policy = by_category[POLICY]
    assert len(policy) == 1
    assert policy[0]["source"] == "/changes/11/details"
    policy_detail = policy[0]["detail"]
    assert isinstance(policy_detail, dict)
    assert "baseline_coverage_issues" not in policy_detail
    assert "candidate_coverage_issues" not in policy_detail
    detail = policy[0]["detail"]
    assert isinstance(detail, dict)
    assert detail["reason"] == "candidate_compiled_subject_unbound"
    unbound = [entry for entry in entries if "candidate_compiled_subject_unbound" in str(entry)]
    assert len(unbound) == 1
    assert unbound[0] is policy[0]
    assert unbound[0]["category"] != ATTRIBUTION


def test_every_residual_reason_code_receives_a_category_and_a_statement() -> None:
    """Prove the five known reason codes split into policy and attribution and none is dropped."""
    reasons: list[CanonicalValue] = [*_KNOWN_RESIDUAL_REASONS, _UNKNOWN_RESIDUAL_REASON]
    receipt = _release_receipt([_residual_change(reasons)])
    entries = model_release_limitations(receipt)
    assert [entry["detail"] for entry in entries] == [{"reason": reason} for reason in reasons]
    assert [entry["category"] for entry in entries] == [
        POLICY,
        POLICY,
        ATTRIBUTION,
        ATTRIBUTION,
        ATTRIBUTION,
        ATTRIBUTION,
    ]
    assert all(entry["source"] == "/changes/0/details" for entry in entries)
    statements = [str(entry["message"]) for entry in entries]
    assert len(set(statements)) == len(statements)
    for statement in statements[:5]:
        assert statement
        assert not statement.startswith(_FALLBACK_STATEMENT_PREFIX)
    assert statements[5] == f"{_FALLBACK_STATEMENT_PREFIX} {_UNKNOWN_RESIDUAL_REASON}."


def test_a_truncated_field_report_adds_a_producer_limitation_carrying_the_producer_counts() -> None:
    """Prove producer truncation is reported with the producer's own returned and total counts."""
    receipt = _release_receipt(
        [_compiled_change("body_mass")],
        certification=_certification(
            _field_report(
                [_descriptive_row("body_mass")],
                truncated=True,
                changed_fields_total=41,
                changed_fields_returned=1,
            )
        ),
    )
    entries = model_release_limitations(receipt)
    producer = [entry for entry in entries if entry["category"] == PRODUCER]
    assert len(producer) == 1
    assert producer[0]["source"] == "/certification_receipt/field_report/changed_fields"
    assert producer[0]["detail"] == {"changed_fields_total": 41, "changed_fields_returned": 1}
    assert "truncated" in str(producer[0]["message"])


def _required_rule(identity: str, field: str) -> dict[str, CanonicalValue]:
    """Build one REQUIRE rule in the shape a receipt actually retains."""
    return {
        "id": identity,
        "effect": "REQUIRE",
        "selector": {
            "object_type": "body",
            "object_name": "link",
            "field": field,
            "change_kind": "MODIFY",
        },
        "before_sha256": None,
        "after_sha256": None,
    }


def test_missing_required_rules_produce_one_policy_limitation_naming_every_rule() -> None:
    """Prove declared rules that were never observed are reported once as a policy condition.

    A receipt retains whole rule objects here, not bare identifiers, so the reported name has to
    come from each object's own id.
    """
    rules = [_required_rule("mass.increase", "mass"), _required_rule("inertia.increase", "inertia")]
    receipt = _release_receipt([], missing_required_rules=rules)
    entries = model_release_limitations(receipt)
    assert len(entries) == 1
    assert entries[0]["category"] == POLICY
    assert entries[0]["source"] == "/missing_required_rules"
    assert entries[0]["detail"] == {"missing_required_rules": rules}
    message = str(entries[0]["message"])
    assert "2 change(s)" in message
    assert "mass.increase, inertia.increase" in message


def test_an_identical_pair_invents_nothing_while_a_differing_pair_declares_the_gap() -> None:
    """Prove a null field report is a gap only when the compiled bytes actually differ."""
    equal_receipt = _real_receipt(_EQUAL_RELEASE)
    certification = equal_receipt["certification_receipt"]
    assert isinstance(certification, dict)
    assert certification["field_report"] is None
    assert model_release_findings(equal_receipt) == []
    equal_entries = model_release_limitations(equal_receipt)
    assert {str(entry["category"]) for entry in equal_entries} == {CLAIM}
    differing = _release_receipt([], certification=_certification(None, byte_equal=False))
    differing_entries = model_release_limitations(differing)
    assert len(differing_entries) == 1
    assert differing_entries[0]["category"] == PRODUCER
    assert differing_entries[0]["source"] == "/certification_receipt/field_report"
    assert "no per-field coverage is recorded" in str(differing_entries[0]["message"])
    assert "detail" not in differing_entries[0]


def test_certification_findings_on_a_real_certification_resolve_under_its_report() -> None:
    """Prove a standalone certification yields one finding per retained descriptive row."""
    receipt = _real_receipt(_CERTIFICATION)
    report = receipt["field_report"]
    assert isinstance(report, dict)
    rows = report["changed_fields"]
    assert isinstance(rows, list)
    findings = certification_findings(receipt)
    assert len(findings) == 9
    assert len(rows) == 9
    for index, finding in enumerate(findings):
        pointer = finding["source"]
        assert isinstance(pointer, str)
        assert pointer == json_pointer("field_report", "changed_fields", index)
        assert pointer.startswith("/field_report/changed_fields/")
        assert _resolve_pointer(receipt, pointer) is rows[index]
        assert finding["data"] is rows[index]
        assert finding["kind"] == "compiled_field"
        assert "descriptive" not in finding
        row = rows[index]
        assert isinstance(row, dict)
        assert str(finding["display"]).startswith(f"compiled field {row['path']}: ")
    categories = {str(entry["category"]) for entry in certification_limitations(receipt)}
    assert categories == {CLAIM, PRODUCER}


def test_a_residual_raised_only_by_a_policy_condition_is_not_called_unexplained() -> None:
    """Refuse to describe a binding condition as an unexplained physical difference."""
    receipt = _residual_receipt(["candidate_compiled_subject_unbound"])
    display = str(model_release_findings(receipt)[0]["display"])
    assert "fail-closed residual recorded" in display
    assert "unexplained" not in display
    assert "no identified explanation" not in display


def test_a_residual_that_records_no_identified_explanation_says_exactly_that() -> None:
    """Name the one reason that really does mean the compiled difference is unexplained."""
    receipt = _residual_receipt(["no_public_or_semantic_field_difference_identified"])
    display = str(model_release_findings(receipt)[0]["display"])
    assert "compiled difference with no identified explanation" in display


def test_a_mixed_residual_does_not_inherit_the_unexplained_wording_from_a_policy_reason() -> None:
    """Keep the stronger wording for the reason that earns it, and only for that reason."""
    mixed = _residual_receipt(
        ["candidate_compiled_subject_unbound", "semantic_name_coverage_incomplete"]
    )
    display = str(model_release_findings(mixed)[0]["display"])
    assert "fail-closed residual recorded" in display
    assert "candidate_compiled_subject_unbound" in display
    assert "semantic_name_coverage_incomplete" in display


def test_the_embedded_certification_boundaries_are_carried_with_their_own_pointers() -> None:
    """Present the certification's own limits, which do not repeat the release ones."""
    receipt = _real_receipt(_CHANGED_RELEASE)
    entries = model_release_limitations(receipt)
    carried = [
        entry
        for entry in entries
        if str(entry["source"]).startswith("/certification_receipt/limitations/")
    ]
    assert len(carried) == 5
    for entry in carried:
        resolved = _resolve_pointer(receipt, str(entry["source"]))
        assert isinstance(resolved, dict)
        detail = entry["detail"]
        assert isinstance(detail, dict)
        assert detail["code"] == resolved["code"]
        assert entry["message"] == resolved["statement"]


def _residual_receipt(reasons: list[str]) -> dict[str, CanonicalValue]:
    """Build one model-release receipt carrying a single opaque residual row."""
    return {
        "schema": "metrifid.model_release_receipt",
        "limitations": [],
        "changes": [
            {
                "source": "OPAQUE_ARTIFACT_RESIDUAL",
                "classification": "UNDECLARED",
                "rule_id": None,
                "before_sha256": "a" * 64,
                "after_sha256": "b" * 64,
                "before_value": None,
                "after_value": None,
                "details": {
                    "reasons": list(reasons),
                    "policy_candidate_compiled_sha256": None,
                    "baseline_coverage_issues": [],
                    "candidate_coverage_issues": [],
                },
                "selector": {
                    "object_type": "opaque",
                    "object_name": "complete_mjb",
                    "field": "compiled_artifact",
                    "change_kind": "MODIFY",
                },
            }
        ],
    }


def _constructed_missing_requirement() -> dict[str, CanonicalValue]:
    """Build one canonically validated receipt that records a failed policy over equal bytes.

    This is a CONSTRUCTED reader fixture, not a recorded native comparison. It exists because no
    retained real receipt records an unmet requirement, and the reader must still show one.
    """
    from metrifid.json_values import canonical_json_bytes, canonical_sha256
    from metrifid.model_release import validate_model_release_receipt
    from metrifid.model_release._decision import ModelReleaseDecision
    from metrifid.model_release._policy import parse_model_release_policy
    from metrifid.model_release._receipt import build_model_release_receipt
    from metrifid.model_release._status import ModelReleaseStatus

    equal = _real_receipt(_EQUAL_RELEASE)
    certification = equal["certification_receipt"]
    assert isinstance(certification, dict)
    policy = parse_model_release_policy(
        canonical_json_bytes(
            {
                "schema": "metrifid.model_release_policy",
                "schema_version": 1,
                "baseline_compiled_sha256": _compiled_digest(certification, "baseline"),
                "candidate_compiled_sha256": _compiled_digest(certification, "candidate"),
                "rules": [
                    {
                        "id": "required-link-mass-increase",
                        "effect": "REQUIRE",
                        "selector": {
                            "object_type": "body",
                            "object_name": "link",
                            "field": "mass",
                            "change_kind": "MODIFY",
                        },
                        "before_sha256": canonical_sha256(_ONE_POINT_FIVE),
                        "after_sha256": canonical_sha256(_TWO),
                    }
                ],
            }
        )
    )
    registry = equal["public_field_registry"]
    assert isinstance(registry, dict)
    receipt = build_model_release_receipt(
        policy=policy,
        decision=ModelReleaseDecision(
            ModelReleaseStatus.OUTSIDE_DECLARED_POLICY, (), (), policy.rules
        ),
        certification_receipt=certification,
        registry_sha256=str(registry["sha256"]),
        registry_count=int(str(registry["field_count"])),
    )
    validate_model_release_receipt(receipt)
    return receipt


def _compiled_digest(certification: dict[str, CanonicalValue], role: str) -> str:
    """Return one role's retained compiled-artifact digest."""
    block = certification[role]
    assert isinstance(block, dict)
    artifact = block["compiled_artifact"]
    assert isinstance(artifact, dict)
    return str(artifact["mjb_sha256"])


def test_an_unmet_required_rule_is_named_by_its_identity_not_by_an_empty_string() -> None:
    """Missing required rules are whole rule objects, so report the id a maintainer wrote."""
    receipt = _constructed_missing_requirement()
    policy_entries = [
        entry for entry in model_release_limitations(receipt) if entry["category"] == POLICY
    ]
    assert len(policy_entries) == 1
    message = str(policy_entries[0]["message"])
    assert "required-link-mass-increase" in message
    assert "observed: ." not in message
    detail = policy_entries[0]["detail"]
    assert isinstance(detail, dict)
    retained = detail["missing_required_rules"]
    assert retained == receipt["missing_required_rules"]


def test_the_producer_limitation_retains_every_omitted_field_row_in_its_source_shape() -> None:
    """Which members were omitted, and why, is the answer to what was not covered."""
    receipt = _real_receipt(_CHANGED_RELEASE)
    certification = receipt["certification_receipt"]
    assert isinstance(certification, dict)
    report = certification["field_report"]
    assert isinstance(report, dict)
    producer = [
        entry for entry in model_release_limitations(receipt) if entry["category"] == PRODUCER
    ]
    assert len(producer) == 1
    detail = producer[0]["detail"]
    assert isinstance(detail, dict)
    assert detail["omitted_fields"] == report["omitted_fields"]
    assert len(list(report["omitted_fields"])) == 46


def test_the_omission_limitation_separates_expanded_members_from_uncompared_ones() -> None:
    """An omitted member is not a member nobody looked at, and the wording must not say so.

    In this receipt the producer omits `stat` with reason EXPANDED_ONE_LEVEL_BELOW, and the very
    same report carries explicit changes for `stat.meanmass` and `stat.meaninertia`. Calling every
    omitted member "never compared" would contradict the rows directly beneath it. The limitation
    bounds the field-level explanation; the compiled-byte comparison is recorded on its own and is
    complete for this pair.
    """
    receipt = json.loads(_CERTIFICATION.read_text(encoding="utf-8"))
    report = receipt["field_report"]

    reasons = {row["path"]: row["reason"] for row in report["omitted_fields"]}
    assert reasons["stat"] == "EXPANDED_ONE_LEVEL_BELOW"
    changed = {row["path"] for row in report["changed_fields"]}
    assert {"stat.meanmass", "stat.meaninertia"} <= changed, changed

    producer = [
        entry for entry in certification_limitations(receipt) if entry["category"] == PRODUCER
    ]
    assert len(producer) == 1
    message = str(producer[0]["message"])
    assert "never compared" not in message, message
    assert "field-level" in message
    assert "recorded reason" in message

    # Every retained row and count is passed through exactly as the producer wrote it, so a
    # reader can see which reason applies to which member rather than taking the prose on trust.
    detail = producer[0]["detail"]
    assert isinstance(detail, dict)
    assert detail["omitted_fields"] == report["omitted_fields"]
    assert detail["fields_omitted_count"] == report["fields_omitted_count"]
    assert detail["fields_compared_count"] == report["fields_compared_count"]

    # The byte comparison is a separate record and stays complete for this pair.
    byte_comparison = receipt["byte_comparison"]
    assert byte_comparison["equal"] is False
    assert byte_comparison["compared_byte_count"] == byte_comparison["baseline_mjb_size_bytes"]
    assert byte_comparison["compared_byte_count"] == byte_comparison["candidate_mjb_size_bytes"]
