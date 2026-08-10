"""Validate descriptive compiled-equivalence receipt fields."""

from __future__ import annotations

from collections.abc import Mapping

from ..json_values import Binary64, CanonicalValue, require_sha256
from ..operational import _require_exact_object_fields
from ._bytes import ByteComparison
from ._field_schema import (
    _CHANGED_FIELD_MEMBERS,
    _FIELD_REPORT_MEMBERS,
    _OMISSION_REASONS,
    _OMITTED_FIELD_MEMBERS,
    _WITNESS_MEMBERS,
    FIELD_DIFFERENCES_IDENTIFIED,
    FIELD_REPORT_SCHEMA,
    FIELD_REPORT_STATUSES,
    MAX_WITNESSES_PER_FIELD,
    NO_PUBLIC_FIELD_DIFFERENCE_IDENTIFIED,
)
from ._receipt_contract import (
    _LIMITATION_STATEMENTS,
    ARTIFACT_CLAIM_EXCLUSIONS,
    BEHAVIOR_IMPLICATION,
    BEHAVIOR_IMPLICATION_PREMISES,
    CERTIFIED_ARTIFACT_CLAIM,
    NOT_CERTIFIED_ARTIFACT_CLAIM,
    NOT_CERTIFIED_GUIDANCE,
    REQUIRED_LIMITATIONS,
)
from ._status import CertifyStatus


def _validate_claims(obj: Mapping[str, object], certified: bool) -> None:
    """Require the frozen claim, implication and limitation text, verbatim."""
    _validate_artifact_claim(obj["artifact_claim"], certified)
    _validate_behavior_implication(obj["behavior_implication"], certified)
    _validate_limitations(obj["limitations"])


def _validate_artifact_claim(value: object, certified: bool) -> None:
    """Validate the frozen workload-free artifact claim and its exclusions."""
    claim = _require_exact_object_fields(
        value,
        {"claim_kind", "workload_free", "statement", "does_not_claim"}
        | ({"guidance"} if not certified else set()),
        "artifact_claim",
    )
    if (
        claim["claim_kind"] != "UNCONDITIONAL_ARTIFACT_STATEMENT"
        or claim["workload_free"] is not True
    ):
        raise ValueError("the artifact claim must be the frozen unconditional workload-free claim")
    if claim["statement"] != (
        CERTIFIED_ARTIFACT_CLAIM if certified else NOT_CERTIFIED_ARTIFACT_CLAIM
    ):
        raise ValueError("the artifact claim statement is not the frozen text")
    if claim["does_not_claim"] != list(ARTIFACT_CLAIM_EXCLUSIONS):
        raise ValueError("the artifact claim exclusion list is not the frozen list")
    if not certified and claim["guidance"] != NOT_CERTIFIED_GUIDANCE:
        raise ValueError("a differing result must carry the frozen negative guidance")


def _validate_behavior_implication(value: object, certified: bool) -> None:
    """Validate the optional non-decision-bearing behavior implication."""
    if not certified:
        if value is not None:
            raise ValueError("a differing result carries no behavior implication")
        return
    entry = _require_exact_object_fields(
        value,
        {"claim_kind", "statement", "premises", "included_in_decision_sha256"},
        "behavior_implication",
    )
    if entry["claim_kind"] != "CONDITIONAL_NON_DECISION_BEARING":
        raise ValueError("the behavior implication must be labelled conditional")
    if entry["statement"] != BEHAVIOR_IMPLICATION:
        raise ValueError("the behavior implication statement is not the frozen text")
    if entry["premises"] != list(BEHAVIOR_IMPLICATION_PREMISES):
        raise ValueError("the behavior implication premises are not the frozen ordered list")
    if entry["included_in_decision_sha256"] is not False:
        raise ValueError("the behavior implication may never enter the decision hash")


def _validate_limitations(value: object) -> None:
    """Validate the complete ordered limitation registry and frozen statements."""
    limitations = value
    if type(limitations) is not list or len(limitations) != len(REQUIRED_LIMITATIONS):
        raise ValueError("every required limitation must be present in the frozen order")
    for entry_value, code in zip(limitations, REQUIRED_LIMITATIONS, strict=True):
        entry = _require_exact_object_fields(entry_value, {"code", "statement"}, "limitation")
        if entry["code"] != code:
            raise ValueError("every required limitation must be present in the frozen order")
        if entry["statement"] != _LIMITATION_STATEMENTS[code]:
            raise ValueError(f"the {code} limitation statement is not the frozen text")


def _validated_report_counts(
    report: Mapping[str, object],
) -> tuple[dict[str, int], list[object], list[object]]:
    """Validate the report's counts and their ordering, and return the two returned lists."""
    counts: dict[str, int] = {}
    for name in (
        "fields_compared_count",
        "fields_omitted_count",
        "changed_fields_total",
        "changed_fields_returned",
    ):
        raw = report[name]
        if type(raw) is not int:
            raise TypeError(f"{name} must be a JSON integer, not a boolean")
        if raw < 0:
            raise ValueError(f"{name} must be nonnegative")
        counts[name] = raw
    changed = report["changed_fields"]
    omitted = report["omitted_fields"]
    if type(changed) is not list or type(omitted) is not list:
        raise TypeError("changed_fields and omitted_fields must be arrays")
    if counts["changed_fields_returned"] != len(changed):
        raise ValueError("changed_fields_returned must equal the returned changed-field count")
    if counts["changed_fields_total"] < counts["changed_fields_returned"]:
        raise ValueError("changed_fields_total may not be smaller than the returned count")
    if counts["fields_compared_count"] < counts["changed_fields_total"]:
        raise ValueError("fields_compared_count may not be smaller than changed_fields_total")
    if counts["fields_omitted_count"] < len(omitted):
        raise ValueError("fields_omitted_count may not be smaller than the returned omitted count")
    expected_status = (
        FIELD_DIFFERENCES_IDENTIFIED
        if counts["changed_fields_total"] > 0
        else NO_PUBLIC_FIELD_DIFFERENCE_IDENTIFIED
    )
    if report["field_report_status"] != expected_status:
        raise ValueError("the field report status does not follow from changed_fields_total")
    return counts, changed, omitted


def _entry_withheld_evidence(entry: object) -> bool:
    """True when a changed field published less element evidence than it observed.

    Two shapes count: a witness list shorter than the reported changed-element count, and an
    array field whose copies were declined by the witness copy budget - which publishes no count
    and no witnesses while still naming equal known shapes on both sides.
    """
    if type(entry) is not dict:
        return False
    count = entry["changed_element_count"]
    witnesses = entry["witnesses"]
    if type(count) is int:
        return count > len(witnesses)
    shapes = (entry["baseline_shape"], entry["candidate_shape"])
    return not witnesses and shapes[0] is not None and shapes[0] == shapes[1]


def _validate_field_report(value: object, certified: bool, comparison: ByteComparison) -> None:
    """A certificate carries no field report; a difference carries one describing this comparison.

    Everything below validates the published report's internal structure and its redundancies
    against facts already validated elsewhere in the receipt. It does not recompute any public
    field value from MJB bytes and does not claim to.
    """
    if certified:
        if value is not None:
            raise ValueError("a certificate carries no field report")
        return
    report = _require_exact_object_fields(value, set(_FIELD_REPORT_MEMBERS), "field_report")
    _validate_field_report_header(report, comparison)
    counts, changed, omitted = _validated_report_counts(report)
    changed_paths = [_validated_changed_field(entry) for entry in changed]
    omitted_paths = [_validated_omitted_field(entry) for entry in omitted]
    _require_sorted_unique(changed_paths, "changed field")
    _require_sorted_unique(omitted_paths, "omitted field")
    if set(changed_paths) & set(omitted_paths):
        raise ValueError("a path may not be both changed and omitted")
    _validate_report_truncation(report, counts, changed, omitted)


def _validate_field_report_header(report: Mapping[str, object], comparison: ByteComparison) -> None:
    """Validate field-report schema identity, status, and byte-comparison bindings."""
    if report["schema"] != FIELD_REPORT_SCHEMA:
        raise ValueError("the field report schema is outside the frozen registry")
    if type(report["schema_version"]) is not int or report["schema_version"] != 1:
        raise ValueError("the field report schema_version is outside the frozen registry")
    if report["field_report_status"] not in FIELD_REPORT_STATUSES:
        raise ValueError("the field report status is outside the frozen registry")
    for name, expected in (
        ("baseline_mjb_size_bytes", comparison.baseline_mjb_size_bytes),
        ("candidate_mjb_size_bytes", comparison.candidate_mjb_size_bytes),
        ("first_differing_byte_offset", comparison.first_differing_byte_offset),
        ("differing_byte_count", comparison.differing_byte_count),
    ):
        if report[name] != expected:
            raise ValueError(f"the field report {name} disagrees with the byte comparison")


def _validate_report_truncation(
    report: Mapping[str, object],
    counts: Mapping[str, int],
    changed: list[object],
    omitted: list[object],
) -> None:
    """Require the truncation flag to match every evidence-withholding condition."""
    truncated = report["truncated"]
    if type(truncated) is not bool:
        raise TypeError("truncated must be a boolean")
    expected_truncated = (
        counts["changed_fields_total"] > counts["changed_fields_returned"]
        or counts["fields_omitted_count"] > len(omitted)
        or any(_entry_withheld_evidence(entry) for entry in changed)
    )
    if truncated is not expected_truncated:
        raise ValueError("truncated must equal the disjunction of the configured bounds")


def _require_sorted_unique(paths: list[str], label: str) -> None:
    """Require evidence paths to be unique and exact-code-point ordered."""
    if paths != sorted(paths):
        raise ValueError(f"{label} paths must be sorted")
    if len(set(paths)) != len(paths):
        raise ValueError(f"{label} paths must be unique")


def _validated_changed_field(value: object) -> str:
    """Validate one changed-field entry and return its path."""
    entry = _require_exact_object_fields(value, set(_CHANGED_FIELD_MEMBERS), "changed field")
    path = entry["path"]
    if type(path) is not str or not path:
        raise ValueError("a changed field must carry a nonempty path")
    baseline = _validated_changed_side(entry, "baseline")
    candidate = _validated_changed_side(entry, "candidate")
    if baseline[3] is None and candidate[3] is None:
        raise ValueError("a changed field must exist on at least one side")
    if baseline == candidate:
        raise ValueError("a changed field may not report identical baseline and candidate facts")
    _validate_changed_witnesses(entry, baseline[2], candidate[2])
    return path


def _validated_changed_side(
    entry: Mapping[str, object], side: str
) -> tuple[object, object, tuple[int, ...] | None, object]:
    """Validate one role-local side of a changed-field report entry."""
    digest = entry[f"{side}_sha256"]
    if digest is not None:
        require_sha256(digest, f"{side}_sha256")
    shape = _validated_shape(entry[f"{side}_shape"], side)
    type_name = entry[f"{side}_type"]
    dtype = entry[f"{side}_dtype"]
    if digest is None:
        _validate_absent_changed_side(type_name, dtype, shape, side)
    else:
        _validate_present_changed_side(type_name, dtype, side)
    return type_name, dtype, shape, digest


def _validate_absent_changed_side(
    type_name: object, dtype: object, shape: tuple[int, ...] | None, side: str
) -> None:
    """Require an absent changed-field side to carry no descriptive metadata."""
    if type_name is not None or dtype is not None or shape is not None:
        raise ValueError(f"an absent {side} side carries no type, dtype or shape")


def _validate_present_changed_side(type_name: object, dtype: object, side: str) -> None:
    """Require a present changed-field side to carry valid type and dtype names."""
    if type(type_name) is not str or not type_name:
        raise ValueError(f"a present {side} side must name its type as a nonempty string")
    if dtype is not None and (type(dtype) is not str or not dtype):
        raise ValueError(f"{side}_dtype must be null or a nonempty string")


def _validate_changed_witnesses(
    entry: Mapping[str, object],
    baseline_shape: tuple[int, ...] | None,
    candidate_shape: tuple[int, ...] | None,
) -> None:
    """Validate changed-element counts and bounded, ordered witness evidence."""
    count = _validated_changed_count(entry["changed_element_count"])
    witnesses = _validated_witness_list(entry["witnesses"], count)
    known_shape = (
        baseline_shape if baseline_shape is not None and baseline_shape == candidate_shape else None
    )
    indices = [_validated_witness(item, known_shape) for item in witnesses]
    if indices != sorted(indices):
        raise ValueError("witness indices must be sorted within a changed field")
    if len(set(indices)) != len(indices):
        raise ValueError("witness indices must be unique within a changed field")


def _validated_changed_count(value: object) -> int | None:
    """Validate and return a changed-element count or its deliberate omission."""
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError("changed_element_count must be a JSON integer or null")
    if value <= 0:
        raise ValueError("changed_element_count must be positive when present")
    return value


def _validated_witness_list(value: object, count: int | None) -> list[object]:
    """Validate the witness container and configured per-field evidence bound."""
    witnesses = value
    if type(witnesses) is not list:
        raise TypeError("witnesses must be an array")
    if len(witnesses) > MAX_WITNESSES_PER_FIELD:
        raise ValueError("a changed field may not return more than the configured witness bound")
    if count is not None and count < len(witnesses):
        raise ValueError("changed_element_count may not be smaller than the witness count")
    return witnesses


def _validated_shape(value: object, side: str) -> tuple[int, ...] | None:
    """Validate and return a role-local nonnegative integer array shape."""
    if value is None:
        return None
    if type(value) is not list:
        raise TypeError(f"{side}_shape must be null or an array")
    for extent in value:
        if type(extent) is not int or extent < 0:
            raise ValueError(f"{side}_shape must hold nonnegative JSON integers")
    return tuple(int(extent) for extent in value)


def _validated_witness(value: object, known_shape: tuple[int, ...] | None) -> tuple[int, ...]:
    """Validate one witness and return its index for the sortedness check."""
    witness = _require_exact_object_fields(value, set(_WITNESS_MEMBERS), "witness")
    index = _validated_witness_index(witness["index"])
    _validate_witness_values(witness)
    _validate_witness_bounds(index, known_shape)
    return index


def _validated_witness_index(value: object) -> tuple[int, ...]:
    """Validate and normalize one witness index array."""
    index = value
    if type(index) is not list:
        raise TypeError("a witness index must be an array")
    for axis in index:
        if type(axis) is not int or axis < 0:
            raise ValueError("a witness index must hold nonnegative JSON integers")
    return tuple(int(axis) for axis in index)


def _validate_witness_values(witness: Mapping[str, object]) -> None:
    """Validate textual and canonical value evidence carried by one witness."""
    for name in ("baseline_text", "candidate_text"):
        if type(witness[name]) is not str:
            raise TypeError(f"{name} must be a string")
    for name in ("baseline_value", "candidate_value"):
        _require_producer_value(witness[name], name)


def _validate_witness_bounds(index: tuple[int, ...], known_shape: tuple[int, ...] | None) -> None:
    """Require an index to fit the known equal array shape when one is available."""
    if known_shape is not None:
        if len(index) != len(known_shape):
            raise ValueError("a witness index rank must match the known equal shape rank")
        if any(axis >= extent for axis, extent in zip(index, known_shape, strict=True)):
            raise ValueError("a witness index must lie inside the known equal shape")


def _require_producer_value(value: object, field: str) -> None:
    """Require one witness value to be a canonical form this product's reporter can emit.

    The reporter emits null, a boolean, an integer, a decoded string, or an exact IEEE-754
    binary64 object. A raw JSON float, an array, or any other object is not something it
    produces, so a receipt carrying one is not internally consistent.
    """
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is not dict:
        raise TypeError(f"{field} must be null, a boolean, an integer, a string or a Binary64")
    if set(value) != {"kind", "bits"} or value.get("kind") != "ieee754_binary64":
        raise ValueError(f"{field} is not an exact Binary64 object")
    Binary64.from_primitive(value)


def _validated_omitted_field(value: object) -> str:
    """Validate one omitted-field entry and return its path."""
    entry = _require_exact_object_fields(value, set(_OMITTED_FIELD_MEMBERS), "omitted field")
    path = entry["path"]
    if type(path) is not str or not path:
        raise ValueError("an omitted field must carry a nonempty path")
    if entry["reason"] not in _OMISSION_REASONS:
        raise ValueError("an omitted field reason is outside the producer's frozen registry")
    return path


def _artifact_claim(status: CertifyStatus) -> dict[str, CanonicalValue]:
    """Select the frozen artifact claim text for a completed Certify status."""
    certified = status is CertifyStatus.CERTIFIED_COMPILED_EQUIVALENCE
    claim: dict[str, CanonicalValue] = {
        "claim_kind": "UNCONDITIONAL_ARTIFACT_STATEMENT",
        "workload_free": True,
        "statement": CERTIFIED_ARTIFACT_CLAIM if certified else NOT_CERTIFIED_ARTIFACT_CLAIM,
        "does_not_claim": list(ARTIFACT_CLAIM_EXCLUSIONS),
    }
    if not certified:
        claim["guidance"] = NOT_CERTIFIED_GUIDANCE
    return claim


def _behavior_implication(status: CertifyStatus) -> dict[str, CanonicalValue] | None:
    """Select the bounded behavior implication permitted by a Certify status."""
    if status is not CertifyStatus.CERTIFIED_COMPILED_EQUIVALENCE:
        return None
    return {
        "claim_kind": "CONDITIONAL_NON_DECISION_BEARING",
        "statement": BEHAVIOR_IMPLICATION,
        "premises": list(BEHAVIOR_IMPLICATION_PREMISES),
        "included_in_decision_sha256": False,
    }
