"""The frozen Certify status, exit mapping, claim separation and receipt schema."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest

from metrifid.certify import validate_receipt
from metrifid.json_values import canonical_sha256

_BASELINE_XML = """
<mujoco model="receipt-fixture">
  <option timestep="0.002"/>
  <worldbody>
    <body name="b" pos="0 0 1">
      <geom name="g" type="sphere" size="0.1" rgba="1 0 0 1" mass="2"/>
      <joint name="j" type="hinge" axis="0 0 1" damping="0.5"/>
    </body>
  </worldbody>
</mujoco>
"""


_CANDIDATE_XML = _BASELINE_XML.replace('mass="2"', 'mass="3"')


def _certify(root: Path, baseline_xml: str, candidate_xml: str) -> Path:
    """Run real certification for fixture XML and return its published output directory."""
    from metrifid.certify import certify_models

    baseline_dir = root / "baseline"
    candidate_dir = root / "candidate"
    baseline_dir.mkdir(parents=True)
    candidate_dir.mkdir(parents=True)
    (baseline_dir / "model.xml").write_text(baseline_xml, encoding="utf-8")
    (candidate_dir / "model.xml").write_text(candidate_xml, encoding="utf-8")
    result = certify_models(
        str(baseline_dir / "model.xml"),
        str(candidate_dir / "model.xml"),
        str(root / "out"),
    )
    return result.certification_json.parent


@pytest.fixture(scope="module")
def certified_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Publish a receipt for equivalent fixture models through the real product path."""
    return _certify(tmp_path_factory.mktemp("certified"), _BASELINE_XML, _BASELINE_XML)


@pytest.fixture(scope="module")
def differing_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Publish a receipt for unequal fixture models through the real product path."""
    return _certify(tmp_path_factory.mktemp("differing"), _BASELINE_XML, _CANDIDATE_XML)


def _load_receipt_json(path: Path) -> dict[str, Any]:
    """Decode a published certification receipt for evidence-tampering scenarios."""
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


_DECISION_NAMES = (
    "schema",
    "schema_version",
    "status",
    "completed_exit_code",
    "tool",
    "runtime_identity",
    "baseline",
    "candidate",
    "byte_comparison",
)


_ROOT_NAMES = (
    *_DECISION_NAMES,
    "field_report",
    "artifact_claim",
    "behavior_implication",
    "limitations",
    "decision_sha256",
    "receipt_sha256",
)


def _resealed(receipt: dict[str, Any]) -> dict[str, Any]:
    """Recompute both hashes exactly as a naive forger would after editing a nested fact."""
    from metrifid.json_values import compute_self_hash

    resealed = copy.deepcopy(receipt)
    resealed["decision_sha256"] = canonical_sha256(
        {name: resealed[name] for name in _DECISION_NAMES}
    )
    resealed["receipt_sha256"] = None
    resealed["receipt_sha256"] = compute_self_hash(resealed, "receipt_sha256")
    return resealed


def _tampered(receipt: dict[str, Any], mutate: Any) -> dict[str, Any]:
    """Copy, mutate, and reseal a receipt so validation reaches its semantic contradiction."""
    copied = copy.deepcopy(receipt)
    mutate(copied)
    return _resealed(copied)


_DIFFERING_TAMPERS: tuple[tuple[str, Any], ...] = (
    (
        "implication injected",
        lambda r: r.__setitem__(
            "behavior_implication",
            {
                "claim_kind": "CONDITIONAL_NON_DECISION_BEARING",
                "statement": "x",
                "premises": [],
                "included_in_decision_sha256": False,
            },
        ),
    ),
    ("field report removed", lambda r: r.__setitem__("field_report", None)),
    (
        "negative guidance rewritten",
        lambda r: r["artifact_claim"].__setitem__("guidance", "It is fine."),
    ),
    ("field report byte facts", lambda r: r["field_report"].__setitem__("differing_byte_count", 1)),
    (
        "field report status",
        lambda r: r["field_report"].__setitem__("field_report_status", "ALL_GOOD"),
    ),
    (
        "comparison offset dropped",
        lambda r: r["byte_comparison"].__setitem__("first_differing_byte_offset", None),
    ),
    (
        "status flipped to certified",
        lambda r: (
            r.__setitem__("status", "CERTIFIED_COMPILED_EQUIVALENCE"),
            r.__setitem__("completed_exit_code", 0),
        ),
    ),
)


def _require_field_report(receipt: dict[str, Any]) -> dict[str, Any]:
    """Return the mapping required on a compiled-difference receipt."""
    report = receipt["field_report"]
    assert isinstance(report, dict)
    return report


def _first_changed(receipt: dict[str, Any]) -> dict[str, Any]:
    """Select the first compiled-field difference from a receipt that must contain one."""
    return cast("dict[str, Any]", _require_field_report(receipt)["changed_fields"][0])


_EXIT_AND_TOOL_TAMPERS: tuple[tuple[str, Any], ...] = (
    ("completed_exit_code = false", lambda r: r.__setitem__("completed_exit_code", False)),
    ("completed_exit_code = true", lambda r: r.__setitem__("completed_exit_code", True)),
    (
        "tool state UNBOUND",
        lambda r: r["tool"].update(execution_identity_state="UNBOUND", distribution_sha256=None),
    ),
    (
        "tool state MISMATCH",
        lambda r: r["tool"].update(execution_identity_state="MISMATCH", distribution_sha256=None),
    ),
    ("tool version differs from runtime", lambda r: r["tool"].__setitem__("version", "9.9.9")),
    (
        "tool digest differs from runtime",
        lambda r: r["tool"].__setitem__("distribution_sha256", "a" * 64),
    ),
)


_FIELD_REPORT_TAMPERS: tuple[tuple[str, Any], ...] = (
    ("field report unknown member", lambda r: _require_field_report(r).__setitem__("extra", 1)),
    ("field report missing member", lambda r: _require_field_report(r).pop("truncated")),
    (
        "negative root count",
        lambda r: _require_field_report(r).__setitem__("fields_omitted_count", -1),
    ),
    (
        "boolean root count",
        lambda r: _require_field_report(r).__setitem__("fields_compared_count", True),
    ),
    (
        "changed_fields_total inconsistent",
        lambda r: _require_field_report(r).__setitem__("changed_fields_total", 0),
    ),
    (
        "changed_fields_returned inconsistent",
        lambda r: _require_field_report(r).__setitem__("changed_fields_returned", 99),
    ),
    (
        "fields_compared_count too small",
        lambda r: _require_field_report(r).__setitem__("fields_compared_count", 1),
    ),
    (
        "fields_omitted_count too small",
        lambda r: _require_field_report(r).__setitem__("fields_omitted_count", 0),
    ),
    (
        "status inconsistent with total",
        lambda r: _require_field_report(r).__setitem__(
            "field_report_status", "NO_PUBLIC_FIELD_DIFFERENCE_IDENTIFIED"
        ),
    ),
    ("changed paths unsorted", lambda r: _require_field_report(r)["changed_fields"].reverse()),
    (
        "changed paths duplicated",
        lambda r: _require_field_report(r)["changed_fields"].__setitem__(
            1, copy.deepcopy(_require_field_report(r)["changed_fields"][0])
        ),
    ),
    ("omitted paths unsorted", lambda r: _require_field_report(r)["omitted_fields"].reverse()),
    (
        "omitted paths duplicated",
        lambda r: _require_field_report(r)["omitted_fields"].__setitem__(
            1, copy.deepcopy(_require_field_report(r)["omitted_fields"][0])
        ),
    ),
    (
        "same path changed and omitted",
        lambda r: _require_field_report(r)["omitted_fields"].__setitem__(
            0, {"path": _first_changed(r)["path"], "reason": "CALLABLE_MEMBER"}
        ),
    ),
    (
        "unknown omitted reason",
        lambda r: _require_field_report(r)["omitted_fields"][0].__setitem__(
            "reason", "BECAUSE_I_SAID_SO"
        ),
    ),
)


_CHANGED_ENTRY_TAMPERS: tuple[tuple[str, Any], ...] = (
    ("changed entry unknown member", lambda r: _first_changed(r).__setitem__("extra", 1)),
    ("changed entry missing member", lambda r: _first_changed(r).pop("witnesses")),
    ("invalid field sha256", lambda r: _first_changed(r).__setitem__("baseline_sha256", "nothex")),
    (
        "identical baseline and candidate facts",
        lambda r: _first_changed(r).update(
            candidate_sha256=_first_changed(r)["baseline_sha256"],
            candidate_type=_first_changed(r)["baseline_type"],
            candidate_dtype=_first_changed(r)["baseline_dtype"],
            candidate_shape=_first_changed(r)["baseline_shape"],
        ),
    ),
    (
        "zero changed_element_count",
        lambda r: _first_changed(r).__setitem__("changed_element_count", 0),
    ),
    (
        "negative changed_element_count",
        lambda r: _first_changed(r).__setitem__("changed_element_count", -3),
    ),
    (
        "more than eight witnesses",
        lambda r: _first_changed(r).__setitem__(
            "witnesses", [copy.deepcopy(_first_changed(r)["witnesses"][0]) for _ in range(9)]
        ),
    ),
    ("witness unknown member", lambda r: _first_changed(r)["witnesses"][0].__setitem__("extra", 1)),
    (
        "negative witness index",
        lambda r: _first_changed(r)["witnesses"][0].__setitem__("index", [-1]),
    ),
    (
        "witness rank inconsistent with shape",
        lambda r: _first_changed(r)["witnesses"][0].__setitem__(
            "index", [*_first_changed(r)["witnesses"][0]["index"], 0]
        ),
    ),
    (
        "witness index outside the shape",
        lambda r: _first_changed(r)["witnesses"][0].__setitem__(
            "index", [10**6] * len(_first_changed(r)["witnesses"][0]["index"])
        ),
    ),
)


_TYPE_TAMPERS: tuple[tuple[str, Any], ...] = (
    (
        "present side with a null type",
        lambda r: _first_changed(r).__setitem__("baseline_type", None),
    ),
    (
        "present side with an empty type",
        lambda r: _first_changed(r).__setitem__("baseline_type", ""),
    ),
    (
        "present side with a non-string type",
        lambda r: _first_changed(r).__setitem__("baseline_type", 7),
    ),
    (
        "present side with an empty dtype",
        lambda r: _first_changed(r).__setitem__("baseline_dtype", ""),
    ),
    (
        "present side with a non-string dtype",
        lambda r: _first_changed(r).__setitem__("baseline_dtype", 3),
    ),
    (
        "witness value is an array",
        lambda r: _first_changed(r)["witnesses"][0].__setitem__("baseline_value", [1, 2]),
    ),
    (
        "witness value is an arbitrary object",
        lambda r: _first_changed(r)["witnesses"][0].__setitem__("baseline_value", {"a": 1}),
    ),
    (
        "witness value has the wrong kind",
        lambda r: _first_changed(r)["witnesses"][0].__setitem__(
            "baseline_value", {"kind": "decimal", "bits": "0" * 16}
        ),
    ),
)


def test_the_truncated_flag_may_not_be_flipped_in_either_direction(
    differing_output: Path,
) -> None:
    """`truncated` is the exact disjunction of the three bounds, so either flip is a contradiction."""
    source = _load_receipt_json(differing_output / "certification.json")
    actual = _require_field_report(source)["truncated"]
    assert isinstance(actual, bool)

    def flip_truncation_flag(receipt: dict[str, Any]) -> None:
        """Invert only the flag so validation must derive truncation from report bounds."""
        _require_field_report(receipt)["truncated"] = not actual

    with pytest.raises(ValueError):
        validate_receipt(_tampered(source, flip_truncation_flag))


def test_truncated_true_with_no_bound_reached_is_rejected(differing_output: Path) -> None:
    """Reject a resealed receipt that claims truncation without reporting a reached bound."""

    def replace_with_empty_truncated_field_report(receipt: dict[str, Any]) -> None:
        """Replace report evidence with empty collections while claiming truncation."""
        report = _require_field_report(receipt)
        report.update(
            truncated=True,
            changed_fields=[],
            changed_fields_returned=0,
            changed_fields_total=0,
            omitted_fields=[],
            fields_omitted_count=0,
            field_report_status="NO_PUBLIC_FIELD_DIFFERENCE_IDENTIFIED",
        )

    receipt = _tampered(
        _load_receipt_json(differing_output / "certification.json"),
        replace_with_empty_truncated_field_report,
    )
    with pytest.raises(ValueError):
        validate_receipt(receipt)


def test_truncated_false_with_a_bound_reached_is_rejected(differing_output: Path) -> None:
    """Force a witness bound, then claim nothing was truncated."""
    source = _load_receipt_json(differing_output / "certification.json")

    def hide_truncation(receipt: dict[str, Any]) -> None:
        """Exceed a witness bound while falsely clearing the derived truncation flag."""
        report = _require_field_report(receipt)
        entry = report["changed_fields"][0]
        entry["changed_element_count"] = len(entry["witnesses"]) + 5
        report["truncated"] = False

    with pytest.raises(ValueError):
        validate_receipt(_tampered(source, hide_truncation))


@pytest.mark.parametrize(
    ("label", "mutate"), _CHANGED_ENTRY_TAMPERS, ids=[c[0] for c in _CHANGED_ENTRY_TAMPERS]
)
def test_a_resealed_changed_entry_tamper_is_rejected(
    differing_output: Path, label: str, mutate: Any
) -> None:
    """Reject each correctly resealed contradiction within one changed-field entry."""
    receipt = _tampered(_load_receipt_json(differing_output / "certification.json"), mutate)
    with pytest.raises((ValueError, TypeError, KeyError)):
        validate_receipt(receipt)


def test_unsorted_or_duplicated_witness_indices_are_rejected(differing_output: Path) -> None:
    """Reject witnesses whose indices are duplicated or no longer canonically ordered."""
    source = _load_receipt_json(differing_output / "certification.json")
    multi = [
        entry
        for entry in _require_field_report(source)["changed_fields"]
        if len(entry["witnesses"]) > 1
    ]
    assert multi, "the differing fixture must contain a field with more than one witness"
    path = multi[0]["path"]

    def reverse_witnesses(receipt: dict[str, Any]) -> None:
        """Reverse one multi-witness collection while preserving its other evidence."""
        entry = next(
            e for e in _require_field_report(receipt)["changed_fields"] if e["path"] == path
        )
        entry["witnesses"] = list(reversed(entry["witnesses"]))

    def duplicate_witness(receipt: dict[str, Any]) -> None:
        """Replace one witness with a duplicate carrying the same flattened index."""
        entry = next(
            e for e in _require_field_report(receipt)["changed_fields"] if e["path"] == path
        )
        entry["witnesses"][1] = copy.deepcopy(entry["witnesses"][0])

    for mutate in (reverse_witnesses, duplicate_witness):
        with pytest.raises(ValueError):
            validate_receipt(_tampered(source, mutate))


def test_changed_element_count_below_the_witness_count_is_rejected(
    differing_output: Path,
) -> None:
    """Reject a changed-element total smaller than its published witness collection."""
    source = _load_receipt_json(differing_output / "certification.json")
    multi = [
        entry
        for entry in _require_field_report(source)["changed_fields"]
        if len(entry["witnesses"]) > 1
    ]
    assert multi
    path = multi[0]["path"]

    def set_changed_element_count_below_witness_count(receipt: dict[str, Any]) -> None:
        """Make one element count smaller than its published witness collection."""
        entry = next(
            e for e in _require_field_report(receipt)["changed_fields"] if e["path"] == path
        )
        entry["changed_element_count"] = 1

    with pytest.raises(ValueError):
        validate_receipt(_tampered(source, set_changed_element_count_below_witness_count))


def test_validator_and_producer_share_the_same_frozen_registries() -> None:
    """The omission reasons and member sets are read from the producer, never restated."""
    from metrifid.certify import _fields, _receipt

    assert _receipt._OMISSION_REASONS is _fields._OMISSION_REASONS
    assert _receipt._FIELD_REPORT_MEMBERS is _fields._FIELD_REPORT_MEMBERS
    assert _receipt._CHANGED_FIELD_MEMBERS is _fields._CHANGED_FIELD_MEMBERS
    assert _receipt._WITNESS_MEMBERS is _fields._WITNESS_MEMBERS


@pytest.mark.parametrize(("label", "mutate"), _TYPE_TAMPERS, ids=[c[0] for c in _TYPE_TAMPERS])
def test_a_resealed_descriptive_type_tamper_is_rejected(
    differing_output: Path, label: str, mutate: Any
) -> None:
    """Reject resealed type metadata that contradicts the producer's field evidence."""
    receipt = _tampered(_load_receipt_json(differing_output / "certification.json"), mutate)
    with pytest.raises((ValueError, TypeError, KeyError)):
        validate_receipt(receipt)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(1.5, id="float"),
        pytest.param({"kind": "ieee754_binary64", "bits": "nothex"}, id="malformed Binary64"),
        pytest.param(
            {"kind": "ieee754_binary64", "bits": "0" * 16, "extra": 1}, id="Binary64 with extras"
        ),
    ],
)
def test_a_witness_value_outside_the_producer_forms_is_rejected(value: Any) -> None:
    """Reject witness forms that the producer's canonical serialization cannot emit."""
    from metrifid.certify._receipt import _require_producer_value
    from metrifid.json_values import canonical_json_bytes

    with pytest.raises((TypeError, ValueError)):
        _require_producer_value(value, "baseline_value")
    with pytest.raises((TypeError, ValueError)):
        canonical_json_bytes({"witness": value})


def test_an_absent_side_still_requires_every_member_to_be_null(differing_output: Path) -> None:
    """Reject an absent-side digest when its corresponding metadata remains populated."""

    def clear_only_baseline_digest(receipt: dict[str, Any]) -> None:
        """Clear only the baseline digest, leaving contradictory side metadata present."""
        entry = _first_changed(receipt)
        entry["baseline_sha256"] = None
        # type, dtype and shape are deliberately left populated

    receipt = _tampered(
        _load_receipt_json(differing_output / "certification.json"), clear_only_baseline_digest
    )
    with pytest.raises(ValueError):
        validate_receipt(receipt)


def test_a_producer_supported_witness_value_is_accepted(differing_output: Path) -> None:
    """The four canonical forms this reporter can emit must all still validate."""
    from metrifid.certify._receipt import _require_producer_value

    for value in (None, True, False, 0, -3, "text", {"kind": "ieee754_binary64", "bits": "0" * 16}):
        _require_producer_value(value, "baseline_value")


def test_the_frozen_receipt_corpus_still_validates_under_the_stricter_rules() -> None:
    """Every receipt the shipped product has published must remain valid."""
    corpus = Path(__file__).resolve().parents[2] / "examples"
    assert corpus.is_dir(), "the example directory ships with the package"
