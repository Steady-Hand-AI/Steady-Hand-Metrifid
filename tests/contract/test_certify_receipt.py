"""The frozen Certify status, exit mapping, claim separation and receipt schema."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest

from metrifid.certify import CertifyStatus, certify_exit_code, validate_receipt
from metrifid.certify._receipt import (
    ARTIFACT_CLAIM_EXCLUSIONS,
    BEHAVIOR_IMPLICATION,
    BEHAVIOR_IMPLICATION_PREMISES,
    CERTIFIED_ARTIFACT_CLAIM,
    NOT_CERTIFIED_ARTIFACT_CLAIM,
    RECEIPT_SCHEMA,
    REQUIRED_LIMITATIONS,
)
from metrifid.certify._status import CERTIFY_COMPLETED_EXIT_CODES
from metrifid.errors import ComparisonStatus, OperationalExitCode
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
    """Compile two MJCF inputs through Certify and return its artifact directory."""
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
    """Publish a receipt for byte-identical models through the real product path."""
    return _certify(tmp_path_factory.mktemp("certified"), _BASELINE_XML, _BASELINE_XML)


@pytest.fixture(scope="module")
def differing_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Publish a receipt for models whose compiled bytes differ by mass."""
    return _certify(tmp_path_factory.mktemp("differing"), _BASELINE_XML, _CANDIDATE_XML)


def _load_receipt_json(path: Path) -> dict[str, Any]:
    """Decode a published certification receipt for contract assertions."""
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
    from metrifid.json_values import canonical_sha256, compute_self_hash

    resealed = copy.deepcopy(receipt)
    resealed["decision_sha256"] = canonical_sha256(
        {name: resealed[name] for name in _DECISION_NAMES}
    )
    resealed["receipt_sha256"] = None
    resealed["receipt_sha256"] = compute_self_hash(resealed, "receipt_sha256")
    return resealed


def _tampered(receipt: dict[str, Any], mutate: Any) -> dict[str, Any]:
    """Copy, mutate, and reseal a receipt so semantic validation is isolated."""
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
    """Return the field report required on a compiled-difference receipt."""
    report = receipt["field_report"]
    assert isinstance(report, dict)
    return report


def _first_changed(receipt: dict[str, Any]) -> dict[str, Any]:
    """Select the first reported public-field difference for focused mutations."""
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


def test_the_completed_status_registry_is_exactly_two_values() -> None:
    """Limit completed certification to byte equality or compiled difference."""
    assert [status.value for status in CertifyStatus] == [
        "CERTIFIED_COMPILED_EQUIVALENCE",
        "NOT_CERTIFIED_COMPILED_DIFFERS",
    ]


def test_the_completed_exit_mapping_is_frozen() -> None:
    """Bind the two completed statuses to public process exits zero and forty."""
    assert certify_exit_code(CertifyStatus.CERTIFIED_COMPILED_EQUIVALENCE) == 0
    assert certify_exit_code(CertifyStatus.NOT_CERTIFIED_COMPILED_DIFFERS) == 40
    assert dict(CERTIFY_COMPLETED_EXIT_CODES) == {
        CertifyStatus.CERTIFIED_COMPILED_EQUIVALENCE: 0,
        CertifyStatus.NOT_CERTIFIED_COMPILED_DIFFERS: 40,
    }


def test_no_certify_status_leaks_into_the_comparison_status_registry() -> None:
    """Keep artifact certification outcomes separate from workload comparison decisions."""
    comparison = {status.value for status in ComparisonStatus}
    for status in CertifyStatus:
        assert status.value not in comparison


def test_the_completed_exits_are_certify_owned_plain_integers() -> None:
    """Keep completed exits plain integers while reserving operational refusal codes."""
    for value in CERTIFY_COMPLETED_EXIT_CODES.values():
        assert type(value) is int
        assert not isinstance(value, OperationalExitCode)
    assert 40 not in {int(code) for code in OperationalExitCode}
    assert int(OperationalExitCode.INVALID_INVOCATION_INPUT_OUTPUT) == 64
    assert int(OperationalExitCode.INTERNAL_PROJECT_FAILURE) == 70


def test_a_certify_status_is_required_for_the_exit_mapping() -> None:
    """Reject raw strings at the typed certification exit-mapping boundary."""
    with pytest.raises(TypeError):
        certify_exit_code("CERTIFIED_COMPILED_EQUIVALENCE")  # type: ignore[arg-type]


def test_a_certificate_carries_the_exact_unconditional_claim(certified_output: Path) -> None:
    """Pin the workload-free claim and its exclusions on an equality certificate."""
    receipt = _load_receipt_json(certified_output / "certification.json")
    claim = receipt["artifact_claim"]
    assert claim["claim_kind"] == "UNCONDITIONAL_ARTIFACT_STATEMENT"
    assert claim["workload_free"] is True
    assert claim["statement"] == CERTIFIED_ARTIFACT_CLAIM
    assert claim["does_not_claim"] == list(ARTIFACT_CLAIM_EXCLUSIONS)
    assert "guidance" not in claim


def test_the_behavior_implication_is_separate_conditional_and_fully_premised(
    certified_output: Path,
) -> None:
    """Keep behavioral implications conditional, fully premised, and outside artifact claims."""
    receipt = _load_receipt_json(certified_output / "certification.json")
    implication = receipt["behavior_implication"]
    assert implication["claim_kind"] == "CONDITIONAL_NON_DECISION_BEARING"
    assert implication["statement"] == BEHAVIOR_IMPLICATION
    assert implication["premises"] == list(BEHAVIOR_IMPLICATION_PREMISES)
    assert implication["included_in_decision_sha256"] is False
    assert implication["statement"] not in json.dumps(receipt["artifact_claim"])


def test_the_decision_hash_excludes_the_implication_report_and_limitations(
    certified_output: Path,
) -> None:
    """Restrict decision hashing to the frozen decision-bearing receipt members."""
    receipt = _load_receipt_json(certified_output / "certification.json")
    decision = {
        name: receipt[name]
        for name in (
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
    }
    assert receipt["decision_sha256"] == canonical_sha256(decision)
    for excluded in ("behavior_implication", "field_report", "artifact_claim", "limitations"):
        assert excluded not in decision


def test_the_decision_hash_ignores_a_changed_implication(certified_output: Path) -> None:
    """Show that editing explanatory implications cannot alter the decision digest."""
    receipt = _load_receipt_json(certified_output / "certification.json")
    before = receipt["decision_sha256"]
    receipt["behavior_implication"]["statement"] = "an entirely different sentence"
    decision = {
        name: receipt[name]
        for name in (
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
    }
    assert canonical_sha256(decision) == before


def test_every_required_limitation_is_present_in_the_frozen_order(certified_output: Path) -> None:
    """Pin every limitation code and its canonical receipt ordering."""
    receipt = _load_receipt_json(certified_output / "certification.json")
    assert [item["code"] for item in receipt["limitations"]] == list(REQUIRED_LIMITATIONS)
    assert all(item["statement"] for item in receipt["limitations"])


def test_the_receipt_schema_is_the_single_frozen_version(certified_output: Path) -> None:
    """Require the sole compiled-equivalence receipt schema and version."""
    receipt = _load_receipt_json(certified_output / "certification.json")
    assert receipt["schema"] == RECEIPT_SCHEMA == "metrifid.compiled_equivalence_receipt"
    assert receipt["schema_version"] == 1


def test_the_published_receipt_revalidates_exactly(certified_output: Path) -> None:
    """Revalidate the exact JSON emitted by the public certification path."""
    validate_receipt(_load_receipt_json(certified_output / "certification.json"))


def test_a_certificate_carries_no_field_report(certified_output: Path) -> None:
    """Exclude difference-only field evidence from byte-equality certificates."""
    assert _load_receipt_json(certified_output / "certification.json")["field_report"] is None


def test_a_difference_carries_the_negative_claim_and_directs_the_user_to_compare(
    differing_output: Path,
) -> None:
    """Direct compiled differences to workload comparison without implying behavior."""
    receipt = _load_receipt_json(differing_output / "certification.json")
    assert receipt["status"] == "NOT_CERTIFIED_COMPILED_DIFFERS"
    assert receipt["completed_exit_code"] == 40
    claim = receipt["artifact_claim"]
    assert claim["statement"] == NOT_CERTIFIED_ARTIFACT_CLAIM
    assert "behavioral significance" in claim["guidance"]
    assert "metrifid compare" in claim["guidance"]
    assert receipt["behavior_implication"] is None
    validate_receipt(receipt)


def test_a_difference_never_emits_a_comparison_decision(differing_output: Path) -> None:
    """Prevent workload comparison statuses from appearing in certification artifacts."""
    text = (differing_output / "certification.json").read_text(encoding="utf-8")
    text += (differing_output / "certification.md").read_text(encoding="utf-8")
    assert "MATERIAL_BEHAVIOR_CHANGE" not in text
    assert "NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD" not in text
