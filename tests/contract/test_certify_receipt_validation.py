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
    """Publish a self-consistent receipt for byte-identical compiled models."""
    return _certify(tmp_path_factory.mktemp("certified"), _BASELINE_XML, _BASELINE_XML)


@pytest.fixture(scope="module")
def differing_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Publish a self-consistent receipt for models with different compiled bytes."""
    return _certify(tmp_path_factory.mktemp("differing"), _BASELINE_XML, _CANDIDATE_XML)


def _load_receipt_json(path: Path) -> dict[str, Any]:
    """Decode a published receipt into a mutable mapping for validation attacks."""
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


def _assert_resealed_receipt_rejected(receipt: dict[str, Any], mutate: Any) -> None:
    """Prove semantic validation rejects one contradiction after both hashes are resealed."""
    validate_receipt(receipt)
    contradicted = copy.deepcopy(receipt)
    mutate(contradicted)
    with pytest.raises((ValueError, TypeError, KeyError)):
        validate_receipt(_resealed(contradicted))


def _tampered(receipt: dict[str, Any], mutate: Any) -> dict[str, Any]:
    """Copy, mutate, and reseal a receipt to isolate semantic validation."""
    copied = copy.deepcopy(receipt)
    mutate(copied)
    return _resealed(copied)


# Keep the mutation registry and its tests dense enough for the audited contract-module limit.
# fmt: off
_CERTIFIED_TAMPERS: tuple[tuple[str, Any], ...] = (
    (
        "runtime self-hash",
        lambda r: r["runtime_identity"].__setitem__("runtime_identity_sha256", "0" * 64),
    ),
    (
        "runtime header word",
        lambda r: r["runtime_identity"].__setitem__(
            "mjb_header_words", [54321, 8, 92, 3010000, 472]
        ),
    ),
    (
        "runtime execution mode",
        lambda r: r["runtime_identity"].__setitem__("execution_mode", "STEPPED"),
    ),
    (
        "artifact runtime binding",
        lambda r: r["baseline"]["compiled_artifact"].__setitem__(
            "runtime_identity_sha256", "1" * 64
        ),
    ),
    (
        "artifact digest",
        lambda r: r["candidate"]["compiled_artifact"].__setitem__("mjb_sha256", "2" * 64),
    ),
    (
        "artifact size",
        lambda r: r["baseline"]["compiled_artifact"].__setitem__("mjb_size_bytes", 999),
    ),
    (
        "artifact method",
        lambda r: r["baseline"]["compiled_artifact"].__setitem__(
            "method", "SHA256_OF_SOMETHING_ELSE"
        ),
    ),
    (
        "artifact magic hex",
        lambda r: r["baseline"]["compiled_artifact"].__setitem__("magic_hex", "0xdeadbeef"),
    ),
    (
        "artifact mjtnum width",
        lambda r: r["baseline"]["compiled_artifact"].__setitem__("sizeof_mjtnum", 4),
    ),
    (
        "artifact version integer",
        lambda r: r["baseline"]["compiled_artifact"].__setitem__(
            "mujoco_version_integer",
            int(r["baseline"]["compiled_artifact"]["mujoco_version_integer"]) + 1,
        ),
    ),
    (
        "source closure total bytes",
        lambda r: r["baseline"].__setitem__("source_closure_total_bytes", 1),
    ),
    (
        "source closure digest",
        lambda r: r["baseline"].__setitem__("source_closure_sha256", "3" * 64),
    ),
    ("role name", lambda r: r["baseline"].__setitem__("role", "candidate")),
    (
        "comparison size",
        lambda r: r["byte_comparison"].__setitem__("baseline_mjb_size_bytes", 12345),
    ),
    ("comparison equal flag", lambda r: r["byte_comparison"].__setitem__("equal", False)),
    (
        "artifact claim statement",
        lambda r: r["artifact_claim"].__setitem__("statement", "Everything is equivalent."),
    ),
    ("artifact claim exclusions", lambda r: r["artifact_claim"]["does_not_claim"].pop()),
    ("implication premise", lambda r: r["behavior_implication"]["premises"].pop()),
    (
        "implication statement",
        lambda r: r["behavior_implication"].__setitem__("statement", "It behaves the same."),
    ),
    (
        "implication in decision hash",
        lambda r: r["behavior_implication"].__setitem__("included_in_decision_sha256", True),
    ),
    (
        "limitation statement",
        lambda r: r["limitations"][0].__setitem__("statement", "No limits apply."),
    ),
    ("tool name", lambda r: r["tool"].__setitem__("name", "some_other_tool")),
    (
        "field report on a certificate",
        lambda r: r.__setitem__(
            "field_report", {"schema": "metrifid.compiled_field_report", "schema_version": 1}
        ),
    ),
    ("unknown root field", lambda r: r.__setitem__("extra_root_field", 1)),
    ("missing root field", lambda r: r.pop("limitations")),
)


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
    """Return the field-report mapping required on a compiled-difference receipt."""
    report = receipt["field_report"]
    assert isinstance(report, dict)
    return report


def _first_changed(receipt: dict[str, Any]) -> dict[str, Any]:
    """Select the first changed-field entry for focused contradiction attacks."""
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
    ("negative root count", lambda r: _require_field_report(r).__setitem__("fields_omitted_count", -1)),
    ("boolean root count", lambda r: _require_field_report(r).__setitem__("fields_compared_count", True)),
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
        lambda r: _require_field_report(r)["omitted_fields"][0].__setitem__("reason", "BECAUSE_I_SAID_SO"),
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


def test_the_field_report_cannot_change_the_status(differing_output: Path) -> None:
    """Prevent descriptive field evidence from upgrading a compiled-difference status."""
    receipt = _load_receipt_json(differing_output / "certification.json")
    report = receipt["field_report"]
    assert report is not None
    assert receipt["byte_comparison"]["equal"] is False
    assert receipt["status"] == "NOT_CERTIFIED_COMPILED_DIFFERS"
    receipt["field_report"] = None
    receipt["status"] = "CERTIFIED_COMPILED_EQUIVALENCE"
    with pytest.raises(ValueError):
        validate_receipt(receipt)


def test_a_tampered_status_fails_revalidation(certified_output: Path) -> None:
    """Reject a status edit whose decision and receipt hashes remain stale."""
    receipt = _load_receipt_json(certified_output / "certification.json")
    receipt["status"] = "NOT_CERTIFIED_COMPILED_DIFFERS"
    with pytest.raises(ValueError):
        validate_receipt(receipt)


def test_a_dropped_limitation_fails_revalidation(certified_output: Path) -> None:
    """Reject receipts that omit any mandatory certification limitation."""
    receipt = _load_receipt_json(certified_output / "certification.json")
    receipt["limitations"] = receipt["limitations"][:-1]
    with pytest.raises(ValueError):
        validate_receipt(receipt)


def test_a_tampered_self_hash_fails_revalidation(certified_output: Path) -> None:
    """Reject a forged receipt self-hash even when every semantic fact is unchanged."""
    receipt = _load_receipt_json(certified_output / "certification.json")
    receipt["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_receipt(receipt)


def test_resealing_an_untouched_receipt_still_validates(certified_output: Path) -> None:
    """The reseal helper itself is faithful, so a rejection below is never an artifact of it."""
    receipt = _load_receipt_json(certified_output / "certification.json")
    resealed = _resealed(receipt)
    assert resealed == receipt
    validate_receipt(resealed)


@pytest.mark.parametrize(
    ("label", "mutate"), _CERTIFIED_TAMPERS, ids=[t[0] for t in _CERTIFIED_TAMPERS]
)
def test_a_resealed_certified_tamper_is_rejected(
    certified_output: Path, label: str, mutate: Any
) -> None:
    """Reject resealed contradictions in certificate roles, runtime, claims, and schema."""
    receipt = _tampered(_load_receipt_json(certified_output / "certification.json"), mutate)
    with pytest.raises((ValueError, TypeError, KeyError)):
        validate_receipt(receipt)


@pytest.mark.parametrize(
    ("label", "mutate"), _DIFFERING_TAMPERS, ids=[t[0] for t in _DIFFERING_TAMPERS]
)
def test_a_resealed_differing_tamper_is_rejected(
    differing_output: Path, label: str, mutate: Any
) -> None:
    """Reject resealed contradictions specific to compiled-difference receipts."""
    receipt = _tampered(_load_receipt_json(differing_output / "certification.json"), mutate)
    with pytest.raises((ValueError, TypeError, KeyError)):
        validate_receipt(receipt)


def test_the_root_member_set_is_exact(certified_output: Path) -> None:
    """Pin the complete root member set for independently parsed receipts."""
    receipt = _load_receipt_json(certified_output / "certification.json")
    assert tuple(sorted(receipt)) == tuple(sorted(_ROOT_NAMES))


def test_the_runtime_and_both_artifacts_are_mutually_bound(certified_output: Path) -> None:
    """Bind both artifacts to the receipt runtime, header words, and MuJoCo version."""
    receipt = _load_receipt_json(certified_output / "certification.json")
    runtime = receipt["runtime_identity"]
    for role in ("baseline", "candidate"):
        artifact = receipt[role]["compiled_artifact"]
        assert artifact["runtime_identity_sha256"] == runtime["runtime_identity_sha256"]
        assert artifact["header_words"] == runtime["mjb_header_words"]
        assert artifact["mujoco_version_integer"] == runtime["mujoco_version_integer"]
        assert artifact["magic_decimal"] == artifact["header_words"][0]
        assert artifact["magic_hex"] == f"0x{artifact['header_words'][0]:08x}"
        assert artifact["sizeof_mjtnum"] == artifact["header_words"][1]


def test_comparison_sizes_digests_and_equality_are_mutually_consistent(
    certified_output: Path, differing_output: Path
) -> None:
    """Derive comparison sizes and equality from both compiled artifact identities."""
    for path, expected_equal in ((certified_output, True), (differing_output, False)):
        receipt = _load_receipt_json(path / "certification.json")
        comparison = receipt["byte_comparison"]
        baseline = receipt["baseline"]["compiled_artifact"]
        candidate = receipt["candidate"]["compiled_artifact"]
        assert comparison["baseline_mjb_size_bytes"] == baseline["mjb_size_bytes"]
        assert comparison["candidate_mjb_size_bytes"] == candidate["mjb_size_bytes"]
        assert comparison["compared_byte_count"] == min(
            baseline["mjb_size_bytes"], candidate["mjb_size_bytes"]
        )
        identical = (
            baseline["mjb_size_bytes"] == candidate["mjb_size_bytes"]
            and baseline["mjb_sha256"] == candidate["mjb_sha256"]
        )
        assert comparison["equal"] is identical is expected_equal


def test_the_nested_runtime_identity_revalidates_on_its_own(certified_output: Path) -> None:
    """Require the embedded runtime identity to round-trip and validate independently."""
    from metrifid._runtime_identity import CertifyRuntimeIdentity

    receipt = _load_receipt_json(certified_output / "certification.json")
    parsed = CertifyRuntimeIdentity.from_primitive(receipt["runtime_identity"])
    parsed.validate_hash()
    assert parsed.to_primitive() == receipt["runtime_identity"]
    assert parsed.execution_mode == "NO_MJDATA_EXECUTION"
    assert len(parsed.mjb_header_words) == 5


def test_certified_receipt_declares_claim_and_limitations(certified_output: Path) -> None:
    """Verify that certification states its claim boundary and limitation codes."""
    receipt = _load_receipt_json(certified_output / "certification.json")
    assert receipt["artifact_claim"]["statement"]
    assert receipt["behavior_implication"]["statement"]
    assert [item["code"] for item in receipt["limitations"]] == [
        "EXACT_RECORDED_RUNTIME_ONLY",
        "POST_CERTIFICATION_MJMODEL_MUTATION_OUTSIDE_CLAIM",
        "EXTERNAL_CODE_AND_INPUT_EQUIVALENCE_NOT_ESTABLISHED",
        "NO_SOURCE_TEXT_LICENSE_OR_VISUAL_INTENT_CLAIM",
        "NO_CROSS_MUJOCO_VERSION_CLAIM",
    ]


@pytest.mark.parametrize(
    ("label", "mutate"), _EXIT_AND_TOOL_TAMPERS, ids=[c[0] for c in _EXIT_AND_TOOL_TAMPERS]
)
def test_a_resealed_exit_or_tool_tamper_is_rejected(
    certified_output: Path, label: str, mutate: Any
) -> None:
    """Reject resealed exit-type and installed-distribution identity contradictions."""
    receipt = _tampered(_load_receipt_json(certified_output / "certification.json"), mutate)
    with pytest.raises((ValueError, TypeError, KeyError)):
        validate_receipt(receipt)


@pytest.mark.parametrize(
    ("label", "mutate"), _FIELD_REPORT_TAMPERS, ids=[c[0] for c in _FIELD_REPORT_TAMPERS]
)
def test_a_resealed_field_report_tamper_is_rejected(
    differing_output: Path, label: str, mutate: Any
) -> None:
    """Reject resealed field-report count, ordering, schema, and omission contradictions."""
    receipt = _tampered(_load_receipt_json(differing_output / "certification.json"), mutate)
    with pytest.raises((ValueError, TypeError, KeyError)):
        validate_receipt(receipt)


def test_resealed_receipt_rejects_boolean_exit_code(certified_output: Path) -> None:
    """Reject boolean exit codes despite Python's integer-compatible boolean semantics."""
    _assert_resealed_receipt_rejected(_load_receipt_json(certified_output / "certification.json"), lambda changed: changed.__setitem__("completed_exit_code", False))


def test_resealed_receipt_rejects_status_exit_mismatch(certified_output: Path) -> None:
    """Reject a differing status that still carries certification's zero exit."""
    _assert_resealed_receipt_rejected(_load_receipt_json(certified_output / "certification.json"), lambda changed: changed.__setitem__("status", "NOT_CERTIFIED_COMPILED_DIFFERS"))


def test_resealed_receipt_rejects_unbound_tool_identity(certified_output: Path) -> None:
    """Reject a completed receipt whose producing tool identity becomes unbound."""
    _assert_resealed_receipt_rejected(_load_receipt_json(certified_output / "certification.json"), lambda changed: changed["tool"].__setitem__("execution_identity_state", "UNBOUND"))


def test_resealed_receipt_rejects_mismatched_tool_version(certified_output: Path) -> None:
    """Reject tool versions that disagree with the bound runtime identity."""
    _assert_resealed_receipt_rejected(_load_receipt_json(certified_output / "certification.json"), lambda changed: changed["tool"].__setitem__("version", "9.9.9"))


def test_resealed_receipt_rejects_distribution_digest_mismatch(certified_output: Path) -> None:
    """Reject tool distribution digests that differ from the runtime binding."""
    _assert_resealed_receipt_rejected(_load_receipt_json(certified_output / "certification.json"), lambda changed: changed["tool"].__setitem__("distribution_sha256", "f" * 64))


def test_resealed_receipt_rejects_impossible_changed_field_count(differing_output: Path) -> None:
    """Reject zero changed-field totals alongside a populated changed-field collection."""
    _assert_resealed_receipt_rejected(_load_receipt_json(differing_output / "certification.json"), lambda changed: _require_field_report(changed).__setitem__("changed_fields_total", 0))


def test_resealed_receipt_rejects_false_truncation_state(differing_output: Path) -> None:
    """Reject false truncation after a valid receipt deliberately withholds witnesses."""
    receipt = _load_receipt_json(differing_output / "certification.json")
    report = _require_field_report(receipt)
    entry = report["changed_fields"][0]
    entry["changed_element_count"] = len(entry["witnesses"]) + 1
    report["truncated"] = True
    _assert_resealed_receipt_rejected(_resealed(receipt), lambda changed: _require_field_report(changed).__setitem__("truncated", False))


def test_resealed_receipt_rejects_missing_changed_fields(differing_output: Path) -> None:
    """Reject an empty changed-field collection while positive counts remain."""
    _assert_resealed_receipt_rejected(_load_receipt_json(differing_output / "certification.json"), lambda changed: _require_field_report(changed).__setitem__("changed_fields", []))


def test_resealed_receipt_rejects_duplicate_changed_field_paths(differing_output: Path) -> None:
    """Reject two changed-field entries that claim the same public model path."""
    receipt = _load_receipt_json(differing_output / "certification.json")
    field_path = _require_field_report(receipt)["changed_fields"][0]["path"]
    _assert_resealed_receipt_rejected(receipt, lambda changed: _require_field_report(changed)["changed_fields"][1].__setitem__("path", field_path))


def test_resealed_receipt_rejects_invalid_witness_value_type(differing_output: Path) -> None:
    """Reject array-valued evidence that the field reporter cannot produce."""
    _assert_resealed_receipt_rejected(_load_receipt_json(differing_output / "certification.json"), lambda changed: _first_changed(changed)["witnesses"][0].__setitem__("baseline_value", []))


def test_resealed_receipt_rejects_artifact_runtime_binding_mismatch(certified_output: Path) -> None:
    """Reject compiled artifacts bound to a different runtime identity digest."""
    _assert_resealed_receipt_rejected(_load_receipt_json(certified_output / "certification.json"), lambda changed: changed["baseline"]["compiled_artifact"].__setitem__("runtime_identity_sha256", "e" * 64))


def test_resealed_receipt_rejects_source_closure_digest_mismatch(certified_output: Path) -> None:
    """Reject role digests that contradict their canonical source-closure evidence."""
    _assert_resealed_receipt_rejected(_load_receipt_json(certified_output / "certification.json"), lambda changed: changed["baseline"].__setitem__("source_closure_sha256", "d" * 64))
# fmt: on
