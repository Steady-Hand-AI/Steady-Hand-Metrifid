"""Contract tests for the pure, self-hashed static model-release receipt."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import textwrap
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from metrifid.json_values import CanonicalValue, canonical_json_bytes, canonical_sha256
from metrifid.model_release import _receipt as release_receipt
from metrifid.model_release._decision import (
    ChangeClassification,
    ClassifiedChange,
    ModelReleaseDecision,
    ModelReleaseDecisionRefusal,
    ObservedChange,
)
from metrifid.model_release._policy import (
    MODEL_RELEASE_POLICY_SCHEMA,
    MODEL_RELEASE_POLICY_SCHEMA_VERSION,
    ChangeKind,
    ModelReleasePolicy,
    PolicyEffect,
    PolicyObjectType,
    PolicyRule,
    PolicySelector,
)
from metrifid.model_release._public_field_registry_catalog import (
    characterized_registry,
)
from metrifid.model_release._receipt import (
    DYNAMIC_BEHAVIOR_CLAIM,
    build_model_release_receipt,
    load_and_validate_model_release_receipt,
    validate_model_release_receipt,
)
from metrifid.model_release._receipt_validation import (
    certification_runtime_base_version,
    model_release_decision_sha256,
)
from metrifid.model_release._status import ModelReleaseStatus

_BASELINE_XML = """
<mujoco model="model-release-receipt">
  <worldbody>
    <body name="b" pos="0 0 1">
      <geom name="g" type="sphere" size="0.1" mass="2"/>
      <joint name="j" type="hinge"/>
    </body>
  </worldbody>
</mujoco>
"""
_CANDIDATE_XML = _BASELINE_XML.replace('mass="2"', 'mass="3"')


@pytest.fixture(scope="module")
def certification_receipt(tmp_path_factory: pytest.TempPathFactory) -> dict[str, CanonicalValue]:
    """Produce one real differing Certify receipt for outer-link validation."""
    from metrifid import _runtime_identity
    from metrifid.certify import _run as certify_run
    from metrifid.certify import certify_models

    root = tmp_path_factory.mktemp("model-release-receipt")
    baseline_root = root / "baseline"
    candidate_root = root / "candidate"
    baseline_root.mkdir()
    candidate_root.mkdir()
    baseline = baseline_root / "model.xml"
    candidate = candidate_root / "model.xml"
    baseline.write_text(_BASELINE_XML, encoding="utf-8")
    candidate.write_text(_CANDIDATE_XML, encoding="utf-8")
    patcher = pytest.MonkeyPatch()
    patcher.setattr(certify_run, "installed_distribution_sha256", lambda: "9" * 64)
    patcher.setattr(_runtime_identity, "installed_distribution_sha256", lambda: "9" * 64)
    try:
        result = certify_models(str(baseline), str(candidate), str(root / "out"))
    finally:
        patcher.undo()
    return result.receipt


def _artifact_sha256(receipt: Mapping[str, CanonicalValue], role: str) -> str:
    """Read one validated role-local MJB digest from a Certify fixture."""
    role_value = cast(dict[str, Any], receipt[role])
    artifact = cast(dict[str, Any], role_value["compiled_artifact"])
    return cast(str, artifact["mjb_sha256"])


def _policy(
    certification: Mapping[str, CanonicalValue],
    rules: tuple[PolicyRule, ...],
) -> ModelReleasePolicy:
    """Construct one canonical policy bound to the real Certify artifact pair."""
    ordered = tuple(sorted(rules, key=PolicyRule.sort_key))
    primitive: dict[str, CanonicalValue] = {
        "schema": MODEL_RELEASE_POLICY_SCHEMA,
        "schema_version": MODEL_RELEASE_POLICY_SCHEMA_VERSION,
        "baseline_compiled_sha256": _artifact_sha256(certification, "baseline"),
        "candidate_compiled_sha256": _artifact_sha256(certification, "candidate"),
        "rules": [rule.to_primitive() for rule in ordered],
    }
    return ModelReleasePolicy(
        MODEL_RELEASE_POLICY_SCHEMA,
        MODEL_RELEASE_POLICY_SCHEMA_VERSION,
        _artifact_sha256(certification, "baseline"),
        _artifact_sha256(certification, "candidate"),
        ordered,
        "a" * 64,
        canonical_sha256(primitive),
    )


def _allowed_semantic_receipt(
    certification: Mapping[str, CanonicalValue],
    *,
    object_type: PolicyObjectType = PolicyObjectType.BODY,
    object_name: str = "b",
    field: str = "parent",
    before_value: CanonicalValue = "world",
    after_value: CanonicalValue = "new_parent",
) -> dict[str, CanonicalValue]:
    """Build one complete receipt with one policy-allowed semantic MODIFY row."""
    selector = PolicySelector(object_type, object_name, field, ChangeKind.MODIFY)
    before_sha256 = canonical_sha256(before_value)
    after_sha256 = canonical_sha256(after_value)
    rule = PolicyRule(
        "allow-change",
        PolicyEffect.ALLOW,
        selector,
        before_sha256,
        after_sha256,
    )
    policy = _policy(certification, (rule,))
    observed = ObservedChange(
        selector,
        "SEMANTIC_OBJECT",
        before_sha256,
        after_sha256,
        before_value,
        after_value,
        {},
    )
    classified = ClassifiedChange(observed, ChangeClassification.ALLOWED, rule.id)
    decision = ModelReleaseDecision(
        ModelReleaseStatus.WITHIN_DECLARED_POLICY,
        (classified,),
        (),
        (),
    )
    registry = characterized_registry(certification_runtime_base_version(certification))
    assert registry is not None
    return build_model_release_receipt(
        policy=policy,
        decision=decision,
        certification_receipt=certification,
        registry_sha256=registry.comparable_registry_sha256,
        registry_count=registry.comparable_registry_count,
    )


def _reseal(receipt: dict[str, Any]) -> dict[str, Any]:
    """Recompute both outer hashes after a semantic contradiction is injected."""
    from metrifid.json_values import compute_self_hash

    copied = copy.deepcopy(receipt)
    copied["decision_sha256"] = model_release_decision_sha256(copied)
    copied["receipt_sha256"] = None
    copied["receipt_sha256"] = compute_self_hash(copied, "receipt_sha256")
    return copied


def test_valid_roundtrip_has_complete_static_evidence_and_certify_links(
    certification_receipt: dict[str, CanonicalValue],
) -> None:
    """Build, serialize, strictly reload, and revalidate one completed receipt."""
    receipt = _allowed_semantic_receipt(certification_receipt)
    validate_model_release_receipt(receipt)
    loaded = load_and_validate_model_release_receipt(canonical_json_bytes(receipt))
    assert loaded == receipt
    assert receipt["dynamic_behavior_claim"] == DYNAMIC_BEHAVIOR_CLAIM
    assert receipt["changes_complete"] is True
    assert receipt["change_count"] == 1
    assert receipt["certification_receipt"] == certification_receipt
    assert receipt["certification_receipt_sha256"] == certification_receipt["receipt_sha256"]
    assert receipt["certification_decision_sha256"] == certification_receipt["decision_sha256"]


def test_builder_enforces_the_serialized_public_reader_bounds_before_returning(
    certification_receipt: dict[str, CanonicalValue],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make every producer-returned receipt admissible through the public serialized loader."""
    original = release_receipt.load_and_validate_model_release_receipt
    observed: list[bytes] = []

    def recording_loader(data: bytes | str) -> dict[str, CanonicalValue]:
        assert isinstance(data, bytes)
        observed.append(data)
        return original(data)

    monkeypatch.setattr(
        release_receipt, "load_and_validate_model_release_receipt", recording_loader
    )
    receipt = _allowed_semantic_receipt(certification_receipt)
    assert observed == [canonical_json_bytes(receipt)]


def test_builder_converts_serialized_reader_capacity_to_a_typed_refusal(
    certification_receipt: dict[str, CanonicalValue],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a bounded valid receipt outside the internal-error exit when it is too large."""
    from metrifid._json_admission import JsonAdmissionError

    def capacity_refusal(_data: bytes | str) -> dict[str, CanonicalValue]:
        raise JsonAdmissionError("synthetic receipt budget")

    monkeypatch.setattr(
        release_receipt, "load_and_validate_model_release_receipt", capacity_refusal
    )
    with pytest.raises(ModelReleaseDecisionRefusal) as captured:
        _allowed_semantic_receipt(certification_receipt)
    assert captured.value.evidence == {
        "issue": "serialized_receipt_budget_exceeded",
        "exception_type": "JsonAdmissionError",
    }


def test_semantic_json_null_is_hashed_as_a_real_modify_value(
    certification_receipt: dict[str, CanonicalValue],
) -> None:
    """Keep geom mesh null->name distinct from structural ADD-side absence."""
    receipt = _allowed_semantic_receipt(
        certification_receipt,
        object_type=PolicyObjectType.GEOM,
        object_name="g",
        field="mesh",
        before_value=None,
        after_value="named_mesh",
    )
    validate_model_release_receipt(receipt)
    change = cast(list[dict[str, Any]], receipt["changes"])[0]
    assert change["before_value"] is None
    assert change["before_sha256"] == canonical_sha256(None)
    assert change["before_sha256"] is not None


@pytest.mark.parametrize(
    ("transmission_type", "object_type", "target_count"),
    [
        ("JOINT", "JOINT", 1),
        ("JOINTINPARENT", "JOINT", 1),
        ("SLIDERCRANK", "SITE", 2),
        ("TENDON", "TENDON", 1),
        ("SITE", "SITE", 2),
        ("BODY", "BODY", 1),
    ],
)
def test_all_frozen_actuator_transmission_target_shapes_revalidate(
    certification_receipt: dict[str, CanonicalValue],
    transmission_type: str,
    object_type: str,
    target_count: int,
) -> None:
    """Admit stable typed/name target references for every supported transmission family."""
    before: dict[str, CanonicalValue] = {
        "transmission_type": transmission_type,
        "references": [
            {"object_type": object_type, "name": f"before-{index}"} for index in range(target_count)
        ],
    }
    after: dict[str, CanonicalValue] = {
        "transmission_type": transmission_type,
        "references": [
            {"object_type": object_type, "name": f"after-{index}"} for index in range(target_count)
        ],
    }
    receipt = _allowed_semantic_receipt(
        certification_receipt,
        object_type=PolicyObjectType.ACTUATOR,
        object_name="motor",
        field="targets",
        before_value=before,
        after_value=after,
    )
    validate_model_release_receipt(receipt)


def test_actuator_transmission_change_cannot_omit_its_redundant_target_row(
    certification_receipt: dict[str, CanonicalValue],
) -> None:
    """Reject a resealed incomplete receipt that changes transmission without target semantics."""
    with pytest.raises(ValueError, match="missing its targets change"):
        _allowed_semantic_receipt(
            certification_receipt,
            object_type=PolicyObjectType.ACTUATOR,
            object_name="motor",
            field="transmission",
            before_value=0,
            after_value=3,
        )


def _wrong_status(receipt: dict[str, Any]) -> None:
    receipt["status"] = ModelReleaseStatus.REVIEW_REQUIRED.value
    receipt["completed_exit_code"] = 40


def _wrong_classification(receipt: dict[str, Any]) -> None:
    receipt["changes"][0]["classification"] = ChangeClassification.FORBIDDEN.value


def _wrong_count(receipt: dict[str, Any]) -> None:
    receipt["classification_counts"][ChangeClassification.ALLOWED.value] = 2


def _wrong_witness(receipt: dict[str, Any]) -> None:
    receipt["first_unexpected_witness"] = copy.deepcopy(receipt["changes"][0])


def _wrong_satisfied_partition(receipt: dict[str, Any]) -> None:
    receipt["satisfied_required_rule_ids"] = ["allow-change"]


def _wrong_completion_marker(receipt: dict[str, Any]) -> None:
    receipt["changes_complete"] = False


@pytest.mark.parametrize(
    "mutate",
    [
        _wrong_status,
        _wrong_classification,
        _wrong_count,
        _wrong_witness,
        _wrong_satisfied_partition,
        _wrong_completion_marker,
    ],
)
def test_resealed_contradiction_corpus_is_rejected(
    certification_receipt: dict[str, CanonicalValue],
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    """Semantic invariants, not only hashes, reject a naively resealed forgery."""
    receipt = cast(dict[str, Any], _allowed_semantic_receipt(certification_receipt))
    mutate(receipt)
    with pytest.raises((TypeError, ValueError)):
        validate_model_release_receipt(_reseal(receipt))


@pytest.mark.parametrize("member", ["root", "change"])
def test_exact_member_sets_reject_unknown_fields(
    certification_receipt: dict[str, CanonicalValue], member: str
) -> None:
    """Unknown fields cannot create a shadow interpretation at either schema level."""
    receipt = cast(dict[str, Any], _allowed_semantic_receipt(certification_receipt))
    if member == "root":
        receipt["unregistered_root_member"] = True
    else:
        receipt["changes"][0]["unregistered_change_member"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        validate_model_release_receipt(_reseal(receipt))


def test_embedded_certification_hash_link_is_not_resealable_from_outer_receipt(
    certification_receipt: dict[str, CanonicalValue],
) -> None:
    """Outer resealing cannot hide a changed embedded Certify identity."""
    receipt = cast(dict[str, Any], _allowed_semantic_receipt(certification_receipt))
    receipt["certification_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match the embedded receipt"):
        validate_model_release_receipt(_reseal(receipt))


def test_dynamic_non_claim_and_limitations_are_frozen_under_resealing(
    certification_receipt: dict[str, CanonicalValue],
) -> None:
    """No forged dynamic or safety claim can survive exact receipt validation."""
    receipt = cast(dict[str, Any], _allowed_semantic_receipt(certification_receipt))
    receipt["dynamic_behavior_claim"] = "DYNAMICALLY_EQUIVALENT"
    with pytest.raises(ValueError, match="NO_DYNAMIC_BEHAVIOR_CLAIM"):
        validate_model_release_receipt(_reseal(receipt))


def test_raw_loader_refuses_duplicate_members() -> None:
    """Strict admission rejects duplicate names before semantic receipt validation."""
    payload = '{"schema":"a","schema":"b"}'
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_and_validate_model_release_receipt(payload)


def test_receipt_import_graph_remains_native_free() -> None:
    """Independent receipt validation imports neither MuJoCo nor NumPy."""
    blocker = textwrap.dedent(
        """
        import sys

        class BlockNative:
            def find_spec(self, name, path=None, target=None):
                if name.split('.', 1)[0] in {'mujoco', 'numpy'}:
                    raise ModuleNotFoundError(name)
                return None

        sys.meta_path.insert(0, BlockNative())
        import metrifid.model_release._receipt
        loaded = set(sys.modules)
        assert not any(name == 'mujoco' or name.startswith('mujoco.') for name in loaded)
        assert not any(name == 'numpy' or name.startswith('numpy.') for name in loaded)
        print('PURE')
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", blocker],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PURE" in completed.stdout


def test_fixture_is_strict_json_serializable(
    certification_receipt: dict[str, CanonicalValue], tmp_path: Path
) -> None:
    """The complete nested receipt contains only canonical JSON values."""
    receipt = _allowed_semantic_receipt(certification_receipt)
    path = tmp_path / "model_release.json"
    path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    assert (
        json.loads(path.read_text(encoding="utf-8"))["receipt_sha256"]
        == (receipt["receipt_sha256"])
    )
