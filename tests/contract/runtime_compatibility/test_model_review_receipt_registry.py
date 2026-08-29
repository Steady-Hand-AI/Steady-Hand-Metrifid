"""Pure receipt contracts for exact runtime-to-registry binding."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from metrifid.json_values import CanonicalValue, canonical_sha256, compute_self_hash
from metrifid.model_release._decision import ModelReleaseDecision
from metrifid.model_release._policy import (
    MODEL_RELEASE_POLICY_SCHEMA,
    MODEL_RELEASE_POLICY_SCHEMA_VERSION,
    ModelReleasePolicy,
)
from metrifid.model_release._public_field_registry_catalog import (
    PUBLIC_FIELD_REGISTRY_SCHEMA_VERSION,
    characterized_registry,
)
from metrifid.model_release._receipt import (
    build_model_release_receipt,
    validate_model_release_receipt,
)
from metrifid.model_release._receipt_validation import (
    MODEL_RELEASE_RECEIPT_SCHEMA_VERSION,
    certification_runtime_base_version,
    model_release_decision_sha256,
)
from metrifid.model_release._status import ModelReleaseStatus

_MODEL_XML = """
<mujoco model="runtime-registry-receipt">
  <worldbody><body name="body"><geom name="geom" size="0.1"/></body></worldbody>
</mujoco>
"""


def _artifact_sha256(receipt: Mapping[str, CanonicalValue], role: str) -> str:
    """Read one role's complete-MJB digest from a validated Certify receipt."""
    role_value = cast(dict[str, Any], receipt[role])
    artifact = cast(dict[str, Any], role_value["compiled_artifact"])
    return cast(str, artifact["mjb_sha256"])


def _empty_policy(certification: Mapping[str, CanonicalValue]) -> ModelReleasePolicy:
    """Build one empty policy bound to an identical compiled-artifact pair."""
    baseline_sha256 = _artifact_sha256(certification, "baseline")
    candidate_sha256 = _artifact_sha256(certification, "candidate")
    primitive: dict[str, CanonicalValue] = {
        "schema": MODEL_RELEASE_POLICY_SCHEMA,
        "schema_version": MODEL_RELEASE_POLICY_SCHEMA_VERSION,
        "baseline_compiled_sha256": baseline_sha256,
        "candidate_compiled_sha256": candidate_sha256,
        "rules": [],
    }
    return ModelReleasePolicy(
        MODEL_RELEASE_POLICY_SCHEMA,
        MODEL_RELEASE_POLICY_SCHEMA_VERSION,
        baseline_sha256,
        candidate_sha256,
        (),
        "a" * 64,
        canonical_sha256(primitive),
    )


@pytest.fixture(scope="module")
def model_release_receipt(tmp_path_factory: pytest.TempPathFactory) -> dict[str, CanonicalValue]:
    """Produce one real linked Certify/model-release receipt under the installed runtime."""
    from metrifid import _runtime_identity
    from metrifid.certify import _run as certify_run
    from metrifid.certify import certify_models

    root = tmp_path_factory.mktemp("runtime-registry-receipt")
    baseline_root = root / "baseline"
    candidate_root = root / "candidate"
    baseline_root.mkdir()
    candidate_root.mkdir()
    baseline = baseline_root / "model.xml"
    candidate = candidate_root / "model.xml"
    baseline.write_text(_MODEL_XML, encoding="utf-8")
    candidate.write_text(_MODEL_XML, encoding="utf-8")
    patcher = pytest.MonkeyPatch()
    patcher.setattr(certify_run, "installed_distribution_sha256", lambda: "9" * 64)
    patcher.setattr(_runtime_identity, "installed_distribution_sha256", lambda: "9" * 64)
    try:
        certification = certify_models(str(baseline), str(candidate), str(root / "out")).receipt
    finally:
        patcher.undo()
    policy = _empty_policy(certification)
    decision = ModelReleaseDecision(
        ModelReleaseStatus.NO_COMPILED_CHANGE,
        (),
        (),
        (),
    )
    base_version = certification_runtime_base_version(certification)
    registry = characterized_registry(base_version)
    assert registry is not None
    return build_model_release_receipt(
        policy=policy,
        decision=decision,
        certification_receipt=certification,
        registry_sha256=registry.comparable_registry_sha256,
        registry_count=registry.comparable_registry_count,
    )


def _reseal_outer(receipt: dict[str, Any]) -> dict[str, Any]:
    """Recompute both model-release hashes after a registry attack."""
    receipt["decision_sha256"] = model_release_decision_sha256(receipt)
    receipt["receipt_sha256"] = None
    receipt["receipt_sha256"] = compute_self_hash(receipt, "receipt_sha256")
    return receipt


def _reseal_certification_runtime(
    receipt: dict[str, Any],
    base_version: str,
    version_integer: int,
) -> None:
    """Create an internally resealed linked Certify runtime for adversarial validation."""
    from metrifid.certify._receipt_contract import _DECISION_MEMBERS

    certification = cast(dict[str, Any], receipt["certification_receipt"])
    runtime = cast(dict[str, Any], certification["runtime_identity"])
    runtime["mujoco_version"] = base_version
    runtime["mujoco_version_string"] = base_version
    runtime["mujoco_version_integer"] = version_integer
    runtime["mjb_header_words"][3] = version_integer
    runtime["runtime_identity_sha256"] = None
    runtime["runtime_identity_sha256"] = compute_self_hash(runtime, "runtime_identity_sha256")
    for role in ("baseline", "candidate"):
        role_value = cast(dict[str, Any], certification[role])
        artifact = cast(dict[str, Any], role_value["compiled_artifact"])
        artifact["header_words"][3] = version_integer
        artifact["mujoco_version_integer"] = version_integer
        artifact["runtime_identity_sha256"] = runtime["runtime_identity_sha256"]
    certification["decision_sha256"] = canonical_sha256(
        cast(CanonicalValue, {name: certification[name] for name in _DECISION_MEMBERS})
    )
    certification["receipt_sha256"] = None
    certification["receipt_sha256"] = compute_self_hash(certification, "receipt_sha256")
    receipt["certification_receipt_sha256"] = certification["receipt_sha256"]
    receipt["certification_decision_sha256"] = certification["decision_sha256"]


def _other_cataloged_base_version(current: str, *, distinct_tuple: bool) -> str:
    """Select another characterized version, optionally requiring a different tuple."""
    current_entry = characterized_registry(current)
    assert current_entry is not None
    for candidate in ("3.9.0", "3.10.0", "3.11.0", "3.12.0"):
        entry = characterized_registry(candidate)
        assert entry is not None
        tuple_differs = (
            entry.comparable_registry_sha256 != current_entry.comparable_registry_sha256
            or entry.comparable_registry_count != current_entry.comparable_registry_count
        )
        if candidate != current and (not distinct_tuple or tuple_differs):
            return candidate
    raise AssertionError("the catalog must contain an alternate measured profile")


def test_new_receipt_binds_registry_to_linked_certify_runtime(
    model_release_receipt: dict[str, CanonicalValue],
) -> None:
    """Preserve schema v1 while deriving its expected tuple from the linked runtime."""
    validate_model_release_receipt(model_release_receipt)
    certification = cast(dict[str, Any], model_release_receipt["certification_receipt"])
    registry = cast(dict[str, Any], model_release_receipt["public_field_registry"])
    assert model_release_receipt["schema_version"] == MODEL_RELEASE_RECEIPT_SCHEMA_VERSION
    assert set(registry) == {"schema", "schema_version", "sha256", "field_count"}
    assert registry["schema_version"] == PUBLIC_FIELD_REGISTRY_SCHEMA_VERSION
    expected = characterized_registry(certification_runtime_base_version(certification))
    assert expected is not None
    assert registry["sha256"] == expected.comparable_registry_sha256
    assert registry["field_count"] == expected.comparable_registry_count


def test_cross_version_registry_reseal_is_rejected(
    model_release_receipt: dict[str, CanonicalValue],
) -> None:
    """Reject another catalog row even after both unsigned outer hashes are recomputed."""
    attacked = cast(dict[str, Any], copy.deepcopy(model_release_receipt))
    registry = cast(dict[str, Any], attacked["public_field_registry"])
    certification = cast(dict[str, Any], attacked["certification_receipt"])
    current = certification_runtime_base_version(certification)
    substituted = _other_cataloged_base_version(current, distinct_tuple=True)
    substituted_entry = characterized_registry(substituted)
    assert substituted_entry is not None
    registry["sha256"] = substituted_entry.comparable_registry_sha256
    registry["field_count"] = substituted_entry.comparable_registry_count
    with pytest.raises(ValueError, match="versioned catalog"):
        validate_model_release_receipt(_reseal_outer(attacked))


@pytest.mark.parametrize(
    ("member", "expected_message"),
    [("sha256", "SHA-256"), ("field_count", "field_count")],
    ids=("changed_hash", "changed_count"),
)
def test_changed_registry_hash_and_count_are_rejected(
    model_release_receipt: dict[str, CanonicalValue],
    member: str,
    expected_message: str,
) -> None:
    """Reject either tuple component after recomputing both outer receipt hashes."""
    attacked = cast(dict[str, Any], copy.deepcopy(model_release_receipt))
    registry = cast(dict[str, Any], attacked["public_field_registry"])
    if member == "sha256":
        registry[member] = "0" * 64
    else:
        registry[member] = cast(int, registry[member]) + 1
    with pytest.raises(ValueError, match=expected_message):
        validate_model_release_receipt(_reseal_outer(attacked))


def test_unknown_linked_runtime_is_rejected_after_complete_reseal(
    model_release_receipt: dict[str, CanonicalValue],
) -> None:
    """Reject an internally coherent Certify reseal with no characterized registry."""
    attacked = cast(dict[str, Any], copy.deepcopy(model_release_receipt))
    _reseal_certification_runtime(attacked, "99.0.0", 99_000_000)
    registry = cast(dict[str, Any], attacked["public_field_registry"])
    registry["sha256"] = "0" * 64
    registry["field_count"] = 0
    with pytest.raises(ValueError, match="no characterized public-field registry"):
        validate_model_release_receipt(_reseal_outer(attacked))


def test_linked_runtime_selects_its_own_catalog_entry(
    model_release_receipt: dict[str, CanonicalValue],
) -> None:
    """A resealed linked runtime cannot retain an older version's distinct registry tuple."""
    attacked = cast(dict[str, Any], copy.deepcopy(model_release_receipt))
    certification = cast(dict[str, Any], attacked["certification_receipt"])
    current = certification_runtime_base_version(certification)
    substituted = _other_cataloged_base_version(current, distinct_tuple=True)
    major, minor, patch = (int(part) for part in substituted.split("."))
    _reseal_certification_runtime(
        attacked,
        substituted,
        major * 1_000_000 + minor * 1_000 + patch,
    )
    with pytest.raises((ValueError, TypeError), match="versioned catalog"):
        validate_model_release_receipt(_reseal_outer(attacked))


def test_pure_registry_validation_does_not_read_external_artifacts(
    model_release_receipt: dict[str, CanonicalValue],
    tmp_path: Path,
) -> None:
    """Validate entirely from linked receipt facts without native or artifact execution."""
    absent_artifact = tmp_path / "never-created.mjb"
    validate_model_release_receipt(model_release_receipt)
    assert not absent_artifact.exists()
