"""Build the frozen, self-hashed static model-release decision receipt."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .._json_admission import JsonAdmissionError
from ..certify import validate_receipt as validate_certification_receipt
from ..json_values import (
    CanonicalValue,
    canonical_json_bytes,
    compute_self_hash,
    freeze_canonical,
    require_sha256,
    thaw_canonical,
)
from ._decision import ChangeClassification, ModelReleaseDecision, ModelReleaseDecisionRefusal
from ._policy import ModelReleasePolicy
from ._public_field_registry_catalog import (
    PUBLIC_FIELD_REGISTRY_SCHEMA,
    PUBLIC_FIELD_REGISTRY_SCHEMA_VERSION,
    characterized_registry,
)
from ._receipt_validation import (
    _LIMITATION_STATEMENTS,
    DYNAMIC_BEHAVIOR_CLAIM,
    MODEL_RELEASE_RECEIPT_SCHEMA,
    MODEL_RELEASE_RECEIPT_SCHEMA_VERSION,
    REQUIRED_LIMITATIONS,
    STATIC_CLAIM_KIND,
    STATIC_CLAIM_STATEMENT,
    certification_runtime_base_version,
    load_and_validate_model_release_receipt,
    model_release_decision_sha256,
    validate_model_release_receipt,
)
from ._status import ModelReleaseStatus, model_release_exit_code


@dataclass(frozen=True, slots=True)
class ModelReleaseResult:
    """One completed static model-release decision and its two published files."""

    status: ModelReleaseStatus
    receipt: dict[str, CanonicalValue]
    model_release_json: Path
    model_release_markdown: Path

    @property
    def receipt_sha256(self) -> str:
        """Return the validated SHA-256 identity of this receipt."""
        return require_sha256(self.receipt.get("receipt_sha256"), "receipt_sha256")


def build_model_release_receipt(
    *,
    policy: ModelReleasePolicy,
    decision: ModelReleaseDecision,
    certification_receipt: Mapping[str, CanonicalValue],
    registry_sha256: str,
    registry_count: int,
) -> dict[str, CanonicalValue]:
    """Build, decision-hash, self-hash, and independently validate one completed receipt."""
    if not isinstance(policy, ModelReleasePolicy):
        raise TypeError("policy must be a ModelReleasePolicy")
    if not isinstance(decision, ModelReleaseDecision):
        raise TypeError("decision must be a ModelReleaseDecision")
    certification = _canonical_object_copy(certification_receipt, "certification_receipt")
    validate_certification_receipt(certification)
    runtime_base_version = certification_runtime_base_version(certification)
    expected_registry = characterized_registry(runtime_base_version)
    if expected_registry is None:
        raise ValueError("linked Certify runtime has no characterized public-field registry")
    if (
        registry_sha256 != expected_registry.comparable_registry_sha256
        or registry_count != expected_registry.comparable_registry_count
    ):
        raise ValueError("measured public-field registry differs from the versioned catalog")

    changes: list[CanonicalValue] = [change.to_primitive() for change in decision.changes]
    missing: list[CanonicalValue] = [
        rule.to_primitive() for rule in decision.missing_required_rules
    ]
    policy_rules: list[CanonicalValue] = [rule.to_primitive() for rule in policy.rules]
    receipt: dict[str, CanonicalValue] = {
        "schema": MODEL_RELEASE_RECEIPT_SCHEMA,
        "schema_version": MODEL_RELEASE_RECEIPT_SCHEMA_VERSION,
        "status": decision.status.value,
        "completed_exit_code": model_release_exit_code(decision.status),
        "certification_receipt": certification,
        "certification_receipt_sha256": _receipt_digest(certification, "receipt_sha256"),
        "certification_decision_sha256": _receipt_digest(certification, "decision_sha256"),
        "policy": {
            "schema": policy.schema,
            "schema_version": policy.schema_version,
            "baseline_compiled_sha256": policy.baseline_compiled_sha256,
            "candidate_compiled_sha256": policy.candidate_compiled_sha256,
            "rules": policy_rules,
            "rule_count": len(policy.rules),
            "raw_sha256": policy.raw_sha256,
            "semantic_sha256": policy.semantic_sha256,
        },
        "public_field_registry": {
            "schema": PUBLIC_FIELD_REGISTRY_SCHEMA,
            "schema_version": PUBLIC_FIELD_REGISTRY_SCHEMA_VERSION,
            "sha256": registry_sha256,
            "field_count": registry_count,
        },
        "changes_complete": True,
        "changes": changes,
        "change_count": len(changes),
        "classification_counts": {
            classification.value: sum(
                change.classification is classification for change in decision.changes
            )
            for classification in ChangeClassification
        },
        "satisfied_required_rule_ids": list(decision.satisfied_required_rule_ids),
        "missing_required_rules": missing,
        "first_unexpected_witness": (
            None if decision.first_unexpected is None else decision.first_unexpected.to_primitive()
        ),
        "first_missing_required_witness": (
            None
            if decision.first_missing_required is None
            else decision.first_missing_required.to_primitive()
        ),
        "static_claim": {
            "claim_kind": STATIC_CLAIM_KIND,
            "statement": STATIC_CLAIM_STATEMENT,
        },
        "dynamic_behavior_claim": DYNAMIC_BEHAVIOR_CLAIM,
        "limitations": [
            {"code": code, "statement": _LIMITATION_STATEMENTS[code]}
            for code in REQUIRED_LIMITATIONS
        ],
        "decision_sha256": None,
        "receipt_sha256": None,
    }
    receipt["decision_sha256"] = model_release_decision_sha256(receipt)
    receipt["receipt_sha256"] = compute_self_hash(receipt, "receipt_sha256")
    validate_model_release_receipt(receipt)
    try:
        load_and_validate_model_release_receipt(canonical_json_bytes(receipt))
    except JsonAdmissionError as exc:
        raise ModelReleaseDecisionRefusal(
            "serialized_receipt_budget_exceeded",
            exception_type=type(exc).__name__,
        ) from exc
    return receipt


def _canonical_object_copy(
    value: Mapping[str, CanonicalValue], field: str
) -> dict[str, CanonicalValue]:
    """Deep-copy one canonical mapping so later caller mutation cannot alter the receipt."""
    frozen = freeze_canonical(cast("CanonicalValue", dict(value)))
    thawed = thaw_canonical(frozen)
    if type(thawed) is not dict:
        raise TypeError(f"{field} must be a canonical object")
    return thawed


def _receipt_digest(receipt: Mapping[str, CanonicalValue], field: str) -> str:
    """Read one required digest from a validated embedded receipt."""
    return require_sha256(receipt.get(field), f"certification_receipt.{field}")


__all__ = [
    "DYNAMIC_BEHAVIOR_CLAIM",
    "MODEL_RELEASE_RECEIPT_SCHEMA",
    "MODEL_RELEASE_RECEIPT_SCHEMA_VERSION",
    "REQUIRED_LIMITATIONS",
    "ModelReleaseResult",
    "build_model_release_receipt",
    "load_and_validate_model_release_receipt",
    "validate_model_release_receipt",
]
