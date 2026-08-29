"""Pure, bounded validation for self-hashed model-release receipts.

The producer classifies changes while private compiled artifacts are available.  This module is
the independent-reader boundary: it admits no native dependency, revalidates the embedded Certify
receipt through its public API, reconstructs policy classification, and rejects contradictions
even when an attacker recomputes both outer hashes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, TypeVar, cast

from .._json_admission import RECEIPT_JSON_LIMITS, JsonAdmissionError, bounded_strict_json_loads
from .._schema_primitives import (
    _exact_int,
    _fields,
    _nonempty_string,
    _object,
    _optional_hash,
    _sequence,
)
from ..certify import CertifyStatus
from ..certify import validate_receipt as validate_certification_receipt
from ..json_values import (
    Binary64,
    CanonicalValue,
    canonical_sha256,
    require_sha256,
    validate_self_hash,
)
from ..schemas import TargetReference
from ._decision import (
    COMPILED_NAME_IDENTITY_MAPPING_FIELDS,
    MAX_COMPILED_CHANGES,
    ChangeClassification,
    ClassifiedChange,
    ObservedChange,
)
from ._policy import (
    MODEL_RELEASE_POLICY_SCHEMA,
    MODEL_RELEASE_POLICY_SCHEMA_VERSION,
    ChangeKind,
    ModelReleasePolicy,
    PolicyEffect,
    PolicyObjectType,
    PolicyRule,
    PolicySelector,
)
from ._public_field_registry_catalog import (
    PUBLIC_FIELD_REGISTRY_SCHEMA,
    PUBLIC_FIELD_REGISTRY_SCHEMA_VERSION,
    characterized_registry,
    coherent_runtime_base_version,
)
from ._status import ModelReleaseStatus, model_release_exit_code

MODEL_RELEASE_RECEIPT_SCHEMA: Final = "metrifid.model_release_receipt"
MODEL_RELEASE_RECEIPT_SCHEMA_VERSION: Final = 1
DYNAMIC_BEHAVIOR_CLAIM: Final = "NO_DYNAMIC_BEHAVIOR_CLAIM"
STATIC_CLAIM_KIND: Final = "STATIC_COMPILED_MODEL_CHANGE_POLICY_CLASSIFICATION"
STATIC_CLAIM_STATEMENT: Final = (
    "This receipt classifies the complete ordered set of statically observed compiled-model "
    "changes against the embedded bounded policy under the recorded Certify artifact identities."
)

REQUIRED_LIMITATIONS: Final = (
    "STATIC_ONLY_NO_DYNAMIC_EQUIVALENCE",
    "NO_HARDWARE_OR_OPERATIONAL_SAFETY_CLAIM",
    "DERIVED_COMPILED_CLOSURE_MUST_BE_DECLARED",
    "FROZEN_PUBLIC_FIELD_REGISTRY_ONLY",
    "NO_EXTERNAL_ARTIFACT_RECOMPUTATION",
)
_LIMITATION_STATEMENTS: Final[Mapping[str, str]] = {
    "STATIC_ONLY_NO_DYNAMIC_EQUIVALENCE": (
        "The decision is static. It does not establish trajectory, controller, workload, or "
        "dynamic-behavior equivalence."
    ),
    "NO_HARDWARE_OR_OPERATIONAL_SAFETY_CLAIM": (
        "The decision makes no hardware-safety, deployment-safety, task-suitability, or "
        "operational-readiness claim."
    ),
    "DERIVED_COMPILED_CLOSURE_MUST_BE_DECLARED": (
        "One intended source edit, including a mass edit, can change many derived compiled "
        "fields; every observed derived change remains independently policy-bearing."
    ),
    "FROZEN_PUBLIC_FIELD_REGISTRY_ONLY": (
        "Typed explanations use only the frozen public-field and semantic registries. Any "
        "unexplained complete-MJB difference is represented by a fail-closed opaque residual."
    ),
    "NO_EXTERNAL_ARTIFACT_RECOMPUTATION": (
        "Receipt revalidation establishes internal consistency and hash linkage; it does not "
        "recompile source models or independently reconstruct the private MJB snapshots."
    ),
}

_ROOT_MEMBERS: Final = (
    "schema",
    "schema_version",
    "status",
    "completed_exit_code",
    "certification_receipt",
    "certification_receipt_sha256",
    "certification_decision_sha256",
    "policy",
    "public_field_registry",
    "changes_complete",
    "changes",
    "change_count",
    "classification_counts",
    "satisfied_required_rule_ids",
    "missing_required_rules",
    "first_unexpected_witness",
    "first_missing_required_witness",
    "static_claim",
    "dynamic_behavior_claim",
    "limitations",
    "decision_sha256",
    "receipt_sha256",
)
_DECISION_MEMBERS: Final = (
    "schema",
    "schema_version",
    "status",
    "completed_exit_code",
    "certification_receipt_sha256",
    "certification_decision_sha256",
    "policy",
    "public_field_registry",
    "changes_complete",
    "changes",
    "change_count",
    "classification_counts",
    "satisfied_required_rule_ids",
    "missing_required_rules",
    "first_unexpected_witness",
    "first_missing_required_witness",
)
_POLICY_MEMBERS: Final = {
    "schema",
    "schema_version",
    "baseline_compiled_sha256",
    "candidate_compiled_sha256",
    "rules",
    "rule_count",
    "raw_sha256",
    "semantic_sha256",
}
_REGISTRY_MEMBERS: Final = {"schema", "schema_version", "sha256", "field_count"}
_CHANGE_MEMBERS: Final = {
    "selector",
    "source",
    "classification",
    "rule_id",
    "before_sha256",
    "after_sha256",
    "before_value",
    "after_value",
    "details",
}
_CLASSIFICATION_MEMBERS: Final = {member.value for member in ChangeClassification}
_SOURCES: Final = {"SEMANTIC_OBJECT", "COMPILED_PUBLIC_FIELD", "OPAQUE_ARTIFACT_RESIDUAL"}
_ACTUATOR_TRANSMISSION_TYPES: Final = {
    0: "JOINT",
    1: "JOINTINPARENT",
    2: "SLIDERCRANK",
    3: "TENDON",
    4: "SITE",
    5: "BODY",
}
_OPAQUE_REASONS: Final = (
    "candidate_compiled_subject_unbound",
    "candidate_compiled_subject_mismatch",
    "no_public_or_semantic_field_difference_identified",
    "semantic_name_coverage_incomplete",
    "compiled_name_identity_mapping_changed",
)
_EnumT = TypeVar("_EnumT", bound=StrEnum)


@dataclass(frozen=True, slots=True)
class _ValidatedChange:
    """One structurally validated row and its reconstructed typed classification."""

    primitive: dict[str, CanonicalValue]
    classified: ClassifiedChange


def model_release_decision_sha256(receipt: Mapping[str, CanonicalValue]) -> str:
    """Hash exactly the frozen decision-bearing root members."""
    decision = {name: receipt[name] for name in _DECISION_MEMBERS}
    return canonical_sha256(cast("CanonicalValue", decision))


def load_and_validate_model_release_receipt(
    data: bytes | str,
) -> dict[str, CanonicalValue]:
    """Strictly admit and validate one serialized model-release receipt.

    Duplicate members, raw floats, nonstandard constants, excessive input size, excessive
    nesting, and semantic contradictions are refused before a mapping is returned.
    """
    parsed = bounded_strict_json_loads(data, RECEIPT_JSON_LIMITS)
    if type(parsed) is not dict:
        raise JsonAdmissionError("model-release receipt must be a JSON object")
    validate_model_release_receipt(parsed)
    return parsed


def validate_model_release_receipt(receipt: Mapping[str, CanonicalValue]) -> None:
    """Revalidate one unsigned receipt as an independent pure-Python reader."""
    obj = _object(receipt, "model-release receipt")
    _fields(obj, set(_ROOT_MEMBERS), "model-release receipt")
    _validate_header(obj)
    certification, baseline_sha256, candidate_sha256, runtime_base_version = (
        _validate_certification(obj)
    )
    policy = _validate_policy(obj["policy"])
    if policy.baseline_compiled_sha256 != baseline_sha256:
        raise ValueError("policy baseline subject does not match the certified baseline MJB")
    _validate_registry(
        obj["public_field_registry"],
        runtime_base_version=runtime_base_version,
    )

    validated_changes = _validate_changes(obj["changes"], policy)
    changes = tuple(row.classified for row in validated_changes)
    _validate_change_summary(obj, validated_changes, policy)
    _validate_artifact_change_linkage(
        certification,
        policy,
        changes,
        baseline_sha256,
        candidate_sha256,
    )
    _validate_claims(obj)

    expected_decision_sha256 = model_release_decision_sha256(receipt)
    if require_sha256(obj["decision_sha256"], "decision_sha256") != expected_decision_sha256:
        raise ValueError("decision_sha256 does not match the decision-bearing members")
    validate_self_hash(cast("dict[str, CanonicalValue]", obj), "receipt_sha256")


def _validate_header(obj: Mapping[str, object]) -> None:
    """Validate schema, completed status, and the status-owned exit code."""
    if obj["schema"] != MODEL_RELEASE_RECEIPT_SCHEMA:
        raise ValueError("model-release receipt schema is outside the frozen registry")
    if _exact_int(obj["schema_version"], "schema_version") != MODEL_RELEASE_RECEIPT_SCHEMA_VERSION:
        raise ValueError("model-release receipt schema_version is outside the frozen registry")
    status = _enum_member(ModelReleaseStatus, obj["status"], "status")
    if _exact_int(obj["completed_exit_code"], "completed_exit_code") != model_release_exit_code(
        status
    ):
        raise ValueError("completed_exit_code does not match the model-release status")


def _validate_certification(
    obj: Mapping[str, object],
) -> tuple[dict[str, object], str, str, str]:
    """Validate the linked Certify receipt, artifact identities, and coherent runtime base."""
    certification = _object(obj["certification_receipt"], "certification_receipt")
    validate_certification_receipt(cast("dict[str, CanonicalValue]", certification))
    embedded_receipt_sha256 = require_sha256(
        certification["receipt_sha256"], "certification_receipt.receipt_sha256"
    )
    if (
        require_sha256(obj["certification_receipt_sha256"], "certification_receipt_sha256")
        != embedded_receipt_sha256
    ):
        raise ValueError("certification_receipt_sha256 does not match the embedded receipt")
    embedded_decision_sha256 = require_sha256(
        certification["decision_sha256"], "certification_receipt.decision_sha256"
    )
    if (
        require_sha256(obj["certification_decision_sha256"], "certification_decision_sha256")
        != embedded_decision_sha256
    ):
        raise ValueError("certification_decision_sha256 does not match the embedded receipt")
    baseline = _object(certification["baseline"], "certification_receipt.baseline")
    candidate = _object(certification["candidate"], "certification_receipt.candidate")
    baseline_artifact = _object(
        baseline["compiled_artifact"], "certification_receipt.baseline.compiled_artifact"
    )
    candidate_artifact = _object(
        candidate["compiled_artifact"], "certification_receipt.candidate.compiled_artifact"
    )
    runtime_base_version = certification_runtime_base_version(certification)
    return (
        certification,
        require_sha256(baseline_artifact["mjb_sha256"], "baseline MJB SHA-256"),
        require_sha256(candidate_artifact["mjb_sha256"], "candidate MJB SHA-256"),
        runtime_base_version,
    )


def certification_runtime_base_version(certification: Mapping[str, object]) -> str:
    """Derive a coherent stable MuJoCo base version from one validated Certify receipt."""
    runtime = _object(
        certification["runtime_identity"],
        "certification_receipt.runtime_identity",
    )
    return coherent_runtime_base_version(
        _nonempty_string(
            runtime["mujoco_version"],
            "certification_receipt.runtime_identity.mujoco_version",
        ),
        _nonempty_string(
            runtime["mujoco_version_string"],
            "certification_receipt.runtime_identity.mujoco_version_string",
        ),
        _exact_int(
            runtime["mujoco_version_integer"],
            "certification_receipt.runtime_identity.mujoco_version_integer",
        ),
    )


def _validate_policy(value: object) -> ModelReleasePolicy:
    """Reconstruct the embedded complete policy and verify its canonical semantic identity."""
    obj = _object(value, "policy")
    _fields(obj, _POLICY_MEMBERS, "policy")
    raw_rules = _sequence(obj["rules"], "policy.rules")
    rules = tuple(PolicyRule.from_primitive(rule) for rule in raw_rules)
    rule_count = _exact_int(obj["rule_count"], "policy.rule_count")
    if rule_count != len(rules):
        raise ValueError("policy.rule_count does not match the complete rule array")
    if obj["schema"] != MODEL_RELEASE_POLICY_SCHEMA:
        raise ValueError("embedded policy schema is outside the frozen registry")
    schema_version = _exact_int(obj["schema_version"], "policy.schema_version")
    if schema_version != MODEL_RELEASE_POLICY_SCHEMA_VERSION:
        raise ValueError("embedded policy schema_version is outside the frozen registry")
    baseline_sha256 = require_sha256(
        obj["baseline_compiled_sha256"], "policy.baseline_compiled_sha256"
    )
    candidate_sha256 = _optional_hash(
        obj["candidate_compiled_sha256"], "policy.candidate_compiled_sha256"
    )
    return ModelReleasePolicy(
        MODEL_RELEASE_POLICY_SCHEMA,
        schema_version,
        baseline_sha256,
        candidate_sha256,
        rules,
        require_sha256(obj["raw_sha256"], "policy.raw_sha256"),
        require_sha256(obj["semantic_sha256"], "policy.semantic_sha256"),
    )


def _validate_registry(
    value: object,
    *,
    runtime_base_version: str,
) -> None:
    """Derive and require the exact versioned public compiled-field registry identity."""
    obj = _object(value, "public_field_registry")
    _fields(obj, _REGISTRY_MEMBERS, "public_field_registry")
    if obj["schema"] != PUBLIC_FIELD_REGISTRY_SCHEMA:
        raise ValueError("public_field_registry schema is outside the frozen registry")
    if (
        _exact_int(obj["schema_version"], "public_field_registry.schema_version")
        != PUBLIC_FIELD_REGISTRY_SCHEMA_VERSION
    ):
        raise ValueError("public_field_registry schema_version is outside the frozen registry")
    expected = characterized_registry(runtime_base_version)
    if expected is None:
        raise ValueError("linked Certify runtime has no characterized public-field registry")
    if require_sha256(obj["sha256"], "public_field_registry.sha256") != (
        expected.comparable_registry_sha256
    ):
        raise ValueError("public_field_registry SHA-256 differs from the versioned catalog")
    if _exact_int(obj["field_count"], "public_field_registry.field_count") != (
        expected.comparable_registry_count
    ):
        raise ValueError("public_field_registry field_count differs from the versioned catalog")


def _validate_changes(value: object, policy: ModelReleasePolicy) -> tuple[_ValidatedChange, ...]:
    """Validate every complete change row, classification, and canonical ordering."""
    raw_changes = _sequence(value, "changes")
    if len(raw_changes) > MAX_COMPILED_CHANGES:
        raise ValueError(f"change count exceeds the maximum of {MAX_COMPILED_CHANGES}")
    rule_index = _build_rule_index(policy.rules)
    changes = tuple(_validate_change(raw, rule_index) for raw in raw_changes)
    sort_keys = [row.classified.observed.sort_key() for row in changes]
    if sort_keys != sorted(sort_keys):
        raise ValueError("changes are not in canonical first-witness order")
    if len(sort_keys) != len(set(sort_keys)):
        raise ValueError("changes contain a duplicate selector witness")
    _validate_actuator_change_redundancy(changes)
    return changes


def _validate_actuator_change_redundancy(changes: tuple[_ValidatedChange, ...]) -> None:
    """Require transmission and target rows to preserve their producer-side redundancy."""
    rows = {
        (
            row.classified.observed.selector.object_name,
            row.classified.observed.selector.field,
            row.classified.observed.selector.change_kind,
        ): row.classified.observed
        for row in changes
        if row.classified.observed.selector.object_type is PolicyObjectType.ACTUATOR
    }
    for row in changes:
        observed = row.classified.observed
        selector = observed.selector
        if (
            selector.object_type is not PolicyObjectType.ACTUATOR
            or selector.change_kind is not ChangeKind.MODIFY
        ):
            continue
        counterpart = (
            selector.object_name,
            "targets" if selector.field == "transmission" else "transmission",
            ChangeKind.MODIFY,
        )
        if selector.field == "transmission" and counterpart not in rows:
            raise ValueError("actuator transmission change is missing its targets change")
        if selector.field != "targets":
            continue
        before = _object(observed.before_value, "actuator targets before value")
        after = _object(observed.after_value, "actuator targets after value")
        if before["transmission_type"] != after["transmission_type"] and counterpart not in rows:
            raise ValueError("actuator targets transmission changed without transmission row")
        if counterpart not in rows:
            continue
        transmission = rows[counterpart]
        before_transmission = _exact_int(
            transmission.before_value, "actuator transmission before value"
        )
        after_transmission = _exact_int(
            transmission.after_value, "actuator transmission after value"
        )
        if (
            before["transmission_type"] != _ACTUATOR_TRANSMISSION_TYPES[before_transmission]
            or after["transmission_type"] != _ACTUATOR_TRANSMISSION_TYPES[after_transmission]
        ):
            raise ValueError("actuator transmission change contradicts its target semantics")


RuleKey = tuple[PolicyObjectType, str, str, ChangeKind]


def _build_rule_index(rules: tuple[PolicyRule, ...]) -> dict[RuleKey, PolicyRule]:
    """Index independently admitted nonoverlapping selectors for bounded validation."""
    return {
        (
            rule.selector.object_type,
            rule.selector.object_name,
            rule.selector.field,
            rule.selector.change_kind,
        ): rule
        for rule in rules
    }


def _validate_change(value: object, rule_index: dict[RuleKey, PolicyRule]) -> _ValidatedChange:
    """Validate one row and independently reconstruct its policy disposition."""
    obj = _object(value, "change")
    _fields(obj, _CHANGE_MEMBERS, "change")
    selector = PolicySelector.from_primitive(obj["selector"])
    source = _nonempty_string(obj["source"], "change.source")
    if source not in _SOURCES:
        raise ValueError("change.source is outside the frozen source registry")
    before_sha256 = _optional_hash(obj["before_sha256"], "change.before_sha256")
    after_sha256 = _optional_hash(obj["after_sha256"], "change.after_sha256")
    details = _object(obj["details"], "change.details")
    primitive = cast("dict[str, CanonicalValue]", obj)
    observed = ObservedChange(
        selector,
        source,
        before_sha256,
        after_sha256,
        cast("CanonicalValue", obj["before_value"]),
        cast("CanonicalValue", obj["after_value"]),
        cast("dict[str, CanonicalValue]", details),
        force_undeclared=source == "OPAQUE_ARTIFACT_RESIDUAL",
    )
    _validate_source_shape(observed)
    expected = _classify_change(observed, rule_index)
    classification = _enum_member(
        ChangeClassification, obj["classification"], "change.classification"
    )
    rule_id = _optional_nonempty_string(obj["rule_id"], "change.rule_id")
    if classification is not expected.classification or rule_id != expected.rule_id:
        raise ValueError("change classification or rule_id contradicts the embedded policy")
    return _ValidatedChange(primitive, ClassifiedChange(observed, classification, rule_id))


def _validate_source_shape(change: ObservedChange) -> None:
    """Bind each producer source token to its exact selector and evidence shape."""
    selector = change.selector
    if change.source == "SEMANTIC_OBJECT":
        if selector.object_type in {PolicyObjectType.COMPILED_FIELD, PolicyObjectType.OPAQUE}:
            raise ValueError("SEMANTIC_OBJECT source has an incompatible object_type")
        if selector.object_name == "*":
            raise ValueError("SEMANTIC_OBJECT change cannot use the policy wildcard as a name")
        if selector.change_kind in {ChangeKind.ADD, ChangeKind.REMOVE}:
            if selector.field != "presence":
                raise ValueError("semantic additions/removals must use the presence field")
        elif selector.field == "presence":
            raise ValueError("semantic presence cannot be modified")
        if change.details:
            raise ValueError("SEMANTIC_OBJECT change.details must be empty")
        _validate_semantic_value_hashes(change)
        return
    if change.source == "COMPILED_PUBLIC_FIELD":
        if selector.object_type is not PolicyObjectType.COMPILED_FIELD:
            raise ValueError("COMPILED_PUBLIC_FIELD source requires compiled_field object_type")
        if change.before_value is not None or change.after_value is not None:
            raise ValueError("compiled public-field values must remain digest-only")
        _validate_compiled_field_details(change)
        return
    _validate_opaque_change(change)


def _validate_semantic_value_hashes(change: ObservedChange) -> None:
    """Recompute semantic-side canonical hashes and enforce ADD/REMOVE direction."""
    kind = change.selector.change_kind
    if kind is ChangeKind.ADD:
        if change.before_value is not None:
            raise ValueError("semantic ADD must have an absent before side")
        _validate_semantic_object(change.selector.object_type, change.after_value)
    elif kind is ChangeKind.REMOVE:
        _validate_semantic_object(change.selector.object_type, change.before_value)
        if change.after_value is not None:
            raise ValueError("semantic REMOVE must have an absent after side")
    else:
        _validate_semantic_field_value(
            change.selector.object_type, change.selector.field, change.before_value
        )
        _validate_semantic_field_value(
            change.selector.object_type, change.selector.field, change.after_value
        )
    expected_before = None if kind is ChangeKind.ADD else canonical_sha256(change.before_value)
    expected_after = None if kind is ChangeKind.REMOVE else canonical_sha256(change.after_value)
    if change.before_sha256 != expected_before or change.after_sha256 != expected_after:
        raise ValueError("semantic change hashes do not match the embedded canonical values")
    if change.before_sha256 == change.after_sha256:
        raise ValueError("semantic change must identify two different sides")


def _validate_semantic_object(object_type: PolicyObjectType, value: CanonicalValue) -> None:
    """Validate the exact full semantic-field object used by a presence ADD or REMOVE."""
    obj = _object(value, f"{object_type.value} presence value")
    field_registry = {
        PolicyObjectType.BODY: ("parent", "mass", "inertia"),
        PolicyObjectType.JOINT: ("body", "type", "limited", "range"),
        PolicyObjectType.GEOM: ("body", "mesh"),
        PolicyObjectType.MESH: ("compiled_geometry_sha256",),
        PolicyObjectType.ACTUATOR: ("transmission", "targets"),
    }
    fields = field_registry[object_type]
    _fields(obj, set(fields), f"{object_type.value} presence value")
    for field in fields:
        _validate_semantic_field_value(
            object_type,
            field,
            cast("CanonicalValue", obj[field]),
        )
    if object_type is PolicyObjectType.ACTUATOR:
        transmission = _exact_int(obj["transmission"], "semantic actuator.transmission")
        targets = _object(obj["targets"], "semantic actuator.targets")
        if targets["transmission_type"] != _ACTUATOR_TRANSMISSION_TYPES[transmission]:
            raise ValueError("actuator presence transmission contradicts its target semantics")


def _validate_semantic_field_value(
    object_type: PolicyObjectType,
    field: str,
    value: CanonicalValue,
) -> None:
    """Validate one value against the closed typed semantic-field registry."""
    if object_type is PolicyObjectType.BODY:
        _validate_body_field(field, value)
        return
    if object_type is PolicyObjectType.JOINT:
        _validate_joint_field(field, value)
        return
    if object_type is PolicyObjectType.GEOM:
        _validate_geom_field(field, value)
        return
    if object_type is PolicyObjectType.MESH:
        if field != "compiled_geometry_sha256":
            raise ValueError("semantic mesh field is outside the closed typed registry")
        require_sha256(value, "semantic mesh.compiled_geometry_sha256")
        return
    if object_type is PolicyObjectType.ACTUATOR:
        _validate_actuator_field(field, value)
        return
    raise ValueError("semantic field value is outside the closed typed registry")


def _validate_body_field(field: str, value: CanonicalValue) -> None:
    """Validate one body semantic value."""
    if field == "parent":
        _nonempty_string(value, "semantic body.parent")
    elif field == "mass":
        Binary64.from_primitive(value)
    elif field == "inertia":
        _binary64_vector(value, 3, "semantic body.inertia")
    else:
        raise ValueError("semantic body field is outside the closed typed registry")


def _validate_joint_field(field: str, value: CanonicalValue) -> None:
    """Validate one joint semantic value."""
    if field == "body":
        _nonempty_string(value, "semantic joint.body")
    elif field == "type":
        joint_type = _exact_int(value, "semantic joint.type")
        if joint_type not in {0, 1, 2, 3}:
            raise ValueError("semantic joint.type is outside the MuJoCo joint-type registry")
    elif field == "limited":
        if type(value) is not bool:
            raise TypeError("semantic joint.limited must be a boolean")
    elif field == "range":
        _binary64_vector(value, 2, "semantic joint.range")
    else:
        raise ValueError("semantic joint field is outside the closed typed registry")


def _validate_geom_field(field: str, value: CanonicalValue) -> None:
    """Validate one geom semantic value, preserving null as a real mesh reference value."""
    if field == "body":
        _nonempty_string(value, "semantic geom.body")
    elif field == "mesh":
        if value is not None:
            _nonempty_string(value, "semantic geom.mesh")
    else:
        raise ValueError("semantic geom field is outside the closed typed registry")


def _validate_actuator_field(field: str, value: CanonicalValue) -> None:
    """Validate one actuator semantic value."""
    if field == "transmission":
        transmission_id = _exact_int(value, "semantic actuator.transmission")
        if transmission_id not in _ACTUATOR_TRANSMISSION_TYPES:
            raise ValueError("semantic actuator.transmission is outside the frozen registry")
    elif field == "targets":
        target_set = _object(value, "semantic actuator.targets")
        _fields(
            target_set,
            {"transmission_type", "references"},
            "semantic actuator.targets",
        )
        transmission_type = _nonempty_string(
            target_set["transmission_type"], "semantic actuator.targets.transmission_type"
        )
        expected_shapes: dict[str, tuple[str, int, int]] = {
            "JOINT": ("JOINT", 1, 1),
            "JOINTINPARENT": ("JOINT", 1, 1),
            "SLIDERCRANK": ("SITE", 2, 2),
            "TENDON": ("TENDON", 1, 1),
            "SITE": ("SITE", 1, 2),
            "BODY": ("BODY", 1, 1),
        }
        try:
            expected_type, minimum, maximum = expected_shapes[transmission_type]
        except KeyError:
            raise ValueError(
                "semantic actuator.targets transmission is outside the frozen registry"
            ) from None
        references = _sequence(target_set["references"], "semantic actuator.targets.references")
        if not minimum <= len(references) <= maximum:
            raise ValueError("semantic actuator target count contradicts transmission")
        for target in references:
            _validate_actuator_target(target, expected_type)
    else:
        raise ValueError("semantic actuator field is outside the closed typed registry")


def _validate_actuator_target(value: object, expected_type: str) -> None:
    """Validate one exact named transmission target from the shared descriptor schema."""
    target = TargetReference.from_primitive(value)
    if target.object_type != expected_type:
        raise ValueError("semantic actuator target object_type contradicts transmission")


def _binary64_vector(value: CanonicalValue, length: int, field: str) -> None:
    """Validate one fixed-length array of exact tagged binary64 values."""
    items = _sequence(value, field)
    if len(items) != length:
        raise ValueError(f"{field} must contain exactly {length} values")
    for item in items:
        Binary64.from_primitive(item)


def _validate_compiled_field_details(change: ObservedChange) -> None:
    """Validate digest direction and the complete baseline/candidate field metadata pair."""
    details = change.details
    _fields(details, {"baseline", "candidate"}, "compiled public-field details")
    baseline = _validate_field_detail(details["baseline"], "baseline")
    candidate = _validate_field_detail(details["candidate"], "candidate")
    kind = change.selector.change_kind
    if kind is ChangeKind.ADD:
        if baseline is not None or candidate is None or change.before_sha256 is not None:
            raise ValueError("compiled-field ADD has contradictory side evidence")
        require_sha256(change.after_sha256, "compiled-field ADD after_sha256")
    elif kind is ChangeKind.REMOVE:
        if baseline is None or candidate is not None or change.after_sha256 is not None:
            raise ValueError("compiled-field REMOVE has contradictory side evidence")
        require_sha256(change.before_sha256, "compiled-field REMOVE before_sha256")
    else:
        if baseline is None or candidate is None:
            raise ValueError("compiled-field MODIFY requires both metadata sides")
        require_sha256(change.before_sha256, "compiled-field MODIFY before_sha256")
        require_sha256(change.after_sha256, "compiled-field MODIFY after_sha256")
    if change.before_sha256 == change.after_sha256:
        raise ValueError("compiled public-field change must identify two different sides")


def _validate_field_detail(value: CanonicalValue, role: str) -> dict[str, object] | None:
    """Validate one optional public-field kind/dtype/shape description."""
    if value is None:
        return None
    obj = _object(value, f"compiled public-field {role} detail")
    _fields(obj, {"kind", "dtype", "shape"}, f"compiled public-field {role} detail")
    kind = _nonempty_string(obj["kind"], f"compiled public-field {role} kind")
    if kind not in {"ndarray", "numpy_scalar", "bool", "int", "float", "bytes", "str"}:
        raise ValueError("compiled public-field kind is outside the frozen registry")
    dtype = obj["dtype"]
    shape = obj["shape"]
    if kind == "ndarray":
        _nonempty_string(dtype, f"compiled public-field {role} dtype")
        dimensions = _sequence(shape, f"compiled public-field {role} shape")
        for dimension in dimensions:
            if _exact_int(dimension, "compiled public-field dimension") < 0:
                raise ValueError("compiled public-field dimensions must be nonnegative")
    elif dtype is not None or shape is not None:
        raise ValueError("scalar public-field details must have null dtype and shape")
    return obj


def _validate_opaque_change(change: ObservedChange) -> None:
    """Require the one fixed fail-closed residual selector and bounded reason evidence."""
    selector = change.selector
    if (
        selector.object_type is not PolicyObjectType.OPAQUE
        or selector.object_name != "complete_mjb"
        or selector.field != "compiled_artifact"
        or selector.change_kind is not ChangeKind.MODIFY
    ):
        raise ValueError("OPAQUE_ARTIFACT_RESIDUAL must use the frozen complete-MJB selector")
    if change.before_value is not None or change.after_value is not None:
        raise ValueError("opaque residual values must remain digest-only")
    require_sha256(change.before_sha256, "opaque residual before_sha256")
    require_sha256(change.after_sha256, "opaque residual after_sha256")
    if change.before_sha256 == change.after_sha256:
        raise ValueError("opaque residual must identify different artifact subjects")
    _fields(
        change.details,
        {
            "reasons",
            "policy_candidate_compiled_sha256",
            "baseline_coverage_issues",
            "candidate_coverage_issues",
        },
        "opaque residual details",
    )
    reasons = tuple(
        _nonempty_string(item, "opaque residual reason")
        for item in _sequence(change.details["reasons"], "opaque residual reasons")
    )
    if not reasons or len(reasons) != len(set(reasons)):
        raise ValueError("opaque residual reasons must be nonempty and unique")
    if reasons != tuple(reason for reason in _OPAQUE_REASONS if reason in reasons):
        raise ValueError("opaque residual reasons are outside the frozen canonical order")
    _optional_hash(
        change.details["policy_candidate_compiled_sha256"],
        "opaque residual policy_candidate_compiled_sha256",
    )
    baseline_coverage = _coverage_issues(
        change.details["baseline_coverage_issues"], "baseline_coverage_issues"
    )
    candidate_coverage = _coverage_issues(
        change.details["candidate_coverage_issues"], "candidate_coverage_issues"
    )
    has_coverage = bool(baseline_coverage or candidate_coverage)
    if ("semantic_name_coverage_incomplete" in reasons) is not has_coverage:
        raise ValueError("opaque residual coverage reason contradicts its coverage issue arrays")


def _coverage_issues(value: CanonicalValue, field: str) -> tuple[str, ...]:
    """Validate deterministic semantic-coverage issue evidence."""
    issues = tuple(_nonempty_string(item, field) for item in _sequence(value, field))
    if issues != tuple(sorted(set(issues))):
        raise ValueError(f"{field} must be unique and sorted")
    return issues


def _validate_change_summary(
    obj: Mapping[str, object],
    rows: tuple[_ValidatedChange, ...],
    policy: ModelReleasePolicy,
) -> None:
    """Recompute counts, REQUIRE partition, witnesses, precedence, and completion marker."""
    if obj["changes_complete"] is not True:
        raise ValueError("changes_complete must be the literal true value")
    if _exact_int(obj["change_count"], "change_count") != len(rows):
        raise ValueError("change_count does not match the complete change array")
    _validate_classification_counts(obj["classification_counts"], rows)

    changes = tuple(row.classified for row in rows)
    required_rules = tuple(rule for rule in policy.rules if rule.effect is PolicyEffect.REQUIRE)
    expected_satisfied = tuple(
        rule.id for rule in required_rules if any(change.rule_id == rule.id for change in changes)
    )
    actual_satisfied = tuple(
        _nonempty_string(item, "satisfied_required_rule_ids")
        for item in _sequence(obj["satisfied_required_rule_ids"], "satisfied_required_rule_ids")
    )
    if actual_satisfied != expected_satisfied:
        raise ValueError("satisfied_required_rule_ids contradict the classified changes")
    expected_missing = tuple(
        rule for rule in required_rules if rule.id not in set(expected_satisfied)
    )
    actual_missing = tuple(
        PolicyRule.from_primitive(item)
        for item in _sequence(obj["missing_required_rules"], "missing_required_rules")
    )
    if actual_missing != expected_missing:
        raise ValueError("missing_required_rules do not complete the REQUIRE-rule partition")

    expected_unexpected = next(
        (
            row.primitive
            for row in rows
            if row.classified.classification
            in {ChangeClassification.FORBIDDEN, ChangeClassification.UNDECLARED}
        ),
        None,
    )
    if obj["first_unexpected_witness"] != expected_unexpected:
        raise ValueError("first_unexpected_witness is not the canonical first unexpected change")
    expected_missing_witness = expected_missing[0].to_primitive() if expected_missing else None
    if obj["first_missing_required_witness"] != expected_missing_witness:
        raise ValueError("first_missing_required_witness is not the first missing REQUIRE rule")

    expected_status = _derived_status(changes, expected_missing)
    actual_status = _enum_member(ModelReleaseStatus, obj["status"], "status")
    if actual_status is not expected_status:
        raise ValueError("status does not follow the frozen decision precedence")


def _validate_classification_counts(value: object, rows: tuple[_ValidatedChange, ...]) -> None:
    """Require all four exact classification counts, including explicit zeros."""
    obj = _object(value, "classification_counts")
    _fields(obj, _CLASSIFICATION_MEMBERS, "classification_counts")
    for classification in ChangeClassification:
        count = _exact_int(obj[classification.value], f"{classification.value} count")
        if count < 0:
            raise ValueError("classification counts must be nonnegative")
        expected = sum(row.classified.classification is classification for row in rows)
        if count != expected:
            raise ValueError(f"{classification.value} count contradicts the change array")


def _validate_artifact_change_linkage(
    certification: Mapping[str, object],
    policy: ModelReleasePolicy,
    changes: tuple[ClassifiedChange, ...],
    baseline_sha256: str,
    candidate_sha256: str,
) -> None:
    """Bind compiled equality/difference, policy subjects, and fail-closed residual evidence."""
    cert_status = _enum_member(CertifyStatus, certification["status"], "certification status")
    artifacts_differ = baseline_sha256 != candidate_sha256
    if (cert_status is CertifyStatus.CERTIFIED_COMPILED_EQUIVALENCE) is artifacts_differ:
        raise ValueError("embedded Certify status contradicts its artifact subjects")

    opaque = tuple(
        change for change in changes if change.observed.source == "OPAQUE_ARTIFACT_RESIDUAL"
    )
    if len(opaque) > 1:
        raise ValueError("at most one opaque complete-MJB residual is permitted")
    nonopaque_count = len(changes) - len(opaque)
    name_mapping_changed = any(
        change.observed.source == "COMPILED_PUBLIC_FIELD"
        and change.observed.selector.object_name in COMPILED_NAME_IDENTITY_MAPPING_FIELDS
        for change in changes
    )
    expected_subject_reason: str | None = None
    if artifacts_differ and policy.candidate_compiled_sha256 is None:
        expected_subject_reason = "candidate_compiled_subject_unbound"
    elif (
        policy.candidate_compiled_sha256 is not None
        and policy.candidate_compiled_sha256 != candidate_sha256
    ):
        expected_subject_reason = "candidate_compiled_subject_mismatch"
    if artifacts_differ and not changes:
        raise ValueError("a differing compiled artifact requires at least one change witness")
    if not artifacts_differ and nonopaque_count:
        raise ValueError("byte-identical compiled artifacts cannot carry typed change witnesses")
    requires_opaque = (
        expected_subject_reason is not None
        or (artifacts_differ and nonopaque_count == 0)
        or (artifacts_differ and name_mapping_changed)
    )
    if requires_opaque and not opaque:
        raise ValueError("the complete-MJB difference requires a fail-closed opaque residual")
    if not artifacts_differ and expected_subject_reason is None and opaque:
        raise ValueError(
            "byte-identical artifacts permit only a candidate-subject mismatch residual"
        )
    if not opaque:
        return
    _validate_opaque_linkage(
        opaque[0].observed,
        policy,
        baseline_sha256,
        candidate_sha256,
        expected_subject_reason,
        nonopaque_count,
        artifacts_differ,
        name_mapping_changed,
    )


def _validate_opaque_linkage(
    change: ObservedChange,
    policy: ModelReleasePolicy,
    baseline_sha256: str,
    candidate_sha256: str,
    expected_subject_reason: str | None,
    nonopaque_count: int,
    artifacts_differ: bool,
    name_mapping_changed: bool,
) -> None:
    """Cross-check opaque reason membership and its exact artifact-side hashes."""
    reasons = tuple(cast("list[str]", change.details["reasons"]))
    subject_reasons = {
        "candidate_compiled_subject_unbound",
        "candidate_compiled_subject_mismatch",
    }
    actual_subject_reasons = subject_reasons.intersection(reasons)
    expected_subject_reasons = (
        set() if expected_subject_reason is None else {expected_subject_reason}
    )
    if actual_subject_reasons != expected_subject_reasons:
        raise ValueError("opaque residual subject reason contradicts policy/candidate linkage")
    has_no_explanation = "no_public_or_semantic_field_difference_identified" in reasons
    if has_no_explanation is not (artifacts_differ and nonopaque_count == 0):
        raise ValueError("opaque no-field-difference reason contradicts the complete change array")
    has_name_mapping_reason = "compiled_name_identity_mapping_changed" in reasons
    if has_name_mapping_reason is not (artifacts_differ and name_mapping_changed):
        raise ValueError("opaque name-mapping reason contradicts the complete change array")
    if change.details["policy_candidate_compiled_sha256"] != policy.candidate_compiled_sha256:
        raise ValueError("opaque residual policy candidate hash contradicts the embedded policy")
    expected_before = (
        policy.candidate_compiled_sha256
        if expected_subject_reason == "candidate_compiled_subject_mismatch"
        else baseline_sha256
    )
    if change.before_sha256 != expected_before or change.after_sha256 != candidate_sha256:
        raise ValueError("opaque residual hashes do not bind the certified artifact subjects")


def _validate_claims(obj: Mapping[str, object]) -> None:
    """Require exact static scope, dynamic non-claim token, and all frozen limitations."""
    claim = _object(obj["static_claim"], "static_claim")
    _fields(claim, {"claim_kind", "statement"}, "static_claim")
    if claim["claim_kind"] != STATIC_CLAIM_KIND or claim["statement"] != STATIC_CLAIM_STATEMENT:
        raise ValueError("static_claim is not the frozen static-only statement")
    if obj["dynamic_behavior_claim"] != DYNAMIC_BEHAVIOR_CLAIM:
        raise ValueError("dynamic_behavior_claim must be NO_DYNAMIC_BEHAVIOR_CLAIM")
    limitations = _sequence(obj["limitations"], "limitations")
    if len(limitations) != len(REQUIRED_LIMITATIONS):
        raise ValueError("every required model-release limitation must be present")
    for raw, code in zip(limitations, REQUIRED_LIMITATIONS, strict=True):
        entry = _object(raw, "limitation")
        _fields(entry, {"code", "statement"}, "limitation")
        if entry["code"] != code or entry["statement"] != _LIMITATION_STATEMENTS[code]:
            raise ValueError("model-release limitations must use the frozen order and text")


def _classify_change(
    change: ObservedChange, rule_index: dict[RuleKey, PolicyRule]
) -> ClassifiedChange:
    """Independently apply the same bounded at-most-one-selector policy semantics."""
    if change.force_undeclared:
        return ClassifiedChange(change, ChangeClassification.UNDECLARED, None)
    selector = change.selector
    exact_key = (
        selector.object_type,
        selector.object_name,
        selector.field,
        selector.change_kind,
    )
    wildcard_key = (
        selector.object_type,
        "*",
        selector.field,
        selector.change_kind,
    )
    rule = rule_index.get(exact_key)
    if rule is None:
        rule = rule_index.get(wildcard_key)
    if rule is None:
        return ClassifiedChange(change, ChangeClassification.UNDECLARED, None)
    if rule.before_sha256 is not None and rule.before_sha256 != change.before_sha256:
        return ClassifiedChange(change, ChangeClassification.UNDECLARED, None)
    if rule.after_sha256 is not None and rule.after_sha256 != change.after_sha256:
        return ClassifiedChange(change, ChangeClassification.UNDECLARED, None)
    classification = {
        PolicyEffect.ALLOW: ChangeClassification.ALLOWED,
        PolicyEffect.REQUIRE: ChangeClassification.REQUIRED,
        PolicyEffect.FORBID: ChangeClassification.FORBIDDEN,
    }[rule.effect]
    return ClassifiedChange(change, classification, rule.id)


def _derived_status(
    changes: tuple[ClassifiedChange, ...], missing: tuple[PolicyRule, ...]
) -> ModelReleaseStatus:
    """Apply the exact forbidden/missing, undeclared, empty, declared precedence."""
    if missing or any(
        change.classification is ChangeClassification.FORBIDDEN for change in changes
    ):
        return ModelReleaseStatus.OUTSIDE_DECLARED_POLICY
    if any(change.classification is ChangeClassification.UNDECLARED for change in changes):
        return ModelReleaseStatus.REVIEW_REQUIRED
    if not changes:
        return ModelReleaseStatus.NO_COMPILED_CHANGE
    return ModelReleaseStatus.WITHIN_DECLARED_POLICY


def _optional_nonempty_string(value: object, field: str) -> str | None:
    """Admit a null or one nonempty UTF-8 string."""
    return None if value is None else _nonempty_string(value, field)


def _enum_member(enum_type: type[_EnumT], value: object, field: str) -> _EnumT:
    """Admit one exact string member of a closed string enum."""
    token = _nonempty_string(value, field)
    try:
        return enum_type(token)
    except ValueError as exc:
        raise ValueError(f"{field} is outside the frozen enum registry") from exc


__all__ = [
    "DYNAMIC_BEHAVIOR_CLAIM",
    "MODEL_RELEASE_RECEIPT_SCHEMA",
    "MODEL_RELEASE_RECEIPT_SCHEMA_VERSION",
    "REQUIRED_LIMITATIONS",
    "STATIC_CLAIM_KIND",
    "STATIC_CLAIM_STATEMENT",
    "load_and_validate_model_release_receipt",
    "model_release_decision_sha256",
    "validate_model_release_receipt",
]
