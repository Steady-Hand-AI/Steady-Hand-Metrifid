"""Deterministic compiled-change construction, policy classification, and status selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

from ..json_values import CanonicalValue, canonical_sha256
from ._policy import (
    ChangeKind,
    ModelReleasePolicy,
    PolicyEffect,
    PolicyObjectType,
    PolicyRule,
    PolicySelector,
)
from ._snapshot import CompiledModelSnapshot, PublicFieldFact, SemanticObjectFact
from ._status import ModelReleaseStatus

MAX_COMPILED_CHANGES: Final = 10_000
COMPILED_NAME_IDENTITY_MAPPING_FIELDS: Final = frozenset(
    {
        "names",
        "names_map",
        "nnames",
        "nnames_map",
        "name_actuatoradr",
        "name_bodyadr",
        "name_camadr",
        "name_eqadr",
        "name_excludeadr",
        "name_flexadr",
        "name_geomadr",
        "name_hfieldadr",
        "name_jntadr",
        "name_keyadr",
        "name_lightadr",
        "name_matadr",
        "name_meshadr",
        "name_numericadr",
        "name_pairadr",
        "name_pluginadr",
        "name_sensoradr",
        "name_siteadr",
        "name_skinadr",
        "name_tendonadr",
        "name_texadr",
        "name_textadr",
        "name_tupleadr",
    }
)

_OBJECT_TYPE_RANK: Final = {
    PolicyObjectType.BODY: 0,
    PolicyObjectType.JOINT: 1,
    PolicyObjectType.GEOM: 2,
    PolicyObjectType.MESH: 3,
    PolicyObjectType.ACTUATOR: 4,
    PolicyObjectType.COMPILED_FIELD: 5,
    PolicyObjectType.OPAQUE: 6,
}
_FIELD_RANK: Final = {
    PolicyObjectType.BODY: {"presence": 0, "parent": 1, "mass": 2, "inertia": 3},
    PolicyObjectType.JOINT: {
        "presence": 0,
        "body": 1,
        "type": 2,
        "limited": 3,
        "range": 4,
    },
    PolicyObjectType.GEOM: {"presence": 0, "body": 1, "mesh": 2},
    PolicyObjectType.MESH: {"presence": 0, "compiled_geometry_sha256": 1},
    PolicyObjectType.ACTUATOR: {"presence": 0, "transmission": 1, "targets": 2},
    PolicyObjectType.COMPILED_FIELD: {"value": 0},
    PolicyObjectType.OPAQUE: {"compiled_artifact": 0},
}
_CHANGE_KIND_RANK: Final = {ChangeKind.REMOVE: 0, ChangeKind.ADD: 1, ChangeKind.MODIFY: 2}


class ChangeClassification(StrEnum):
    """The four policy classifications attached to an observed compiled change."""

    ALLOWED = "ALLOWED"
    REQUIRED = "REQUIRED"
    FORBIDDEN = "FORBIDDEN"
    UNDECLARED = "UNDECLARED"


class ModelReleaseDecisionRefusal(ValueError):
    """A bounded decision input that cannot produce a complete receipt."""

    def __init__(self, issue: str, **evidence: CanonicalValue) -> None:
        """Capture one stable issue token and canonical refusal evidence."""
        self.issue = issue
        self.evidence: dict[str, CanonicalValue] = {"issue": issue, **evidence}
        super().__init__(issue)


@dataclass(frozen=True, slots=True)
class ObservedChange:
    """One ordered semantic, public-field, or fail-closed residual change."""

    selector: PolicySelector
    source: str
    before_sha256: str | None
    after_sha256: str | None
    before_value: CanonicalValue
    after_value: CanonicalValue
    details: dict[str, CanonicalValue]
    force_undeclared: bool = False

    def sort_key(self) -> tuple[int, bytes, int, int]:
        """Return the frozen source-order-independent witness ordering key."""
        return (
            _OBJECT_TYPE_RANK[self.selector.object_type],
            self.selector.object_name.encode("utf-8", errors="strict"),
            _FIELD_RANK[self.selector.object_type][self.selector.field],
            _CHANGE_KIND_RANK[self.selector.change_kind],
        )


@dataclass(frozen=True, slots=True)
class ClassifiedChange:
    """One observed change plus its exact policy disposition."""

    observed: ObservedChange
    classification: ChangeClassification
    rule_id: str | None

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return the complete decision-bearing change row."""
        change = self.observed
        return {
            "selector": change.selector.to_primitive(),
            "source": change.source,
            "classification": self.classification.value,
            "rule_id": self.rule_id,
            "before_sha256": change.before_sha256,
            "after_sha256": change.after_sha256,
            "before_value": change.before_value,
            "after_value": change.after_value,
            "details": dict(change.details),
        }


@dataclass(frozen=True, slots=True)
class ModelReleaseDecision:
    """The complete ordered change set and status derived from one admitted policy."""

    status: ModelReleaseStatus
    changes: tuple[ClassifiedChange, ...]
    satisfied_required_rule_ids: tuple[str, ...]
    missing_required_rules: tuple[PolicyRule, ...]

    @property
    def first_unexpected(self) -> ClassifiedChange | None:
        """Return the first forbidden or undeclared change in canonical order."""
        return next(
            (
                change
                for change in self.changes
                if change.classification
                in {ChangeClassification.FORBIDDEN, ChangeClassification.UNDECLARED}
            ),
            None,
        )

    @property
    def first_missing_required(self) -> PolicyRule | None:
        """Return the first unsatisfied REQUIRE rule in canonical policy order."""
        return self.missing_required_rules[0] if self.missing_required_rules else None


def decide_model_release(
    *,
    policy: ModelReleasePolicy,
    baseline: CompiledModelSnapshot,
    candidate: CompiledModelSnapshot,
    baseline_mjb_sha256: str,
    candidate_mjb_sha256: str,
) -> ModelReleaseDecision:
    """Build every compiled change, classify it, and apply the frozen outcome precedence."""
    observed = _observed_changes(
        policy,
        baseline,
        candidate,
        baseline_mjb_sha256,
        candidate_mjb_sha256,
    )
    if len(observed) > MAX_COMPILED_CHANGES:
        raise ModelReleaseDecisionRefusal(
            "compiled_change_budget_exceeded",
            observed_change_count=len(observed),
            maximum_change_count=MAX_COMPILED_CHANGES,
        )
    rule_index = _build_rule_index(policy.rules)
    classified = tuple(_classify_change(change, rule_index) for change in observed)
    satisfied = tuple(
        rule.id
        for rule in policy.rules
        if rule.effect is PolicyEffect.REQUIRE
        and any(change.rule_id == rule.id for change in classified)
    )
    satisfied_set = set(satisfied)
    missing = tuple(
        rule
        for rule in policy.rules
        if rule.effect is PolicyEffect.REQUIRE and rule.id not in satisfied_set
    )
    has_forbidden = any(
        change.classification is ChangeClassification.FORBIDDEN for change in classified
    )
    has_undeclared = any(
        change.classification is ChangeClassification.UNDECLARED for change in classified
    )
    if has_forbidden or missing:
        status = ModelReleaseStatus.OUTSIDE_DECLARED_POLICY
    elif has_undeclared:
        status = ModelReleaseStatus.REVIEW_REQUIRED
    elif not classified:
        status = ModelReleaseStatus.NO_COMPILED_CHANGE
    else:
        status = ModelReleaseStatus.WITHIN_DECLARED_POLICY
    return ModelReleaseDecision(status, classified, satisfied, missing)


def _observed_changes(
    policy: ModelReleasePolicy,
    baseline: CompiledModelSnapshot,
    candidate: CompiledModelSnapshot,
    baseline_mjb_sha256: str,
    candidate_mjb_sha256: str,
) -> tuple[ObservedChange, ...]:
    """Return semantic changes, complete public-field changes, and necessary residuals."""
    changes = [*_semantic_changes(baseline, candidate), *_public_field_changes(baseline, candidate)]
    artifact_differs = baseline_mjb_sha256 != candidate_mjb_sha256
    residual_reasons: list[str] = []
    if artifact_differs and policy.candidate_compiled_sha256 is None:
        residual_reasons.append("candidate_compiled_subject_unbound")
    elif (
        policy.candidate_compiled_sha256 is not None
        and policy.candidate_compiled_sha256 != candidate_mjb_sha256
    ):
        residual_reasons.append("candidate_compiled_subject_mismatch")
    if artifact_differs and not changes:
        residual_reasons.append("no_public_or_semantic_field_difference_identified")
    if artifact_differs and (baseline.coverage_issues or candidate.coverage_issues):
        residual_reasons.append("semantic_name_coverage_incomplete")
    if artifact_differs and any(
        change.source == "COMPILED_PUBLIC_FIELD"
        and change.selector.object_name in COMPILED_NAME_IDENTITY_MAPPING_FIELDS
        for change in changes
    ):
        residual_reasons.append("compiled_name_identity_mapping_changed")
    if residual_reasons:
        expected = policy.candidate_compiled_sha256
        changes.append(
            ObservedChange(
                PolicySelector(
                    PolicyObjectType.OPAQUE,
                    "complete_mjb",
                    "compiled_artifact",
                    ChangeKind.MODIFY,
                ),
                "OPAQUE_ARTIFACT_RESIDUAL",
                expected
                if "candidate_compiled_subject_mismatch" in residual_reasons
                else baseline_mjb_sha256,
                candidate_mjb_sha256,
                None,
                None,
                {
                    "reasons": cast("list[CanonicalValue]", residual_reasons),
                    "policy_candidate_compiled_sha256": expected,
                    "baseline_coverage_issues": cast(
                        "list[CanonicalValue]", list(baseline.coverage_issues)
                    ),
                    "candidate_coverage_issues": cast(
                        "list[CanonicalValue]", list(candidate.coverage_issues)
                    ),
                },
                force_undeclared=True,
            )
        )
    return tuple(sorted(changes, key=ObservedChange.sort_key))


def _semantic_changes(
    baseline: CompiledModelSnapshot, candidate: CompiledModelSnapshot
) -> list[ObservedChange]:
    """Diff named semantic objects; additions/removals collapse to one presence row."""
    changes: list[ObservedChange] = []
    keys = sorted(
        set(baseline.semantic_objects) | set(candidate.semantic_objects),
        key=lambda key: (_OBJECT_TYPE_RANK[key[0]], key[1].encode("utf-8", errors="strict")),
    )
    for key in keys:
        left = baseline.semantic_objects.get(key)
        right = candidate.semantic_objects.get(key)
        if left is None:
            assert right is not None
            changes.append(_presence_change(right, ChangeKind.ADD, None, right.object_sha256))
            continue
        if right is None:
            changes.append(_presence_change(left, ChangeKind.REMOVE, left.object_sha256, None))
            continue
        for field in sorted(
            set(left.fields) | set(right.fields),
            key=lambda name: _FIELD_RANK[left.object_type][name],
        ):
            before_value = left.fields.get(field)
            after_value = right.fields.get(field)
            before_sha256 = _value_sha256(before_value)
            after_sha256 = _value_sha256(after_value)
            if before_sha256 == after_sha256:
                continue
            changes.append(
                ObservedChange(
                    PolicySelector(
                        left.object_type,
                        left.object_name,
                        field,
                        ChangeKind.MODIFY,
                    ),
                    "SEMANTIC_OBJECT",
                    before_sha256,
                    after_sha256,
                    before_value,
                    after_value,
                    {},
                )
            )
    return changes


def _presence_change(
    obj: SemanticObjectFact,
    kind: ChangeKind,
    before_sha256: str | None,
    after_sha256: str | None,
) -> ObservedChange:
    """Represent one object addition/removal without fabricating field-by-field pairing."""
    value = cast("CanonicalValue", dict(obj.fields))
    return ObservedChange(
        PolicySelector(obj.object_type, obj.object_name, "presence", kind),
        "SEMANTIC_OBJECT",
        before_sha256,
        after_sha256,
        value if kind is ChangeKind.REMOVE else None,
        value if kind is ChangeKind.ADD else None,
        {},
    )


def _public_field_changes(
    baseline: CompiledModelSnapshot, candidate: CompiledModelSnapshot
) -> list[ObservedChange]:
    """List every changed field in the frozen complete public-field registry."""
    changes: list[ObservedChange] = []
    paths = sorted(set(baseline.public_fields) | set(candidate.public_fields))
    for path in paths:
        left = baseline.public_fields.get(path)
        right = candidate.public_fields.get(path)
        if left == right:
            continue
        changes.append(
            ObservedChange(
                PolicySelector(
                    PolicyObjectType.COMPILED_FIELD,
                    path,
                    "value",
                    _public_change_kind(left, right),
                ),
                "COMPILED_PUBLIC_FIELD",
                None if left is None else left.value_sha256,
                None if right is None else right.value_sha256,
                None,
                None,
                {
                    "baseline": _field_detail(left),
                    "candidate": _field_detail(right),
                },
            )
        )
    return changes


def _field_detail(value: PublicFieldFact | None) -> CanonicalValue:
    """Return field metadata or null for an absent registry field."""
    return None if value is None else value.detail_primitive()


def _public_change_kind(
    baseline: PublicFieldFact | None, candidate: PublicFieldFact | None
) -> ChangeKind:
    """Classify public registry membership changes, normally a value modification."""
    if baseline is None:
        return ChangeKind.ADD
    if candidate is None:
        return ChangeKind.REMOVE
    return ChangeKind.MODIFY


def _value_sha256(value: CanonicalValue) -> str:
    """Return an exact canonical identity, including the JSON null value."""
    return canonical_sha256(value)


RuleKey = tuple[PolicyObjectType, str, str, ChangeKind]


def _build_rule_index(rules: tuple[PolicyRule, ...]) -> dict[RuleKey, PolicyRule]:
    """Index the admitted nonoverlapping selector set for bounded classification."""
    return {
        (
            rule.selector.object_type,
            rule.selector.object_name,
            rule.selector.field,
            rule.selector.change_kind,
        ): rule
        for rule in rules
    }


def _classify_change(
    change: ObservedChange, rule_index: dict[RuleKey, PolicyRule]
) -> ClassifiedChange:
    """Apply the at-most-one admitted selector and its optional exact value constraints."""
    if change.force_undeclared:
        return ClassifiedChange(change, ChangeClassification.UNDECLARED, None)
    selector = change.selector
    shape = (selector.object_type, selector.field, selector.change_kind)
    rule = rule_index.get((shape[0], selector.object_name, shape[1], shape[2]))
    if rule is None:
        rule = rule_index.get((shape[0], "*", shape[1], shape[2]))
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


__all__ = [
    "MAX_COMPILED_CHANGES",
    "ChangeClassification",
    "ClassifiedChange",
    "ModelReleaseDecisionRefusal",
    "ModelReleaseDecision",
    "ObservedChange",
    "decide_model_release",
]
