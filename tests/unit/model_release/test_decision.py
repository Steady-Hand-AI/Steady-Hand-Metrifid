"""Pure status, policy-decision, ordering, and fail-closed residual scenarios."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import pytest

from metrifid.json_values import Binary64, CanonicalValue, canonical_sha256
from metrifid.model_release import _decision as decision_module
from metrifid.model_release._decision import (
    ChangeClassification,
    ModelReleaseDecision,
    decide_model_release,
)
from metrifid.model_release._policy import (
    ChangeKind,
    ModelReleasePolicy,
    PolicyEffect,
    PolicyObjectType,
    parse_model_release_policy,
)
from metrifid.model_release._snapshot import (
    PUBLIC_FIELD_REGISTRY_SHA256,
    CompiledModelSnapshot,
    PublicFieldFact,
    SemanticObjectFact,
)
from metrifid.model_release._status import (
    MODEL_RELEASE_COMPLETED_EXIT_CODES,
    ModelReleaseStatus,
    model_release_exit_code,
)

BASELINE_MJB = "1" * 64
CANDIDATE_MJB = "2" * 64
OTHER_MJB = "3" * 64
PUBLIC_BEFORE = "4" * 64
PUBLIC_AFTER = "5" * 64
WRONG_VALUE = "f" * 64

_CHANGE_ROW_FIELDS = {
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


def _binary64(value: float) -> dict[str, CanonicalValue]:
    """Represent a synthetic semantic number with its exact binary64 bits."""
    return Binary64.from_float(value).to_primitive()


def _body(
    name: str,
    mass: CanonicalValue,
    *,
    parent: str = "world",
    reverse_fields: bool = False,
) -> SemanticObjectFact:
    """Build one complete synthetic body fact, optionally reversing field insertion order."""
    fields: dict[str, CanonicalValue] = {
        "parent": parent,
        "mass": mass,
        "inertia": [_binary64(1.0), _binary64(1.0), _binary64(1.0)],
    }
    if reverse_fields:
        fields = dict(reversed(tuple(fields.items())))
    return SemanticObjectFact(
        PolicyObjectType.BODY,
        name,
        MappingProxyType(fields),
    )


def _joint(name: str, range_value: CanonicalValue) -> SemanticObjectFact:
    """Build one complete synthetic joint fact."""
    fields: dict[str, CanonicalValue] = {
        "body": "alpha",
        "type": 3,
        "limited": True,
        "range": range_value,
    }
    return SemanticObjectFact(
        PolicyObjectType.JOINT,
        name,
        MappingProxyType(fields),
    )


def _geom(name: str, mesh: str | None) -> SemanticObjectFact:
    """Build one synthetic geom fact whose mesh reference may be semantic null."""
    fields: dict[str, CanonicalValue] = {"body": "alpha", "mesh": mesh}
    return SemanticObjectFact(
        PolicyObjectType.GEOM,
        name,
        MappingProxyType(fields),
    )


def _public_field(path: str, value_sha256: str) -> PublicFieldFact:
    """Build one synthetic complete-public-registry field fact."""
    return PublicFieldFact(path, "ndarray", "<f8", (1,), value_sha256)


def _snapshot(
    objects: Sequence[SemanticObjectFact] = (),
    *,
    public_fields: Sequence[PublicFieldFact] = (),
    coverage_issues: tuple[str, ...] = (),
) -> CompiledModelSnapshot:
    """Build an immutable snapshot without importing MuJoCo or NumPy."""
    semantic = {(fact.object_type, fact.object_name): fact for fact in objects}
    public = {fact.path: fact for fact in public_fields}
    return CompiledModelSnapshot(
        MappingProxyType(public),
        MappingProxyType(semantic),
        coverage_issues,
        PUBLIC_FIELD_REGISTRY_SHA256,
    )


def _rule(
    rule_id: str,
    effect: PolicyEffect,
    object_type: PolicyObjectType,
    object_name: str,
    field: str,
    *,
    change_kind: ChangeKind = ChangeKind.MODIFY,
    before_sha256: str | None = None,
    after_sha256: str | None = None,
) -> dict[str, object]:
    """Build one strict policy-rule primitive."""
    return {
        "id": rule_id,
        "effect": effect.value,
        "selector": {
            "object_type": object_type.value,
            "object_name": object_name,
            "field": field,
            "change_kind": change_kind.value,
        },
        "before_sha256": before_sha256,
        "after_sha256": after_sha256,
    }


def _policy(
    rules: Sequence[Mapping[str, object]] = (),
    *,
    candidate_sha256: str | None = CANDIDATE_MJB,
) -> ModelReleasePolicy:
    """Admit one synthetic policy through the real strict parser."""
    primitive = {
        "schema": "metrifid.model_release_policy",
        "schema_version": 1,
        "baseline_compiled_sha256": BASELINE_MJB,
        "candidate_compiled_sha256": candidate_sha256,
        "rules": [dict(rule) for rule in rules],
    }
    return parse_model_release_policy(
        json.dumps(primitive, ensure_ascii=False, separators=(",", ":")).encode()
    )


def _decide(
    policy: ModelReleasePolicy,
    baseline: CompiledModelSnapshot,
    candidate: CompiledModelSnapshot,
    *,
    baseline_mjb_sha256: str = BASELINE_MJB,
    candidate_mjb_sha256: str = CANDIDATE_MJB,
) -> ModelReleaseDecision:
    """Invoke the pure decision boundary with synthetic artifact identities."""
    return decide_model_release(
        policy=policy,
        baseline=baseline,
        candidate=candidate,
        baseline_mjb_sha256=baseline_mjb_sha256,
        candidate_mjb_sha256=candidate_mjb_sha256,
    )


def _assert_complete_rows(decision: ModelReleaseDecision, count: int) -> None:
    """Require every observed change to survive as one complete decision-bearing row."""
    assert len(decision.changes) == count
    for change in decision.changes:
        assert set(change.to_primitive()) == _CHANGE_ROW_FIELDS


def test_exact_completed_status_vocabulary_and_exit_mapping() -> None:
    """Pin all four completed meanings and their exact zero-or-forty transport."""
    expected = {
        ModelReleaseStatus.NO_COMPILED_CHANGE: 0,
        ModelReleaseStatus.WITHIN_DECLARED_POLICY: 0,
        ModelReleaseStatus.REVIEW_REQUIRED: 40,
        ModelReleaseStatus.OUTSIDE_DECLARED_POLICY: 40,
    }
    assert set(ModelReleaseStatus) == set(expected)
    assert dict(MODEL_RELEASE_COMPLETED_EXIT_CODES) == expected
    assert {status: model_release_exit_code(status) for status in expected} == expected


def test_compiled_change_budget_is_a_typed_refusal_not_an_internal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the refusal boundary when a valid comparison exceeds decision capacity."""
    monkeypatch.setattr(decision_module, "MAX_COMPILED_CHANGES", 0)
    with pytest.raises(decision_module.ModelReleaseDecisionRefusal) as captured:
        _decide(
            _policy(),
            _snapshot((_body("arm", _binary64(1.0)),)),
            _snapshot((_body("arm", _binary64(2.0)),)),
        )
    assert captured.value.evidence == {
        "issue": "compiled_change_budget_exceeded",
        "observed_change_count": 1,
        "maximum_change_count": 0,
    }


def test_identical_snapshots_and_artifacts_have_no_compiled_change() -> None:
    """Return NO_COMPILED_CHANGE only when no requirement is missing."""
    body = _body("arm", _binary64(1.0))
    snapshot = _snapshot((body,))
    decision = _decide(
        _policy(candidate_sha256=BASELINE_MJB),
        snapshot,
        snapshot,
        candidate_mjb_sha256=BASELINE_MJB,
    )
    assert decision.status is ModelReleaseStatus.NO_COMPILED_CHANGE
    assert model_release_exit_code(decision.status) == 0
    assert decision.changes == ()
    assert decision.first_unexpected is None
    assert decision.first_missing_required is None


def test_missing_requirement_outranks_an_empty_change_set() -> None:
    """Return OUTSIDE_DECLARED_POLICY instead of NO change when REQUIRE is absent."""
    mass = _binary64(1.0)
    snapshot = _snapshot((_body("arm", mass),))
    rule = _rule(
        "required-mass",
        PolicyEffect.REQUIRE,
        PolicyObjectType.BODY,
        "arm",
        "mass",
        before_sha256=canonical_sha256(mass),
        after_sha256=canonical_sha256(_binary64(2.0)),
    )
    decision = _decide(
        _policy((rule,), candidate_sha256=BASELINE_MJB),
        snapshot,
        snapshot,
        candidate_mjb_sha256=BASELINE_MJB,
    )
    assert decision.status is ModelReleaseStatus.OUTSIDE_DECLARED_POLICY
    assert model_release_exit_code(decision.status) == 40
    assert decision.changes == ()
    assert decision.satisfied_required_rule_ids == ()
    assert tuple(item.id for item in decision.missing_required_rules) == ("required-mass",)
    assert decision.first_missing_required is decision.missing_required_rules[0]


def test_exactly_allowed_change_is_within_declared_policy() -> None:
    """Classify one exact allowed semantic change and retain its complete evidence row."""
    before = _binary64(1.0)
    after = _binary64(2.0)
    rule = _rule(
        "allow-mass",
        PolicyEffect.ALLOW,
        PolicyObjectType.BODY,
        "arm",
        "mass",
        before_sha256=canonical_sha256(before),
        after_sha256=canonical_sha256(after),
    )
    decision = _decide(
        _policy((rule,)),
        _snapshot((_body("arm", before),)),
        _snapshot((_body("arm", after),)),
    )
    assert decision.status is ModelReleaseStatus.WITHIN_DECLARED_POLICY
    _assert_complete_rows(decision, 1)
    change = decision.changes[0]
    assert change.classification is ChangeClassification.ALLOWED
    assert change.rule_id == "allow-mass"
    assert change.observed.before_value == before
    assert change.observed.after_value == after
    assert decision.first_unexpected is None


def test_forbidden_change_is_outside_declared_policy() -> None:
    """Make one matching FORBID row the stable unexpected witness."""
    before = _binary64(1.0)
    after = _binary64(2.0)
    rule = _rule(
        "forbid-mass",
        PolicyEffect.FORBID,
        PolicyObjectType.BODY,
        "arm",
        "mass",
    )
    decision = _decide(
        _policy((rule,)),
        _snapshot((_body("arm", before),)),
        _snapshot((_body("arm", after),)),
    )
    assert decision.status is ModelReleaseStatus.OUTSIDE_DECLARED_POLICY
    _assert_complete_rows(decision, 1)
    assert decision.changes[0].classification is ChangeClassification.FORBIDDEN
    assert decision.first_unexpected is decision.changes[0]


def test_allowed_plus_undeclared_change_requires_review_and_keeps_both_rows() -> None:
    """Never let one allowed field conceal a second undeclared field."""
    before = _binary64(1.0)
    after = _binary64(2.0)
    rule = _rule(
        "allow-mass",
        PolicyEffect.ALLOW,
        PolicyObjectType.BODY,
        "arm",
        "mass",
    )
    decision = _decide(
        _policy((rule,)),
        _snapshot((_body("arm", before),)),
        _snapshot((_body("arm", after, parent="new-parent"),)),
    )
    assert decision.status is ModelReleaseStatus.REVIEW_REQUIRED
    _assert_complete_rows(decision, 2)
    assert [change.observed.selector.field for change in decision.changes] == ["parent", "mass"]
    assert [change.classification for change in decision.changes] == [
        ChangeClassification.UNDECLARED,
        ChangeClassification.ALLOWED,
    ]
    assert decision.first_unexpected is decision.changes[0]


def test_exact_required_change_is_present_and_satisfied() -> None:
    """Classify an exact REQUIRE match and record its satisfaction once."""
    before = _binary64(1.0)
    after = _binary64(2.0)
    rule = _rule(
        "required-mass",
        PolicyEffect.REQUIRE,
        PolicyObjectType.BODY,
        "arm",
        "mass",
        before_sha256=canonical_sha256(before),
        after_sha256=canonical_sha256(after),
    )
    decision = _decide(
        _policy((rule,)),
        _snapshot((_body("arm", before),)),
        _snapshot((_body("arm", after),)),
    )
    assert decision.status is ModelReleaseStatus.WITHIN_DECLARED_POLICY
    _assert_complete_rows(decision, 1)
    assert decision.changes[0].classification is ChangeClassification.REQUIRED
    assert decision.satisfied_required_rule_ids == ("required-mass",)
    assert decision.missing_required_rules == ()


def test_exact_required_add_and_remove_are_present_and_satisfied() -> None:
    """Classify structural requirements without fabricating a digest for an absent side."""
    body = _body("arm", _binary64(1.0))
    cases = (
        (
            ChangeKind.ADD,
            _snapshot(),
            _snapshot((body,)),
            None,
            body.object_sha256,
        ),
        (
            ChangeKind.REMOVE,
            _snapshot((body,)),
            _snapshot(),
            body.object_sha256,
            None,
        ),
    )
    for kind, baseline, candidate, before_sha256, after_sha256 in cases:
        rule = _rule(
            f"required-{kind.value.lower()}",
            PolicyEffect.REQUIRE,
            PolicyObjectType.BODY,
            "arm",
            "presence",
            change_kind=kind,
            before_sha256=before_sha256,
            after_sha256=after_sha256,
        )
        decision = _decide(_policy((rule,)), baseline, candidate)
        assert decision.status is ModelReleaseStatus.WITHIN_DECLARED_POLICY
        _assert_complete_rows(decision, 1)
        assert decision.changes[0].classification is ChangeClassification.REQUIRED
        assert decision.satisfied_required_rule_ids == (rule["id"],)
        assert decision.missing_required_rules == ()


def test_wrong_required_after_hash_is_undeclared_and_required_is_missing() -> None:
    """Preserve both witnesses when a selector matches but its exact after value does not."""
    before = _binary64(1.0)
    after = _binary64(2.0)
    rule = _rule(
        "required-mass",
        PolicyEffect.REQUIRE,
        PolicyObjectType.BODY,
        "arm",
        "mass",
        before_sha256=canonical_sha256(before),
        after_sha256=WRONG_VALUE,
    )
    decision = _decide(
        _policy((rule,)),
        _snapshot((_body("arm", before),)),
        _snapshot((_body("arm", after),)),
    )
    assert decision.status is ModelReleaseStatus.OUTSIDE_DECLARED_POLICY
    _assert_complete_rows(decision, 1)
    assert decision.changes[0].classification is ChangeClassification.UNDECLARED
    assert decision.changes[0].rule_id is None
    assert decision.first_unexpected is decision.changes[0]
    assert decision.first_missing_required is decision.missing_required_rules[0]
    assert decision.first_missing_required.id == "required-mass"


def test_null_candidate_binding_forces_an_opaque_review_on_artifact_difference() -> None:
    """Keep discovery-mode artifact difference nonpositive even when typed change is allowed."""
    before = _binary64(1.0)
    after = _binary64(2.0)
    rule = _rule(
        "allow-mass",
        PolicyEffect.ALLOW,
        PolicyObjectType.BODY,
        "arm",
        "mass",
    )
    decision = _decide(
        _policy((rule,), candidate_sha256=None),
        _snapshot((_body("arm", before),)),
        _snapshot((_body("arm", after),)),
    )
    assert decision.status is ModelReleaseStatus.REVIEW_REQUIRED
    _assert_complete_rows(decision, 2)
    opaque = decision.changes[-1]
    assert opaque.observed.selector.object_type is PolicyObjectType.OPAQUE
    assert opaque.classification is ChangeClassification.UNDECLARED
    assert opaque.rule_id is None
    assert opaque.observed.details["reasons"] == ["candidate_compiled_subject_unbound"]
    assert decision.first_unexpected is opaque


def test_wrong_candidate_binding_forces_an_opaque_review() -> None:
    """Emit an opaque mismatch witness instead of accepting otherwise allowed typed changes."""
    before = _binary64(1.0)
    after = _binary64(2.0)
    rule = _rule(
        "allow-mass",
        PolicyEffect.ALLOW,
        PolicyObjectType.BODY,
        "arm",
        "mass",
    )
    decision = _decide(
        _policy((rule,), candidate_sha256=OTHER_MJB),
        _snapshot((_body("arm", before),)),
        _snapshot((_body("arm", after),)),
    )
    assert decision.status is ModelReleaseStatus.REVIEW_REQUIRED
    _assert_complete_rows(decision, 2)
    opaque = decision.changes[-1]
    assert opaque.observed.details["reasons"] == ["candidate_compiled_subject_mismatch"]
    assert opaque.observed.before_sha256 == OTHER_MJB
    assert opaque.observed.after_sha256 == CANDIDATE_MJB
    assert opaque.classification is ChangeClassification.UNDECLARED


def test_exact_candidate_binding_with_every_change_allowed_is_within_policy() -> None:
    """Return green only when every semantic and complete-public-field row is declared."""
    before = _binary64(1.0)
    after = _binary64(2.0)
    rules = (
        _rule(
            "allow-mass",
            PolicyEffect.ALLOW,
            PolicyObjectType.BODY,
            "arm",
            "mass",
        ),
        _rule(
            "allow-public",
            PolicyEffect.ALLOW,
            PolicyObjectType.COMPILED_FIELD,
            "body_mass",
            "value",
            before_sha256=PUBLIC_BEFORE,
            after_sha256=PUBLIC_AFTER,
        ),
    )
    decision = _decide(
        _policy(rules, candidate_sha256=CANDIDATE_MJB),
        _snapshot(
            (_body("arm", before),),
            public_fields=(_public_field("body_mass", PUBLIC_BEFORE),),
        ),
        _snapshot(
            (_body("arm", after),),
            public_fields=(_public_field("body_mass", PUBLIC_AFTER),),
        ),
    )
    assert decision.status is ModelReleaseStatus.WITHIN_DECLARED_POLICY
    _assert_complete_rows(decision, 2)
    assert all(change.classification is ChangeClassification.ALLOWED for change in decision.changes)
    assert all(
        change.observed.selector.object_type is not PolicyObjectType.OPAQUE
        for change in decision.changes
    )


def test_decision_order_and_first_witness_ignore_policy_and_input_permutations() -> None:
    """Order by typed semantic locus, never rule or mapping insertion order."""
    before_objects = (
        _body("zeta", _binary64(1.0)),
        _joint("hinge", [_binary64(-1.0), _binary64(1.0)]),
        _body("alpha", _binary64(1.0)),
    )
    after_objects = (
        _body("zeta", _binary64(3.0), reverse_fields=True),
        _joint("hinge", [_binary64(-2.0), _binary64(2.0)]),
        _body("alpha", _binary64(2.0), parent="zeta", reverse_fields=True),
    )
    rules = [
        _rule(
            "allow-public",
            PolicyEffect.ALLOW,
            PolicyObjectType.COMPILED_FIELD,
            "zz_field",
            "value",
        ),
        _rule(
            "allow-joint",
            PolicyEffect.ALLOW,
            PolicyObjectType.JOINT,
            "hinge",
            "range",
        ),
        _rule(
            "forbid-zeta",
            PolicyEffect.FORBID,
            PolicyObjectType.BODY,
            "zeta",
            "mass",
        ),
        _rule(
            "allow-alpha-mass",
            PolicyEffect.ALLOW,
            PolicyObjectType.BODY,
            "alpha",
            "mass",
        ),
    ]
    baseline_public = (_public_field("zz_field", PUBLIC_BEFORE),)
    candidate_public = (_public_field("zz_field", PUBLIC_AFTER),)

    first = _decide(
        _policy(rules),
        _snapshot(before_objects, public_fields=baseline_public),
        _snapshot(after_objects, public_fields=candidate_public),
    )
    second = _decide(
        _policy(tuple(reversed(rules))),
        _snapshot(tuple(reversed(before_objects)), public_fields=tuple(reversed(baseline_public))),
        _snapshot(tuple(reversed(after_objects)), public_fields=tuple(reversed(candidate_public))),
    )

    first_rows = tuple(change.to_primitive() for change in first.changes)
    second_rows = tuple(change.to_primitive() for change in second.changes)
    assert first_rows == second_rows
    assert first.status is second.status is ModelReleaseStatus.OUTSIDE_DECLARED_POLICY
    _assert_complete_rows(first, 5)
    assert [
        (
            change.observed.selector.object_type,
            change.observed.selector.object_name,
            change.observed.selector.field,
        )
        for change in first.changes
    ] == [
        (PolicyObjectType.BODY, "alpha", "parent"),
        (PolicyObjectType.BODY, "alpha", "mass"),
        (PolicyObjectType.BODY, "zeta", "mass"),
        (PolicyObjectType.JOINT, "hinge", "range"),
        (PolicyObjectType.COMPILED_FIELD, "zz_field", "value"),
    ]
    assert first.first_unexpected is first.changes[0]
    assert second.first_unexpected is second.changes[0]
    assert first.first_unexpected.classification is ChangeClassification.UNDECLARED


def test_semantic_coverage_issue_forces_an_opaque_review() -> None:
    """Refuse positive omission when either snapshot could not name every semantic object."""
    before = _binary64(1.0)
    after = _binary64(2.0)
    rule = _rule(
        "allow-mass",
        PolicyEffect.ALLOW,
        PolicyObjectType.BODY,
        "arm",
        "mass",
    )
    decision = _decide(
        _policy((rule,)),
        _snapshot((_body("arm", before),)),
        _snapshot(
            (_body("arm", after),),
            coverage_issues=("body:7:unnamed_or_overlong",),
        ),
    )
    assert decision.status is ModelReleaseStatus.REVIEW_REQUIRED
    _assert_complete_rows(decision, 2)
    opaque = decision.changes[-1]
    assert opaque.observed.selector.object_type is PolicyObjectType.OPAQUE
    assert opaque.observed.details["reasons"] == ["semantic_name_coverage_incomplete"]
    assert opaque.observed.details["candidate_coverage_issues"] == ["body:7:unnamed_or_overlong"]


def test_semantic_json_null_value_receives_a_real_digest() -> None:
    """Distinguish a semantic null value from the absence of a hash constraint."""
    after = "mesh-a"
    before_sha256 = canonical_sha256(None)
    after_sha256 = canonical_sha256(after)
    rule = _rule(
        "allow-mesh-reference",
        PolicyEffect.ALLOW,
        PolicyObjectType.GEOM,
        "visual",
        "mesh",
        before_sha256=before_sha256,
        after_sha256=after_sha256,
    )
    decision = _decide(
        _policy((rule,)),
        _snapshot((_geom("visual", None),)),
        _snapshot((_geom("visual", after),)),
    )
    assert decision.status is ModelReleaseStatus.WITHIN_DECLARED_POLICY
    _assert_complete_rows(decision, 1)
    observed = decision.changes[0].observed
    assert observed.before_value is None
    assert observed.before_sha256 == before_sha256
    assert observed.before_sha256 is not None
    assert observed.after_value == after
    assert observed.after_sha256 == after_sha256
