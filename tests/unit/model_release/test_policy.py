"""Unit contract for bounded, immutable model-release policy admission."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from metrifid._json_admission import JsonAdmissionError
from metrifid.json_values import canonical_sha256
from metrifid.model_release._policy import (
    MODEL_RELEASE_POLICY_MAX_BYTES,
    MODEL_RELEASE_POLICY_MAX_RULES,
    MODEL_RELEASE_POLICY_SCHEMA,
    ChangeKind,
    ModelReleasePolicy,
    PolicyEffect,
    PolicyObjectType,
    PolicyRule,
    PolicySelector,
    load_model_release_policy,
    parse_model_release_policy,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _selector(
    *,
    object_type: str = "body",
    object_name: str = "arm",
    field: str = "mass",
    change_kind: str = "MODIFY",
) -> dict[str, object]:
    """Build one selector primitive for focused mutation by a test."""
    return {
        "object_type": object_type,
        "object_name": object_name,
        "field": field,
        "change_kind": change_kind,
    }


def _rule(
    rule_id: str = "mass-change",
    *,
    effect: str = "ALLOW",
    selector: dict[str, object] | None = None,
    before_sha256: str | None = None,
    after_sha256: str | None = None,
) -> dict[str, object]:
    """Build one rule primitive for focused mutation by a test."""
    return {
        "id": rule_id,
        "effect": effect,
        "selector": selector or _selector(),
        "before_sha256": before_sha256,
        "after_sha256": after_sha256,
    }


def _policy(
    rules: list[dict[str, object]] | None = None,
    *,
    candidate_sha256: str | None = None,
) -> dict[str, object]:
    """Build one complete policy primitive."""
    return {
        "schema": MODEL_RELEASE_POLICY_SCHEMA,
        "schema_version": 1,
        "baseline_compiled_sha256": SHA_A,
        "candidate_compiled_sha256": candidate_sha256,
        "rules": [_rule()] if rules is None else rules,
    }


def _encode(value: object, *, sort_keys: bool = False, indent: int | None = None) -> bytes:
    """Encode test input while leaving production canonicalization under test."""
    return json.dumps(value, sort_keys=sort_keys, indent=indent, ensure_ascii=False).encode()


def test_parse_canonicalizes_rules_and_carries_raw_and_semantic_hashes() -> None:
    """Make source order affect only raw identity, never admitted semantic identity."""
    rules = [
        _rule(
            "joint-z",
            selector=_selector(
                object_type="joint", object_name="z", field="range", change_kind="MODIFY"
            ),
        ),
        _rule("body-b", selector=_selector(object_name="b")),
        _rule("body-a", selector=_selector(object_name="a")),
    ]
    compact = _encode(_policy(rules))
    reordered = _encode(_policy(list(reversed(rules))), sort_keys=True, indent=2)

    first = parse_model_release_policy(compact)
    second = parse_model_release_policy(reordered)

    assert isinstance(first, ModelReleasePolicy)
    assert [rule.id for rule in first.rules] == ["body-a", "body-b", "joint-z"]
    assert first.rules == second.rules
    assert first.raw_sha256 == hashlib.sha256(compact).hexdigest()
    assert second.raw_sha256 == hashlib.sha256(reordered).hexdigest()
    assert first.raw_sha256 != second.raw_sha256
    assert first.semantic_sha256 == second.semantic_sha256
    assert first.semantic_sha256 == canonical_sha256(first.to_primitive())
    assert first.candidate_compiled_sha256 is None


def test_policy_value_graph_is_frozen() -> None:
    """Expose only frozen dataclasses and a tuple from an admitted policy."""
    policy = parse_model_release_policy(_encode(_policy()))
    assert type(policy.rules) is tuple
    with pytest.raises(FrozenInstanceError):
        policy.raw_sha256 = SHA_B  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        policy.rules[0].id = "different"  # type: ignore[misc]


def test_load_reads_the_exact_regular_file_bytes(tmp_path: Path) -> None:
    """Load through the bounded no-follow file admission path."""
    raw = _encode(_policy(candidate_sha256=SHA_B), indent=2)
    path = tmp_path / "policy.json"
    path.write_bytes(raw)
    policy = load_model_release_policy(path)
    assert policy.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert policy.candidate_compiled_sha256 == SHA_B


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema":"metrifid.model_release_policy","schema":"duplicate"}',
        b'{"schema":1.5}',
        b'{"schema":1e3}',
    ],
)
def test_duplicate_keys_and_raw_floats_are_refused(raw: bytes) -> None:
    """Delegate strict JSON syntax, including duplicate names and floats, to admission."""
    with pytest.raises(JsonAdmissionError):
        parse_model_release_policy(raw)


@pytest.mark.parametrize("level", ["root", "rule", "selector"])
def test_unknown_fields_are_refused_at_every_schema_level(level: str) -> None:
    """Keep the policy language closed at root, rule, and selector objects."""
    value = _policy()
    if level == "root":
        value["unknown"] = True
    elif level == "rule":
        rules = value["rules"]
        assert isinstance(rules, list)
        rules[0]["unknown"] = True
    else:
        rules = value["rules"]
        assert isinstance(rules, list)
        rules[0]["selector"]["unknown"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        parse_model_release_policy(_encode(value))


@pytest.mark.parametrize("level", ["root", "rule", "selector"])
def test_missing_fields_are_refused_at_every_schema_level(level: str) -> None:
    """Require the full frozen shape at root, rule, and selector objects."""
    value = _policy()
    if level == "root":
        del value["candidate_compiled_sha256"]
    elif level == "rule":
        rules = value["rules"]
        assert isinstance(rules, list)
        del rules[0]["after_sha256"]
    else:
        rules = value["rules"]
        assert isinstance(rules, list)
        del rules[0]["selector"]["change_kind"]
    with pytest.raises(ValueError, match="missing fields"):
        parse_model_release_policy(_encode(value))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "metrifid.not_the_policy"),
        ("schema_version", 2),
        ("schema_version", True),
        ("baseline_compiled_sha256", None),
        ("baseline_compiled_sha256", "A" * 64),
        ("candidate_compiled_sha256", "short"),
    ],
)
def test_root_discriminators_and_hashes_are_strict(field: str, value: object) -> None:
    """Refuse invalid schema discriminators and compiled-identity bindings."""
    document = _policy()
    document[field] = value
    with pytest.raises((TypeError, ValueError)):
        parse_model_release_policy(_encode(document))


@pytest.mark.parametrize(
    ("object_type", "field"),
    [
        ("body", "presence"),
        ("body", "parent"),
        ("body", "mass"),
        ("body", "inertia"),
        ("joint", "presence"),
        ("joint", "body"),
        ("joint", "type"),
        ("joint", "limited"),
        ("joint", "range"),
        ("geom", "presence"),
        ("geom", "body"),
        ("geom", "mesh"),
        ("mesh", "presence"),
        ("mesh", "compiled_geometry_sha256"),
        ("actuator", "presence"),
        ("actuator", "transmission"),
        ("actuator", "targets"),
        ("compiled_field", "value"),
        ("opaque", "compiled_artifact"),
    ],
)
def test_closed_object_type_field_pairs_are_admitted(object_type: str, field: str) -> None:
    """Pin every selector field exposed by the policy language."""
    selector = _selector(object_type=object_type, field=field)
    policy = parse_model_release_policy(_encode(_policy([_rule(selector=selector)])))
    assert policy.rules[0].selector.field == field


@pytest.mark.parametrize(
    "selector",
    [
        _selector(object_type="site"),
        _selector(change_kind="RENAME"),
        _selector(object_type="body", field="range"),
        _selector(object_type="opaque", field="value"),
        _selector(object_name="arm*"),
        _selector(object_name="**"),
        _selector(object_name=""),
        _selector(object_name="nul\x00name"),
    ],
)
def test_unknown_or_ambiguous_selectors_are_refused(selector: dict[str, object]) -> None:
    """Reject open vocabulary, incompatible fields, and partial wildcard syntax."""
    with pytest.raises((TypeError, ValueError)):
        parse_model_release_policy(_encode(_policy([_rule(selector=selector)])))


@pytest.mark.parametrize("effect", ["allow", "DENY", "", 1, None])
def test_effect_vocabulary_is_closed(effect: object) -> None:
    """Accept no policy effect outside uppercase ALLOW, REQUIRE, and FORBID."""
    rule = _rule()
    rule["effect"] = effect
    with pytest.raises((TypeError, ValueError)):
        parse_model_release_policy(_encode(_policy([rule])))


@pytest.mark.parametrize("effect", ["ALLOW", "FORBID"])
def test_allow_and_forbid_may_use_null_as_any_digest(effect: str) -> None:
    """Treat null digest constraints as ANY only for ALLOW and FORBID."""
    policy = parse_model_release_policy(_encode(_policy([_rule(effect=effect)])))
    assert policy.rules[0].before_sha256 is None
    assert policy.rules[0].after_sha256 is None


def test_modify_require_needs_an_exact_name_and_both_hashes() -> None:
    """Make a MODIFY requirement identify one exact expected transition."""
    valid = _rule(effect="REQUIRE", before_sha256=SHA_B, after_sha256=SHA_C)
    parsed = parse_model_release_policy(_encode(_policy([valid])))
    assert parsed.rules[0].effect is PolicyEffect.REQUIRE
    for invalid in (
        _rule(effect="REQUIRE", before_sha256=None, after_sha256=SHA_C),
        _rule(effect="REQUIRE", before_sha256=SHA_B, after_sha256=None),
        _rule(
            effect="REQUIRE",
            selector=_selector(object_name="*"),
            before_sha256=SHA_B,
            after_sha256=SHA_C,
        ),
    ):
        with pytest.raises(ValueError, match="REQUIRE"):
            parse_model_release_policy(_encode(_policy([invalid])))


@pytest.mark.parametrize(
    ("change_kind", "before_sha256", "after_sha256"),
    [
        ("ADD", None, SHA_C),
        ("REMOVE", SHA_B, None),
    ],
)
def test_structural_require_binds_the_present_side_and_exact_absence(
    change_kind: str,
    before_sha256: str | None,
    after_sha256: str | None,
) -> None:
    """Admit exact ADD and REMOVE requirements without inventing an absent value hash."""
    rule = _rule(
        effect="REQUIRE",
        selector=_selector(change_kind=change_kind),
        before_sha256=before_sha256,
        after_sha256=after_sha256,
    )
    parsed = parse_model_release_policy(_encode(_policy([rule])))
    assert parsed.rules[0].before_sha256 == before_sha256
    assert parsed.rules[0].after_sha256 == after_sha256


@pytest.mark.parametrize(
    ("change_kind", "before_sha256", "after_sha256"),
    [
        ("ADD", SHA_B, SHA_C),
        ("ADD", None, None),
        ("REMOVE", SHA_B, SHA_C),
        ("REMOVE", None, None),
    ],
)
def test_structural_require_refuses_inexact_absence_or_present_value(
    change_kind: str,
    before_sha256: str | None,
    after_sha256: str | None,
) -> None:
    """Prevent an ADD or REMOVE requirement from weakening either side's meaning."""
    rule = _rule(
        effect="REQUIRE",
        selector=_selector(change_kind=change_kind),
        before_sha256=before_sha256,
        after_sha256=after_sha256,
    )
    with pytest.raises(ValueError, match="REQUIRE"):
        parse_model_release_policy(_encode(_policy([rule])))


def test_duplicate_ids_are_refused_even_for_distinct_selectors() -> None:
    """Keep rule references unambiguous independent of selector differences."""
    rules = [
        _rule("same", selector=_selector(object_name="a")),
        _rule("same", selector=_selector(object_name="b")),
    ]
    with pytest.raises(ValueError, match="duplicate policy rule id"):
        parse_model_release_policy(_encode(_policy(rules)))


def test_duplicate_selectors_are_refused_even_with_different_constraints() -> None:
    """Allow no order- or effect-dependent resolution for an identical selector."""
    rules = [
        _rule("one", effect="ALLOW"),
        _rule("two", effect="FORBID", before_sha256=SHA_B, after_sha256=SHA_C),
    ]
    with pytest.raises(ValueError, match="duplicate policy selector"):
        parse_model_release_policy(_encode(_policy(rules)))


@pytest.mark.parametrize("reverse", [False, True])
def test_potential_wildcard_overlap_is_refused_independent_of_effect_and_order(
    reverse: bool,
) -> None:
    """Reject any wildcard/exact pair capable of selecting the same change."""
    rules = [
        _rule("wild", effect="FORBID", selector=_selector(object_name="*")),
        _rule("exact", effect="ALLOW", selector=_selector(object_name="arm")),
    ]
    if reverse:
        rules.reverse()
    with pytest.raises(ValueError, match="potentially overlapping"):
        parse_model_release_policy(_encode(_policy(rules)))


def test_nonoverlapping_wildcards_are_admitted() -> None:
    """Allow wildcards when field or change kind makes the selected changes disjoint."""
    rules = [
        _rule("mass", selector=_selector(object_name="*", field="mass")),
        _rule("parent", selector=_selector(object_name="*", field="parent")),
        _rule(
            "added-mass",
            selector=_selector(object_name="*", field="mass", change_kind="ADD"),
        ),
    ]
    assert len(parse_model_release_policy(_encode(_policy(rules))).rules) == 3


def test_rule_count_limit_is_inclusive_and_boundary_plus_one_refuses() -> None:
    """Admit exactly 4096 rules and refuse the next one."""
    rules = [
        _rule(f"r{index}", selector=_selector(object_name=f"body-{index}"))
        for index in range(MODEL_RELEASE_POLICY_MAX_RULES)
    ]
    assert len(parse_model_release_policy(_encode(_policy(rules))).rules) == 4096
    rules.append(_rule("extra", selector=_selector(object_name="extra")))
    with pytest.raises(ValueError, match="maximum of 4096"):
        parse_model_release_policy(_encode(_policy(rules)))


def test_id_and_name_limits_measure_strict_utf8_bytes() -> None:
    """Admit 256 encoded bytes and refuse 257 or more for IDs and names."""
    exact = "é" * 128
    policy = parse_model_release_policy(
        _encode(_policy([_rule(exact, selector=_selector(object_name=exact))]))
    )
    assert policy.rules[0].id == exact
    for rule in (
        _rule("é" * 129),
        _rule("id", selector=_selector(object_name="é" * 129)),
    ):
        with pytest.raises((JsonAdmissionError, ValueError)):
            parse_model_release_policy(_encode(_policy([rule])))


def test_whole_document_limit_is_inclusive() -> None:
    """Admit exactly one MiB and refuse the next byte before structural parsing."""
    raw = _encode(_policy([]))
    at_limit = raw + b" " * (MODEL_RELEASE_POLICY_MAX_BYTES - len(raw))
    assert parse_model_release_policy(at_limit).rules == ()
    padded = at_limit + b" "
    with pytest.raises(JsonAdmissionError, match="exceeds"):
        parse_model_release_policy(padded)


def test_direct_types_enforce_enum_and_canonical_policy_invariants() -> None:
    """Keep direct SDK construction as strict as JSON-backed construction."""
    selector = PolicySelector(PolicyObjectType.BODY, "arm", "mass", ChangeKind.MODIFY)
    rule = PolicyRule("change", PolicyEffect.ALLOW, selector, None, None)
    primitive = _policy([_rule("change")])
    semantic = canonical_sha256(primitive)  # type: ignore[arg-type]
    policy = ModelReleasePolicy(
        MODEL_RELEASE_POLICY_SCHEMA,
        1,
        SHA_A,
        None,
        (rule,),
        SHA_B,
        semantic,
    )
    assert policy.rules == (rule,)
    with pytest.raises(ValueError, match="semantic_sha256"):
        ModelReleasePolicy(
            MODEL_RELEASE_POLICY_SCHEMA,
            1,
            SHA_A,
            None,
            (rule,),
            SHA_B,
            SHA_C,
        )


def test_parse_requires_exact_bytes_input() -> None:
    """Make text-to-byte encoding an explicit caller decision."""
    with pytest.raises(TypeError, match="must be bytes"):
        parse_model_release_policy("{}")  # type: ignore[arg-type]
