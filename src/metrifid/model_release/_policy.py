"""Bounded, fail-closed admission for model-release policy documents.

The policy language is deliberately small.  Rules select one typed compiled-model change and
declare it allowed, required, or forbidden.  Admission happens without importing MuJoCo or NumPy,
and every accepted policy has both an exact raw-byte identity and a canonical semantic identity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, TypeVar, cast

from .._json_admission import (
    JsonAdmissionLimits,
    bounded_strict_json_loads,
    read_bounded_regular_file,
)
from .._schema_primitives import _exact_int, _fields, _object, _optional_hash, _sequence
from ..json_values import CanonicalValue, canonical_sha256, require_sha256

__all__ = [
    "MODEL_RELEASE_POLICY_MAX_BYTES",
    "MODEL_RELEASE_POLICY_MAX_RULES",
    "MODEL_RELEASE_POLICY_SCHEMA",
    "MODEL_RELEASE_POLICY_SCHEMA_VERSION",
    "ChangeKind",
    "ModelReleasePolicy",
    "PolicyEffect",
    "PolicyObjectType",
    "PolicyRule",
    "PolicySelector",
    "load_model_release_policy",
    "parse_model_release_policy",
]

MODEL_RELEASE_POLICY_SCHEMA: Final = "metrifid.model_release_policy"
MODEL_RELEASE_POLICY_SCHEMA_VERSION: Final = 1
MODEL_RELEASE_POLICY_MAX_BYTES: Final = 1024 * 1024
MODEL_RELEASE_POLICY_MAX_RULES: Final = 4096
_MAX_TOKEN_BYTES: Final = 256
_POLICY_LIMITS: Final = JsonAdmissionLimits(
    max_bytes=MODEL_RELEASE_POLICY_MAX_BYTES,
    max_depth=8,
    max_nodes=100_000,
    max_string_bytes=_MAX_TOKEN_BYTES,
)
_ROOT_FIELDS: Final = {
    "schema",
    "schema_version",
    "baseline_compiled_sha256",
    "candidate_compiled_sha256",
    "rules",
}
_RULE_FIELDS: Final = {"id", "effect", "selector", "before_sha256", "after_sha256"}
_SELECTOR_FIELDS: Final = {"object_type", "object_name", "field", "change_kind"}
_EnumT = TypeVar("_EnumT", bound=StrEnum)


class PolicyEffect(StrEnum):
    """The closed decision vocabulary for one model-release policy rule."""

    ALLOW = "ALLOW"
    REQUIRE = "REQUIRE"
    FORBID = "FORBID"


class ChangeKind(StrEnum):
    """The closed structural-change vocabulary admitted by policy selectors."""

    ADD = "ADD"
    REMOVE = "REMOVE"
    MODIFY = "MODIFY"


class PolicyObjectType(StrEnum):
    """The closed object vocabulary exposed by the compiled-model diff."""

    BODY = "body"
    JOINT = "joint"
    GEOM = "geom"
    MESH = "mesh"
    ACTUATOR = "actuator"
    COMPILED_FIELD = "compiled_field"
    OPAQUE = "opaque"


_FIELDS_BY_OBJECT_TYPE: Final[dict[PolicyObjectType, frozenset[str]]] = {
    PolicyObjectType.BODY: frozenset({"presence", "parent", "mass", "inertia"}),
    PolicyObjectType.JOINT: frozenset({"presence", "body", "type", "limited", "range"}),
    PolicyObjectType.GEOM: frozenset({"presence", "body", "mesh"}),
    PolicyObjectType.MESH: frozenset({"presence", "compiled_geometry_sha256"}),
    PolicyObjectType.ACTUATOR: frozenset({"presence", "transmission", "targets"}),
    PolicyObjectType.COMPILED_FIELD: frozenset({"value"}),
    PolicyObjectType.OPAQUE: frozenset({"compiled_artifact"}),
}


@dataclass(frozen=True, slots=True)
class PolicySelector:
    """One exact or whole-name-wildcard compiled-change selector."""

    object_type: PolicyObjectType
    object_name: str
    field: str
    change_kind: ChangeKind

    def __post_init__(self) -> None:
        """Validate the selector against the closed type/field registry."""
        if not isinstance(self.object_type, PolicyObjectType):
            raise TypeError("selector object_type must be a PolicyObjectType")
        _validate_object_name(self.object_name)
        field = _bounded_nonempty_text(self.field, "selector field")
        if field not in _FIELDS_BY_OBJECT_TYPE[self.object_type]:
            raise ValueError(
                f"selector field {field!r} is not valid for object_type {self.object_type.value!r}"
            )
        if not isinstance(self.change_kind, ChangeKind):
            raise TypeError("selector change_kind must be a ChangeKind")

    @classmethod
    def from_primitive(cls, value: object) -> PolicySelector:
        """Parse one selector object with an exact, closed field set."""
        obj = _object(value, "model-release policy selector")
        _fields(obj, _SELECTOR_FIELDS, "model-release policy selector")
        return cls(
            _enum_member(
                PolicyObjectType,
                obj["object_type"],
                "selector object_type",
            ),
            _validate_object_name(obj["object_name"]),
            _bounded_nonempty_text(obj["field"], "selector field"),
            _enum_member(ChangeKind, obj["change_kind"], "selector change_kind"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return the canonical JSON primitive for this selector."""
        return {
            "object_type": self.object_type.value,
            "object_name": self.object_name,
            "field": self.field,
            "change_kind": self.change_kind.value,
        }

    def sort_key(self) -> tuple[str, str, str, str]:
        """Return the frozen Unicode-code-point ordering key for this selector."""
        return (
            self.object_type.value,
            self.object_name,
            self.field,
            self.change_kind.value,
        )


@dataclass(frozen=True, slots=True)
class PolicyRule:
    """One admitted allow, require, or forbid declaration."""

    id: str
    effect: PolicyEffect
    selector: PolicySelector
    before_sha256: str | None
    after_sha256: str | None

    def __post_init__(self) -> None:
        """Validate identity, effect, selector, and digest-constraint semantics."""
        _bounded_nonempty_text(self.id, "policy rule id")
        if not isinstance(self.effect, PolicyEffect):
            raise TypeError("policy rule effect must be a PolicyEffect")
        if not isinstance(self.selector, PolicySelector):
            raise TypeError("policy rule selector must be a PolicySelector")
        _optional_hash(self.before_sha256, "before_sha256")
        _optional_hash(self.after_sha256, "after_sha256")
        if self.effect is PolicyEffect.REQUIRE:
            if self.selector.object_name == "*":
                raise ValueError("REQUIRE rules must use an exact object_name")
            if self.selector.change_kind is ChangeKind.MODIFY and (
                self.before_sha256 is None or self.after_sha256 is None
            ):
                raise ValueError("MODIFY REQUIRE rules must bind both exact value digests")
            if self.selector.change_kind is ChangeKind.ADD and (
                self.before_sha256 is not None or self.after_sha256 is None
            ):
                raise ValueError(
                    "ADD REQUIRE rules must bind exact absence before and an exact after digest"
                )
            if self.selector.change_kind is ChangeKind.REMOVE and (
                self.before_sha256 is None or self.after_sha256 is not None
            ):
                raise ValueError(
                    "REMOVE REQUIRE rules must bind an exact before digest and exact absence after"
                )

    @classmethod
    def from_primitive(cls, value: object) -> PolicyRule:
        """Parse one policy rule object with an exact, closed field set."""
        obj = _object(value, "model-release policy rule")
        _fields(obj, _RULE_FIELDS, "model-release policy rule")
        return cls(
            _bounded_nonempty_text(obj["id"], "policy rule id"),
            _enum_member(PolicyEffect, obj["effect"], "policy rule effect"),
            PolicySelector.from_primitive(obj["selector"]),
            _optional_hash(obj["before_sha256"], "before_sha256"),
            _optional_hash(obj["after_sha256"], "after_sha256"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return the canonical JSON primitive for this rule."""
        return {
            "id": self.id,
            "effect": self.effect.value,
            "selector": self.selector.to_primitive(),
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
        }

    def sort_key(self) -> tuple[str, str, str, str, str]:
        """Order rules deterministically by selector and then rule identity."""
        return (*self.selector.sort_key(), self.id)


@dataclass(frozen=True, slots=True)
class ModelReleasePolicy:
    """An immutable admitted policy with exact raw and semantic identities."""

    schema: str
    schema_version: int
    baseline_compiled_sha256: str
    candidate_compiled_sha256: str | None
    rules: tuple[PolicyRule, ...]
    raw_sha256: str
    semantic_sha256: str

    def __post_init__(self) -> None:
        """Revalidate direct construction and require canonical rule ordering."""
        if self.schema != MODEL_RELEASE_POLICY_SCHEMA:
            raise ValueError(f"policy schema must be {MODEL_RELEASE_POLICY_SCHEMA!r}")
        if type(self.schema_version) is not int or (
            self.schema_version != MODEL_RELEASE_POLICY_SCHEMA_VERSION
        ):
            raise ValueError(
                f"policy schema_version must be exactly {MODEL_RELEASE_POLICY_SCHEMA_VERSION}"
            )
        require_sha256(self.baseline_compiled_sha256, "baseline_compiled_sha256")
        _optional_hash(self.candidate_compiled_sha256, "candidate_compiled_sha256")
        if type(self.rules) is not tuple:
            raise TypeError("policy rules must be an immutable tuple")
        if len(self.rules) > MODEL_RELEASE_POLICY_MAX_RULES:
            raise ValueError(f"policy rules exceed the maximum of {MODEL_RELEASE_POLICY_MAX_RULES}")
        if any(not isinstance(rule, PolicyRule) for rule in self.rules):
            raise TypeError("policy rules must contain only PolicyRule values")
        _validate_rule_set(self.rules)
        if self.rules != tuple(sorted(self.rules, key=PolicyRule.sort_key)):
            raise ValueError("policy rules must be in canonical selector/id order")
        require_sha256(self.raw_sha256, "raw_sha256")
        require_sha256(self.semantic_sha256, "semantic_sha256")
        if self.semantic_sha256 != canonical_sha256(self.to_primitive()):
            raise ValueError("semantic_sha256 does not match canonical policy semantics")

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return the canonical semantic policy object, excluding identity metadata."""
        rules: list[CanonicalValue] = [rule.to_primitive() for rule in self.rules]
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "baseline_compiled_sha256": self.baseline_compiled_sha256,
            "candidate_compiled_sha256": self.candidate_compiled_sha256,
            "rules": rules,
        }


def parse_model_release_policy(data: bytes) -> ModelReleasePolicy:
    """Admit one strict UTF-8 policy document from bounded raw bytes.

    Duplicate JSON names, raw floating-point tokens, excessive structure, unknown fields, invalid
    enum values, ambiguous rules, and out-of-bound content are all refused before a policy is
    returned.

    Args:
        data: Exact policy-file bytes.  Text callers must encode deliberately before admission.

    Returns:
        The immutable, canonicalized policy and both of its identities.
    """
    if type(data) is not bytes:
        raise TypeError("model-release policy data must be bytes")
    value = bounded_strict_json_loads(data, _POLICY_LIMITS)
    obj = _object(value, "model-release policy")
    _fields(obj, _ROOT_FIELDS, "model-release policy")
    schema = _bounded_nonempty_text(obj["schema"], "policy schema")
    if schema != MODEL_RELEASE_POLICY_SCHEMA:
        raise ValueError(f"policy schema must be {MODEL_RELEASE_POLICY_SCHEMA!r}")
    schema_version = _exact_int(obj["schema_version"], "policy schema_version")
    if schema_version != MODEL_RELEASE_POLICY_SCHEMA_VERSION:
        raise ValueError(
            f"policy schema_version must be exactly {MODEL_RELEASE_POLICY_SCHEMA_VERSION}"
        )
    raw_rules = _sequence(obj["rules"], "policy rules")
    if len(raw_rules) > MODEL_RELEASE_POLICY_MAX_RULES:
        raise ValueError(f"policy rules exceed the maximum of {MODEL_RELEASE_POLICY_MAX_RULES}")
    rules = tuple(PolicyRule.from_primitive(rule) for rule in raw_rules)
    _validate_rule_set(rules)
    canonical_rules = tuple(sorted(rules, key=PolicyRule.sort_key))
    primitive: dict[str, CanonicalValue] = {
        "schema": schema,
        "schema_version": schema_version,
        "baseline_compiled_sha256": require_sha256(
            obj["baseline_compiled_sha256"], "baseline_compiled_sha256"
        ),
        "candidate_compiled_sha256": _optional_hash(
            obj["candidate_compiled_sha256"], "candidate_compiled_sha256"
        ),
        "rules": [rule.to_primitive() for rule in canonical_rules],
    }
    return ModelReleasePolicy(
        schema,
        schema_version,
        cast(str, primitive["baseline_compiled_sha256"]),
        cast(str | None, primitive["candidate_compiled_sha256"]),
        canonical_rules,
        hashlib.sha256(data).hexdigest(),
        canonical_sha256(primitive),
    )


def load_model_release_policy(path: str | Path) -> ModelReleasePolicy:
    """Read one no-follow regular file and admit it as a model-release policy."""
    return parse_model_release_policy(
        read_bounded_regular_file(path, MODEL_RELEASE_POLICY_MAX_BYTES)
    )


def _bounded_nonempty_text(value: object, field: str) -> str:
    """Admit one nonempty strict UTF-8 token no larger than 256 encoded bytes."""
    if type(value) is not str:
        raise TypeError(f"{field} must be a string")
    result = value
    if not result:
        raise ValueError(f"{field} must be nonempty")
    try:
        encoded = result.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must be strict UTF-8") from exc
    if len(encoded) > _MAX_TOKEN_BYTES:
        raise ValueError(f"{field} exceeds the {_MAX_TOKEN_BYTES} UTF-8 byte limit")
    return result


def _validate_object_name(value: object) -> str:
    """Admit one exact semantic name or the entire wildcard token ``*``."""
    result = _bounded_nonempty_text(value, "selector object_name")
    if "\x00" in result:
        raise ValueError("selector object_name must not contain U+0000")
    if "*" in result and result != "*":
        raise ValueError("selector object_name wildcard must be the whole token '*'")
    return result


def _enum_member(enum_type: type[_EnumT], value: object, field: str) -> _EnumT:
    """Admit a strict string member of one closed string enum."""
    token = _bounded_nonempty_text(value, field)
    try:
        return enum_type(token)
    except ValueError as exc:
        allowed = sorted(member.value for member in enum_type)
        raise ValueError(f"{field} must be one of {allowed}") from exc


def _validate_rule_set(rules: tuple[PolicyRule, ...]) -> None:
    """Reject duplicate identities, selectors, and wildcard/exact selector overlap."""
    _reject_duplicate_ids(rules)
    _reject_duplicate_selectors(rules)
    _reject_overlapping_selectors(rules)


def _reject_duplicate_ids(rules: tuple[PolicyRule, ...]) -> None:
    """Reject repeated rule identifiers independent of rule ordering."""
    seen: set[str] = set()
    for rule in rules:
        if rule.id in seen:
            raise ValueError(f"duplicate policy rule id: {rule.id!r}")
        seen.add(rule.id)


def _reject_duplicate_selectors(rules: tuple[PolicyRule, ...]) -> None:
    """Reject two rules that carry the same exact selector."""
    seen: set[PolicySelector] = set()
    for rule in rules:
        if rule.selector in seen:
            raise ValueError(f"duplicate policy selector: {rule.selector.sort_key()!r}")
        seen.add(rule.selector)


def _reject_overlapping_selectors(rules: tuple[PolicyRule, ...]) -> None:
    """Reject wildcard/exact selectors that could select the same compiled change."""
    by_shape: dict[tuple[PolicyObjectType, str, ChangeKind], list[str]] = {}
    for rule in rules:
        selector = rule.selector
        shape = (selector.object_type, selector.field, selector.change_kind)
        names = by_shape.setdefault(shape, [])
        if names and (selector.object_name == "*" or "*" in names):
            raise ValueError(
                "potentially overlapping policy selectors for "
                f"{(selector.object_type.value, selector.field, selector.change_kind.value)!r}"
            )
        names.append(selector.object_name)
