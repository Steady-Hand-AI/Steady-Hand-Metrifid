"""Compiled joint/actuator identity, explicit aliases, and deterministic alignment."""

# Keep the baseline import statement order for rename-normalized AST identity.
from __future__ import annotations  # noqa: I001

import hashlib
from dataclasses import dataclass, replace
from typing import ClassVar, Self, cast

from ._model_admission import (
    CompiledModelIdentity,
)
from ._model_closure import (
    AlignedActuator,
    AlignedJoint,
    _completed_hash,
    _require_exact_object_fields,
    _nonnegative_int,
    _strict_array,
    _strict_name,
    _unique_sorted_names,
    refuse,
)
from ._model_identity_validation import (
    strict_alias_bindings_from_primitive,
    validate_model_pair_semantics,
    validate_semantic_alignment_local,
)
from .json_values import (
    CanonicalValue,
    FrozenCanonicalObject,
    canonical_json_bytes,
    canonical_sha256,
    compute_self_hash,
    require_sha256,
    strict_json_loads,
    thaw_canonical,
    validate_self_hash,
)
from .operational import OperationalReasonCode
from .schemas import (
    AliasArtifact,
    AlignmentSummary,
    ModelClosureIdentity,
)


def _validate_alignment_aliases(alignment: SemanticAlignment) -> None:
    """Validate alias hashes and canonical alias-binding order."""
    if (alignment.aliases_raw_sha256 is None) != (alignment.aliases_semantic_sha256 is None):
        raise ValueError("alias hashes must both be present or absent")
    if alignment.aliases_raw_sha256 is not None:
        require_sha256(alignment.aliases_raw_sha256, "aliases_raw_sha256")
        require_sha256(cast(str, alignment.aliases_semantic_sha256), "aliases_semantic_sha256")
    binding_bytes = [
        canonical_json_bytes(thaw_canonical(item)) for item in alignment.alias_bindings
    ]
    if binding_bytes != sorted(binding_bytes) or len(binding_bytes) != len(set(binding_bytes)):
        raise ValueError("alias bindings must be uniquely sorted by canonical bytes")


def _validate_alignment_members(alignment: SemanticAlignment) -> None:
    """Validate aligned joint and actuator collections."""
    if not all(isinstance(item, AlignedJoint) for item in alignment.joints):
        raise TypeError("joints must contain AlignedJoint values")
    if not all(isinstance(item, AlignedActuator) for item in alignment.actuators):
        raise TypeError("actuators must contain AlignedActuator values")
    if not _unique_sorted_names(alignment.joints) or not _unique_sorted_names(alignment.actuators):
        raise ValueError("aligned identities must be uniquely sorted")


@dataclass(frozen=True, slots=True)
class SemanticAlignment:
    """Bind canonical joints and actuators across roles, including explicit alias evidence."""

    schema: str
    schema_version: int
    semantic_alignment_sha256: str | None
    aliases_raw_sha256: str | None
    aliases_semantic_sha256: str | None
    joints: tuple[AlignedJoint, ...]
    actuators: tuple[AlignedActuator, ...]
    alias_bindings: tuple[FrozenCanonicalObject, ...]

    _SCHEMA: ClassVar[str] = "metrifid.semantic_alignment"
    _SCHEMA_VERSION: ClassVar[int] = 1
    _HASH_FIELD: ClassVar[str] = "semantic_alignment_sha256"

    def __post_init__(self) -> None:
        """Validate schema, aliases, aligned members, local invariants, and optional self-hash."""
        if self.schema != self._SCHEMA:
            raise ValueError("invalid semantic alignment schema")
        if type(self.schema_version) is not int or self.schema_version != self._SCHEMA_VERSION:
            raise ValueError("invalid semantic alignment schema_version")
        _validate_alignment_aliases(self)
        _validate_alignment_members(self)
        validate_semantic_alignment_local(
            self.joints,
            self.actuators,
            self.aliases_raw_sha256,
            self.aliases_semantic_sha256,
            self.alias_bindings,
        )
        if self.semantic_alignment_sha256 is not None:
            validate_self_hash(self._primitive(), self._HASH_FIELD)

    def _primitive(self) -> dict[str, CanonicalValue]:
        """Assemble canonical semantic-alignment values, including unhashed drafts."""
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "semantic_alignment_sha256": self.semantic_alignment_sha256,
            "aliases_raw_sha256": self.aliases_raw_sha256,
            "aliases_semantic_sha256": self.aliases_semantic_sha256,
            "joints": [item.to_primitive() for item in self.joints],
            "actuators": [item.to_primitive() for item in self.actuators],
            "alias_bindings": [thaw_canonical(item) for item in self.alias_bindings],
        }

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit a completed semantic alignment; reject an unhashed draft."""
        if self.semantic_alignment_sha256 is None:
            raise ValueError("semantic alignment is not completed")
        return self._primitive()

    def finalized(self) -> Self:
        """Return this alignment with its canonical self-hash populated."""
        if self.semantic_alignment_sha256 is not None:
            return self
        return replace(
            self,
            semantic_alignment_sha256=compute_self_hash(self._primitive(), self._HASH_FIELD),
        )

    def summary(self) -> AlignmentSummary:
        """Build the self-hashed canonical joint, actuator, and alias ordering summary."""
        return AlignmentSummary(
            None,
            tuple(item.canonical_name for item in self.joints),
            tuple(item.canonical_name for item in self.actuators),
            self.alias_bindings,
        ).finalized()

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode and self-hash-validate complete cross-role semantic alignment evidence."""
        fields = {
            "schema",
            "schema_version",
            "semantic_alignment_sha256",
            "aliases_raw_sha256",
            "aliases_semantic_sha256",
            "joints",
            "actuators",
            "alias_bindings",
        }
        obj = _require_exact_object_fields(value, fields, "SemanticAlignment")
        bindings = strict_alias_bindings_from_primitive(obj["alias_bindings"])
        result = cls(
            cast(str, _strict_name(obj["schema"], "schema")),
            cast(int, _nonnegative_int(obj["schema_version"], "schema_version")),
            _completed_hash(obj["semantic_alignment_sha256"], "semantic_alignment_sha256"),
            None
            if obj["aliases_raw_sha256"] is None
            else require_sha256(obj["aliases_raw_sha256"], "aliases_raw_sha256"),
            None
            if obj["aliases_semantic_sha256"] is None
            else require_sha256(obj["aliases_semantic_sha256"], "aliases_semantic_sha256"),
            tuple(
                AlignedJoint.from_primitive(item) for item in _strict_array(obj["joints"], "joints")
            ),
            tuple(
                AlignedActuator.from_primitive(item)
                for item in _strict_array(obj["actuators"], "actuators")
            ),
            bindings,
        )
        validate_self_hash(result.to_primitive(), result._HASH_FIELD)
        return result


@dataclass(frozen=True, slots=True)
class ModelPairIdentity:
    """Bind two source closures, compiled identities, and their semantic alignment."""

    schema: str
    schema_version: int
    model_pair_identity_sha256: str | None
    baseline_closure: ModelClosureIdentity
    candidate_closure: ModelClosureIdentity
    baseline_compiled: CompiledModelIdentity
    candidate_compiled: CompiledModelIdentity
    alignment: SemanticAlignment
    alignment_summary: AlignmentSummary

    _SCHEMA: ClassVar[str] = "metrifid.model_pair_identity"
    _SCHEMA_VERSION: ClassVar[int] = 1
    _HASH_FIELD: ClassVar[str] = "model_pair_identity_sha256"

    def __post_init__(self) -> None:
        """Validate closure bindings, generated alignment semantics, summary, and self-hash."""
        if self.schema != self._SCHEMA:
            raise ValueError("invalid model pair identity schema")
        if type(self.schema_version) is not int or self.schema_version != self._SCHEMA_VERSION:
            raise ValueError("invalid model pair identity schema_version")
        if self.baseline_compiled.model_closure_sha256 != self.baseline_closure.sha256():
            raise ValueError("baseline compiled identity does not bind its closure")
        if self.candidate_compiled.model_closure_sha256 != self.candidate_closure.sha256():
            raise ValueError("candidate compiled identity does not bind its closure")
        validate_model_pair_semantics(
            self.baseline_closure.sha256(),
            self.candidate_closure.sha256(),
            self.baseline_compiled,
            self.candidate_compiled,
            self.alignment,
        )
        if self.alignment.summary() != self.alignment_summary:
            raise ValueError("alignment summary does not match semantic alignment")
        if self.model_pair_identity_sha256 is not None:
            validate_self_hash(self._primitive(), self._HASH_FIELD)

    def _primitive(self) -> dict[str, CanonicalValue]:
        """Assemble the nested canonical model-pair object, including an optional self-hash."""
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "model_pair_identity_sha256": self.model_pair_identity_sha256,
            "baseline_closure": self.baseline_closure.to_primitive(),
            "candidate_closure": self.candidate_closure.to_primitive(),
            "baseline_compiled": self.baseline_compiled.to_primitive(),
            "candidate_compiled": self.candidate_compiled.to_primitive(),
            "alignment": self.alignment.to_primitive(),
            "alignment_summary": self.alignment_summary.to_primitive(),
        }

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit a completed model-pair identity; reject an unhashed draft."""
        if self.model_pair_identity_sha256 is None:
            raise ValueError("model pair identity is not completed")
        return self._primitive()

    def finalized(self) -> Self:
        """Return this model-pair identity with its canonical self-hash populated."""
        if self.model_pair_identity_sha256 is not None:
            return self
        return replace(
            self,
            model_pair_identity_sha256=compute_self_hash(self._primitive(), self._HASH_FIELD),
        )

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode and validate a complete nested model-pair identity object."""
        fields = {
            "schema",
            "schema_version",
            "model_pair_identity_sha256",
            "baseline_closure",
            "candidate_closure",
            "baseline_compiled",
            "candidate_compiled",
            "alignment",
            "alignment_summary",
        }
        obj = _require_exact_object_fields(value, fields, "ModelPairIdentity")
        result = cls(
            cast(str, _strict_name(obj["schema"], "schema")),
            cast(int, _nonnegative_int(obj["schema_version"], "schema_version")),
            _completed_hash(obj["model_pair_identity_sha256"], "model_pair_identity_sha256"),
            ModelClosureIdentity.from_primitive(obj["baseline_closure"]),
            ModelClosureIdentity.from_primitive(obj["candidate_closure"]),
            CompiledModelIdentity.from_primitive(obj["baseline_compiled"]),
            CompiledModelIdentity.from_primitive(obj["candidate_compiled"]),
            SemanticAlignment.from_primitive(obj["alignment"]),
            AlignmentSummary.from_primitive(obj["alignment_summary"]),
        )
        validate_self_hash(result.to_primitive(), result._HASH_FIELD)
        return result


def _parse_aliases(
    raw: str | bytes | None, baseline_hash: str, candidate_hash: str
) -> tuple[AliasArtifact | None, str | None, str | None]:
    """Parse raw, baseline hash and candidate hash into the parse aliases representation used by model identity types, rejecting invalid input with refuse, TypeError."""
    if raw is None:
        return None, None, None
    try:
        if isinstance(raw, str):
            raw_bytes = raw.encode("utf-8", errors="strict")
        elif type(raw) is bytes:
            raw_bytes = raw
        else:
            raise TypeError("aliases must be UTF-8 text or bytes")
        artifact = AliasArtifact.from_primitive(strict_json_loads(raw_bytes))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise refuse(
            OperationalReasonCode.ALIAS_SCHEMA_INVALID,
            "comparison",
            exception_type=type(exc).__name__,
            message=str(exc),
        ) from exc
    if (
        artifact.baseline_model_closure_sha256 != baseline_hash
        or artifact.candidate_model_closure_sha256 != candidate_hash
    ):
        raise refuse(
            OperationalReasonCode.ALIAS_CLOSURE_HASH_MISMATCH,
            "comparison",
            measured_baseline=baseline_hash,
            measured_candidate=candidate_hash,
            declared_baseline=artifact.baseline_model_closure_sha256,
            declared_candidate=artifact.candidate_model_closure_sha256,
        )
    return (
        artifact,
        hashlib.sha256(raw_bytes).hexdigest(),
        canonical_sha256(artifact.to_primitive()),
    )
