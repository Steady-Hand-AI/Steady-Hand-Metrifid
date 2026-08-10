"""Strict model-closure and comparison-input identity schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, cast

from ._schema_primitives import (
    _fields,
    _nonnegative_int,
    _object,
    _optional_hash,
    _relative_posix_path,
    _require_instance,
    _require_typed_tuple,
    _sequence,
)
from .json_values import CanonicalValue, canonical_sha256, require_sha256


@dataclass(frozen=True, slots=True)
class ModelClosureMember:
    """Identify one measured model file by relative path, byte count, and content hash."""

    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        """Validate the confined relative path, nonnegative size, and SHA-256 digest."""
        _relative_posix_path(self.path, "path")
        _nonnegative_int(self.size_bytes, "size_bytes")
        require_sha256(self.sha256, "sha256")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode an exact path/size/hash model-closure member object."""
        obj = _object(value, "ModelClosureMember")
        _fields(obj, {"path", "size_bytes", "sha256"}, "ModelClosureMember")
        return cls(
            _relative_posix_path(obj["path"], "path"),
            _nonnegative_int(obj["size_bytes"], "size_bytes"),
            require_sha256(obj["sha256"], "sha256"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit this member's relative path, measured size, and content hash."""
        return {"path": self.path, "size_bytes": self.size_bytes, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ModelClosureIdentity:
    """Identify a model entrypoint and its complete ordered set of measured files."""

    entrypoint: str
    member_count: int
    members: tuple[ModelClosureMember, ...]

    def __post_init__(self) -> None:
        """Require unique path-ordered members containing the declared entrypoint."""
        _relative_posix_path(self.entrypoint, "entrypoint")
        _nonnegative_int(self.member_count, "member_count")
        _require_typed_tuple(self.members, ModelClosureMember, "members")
        if self.member_count != len(self.members):
            raise ValueError("member_count must equal len(members)")
        paths = tuple(member.path for member in self.members)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("model closure members must be uniquely sorted by path")
        if self.entrypoint not in paths:
            raise ValueError("entrypoint must be one of the closure members")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode an entrypoint and exact ordered member list into a closure identity."""
        obj = _object(value, "ModelClosureIdentity")
        _fields(obj, {"entrypoint", "member_count", "members"}, "ModelClosureIdentity")
        members = tuple(
            ModelClosureMember.from_primitive(item) for item in _sequence(obj["members"], "members")
        )
        return cls(
            _relative_posix_path(obj["entrypoint"], "entrypoint"),
            _nonnegative_int(obj["member_count"], "member_count"),
            members,
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the entrypoint, asserted member count, and measured member objects."""
        return {
            "entrypoint": self.entrypoint,
            "member_count": self.member_count,
            "members": [member.to_primitive() for member in self.members],
        }

    def sha256(self) -> str:
        """Hash the complete canonical model-closure identity."""
        return canonical_sha256(self.to_primitive())


@dataclass(frozen=True, slots=True)
class ModelClosures:
    """Pair the measured baseline and candidate model-closure identities."""

    baseline: ModelClosureIdentity
    candidate: ModelClosureIdentity

    def __post_init__(self) -> None:
        """Require typed closure identities for both comparison roles."""
        _require_instance(self.baseline, ModelClosureIdentity, "baseline")
        _require_instance(self.candidate, ModelClosureIdentity, "candidate")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode the exact baseline/candidate closure pair."""
        obj = _object(value, "ModelClosures")
        _fields(obj, {"baseline", "candidate"}, "ModelClosures")
        return cls(
            ModelClosureIdentity.from_primitive(obj["baseline"]),
            ModelClosureIdentity.from_primitive(obj["candidate"]),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit nested baseline and candidate closure identities."""
        return {
            "baseline": self.baseline.to_primitive(),
            "candidate": self.candidate.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class ComparisonInputsIdentity:
    """Bind every raw and semantic comparison input to its SHA-256 identity."""

    configuration_raw_sha256: str
    comparison_contract_sha256: str
    baseline_model_closure_sha256: str
    candidate_model_closure_sha256: str
    initial_state_raw_sha256: str
    initial_state_semantic_sha256: str
    actions_raw_sha256: str
    actions_semantic_sha256: str
    aliases_raw_sha256: str | None
    aliases_semantic_sha256: str | None

    def __post_init__(self) -> None:
        """Validate all required digests and the all-or-none alias digest pair."""
        for field_name in (
            "configuration_raw_sha256",
            "comparison_contract_sha256",
            "baseline_model_closure_sha256",
            "candidate_model_closure_sha256",
            "initial_state_raw_sha256",
            "initial_state_semantic_sha256",
            "actions_raw_sha256",
            "actions_semantic_sha256",
        ):
            require_sha256(getattr(self, field_name), field_name)
        if (self.aliases_raw_sha256 is None) != (self.aliases_semantic_sha256 is None):
            raise ValueError("alias raw and semantic hashes must both be null or both be present")
        if self.aliases_raw_sha256 is not None:
            require_sha256(self.aliases_raw_sha256, "aliases_raw_sha256")
            require_sha256(cast(str, self.aliases_semantic_sha256), "aliases_semantic_sha256")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode the exact digest bindings for configuration, models, workload, and aliases."""
        obj = _object(value, "ComparisonInputsIdentity")
        expected = {
            "configuration_raw_sha256",
            "comparison_contract_sha256",
            "baseline_model_closure_sha256",
            "candidate_model_closure_sha256",
            "initial_state_raw_sha256",
            "initial_state_semantic_sha256",
            "actions_raw_sha256",
            "actions_semantic_sha256",
            "aliases_raw_sha256",
            "aliases_semantic_sha256",
        }
        _fields(obj, expected, "ComparisonInputsIdentity")
        return cls(
            require_sha256(obj["configuration_raw_sha256"], "configuration_raw_sha256"),
            require_sha256(obj["comparison_contract_sha256"], "comparison_contract_sha256"),
            require_sha256(obj["baseline_model_closure_sha256"], "baseline_model_closure_sha256"),
            require_sha256(obj["candidate_model_closure_sha256"], "candidate_model_closure_sha256"),
            require_sha256(obj["initial_state_raw_sha256"], "initial_state_raw_sha256"),
            require_sha256(obj["initial_state_semantic_sha256"], "initial_state_semantic_sha256"),
            require_sha256(obj["actions_raw_sha256"], "actions_raw_sha256"),
            require_sha256(obj["actions_semantic_sha256"], "actions_semantic_sha256"),
            _optional_hash(obj["aliases_raw_sha256"], "aliases_raw_sha256"),
            _optional_hash(obj["aliases_semantic_sha256"], "aliases_semantic_sha256"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit every raw, semantic, closure, and contract digest without recomputation."""
        return {
            "configuration_raw_sha256": self.configuration_raw_sha256,
            "comparison_contract_sha256": self.comparison_contract_sha256,
            "baseline_model_closure_sha256": self.baseline_model_closure_sha256,
            "candidate_model_closure_sha256": self.candidate_model_closure_sha256,
            "initial_state_raw_sha256": self.initial_state_raw_sha256,
            "initial_state_semantic_sha256": self.initial_state_semantic_sha256,
            "actions_raw_sha256": self.actions_raw_sha256,
            "actions_semantic_sha256": self.actions_semantic_sha256,
            "aliases_raw_sha256": self.aliases_raw_sha256,
            "aliases_semantic_sha256": self.aliases_semantic_sha256,
        }
