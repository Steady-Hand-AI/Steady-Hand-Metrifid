"""Strict pre-contract operational-failure ABI and frozen registries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import cast

from ._operational_registry import (
    _EXECUTION_IDENTITY_MISMATCH_REASONS,
    _EXECUTION_IDENTITY_UNBOUND_REASONS,
    _INPUT_DIGEST_RANK,
    _OPERATIONAL_FAILURE_SCHEMA,
    _OPERATIONAL_FAILURE_SCHEMA_VERSION,
    _OPERATIONAL_RULE_SCHEMA,
    _OPERATIONAL_RULE_SCHEMA_VERSION,
    _OPERATIONS,
    _POST_IDENTITY_STAGES,
)
from ._operational_registry import (
    OPERATIONAL_REASON_REGISTRY as OPERATIONAL_REASON_REGISTRY,
)
from ._operational_registry import (
    ExecutionIdentityState as ExecutionIdentityState,
)
from ._operational_registry import (
    InputDigestCode as InputDigestCode,
)
from ._operational_registry import (
    OperationalReasonCode as OperationalReasonCode,
)
from ._operational_registry import (
    OperationalReasonRule as OperationalReasonRule,
)
from ._operational_registry import (
    OperationalStage as OperationalStage,
)
from .errors import OperationalExitCode, ReasonRole
from .json_values import (
    CanonicalValue,
    FrozenCanonicalObject,
    canonical_sha256,
    freeze_canonical,
    require_sha256,
    thaw_canonical,
    validate_self_hash,
)
from .schemas import EnvironmentIdentity


@dataclass(frozen=True, slots=True)
class OperationalToolObservation:
    """Tool version and honest execution-identity observation."""

    version: str
    execution_identity_state: ExecutionIdentityState
    distribution_sha256: str | None

    def __post_init__(self) -> None:
        """Match verified, unbound, or mismatched execution state to its permitted digest."""
        _nonempty_string(self.version, "version")
        if self.execution_identity_state not in {
            "VERIFIED_INSTALLED_DISTRIBUTION",
            "UNBOUND",
            "MISMATCH",
        }:
            raise ValueError("execution_identity_state is outside the frozen registry")
        if self.execution_identity_state == "VERIFIED_INSTALLED_DISTRIBUTION":
            if self.distribution_sha256 is None:
                raise ValueError("verified execution identity requires distribution_sha256")
            require_sha256(self.distribution_sha256, "distribution_sha256")
            if self.distribution_sha256 == "0" * 64:
                raise ValueError("verified execution identity forbids a placeholder hash")
        elif self.distribution_sha256 is not None:
            raise ValueError("unverified execution identity requires null distribution_sha256")

    @classmethod
    def from_primitive(cls, value: object) -> OperationalToolObservation:
        """Decode exact tool-version and execution-identity observations."""
        obj = _require_exact_object_fields(
            value,
            {"version", "execution_identity_state", "distribution_sha256"},
            "OperationalToolObservation",
        )
        state = _nonempty_string(obj["execution_identity_state"], "execution_identity_state")
        if state not in {"VERIFIED_INSTALLED_DISTRIBUTION", "UNBOUND", "MISMATCH"}:
            raise ValueError("execution_identity_state is outside the frozen registry")
        return cls(
            _nonempty_string(obj["version"], "version"),
            cast(ExecutionIdentityState, state),
            _optional_hash(obj["distribution_sha256"], "distribution_sha256"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit tool version, binding state, and its permitted distribution digest."""
        return {
            "version": self.version,
            "execution_identity_state": self.execution_identity_state,
            "distribution_sha256": self.distribution_sha256,
        }


@dataclass(frozen=True, slots=True)
class InputDigest:
    """One available pre-contract input digest."""

    code: InputDigestCode
    sha256: str

    def __post_init__(self) -> None:
        """Validate the closed input-digest code and reject malformed or placeholder hashes."""
        if not isinstance(self.code, InputDigestCode):
            raise TypeError("code must be an InputDigestCode")
        require_sha256(self.sha256, "sha256")
        if self.sha256 == "0" * 64:
            raise ValueError("placeholder input digests are forbidden")

    @classmethod
    def from_primitive(cls, value: object) -> InputDigest:
        """Decode one registry-typed available-input digest."""
        obj = _require_exact_object_fields(value, {"code", "sha256"}, "InputDigest")
        token = _nonempty_string(obj["code"], "code")
        try:
            code = InputDigestCode(token)
        except ValueError as exc:
            raise ValueError(f"unknown input digest code: {token}") from exc
        return cls(code, require_sha256(obj["sha256"], "sha256"))

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the input kind token and its measured SHA-256 digest."""
        return {"code": self.code.value, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class OperationalReason:
    """One strict operational reason embedded in OperationalFailure."""

    code: OperationalReasonCode
    role: ReasonRole
    field: str | None
    object_name: str | None
    evidence: FrozenCanonicalObject | dict[str, CanonicalValue]

    def __post_init__(self) -> None:
        """Validate reason code, role, optional subject fields, and canonical evidence."""
        if not isinstance(self.code, OperationalReasonCode):
            raise TypeError("code must be an OperationalReasonCode")
        if self.role not in {None, "baseline", "candidate", "comparison"}:
            raise ValueError("role is outside the frozen role registry")
        _optional_nonempty_string(self.field, "field")
        _optional_nonempty_string(self.object_name, "object_name")
        frozen = _freeze_object(self.evidence, "evidence")
        object.__setattr__(self, "evidence", frozen)

    @classmethod
    def from_primitive(cls, value: object) -> OperationalReason:
        """Decode a closed operational reason and immutably freeze its evidence object."""
        obj = _require_exact_object_fields(
            value,
            {"code", "role", "field", "object_name", "evidence"},
            "OperationalReason",
        )
        token = _nonempty_string(obj["code"], "code")
        try:
            code = OperationalReasonCode(token)
        except ValueError as exc:
            raise ValueError(f"unknown operational reason code: {token}") from exc
        role_raw = obj["role"]
        if role_raw not in {None, "baseline", "candidate", "comparison"}:
            raise ValueError("role is outside the frozen role registry")
        evidence_raw = obj["evidence"]
        if type(evidence_raw) is not dict:
            raise TypeError("evidence must be an object")
        return cls(
            code,
            cast(ReasonRole, role_raw),
            _optional_nonempty_string(obj["field"], "field"),
            _optional_nonempty_string(obj["object_name"], "object_name"),
            cast(dict[str, CanonicalValue], evidence_raw),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit reason subject fields and a fresh canonical evidence object."""
        return {
            "code": self.code.value,
            "role": self.role,
            "field": self.field,
            "object_name": self.object_name,
            "evidence": thaw_canonical(cast(FrozenCanonicalObject, self.evidence)),
        }


def _validate_failure_contract(failure: OperationalFailure) -> None:
    """Validate failure schemas, tool, operation, and typed registry members."""
    if failure.schema != _OPERATIONAL_FAILURE_SCHEMA:
        raise ValueError("invalid operational failure schema")
    if failure.schema_version != _OPERATIONAL_FAILURE_SCHEMA_VERSION:
        raise ValueError("invalid operational failure schema_version")
    if failure.failure_rule_schema != _OPERATIONAL_RULE_SCHEMA:
        raise ValueError("invalid failure_rule_schema")
    if failure.failure_rule_schema_version != _OPERATIONAL_RULE_SCHEMA_VERSION:
        raise ValueError("invalid failure_rule_schema_version")
    _schema_version(failure.schema_version, "schema_version")
    _schema_version(failure.failure_rule_schema_version, "failure_rule_schema_version")
    if not isinstance(failure.tool, OperationalToolObservation):
        raise TypeError("tool must be an OperationalToolObservation")
    if failure.operation not in _OPERATIONS:
        raise ValueError("operation must be compare, audit-timestep or certify")
    if not isinstance(failure.stage, OperationalStage):
        raise TypeError("stage must be an OperationalStage")
    if not isinstance(failure.reason, OperationalReason):
        raise TypeError("reason must be an OperationalReason")
    if not isinstance(failure.exit_code, OperationalExitCode):
        raise TypeError("exit_code must be an OperationalExitCode")


def _validate_failure_outcome(failure: OperationalFailure) -> None:
    """Validate registry binding and optional environment evidence."""
    expected = OPERATIONAL_REASON_REGISTRY[failure.reason.code]
    if failure.stage is not expected.stage:
        raise ValueError("stage does not match the operational reason registry")
    if failure.exit_code is not expected.exit_code:
        raise ValueError("exit_code does not match the operational reason registry")
    _validate_execution_identity_binding(failure)
    if failure.environment is not None:
        if not isinstance(failure.environment, EnvironmentIdentity):
            raise TypeError("environment must be an EnvironmentIdentity or null")
        failure.environment.validate_hash()


@dataclass(frozen=True, slots=True)
class OperationalFailure:
    """Strict self-hashed failure emitted before a comparison contract exists."""

    schema: str
    schema_version: int
    failure_rule_schema: str
    failure_rule_schema_version: int
    tool: OperationalToolObservation
    operation: str
    stage: OperationalStage
    reason: OperationalReason
    available_inputs: tuple[InputDigest, ...]
    environment: EnvironmentIdentity | None
    exit_code: OperationalExitCode
    failure_sha256: str | None

    def __post_init__(self) -> None:
        """Validate registry bindings, canonical input order, environment, and optional self-hash."""
        _validate_failure_contract(self)
        _validate_failure_outcome(self)
        inputs = _canonical_inputs(self.available_inputs)
        object.__setattr__(self, "available_inputs", inputs)
        if self.failure_sha256 is not None:
            require_sha256(self.failure_sha256, "failure_sha256")
            validate_self_hash(self.to_primitive(), "failure_sha256")

    @classmethod
    def from_primitive(cls, value: object) -> OperationalFailure:
        """Decode and self-hash-validate a complete pre-contract failure artifact."""
        obj = _require_exact_object_fields(
            value,
            {
                "schema",
                "schema_version",
                "failure_rule_schema",
                "failure_rule_schema_version",
                "tool",
                "operation",
                "stage",
                "reason",
                "available_inputs",
                "environment",
                "exit_code",
                "failure_sha256",
            },
            "OperationalFailure",
        )
        failure = cls(
            _nonempty_string(obj["schema"], "schema"),
            _schema_version(obj["schema_version"], "schema_version"),
            _nonempty_string(obj["failure_rule_schema"], "failure_rule_schema"),
            _schema_version(obj["failure_rule_schema_version"], "failure_rule_schema_version"),
            OperationalToolObservation.from_primitive(obj["tool"]),
            _nonempty_string(obj["operation"], "operation"),
            _operational_stage(obj["stage"]),
            OperationalReason.from_primitive(obj["reason"]),
            tuple(
                InputDigest.from_primitive(item)
                for item in _array(obj["available_inputs"], "available_inputs")
            ),
            _optional_environment(obj["environment"]),
            _operational_exit_code(obj["exit_code"]),
            require_sha256(obj["failure_sha256"], "failure_sha256"),
        )
        failure.validate_hash()
        return failure

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit a completed operational failure; reject an unhashed draft."""
        if self.failure_sha256 is None:
            raise ValueError("failure_sha256 is required before serialization")
        return {
            **_operational_failure_unhashed_primitive(self),
            "failure_sha256": self.failure_sha256,
        }

    def finalized(self) -> OperationalFailure:
        """Return a new failure with its canonical self-hash populated."""
        if self.failure_sha256 is not None:
            self.validate_hash()
            return self
        digest = canonical_sha256(_operational_failure_unhashed_primitive(self))
        finalized = replace(self, failure_sha256=digest)
        finalized.validate_hash()
        return finalized

    def validate_hash(self) -> None:
        """Validate the required canonical operational-failure self-hash."""
        validate_self_hash(self.to_primitive(), "failure_sha256")


def _operational_stage(value: object) -> OperationalStage:
    """Parse one operational stage under the closed stage registry."""
    token = _nonempty_string(value, "stage")
    try:
        return OperationalStage(token)
    except ValueError as exc:
        raise ValueError(f"unknown operational stage: {token}") from exc


def _operational_exit_code(value: object) -> OperationalExitCode:
    """Parse one strict integer operational exit code under the closed registry."""
    if type(value) is not int:
        raise TypeError("exit_code must be an integer and not a boolean")
    try:
        return OperationalExitCode(value)
    except ValueError as exc:
        raise ValueError(f"unknown operational exit code: {value}") from exc


def _optional_environment(value: object) -> EnvironmentIdentity | None:
    """Parse an optional environment identity from an operational failure."""
    return None if value is None else EnvironmentIdentity.from_primitive(value)


def _operational_failure_unhashed_primitive(
    failure: OperationalFailure,
) -> dict[str, CanonicalValue]:
    """Assemble every operational-failure field except ``failure_sha256``."""
    return {
        "schema": failure.schema,
        "schema_version": failure.schema_version,
        "failure_rule_schema": failure.failure_rule_schema,
        "failure_rule_schema_version": failure.failure_rule_schema_version,
        "tool": failure.tool.to_primitive(),
        "operation": failure.operation,
        "stage": failure.stage.value,
        "reason": failure.reason.to_primitive(),
        "available_inputs": [item.to_primitive() for item in failure.available_inputs],
        "environment": (
            None if failure.environment is None else failure.environment.to_primitive()
        ),
        "exit_code": int(failure.exit_code),
    }


def _validate_execution_identity_binding(failure: OperationalFailure) -> None:
    """Match failure reason and stage to the observed execution-identity state."""
    code = failure.reason.code
    state = failure.tool.execution_identity_state
    if code in _EXECUTION_IDENTITY_MISMATCH_REASONS:
        if state != "MISMATCH":
            raise ValueError("execution-identity mismatch reason requires MISMATCH tool state")
        return
    if code in _EXECUTION_IDENTITY_UNBOUND_REASONS:
        if state != "UNBOUND":
            raise ValueError("execution-identity unbound reason requires UNBOUND tool state")
        return
    if failure.stage in _POST_IDENTITY_STAGES:
        if state != "VERIFIED_INSTALLED_DISTRIBUTION":
            raise ValueError("post-identity operational stage requires verified tool identity")
        return
    if failure.stage in {OperationalStage.INVOCATION, OperationalStage.INTERNAL}:
        if state not in {"UNBOUND", "VERIFIED_INSTALLED_DISTRIBUTION"}:
            raise ValueError("INVOCATION and INTERNAL forbid MISMATCH tool state")
        return
    raise ValueError("operational execution-identity binding is incomplete")


def _canonical_inputs(values: Sequence[InputDigest]) -> tuple[InputDigest, ...]:
    """Validate, deduplicate, and registry-order available pre-contract input digests."""
    raw: object = values
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("available_inputs must be a sequence of InputDigest values")
    if any(not isinstance(value, InputDigest) for value in values):
        raise TypeError("available_inputs must contain only InputDigest values")
    codes = [value.code for value in values]
    if len(codes) != len(set(codes)):
        raise ValueError("available_inputs must not contain duplicate digest codes")
    return tuple(sorted(values, key=lambda value: _INPUT_DIGEST_RANK[value.code]))


def _freeze_object(
    value: FrozenCanonicalObject | dict[str, CanonicalValue], field: str
) -> FrozenCanonicalObject:
    """Recursively canonicalize and freeze an operational evidence object."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a canonical object")
    if type(value) is dict:
        primitive = cast(dict[str, CanonicalValue], value)
    else:
        thawed = thaw_canonical(cast(FrozenCanonicalObject, value))
        if type(thawed) is not dict:
            raise TypeError(f"{field} must be a canonical object")
        primitive = thawed
    frozen = freeze_canonical(primitive)
    if not isinstance(frozen, Mapping):
        raise TypeError(f"{field} must be a canonical object")
    return frozen


def _require_exact_object_fields(
    value: object, fields: set[str], context: str
) -> dict[str, object]:
    """Require a concrete object with exactly the frozen operational schema fields."""
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    obj = cast(dict[str, object], value)
    actual = set(obj)
    missing = fields - actual
    unknown = actual - fields
    if missing:
        raise ValueError(f"{context} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{context} unknown fields: {sorted(unknown)}")
    return obj


def _array(value: object, field: str) -> list[object]:
    """Admit only a concrete JSON array for an operational artifact field."""
    if type(value) is not list:
        raise TypeError(f"{field} must be an array")
    return cast(list[object], value)


def _nonempty_string(value: object, field: str) -> str:
    """Admit a nonempty UTF-8 string for an operational artifact field."""
    if type(value) is not str:
        raise TypeError(f"{field} must be a string")
    value.encode("utf-8", errors="strict")
    if not value:
        raise ValueError(f"{field} must be nonempty")
    return value


def _schema_version(value: object, field: str) -> int:
    """Require an integer schema version while rejecting booleans."""
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer and not a boolean")
    return value


def _optional_nonempty_string(value: object, field: str) -> str | None:
    """Admit either ``None`` or a nonempty UTF-8 operational subject string."""
    if value is None:
        return None
    return _nonempty_string(value, field)


def _optional_hash(value: object, field: str) -> str | None:
    """Admit either ``None`` or a lowercase SHA-256 operational identity."""
    if value is None:
        return None
    return require_sha256(value, field)
