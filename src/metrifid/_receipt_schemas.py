"""Strict comparison receipt and validation schemas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import ClassVar, Self, cast

from ._closure_schemas import ComparisonInputsIdentity, ModelClosures
from ._contract_schemas import ComparisonContractIdentity, MonitoredJoint
from ._runtime_schemas import AlignmentSummary, EnvironmentIdentity, TimeContract, ToolIdentity
from ._schema_constants import (
    _METRICS_BY_JOINT_TYPE,
    _RECEIPT_SCHEMA,
    _RECEIPT_SCHEMA_VERSION,
    _STATUS_RULE_SCHEMA,
    _STATUS_RULE_SCHEMA_VERSION,
)
from ._schema_primitives import (
    _exact_int,
    _fields,
    _nonempty_string,
    _object,
    _require_instance,
    _require_typed_tuple,
    _sequence,
    _string,
)
from .errors import (
    ComparisonStatus,
    EngineThreadpoolState,
    LimitationCode,
    ReasonCode,
    ReasonRecord,
    canonical_limitations,
    derive_comparison_status,
    ordered_reasons,
    projected_reason_codes,
)
from .json_values import (
    CanonicalValue,
    FrozenCanonicalObject,
    canonical_sha256,
    freeze_canonical,
    require_sha256,
    thaw_canonical,
    validate_self_hash,
)


@dataclass(frozen=True, slots=True)
class CanonicalSummary:
    """A strictly canonical immutable object for a later phase's bounded evidence summary."""

    value: FrozenCanonicalObject

    def __post_init__(self) -> None:
        """Require an object-valued summary and recursively freeze its canonical values."""
        if not isinstance(self.value, Mapping):
            raise TypeError("summary must be an object")
        thawed = {key: thaw_canonical(item) for key, item in self.value.items()}
        frozen = freeze_canonical(cast(CanonicalValue, thawed))
        object.__setattr__(self, "value", cast(FrozenCanonicalObject, frozen))

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode and immutably freeze one canonical bounded-evidence object."""
        obj = _object(value, cls.__name__)
        frozen = freeze_canonical(cast(CanonicalValue, obj))
        return cls(cast(FrozenCanonicalObject, frozen))

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return a fresh mutable canonical object for receipt serialization."""
        return cast(dict[str, CanonicalValue], thaw_canonical(self.value))


class RepeatabilitySummary(CanonicalSummary):
    """Strict canonical repeatability evidence envelope."""


class NumericalEvidenceSummary(CanonicalSummary):
    """Strict canonical numerical evidence envelope."""


class MetricEvidenceSummary(CanonicalSummary):
    """Strict canonical metric evidence envelope."""


def _validate_receipt_envelope(receipt: ComparisonReceipt) -> None:
    """Validate receipt schemas and required nested object types."""
    if receipt.schema != _RECEIPT_SCHEMA:
        raise ValueError("invalid receipt schema")
    if type(receipt.schema_version) is not int or receipt.schema_version != _RECEIPT_SCHEMA_VERSION:
        raise ValueError("invalid receipt schema_version")
    if receipt.status_rule_schema != _STATUS_RULE_SCHEMA:
        raise ValueError("invalid status_rule_schema")
    if (
        type(receipt.status_rule_schema_version) is not int
        or receipt.status_rule_schema_version != _STATUS_RULE_SCHEMA_VERSION
    ):
        raise ValueError("invalid status_rule_schema_version")
    _require_instance(receipt.tool, ToolIdentity, "tool")
    if not isinstance(receipt.status, ComparisonStatus):
        raise TypeError("status must be a ComparisonStatus")
    _require_typed_tuple(receipt.reason_codes, ReasonCode, "reason_codes")
    _require_typed_tuple(receipt.reasons, ReasonRecord, "reasons")
    ordered_reasons(receipt.reasons)


def _validate_receipt_identities(receipt: ComparisonReceipt) -> None:
    """Validate embedded identities and monitored-joint ordering."""
    _require_instance(receipt.environment, EnvironmentIdentity, "environment")
    _require_instance(receipt.inputs, ComparisonInputsIdentity, "inputs")
    _require_instance(
        receipt.comparison_contract, ComparisonContractIdentity, "comparison_contract"
    )
    _require_instance(receipt.model_closures, ModelClosures, "model_closures")
    _require_instance(receipt.time, TimeContract, "time")
    _require_instance(receipt.alignment, AlignmentSummary, "alignment")
    _require_typed_tuple(receipt.monitored_joints, MonitoredJoint, "monitored_joints")
    if not receipt.monitored_joints:
        raise ValueError("monitored_joints must be nonempty")
    names = tuple(joint.canonical_name for joint in receipt.monitored_joints)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError("monitored_joints must be uniquely ordered by canonical_name")


def _validate_receipt_evidence(receipt: ComparisonReceipt) -> None:
    """Validate canonical evidence summary types."""
    _require_instance(receipt.tolerances, CanonicalSummary, "tolerances")
    _require_instance(receipt.repeatability, RepeatabilitySummary, "repeatability")
    _require_instance(receipt.numerical_evidence, NumericalEvidenceSummary, "numerical_evidence")
    _require_instance(receipt.metrics, MetricEvidenceSummary, "metrics")
    if receipt.first_crossing is not None:
        _require_instance(receipt.first_crossing, CanonicalSummary, "first_crossing")


@dataclass(frozen=True, slots=True)
class ComparisonReceipt:
    """The complete minimum comparison receipt envelope."""

    schema: str
    schema_version: int
    status_rule_schema: str
    status_rule_schema_version: int
    tool: ToolIdentity
    status: ComparisonStatus
    reason_codes: tuple[ReasonCode, ...]
    reasons: tuple[ReasonRecord, ...]
    environment: EnvironmentIdentity
    inputs: ComparisonInputsIdentity
    comparison_contract: ComparisonContractIdentity
    model_closures: ModelClosures
    time: TimeContract
    alignment: AlignmentSummary
    monitored_joints: tuple[MonitoredJoint, ...]
    tolerances: CanonicalSummary
    repeatability: RepeatabilitySummary
    numerical_evidence: NumericalEvidenceSummary
    metrics: MetricEvidenceSummary
    first_crossing: CanonicalSummary | None
    limitations: tuple[LimitationCode, ...]
    receipt_sha256: str | None

    _HASH_FIELD: ClassVar[str] = "receipt_sha256"

    def __post_init__(self) -> None:
        """Validate envelope types, embedded identities, evidence, limitations, and bindings."""
        _validate_receipt_envelope(self)
        _validate_receipt_identities(self)
        _validate_receipt_evidence(self)
        limitations = canonical_limitations(self.limitations)
        object.__setattr__(self, "limitations", limitations)
        _validate_tolerance_binding(self)
        if self.receipt_sha256 is not None:
            require_sha256(self.receipt_sha256, self._HASH_FIELD)
            _validate_receipt_semantics(self, require_hash=True)

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode and fully validate the exact comparison-receipt field set."""
        obj = _object(value, "ComparisonReceipt")
        _fields(obj, set(_COMPARISON_RECEIPT_FIELDS), "ComparisonReceipt")
        receipt = _comparison_receipt_from_object(cls, obj)
        return cast(Self, validate_receipt(receipt))

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit a completed receipt with every nested identity and its required self-hash."""
        if self.receipt_sha256 is None:
            raise ValueError("receipt_sha256 is required before serialization")
        return {
            **_receipt_unhashed_primitive(self),
            "receipt_sha256": self.receipt_sha256,
        }


_COMPARISON_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "status_rule_schema",
        "status_rule_schema_version",
        "tool",
        "status",
        "reason_codes",
        "reasons",
        "environment",
        "inputs",
        "comparison_contract",
        "model_closures",
        "time",
        "alignment",
        "monitored_joints",
        "tolerances",
        "repeatability",
        "numerical_evidence",
        "metrics",
        "first_crossing",
        "limitations",
        "receipt_sha256",
    }
)


def _comparison_receipt_from_object(
    receipt_type: type[ComparisonReceipt], obj: dict[str, object]
) -> ComparisonReceipt:
    """Parse one exact receipt object into its complete typed contract."""
    return receipt_type(
        _string(obj["schema"], "schema"),
        _exact_int(obj["schema_version"], "schema_version"),
        _string(obj["status_rule_schema"], "status_rule_schema"),
        _exact_int(obj["status_rule_schema_version"], "status_rule_schema_version"),
        ToolIdentity.from_primitive(obj["tool"]),
        _comparison_status(obj["status"]),
        _reason_code_values(obj["reason_codes"]),
        tuple(ReasonRecord.from_primitive(item) for item in _sequence(obj["reasons"], "reasons")),
        EnvironmentIdentity.from_primitive(obj["environment"]),
        ComparisonInputsIdentity.from_primitive(obj["inputs"]),
        ComparisonContractIdentity.from_primitive(obj["comparison_contract"]),
        ModelClosures.from_primitive(obj["model_closures"]),
        TimeContract.from_primitive(obj["time"]),
        AlignmentSummary.from_primitive(obj["alignment"]),
        tuple(
            MonitoredJoint.from_primitive(item)
            for item in _sequence(obj["monitored_joints"], "monitored_joints")
        ),
        CanonicalSummary.from_primitive(obj["tolerances"]),
        RepeatabilitySummary.from_primitive(obj["repeatability"]),
        NumericalEvidenceSummary.from_primitive(obj["numerical_evidence"]),
        MetricEvidenceSummary.from_primitive(obj["metrics"]),
        _optional_summary(obj["first_crossing"]),
        tuple(
            LimitationCode(_nonempty_string(item, "limitation"))
            for item in _sequence(obj["limitations"], "limitations")
        ),
        require_sha256(obj["receipt_sha256"], "receipt_sha256"),
    )


def _comparison_status(value: object) -> ComparisonStatus:
    """Parse one comparison status with the contract's closed registry error."""
    token = _string(value, "status")
    try:
        return ComparisonStatus(token)
    except ValueError as exc:
        raise ValueError(f"unknown comparison status: {token}") from exc


def _reason_code_values(value: object) -> tuple[ReasonCode, ...]:
    """Parse the ordered reason-code projection under the closed registry."""
    values: list[ReasonCode] = []
    for raw in _sequence(value, "reason_codes"):
        token = _string(raw, "reason_code")
        try:
            values.append(ReasonCode(token))
        except ValueError as exc:
            raise ValueError(f"unknown reason code: {token}") from exc
    return tuple(values)


def _optional_summary(value: object) -> CanonicalSummary | None:
    """Parse an optional canonical summary object."""
    return None if value is None else CanonicalSummary.from_primitive(value)


def _receipt_unhashed_primitive(receipt: ComparisonReceipt) -> dict[str, CanonicalValue]:
    """Assemble every decision-bearing receipt field except ``receipt_sha256``."""
    return {
        "schema": receipt.schema,
        "schema_version": receipt.schema_version,
        "status_rule_schema": receipt.status_rule_schema,
        "status_rule_schema_version": receipt.status_rule_schema_version,
        "tool": receipt.tool.to_primitive(),
        "status": receipt.status.value,
        "reason_codes": [code.value for code in receipt.reason_codes],
        "reasons": [reason.to_primitive() for reason in receipt.reasons],
        "environment": receipt.environment.to_primitive(),
        "inputs": receipt.inputs.to_primitive(),
        "comparison_contract": receipt.comparison_contract.to_primitive(),
        "model_closures": receipt.model_closures.to_primitive(),
        "time": receipt.time.to_primitive(),
        "alignment": receipt.alignment.to_primitive(),
        "monitored_joints": [joint.to_primitive() for joint in receipt.monitored_joints],
        "tolerances": receipt.tolerances.to_primitive(),
        "repeatability": receipt.repeatability.to_primitive(),
        "numerical_evidence": receipt.numerical_evidence.to_primitive(),
        "metrics": receipt.metrics.to_primitive(),
        "first_crossing": (
            None if receipt.first_crossing is None else receipt.first_crossing.to_primitive()
        ),
        "limitations": [value.value for value in receipt.limitations],
    }


def finalize_receipt(receipt: ComparisonReceipt) -> ComparisonReceipt:
    """Return a new, deterministically ordered and self-hashed receipt."""
    if not isinstance(receipt, ComparisonReceipt):
        raise TypeError("receipt must be a ComparisonReceipt")
    if receipt.receipt_sha256 is not None:
        raise ValueError("finalize_receipt requires an unhashed receipt candidate")
    finalized_environment = receipt.environment.finalized()
    finalized_alignment = receipt.alignment.finalized()
    reasons = ordered_reasons(receipt.reasons)
    reason_codes = projected_reason_codes(reasons)
    candidate = replace(
        receipt,
        environment=finalized_environment,
        alignment=finalized_alignment,
        reasons=reasons,
        reason_codes=reason_codes,
    )
    _validate_receipt_semantics(candidate, require_hash=False)
    digest = canonical_sha256(_receipt_unhashed_primitive(candidate))
    finalized = replace(candidate, receipt_sha256=digest)
    return validate_receipt(finalized)


def validate_receipt(receipt: ComparisonReceipt) -> ComparisonReceipt:
    """Validate one receipt without reordering, repairing, or mutating it."""
    if not isinstance(receipt, ComparisonReceipt):
        raise TypeError("receipt must be a ComparisonReceipt")
    _validate_receipt_semantics(receipt, require_hash=True)
    return receipt


def _validate_receipt_semantics(receipt: ComparisonReceipt, *, require_hash: bool) -> None:
    """Cross-check reason ordering, status, identities, evidence bindings, and optional hash."""
    if receipt.reasons != ordered_reasons(receipt.reasons):
        raise ValueError("reasons are not in frozen deterministic order")
    expected_codes = projected_reason_codes(receipt.reasons)
    if receipt.reason_codes != expected_codes:
        raise ValueError("reason_codes is not the exact first-occurrence projection")
    if receipt.limitations != tuple(LimitationCode):
        raise ValueError("limitations are not the exact canonical registry")
    if receipt.status is ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD:
        if receipt.reasons or receipt.reason_codes:
            raise ValueError("green receipt must not contain reasons")
    elif not receipt.reasons:
        raise ValueError("every non-green receipt requires at least one reason")
    expected_status = derive_comparison_status(receipt.reasons)
    if receipt.status != expected_status:
        raise ValueError("status is inconsistent with the frozen precedence table")
    receipt.environment.validate_hash()
    receipt.alignment.validate_hash()
    _validate_comparison_contract_binding(receipt)
    _validate_reason_semantics(receipt)
    _validate_tolerance_binding(receipt)
    _validate_threadpool_reason_binding(receipt)
    if require_hash:
        if receipt.receipt_sha256 is None:
            raise ValueError("receipt_sha256 is required")
        validate_self_hash(receipt.to_primitive(), ComparisonReceipt._HASH_FIELD)


def _validate_tolerance_binding(receipt: ComparisonReceipt) -> None:
    """Require the tolerance summary to exactly restate every monitored joint."""
    expected_tolerances = {
        joint.canonical_name: {
            "joint_type": joint.joint_type,
            **{
                metric: joint.tolerances[metric].to_primitive()
                for metric in _METRICS_BY_JOINT_TYPE[joint.joint_type]
            },
        }
        for joint in receipt.monitored_joints
    }
    if receipt.tolerances.to_primitive() != expected_tolerances:
        raise ValueError("receipt tolerances do not match monitored_joints")


def _validate_comparison_contract_binding(receipt: ComparisonReceipt) -> None:
    """Bind the embedded contract digest to closures, workload, time, and monitoring."""
    contract = receipt.comparison_contract
    if receipt.inputs.comparison_contract_sha256 != contract.sha256():
        raise ValueError("comparison_contract_sha256 does not match comparison_contract")
    _validate_closure_bindings(receipt)
    _validate_input_bindings(receipt)
    _validate_time_and_monitoring_bindings(receipt)


def _validate_closure_bindings(receipt: ComparisonReceipt) -> None:
    """Validate embedded baseline and candidate closure hashes."""
    contract = receipt.comparison_contract

    baseline_closure_sha256 = receipt.model_closures.baseline.sha256()
    if receipt.inputs.baseline_model_closure_sha256 != baseline_closure_sha256:
        raise ValueError("baseline model closure hash does not match embedded identity")
    if contract.baseline_model_closure_sha256 != baseline_closure_sha256:
        raise ValueError("comparison_contract baseline model closure hash does not match")

    candidate_closure_sha256 = receipt.model_closures.candidate.sha256()
    if receipt.inputs.candidate_model_closure_sha256 != candidate_closure_sha256:
        raise ValueError("candidate model closure hash does not match embedded identity")
    if contract.candidate_model_closure_sha256 != candidate_closure_sha256:
        raise ValueError("comparison_contract candidate model closure hash does not match")


def _validate_input_bindings(receipt: ComparisonReceipt) -> None:
    """Validate workload and alias hashes against the comparison contract."""
    contract = receipt.comparison_contract
    if receipt.inputs.initial_state_semantic_sha256 != contract.initial_state_semantic_sha256:
        raise ValueError("initial_state_semantic_sha256 does not match comparison_contract")
    if receipt.inputs.actions_semantic_sha256 != contract.actions_semantic_sha256:
        raise ValueError("actions_semantic_sha256 does not match comparison_contract")
    if receipt.inputs.aliases_semantic_sha256 != contract.aliases_semantic_sha256:
        raise ValueError("aliases_semantic_sha256 does not match comparison_contract")


def _validate_time_and_monitoring_bindings(receipt: ComparisonReceipt) -> None:
    """Validate time and monitored-joint bindings."""
    contract = receipt.comparison_contract
    for field_name in ("baseline_step_dt", "candidate_step_dt", "control_dt"):
        if getattr(receipt.time, field_name) != getattr(contract, field_name):
            raise ValueError(f"time.{field_name} does not match comparison_contract")

    if receipt.monitored_joints != contract.monitored_joints:
        raise ValueError("receipt monitored_joints do not match comparison_contract")
    aligned_names = set(receipt.alignment.joint_order)
    missing = [
        joint.canonical_name
        for joint in receipt.monitored_joints
        if joint.canonical_name not in aligned_names
    ]
    if missing:
        raise ValueError(f"monitored joints are absent from alignment: {missing}")


def _validate_reason_semantics(receipt: ComparisonReceipt) -> None:
    """Require metric-tolerance reasons to name a valid monitored joint metric."""
    monitored = {joint.canonical_name: joint for joint in receipt.monitored_joints}
    for reason in receipt.reasons:
        if reason.code is not ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED:
            continue
        if reason.object_name not in monitored:
            raise ValueError(
                "JOINT_METRIC_TOLERANCE_EXCEEDED object_name must be a monitored joint"
            )
        joint = monitored[reason.object_name]
        if reason.metric not in _METRICS_BY_JOINT_TYPE[joint.joint_type]:
            raise ValueError(
                "JOINT_METRIC_TOLERANCE_EXCEEDED metric is invalid for the monitored joint type"
            )


def _validate_threadpool_reason_binding(receipt: ComparisonReceipt) -> None:
    """Match threadpool refusal reasons exactly to the observed engine state."""
    codes = set(receipt.reason_codes)
    active = ReasonCode.ENGINE_THREADPOOL_ACTIVE in codes
    unknown = ReasonCode.ENGINE_THREADPOOL_STATE_UNKNOWN in codes
    state = receipt.environment.engine_threadpool_state
    if state is EngineThreadpoolState.DISABLED and (active or unknown):
        raise ValueError("disabled threadpool state must not carry a threadpool refusal reason")
    if state is EngineThreadpoolState.ACTIVE and (not active or unknown):
        raise ValueError("active threadpool state requires ENGINE_THREADPOOL_ACTIVE only")
    if state is EngineThreadpoolState.UNKNOWN and (not unknown or active):
        raise ValueError("unknown threadpool state requires ENGINE_THREADPOOL_STATE_UNKNOWN only")
