"""Full-horizon product mapping over the selected private runtime evaluator."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Final, cast

from .._native_upgrade import CaseEvidence, evaluate_case
from ..json_values import CanonicalValue
from ._evidence import AdmittedRuntimeEvidence
from ._status import RuntimeReviewReasonCode, RuntimeReviewStatus
from ._witness import (
    StableWitness,
    first_decisive_witness,
    stable_witnesses,
    stable_worst_witness,
)

_WITHIN: Final = RuntimeReviewStatus.WITHIN_DECLARED_MIGRATION_ENVELOPE.value
_UNRESOLVED: Final = RuntimeReviewStatus.UNRESOLVED_NEAR_BOUNDARY.value
_OUTSIDE: Final = RuntimeReviewStatus.OUTSIDE_DECLARED_MIGRATION_ENVELOPE.value
_REASONS: Final = frozenset(item.value for item in RuntimeReviewReasonCode)


@dataclass(frozen=True, slots=True)
class RuntimeReviewDecision:
    """One typed full-horizon decision and its stable public witness projection."""

    status: RuntimeReviewStatus
    reason_code: RuntimeReviewReasonCode | None
    admitted_prefix: str
    first_failing_gate: dict[str, CanonicalValue] | None
    first_decisive_witness: StableWitness | None
    worst_witness: StableWitness | None
    witness_counts: dict[str, CanonicalValue]
    witnesses: tuple[StableWitness, ...]
    private_result: Mapping[str, object]

    def __post_init__(self) -> None:
        """Enforce reason/status and first-decisive-witness invariants."""
        if (self.status is RuntimeReviewStatus.INSUFFICIENT_EVIDENCE) != (
            self.reason_code is not None
        ):
            raise ValueError("only INSUFFICIENT_EVIDENCE carries a reason_code")
        expected = {
            _WITHIN,
            _UNRESOLVED,
            _OUTSIDE,
        }
        if set(self.witness_counts) != expected:
            raise ValueError("witness_counts must contain the exact three classifications")
        if (
            self.status
            in {
                RuntimeReviewStatus.WITHIN_DECLARED_MIGRATION_ENVELOPE,
                RuntimeReviewStatus.INSUFFICIENT_EVIDENCE,
            }
            and self.first_decisive_witness is not None
        ):
            raise ValueError("WITHIN and INSUFFICIENT decisions have no first decisive witness")

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return the canonical product-decision fields used by receipt construction."""
        return {
            "status": self.status.value,
            "reason_code": None if self.reason_code is None else self.reason_code.value,
            "admitted_prefix": self.admitted_prefix,
            "first_failing_gate": self.first_failing_gate,
            "first_decisive_witness": (
                None
                if self.first_decisive_witness is None
                else self.first_decisive_witness.to_primitive()
            ),
            "worst_witness": (
                None if self.worst_witness is None else self.worst_witness.to_primitive()
            ),
            "witness_counts": dict(self.witness_counts),
            "witnesses": [item.to_primitive() for item in self.witnesses],
        }


def evaluate_runtime_evidence(evidence: AdmittedRuntimeEvidence) -> RuntimeReviewDecision:
    """Evaluate admitted raw evidence and apply the stricter full-horizon product rule."""
    if not isinstance(evidence, AdmittedRuntimeEvidence):
        raise TypeError("evidence must be AdmittedRuntimeEvidence")
    return evaluate_runtime_case(evidence.case_evidence)


def evaluate_runtime_case(case: CaseEvidence) -> RuntimeReviewDecision:
    """Apply product mapping to one reconstructed or synthetic private evidence case."""
    if not isinstance(case, CaseEvidence):
        raise TypeError("case must be CaseEvidence")
    private = evaluate_case(case)
    rows = _private_rows(private.get("witnesses"))
    witnesses = stable_witnesses(case, rows)
    private_status = _private_string(private, "status")
    admitted_prefix_value = _private_number(private, "admitted_prefix")
    first_gate = _stable_gate(private.get("first_failing_gate"))
    unqualified_suffix = private.get("unqualified_suffix")
    status, reason = _map_product_status(
        private_status,
        admitted_prefix_value,
        case.horizon,
        first_gate,
        unqualified_suffix,
        witnesses,
    )
    counts: dict[str, CanonicalValue] = {
        classification: sum(item.classification == classification for item in witnesses)
        for classification in (_WITHIN, _UNRESOLVED, _OUTSIDE)
    }
    decisive = first_decisive_witness(status, witnesses)
    if (
        status
        in {
            RuntimeReviewStatus.OUTSIDE_DECLARED_MIGRATION_ENVELOPE,
            RuntimeReviewStatus.UNRESOLVED_NEAR_BOUNDARY,
        }
        and decisive is None
    ):
        raise ValueError("completed decisive status lacks its proving witness")
    return RuntimeReviewDecision(
        status=status,
        reason_code=reason,
        admitted_prefix=_decimal_token(admitted_prefix_value),
        first_failing_gate=first_gate,
        first_decisive_witness=decisive,
        worst_witness=stable_worst_witness(private.get("worst_witness"), witnesses),
        witness_counts=counts,
        witnesses=witnesses,
        private_result=private,
    )


def _map_product_status(
    private_status: str,
    admitted_prefix: float,
    horizon: float,
    first_gate: dict[str, CanonicalValue] | None,
    unqualified_suffix: object,
    witnesses: tuple[StableWitness, ...],
) -> tuple[RuntimeReviewStatus, RuntimeReviewReasonCode | None]:
    """Apply exact OUTSIDE-first then full-horizon completed-decision precedence."""
    if any(item.classification == _OUTSIDE for item in witnesses):
        return RuntimeReviewStatus.OUTSIDE_DECLARED_MIGRATION_ENVELOPE, None
    if private_status == RuntimeReviewReasonCode.PREFIX_TOO_SHORT.value:
        return (
            RuntimeReviewStatus.INSUFFICIENT_EVIDENCE,
            RuntimeReviewReasonCode.PREFIX_TOO_SHORT,
        )
    incomplete = (
        admitted_prefix != horizon or first_gate is not None or unqualified_suffix is not None
    )
    if incomplete:
        gate_status = None if first_gate is None else first_gate.get("status")
        reason_token = (
            gate_status
            if isinstance(gate_status, str) and gate_status in _REASONS
            else (RuntimeReviewReasonCode.PREFIX_TOO_SHORT.value)
        )
        return RuntimeReviewStatus.INSUFFICIENT_EVIDENCE, RuntimeReviewReasonCode(reason_token)
    if any(item.classification == _UNRESOLVED for item in witnesses):
        return RuntimeReviewStatus.UNRESOLVED_NEAR_BOUNDARY, None
    if witnesses and all(item.classification == _WITHIN for item in witnesses):
        return RuntimeReviewStatus.WITHIN_DECLARED_MIGRATION_ENVELOPE, None
    raise ValueError(f"private evaluator result cannot produce a public decision: {private_status}")


def _stable_gate(value: object) -> dict[str, CanonicalValue] | None:
    """Convert a private gate row into stable exact-decimal canonical fields."""
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("private first_failing_gate must be an object or null")
    gate = cast(dict[str, object], value)
    expected = {"channel_id", "detail", "status", "time"}
    if set(gate) != expected:
        raise ValueError("private first_failing_gate has unexpected fields")
    return {
        "channel_id": _private_nonempty_string(gate["channel_id"], "gate channel_id"),
        "detail": _private_nonempty_string(gate["detail"], "gate detail"),
        "status": _private_nonempty_string(gate["status"], "gate status"),
        "time": _decimal_token(_number(gate["time"], "gate time")),
    }


def _private_rows(value: object) -> list[dict[str, object]]:
    """Admit the private evaluator's bounded witness-row sequence."""
    if type(value) is not list:
        raise ValueError("private witnesses must be an array")
    rows = cast(list[object], value)
    if any(type(item) is not dict for item in rows):
        raise ValueError("private witness rows must be objects")
    return cast(list[dict[str, object]], rows)


def _private_string(value: Mapping[str, object], field: str) -> str:
    """Read one required nonempty private-result string."""
    return _private_nonempty_string(value.get(field), field)


def _private_nonempty_string(value: object, field: str) -> str:
    """Require one nonempty string from private evaluator output."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"private {field} must be a nonempty string")
    return value


def _private_number(value: Mapping[str, object], field: str) -> float:
    """Read one required finite private-result number."""
    return _number(value.get(field), field)


def _number(value: object, field: str) -> float:
    """Require a private result's finite binary64 number."""
    if type(value) not in {int, float}:
        raise ValueError(f"private {field} must be a number")
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        raise ValueError(f"private {field} must be finite")
    return number


def _decimal_token(value: float) -> str:
    """Render one finite private binary64 as a canonical ordinary decimal token."""
    decimal = Decimal(str(value))
    if not decimal.is_finite():
        raise ValueError("decision decimal token must be finite")
    if decimal == 0:
        return "0"
    token = format(decimal, "f")
    return token.rstrip("0").rstrip(".") if "." in token else token


__all__ = ["RuntimeReviewDecision", "evaluate_runtime_case", "evaluate_runtime_evidence"]
