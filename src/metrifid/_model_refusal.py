"""Typed operational refusals shared by model admission responsibilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypeAlias, cast

from .json_values import CanonicalValue, FrozenCanonicalObject, freeze_canonical, thaw_canonical
from .operational import OperationalReasonCode

ModelRole: TypeAlias = Literal["baseline", "candidate", "comparison"]


class ModelAdmissionRefusal(Exception):
    """One expected model admission refusal with a frozen reason, role, and evidence object."""

    __slots__ = ("evidence", "reason", "role")
    evidence: FrozenCanonicalObject
    reason: OperationalReasonCode
    role: ModelRole

    def __init__(
        self,
        reason: OperationalReasonCode,
        role: ModelRole,
        evidence: Mapping[str, CanonicalValue] | None = None,
    ) -> None:
        """Freeze validated refusal inputs into deterministic operational evidence."""
        if role not in {"baseline", "candidate", "comparison"}:
            raise ValueError("invalid model admission refusal role")
        frozen = freeze_canonical(cast(CanonicalValue, dict(evidence or {})))
        if not isinstance(frozen, Mapping):
            raise TypeError("model admission refusal evidence must be an object")
        self.evidence = frozen
        self.reason = reason
        self.role = role
        super().__init__(f"{reason.value}:{role}")

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return the deterministic internal refusal representation."""
        return {
            "reason": self.reason.value,
            "role": self.role,
            "evidence": thaw_canonical(self.evidence),
        }


def refuse(
    reason: OperationalReasonCode,
    role: ModelRole,
    **evidence: CanonicalValue,
) -> ModelAdmissionRefusal:
    """Construct one expected model-admission refusal without raising it."""
    return ModelAdmissionRefusal(reason, role, evidence)


__all__ = ["ModelAdmissionRefusal", "ModelRole", "refuse"]
