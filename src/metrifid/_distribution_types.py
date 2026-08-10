"""Value objects and errors for installed-distribution identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from .json_values import CanonicalValue
from .operational import (
    ExecutionIdentityState,
    OperationalFailure,
    OperationalReason,
    OperationalReasonCode,
    OperationalToolObservation,
)
from .version import __version__

# The exact operations that may own a distribution-identity failure. A refusal must name the
# command that actually refused, so this method takes the operation with no default.
OperationName: TypeAlias = Literal["compare", "certify", "audit-timestep"]


@dataclass(frozen=True, slots=True)
class _ManifestRecord:
    """One strict row from the installed wheel RECORD file."""

    path: str
    hash_mode: str | None
    hash_value: str | None
    size: int | None


@dataclass(frozen=True, slots=True)
class _InstalledManifest:
    """Strict in-root RECORD members plus untrusted out-of-root rows."""

    records: dict[str, _ManifestRecord]
    external_records: tuple[_ManifestRecord, ...]


class DistributionIdentityError(ValueError):
    """A refused execution identity with a canonical operational artifact."""

    def __init__(
        self,
        reason_code: OperationalReasonCode,
        message: str,
        *,
        field: str | None = None,
        evidence: dict[str, CanonicalValue] | None = None,
    ) -> None:
        """Capture a closed identity-failure reason and its optional manifest evidence."""
        super().__init__(message)
        self.reason_code = reason_code
        self.field = field
        self.evidence = {} if evidence is None else dict(evidence)

    def to_operational_failure(self, operation: OperationName) -> OperationalFailure:
        """Return the strict self-hashed pre-contract failure artifact for one operation.

        Args:
            operation: The command that actually refused. A distribution-identity failure raised
                under ``certify`` must be reported as ``certify``, not as ``compare``.
        """
        mismatch_reasons = {
            OperationalReasonCode.EXECUTING_CODE_NOT_INSTALLED_DISTRIBUTION,
            OperationalReasonCode.MIXED_METRIFID_MODULE_ROOTS,
        }
        state: ExecutionIdentityState = (
            "MISMATCH" if self.reason_code in mismatch_reasons else "UNBOUND"
        )
        return OperationalFailure(
            schema="metrifid.operational_failure",
            schema_version=1,
            failure_rule_schema="metrifid.operational_failure_rules",
            failure_rule_schema_version=1,
            tool=OperationalToolObservation(__version__, state, None),
            operation=operation,
            stage=self.reason_code.stage,
            reason=OperationalReason(
                code=self.reason_code,
                role=None,
                field=self.field,
                object_name=None,
                evidence=self.evidence,
            ),
            available_inputs=(),
            environment=None,
            exit_code=self.reason_code.exit_code,
            failure_sha256=None,
        ).finalized()
