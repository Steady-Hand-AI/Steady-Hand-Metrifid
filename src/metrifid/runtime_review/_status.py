"""Frozen completed statuses, reasons, and exit codes for Native Runtime Review."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from types import MappingProxyType
from typing import Final


class RuntimeReviewStatus(StrEnum):
    """The complete registry of completed runtime-review decisions."""

    WITHIN_DECLARED_MIGRATION_ENVELOPE = "WITHIN_DECLARED_MIGRATION_ENVELOPE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNRESOLVED_NEAR_BOUNDARY = "UNRESOLVED_NEAR_BOUNDARY"
    OUTSIDE_DECLARED_MIGRATION_ENVELOPE = "OUTSIDE_DECLARED_MIGRATION_ENVELOPE"


class RuntimeReviewReasonCode(StrEnum):
    """The complete registry of completed insufficient-evidence reasons."""

    REPEATABILITY_FAILED = "REPEATABILITY_FAILED"
    SOLVER_NOT_CONVERGED = "SOLVER_NOT_CONVERGED"
    CONTACT_EVENT_TOPOLOGY_CHANGED = "CONTACT_EVENT_TOPOLOGY_CHANGED"
    NON_ASYMPTOTIC_REGIME = "NON_ASYMPTOTIC_REGIME"
    PREFIX_TOO_SHORT = "PREFIX_TOO_SHORT"


class RuntimeReviewExitCode(IntEnum):
    """Process exit codes for the four completed runtime-review decisions."""

    WITHIN_DECLARED_MIGRATION_ENVELOPE = 0
    INSUFFICIENT_EVIDENCE = 20
    UNRESOLVED_NEAR_BOUNDARY = 30
    OUTSIDE_DECLARED_MIGRATION_ENVELOPE = 40


_EXIT_CODES: Final[MappingProxyType[RuntimeReviewStatus, RuntimeReviewExitCode]] = MappingProxyType(
    {
        RuntimeReviewStatus.WITHIN_DECLARED_MIGRATION_ENVELOPE: (
            RuntimeReviewExitCode.WITHIN_DECLARED_MIGRATION_ENVELOPE
        ),
        RuntimeReviewStatus.INSUFFICIENT_EVIDENCE: (RuntimeReviewExitCode.INSUFFICIENT_EVIDENCE),
        RuntimeReviewStatus.UNRESOLVED_NEAR_BOUNDARY: (
            RuntimeReviewExitCode.UNRESOLVED_NEAR_BOUNDARY
        ),
        RuntimeReviewStatus.OUTSIDE_DECLARED_MIGRATION_ENVELOPE: (
            RuntimeReviewExitCode.OUTSIDE_DECLARED_MIGRATION_ENVELOPE
        ),
    }
)


def runtime_review_exit_code(status: RuntimeReviewStatus) -> RuntimeReviewExitCode:
    """Map one completed runtime-review status to its frozen process exit code."""
    if not isinstance(status, RuntimeReviewStatus):
        raise TypeError("status must be a RuntimeReviewStatus")
    return _EXIT_CODES[status]
