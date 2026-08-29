"""Frozen qualification statuses, exit codes, and claim limitations.

The four completed statuses answer one question: do the selected workloads detect every declared
perturbation at or above its required magnitude. They are deliberately distinct from the comparison
statuses they are built from, and from the research statuses that preceded this product, which are
not published anywhere in this package.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from types import MappingProxyType
from typing import Final


class QualificationStatus(StrEnum):
    """The complete workload-qualification status registry."""

    QUALIFIED_FOR_DECLARED_PROBES = "QUALIFIED_FOR_DECLARED_PROBES"
    PARTIALLY_QUALIFIED = "PARTIALLY_QUALIFIED"
    INSUFFICIENT_EXCITATION = "INSUFFICIENT_EXCITATION"
    UNRESOLVED = "UNRESOLVED"


class ProbeGroupStatus(StrEnum):
    """Per-probe-group adjudication, decided from the ordered detection signature."""

    QUALIFIED = "QUALIFIED"
    UNRESOLVED = "UNRESOLVED"
    INSUFFICIENT = "INSUFFICIENT"


class CellOutcome(StrEnum):
    """One workload-versus-variant detection outcome, mapped from a completed comparison."""

    DETECTED = "DETECTED"
    NOT_DETECTED = "NOT_DETECTED"
    UNRESOLVED = "UNRESOLVED"


class QualificationExitCode(IntEnum):
    """Process exit codes for completed qualification statuses."""

    QUALIFIED_FOR_DECLARED_PROBES = 0
    PARTIALLY_QUALIFIED = 20
    INSUFFICIENT_EXCITATION = 20
    UNRESOLVED = 30


class QualificationLimitationCode(StrEnum):
    """What a completed qualification does not claim, in canonical order.

    This is a separate registry from the comparison limitation codes. A qualification carries claims
    the comparison surface has no vocabulary for, and the comparison registry is a frozen contract
    that this product must not extend.

    ``USER_DECLARED_PROBE_SEMANTICS_NOT_VERIFIED`` is the honest boundary on probe declarations:
    Metrifid compares the exact supplied admitted model closures and preserves the user's parameter,
    direction, magnitude, and magnitude-semantics labels. It does not independently establish that
    those labels faithfully describe the source edits or that no other source change exists.
    """

    DECLARED_PROBES_ONLY = "DECLARED_PROBES_ONLY"
    USER_DECLARED_PROBE_SEMANTICS_NOT_VERIFIED = "USER_DECLARED_PROBE_SEMANTICS_NOT_VERIFIED"
    DECLARED_WORKLOAD_CANDIDATES_ONLY = "DECLARED_WORKLOAD_CANDIDATES_ONLY"
    MONITORED_JOINT_COORDINATES_ONLY = "MONITORED_JOINT_COORDINATES_ONLY"
    NO_GLOBAL_EQUIVALENCE_CLAIM = "NO_GLOBAL_EQUIVALENCE_CLAIM"
    NO_TASK_SAFETY_OR_REAL_WORLD_TRANSFER_CLAIM = "NO_TASK_SAFETY_OR_REAL_WORLD_TRANSFER_CLAIM"


QUALIFICATION_LIMITATIONS: Final[tuple[QualificationLimitationCode, ...]] = tuple(
    QualificationLimitationCode
)

_EXIT_CODES: Final[MappingProxyType[QualificationStatus, QualificationExitCode]] = MappingProxyType(
    {
        QualificationStatus.QUALIFIED_FOR_DECLARED_PROBES: (
            QualificationExitCode.QUALIFIED_FOR_DECLARED_PROBES
        ),
        QualificationStatus.PARTIALLY_QUALIFIED: QualificationExitCode.PARTIALLY_QUALIFIED,
        QualificationStatus.INSUFFICIENT_EXCITATION: QualificationExitCode.INSUFFICIENT_EXCITATION,
        QualificationStatus.UNRESOLVED: QualificationExitCode.UNRESOLVED,
    }
)


def qualification_exit_code(status: QualificationStatus) -> QualificationExitCode:
    """Map every completed qualification status to its frozen process exit code."""
    if not isinstance(status, QualificationStatus):
        raise TypeError("status must be a QualificationStatus")
    return _EXIT_CODES[status]
