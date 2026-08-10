"""Certify-owned completed status and completed exit mapping.

Certify answers a different question from Compare, so it owns its own status type and its own
completed exit code. Neither value is added to ``ComparisonStatus`` and neither reuses
``OperationalExitCode``, which stays reserved for refusals (64) and internal failures (70).
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType


class CertifyStatus(StrEnum):
    """The only two completed Certify outcomes."""

    CERTIFIED_COMPILED_EQUIVALENCE = "CERTIFIED_COMPILED_EQUIVALENCE"
    NOT_CERTIFIED_COMPILED_DIFFERS = "NOT_CERTIFIED_COMPILED_DIFFERS"


CERTIFY_COMPLETED_EXIT_CODES: Mapping[CertifyStatus, int] = MappingProxyType(
    {
        CertifyStatus.CERTIFIED_COMPILED_EQUIVALENCE: 0,
        CertifyStatus.NOT_CERTIFIED_COMPILED_DIFFERS: 40,
    }
)


def certify_exit_code(status: CertifyStatus) -> int:
    """Return the completed process exit code Certify owns for this status."""
    if not isinstance(status, CertifyStatus):
        raise TypeError("status must be a CertifyStatus")
    return CERTIFY_COMPLETED_EXIT_CODES[status]


__all__ = ["CERTIFY_COMPLETED_EXIT_CODES", "CertifyStatus", "certify_exit_code"]
