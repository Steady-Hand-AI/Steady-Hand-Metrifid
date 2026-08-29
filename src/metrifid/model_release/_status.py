"""Model Change Gate completed statuses and their frozen exit mapping."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType


class ModelReleaseStatus(StrEnum):
    """The four completed outcomes owned by ``review-model``."""

    NO_COMPILED_CHANGE = "NO_COMPILED_CHANGE"
    WITHIN_DECLARED_POLICY = "WITHIN_DECLARED_POLICY"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    OUTSIDE_DECLARED_POLICY = "OUTSIDE_DECLARED_POLICY"


MODEL_RELEASE_COMPLETED_EXIT_CODES: Mapping[ModelReleaseStatus, int] = MappingProxyType(
    {
        ModelReleaseStatus.NO_COMPILED_CHANGE: 0,
        ModelReleaseStatus.WITHIN_DECLARED_POLICY: 0,
        ModelReleaseStatus.REVIEW_REQUIRED: 40,
        ModelReleaseStatus.OUTSIDE_DECLARED_POLICY: 40,
    }
)


def model_release_exit_code(status: ModelReleaseStatus) -> int:
    """Return the completed process exit code owned by ``review-model``."""
    if not isinstance(status, ModelReleaseStatus):
        raise TypeError("status must be a ModelReleaseStatus")
    return MODEL_RELEASE_COMPLETED_EXIT_CODES[status]


__all__ = [
    "MODEL_RELEASE_COMPLETED_EXIT_CODES",
    "ModelReleaseStatus",
    "model_release_exit_code",
]
