"""Public, lazily loaded API for exact native-runtime migration review.

Runtime Review answers one bounded question from already-produced evidence: whether one exact
candidate MuJoCo native profile may replace one exact baseline profile for the declared source
closure, workload, tolerances, and complete one-second horizon. Importing this package performs no
filesystem, subprocess, network, NumPy, or MuJoCo work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..compare._failure import ComparisonOperationError as RuntimeReviewOperationError
    from ._execution import (
        RuntimeReviewRunResult,
        run_runtime_review_configuration_file,
    )
    from ._receipt_validation import load_and_validate_runtime_review_receipt
    from ._run import RuntimeReviewResult, review_runtime_configuration_file
    from ._status import (
        RuntimeReviewExitCode,
        RuntimeReviewReasonCode,
        RuntimeReviewStatus,
        runtime_review_exit_code,
    )

_DEFERRED: dict[str, tuple[str, str]] = {
    "RuntimeReviewStatus": ("._status", "RuntimeReviewStatus"),
    "RuntimeReviewReasonCode": ("._status", "RuntimeReviewReasonCode"),
    "RuntimeReviewExitCode": ("._status", "RuntimeReviewExitCode"),
    "RuntimeReviewResult": ("._run", "RuntimeReviewResult"),
    "RuntimeReviewRunResult": ("._execution", "RuntimeReviewRunResult"),
    "RuntimeReviewOperationError": ("..compare._failure", "ComparisonOperationError"),
    "runtime_review_exit_code": ("._status", "runtime_review_exit_code"),
    "review_runtime_configuration_file": ("._run", "review_runtime_configuration_file"),
    "run_runtime_review_configuration_file": (
        "._execution",
        "run_runtime_review_configuration_file",
    ),
    "load_and_validate_runtime_review_receipt": (
        "._receipt_validation",
        "load_and_validate_runtime_review_receipt",
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve one supported name only when a caller explicitly requests it."""
    try:
        module_name, attribute = _DEFERRED[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module

    module = import_module(module_name, __package__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Advertise exactly the supported Runtime Review compatibility surface."""
    return sorted(__all__)


__all__ = [
    "RuntimeReviewStatus",
    "RuntimeReviewReasonCode",
    "RuntimeReviewExitCode",
    "RuntimeReviewResult",
    "RuntimeReviewRunResult",
    "RuntimeReviewOperationError",
    "runtime_review_exit_code",
    "review_runtime_configuration_file",
    "run_runtime_review_configuration_file",
    "load_and_validate_runtime_review_receipt",
]
