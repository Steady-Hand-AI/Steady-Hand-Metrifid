"""Public, lazily native-loaded API for the static Model Change Gate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._policy import (
    ChangeKind,
    ModelReleasePolicy,
    PolicyEffect,
    PolicyObjectType,
    PolicyRule,
    PolicySelector,
    load_model_release_policy,
    parse_model_release_policy,
)
from ._status import (
    MODEL_RELEASE_COMPLETED_EXIT_CODES,
    ModelReleaseStatus,
    model_release_exit_code,
)

if TYPE_CHECKING:
    from ..compare._failure import ComparisonOperationError as ModelReleaseOperationError
    from ._receipt import (
        ModelReleaseResult,
        load_and_validate_model_release_receipt,
        validate_model_release_receipt,
    )
    from ._run import review_model_release

_DEFERRED: dict[str, tuple[str, str]] = {
    "ModelReleaseOperationError": ("..compare._failure", "ComparisonOperationError"),
    "ModelReleaseResult": ("._receipt", "ModelReleaseResult"),
    "review_model_release": ("._run", "review_model_release"),
    "load_and_validate_model_release_receipt": (
        "._receipt",
        "load_and_validate_model_release_receipt",
    ),
    "validate_model_release_receipt": ("._receipt", "validate_model_release_receipt"),
}


def __getattr__(name: str) -> Any:
    """Resolve native execution and receipt helpers only when explicitly requested."""
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
    """Expose the deliberate policy, status, receipt, and execution surface."""
    return sorted(set(globals()) | set(_DEFERRED))


__all__ = [
    "MODEL_RELEASE_COMPLETED_EXIT_CODES",
    "ChangeKind",
    "ModelReleaseOperationError",
    "ModelReleasePolicy",
    "ModelReleaseResult",
    "ModelReleaseStatus",
    "PolicyEffect",
    "PolicyObjectType",
    "PolicyRule",
    "PolicySelector",
    "load_and_validate_model_release_receipt",
    "load_model_release_policy",
    "model_release_exit_code",
    "parse_model_release_policy",
    "review_model_release",
    "validate_model_release_receipt",
]
