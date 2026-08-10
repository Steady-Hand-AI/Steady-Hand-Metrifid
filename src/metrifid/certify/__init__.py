"""Public surface for `metrifid certify`.

Importing this package must not import MuJoCo, NumPy or the closure machinery, so every heavy
name is resolved on first attribute access. Only the status registry, which is pure enum data,
is imported eagerly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._status import CERTIFY_COMPLETED_EXIT_CODES, CertifyStatus, certify_exit_code

if TYPE_CHECKING:
    from ..compare._failure import ComparisonOperationError as CertifyOperationError
    from ._receipt import CertifyResult, load_and_validate_certification_receipt
    from ._run import certify_models

_DEFERRED: dict[str, tuple[str, str]] = {
    "CertifyOperationError": ("..compare._failure", "ComparisonOperationError"),
    "CertifyResult": ("._receipt", "CertifyResult"),
    "RECEIPT_SCHEMA": ("._receipt", "RECEIPT_SCHEMA"),
    "REQUIRED_LIMITATIONS": ("._receipt", "REQUIRED_LIMITATIONS"),
    "certify_models": ("._run", "certify_models"),
    "load_and_validate_certification_receipt": (
        "._receipt",
        "load_and_validate_certification_receipt",
    ),
    "validate_receipt": ("._receipt", "validate_receipt"),
}


def __getattr__(name: str) -> Any:
    """Lazily load the deliberate Certify API without importing MuJoCo eagerly."""
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
    """Expose only deliberate public Certify names to introspection."""
    return sorted(set(globals()) | set(_DEFERRED))


__all__ = [
    "CERTIFY_COMPLETED_EXIT_CODES",
    "RECEIPT_SCHEMA",
    "REQUIRED_LIMITATIONS",
    "CertifyOperationError",
    "CertifyResult",
    "CertifyStatus",
    "certify_exit_code",
    "certify_models",
    "load_and_validate_certification_receipt",
    "validate_receipt",
]
