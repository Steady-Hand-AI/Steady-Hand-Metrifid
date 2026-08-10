"""Deliberate public interface for Metrifid."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import TYPE_CHECKING, Final

from .errors import (
    REASON_REGISTRY,
    STATUS_PRECEDENCE,
    ComparisonStatus,
    EngineThreadpoolState,
    LimitationCode,
    OperationalExitCode,
    ReasonCode,
    ReasonRecord,
)
from .json_values import (
    Binary64,
    ExactRational,
    canonical_json_bytes,
    canonical_sha256,
    strict_json_loads,
)
from .operational import (
    InputDigest,
    InputDigestCode,
    OperationalFailure,
    OperationalReasonCode,
    OperationalStage,
    OperationalToolObservation,
)
from .schemas import (
    ComparisonConfig,
    ComparisonContractIdentity,
    ComparisonReceipt,
    finalize_receipt,
    validate_receipt,
)
from .version import __version__

if TYPE_CHECKING:
    from .workload_writers import write_actions_artifact, write_state_artifact

# The workload writers depend on NumPy. Receipt parsing and validation must stay importable in an
# environment that has neither NumPy nor MuJoCo, so these two names resolve on first attribute
# access instead of at package import. They remain part of the public surface.
_LAZY_EXPORTS: Final[Mapping[str, str]] = {
    "write_actions_artifact": ".workload_writers",
    "write_state_artifact": ".workload_writers",
}

__all__ = [
    "__version__",
    "Binary64",
    "ExactRational",
    "canonical_json_bytes",
    "canonical_sha256",
    "strict_json_loads",
    "ComparisonStatus",
    "EngineThreadpoolState",
    "LimitationCode",
    "OperationalExitCode",
    "ReasonCode",
    "ReasonRecord",
    "STATUS_PRECEDENCE",
    "REASON_REGISTRY",
    "ComparisonConfig",
    "ComparisonContractIdentity",
    "ComparisonReceipt",
    "finalize_receipt",
    "validate_receipt",
    "OperationalStage",
    "OperationalReasonCode",
    "OperationalFailure",
    "OperationalToolObservation",
    "InputDigestCode",
    "InputDigest",
    "write_state_artifact",
    "write_actions_artifact",
]


def __dir__() -> list[str]:
    """Expose only the deliberate public names to API introspection."""
    return sorted(__all__)


def __getattr__(name: str) -> object:
    """Resolve the NumPy-backed public writers on first access.

    Submodules imported with ``import metrifid.errors`` keep the normal Python behavior of binding
    themselves as package attributes; only the deferred names above are handled here.
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
