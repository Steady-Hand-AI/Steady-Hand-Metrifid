"""Public, lazily native-loaded API for compare-backed Workload Qualification.

One user question: do the selected workloads detect every declared model perturbation at or above
its required magnitude under the declared comparison tolerances, and which perturbations remain
blind?

Every answer here is built from completed `metrifid compare` runs against user-supplied probe
models. Nothing in this package estimates detectability from a sensitivity surrogate.

The supported surface is deliberately small, and `__all__` is a compatibility commitment:

```text
qualify_configuration_file                          run one campaign from a strict JSON file
QualificationResult                                 the completed result it returns
QualificationStatus                                 the completed status registry
QualificationExitCode                               the completed process exit-code registry
WorkloadQualificationOperationError                 the bounded operational refusal
load_and_validate_workload_qualification_receipt    validate one published receipt and its evidence
```

The strict JSON configuration file is the only supported way to describe a campaign. Configuration,
probe, workload, cell, group, limitation and cardinality types are internal implementation detail:
they remain reachable from their private modules for testing, but they are not part of this
compatibility surface, and several of their constructors require internal types.

Importing this module performs no filesystem, network, or native-runtime work. MuJoCo is loaded only
when an execution or receipt-validation name is first resolved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..compare._failure import ComparisonOperationError as WorkloadQualificationOperationError
    from ._receipt import load_and_validate_workload_qualification_receipt
    from ._run import QualificationResult, qualify_configuration_file
    from ._status import QualificationExitCode, QualificationStatus

_DEFERRED: dict[str, tuple[str, str]] = {
    "QualificationExitCode": ("._status", "QualificationExitCode"),
    "QualificationResult": ("._run", "QualificationResult"),
    "QualificationStatus": ("._status", "QualificationStatus"),
    "WorkloadQualificationOperationError": ("..compare._failure", "ComparisonOperationError"),
    "load_and_validate_workload_qualification_receipt": (
        "._receipt",
        "load_and_validate_workload_qualification_receipt",
    ),
    "qualify_configuration_file": ("._run", "qualify_configuration_file"),
}


def __getattr__(name: str) -> Any:
    """Resolve one supported name on first use, keeping native loading lazy."""
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
    """Expose exactly the supported surface plus this module's own attributes."""
    return sorted(set(globals()) | set(_DEFERRED))


__all__ = [
    "QualificationExitCode",
    "QualificationResult",
    "QualificationStatus",
    "WorkloadQualificationOperationError",
    "load_and_validate_workload_qualification_receipt",
    "qualify_configuration_file",
]
