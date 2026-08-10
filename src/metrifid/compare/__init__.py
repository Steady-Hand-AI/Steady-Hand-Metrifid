"""Installed comparison comparison API with MuJoCo imports deferred until execution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._failure import ComparisonOperationError

if TYPE_CHECKING:
    from ._orchestrator import ComparisonRunResult


def compare_configuration_file(config_path: str | Path) -> ComparisonRunResult:
    """Run one comparison, importing the MuJoCo runtime only on demand."""
    from ._orchestrator import compare_configuration_file as _run

    return _run(config_path)


def __getattr__(name: str) -> Any:
    """Lazily load the comparison API without importing MuJoCo eagerly."""
    if name == "ComparisonRunResult":
        from ._orchestrator import ComparisonRunResult

        return ComparisonRunResult
    raise AttributeError(name)


__all__ = [
    "ComparisonOperationError",
    "ComparisonRunResult",
    "compare_configuration_file",
]
