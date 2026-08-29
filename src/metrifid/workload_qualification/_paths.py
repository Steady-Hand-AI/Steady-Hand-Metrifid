"""Path admission, ordinal storage naming, and locator typing for one qualification.

Three separate ideas that the previous implementation conflated:

*Declared source paths* are user data. They are admitted as normalized traversal-free relative POSIX
paths and resolved against the base each field is defined against. An entrypoint is defined against
its own model root, not against the configuration directory.

*Storage components* are chosen by Metrifid, never by the user. Semantic labels such as
``workload_id`` and ``probe_id`` are report fields; using one as a directory component let an
absolute label replace the intended parent and place evidence outside the owned output root
entirely. Storage names here are fixed-width ordinals derived only from admitted sequence position,
so they are stable for one admitted configuration, unique, version-neutral, independent of label
bytes, and not a hash of untrusted text.

*Locators* are recorded data describing where retained evidence lives relative to the owned output
root. A locator is read back, re-admitted, and then resolved only through the owned root's retained
directory descriptor; it is never joined onto a pathname and never used as a write target. Lexical
containment was the previous answer and it was not containment: joining an admitted relative locator
onto the root still traverses whatever the intermediate components happen to be at read time.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Final

from .._schema_primitives import _relative_posix_path

CONTROLS_DIRECTORY: Final[str] = "controls"
PROBES_DIRECTORY: Final[str] = "probes"
EVIDENCE_DIRECTORY: Final[str] = "evidence"
RECEIPT_DIRECTORY: Final[str] = "receipt"
ADMITTED_CONFIG_NAME: Final[str] = "qualification.json"
COMPARISON_CONFIG_NAME: Final[str] = "comparison.json"
COMPARISON_OUTPUT_NAME: Final[str] = "comparison_out"
COMPARISON_RECEIPT_NAME: Final[str] = "comparison.json"
_ORDINAL_WIDTH: Final[int] = 3


class PathAdmissionError(ValueError):
    """Raised when a declared path or a recorded locator is not admissible."""


def admit_relative_path(value: object, field: str) -> str:
    """Admit one declared path as a normalized, traversal-free, relative POSIX path.

    Backslashes, absolute paths, empty components, ``.``, ``..``, and any non-normalized spelling
    are refused rather than repaired, because repairing a path silently changes which bytes the
    receipt describes.
    """
    try:
        return _relative_posix_path(value, field)
    except (TypeError, ValueError) as exc:
        raise PathAdmissionError(str(exc)) from exc


def resolve_under(base: Path, relative: str, field: str) -> Path:
    """Join one admitted relative path onto its declared base without escaping it."""
    admitted = admit_relative_path(relative, field)
    resolved = (base / admitted).absolute()
    try:
        resolved.relative_to(base.absolute())
    except ValueError as exc:  # pragma: no cover - unreachable for admitted relative paths
        raise PathAdmissionError(f"{field} escapes its declared base") from exc
    return resolved


def ordinal(prefix: str, index: int) -> str:
    """Return one fixed-width storage component derived only from sequence position."""
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise PathAdmissionError("ordinal index must be a non-negative integer")
    return f"{prefix}_{index:0{_ORDINAL_WIDTH}d}"


def control_locator(workload_index: int) -> PurePosixPath:
    """Return the owned-root-relative directory for one zero-change control."""
    return (
        PurePosixPath(EVIDENCE_DIRECTORY) / CONTROLS_DIRECTORY / ordinal("workload", workload_index)
    )


def probe_locator(workload_index: int, group_index: int, variant_index: int) -> PurePosixPath:
    """Return the owned-root-relative directory for one probe cell."""
    return (
        PurePosixPath(EVIDENCE_DIRECTORY)
        / PROBES_DIRECTORY
        / ordinal("workload", workload_index)
        / ordinal("group", group_index)
        / ordinal("rung", variant_index)
    )


def admit_locator(value: object, field: str) -> PurePosixPath:
    """Re-admit one recorded locator read back from a receipt."""
    return PurePosixPath(admit_relative_path(value, field))
