"""Portable path admission and overlap checks for Native Runtime Review.

Declared evidence paths are caller-controlled input.  They are admitted as normalized relative
POSIX paths, resolved beneath the configuration directory, and walked without accepting a symbolic
link in any declared component.  Output ownership remains a separate atomic-publication concern;
this module decides whether the proposed not-yet-existing root is safe to own before publication
creates anything.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path

from .._schema_primitives import _relative_posix_path


class PathAdmissionError(ValueError):
    """Raised when a declared runtime-review path is unsafe or ambiguous."""


def admit_relative_portable_path(value: object, field: str) -> str:
    """Admit one normalized, traversal-free relative path using POSIX separators."""
    try:
        return _relative_posix_path(value, field)
    except (TypeError, ValueError) as exc:
        raise PathAdmissionError(str(exc)) from exc


def resolve_existing_directory(base: Path, relative: str, field: str) -> Path:
    """Resolve one declared path beneath ``base`` and require real directory components.

    Every component below the configuration directory is inspected with ``lstat``.  Symbolic links,
    missing components, and non-directories are refused instead of followed or repaired.
    """
    admitted = admit_relative_portable_path(relative, field)
    root = _admit_base_directory(base)
    current = root
    for component in admitted.split("/"):
        current = current / component
        metadata = _lstat(current, field)
        if stat.S_ISLNK(metadata.st_mode):
            raise PathAdmissionError(f"{field} must not contain symbolic links")
        if not stat.S_ISDIR(metadata.st_mode):
            raise PathAdmissionError(f"{field} must name a directory")
    _require_beneath(root, current, field)
    return current


def resolve_new_output_path(base: Path, relative: str, field: str = "output_dir") -> Path:
    """Resolve a proposed output root beneath ``base`` without following declared links.

    Existing ancestors must be real directories.  The proposed root itself must not exist, because
    runtime-review evidence is immutable and publication must never overwrite an earlier result.
    A missing suffix is allowed so the atomic output owner can create it safely later.
    """
    admitted = admit_relative_portable_path(relative, field)
    root = _admit_base_directory(base)
    current = root
    components = admitted.split("/")
    for index, component in enumerate(components):
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            candidate = current.joinpath(*components[index + 1 :])
            _require_beneath(root, candidate, field)
            return candidate
        except OSError as exc:
            raise PathAdmissionError(f"{field} could not be inspected: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise PathAdmissionError(f"{field} must not contain symbolic links")
        if not stat.S_ISDIR(metadata.st_mode):
            raise PathAdmissionError(f"{field} ancestors must be directories")
    raise PathAdmissionError(f"{field} already exists; runtime-review output is never overwritten")


def ensure_nonoverlapping(output: Path, protected: Mapping[str, Path]) -> None:
    """Require a proposed output root to be disjoint from every protected input path.

    Overlap is symmetric: the output may neither contain a protected path nor be contained by one.
    Existing protected paths are resolved strictly so a different spelling through an ancestor link
    cannot defeat the comparison.
    """
    canonical_output = _canonical_missing_path(output, "output_dir")
    for label, path in protected.items():
        try:
            canonical_protected = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise PathAdmissionError(f"{label} could not be resolved: {exc}") from exc
        if _overlaps(canonical_output, canonical_protected):
            raise PathAdmissionError(f"output_dir overlaps {label}")


def _admit_base_directory(base: Path) -> Path:
    """Return the canonical existing configuration directory."""
    if not isinstance(base, Path):
        raise TypeError("base must be a Path")
    try:
        resolved = base.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PathAdmissionError(f"configuration directory could not be resolved: {exc}") from exc
    if not resolved.is_dir():
        raise PathAdmissionError("configuration directory must be a directory")
    return resolved


def _lstat(path: Path, field: str) -> os.stat_result:
    """Inspect one declared component without following a final symbolic link."""
    try:
        return os.lstat(path)
    except OSError as exc:
        raise PathAdmissionError(f"{field} could not be admitted as a directory: {exc}") from exc


def _require_beneath(root: Path, candidate: Path, field: str) -> None:
    """Require one lexical candidate to stay strictly beneath its declared root."""
    try:
        candidate.relative_to(root)
    except ValueError as exc:  # pragma: no cover - traversal is rejected before resolution
        raise PathAdmissionError(f"{field} escapes the configuration directory") from exc


def _canonical_missing_path(path: Path, field: str) -> Path:
    """Canonicalize a possibly missing path through its nearest existing ancestor."""
    absolute = path.absolute()
    existing = absolute
    suffix: list[str] = []
    while True:
        try:
            metadata = os.lstat(existing)
        except FileNotFoundError:
            if existing.parent == existing:
                raise PathAdmissionError(f"{field} has no existing ancestor") from None
            suffix.append(existing.name)
            existing = existing.parent
            continue
        except OSError as exc:
            raise PathAdmissionError(f"{field} could not be inspected: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise PathAdmissionError(f"{field} must not contain symbolic links")
        if not stat.S_ISDIR(metadata.st_mode):
            raise PathAdmissionError(f"{field} ancestor must be a directory")
        break
    return existing.resolve(strict=True).joinpath(*reversed(suffix))


def _overlaps(left: Path, right: Path) -> bool:
    """Return whether either path is equal to or nested beneath the other."""
    return left == right or left in right.parents or right in left.parents
