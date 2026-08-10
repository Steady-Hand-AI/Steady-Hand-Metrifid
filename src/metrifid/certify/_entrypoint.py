"""Resolve one MJCF path plus optional root into the accepted (root, relative entrypoint) pair.

Certify takes model paths on the command line rather than from a JSON configuration, so this
module is the only place that turns a user path into the two values the accepted closure
machinery already understands. It resolves nothing else: measurement, symlink policy inside the
tree and byte identity all remain the accepted closure implementation's job.
"""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .._model_closure import ModelRole, refuse
from ..operational import OperationalReasonCode


@dataclass(frozen=True, slots=True)
class ResolvedEntrypoint:
    """One real model root and the normal relative POSIX entrypoint beneath it."""

    model_root: Path
    entrypoint: str


def resolve_entrypoint(
    mjcf_path: str, root_argument: str | None, role: ModelRole
) -> ResolvedEntrypoint:
    """Resolve a command-line MJCF path, with or without an explicit root."""
    absolute = Path(mjcf_path).absolute()
    _require_regular_file(absolute, role)
    real_parent = _real_directory(absolute.parent, role, issue="entrypoint_parent_unavailable")
    if root_argument is None:
        return _verified(real_parent, absolute.name, role)
    root = _real_directory(Path(root_argument).absolute(), role, issue="root_unavailable")
    try:
        relative_parent = real_parent.relative_to(root)
    except ValueError:
        raise refuse(
            OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE,
            role,
            issue="entrypoint_outside_supplied_root",
        ) from None
    entrypoint = PurePosixPath(*relative_parent.parts, absolute.name).as_posix()
    return _verified(root, entrypoint, role)


def _verified(root: Path, entrypoint: str, role: ModelRole) -> ResolvedEntrypoint:
    """Require a normal relative POSIX entrypoint that names the same regular file."""
    posix = PurePosixPath(entrypoint)
    if (
        posix.is_absolute()
        or not posix.parts
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise refuse(
            OperationalReasonCode.MODEL_ENTRYPOINT_INVALID,
            role,
            entrypoint=entrypoint,
            issue="entrypoint_not_a_normal_relative_posix_path",
        )
    _require_regular_file(root.joinpath(*posix.parts), role)
    return ResolvedEntrypoint(root, posix.as_posix())


def _require_regular_file(path: Path, role: ModelRole) -> None:
    """Admit a real regular role entrypoint without a final symlink."""
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise refuse(
            OperationalReasonCode.MODEL_ENTRYPOINT_INVALID,
            role,
            issue="entrypoint_unavailable",
            exception_type=type(exc).__name__,
        ) from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise refuse(
            OperationalReasonCode.MODEL_CLOSURE_SYMLINK_REFUSED,
            role,
            issue="entrypoint_is_a_symlink",
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise refuse(
            OperationalReasonCode.MODEL_ENTRYPOINT_INVALID,
            role,
            issue="entrypoint_not_a_regular_file",
        )


def _real_directory(path: Path, role: ModelRole, *, issue: str) -> Path:
    """Resolve every symlink in a directory path and require a real directory."""
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.lstat()
    except OSError as exc:
        raise refuse(
            OperationalReasonCode.MODEL_ROOT_INVALID,
            role,
            issue=issue,
            exception_type=type(exc).__name__,
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise refuse(
            OperationalReasonCode.MODEL_ROOT_INVALID,
            role,
            issue="root_not_real_directory",
        )
    return resolved


__all__ = ["ResolvedEntrypoint", "resolve_entrypoint"]
