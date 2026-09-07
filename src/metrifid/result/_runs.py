"""Allocate one retained run directory when the caller chose no output directory.

A caller who supplies no ``--output`` still gets retained evidence.  The run directory is created
exclusively beneath the tool's own subtree, so a comparison never writes over anything and never
has to invent a name inside a directory someone else owns.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .._model_closure import ModelRole, refuse
from ..operational import OperationalReasonCode

if TYPE_CHECKING:
    from .._atomic_output import PairedOutputDirectory

__all__ = [
    "MAX_RUN_ALLOCATION_ATTEMPTS",
    "RUNS_DIRECTORY_MODE",
    "allocate_run_directory",
    "require_admissible_model_root",
    "runs_root",
]

MAX_RUN_ALLOCATION_ATTEMPTS: Final = 100
RUNS_DIRECTORY_MODE: Final = 0o700
_RUN_NAME_BYTES: Final = 8


def runs_root() -> Path:
    """Return the retained-run root beneath the current user's home directory.

    Returns:
        The ``~/.metrifid/runs`` path, not necessarily existing yet.

    Raises:
        ComparisonOperationError: The home directory could not be determined.
    """
    try:
        home = Path.home()
        if not home.is_absolute() or not home.is_dir():
            raise OSError("home directory is not an existing absolute directory")
        home = home.resolve(strict=True)
        if home.parent == home:
            # An empty HOME resolves to the filesystem root on this platform. Retaining runs
            # directly under it is never what the caller meant.
            raise OSError("home directory resolved to the filesystem root")
    except (OSError, RuntimeError) as exc:
        raise refuse(
            OperationalReasonCode.OUTPUT_PATH_INVALID,
            "comparison",
            issue="home_directory_unavailable",
            remedy="supply an explicit --output directory",
            exception_type=type(exc).__name__,
        ) from exc
    return home / ".metrifid" / "runs"


def _require_no_overlap(prospective: Path, model_roots: tuple[Path, ...]) -> None:
    """Refuse a retained-run root equal to or inside either admitted model root."""
    for root in model_roots:
        if prospective == root or root in prospective.parents:
            raise refuse(
                OperationalReasonCode.OUTPUT_PATH_INVALID,
                "comparison",
                issue="retained_run_root_inside_model_root",
                remedy="supply an explicit --output directory outside both model roots",
            )


def allocate_run_directory(model_roots: tuple[Path, ...]) -> PairedOutputDirectory:
    """Create one new run directory and return it already bound to its own descriptor.

    The home directory is opened one component at a time without following a link, and every step
    below it happens at the descriptor already held: the tool's two ancestors are created and
    admitted there, and the run itself is created there too. No step resolves a pathname a second
    time, so replacing a directory after it was admitted cannot redirect a later creation into
    another tree. Each directory is created private to the user. A name collision is retried with
    fresh random bytes, and exhausting the bounded attempts is an internal failure rather than a
    silent reuse of somebody else's directory.

    Args:
        model_roots: The resolved baseline and candidate roots, checked for overlap first.

    Returns:
        The bound output directory the comparison will publish into.

    Raises:
        ComparisonOperationError: The root overlapped a model root, could not be created, or no
            free run name was found within the bounded attempts.
    """
    # Imported here so reading a saved receipt never pulls the native-backed publisher in.
    from .._atomic_output import create_owned_paired_output
    from ..model_release._run import MODEL_RELEASE_OUTPUT_NAMES

    root = runs_root()
    _require_no_overlap(root, model_roots)
    runs_fd = _open_retention_root(root)
    try:
        for _ in range(MAX_RUN_ALLOCATION_ATTEMPTS):
            try:
                return create_owned_paired_output(
                    runs_fd,
                    root,
                    secrets.token_hex(_RUN_NAME_BYTES),
                    MODEL_RELEASE_OUTPUT_NAMES,
                    mode=RUNS_DIRECTORY_MODE,
                )
            except FileExistsError:
                continue
    finally:
        os.close(runs_fd)
    raise refuse(
        OperationalReasonCode.INTERNAL_INVARIANT_FAILED,
        "comparison",
        issue="retained_run_name_allocation_exhausted",
        attempts=MAX_RUN_ALLOCATION_ATTEMPTS,
    )


def _open_retention_root(root: Path) -> int:
    """Create and open the tool's two retention ancestors relative to the home descriptor.

    Args:
        root: The retention root, whose parent and grandparent are the tool directory and home.

    Returns:
        An owned descriptor for the retention root, which the caller closes.

    Raises:
        ComparisonOperationError: Home could not be opened, or an ancestor is not a real
            directory the tool may create and enter.
    """
    from .._atomic_output import _open_real_directory, open_owned_subdirectory

    try:
        descriptor = _open_real_directory(root.parent.parent)
    except OSError as exc:
        raise refuse(
            OperationalReasonCode.OUTPUT_PATH_INVALID,
            "comparison",
            issue="retained_run_root_create_failed",
            exception_type=type(exc).__name__,
        ) from exc
    for name in (root.parent.name, root.name):
        try:
            child = open_owned_subdirectory(descriptor, name, mode=RUNS_DIRECTORY_MODE)
        finally:
            os.close(descriptor)
        descriptor = child
    return descriptor


def require_admissible_model_root(root: Path, role: ModelRole) -> None:
    """Refuse a model root so broad that measuring it cannot be what the caller meant.

    The filesystem root, the current home directory itself, and any ancestor of it are refused
    before the tree is enumerated, because Metrifid measures a model root whole.

    Args:
        root: One already-resolved model root.
        role: The role that root belongs to.

    Raises:
        ComparisonOperationError: The root was the filesystem root, the home directory, or an
            ancestor of the home directory.
    """
    if root.parent == root:
        raise _broad_root(role, "model_root_is_filesystem_root")
    try:
        home = Path.home().resolve()
    except (OSError, RuntimeError):
        return
    if root == home:
        raise _broad_root(role, "model_root_is_home_directory")
    if root in home.parents:
        raise _broad_root(role, "model_root_is_ancestor_of_home_directory")


def _broad_root(role: ModelRole, issue: str) -> Exception:
    """Build the refusal naming the remedy for a root that is too broad to measure."""
    return refuse(
        OperationalReasonCode.MODEL_ROOT_INVALID,
        role,
        issue=issue,
        remedy=(
            "point the model path at the directory holding the model, or pass an explicit "
            "--baseline-root/--candidate-root naming it"
        ),
    )
