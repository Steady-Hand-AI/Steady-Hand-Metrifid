"""Default run allocation must own what it writes into.

A run directory is chosen by this code, not by the caller, so the checks that admit it and the
object finally written into have to be the same object. These regressions substitute the parent
and the run itself at each seam and require a refusal rather than a redirected write, a silently
adopted stranger, or a deleted replacement.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from metrifid import _atomic_output
from metrifid._atomic_output import PairedOutputNames, create_owned_paired_output
from metrifid._model_refusal import ModelAdmissionRefusal
from metrifid._npz import ArtifactAdmissionRefusal
from metrifid.operational import OperationalReasonCode
from metrifid.result import _runs

_NAMES = PairedOutputNames("model_release.json", "model_release.md")
# Allocation refuses through two existing primitives, which carry the same reason and evidence
# under two exception names. Either is a correct refusal; neither may be a silent write.
_REFUSALS = (ArtifactAdmissionRefusal, ModelAdmissionRefusal)


def _retention_root(tmp_path: Path) -> Path:
    """Return an isolated retention root under a real home, never the user's own."""
    home = (tmp_path / "home").resolve()
    home.mkdir(parents=True, exist_ok=True)
    return home / ".metrifid" / "runs"


def _model_root(tmp_path: Path) -> Path:
    """Return one isolated model root that no allocation may ever write into."""
    root = (tmp_path / "models").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_a_retention_ancestor_substituted_after_admission_creates_nothing_in_the_target(
    tmp_path: Path,
) -> None:
    """Every step below home happens at a held descriptor, so a late swap redirects nothing.

    This is the reviewer's parent-setup reproduction turned around: replacing the tool directory
    once it had been admitted used to let the next pathname mkdir land inside the model root.
    """
    root = _retention_root(tmp_path)
    models = _model_root(tmp_path)
    tool = root.parent
    real_subdirectory = _atomic_output.open_owned_subdirectory
    admitted = 0

    def substitute_after_first(parent_fd: int, child_name: str, *, mode: int) -> int:
        nonlocal admitted
        descriptor = real_subdirectory(parent_fd, child_name, mode=mode)
        admitted += 1
        if admitted == 1:
            tool.rename(tool.with_name(".metrifid-original"))
            tool.symlink_to(models, target_is_directory=True)
        return descriptor

    with (
        patch.object(_runs, "runs_root", return_value=root),
        patch.object(_atomic_output, "open_owned_subdirectory", side_effect=substitute_after_first),
    ):
        output = _runs.allocate_run_directory((models,))
    try:
        assert list(models.iterdir()) == [], "nothing may be created inside the model root"
        assert (tool.with_name(".metrifid-original") / "runs").is_dir()
    finally:
        output.close()


def test_a_retention_ancestor_that_is_already_a_link_is_refused(tmp_path: Path) -> None:
    """A tool directory that is a link when first observed is never entered or created through."""
    root = _retention_root(tmp_path)
    models = _model_root(tmp_path)
    root.parent.symlink_to(models, target_is_directory=True)

    with (
        patch.object(_runs, "runs_root", return_value=root),
        pytest.raises(_REFUSALS) as refusal,
    ):
        _runs.allocate_run_directory((models,))

    assert refusal.value.reason is OperationalReasonCode.OUTPUT_PATH_INVALID
    assert refusal.value.evidence["issue"] == "output_path_not_real_directory"
    assert list(models.iterdir()) == [], "nothing may be created through the substitution"


def test_a_run_substituted_before_binding_is_refused_rather_than_adopted(
    tmp_path: Path,
) -> None:
    """Require the bound object to be the one just created, so a stranger is never adopted."""
    parent = (tmp_path / "parent").resolve()
    parent.mkdir()
    foreign = (tmp_path / "foreign").resolve()
    foreign.mkdir()
    real_stat = os.stat

    def substitute_then_stat(name: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if name == "run" and "dir_fd" in kwargs:
            created = parent / "run"
            if created.is_dir() and not created.is_symlink():
                created.rmdir()
                created.symlink_to(foreign, target_is_directory=True)
        return real_stat(name, *args, **kwargs)

    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with (
            patch.object(_atomic_output.os, "stat", side_effect=substitute_then_stat),
            pytest.raises(_REFUSALS) as refusal,
        ):
            create_owned_paired_output(parent_fd, parent, "run", _NAMES, mode=0o700)
    finally:
        os.close(parent_fd)

    assert refusal.value.reason is OperationalReasonCode.OUTPUT_PATH_INVALID
    assert list(foreign.iterdir()) == [], "nothing may be published into the substituted object"


def test_an_allocated_run_is_private_empty_and_bound_to_the_object_created(
    tmp_path: Path,
) -> None:
    """The returned output names the created directory and holds a descriptor on that object."""
    root = _retention_root(tmp_path)
    models = _model_root(tmp_path)
    with patch.object(_runs, "runs_root", return_value=root):
        output = _runs.allocate_run_directory((models,))
    try:
        assert output.path.parent == root
        assert list(output.path.iterdir()) == []
        assert stat.S_IMODE(output.path.stat().st_mode) == _runs.RUNS_DIRECTORY_MODE
        bound = os.fstat(output.directory_fd)
        created = output.path.stat()
        assert (bound.st_dev, bound.st_ino) == (created.st_dev, created.st_ino)
        for ancestor in (root, root.parent):
            assert stat.S_IMODE(ancestor.stat().st_mode) == _runs.RUNS_DIRECTORY_MODE
    finally:
        output.close()


def test_name_collisions_retry_with_fresh_names_and_exhaust_into_an_internal_failure(
    tmp_path: Path,
) -> None:
    """Retries are collision-only and finite; exhaustion is an internal failure, not reuse."""
    root = _retention_root(tmp_path)
    models = _model_root(tmp_path)
    root.mkdir(parents=True)
    (root / "cafebabecafebabe").mkdir()
    attempts = 0

    def always_collide(_count: int) -> str:
        nonlocal attempts
        attempts += 1
        return "cafebabecafebabe"

    with (
        patch.object(_runs, "runs_root", return_value=root),
        patch.object(_runs.secrets, "token_hex", side_effect=always_collide),
        pytest.raises(_REFUSALS) as refusal,
    ):
        _runs.allocate_run_directory((models,))

    assert refusal.value.reason is OperationalReasonCode.INTERNAL_INVARIANT_FAILED
    assert int(refusal.value.reason.exit_code) == 70
    assert attempts == _runs.MAX_RUN_ALLOCATION_ATTEMPTS
    assert sorted(entry.name for entry in root.iterdir()) == ["cafebabecafebabe"]


def test_a_retention_root_inside_a_model_root_is_refused_before_anything_is_created(
    tmp_path: Path,
) -> None:
    """Check the prospective root against the model roots before creating any parent."""
    models = _model_root(tmp_path)
    root = models / "nested" / ".metrifid" / "runs"
    with (
        patch.object(_runs, "runs_root", return_value=root),
        pytest.raises(_REFUSALS) as refusal,
    ):
        _runs.allocate_run_directory((models,))

    assert refusal.value.evidence["issue"] == "retained_run_root_inside_model_root"
    assert list(models.iterdir()) == [], "no parent may be created before the overlap check"


def test_the_removed_failure_reaper_is_gone_from_the_allocator_surface() -> None:
    """Emptiness is not ownership, so no post-failure directory inspection may return."""
    assert not hasattr(_runs, "discard_unused_run_directory")
    assert "discard_unused_run_directory" not in _runs.__all__


def test_the_retention_root_and_its_ancestors_are_private_to_the_user(tmp_path: Path) -> None:
    """The parent, not the allocator, is what excludes another account from the run.

    Creating a directory and receiving its descriptor is not one step on POSIX, so the instant
    between them cannot be closed in code. A retention root only its owner can enter is what makes
    that instant unreachable, which is why the mode is asserted here rather than assumed.
    """
    root = _retention_root(tmp_path)
    models = _model_root(tmp_path)
    with patch.object(_runs, "runs_root", return_value=root):
        output = _runs.allocate_run_directory((models,))
    try:
        for directory in (output.path, root, root.parent):
            mode = stat.S_IMODE(directory.stat().st_mode)
            assert mode == _runs.RUNS_DIRECTORY_MODE, directory
            assert not mode & (stat.S_IRWXG | stat.S_IRWXO), directory
    finally:
        output.close()
