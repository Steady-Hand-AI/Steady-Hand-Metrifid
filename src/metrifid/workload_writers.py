"""Deterministic public writers for strict metrifid state and action artifacts."""

from __future__ import annotations

import io
import os
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO, Final, cast

import numpy as np
import numpy.typing as npt

from ._owned_artifacts import (
    OwnedArtifact,
    OwnedArtifactError,
    commit_owned_artifact,
    create_owned_artifact,
    write_owned_stream,
)

_STATE_SCHEMA: Final[str] = "metrifid.state"
_ACTIONS_SCHEMA: Final[str] = "metrifid.actions"
_SCHEMA_VERSION: Final[int] = 1
_ZIP_TIMESTAMP: Final[tuple[int, int, int, int, int, int]] = (1980, 1, 1, 0, 0, 0)


def write_state_artifact(
    path: str | os.PathLike[str],
    *,
    joint_names: Sequence[str],
    qpos_offsets: Sequence[int],
    qpos: npt.ArrayLike,
    qvel_offsets: Sequence[int],
    qvel: npt.ArrayLike,
    actuator_names: Sequence[str],
    act_offsets: Sequence[int],
    act: npt.ArrayLike,
) -> None:
    """Write one canonical state NPZ.

    Names may arrive in any unique order. Segmented values are reordered into exact
    Unicode-code-point name order before deterministic serialization.
    """
    joints = _validated_names(joint_names, "joint_names")
    actuators = _validated_names(actuator_names, "actuator_names")
    qpos_values = _float_array(qpos, "qpos", dimensions=1)
    qvel_values = _float_array(qvel, "qvel", dimensions=1)
    act_values = _float_array(act, "act", dimensions=1)
    qpos_bounds = _validated_offsets(qpos_offsets, len(joints), qpos_values.size, "qpos_offsets")
    qvel_bounds = _validated_offsets(qvel_offsets, len(joints), qvel_values.size, "qvel_offsets")
    act_bounds = _validated_offsets(act_offsets, len(actuators), act_values.size, "act_offsets")

    sorted_joints, sorted_qpos_offsets, sorted_qpos = _canonical_segments(
        joints, qpos_bounds, qpos_values
    )
    _, sorted_qvel_offsets, sorted_qvel = _canonical_segments(joints, qvel_bounds, qvel_values)
    sorted_actuators, sorted_act_offsets, sorted_act = _canonical_segments(
        actuators, act_bounds, act_values
    )
    arrays: dict[str, npt.NDArray[np.generic]] = {
        "schema.npy": _unicode_scalar(_STATE_SCHEMA),
        "schema_version.npy": np.asarray(_SCHEMA_VERSION, dtype="<i8"),
        "joint_names.npy": _unicode_array(sorted_joints),
        "qpos_offsets.npy": np.asarray(sorted_qpos_offsets, dtype="<i8"),
        "qpos.npy": sorted_qpos,
        "qvel_offsets.npy": np.asarray(sorted_qvel_offsets, dtype="<i8"),
        "qvel.npy": sorted_qvel,
        "actuator_names.npy": _unicode_array(sorted_actuators),
        "act_offsets.npy": np.asarray(sorted_act_offsets, dtype="<i8"),
        "act.npy": sorted_act,
    }
    _write_npz_atomic(path, arrays)


def write_actions_artifact(
    path: str | os.PathLike[str],
    *,
    actuator_names: Sequence[str],
    values: npt.ArrayLike,
) -> None:
    """Write one canonical actions NPZ.

    Action columns are reordered with their unique names into exact canonical order.
    """
    names = _validated_names(actuator_names, "actuator_names")
    action_values = _float_array(values, "values", dimensions=2)
    if not 1 <= action_values.shape[0] <= 100_000:
        raise ValueError("values must contain between 1 and 100000 control rows")
    if action_values.shape[1] != len(names):
        raise ValueError("values column count must equal actuator_names length")
    order = tuple(sorted(range(len(names)), key=names.__getitem__))
    sorted_names = tuple(names[index] for index in order)
    sorted_values = np.ascontiguousarray(action_values[:, order], dtype="<f8")
    arrays: dict[str, npt.NDArray[np.generic]] = {
        "schema.npy": _unicode_scalar(_ACTIONS_SCHEMA),
        "schema_version.npy": np.asarray(_SCHEMA_VERSION, dtype="<i8"),
        "actuator_names.npy": _unicode_array(sorted_names),
        "values.npy": sorted_values,
    }
    _write_npz_atomic(path, arrays)


def _validated_names(values: Sequence[str], field: str) -> tuple[str, ...]:
    """Validate unique workload names and return canonical code-point order."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field} must be a sequence of names")
    result: list[str] = []
    for value in values:
        if type(value) is not str:
            raise TypeError(f"{field} entries must be strings")
        value.encode("utf-8", errors="strict")
        if not value or "\x00" in value:
            raise ValueError(f"{field} entries must be nonempty valid names")
        result.append(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} must not contain duplicate names")
    return tuple(result)


def _float_array(value: npt.ArrayLike, field: str, *, dimensions: int) -> npt.NDArray[np.float64]:
    """Copy finite input into a C-order little-endian float64 array of fixed rank."""
    try:
        array = np.asarray(value, dtype="<f8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{field} must be convertible to float64") from exc
    if array.ndim != dimensions:
        raise ValueError(f"{field} must have rank {dimensions}")
    result = np.ascontiguousarray(array, dtype="<f8")
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{field} must contain only finite values")
    return cast(npt.NDArray[np.float64], result)


def _validated_offsets(
    values: Sequence[int], item_count: int, value_count: int, field: str
) -> tuple[int, ...]:
    """Validate segmentation offsets against item and flattened-value counts."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field} must be a sequence of integers")
    result: list[int] = []
    for value in values:
        if type(value) is not int:
            raise TypeError(f"{field} entries must be integers and not booleans")
        if value < 0:
            raise ValueError(f"{field} entries must be nonnegative")
        result.append(value)
    valid = (
        len(result) == item_count + 1
        and bool(result)
        and result[0] == 0
        and result[-1] == value_count
        and all(result[index] <= result[index + 1] for index in range(len(result) - 1))
    )
    if not valid:
        raise ValueError(f"{field} does not describe the supplied segmented values")
    return tuple(result)


def _canonical_segments(
    names: tuple[str, ...],
    offsets: tuple[int, ...],
    values: npt.NDArray[np.float64],
) -> tuple[tuple[str, ...], tuple[int, ...], npt.NDArray[np.float64]]:
    """Reorder named flattened segments and rebuild offsets in canonical name order."""
    order = tuple(sorted(range(len(names)), key=names.__getitem__))
    sorted_names = tuple(names[index] for index in order)
    pieces = [values[offsets[index] : offsets[index + 1]] for index in order]
    widths = [piece.size for piece in pieces]
    sorted_offsets = [0]
    for width in widths:
        sorted_offsets.append(sorted_offsets[-1] + width)
    if pieces:
        sorted_values = np.ascontiguousarray(np.concatenate(pieces), dtype="<f8")
    else:
        sorted_values = np.empty((0,), dtype="<f8")
    return sorted_names, tuple(sorted_offsets), cast(npt.NDArray[np.float64], sorted_values)


def _unicode_scalar(value: str) -> npt.NDArray[np.str_]:
    """Encode one metadata token as a scalar little-endian NumPy Unicode array."""
    width = max(1, len(value))
    return np.asarray(value, dtype=f"<U{width}")


def _unicode_array(values: tuple[str, ...]) -> npt.NDArray[np.str_]:
    """Encode names using the minimal fixed-width little-endian NumPy Unicode dtype."""
    width = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"<U{width}")


def _npy_bytes(array: npt.NDArray[np.generic]) -> bytes:
    """Serialize one admitted array to deterministic unpickled NPY bytes."""
    buffer = io.BytesIO()
    # numpy ships this helper untyped; the call itself is exact.
    np.lib.format.write_array(buffer, array, allow_pickle=False)  # type: ignore[no-untyped-call]
    return buffer.getvalue()


def _write_npz_atomic(
    path: str | os.PathLike[str], arrays: dict[str, npt.NDArray[np.generic]]
) -> None:
    """No-clobber publish deterministic NPZ bytes through one retained owned object."""
    target = Path(path)
    if not target.name or target.name in {".", ".."} or "/" in target.name:
        raise ValueError("artifact path must name one file")
    parent = target.parent
    parent_fd, identity = _bind_writer_parent(parent)
    artifact: OwnedArtifact | None = None
    try:
        _require_regular_writer_target(parent_fd, target.name)
        artifact = _open_writer_temp(parent_fd, target.name)
        _write_npz_to_artifact(artifact, arrays)
        _verify_writer_parent(parent, identity)
        _commit_writer_artifact(artifact, target.name)
        _verify_writer_parent(parent, identity)
        artifact.verify()
    except BaseException:
        if artifact is not None:
            artifact.cleanup()
        raise
    finally:
        if artifact is not None:
            artifact.close()
        os.close(parent_fd)


def _commit_writer_artifact(artifact: OwnedArtifact, target_name: str) -> None:
    """Commit one sealed writer artifact while translating ownership failures to ValueError."""
    try:
        commit_owned_artifact(artifact, target_name)
    except FileExistsError as exc:
        raise ValueError("artifact destination must be absent") from exc
    except (OSError, OwnedArtifactError) as exc:
        raise ValueError("artifact changed during publication") from exc


def _bind_writer_parent(parent: Path) -> tuple[int, tuple[int, int]]:
    """Bind one existing real parent through no-follow component traversal."""
    try:
        descriptor = _open_writer_directory(parent)
        bound = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError("artifact parent must be an existing real directory") from exc
    try:
        _verify_writer_parent(parent, (bound.st_dev, bound.st_ino))
    except ValueError:
        os.close(descriptor)
        raise
    return descriptor, (bound.st_dev, bound.st_ino)


def _open_writer_directory(path: Path) -> int:
    """Open an absolute directory one no-follow component at a time."""
    absolute = path.absolute()
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            child_fd = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child_fd
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _require_regular_writer_target(parent_fd: int, name: str) -> None:
    """Require the destination name to be absent, preserving every existing object."""
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError("artifact destination must be absent") from exc
    raise ValueError("artifact destination must be absent")


def _open_writer_temp(parent_fd: int, target_name: str) -> OwnedArtifact:
    """Create one retained private artifact using the shared ownership implementation."""
    return create_owned_artifact(parent_fd, f".{target_name}.")


def _write_npz_to_artifact(
    artifact: OwnedArtifact, arrays: dict[str, npt.NDArray[np.generic]]
) -> None:
    """Serialize one canonical NPZ through a duplicate of the retained O_RDWR descriptor."""

    def writer(stream: BinaryIO) -> None:
        """Write deterministic uncompressed ZIP members to the supplied duplicate stream."""
        with zipfile.ZipFile(
            stream, "w", compression=zipfile.ZIP_STORED, strict_timestamps=True
        ) as archive:
            for member in sorted(arrays):
                info = zipfile.ZipInfo(member, _ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, _npy_bytes(arrays[member]))

    write_owned_stream(artifact, writer)


def _verify_writer_parent(parent: Path, identity: tuple[int, int]) -> None:
    """Refuse when the public parent pathname no longer names the bound object."""
    try:
        current = parent.lstat()
    except OSError as exc:
        raise ValueError("artifact parent changed during publication") from exc
    changed = current.st_mode & 0o170000 != 0o040000 or (current.st_dev, current.st_ino) != identity
    if changed:
        raise ValueError("artifact parent changed during publication")


__all__ = ["write_actions_artifact", "write_state_artifact"]
