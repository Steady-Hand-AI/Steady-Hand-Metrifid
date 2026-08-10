"""Strict bounded NPZ admission shared by the internal artifact admission workload loaders."""

from __future__ import annotations

import hashlib
import io
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, Literal, TypeAlias, cast
from zipfile import BadZipFile, ZipFile

import numpy as np
import numpy.typing as npt

from .json_values import (
    CanonicalValue,
    FrozenCanonicalObject,
    freeze_canonical,
    require_sha256,
    thaw_canonical,
)
from .operational import OperationalReasonCode

ArtifactRole: TypeAlias = Literal["baseline", "candidate", "comparison"]
MAX_NPZ_BYTES: Final[int] = 512 * 1024 * 1024


class ArtifactAdmissionRefusal(Exception):
    """One expected artifact admission refusal carrying an existing operational reason code."""

    __slots__ = ("evidence", "reason", "role")
    evidence: FrozenCanonicalObject
    reason: OperationalReasonCode
    role: ArtifactRole

    def __init__(
        self,
        reason: OperationalReasonCode,
        role: ArtifactRole,
        evidence: Mapping[str, CanonicalValue] | None = None,
    ) -> None:
        """Capture one expected artifact refusal with frozen role-local evidence."""
        if role not in {"baseline", "candidate", "comparison"}:
            raise ValueError("invalid artifact admission refusal role")
        frozen = freeze_canonical(cast(CanonicalValue, dict(evidence or {})))
        self.reason = reason
        self.role = role
        self.evidence = cast(FrozenCanonicalObject, frozen)
        super().__init__(f"{reason.value}:{role}")

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return the deterministic internal refusal representation."""
        return {
            "reason": self.reason.value,
            "role": self.role,
            "evidence": thaw_canonical(self.evidence),
        }


def refuse(
    reason: OperationalReasonCode,
    role: ArtifactRole = "comparison",
    **evidence: CanonicalValue,
) -> ArtifactAdmissionRefusal:
    """Construct one expected artifact admission refusal without raising it."""
    return ArtifactAdmissionRefusal(reason, role, evidence)


@dataclass(frozen=True, slots=True)
class LoadedNpz:
    """The raw digest and exact arrays loaded from one prevalidated NPZ byte string."""

    raw_file_sha256: str
    arrays: Mapping[str, npt.NDArray[np.generic]]

    def __post_init__(self) -> None:
        """Validate immutable raw bytes, raw hash, and named loaded arrays."""
        require_sha256(self.raw_file_sha256, "raw_file_sha256")
        if type(self.arrays) is not MappingProxyType:
            object.__setattr__(self, "arrays", MappingProxyType(dict(self.arrays)))


def load_npz_arrays(
    path: str | Path,
    *,
    expected_members: frozenset[str],
    invalid_reason: OperationalReasonCode,
) -> LoadedNpz:
    """Read once, preflight the ZIP container, then load arrays from the same bytes."""
    raw = _read_bounded_bytes(path, invalid_reason)
    return _load_npz_arrays_from_bytes(
        raw,
        expected_members=expected_members,
        invalid_reason=invalid_reason,
    )


def _load_npz_arrays_from_bytes(
    raw: bytes,
    *,
    expected_members: frozenset[str],
    invalid_reason: OperationalReasonCode,
) -> LoadedNpz:
    """Preflight, parse, and hash one exact bounded NPZ byte string."""
    if not isinstance(raw, bytes):
        raise TypeError("raw must be bytes")
    if len(raw) > MAX_NPZ_BYTES:
        raise refuse(
            OperationalReasonCode.NPZ_SIZE_BUDGET_EXCEEDED,
            issue="raw_file",
            size_bytes=len(raw),
            maximum_bytes=MAX_NPZ_BYTES,
        )
    _validate_zip_container(raw, expected_members)
    arrays = _load_arrays_from_same_bytes(raw, expected_members, invalid_reason)
    return LoadedNpz(hashlib.sha256(raw).hexdigest(), MappingProxyType(arrays))


def _read_bounded_bytes(
    path: str | Path,
    invalid_reason: OperationalReasonCode,
) -> bytes:
    """Open once without following links and read no more than the raw byte budget plus one."""
    try:
        artifact_path = Path(path)
    except TypeError as exc:
        raise refuse(invalid_reason, issue="invalid_path_type") from exc
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(artifact_path, flags)
    except OSError as exc:
        raise refuse(invalid_reason, issue="artifact_stat_failed") from exc
    try:
        try:
            metadata = os.fstat(descriptor)
        except OSError as exc:
            raise refuse(invalid_reason, issue="artifact_stat_failed") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise refuse(invalid_reason, issue="artifact_not_regular_file")
        _require_raw_size_budget(metadata.st_size)
        raw = _read_descriptor_bounded(descriptor, invalid_reason)
    finally:
        os.close(descriptor)
    _require_raw_size_budget(len(raw))
    return raw


def _read_descriptor_bounded(
    descriptor: int,
    invalid_reason: OperationalReasonCode,
) -> bytes:
    """Read at most the configured maximum plus one byte from an open descriptor."""
    payload = bytearray()
    remaining = MAX_NPZ_BYTES + 1
    try:
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            payload.extend(chunk)
            remaining -= len(chunk)
    except OSError as exc:
        raise refuse(invalid_reason, issue="artifact_read_failed") from exc
    return bytes(payload)


def _require_raw_size_budget(size: int) -> None:
    """Refuse a raw NPZ size above the configured byte budget."""
    if size > MAX_NPZ_BYTES:
        raise refuse(
            OperationalReasonCode.NPZ_SIZE_BUDGET_EXCEEDED,
            issue="raw_file",
            size_bytes=size,
            maximum_bytes=MAX_NPZ_BYTES,
        )


def _validate_zip_container(raw: bytes, expected_members: frozenset[str]) -> None:
    """Validate ZIP structure, compression limits, and the exact expected NPZ member set."""
    try:
        with ZipFile(io.BytesIO(raw), "r") as archive:
            infos = archive.infolist()
    except (BadZipFile, OSError, ValueError) as exc:
        raise refuse(
            OperationalReasonCode.NPZ_MEMBER_SET_INVALID,
            issue="invalid_zip_container",
        ) from exc
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        raise refuse(
            OperationalReasonCode.NPZ_DUPLICATE_MEMBER,
            duplicate_members=cast(CanonicalValue, duplicates),
        )
    for info in infos:
        _validate_member_path(info.filename, info.is_dir())
        if info.flag_bits & 0x1:
            raise refuse(
                OperationalReasonCode.NPZ_ENCRYPTED_MEMBER,
                member=info.filename,
            )
    total = sum(info.file_size for info in infos)
    if total > MAX_NPZ_BYTES:
        raise refuse(
            OperationalReasonCode.NPZ_SIZE_BUDGET_EXCEEDED,
            issue="uncompressed_content",
            size_bytes=total,
            maximum_bytes=MAX_NPZ_BYTES,
        )
    if set(names) != expected_members:
        raise refuse(
            OperationalReasonCode.NPZ_MEMBER_SET_INVALID,
            missing_members=cast(CanonicalValue, sorted(expected_members - set(names))),
            extra_members=cast(CanonicalValue, sorted(set(names) - expected_members)),
        )


def _validate_member_path(name: str, is_directory: bool) -> None:
    """Reject directory, absolute, nested, duplicate, or non-NPY NPZ member paths."""
    path = PurePosixPath(name)
    invalid = (
        is_directory
        or not name
        or path.is_absolute()
        or "/" in name
        or "\\" in name
        or ".." in name
    )
    if invalid:
        raise refuse(
            OperationalReasonCode.NPZ_PATH_MEMBER_INVALID,
            member=name,
            directory=is_directory,
        )


def _load_arrays_from_same_bytes(
    raw: bytes,
    expected_members: frozenset[str],
    invalid_reason: OperationalReasonCode,
) -> dict[str, npt.NDArray[np.generic]]:
    """Load arrays from same bytes from raw, expected members and invalid reason for npz, rejecting invalid input with refuse, TypeError."""
    arrays: dict[str, npt.NDArray[np.generic]] = {}
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            for member in sorted(expected_members):
                key = member.removesuffix(".npy")
                value = archive[key]
                if not isinstance(value, np.ndarray):
                    raise TypeError("NPZ member is not an ndarray")
                arrays[key] = value
    except ValueError as exc:
        if "Object arrays cannot be loaded when allow_pickle=False" in str(exc):
            raise refuse(
                OperationalReasonCode.NPZ_OBJECT_ARRAY_REFUSED,
                issue="object_or_pickle_backed_array",
            ) from exc
        raise refuse(invalid_reason, issue="numpy_load_failed") from exc
    except (BadZipFile, KeyError, OSError, TypeError, EOFError) as exc:
        raise refuse(invalid_reason, issue="numpy_load_failed") from exc
    return arrays


__all__ = [
    "ArtifactAdmissionRefusal",
    "LoadedNpz",
    "MAX_NPZ_BYTES",
    "load_npz_arrays",
    "refuse",
]
