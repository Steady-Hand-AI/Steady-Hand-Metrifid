"""Retained-object publication and cleanup for Metrifid-created files."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
from collections.abc import Callable
from dataclasses import dataclass
from typing import BinaryIO, Final, Literal

_CHUNK_BYTES: Final[int] = 1024 * 1024
ArtifactStage = Literal["private-temporary", "committed-final"]


class OwnedArtifactError(RuntimeError):
    """Report a retained artifact ownership or byte-verification failure."""


@dataclass(slots=True)
class OwnedArtifact:
    """Retain one product-created file and every fact needed for safe verification."""

    parent_fd: int
    temporary_name: str
    fd: int
    device: int
    inode: int
    mode: int
    expected_length: int | None = None
    expected_sha256: str | None = None
    final_name: str | None = None
    stage: ArtifactStage = "private-temporary"
    closed: bool = False
    temporary_owned: bool = True

    @property
    def current_name(self) -> str:
        """Return the sole authoritative name after the current lifecycle stage."""
        if self.stage == "committed-final" and self.final_name is not None:
            return self.final_name
        return self.temporary_name

    def verify(self) -> None:
        """Require the retained descriptor and current directory entry to match exactly."""
        _require_open(self)
        _verify_descriptor_bytes(self)
        _verify_named_entry(self, self.current_name)

    def cleanup(self) -> None:
        """Remove only the still-owned private temporary and preserve every public final."""
        if self.closed:
            return
        if self.temporary_owned:
            _unlink_if_exact(self, self.temporary_name)
            self.temporary_owned = False
        _fsync_quietly(self.parent_fd)

    def close(self) -> None:
        """Close retained file and parent descriptors exactly once."""
        if self.closed:
            return
        self.closed = True
        _close_quietly(self.fd)
        _close_quietly(self.parent_fd)
        self.fd = -1
        self.parent_fd = -1

    def __del__(self) -> None:
        """Release retained descriptors if an exceptional owner abandons the handle."""
        self.close()


@dataclass(slots=True)
class RetainedArtifactPair:
    """Own the two retained files produced by one paired publication transaction."""

    first: OwnedArtifact
    second: OwnedArtifact
    closed: bool = False

    def verify(self) -> None:
        """Reverify both committed files in deterministic publication order."""
        if self.closed:
            raise OwnedArtifactError("retained artifact pair is closed")
        self.first.verify()
        self.second.verify()

    def cleanup(self) -> None:
        """Remove only pair private temporaries while preserving every public final."""
        if self.closed:
            return
        self.second.cleanup()
        self.first.cleanup()

    def close(self) -> None:
        """Close both retained artifacts exactly once."""
        if self.closed:
            return
        self.closed = True
        self.second.close()
        self.first.close()

    def __del__(self) -> None:
        """Release both retained artifacts if a caller abandons the pair."""
        self.close()


def create_owned_artifact(parent_fd: int, prefix: str) -> OwnedArtifact:
    """Create one exclusive O_RDWR private temporary under a retained parent duplicate."""
    _require_plain_prefix(prefix)
    retained_parent = os.dup(parent_fd)
    descriptor = -1
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor, name = _open_unique(retained_parent, prefix, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OwnedArtifactError("private artifact is not a regular file")
        return OwnedArtifact(
            retained_parent,
            name,
            descriptor,
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
        )
    except BaseException:
        _close_quietly(descriptor)
        _close_quietly(retained_parent)
        raise


def write_owned_bytes(artifact: OwnedArtifact, payload: bytes) -> None:
    """Write exact bytes through a duplicate handle, fsync, seal, and verify them."""
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")

    def writer(stream: BinaryIO) -> None:
        """Write the supplied immutable payload to the duplicate stream."""
        stream.write(payload)

    write_owned_stream(artifact, writer)


def write_owned_stream(artifact: OwnedArtifact, writer: Callable[[BinaryIO], None]) -> None:
    """Run one stream writer through a duplicate handle and seal the resulting bytes."""
    _require_unsealed(artifact)
    duplicate = os.dup(artifact.fd)
    try:
        stream = os.fdopen(duplicate, "w+b")
    except BaseException:
        _close_quietly(duplicate)
        raise
    try:
        with stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        _seal_partial_bytes(artifact)
        raise
    length, digest = _measure_descriptor(artifact.fd)
    artifact.expected_length = length
    artifact.expected_sha256 = digest
    _verify_descriptor_bytes(artifact)
    _verify_named_entry(artifact, artifact.temporary_name)


def _seal_partial_bytes(artifact: OwnedArtifact) -> None:
    """Seal whatever partial bytes remain so failure cleanup can remain ownership-safe."""
    try:
        os.fsync(artifact.fd)
        length, digest = _measure_descriptor(artifact.fd)
    except (OSError, OwnedArtifactError):
        return
    artifact.expected_length = length
    artifact.expected_sha256 = digest


def link_owned_artifact(artifact: OwnedArtifact, final_name: str) -> None:
    """Acquire one exact public link before recording its final name."""
    _require_plain_name(final_name)
    if artifact.final_name is not None:
        raise OwnedArtifactError("artifact already has a linked final name")
    if artifact.stage != "private-temporary" or not artifact.temporary_owned:
        raise OwnedArtifactError("artifact is not a private temporary")
    _verify_descriptor_bytes(artifact)
    _verify_named_entry(artifact, artifact.temporary_name)
    os.link(
        artifact.temporary_name,
        final_name,
        src_dir_fd=artifact.parent_fd,
        dst_dir_fd=artifact.parent_fd,
        follow_symlinks=False,
    )
    _verify_descriptor_bytes(artifact)
    _verify_named_entry(artifact, final_name)
    artifact.final_name = final_name


def finalize_owned_artifact(artifact: OwnedArtifact) -> None:
    """Verify the linked final, remove the exact temporary, fsync, and retain the final."""
    if artifact.final_name is None:
        raise OwnedArtifactError("artifact has no linked final name")
    _verify_descriptor_bytes(artifact)
    _verify_named_entry(artifact, artifact.final_name)
    if not artifact.temporary_owned or not _unlink_if_exact(artifact, artifact.temporary_name):
        raise OwnedArtifactError("private temporary changed before finalization")
    artifact.temporary_owned = False
    artifact.stage = "committed-final"
    os.fsync(artifact.parent_fd)
    artifact.verify()


def commit_owned_artifact(artifact: OwnedArtifact, final_name: str) -> OwnedArtifact:
    """No-clobber publish one sealed artifact and retain its final descriptor."""
    try:
        link_owned_artifact(artifact, final_name)
        finalize_owned_artifact(artifact)
        return artifact
    except BaseException:
        artifact.cleanup()
        raise


def commit_owned_pair(
    first: OwnedArtifact,
    first_name: str,
    second: OwnedArtifact,
    second_name: str,
) -> RetainedArtifactPair:
    """Link both sealed temporaries before verifying and finalizing either committed file."""
    pair = RetainedArtifactPair(first, second)
    try:
        link_owned_artifact(first, first_name)
        link_owned_artifact(second, second_name)
        finalize_owned_artifact(first)
        finalize_owned_artifact(second)
        pair.verify()
        return pair
    except BaseException:
        pair.cleanup()
        pair.close()
        raise


def _open_unique(parent_fd: int, prefix: str, flags: int) -> tuple[int, str]:
    """Allocate one unpredictable private name using exclusive descriptor-relative creation."""
    for _attempt in range(100):
        name = f"{prefix}{secrets.token_hex(8)}.tmp"
        try:
            return os.open(name, flags, 0o600, dir_fd=parent_fd), name
        except FileExistsError:
            continue
    raise FileExistsError("could not allocate a private owned artifact")


def _measure_descriptor(descriptor: int) -> tuple[int, str]:
    """Return exact length and SHA-256 while proving EOF through positional reads."""
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise OwnedArtifactError("owned descriptor is not a regular file")
    digest = hashlib.sha256()
    offset = 0
    while offset < metadata.st_size:
        chunk = os.pread(descriptor, min(_CHUNK_BYTES, metadata.st_size - offset), offset)
        if not chunk:
            raise OwnedArtifactError("owned artifact ended before its retained length")
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, offset):
        raise OwnedArtifactError("owned artifact extends beyond its retained length")
    return offset, digest.hexdigest()


def _verify_descriptor_bytes(artifact: OwnedArtifact) -> None:
    """Require the open file to retain its regular mode, exact length, and SHA-256."""
    _require_open(artifact)
    if artifact.expected_length is None or artifact.expected_sha256 is None:
        raise OwnedArtifactError("owned artifact bytes are not sealed")
    metadata = os.fstat(artifact.fd)
    identity = (metadata.st_dev, metadata.st_ino, metadata.st_mode)
    expected = (artifact.device, artifact.inode, artifact.mode)
    if identity != expected or not stat.S_ISREG(metadata.st_mode):
        raise OwnedArtifactError("owned artifact descriptor identity changed")
    length, digest = _measure_descriptor(artifact.fd)
    if length != artifact.expected_length or digest != artifact.expected_sha256:
        raise OwnedArtifactError("owned artifact bytes changed")


def _verify_named_entry(artifact: OwnedArtifact, name: str) -> None:
    """Require a relative name to resolve to the exact retained regular-file object."""
    try:
        metadata = os.stat(name, dir_fd=artifact.parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise OwnedArtifactError("owned artifact name is unavailable") from exc
    identity = (metadata.st_dev, metadata.st_ino, metadata.st_mode)
    expected = (artifact.device, artifact.inode, artifact.mode)
    if identity != expected or not stat.S_ISREG(metadata.st_mode):
        raise OwnedArtifactError("owned artifact name no longer identifies the retained object")


def _entry_is_exact(artifact: OwnedArtifact, name: str) -> bool:
    """Return whether one name still identifies this exact retained object and bytes."""
    try:
        _verify_descriptor_bytes(artifact)
        _verify_named_entry(artifact, name)
    except (OSError, OwnedArtifactError):
        return False
    return True


def _unlink_if_exact(artifact: OwnedArtifact, name: str) -> bool:
    """Unlink one entry only after exact retained-object and byte verification."""
    if not _entry_is_exact(artifact, name):
        return False
    try:
        os.unlink(name, dir_fd=artifact.parent_fd)
    except FileNotFoundError:
        return False
    return True


def _require_open(artifact: OwnedArtifact) -> None:
    """Refuse operations after retained descriptors have been closed."""
    if artifact.closed or artifact.fd < 0 or artifact.parent_fd < 0:
        raise OwnedArtifactError("owned artifact is closed")


def _require_unsealed(artifact: OwnedArtifact) -> None:
    """Require one new private temporary before its only write operation."""
    _require_open(artifact)
    if artifact.expected_length is not None or artifact.expected_sha256 is not None:
        raise OwnedArtifactError("owned artifact is already sealed")
    if (
        artifact.final_name is not None
        or artifact.stage != "private-temporary"
        or not artifact.temporary_owned
    ):
        raise OwnedArtifactError("owned artifact is not a writable private temporary")


def _require_plain_prefix(prefix: str) -> None:
    """Validate a nonempty descriptor-relative temporary-name prefix."""
    if not isinstance(prefix, str) or not prefix or "/" in prefix or "\x00" in prefix:
        raise ValueError("artifact temporary prefix must be plain")


def _require_plain_name(name: str) -> None:
    """Validate one nonempty descriptor-relative final file name."""
    if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
        raise ValueError("artifact final name must be plain")
    if "\x00" in name:
        raise ValueError("artifact final name must not contain NUL")


def _fsync_quietly(descriptor: int) -> None:
    """Best-effort synchronize a retained directory during failure cleanup."""
    try:
        os.fsync(descriptor)
    except OSError:
        pass


def _close_quietly(descriptor: int) -> None:
    """Best-effort close one possibly already-closed descriptor."""
    if descriptor < 0:
        return
    try:
        os.close(descriptor)
    except OSError:
        pass


__all__ = [
    "OwnedArtifact",
    "OwnedArtifactError",
    "RetainedArtifactPair",
    "commit_owned_artifact",
    "commit_owned_pair",
    "create_owned_artifact",
    "link_owned_artifact",
    "finalize_owned_artifact",
    "write_owned_bytes",
    "write_owned_stream",
]
