"""Descriptor-confined no-clobber publication of one retained output pair."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

from ._npz import ArtifactAdmissionRefusal, refuse
from ._owned_artifacts import (
    OwnedArtifact,
    OwnedArtifactError,
    RetainedArtifactPair,
    commit_owned_pair,
    create_owned_artifact,
    write_owned_bytes,
)
from .operational import OperationalReasonCode


@dataclass(frozen=True, slots=True)
class PairedOutputNames:
    """The two final file names one operation publishes together."""

    json_name: str
    markdown_name: str

    def __post_init__(self) -> None:
        """Require two distinct plain file names with no path components."""
        for name in (self.json_name, self.markdown_name):
            if not name or "/" in name or name in {".", ".."}:
                raise ValueError("paired output names must be plain file names")
        if self.json_name == self.markdown_name:
            raise ValueError("paired output names must differ")


class _DirectoryHandle:
    """Own one open directory descriptor and its admitted object identity."""

    __slots__ = ("closed", "device", "fd", "inode")

    def __init__(self, fd: int, *, device: int, inode: int) -> None:
        """Retain one already-open descriptor and its stable identity."""
        self.fd = fd
        self.device = device
        self.inode = inode
        self.closed = False

    def close(self) -> None:
        """Close the retained directory descriptor exactly once."""
        if self.closed:
            return
        self.closed = True
        try:
            os.close(self.fd)
        except OSError:
            pass
        self.fd = -1

    def __del__(self) -> None:
        """Release the retained descriptor when its final owner disappears."""
        self.close()


@dataclass(frozen=True, slots=True)
class PairedOutputDirectory:
    """One descriptor-bound real directory plus the pair published into it."""

    path: Path
    names: PairedOutputNames
    _handle: _DirectoryHandle = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Bind the directory once without following its final path component."""
        object.__setattr__(self, "_handle", _bind_directory(self.path))

    @classmethod
    def _from_handle(
        cls,
        path: Path,
        names: PairedOutputNames,
        handle: _DirectoryHandle,
    ) -> PairedOutputDirectory:
        """Construct an output around an already descriptor-bound directory."""
        instance = cls.__new__(cls)
        object.__setattr__(instance, "path", path)
        object.__setattr__(instance, "names", names)
        object.__setattr__(instance, "_handle", handle)
        return instance

    @classmethod
    def _from_descriptor(
        cls, path: Path, names: PairedOutputNames, descriptor: int
    ) -> PairedOutputDirectory:
        """Own one supplied directory descriptor without resolving its public path."""
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("paired output descriptor must name a directory")
        except BaseException:
            os.close(descriptor)
            raise
        handle = _DirectoryHandle(descriptor, device=metadata.st_dev, inode=metadata.st_ino)
        return cls._from_handle(path, names, handle)

    @property
    def directory_fd(self) -> int:
        """Return the retained directory descriptor for internal confined operations."""
        if self._handle.closed:
            raise ValueError("paired output directory is closed")
        return self._handle.fd

    @property
    def json_path(self) -> Path:
        """Return the final JSON path inside the admitted output directory."""
        return self.path / self.names.json_name

    @property
    def markdown_path(self) -> Path:
        """Return the final Markdown path inside the admitted output directory."""
        return self.path / self.names.markdown_name

    def close(self) -> None:
        """Close the retained directory descriptor exactly once."""
        self._handle.close()


def prepare_paired_output_directory(path: Path, names: PairedOutputNames) -> PairedOutputDirectory:
    """Create an absent final directory or bind one empty real directory."""
    absolute = path.absolute()
    try:
        parent_fd = _open_real_directory(absolute.parent)
    except OSError as exc:
        raise refuse(
            OperationalReasonCode.OUTPUT_PATH_INVALID,
            issue="output_parent_unavailable",
            exception_type=type(exc).__name__,
        ) from exc
    try:
        output = _open_paired_output_child(parent_fd, absolute, absolute.name, names, create=True)
    finally:
        os.close(parent_fd)
    try:
        _require_empty_directory(output)
        verify_paired_output_path_unchanged(output)
        return output
    except BaseException:
        output.close()
        raise


def _adopt_paired_output_descriptor(
    path: Path, names: PairedOutputNames, descriptor: int
) -> PairedOutputDirectory:
    """Adopt one already-retained empty directory descriptor as a paired output.

    The caller has already bound the object it wants written into. Taking that descriptor, rather
    than re-traversing its pathname, is what keeps the publication inside that exact object: there
    is no second lookup for another process to redirect. The supplied descriptor is owned from here
    on and is closed with the returned output.
    """
    output = PairedOutputDirectory._from_descriptor(path, names, descriptor)
    try:
        _require_empty_directory(output)
        return output
    except BaseException:
        output.close()
        raise


def _open_paired_output_child(
    parent_fd: int,
    public_path: Path,
    child_name: str,
    names: PairedOutputNames,
    *,
    create: bool,
) -> PairedOutputDirectory:
    """Bind one real descriptor-relative child, optionally creating it with mkdirat."""
    if not child_name or "/" in child_name or child_name in {".", ".."}:
        raise ValueError("output child name must be a plain file name")
    metadata = _admit_output_child(parent_fd, child_name, create=create)
    return _bind_paired_child(parent_fd, public_path, child_name, names, metadata)


def publish_paired_results(
    output: PairedOutputDirectory,
    *,
    json_bytes: bytes,
    markdown_text: str,
) -> RetainedArtifactPair:
    """No-clobber publish both outputs and return their retained ownership handle."""
    return _publish_paired_results(
        output,
        json_bytes=json_bytes,
        markdown_text=markdown_text,
        require_empty=True,
    )


def _publish_paired_results(
    output: PairedOutputDirectory,
    *,
    json_bytes: bytes,
    markdown_text: str,
    require_empty: bool,
) -> RetainedArtifactPair:
    """Publish a pair inside the bound directory and retain both final descriptors."""
    if not isinstance(json_bytes, bytes):
        raise TypeError("json_bytes must be bytes")
    markdown_bytes = markdown_text.encode("utf-8", errors="strict")
    artifacts: list[OwnedArtifact] = []
    try:
        verify_paired_output_path_unchanged(output)
        if require_empty:
            _require_empty_directory(output)
        artifacts.append(_write_temp(output, _temp_prefix(output.names.json_name), json_bytes))
        artifacts.append(
            _write_temp(output, _temp_prefix(output.names.markdown_name), markdown_bytes)
        )
        pair = commit_owned_pair(
            artifacts[0],
            output.names.json_name,
            artifacts[1],
            output.names.markdown_name,
        )
        verify_paired_output_path_unchanged(output)
        pair.verify()
        return pair
    except ArtifactAdmissionRefusal:
        _cleanup_artifacts(artifacts)
        raise
    except (OSError, OwnedArtifactError, UnicodeError) as exc:
        _cleanup_artifacts(artifacts)
        raise refuse(
            OperationalReasonCode.OUTPUT_WRITE_FAILED,
            issue="paired_atomic_publish_failed",
            exception_type=type(exc).__name__,
        ) from exc


def verify_paired_output_path_unchanged(output: PairedOutputDirectory) -> None:
    """Refuse when the public output pathname no longer names the bound directory."""
    try:
        metadata = output.path.lstat()
    except OSError as exc:
        raise refuse(
            OperationalReasonCode.OUTPUT_PATH_INVALID,
            issue="output_path_replaced",
            exception_type=type(exc).__name__,
        ) from exc
    identity = (metadata.st_dev, metadata.st_ino)
    admitted = (output._handle.device, output._handle.inode)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode) or identity != admitted:
        raise refuse(OperationalReasonCode.OUTPUT_PATH_INVALID, issue="output_path_replaced")


def verify_paired_results(output: PairedOutputDirectory, retained: RetainedArtifactPair) -> None:
    """Require the public path and both retained final byte strings to remain exact."""
    verify_paired_output_path_unchanged(output)
    retained.verify()


def cleanup_paired_output_after_failure(
    output: PairedOutputDirectory | None,
    retained: RetainedArtifactPair | None = None,
) -> None:
    """Remove retained private temporaries, preserve public finals, and close descriptors."""
    if retained is not None:
        retained.cleanup()
        retained.close()
    if output is not None:
        output.close()


def _admit_output_child(parent_fd: int, child_name: str, *, create: bool) -> os.stat_result:
    """Return no-follow metadata for one admitted or newly created output child."""
    try:
        metadata = os.stat(child_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        if not create:
            raise refuse(
                OperationalReasonCode.OUTPUT_PATH_INVALID, issue="output_directory_missing"
            ) from None
        metadata = _create_output_child(parent_fd, child_name)
    except OSError as exc:
        raise refuse(
            OperationalReasonCode.OUTPUT_PATH_INVALID,
            issue="output_path_stat_failed",
            exception_type=type(exc).__name__,
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise refuse(
            OperationalReasonCode.OUTPUT_PATH_INVALID,
            issue="output_path_not_real_directory",
        )
    return metadata


def _create_output_child(parent_fd: int, child_name: str) -> os.stat_result:
    """Create one output child and return its no-follow metadata."""
    try:
        os.mkdir(child_name, mode=0o755, dir_fd=parent_fd)
        return os.stat(child_name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise refuse(
            OperationalReasonCode.OUTPUT_PATH_INVALID,
            issue="output_directory_create_failed",
            exception_type=type(exc).__name__,
        ) from exc


def _bind_directory(path: Path) -> _DirectoryHandle:
    """Open and retain a real directory through no-follow component traversal."""
    descriptor = -1
    try:
        descriptor = _open_real_directory(path)
        metadata = os.fstat(descriptor)
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise refuse(
            OperationalReasonCode.OUTPUT_PATH_INVALID,
            issue="output_directory_bind_failed",
            exception_type=type(exc).__name__,
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise refuse(
            OperationalReasonCode.OUTPUT_PATH_INVALID,
            issue="output_path_not_real_directory",
        )
    return _DirectoryHandle(descriptor, device=metadata.st_dev, inode=metadata.st_ino)


def _bind_paired_child(
    parent_fd: int,
    public_path: Path,
    child_name: str,
    names: PairedOutputNames,
    admitted: os.stat_result,
) -> PairedOutputDirectory:
    """Open a child no-follow and require the same object observed during admission."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(child_name, flags, dir_fd=parent_fd)
        bound = os.fstat(descriptor)
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise refuse(
            OperationalReasonCode.OUTPUT_PATH_INVALID,
            issue="output_directory_bind_failed",
            exception_type=type(exc).__name__,
        ) from exc
    if (admitted.st_dev, admitted.st_ino) != (bound.st_dev, bound.st_ino):
        os.close(descriptor)
        raise refuse(OperationalReasonCode.OUTPUT_PATH_INVALID, issue="output_path_replaced")
    handle = _DirectoryHandle(descriptor, device=bound.st_dev, inode=bound.st_ino)
    return PairedOutputDirectory._from_handle(public_path, names, handle)


def _open_real_directory(path: Path) -> int:
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


def _write_temp(output: PairedOutputDirectory, prefix: str, payload: bytes) -> OwnedArtifact:
    """Create, completely write, fsync, seal, and verify one private temporary."""
    artifact = create_owned_artifact(output.directory_fd, prefix)
    try:
        write_owned_bytes(artifact, payload)
        return artifact
    except BaseException:
        artifact.cleanup()
        artifact.close()
        raise


def _cleanup_artifacts(artifacts: list[OwnedArtifact]) -> None:
    """Best-effort clean and close partially created retained artifacts in reverse order."""
    for artifact in reversed(artifacts):
        artifact.cleanup()
        artifact.close()


def _temp_prefix(name: str) -> str:
    """Build the hidden same-directory prefix used for an output temporary."""
    return f".{name}."


def _require_empty_directory(output: PairedOutputDirectory) -> None:
    """Refuse a bound output directory that cannot be scanned or is not empty."""
    entries = _directory_entries(output)
    if entries:
        names = sorted(entry.name for entry in entries)
        raise refuse(
            OperationalReasonCode.OUTPUT_DIRECTORY_NOT_EMPTY,
            entry_count=len(names),
            first_entry=names[0],
        )


def _directory_entries(output: PairedOutputDirectory) -> list[os.DirEntry[str]]:
    """Scan the admitted directory through its retained descriptor."""
    try:
        with os.scandir(output.directory_fd) as iterator:
            return list(iterator)
    except OSError as exc:
        raise refuse(
            OperationalReasonCode.OUTPUT_PATH_INVALID,
            issue="output_directory_scan_failed",
            exception_type=type(exc).__name__,
        ) from exc


__all__ = [
    "PairedOutputDirectory",
    "PairedOutputNames",
    "cleanup_paired_output_after_failure",
    "prepare_paired_output_directory",
    "publish_paired_results",
    "verify_paired_output_path_unchanged",
    "verify_paired_results",
]
