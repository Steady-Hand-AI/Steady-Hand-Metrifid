"""Audit-specific retained directory registry and ownership-safe cleanup."""

from __future__ import annotations

import os
import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from ._atomic_output import (
    PairedOutputDirectory,
    PairedOutputNames,
    verify_paired_output_path_unchanged,
)
from ._npz import ArtifactAdmissionRefusal, refuse
from ._owned_artifacts import (
    OwnedArtifact,
    OwnedArtifactError,
    RetainedArtifactPair,
    commit_owned_artifact,
    create_owned_artifact,
    write_owned_bytes,
)
from .operational import OperationalReasonCode

DirectoryKey: TypeAlias = tuple[str, ...]
EvidenceGroup = Literal["candidate", "aggregate", "private"]
RetainedEvidence: TypeAlias = OwnedArtifact | RetainedArtifactPair


@dataclass(slots=True)
class AuditOwnedDirectory:
    """Retain one exclusively created audit directory and its parent binding."""

    key: DirectoryKey
    path: Path
    parent_fd: int
    name: str
    fd: int
    identity: tuple[int, int, int]
    group: EvidenceGroup
    closed: bool = False

    def verify(self) -> None:
        """Require the descriptor and parent entry to retain the owned identity."""
        if self.closed:
            raise OwnedArtifactError("audit directory is closed")
        bound = os.fstat(self.fd)
        current = os.stat(self.name, dir_fd=self.parent_fd, follow_symlinks=False)
        if (bound.st_dev, bound.st_ino, bound.st_mode) != self.identity:
            raise OwnedArtifactError("audit directory descriptor changed")
        if (current.st_dev, current.st_ino, current.st_mode) != self.identity:
            raise OwnedArtifactError("audit directory entry changed")
        if not stat.S_ISDIR(current.st_mode):
            raise OwnedArtifactError("audit directory entry is not a directory")

    def close(self) -> None:
        """Close retained child and parent descriptors exactly once."""
        if self.closed:
            return
        self.closed = True
        _close_fd(self.fd)
        _close_fd(self.parent_fd)
        self.fd = self.parent_fd = -1


@dataclass(slots=True)
class _EvidenceRecord:
    """Associate retained evidence with its parent, names, and visibility group."""

    parent: DirectoryKey
    retained: RetainedEvidence
    names: tuple[str, ...]
    group: EvidenceGroup


class AuditArtifactRegistry:
    """Own every audit-created child and verify the exact public evidence tree."""

    def __init__(self, root: PairedOutputDirectory) -> None:
        """Bind the caller output root while initially expecting no child entries."""
        self.root = root
        self._directories: dict[DirectoryKey, AuditOwnedDirectory] = {}
        self._evidence: list[_EvidenceRecord] = []
        self._expected: dict[DirectoryKey, set[str]] = {(): set()}
        self.closed = False

    def create_directory(
        self, parent: DirectoryKey, name: str, *, group: EvidenceGroup
    ) -> AuditOwnedDirectory:
        """Exclusively create, bind, and register one product-named directory."""
        _require_plain_name(name)
        key = (*parent, name)
        expected = self._expected.setdefault(parent, set())
        if key in self._directories or name in expected:
            raise ValueError("audit directory entry already registered")
        record = _create_directory(
            self._directory_fd(parent), self.root.path.joinpath(*key), key, name, group
        )
        self._directories[key] = record
        expected.add(name)
        self._expected[key] = set()
        return record

    def paired_output(
        self, directory: AuditOwnedDirectory, names: PairedOutputNames
    ) -> PairedOutputDirectory:
        """Create a paired-output facade over a registered directory duplicate."""
        if self._directories.get(directory.key) is not directory:
            raise ValueError("audit directory is not registered here")
        directory.verify()
        return PairedOutputDirectory._from_descriptor(directory.path, names, os.dup(directory.fd))

    def write_file(
        self,
        parent: DirectoryKey,
        name: str,
        payload: bytes,
        *,
        group: EvidenceGroup,
    ) -> OwnedArtifact:
        """No-clobber publish and register one retained audit file."""
        _require_plain_name(name)
        if name in self._expected.setdefault(parent, set()):
            raise ValueError("audit file entry already registered")
        artifact = create_owned_artifact(self._directory_fd(parent), f".{name}.")
        try:
            write_owned_bytes(artifact, payload)
            commit_owned_artifact(artifact, name)
            artifact.verify()
        except BaseException as exc:
            _cleanup_retained(artifact)
            raise _failure("audit_file_publication_failed", exc) from exc
        self._evidence.append(_EvidenceRecord(parent, artifact, (name,), group))
        self._expected[parent].add(name)
        return artifact

    def register_pair(
        self, parent: DirectoryKey, pair: RetainedArtifactPair, *, group: EvidenceGroup
    ) -> None:
        """Take ownership of one committed pair and register both final names."""
        try:
            _require_pair_parent(pair, self._directory_fd(parent))
            pair.verify()
            first, second = pair.first.final_name, pair.second.final_name
            if first is None or second is None:
                raise OwnedArtifactError("retained pair lacks committed final names")
            names = (first, second)
            if self._expected.setdefault(parent, set()).intersection(names):
                raise OwnedArtifactError("audit pair entry already registered")
        except BaseException as exc:
            _cleanup_retained(pair)
            raise _failure("audit_pair_registration_failed", exc) from exc
        self._evidence.append(_EvidenceRecord(parent, pair, names, group))
        self._expected[parent].update(names)

    def remove_private(self) -> None:
        """Remove exact private evidence and directories before public publication."""
        try:
            private_evidence = [record for record in self._evidence if record.group == "private"]
            for evidence_record in reversed(private_evidence):
                _remove_exact_evidence(evidence_record)
                self._expected[evidence_record.parent].difference_update(evidence_record.names)
            self._evidence = [record for record in self._evidence if record.group != "private"]
            private_directories = [
                record for record in self._directories.values() if record.group == "private"
            ]
            for directory_record in sorted(
                private_directories, key=lambda item: len(item.key), reverse=True
            ):
                self._remove_exact_directory(directory_record)
        except BaseException as exc:
            raise _failure("audit_private_cleanup_failed", exc) from exc

    def verify_candidate_evidence(self) -> None:
        """Reverify every retained candidate comparison or failure artifact."""
        self._verify_group("candidate")

    def verify_aggregate(self) -> None:
        """Reverify the retained aggregate JSON and Markdown pair."""
        self._verify_group("aggregate")

    def verify_public_tree(self) -> None:
        """Require every public directory to have exactly its registered entries."""
        try:
            verify_paired_output_path_unchanged(self.root)
            directories = sorted(
                (item for item in self._directories.values() if item.group != "private"),
                key=lambda item: (len(item.key), item.key),
            )
            for record in directories:
                record.verify()
            for key in [(), *(record.key for record in directories)]:
                actual = set(os.listdir(self._directory_fd(key)))  # noqa: PTH208
                if actual != self._expected.get(key, set()):
                    raise OwnedArtifactError(f"audit public tree changed at {'/'.join(key) or '.'}")
        except ArtifactAdmissionRefusal:
            raise
        except BaseException as exc:
            raise _failure("audit_public_tree_changed", exc) from exc

    def cleanup(self) -> None:
        """Remove private workspace evidence, preserve public evidence, and close descriptors."""
        if self.closed:
            return
        for evidence_record in reversed(self._evidence):
            if evidence_record.group == "private":
                with suppress(OSError, OwnedArtifactError):
                    _remove_exact_evidence(evidence_record)
                    continue
            _cleanup_retained(evidence_record.retained)
        ordered_directories = sorted(
            self._directories.values(), key=lambda item: len(item.key), reverse=True
        )
        for directory_record in ordered_directories:
            _remove_directory_if_exact(directory_record)
        self._close_directories()

    def close(self) -> None:
        """Close every retained artifact and directory without deleting public evidence."""
        if self.closed:
            return
        for record in self._evidence:
            record.retained.close()
        self._close_directories()

    def _verify_group(self, group: EvidenceGroup) -> None:
        """Reverify every retained item assigned to one evidence group."""
        try:
            for record in self._evidence:
                if record.group == group:
                    record.retained.verify()
        except BaseException as exc:
            raise _failure(f"audit_{group}_evidence_changed", exc) from exc

    def _remove_exact_directory(self, record: AuditOwnedDirectory) -> None:
        """Require one private directory to remain exact and empty, then remove it."""
        record.verify()
        if os.listdir(record.fd):  # noqa: PTH208
            raise OwnedArtifactError("private audit directory is not empty")
        os.rmdir(record.name, dir_fd=record.parent_fd)
        record.close()
        self._expected[record.key[:-1]].discard(record.name)
        self._expected.pop(record.key, None)
        self._directories.pop(record.key, None)

    def _directory_fd(self, key: DirectoryKey) -> int:
        """Return the retained descriptor for the root or one registered child."""
        if self.closed:
            raise RuntimeError("audit artifact registry is closed")
        return self.root.directory_fd if not key else self._directories[key].fd

    def _close_directories(self) -> None:
        """Close all retained directory records and mark the registry closed."""
        for record in self._directories.values():
            record.close()
        self.closed = True


def _create_directory(
    parent_fd: int,
    path: Path,
    key: DirectoryKey,
    name: str,
    group: EvidenceGroup,
) -> AuditOwnedDirectory:
    """Exclusively create and bind one child without adopting an existing entry."""
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except OSError as exc:
        raise _failure("audit_directory_exclusive_create_failed", exc) from exc
    parent_copy = descriptor = -1
    identity: tuple[int, int, int] | None = None
    try:
        parent_copy = os.dup(parent_fd)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=parent_fd)
        bound = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        candidate = (bound.st_dev, bound.st_ino, bound.st_mode)
        if candidate != (current.st_dev, current.st_ino, current.st_mode):
            raise OwnedArtifactError("audit directory changed during exclusive bind")
        if not stat.S_ISDIR(bound.st_mode):
            raise OwnedArtifactError("audit child is not a directory")
        identity = candidate
        return AuditOwnedDirectory(key, path, parent_copy, name, descriptor, identity, group)
    except BaseException as exc:
        if identity is not None:
            record = AuditOwnedDirectory(key, path, parent_copy, name, descriptor, identity, group)
            _remove_directory_if_exact(record)
            parent_copy = descriptor = -1
        _close_fd(descriptor)
        _close_fd(parent_copy)
        raise _failure("audit_directory_bind_failed", exc) from exc


def _require_pair_parent(pair: RetainedArtifactPair, target_fd: int) -> None:
    """Require both pair parent descriptors to identify the registered directory."""
    target = os.fstat(target_fd)
    identity = (target.st_dev, target.st_ino)
    for artifact in (pair.first, pair.second):
        parent = os.fstat(artifact.parent_fd)
        if (parent.st_dev, parent.st_ino) != identity:
            raise OwnedArtifactError("retained pair belongs to another directory")


def _remove_exact_evidence(record: _EvidenceRecord) -> None:
    """Require cleanup to unlink exact registered private finals and no replacement."""
    retained = record.retained
    artifacts: tuple[OwnedArtifact, ...]
    if isinstance(retained, RetainedArtifactPair):
        artifacts = (retained.first, retained.second)
    else:
        artifacts = (retained,)
    for artifact, name in zip(artifacts, record.names, strict=True):
        if artifact.final_name != name or artifact.stage != "committed-final":
            raise OwnedArtifactError("private audit evidence is not a committed final")
        artifact.verify()
        os.unlink(name, dir_fd=artifact.parent_fd)
    os.fsync(artifacts[0].parent_fd)
    retained.close()


def _cleanup_retained(retained: RetainedEvidence) -> None:
    """Best-effort private-temporary cleanup and closure of one retained evidence handle."""
    with suppress(OSError):
        retained.cleanup()
    retained.close()


def _remove_directory_if_exact(record: AuditOwnedDirectory) -> None:
    """Best-effort remove one exact empty directory during failure cleanup."""
    try:
        record.verify()
        if not os.listdir(record.fd):  # noqa: PTH208
            os.rmdir(record.name, dir_fd=record.parent_fd)
    except (OSError, OwnedArtifactError):
        pass
    record.close()


def _failure(issue: str, exc: BaseException) -> ArtifactAdmissionRefusal:
    """Map one ownership failure to the frozen audit output reason surface."""
    return refuse(
        OperationalReasonCode.OUTPUT_WRITE_FAILED, issue=issue, exception_type=type(exc).__name__
    )


def _require_plain_name(name: str) -> None:
    """Require one descriptor-relative audit child name."""
    if not name or "/" in name or name in {".", ".."} or "\x00" in name:
        raise ValueError("audit artifact name must be plain")


def _close_fd(descriptor: int) -> None:
    """Best-effort close one retained descriptor."""
    if descriptor >= 0:
        with suppress(OSError):
            os.close(descriptor)


__all__ = ["AuditArtifactRegistry", "AuditOwnedDirectory", "DirectoryKey"]
