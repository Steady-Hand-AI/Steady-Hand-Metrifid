"""Create one immutable, portable Runtime Review output tree.

The scientific evidence is input, but the published decision must remain reviewable after that
input moves.  This module therefore copies every admitted byte into a newly owned tree, verifies the
copy against both the admitted digests and each cell's checksum manifest, and exposes the copied
identities to the receipt builder.  The complete tree is assembled under a private staging name and
made visible as ``runtime_review`` by one directory rename.

No source file is linked into the output.  Every destination is created with ``O_EXCL`` and
``O_NOFOLLOW`` where the platform provides it, so the resulting snapshot contains only independent
regular-file byte copies.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import re
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Protocol

from ..json_values import CanonicalValue, canonical_json_bytes, require_sha256

_CELL_MEMBERS: Final[tuple[str, ...]] = (
    "CHECKSUMS.sha256",
    "fixture.xml",
    "input_manifest.json",
    "model.mjb",
    "result.json",
    "trace.npz",
)
_CHECKSUM_MEMBERS: Final[tuple[str, ...]] = _CELL_MEMBERS[1:]
_STEP_TOKENS: Final[Mapping[str, str]] = {
    "0.004": "0p004",
    "0.002": "0p002",
    "0.001": "0p001",
}
_CANONICAL_CELL_KEYS: Final[tuple[tuple[str, str, int], ...]] = tuple(
    (role, step_dt, repeat_id)
    for role in ("baseline", "candidate")
    for step_dt in ("0.004", "0.002", "0.001")
    for repeat_id in (0, 1)
)
_MEMBER_LIMITS: Final[Mapping[str, int]] = {
    "CHECKSUMS.sha256": 16 * 1024,
    "fixture.xml": 16 * 1024 * 1024,
    "input_manifest.json": 4 * 1024 * 1024,
    "model.mjb": 256 * 1024 * 1024,
    "result.json": 64 * 1024 * 1024,
    "trace.npz": 512 * 1024 * 1024,
}
_ROOT_FILE_LIMITS: Final[Mapping[str, int]] = {
    "admitted_runtime_review_config.json": 4 * 1024 * 1024,
    "runtime_review.json": 64 * 1024 * 1024,
    "runtime_review.md": 64 * 1024 * 1024,
}
_V1_ROOT_MEMBERS: Final[tuple[str, ...]] = (
    "admitted_runtime_review_config.json",
    "evidence",
    "runtime_review.json",
    "runtime_review.md",
)
_V2_ROOT_MEMBERS: Final[tuple[str, ...]] = (*_V1_ROOT_MEMBERS, "profile_identities")
_PROFILE_IDENTITY_MEMBERS: Final[tuple[str, str]] = ("baseline.json", "candidate.json")
_PROFILE_IDENTITY_LIMIT: Final[int] = 4 * 1024 * 1024
_CHECKSUM_LINE = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)\Z")
_DIR_FLAGS: Final[int] = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_READ_FLAGS: Final[int] = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
_WRITE_FLAGS: Final[int] = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
_STAGING_NAME: Final[str] = ".runtime_review.staging"
_FINAL_NAME: Final[str] = "runtime_review"
_LINUX_RENAME_NOREPLACE: Final[int] = 1
_MACOS_RENAME_EXCL: Final[int] = 0x00000004


class RuntimeReviewEvidenceSource(Protocol):
    """Structural interface required from one already admitted evidence cell."""

    @property
    def profile_role(self) -> str:
        """Return the baseline or candidate role."""
        ...

    @property
    def step_dt(self) -> str:
        """Return the exact configured step token."""
        ...

    @property
    def repeat_id(self) -> int:
        """Return the configured repeat identifier."""
        ...

    @property
    def source_directory(self) -> Path:
        """Return the already admitted source directory."""
        ...

    @property
    def member_sha256(self) -> Mapping[str, str]:
        """Return all six admitted source-member digests."""
        ...


class OwnedRuntimeReviewOutputError(ValueError):
    """Raised when an output tree cannot be owned, copied, or published safely."""


@dataclass(frozen=True, slots=True)
class _OwnedDirectoryIdentity:
    """Path and inode of one directory created while owning the configured output path."""

    path: Path
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class OwnedEvidenceMember:
    """Identity of one regular byte copy in the owned evidence snapshot."""

    name: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        """Require one canonical member name, digest, and nonnegative byte count."""
        if self.name not in _CELL_MEMBERS:
            raise ValueError("owned evidence member name is outside the closed registry")
        require_sha256(self.sha256, "owned evidence member SHA-256")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("owned evidence member size must be a nonnegative integer")

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return the canonical receipt representation of this member."""
        return {
            "name": self.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class OwnedEvidenceCell:
    """Canonical owned locator and complete member identities for one evidence slot."""

    profile_role: str
    step_dt: str
    repeat_id: int
    directory: PurePosixPath
    members: tuple[OwnedEvidenceMember, ...]

    def __post_init__(self) -> None:
        """Require the frozen slot locator and complete member order to agree."""
        expected = _owned_cell_locator(self.profile_role, self.step_dt, self.repeat_id)
        if self.directory != expected:
            raise ValueError("owned evidence directory is not the canonical slot locator")
        if tuple(member.name for member in self.members) != _CELL_MEMBERS:
            raise ValueError("owned evidence cell must list all six members in canonical order")

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return the stable receipt representation of the copied cell."""
        return {
            "profile_role": self.profile_role,
            "step_dt": self.step_dt,
            "repeat_id": self.repeat_id,
            "directory": self.directory.as_posix(),
            "members": [member.to_primitive() for member in self.members],
        }


@dataclass(frozen=True, slots=True)
class OwnedProfileIdentity:
    """Canonical owned locator and byte identity for one v2 profile preflight."""

    profile_role: str
    locator: PurePosixPath
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        """Require a semantic role, canonical locator, digest, and positive byte count."""
        if self.profile_role not in {"baseline", "candidate"}:
            raise ValueError("owned profile identity role must be baseline or candidate")
        if self.locator != PurePosixPath("profile_identities", f"{self.profile_role}.json"):
            raise ValueError("owned profile identity locator is not canonical")
        require_sha256(self.sha256, "owned profile identity SHA-256")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise ValueError("owned profile identity size must be a positive integer")

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return the portable role, locator, digest, and byte count."""
        return {
            "profile_role": self.profile_role,
            "locator": self.locator.as_posix(),
            "raw_sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class PublishedRuntimeReviewOutput:
    """Paths and evidence identities from one completely published output."""

    root: Path
    runtime_review_json: Path
    runtime_review_markdown: Path
    admitted_configuration: Path
    evidence_cells: tuple[OwnedEvidenceCell, ...]
    profile_identities: tuple[OwnedProfileIdentity, ...]
    output_device: int
    output_inode: int
    root_device: int
    root_inode: int
    runtime_review_json_sha256: str
    runtime_review_markdown_sha256: str
    admitted_configuration_sha256: str
    _created_directories: tuple[_OwnedDirectoryIdentity, ...]


class OwnedRuntimeReviewStaging:
    """Descriptor-bound staging tree for a single no-clobber publication."""

    __slots__ = (
        "_closed",
        "_configuration_bytes",
        "_created_directories",
        "_output_fd",
        "_published",
        "_staging_fd",
        "evidence_cells",
        "output_dir",
        "profile_identities",
    )

    def __init__(self, output_dir: Path, configuration_bytes: bytes) -> None:
        """Own an absent output root and stage the exact admitted configuration bytes."""
        if type(configuration_bytes) is not bytes:
            raise TypeError("configuration_bytes must be bytes")
        self.output_dir = output_dir.absolute()
        self._configuration_bytes = configuration_bytes
        self._closed = False
        self._published = False
        self.evidence_cells: tuple[OwnedEvidenceCell, ...] = ()
        self.profile_identities: tuple[OwnedProfileIdentity, ...] = ()
        self._output_fd, self._created_directories = _create_owned_output_root(self.output_dir)
        try:
            os.mkdir(_STAGING_NAME, mode=0o700, dir_fd=self._output_fd)
            self._staging_fd = os.open(_STAGING_NAME, _DIR_FLAGS, dir_fd=self._output_fd)
            _write_new_file(
                self._staging_fd,
                "admitted_runtime_review_config.json",
                configuration_bytes,
            )
        except BaseException:
            staging_fd = getattr(self, "_staging_fd", None)
            if isinstance(staging_fd, int):
                os.close(staging_fd)
            os.close(self._output_fd)
            raise

    def copy_evidence_cells(
        self, sources: Sequence[RuntimeReviewEvidenceSource]
    ) -> tuple[OwnedEvidenceCell, ...]:
        """Copy exactly the canonical 12-cell grid and return its receipt identities."""
        if self.evidence_cells:
            raise OwnedRuntimeReviewOutputError("evidence cells were already copied")
        source_by_key: dict[tuple[str, str, int], RuntimeReviewEvidenceSource] = {}
        for source in sources:
            key = (source.profile_role, source.step_dt, source.repeat_id)
            if key in source_by_key:
                raise OwnedRuntimeReviewOutputError(f"duplicate evidence source slot: {key}")
            source_by_key[key] = source
        if tuple(sorted(source_by_key, key=_cell_rank)) != _CANONICAL_CELL_KEYS:
            raise OwnedRuntimeReviewOutputError(
                "owned publication requires the complete canonical 12-cell evidence grid"
            )

        copied = tuple(self._copy_cell(source_by_key[key]) for key in _CANONICAL_CELL_KEYS)
        self.evidence_cells = copied
        return copied

    def copy_profile_identities(
        self,
        sources: Mapping[str, tuple[Path, str]],
    ) -> tuple[OwnedProfileIdentity, ...]:
        """Copy both exact v2 preflight identities into the portable owned tree."""
        if self.profile_identities:
            raise OwnedRuntimeReviewOutputError("profile identities were already copied")
        if set(sources) != {"baseline", "candidate"}:
            raise OwnedRuntimeReviewOutputError(
                "v2 publication requires baseline and candidate profile identities"
            )
        copied: list[OwnedProfileIdentity] = []
        for role in ("baseline", "candidate"):
            source, expected_sha256 = sources[role]
            require_sha256(expected_sha256, f"{role} profile identity SHA-256")
            payload = _read_absolute_regular_file(
                source, _PROFILE_IDENTITY_LIMIT, f"{role} profile identity"
            )
            measured = _sha256(payload)
            if measured != expected_sha256:
                raise OwnedRuntimeReviewOutputError(
                    f"{role} profile identity changed after configuration admission"
                )
            locator = PurePosixPath("profile_identities", f"{role}.json")
            parent_fd = _make_directories(self._staging_fd, locator.parent)
            try:
                _write_new_file(parent_fd, locator.name, payload)
                observed = _read_regular_member(
                    parent_fd,
                    locator.name,
                    _PROFILE_IDENTITY_LIMIT,
                    require_single_link=True,
                )
            finally:
                os.close(parent_fd)
            if observed != payload:
                raise OwnedRuntimeReviewOutputError(
                    f"owned {role} profile identity differs from admitted bytes"
                )
            copied.append(OwnedProfileIdentity(role, locator, measured, len(payload)))
        self.profile_identities = tuple(copied)
        return self.profile_identities

    def publish(
        self,
        receipt: Mapping[str, CanonicalValue],
        markdown: str,
        *,
        prepublication_validator: Callable[[Path], object] | None = None,
    ) -> PublishedRuntimeReviewOutput:
        """Independently validate staging, then atomically expose it without overwriting."""
        if self._published:
            raise OwnedRuntimeReviewOutputError("runtime review output was already published")
        if len(self.evidence_cells) != len(_CANONICAL_CELL_KEYS):
            raise OwnedRuntimeReviewOutputError(
                "all evidence cells must be copied before publication"
            )
        if type(markdown) is not str:
            raise TypeError("markdown must be text")
        json_bytes = canonical_json_bytes(dict(receipt)) + b"\n"
        markdown_bytes = markdown.encode("utf-8")
        _write_new_file(
            self._staging_fd,
            "runtime_review.json",
            json_bytes,
        )
        _write_new_file(self._staging_fd, "runtime_review.md", markdown_bytes)
        self._verify_staging()
        if prepublication_validator is not None:
            prepublication_validator(self.output_dir / _STAGING_NAME)
        try:
            os.stat(_FINAL_NAME, dir_fd=self._output_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise OwnedRuntimeReviewOutputError("runtime_review output already exists")
        os.fsync(self._staging_fd)
        try:
            _rename_directory_noreplace(self._output_fd, _STAGING_NAME, _FINAL_NAME)
        except FileExistsError as exc:
            raise OwnedRuntimeReviewOutputError("runtime_review output already exists") from exc
        except OSError as exc:
            raise OwnedRuntimeReviewOutputError("atomic runtime_review publication failed") from exc
        self._published = True
        os.fsync(self._output_fd)
        root = self.output_dir / _FINAL_NAME
        output_metadata = os.fstat(self._output_fd)
        root_metadata = os.fstat(self._staging_fd)
        published = PublishedRuntimeReviewOutput(
            root=root,
            runtime_review_json=root / "runtime_review.json",
            runtime_review_markdown=root / "runtime_review.md",
            admitted_configuration=root / "admitted_runtime_review_config.json",
            evidence_cells=self.evidence_cells,
            profile_identities=self.profile_identities,
            output_device=output_metadata.st_dev,
            output_inode=output_metadata.st_ino,
            root_device=root_metadata.st_dev,
            root_inode=root_metadata.st_ino,
            runtime_review_json_sha256=_sha256(json_bytes),
            runtime_review_markdown_sha256=_sha256(markdown_bytes),
            admitted_configuration_sha256=_sha256(self._configuration_bytes),
            _created_directories=self._created_directories,
        )
        self.close()
        return published

    def _copy_cell(self, source: RuntimeReviewEvidenceSource) -> OwnedEvidenceCell:
        """Copy and independently checksum one source cell into its canonical owned slot."""
        locator = _owned_cell_locator(source.profile_role, source.step_dt, source.repeat_id)
        destination_fd = _make_directories(self._staging_fd, locator)
        source_fd = _open_source_cell(source.source_directory)
        try:
            names = tuple(sorted(os.listdir(source_fd)))  # noqa: PTH208 - descriptor confinement
            if names != tuple(sorted(_CELL_MEMBERS)):
                raise OwnedRuntimeReviewOutputError(
                    f"evidence source {source.source_directory} does not contain exactly six members"
                )
            payloads = {
                name: _read_regular_member(source_fd, name, _MEMBER_LIMITS[name])
                for name in _CELL_MEMBERS
            }
            measured = {name: _sha256(payload) for name, payload in payloads.items()}
            expected = dict(source.member_sha256)
            if set(expected) != set(_CELL_MEMBERS):
                raise OwnedRuntimeReviewOutputError(
                    "admitted evidence member identity is not the exact six-member set"
                )
            for name in _CELL_MEMBERS:
                if require_sha256(expected[name], f"{name} SHA-256") != measured[name]:
                    raise OwnedRuntimeReviewOutputError(
                        f"evidence source member changed after admission: {name}"
                    )
            _verify_checksum_manifest(payloads, measured)
            for name in _CELL_MEMBERS:
                _write_new_file(destination_fd, name, payloads[name])
            copied = {
                name: _read_regular_member(
                    destination_fd,
                    name,
                    _MEMBER_LIMITS[name],
                    require_single_link=True,
                )
                for name in _CELL_MEMBERS
            }
            copied_hashes = {name: _sha256(payload) for name, payload in copied.items()}
            if copied_hashes != measured:
                raise OwnedRuntimeReviewOutputError(
                    "owned evidence copy differs from admitted bytes"
                )
            _verify_checksum_manifest(copied, copied_hashes)
            os.fsync(destination_fd)
        finally:
            os.close(source_fd)
            os.close(destination_fd)
        members = tuple(
            OwnedEvidenceMember(name, measured[name], len(payloads[name])) for name in _CELL_MEMBERS
        )
        return OwnedEvidenceCell(
            source.profile_role,
            source.step_dt,
            source.repeat_id,
            locator,
            members,
        )

    def _verify_staging(self) -> None:
        """Re-read every decision-bearing staged byte before the final rename."""
        retained_config = _read_regular_member(
            self._staging_fd,
            "admitted_runtime_review_config.json",
            max(1, len(self._configuration_bytes)),
            require_single_link=True,
        )
        if retained_config != self._configuration_bytes:
            raise OwnedRuntimeReviewOutputError(
                "retained configuration bytes changed during staging"
            )
        for cell in self.evidence_cells:
            cell_fd = _walk_directory(self._staging_fd, cell.directory)
            try:
                expected = {member.name: member.sha256 for member in cell.members}
                payloads = {
                    name: _read_regular_member(
                        cell_fd,
                        name,
                        _MEMBER_LIMITS[name],
                        require_single_link=True,
                    )
                    for name in _CELL_MEMBERS
                }
                measured = {name: _sha256(payload) for name, payload in payloads.items()}
                if measured != expected:
                    raise OwnedRuntimeReviewOutputError(
                        "owned evidence member changed before publication"
                    )
                _verify_checksum_manifest(payloads, measured)
            finally:
                os.close(cell_fd)
        for identity in self.profile_identities:
            payload = _read_regular_member(
                self._staging_fd,
                identity.locator.as_posix(),
                _PROFILE_IDENTITY_LIMIT,
                require_single_link=True,
            )
            if _sha256(payload) != identity.sha256 or len(payload) != identity.size_bytes:
                raise OwnedRuntimeReviewOutputError(
                    f"owned profile identity changed before publication: {identity.profile_role}"
                )
        _read_regular_member(
            self._staging_fd,
            "runtime_review.json",
            64 * 1024 * 1024,
            require_single_link=True,
        )
        _read_regular_member(
            self._staging_fd,
            "runtime_review.md",
            64 * 1024 * 1024,
            require_single_link=True,
        )

    def close(self) -> None:
        """Close descriptors while preserving unpublished bytes against cleanup substitution."""
        if self._closed:
            return
        self._closed = True
        os.close(self._staging_fd)
        os.close(self._output_fd)

    def __enter__(self) -> OwnedRuntimeReviewStaging:
        """Enter the staging context."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release descriptors without recursively deleting unpublished evidence."""
        self.close()


def prepare_owned_runtime_review_output(
    output_dir: Path,
    configuration_bytes: bytes,
) -> OwnedRuntimeReviewStaging:
    """Own an absent output directory and return its private staging handle."""
    return OwnedRuntimeReviewStaging(output_dir, configuration_bytes)


def read_complete_owned_runtime_review_tree(root: Path) -> dict[str, bytes]:
    """Read root documents while requiring the complete descriptor-bound owned tree shape."""
    root_fd = _open_absolute_directory(root)
    try:
        return _validate_complete_owned_tree(root_fd)
    finally:
        os.close(root_fd)


def validate_owned_evidence_cell(root: Path, cell: OwnedEvidenceCell) -> Path:
    """Re-read one published owned cell and bind every recorded member to its exact bytes."""
    root_fd = _open_absolute_directory(root)
    try:
        cell_fd = _walk_directory(root_fd, cell.directory)
    finally:
        os.close(root_fd)
    try:
        names = tuple(sorted(os.listdir(cell_fd)))  # noqa: PTH208 - descriptor confinement
        if names != tuple(sorted(_CELL_MEMBERS)):
            raise OwnedRuntimeReviewOutputError(
                "published owned evidence does not contain exactly six members"
            )
        payloads = {
            name: _read_regular_member(
                cell_fd,
                name,
                _MEMBER_LIMITS[name],
                require_single_link=True,
            )
            for name in _CELL_MEMBERS
        }
        measured = {name: _sha256(payload) for name, payload in payloads.items()}
        expected = {member.name: member.sha256 for member in cell.members}
        sizes = {member.name: member.size_bytes for member in cell.members}
        if tuple(expected) != _CELL_MEMBERS:
            raise OwnedRuntimeReviewOutputError(
                "published owned evidence member identities are not in canonical order"
            )
        if measured != expected or any(
            len(payloads[name]) != sizes[name] for name in _CELL_MEMBERS
        ):
            raise OwnedRuntimeReviewOutputError(
                "published owned evidence bytes do not match the receipt identities"
            )
        _verify_checksum_manifest(payloads, measured)
    finally:
        os.close(cell_fd)
    return root / cell.directory


def verify_published_runtime_review_output(output: PublishedRuntimeReviewOutput) -> None:
    """Rebind the final pathname and recheck every published byte at the success boundary."""
    root_payloads = _read_bound_published_root(output)
    expected_hashes = {
        "runtime_review.json": output.runtime_review_json_sha256,
        "runtime_review.md": output.runtime_review_markdown_sha256,
        "admitted_runtime_review_config.json": output.admitted_configuration_sha256,
    }
    for name, expected_sha256 in expected_hashes.items():
        if _sha256(root_payloads[name]) != expected_sha256:
            raise OwnedRuntimeReviewOutputError(f"published output bytes changed: {name}")
    for cell in output.evidence_cells:
        validate_owned_evidence_cell(output.root, cell)
    for identity in output.profile_identities:
        payload = root_payloads.get(identity.locator.as_posix())
        if payload is None or _sha256(payload) != identity.sha256:
            raise OwnedRuntimeReviewOutputError(
                f"published profile identity changed: {identity.profile_role}"
            )
    if _read_bound_published_root(output) != root_payloads:
        raise OwnedRuntimeReviewOutputError(
            "published runtime_review root bytes changed during final verification"
        )


def _read_bound_published_root(output: PublishedRuntimeReviewOutput) -> dict[str, bytes]:
    """Rebind one publication identity and validate its complete closed tree."""
    root_fd = _open_absolute_directory(output.root)
    try:
        metadata = os.fstat(root_fd)
        if (metadata.st_dev, metadata.st_ino) != (output.root_device, output.root_inode):
            raise OwnedRuntimeReviewOutputError("published runtime_review pathname was replaced")
        return _validate_complete_owned_tree(root_fd)
    finally:
        os.close(root_fd)


def _cell_rank(key: tuple[str, str, int]) -> tuple[int, int, int]:
    """Return the frozen canonical ordering key for one evidence slot."""
    role, step_dt, repeat_id = key
    role_rank = {"baseline": 0, "candidate": 1}.get(role, 99)
    step_rank = {"0.004": 0, "0.002": 1, "0.001": 2}.get(step_dt, 99)
    return role_rank, step_rank, repeat_id


def _owned_cell_locator(profile_role: str, step_dt: str, repeat_id: int) -> PurePosixPath:
    """Construct the exact canonical destination locator for one admitted slot."""
    if profile_role not in {"baseline", "candidate"}:
        raise OwnedRuntimeReviewOutputError("unknown evidence profile role")
    try:
        step_token = _STEP_TOKENS[step_dt]
    except KeyError as exc:
        raise OwnedRuntimeReviewOutputError("unknown evidence step size") from exc
    if repeat_id not in {0, 1} or type(repeat_id) is not int:
        raise OwnedRuntimeReviewOutputError("unknown evidence repeat ID")
    return PurePosixPath("evidence", profile_role, step_token, f"repeat_{repeat_id}")


def _verify_checksum_manifest(payloads: Mapping[str, bytes], measured: Mapping[str, str]) -> None:
    """Require one exact manifest over the five non-manifest members."""
    try:
        text = payloads["CHECKSUMS.sha256"].decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise OwnedRuntimeReviewOutputError("cell checksum manifest is not UTF-8") from exc
    lines = text.splitlines()
    entries: dict[str, str] = {}
    for line in lines:
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise OwnedRuntimeReviewOutputError("cell checksum manifest has invalid syntax")
        digest, name = match.groups()
        if name in entries:
            raise OwnedRuntimeReviewOutputError("cell checksum manifest repeats a member")
        entries[name] = digest
    if set(entries) != set(_CHECKSUM_MEMBERS):
        raise OwnedRuntimeReviewOutputError(
            "cell checksum manifest must cover the five evidence members exactly once"
        )
    if any(entries[name] != measured[name] for name in _CHECKSUM_MEMBERS):
        raise OwnedRuntimeReviewOutputError("cell checksum manifest does not match member bytes")


def _open_source_cell(path: Path) -> int:
    """Open one already admitted source cell without following its final component."""
    try:
        descriptor = os.open(path, _DIR_FLAGS)
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise OwnedRuntimeReviewOutputError("evidence source directory is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise OwnedRuntimeReviewOutputError("evidence source is not a directory")
    return descriptor


def _read_absolute_regular_file(path: Path, maximum: int, field: str) -> bytes:
    """Read one absolute nonsymlink regular file through a retained parent descriptor."""
    if not path.is_absolute() or not path.name:
        raise OwnedRuntimeReviewOutputError(f"{field} must use an absolute non-root path")
    parent_fd = _open_absolute_directory(path.parent)
    try:
        return _read_regular_member(
            parent_fd,
            path.name,
            maximum,
            require_single_link=False,
        )
    finally:
        os.close(parent_fd)


def _create_owned_output_root(
    path: Path,
) -> tuple[int, tuple[_OwnedDirectoryIdentity, ...]]:
    """Create an absent possibly nested output path through no-follow descriptors."""
    absolute = path.absolute()
    descriptor = os.open(absolute.anchor, _DIR_FLAGS)
    current_path = Path(absolute.anchor)
    created: list[_OwnedDirectoryIdentity] = []
    try:
        for index, component in enumerate(absolute.parts[1:]):
            final = index == len(absolute.parts[1:]) - 1
            current_path = current_path / component
            try:
                metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
                metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
                created.append(
                    _OwnedDirectoryIdentity(current_path, metadata.st_dev, metadata.st_ino)
                )
            else:
                if final:
                    raise OwnedRuntimeReviewOutputError(
                        "runtime review output directory already exists; output is never overwritten"
                    )
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise OwnedRuntimeReviewOutputError(
                        "runtime review output path contains an unsafe ancestor"
                    )
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            bound = os.fstat(child)
            if (bound.st_dev, bound.st_ino) != (metadata.st_dev, metadata.st_ino):
                os.close(child)
                raise OwnedRuntimeReviewOutputError("runtime review output ancestor was replaced")
            os.close(descriptor)
            descriptor = child
        if not created or created[-1].path != absolute:
            raise OwnedRuntimeReviewOutputError("runtime review output root was not newly owned")
        return descriptor, tuple(created)
    except BaseException:
        os.close(descriptor)
        raise


def _open_absolute_directory(path: Path) -> int:
    """Open an absolute directory one no-follow component at a time."""
    absolute = path.absolute()
    descriptor = os.open(absolute.anchor, _DIR_FLAGS)
    try:
        for component in absolute.parts[1:]:
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _make_directories(root_fd: int, locator: PurePosixPath) -> int:
    """Create and open a fresh confined directory chain."""
    current = os.dup(root_fd)
    try:
        for component in locator.parts:
            try:
                os.mkdir(component, mode=0o700, dir_fd=current)
            except FileExistsError:
                pass
            child = os.open(component, _DIR_FLAGS, dir_fd=current)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _walk_directory(root_fd: int, locator: PurePosixPath) -> int:
    """Open an existing confined directory chain without following links."""
    current = os.dup(root_fd)
    try:
        for component in locator.parts:
            child = os.open(component, _DIR_FLAGS, dir_fd=current)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _validate_complete_owned_tree(root_fd: int) -> dict[str, bytes]:
    """Require the exact complete Runtime Review hierarchy through retained descriptors."""
    actual_root_members = tuple(sorted(os.listdir(root_fd)))
    if actual_root_members == tuple(sorted(_V1_ROOT_MEMBERS)):
        has_profile_identities = False
    elif actual_root_members == tuple(sorted(_V2_ROOT_MEMBERS)):
        has_profile_identities = True
    else:
        _require_exact_directory_members(root_fd, _V1_ROOT_MEMBERS, "runtime_review root")
        raise AssertionError("unreachable invalid runtime-review root")
    root_payloads = {
        name: _read_regular_member(
            root_fd,
            name,
            limit,
            require_single_link=True,
        )
        for name, limit in _ROOT_FILE_LIMITS.items()
    }
    if has_profile_identities:
        identities_fd = _open_child_directory(
            root_fd, "profile_identities", "runtime_review profile identities"
        )
        try:
            _require_exact_directory_members(
                identities_fd,
                _PROFILE_IDENTITY_MEMBERS,
                "runtime_review profile identities",
            )
            for name in _PROFILE_IDENTITY_MEMBERS:
                locator = f"profile_identities/{name}"
                root_payloads[locator] = _read_regular_member(
                    identities_fd,
                    name,
                    _PROFILE_IDENTITY_LIMIT,
                    require_single_link=True,
                )
        finally:
            os.close(identities_fd)
    evidence_fd = _open_child_directory(root_fd, "evidence", "runtime_review evidence")
    try:
        _require_exact_directory_members(
            evidence_fd,
            ("baseline", "candidate"),
            "runtime_review evidence",
        )
        for role in ("baseline", "candidate"):
            role_fd = _open_child_directory(evidence_fd, role, f"evidence/{role}")
            try:
                step_names = tuple(_STEP_TOKENS[step_dt] for step_dt in _STEP_TOKENS)
                _require_exact_directory_members(role_fd, step_names, f"evidence/{role}")
                for step_name in step_names:
                    step_fd = _open_child_directory(
                        role_fd,
                        step_name,
                        f"evidence/{role}/{step_name}",
                    )
                    try:
                        _require_exact_directory_members(
                            step_fd,
                            ("repeat_0", "repeat_1"),
                            f"evidence/{role}/{step_name}",
                        )
                        for repeat_name in ("repeat_0", "repeat_1"):
                            cell_label = f"evidence/{role}/{step_name}/{repeat_name}"
                            cell_fd = _open_child_directory(step_fd, repeat_name, cell_label)
                            try:
                                _require_exact_directory_members(
                                    cell_fd,
                                    _CELL_MEMBERS,
                                    cell_label,
                                )
                                for name in _CELL_MEMBERS:
                                    member_fd = _validate_regular_member(
                                        cell_fd,
                                        name,
                                        _MEMBER_LIMITS[name],
                                        require_single_link=True,
                                    )
                                    os.close(member_fd)
                            finally:
                                os.close(cell_fd)
                    finally:
                        os.close(step_fd)
            finally:
                os.close(role_fd)
    finally:
        os.close(evidence_fd)
    return root_payloads


def _require_exact_directory_members(
    descriptor: int,
    expected: Sequence[str],
    label: str,
) -> None:
    """Require one bounded directory to contain exactly its closed member registry."""
    try:
        actual = tuple(sorted(os.listdir(descriptor)))
    except OSError as exc:
        raise OwnedRuntimeReviewOutputError(f"{label} could not be listed safely") from exc
    frozen = tuple(sorted(expected))
    if actual != frozen:
        missing = sorted(set(frozen) - set(actual))
        extra = sorted(set(actual) - set(frozen))
        raise OwnedRuntimeReviewOutputError(
            f"{label} has an invalid closed shape; missing={missing}, extra={extra}"
        )


def _open_child_directory(parent_fd: int, name: str, label: str) -> int:
    """Open one expected child directory without following its pathname."""
    try:
        descriptor = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise OwnedRuntimeReviewOutputError(f"{label} is not a nonsymlink directory") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise OwnedRuntimeReviewOutputError(f"{label} is not a directory")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _validate_regular_member(
    parent_fd: int,
    name: str,
    limit: int,
    *,
    require_single_link: bool = False,
) -> int:
    """Open and validate one bounded regular member, returning its retained descriptor."""
    try:
        descriptor = os.open(name, _READ_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise OwnedRuntimeReviewOutputError(f"evidence member is unavailable: {name}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OwnedRuntimeReviewOutputError(f"evidence member is not regular: {name}")
        if require_single_link and metadata.st_nlink != 1:
            raise OwnedRuntimeReviewOutputError(
                f"owned evidence member does not have exactly one link: {name}"
            )
        if metadata.st_size > limit:
            raise OwnedRuntimeReviewOutputError(f"evidence member exceeds its bound: {name}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_regular_member(
    parent_fd: int,
    name: str,
    limit: int,
    *,
    require_single_link: bool = False,
) -> bytes:
    """Read one bounded regular member through a retained parent descriptor."""
    descriptor = _validate_regular_member(
        parent_fd,
        name,
        limit,
        require_single_link=require_single_link,
    )
    try:
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        payload = b"".join(chunks)
        if len(payload) > limit:
            raise OwnedRuntimeReviewOutputError(f"evidence member exceeds its bound: {name}")
        return payload
    finally:
        os.close(descriptor)


def _write_new_file(parent_fd: int, name: str, payload: bytes) -> None:
    """Write, flush, and seal one new regular file without following or replacing."""
    try:
        descriptor = os.open(name, _WRITE_FLAGS, 0o600, dir_fd=parent_fd)
    except OSError as exc:
        raise OwnedRuntimeReviewOutputError(
            f"owned output member could not be created: {name}"
        ) from exc
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            written += os.write(descriptor, view[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(parent_fd: int, source: str, destination: str) -> None:
    """Atomically rename one directory only when the destination is still absent."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        symbol = "renameatx_np"
        flags = _MACOS_RENAME_EXCL
    elif sys.platform.startswith("linux"):
        symbol = "renameat2"
        flags = _LINUX_RENAME_NOREPLACE
    else:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename is unsupported")
    try:
        rename = getattr(library, symbol)
    except AttributeError as exc:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable") from exc
    rename.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = rename(
        parent_fd,
        os.fsencode(source),
        parent_fd,
        os.fsencode(destination),
        flags,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error, os.strerror(error), destination)
    raise OSError(error, os.strerror(error), destination)


def _sha256(payload: bytes) -> str:
    """Return lowercase SHA-256 for one exact byte string."""
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "OwnedEvidenceCell",
    "OwnedEvidenceMember",
    "OwnedRuntimeReviewOutputError",
    "OwnedRuntimeReviewStaging",
    "PublishedRuntimeReviewOutput",
    "RuntimeReviewEvidenceSource",
    "prepare_owned_runtime_review_output",
    "read_complete_owned_runtime_review_tree",
    "validate_owned_evidence_cell",
    "verify_published_runtime_review_output",
]
