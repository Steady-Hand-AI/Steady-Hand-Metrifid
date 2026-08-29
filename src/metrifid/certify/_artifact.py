"""Complete compiled-artifact identity over every serialized MJB byte.

The identity is SHA-256 over the complete ``mj_saveModel`` output. Nothing is projected,
normalized, rounded or tolerated. The buffer is streamed straight into a private file so a
role's bytes never have to stay resident while the other role compiles.

That private file is then retained only through its own descriptor: its directory entry is
removed as soon as the bytes are sealed, so the object every later consumer reads has no
pathname a same-user process could rename or replace. Consumers reach it through positional
descriptor reads, or through the operating system's own descriptor path when MuJoCo needs a
name, and the retained bytes are reverified before any completed decision is published.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mujoco  # type: ignore[import-untyped]

import hashlib
import os
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self, cast

from .._model_closure import ModelAdmissionRefusal, ModelRole, refuse
from ..json_values import CanonicalValue, require_sha256
from ..operational import OperationalReasonCode, _require_exact_object_fields

ARTIFACT_IDENTITY_SCHEMA = "metrifid.compiled_artifact_identity"
ARTIFACT_IDENTITY_SCHEMA_VERSION = 1
COMPLETE_MJB_METHOD = "MUJOCO_COMPLETE_MJB_SHA256"

# Post-compilation serialization and diagnostic bound. It is not a compile-memory guarantee:
# a model that needs more than this to compile fails inside MuJoCo long before this check.
MAX_SERIALIZED_ARTIFACT_BYTES = 512 * 1024 * 1024

_MJB_MAGIC = 54321
_HEADER_WORD_COUNT = 5
_HEADER_FORMAT = "=5i"
_HEADER_BYTES = struct.calcsize(_HEADER_FORMAT)

_ARTIFACT_TEXT_MEMBERS = (
    "schema",
    "method",
    "mjb_sha256",
    "magic_hex",
    "runtime_identity_sha256",
)
_ARTIFACT_INT_MEMBERS = (
    "mjb_size_bytes",
    "magic_decimal",
    "sizeof_mjtnum",
    "mujoco_version_integer",
)
_ARTIFACT_MEMBERS = (
    *_ARTIFACT_TEXT_MEMBERS,
    *_ARTIFACT_INT_MEMBERS,
    "schema_version",
    "header_words",
)


def _validate_artifact_header(identity: CompiledArtifactIdentity) -> None:
    """Validate compiled artifact header words and their scalar restatements."""
    if len(identity.header_words) != _HEADER_WORD_COUNT:
        raise ValueError("header_words must hold exactly five native integers")
    if any(type(word) is not int for word in identity.header_words):
        raise TypeError("header_words must hold native integers")
    if identity.magic_decimal != identity.header_words[0]:
        raise ValueError("magic_decimal must equal the first header word")
    if identity.magic_hex != f"0x{identity.header_words[0]:08x}":
        raise ValueError("magic_hex must render the first header word")
    if identity.sizeof_mjtnum != identity.header_words[1]:
        raise ValueError("sizeof_mjtnum must equal the second header word")
    if identity.mujoco_version_integer != identity.header_words[3]:
        raise ValueError("mujoco_version_integer must equal the fourth header word")


@dataclass(frozen=True, slots=True)
class CompiledArtifactIdentity:
    """One complete MJB artifact identity plus its recorded header words."""

    schema: str
    schema_version: int
    method: str
    mjb_sha256: str
    mjb_size_bytes: int
    header_words: tuple[int, ...]
    magic_decimal: int
    magic_hex: str
    sizeof_mjtnum: int
    mujoco_version_integer: int
    runtime_identity_sha256: str

    def __post_init__(self) -> None:
        """Validate MJB size, digest, header words, and runtime-identity binding."""
        if self.schema != ARTIFACT_IDENTITY_SCHEMA:
            raise ValueError("invalid compiled artifact identity schema")
        if (
            type(self.schema_version) is not int
            or self.schema_version != ARTIFACT_IDENTITY_SCHEMA_VERSION
        ):
            raise ValueError("invalid compiled artifact identity schema_version")
        if self.method != COMPLETE_MJB_METHOD:
            raise ValueError("compiled artifact identity uses one frozen method")
        require_sha256(self.mjb_sha256, "mjb_sha256")
        require_sha256(self.runtime_identity_sha256, "runtime_identity_sha256")
        if self.mjb_size_bytes <= 0:
            raise ValueError("mjb_size_bytes must be positive")
        _validate_artifact_header(self)

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Parse one compiled artifact identity strictly, requiring its exact member set."""
        obj = _require_exact_object_fields(
            value, set(_ARTIFACT_MEMBERS), "CompiledArtifactIdentity"
        )
        words = obj["header_words"]
        if type(words) is not list or any(type(word) is not int for word in words):
            raise TypeError("header_words must be an array of native integers")
        for name in _ARTIFACT_TEXT_MEMBERS:
            if type(obj[name]) is not str:
                raise TypeError(f"{name} must be a string")
        for name in _ARTIFACT_INT_MEMBERS:
            if type(obj[name]) is not int:
                raise TypeError(f"{name} must be an integer")
        if type(obj["schema_version"]) is not int:
            raise TypeError("schema_version must be an integer")
        return cls(
            schema=cast(str, obj["schema"]),
            schema_version=obj["schema_version"],
            method=cast(str, obj["method"]),
            mjb_sha256=cast(str, obj["mjb_sha256"]),
            mjb_size_bytes=cast(int, obj["mjb_size_bytes"]),
            header_words=tuple(int(word) for word in words),
            magic_decimal=cast(int, obj["magic_decimal"]),
            magic_hex=cast(str, obj["magic_hex"]),
            sizeof_mjtnum=cast(int, obj["sizeof_mjtnum"]),
            mujoco_version_integer=cast(int, obj["mujoco_version_integer"]),
            runtime_identity_sha256=cast(str, obj["runtime_identity_sha256"]),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the complete compiled-artifact identity and header evidence."""
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "method": self.method,
            "mjb_sha256": self.mjb_sha256,
            "mjb_size_bytes": self.mjb_size_bytes,
            # Words 3 and 5 are recorded exactly as MuJoCo wrote them. They are opaque
            # build/layout words here; naming them would invent a contract MuJoCo has not made.
            "header_words": list(self.header_words),
            "magic_decimal": self.magic_decimal,
            "magic_hex": self.magic_hex,
            "sizeof_mjtnum": self.sizeof_mjtnum,
            "mujoco_version_integer": self.mujoco_version_integer,
            "runtime_identity_sha256": self.runtime_identity_sha256,
        }


# The operating system's own view of an open descriptor. Linux publishes both; macOS publishes
# only /dev/fd. Each candidate is admitted only after it is proven to name the exact retained
# object, and a platform offering neither is refused rather than silently sent back to a pathname.
_DESCRIPTOR_PATH_TEMPLATES: Final = ("/proc/self/fd/{fd}", "/dev/fd/{fd}")
_ARTIFACT_READ_CHUNK_BYTES: Final = 1 << 20
_MODEL_LOAD_MAX_ATTEMPTS: Final = 2
_RETRYABLE_MODEL_LOAD_MESSAGE: Final = "mj_loadModel: failed to load from mjb"


class RetainedCompiledArtifact:
    """One sealed complete MJB owned solely by a retained descriptor, with no directory entry.

    The bytes are measured once, while they are being written, and the name is removed
    immediately afterwards. Every later read goes through this descriptor, so a distinct process
    running as the same operating-system user has no name left to rename or replace, and cannot
    substitute a different compiled object for this one.

    It can still mutate this object in place: on Linux by reopening it through ``/proc/<pid>/fd``,
    and on any platform through a descriptor obtained during the brief window before the name was
    removed. That residual case is detected rather than prevented, and detecting it is exactly what
    :meth:`verify` is for. Callers reverify on both sides of every read and again before any
    completed decision is published, so a mutated subject fails the run closed.
    """

    __slots__ = ("closed", "device", "fd", "inode", "mjb_sha256", "mjb_size_bytes", "role")

    def __init__(
        self,
        fd: int,
        *,
        role: ModelRole,
        device: int,
        inode: int,
        mjb_size_bytes: int,
        mjb_sha256: str,
    ) -> None:
        """Retain one already-sealed and already-unlinked private artifact descriptor."""
        self.fd = fd
        self.role = role
        self.device = device
        self.inode = inode
        self.mjb_size_bytes = mjb_size_bytes
        self.mjb_sha256 = mjb_sha256
        self.closed = False

    def _refuse(self, issue: str, **evidence: CanonicalValue) -> ModelAdmissionRefusal:
        """Build one role-local refusal for a broken retained-subject binding."""
        return refuse(
            OperationalReasonCode.COMPILED_ARTIFACT_INVALID,
            self.role,
            issue=issue,
            **evidence,
        )

    def _require_open(self) -> None:
        """Refuse any read or verification after the retained descriptor is released."""
        if self.closed or self.fd < 0:
            raise self._refuse("retained_artifact_descriptor_released")

    def descriptor_path(self) -> str:
        """Return an operating-system descriptor path proven to name this exact object.

        MuJoCo's loader accepts only a pathname. This is the one pathname whose meaning a
        same-user process cannot change: it is resolved by the kernel from this process's own
        descriptor table rather than from a directory entry.
        """
        self._require_open()
        examined: list[CanonicalValue] = []
        for template in _DESCRIPTOR_PATH_TEMPLATES:
            candidate = template.format(fd=self.fd)
            examined.append(candidate)
            probe = -1
            try:
                probe = os.open(candidate, os.O_RDONLY)
                metadata = os.fstat(probe)
            except OSError:
                continue
            finally:
                if probe >= 0:
                    os.close(probe)
            if not stat.S_ISREG(metadata.st_mode):
                continue
            if (metadata.st_dev, metadata.st_ino) != (self.device, self.inode):
                continue
            # Opening /dev/fd/N duplicates this descriptor on macOS rather than reopening the
            # object, so the caller inherits our file offset. Rewinding here is what lets a
            # reader that does not seek for itself still see the artifact from its first byte.
            os.lseek(self.fd, 0, os.SEEK_SET)
            return candidate
        raise self._refuse(
            "retained_artifact_descriptor_path_unavailable",
            examined_descriptor_paths=examined,
        )

    def read_exact(self, offset: int, span: int) -> bytes:
        """Read one exact positional span through the retained descriptor."""
        self._require_open()
        chunk = os.pread(self.fd, span, offset)
        if len(chunk) != span:
            raise self._refuse(
                "retained_artifact_shorter_than_measured",
                requested_offset=offset,
                requested_bytes=span,
                available_bytes=len(chunk),
            )
        return chunk

    def measured_digest(self) -> str:
        """Recompute SHA-256 over every byte the retained descriptor currently holds."""
        self._require_open()
        digest = hashlib.sha256()
        offset = 0
        while offset < self.mjb_size_bytes:
            span = min(_ARTIFACT_READ_CHUNK_BYTES, self.mjb_size_bytes - offset)
            digest.update(self.read_exact(offset, span))
            offset += span
        if os.pread(self.fd, 1, offset):
            raise self._refuse("retained_artifact_longer_than_measured")
        return digest.hexdigest()

    def verify(self) -> None:
        """Require this descriptor to still hold the exact measured, nameless subject."""
        self._require_open()
        metadata = os.fstat(self.fd)
        if not stat.S_ISREG(metadata.st_mode) or (metadata.st_dev, metadata.st_ino) != (
            self.device,
            self.inode,
        ):
            raise self._refuse("retained_artifact_identity_changed")
        if metadata.st_nlink != 0:
            raise self._refuse(
                "retained_artifact_name_reappeared",
                link_count=int(metadata.st_nlink),
            )
        if metadata.st_size != self.mjb_size_bytes:
            raise self._refuse(
                "retained_artifact_size_changed",
                expected_mjb_size_bytes=self.mjb_size_bytes,
                observed_mjb_size_bytes=int(metadata.st_size),
            )
        observed = self.measured_digest()
        if observed != self.mjb_sha256:
            raise self._refuse(
                "retained_artifact_bytes_changed",
                expected_mjb_sha256=self.mjb_sha256,
                observed_mjb_sha256=observed,
            )

    def close(self) -> None:
        """Release the retained descriptor exactly once, discarding the nameless object."""
        if self.closed:
            return
        self.closed = True
        if self.fd >= 0:
            try:
                os.close(self.fd)
            except OSError:
                pass
        self.fd = -1

    def __del__(self) -> None:
        """Release the descriptor if an exceptional owner abandons the artifact."""
        self.close()


@dataclass(frozen=True, slots=True)
class SerializedArtifact:
    """A retained nameless complete-MJB subject plus the facts measured while writing it."""

    retained: RetainedCompiledArtifact
    quarantined_name: Path
    mjb_sha256: str
    mjb_size_bytes: int
    header_words: tuple[int, ...]
    sizeof_mjtnum: int

    @property
    def path(self) -> Path:
        """Return the pathname this artifact briefly occupied before it was unlinked.

        It is retained for evidence and diagnostics only. Nothing in a decision may read it:
        after serialization it names no object, and anything a later process creates there is by
        definition not the subject this artifact identifies.
        """
        return self.quarantined_name

    def verify(self) -> None:
        """Require the retained subject to still hold the exact bytes recorded here."""
        self.retained.verify()

    def identity(self, runtime_identity_sha256: str) -> CompiledArtifactIdentity:
        """Bind this serialized MJB to the finalized Certify runtime identity."""
        return CompiledArtifactIdentity(
            schema=ARTIFACT_IDENTITY_SCHEMA,
            schema_version=ARTIFACT_IDENTITY_SCHEMA_VERSION,
            method=COMPLETE_MJB_METHOD,
            mjb_sha256=self.mjb_sha256,
            mjb_size_bytes=self.mjb_size_bytes,
            header_words=self.header_words,
            magic_decimal=self.header_words[0],
            magic_hex=f"0x{self.header_words[0]:08x}",
            sizeof_mjtnum=self.sizeof_mjtnum,
            mujoco_version_integer=self.header_words[3],
            runtime_identity_sha256=runtime_identity_sha256,
        )


def serialize_complete_artifact(
    model: mujoco.MjModel, role: ModelRole, directory: Path
) -> SerializedArtifact:
    """Serialize every byte of one compiled model into a private file and identify it."""
    import mujoco
    import numpy as np

    size = int(mujoco.mj_sizeModel(model))
    if size > MAX_SERIALIZED_ARTIFACT_BYTES:
        raise refuse(
            OperationalReasonCode.COMPILED_ARTIFACT_SIZE_EXCEEDED,
            role,
            mjb_size_bytes=size,
            limit_bytes=MAX_SERIALIZED_ARTIFACT_BYTES,
        )
    # The buffer is hashed and streamed directly. No copy is taken, so exactly one artifact's
    # worth of bytes is resident, and the file receives every byte MuJoCo wrote.
    buffer = np.empty(size, dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    width = _mjtnum_width(model)
    payload = buffer.data
    try:
        if payload.nbytes != size:
            raise refuse(
                OperationalReasonCode.COMPILED_ARTIFACT_INVALID,
                role,
                issue="serialized_length_differs_from_mj_sizeModel",
                mj_size_model=size,
                serialized_bytes=payload.nbytes,
            )
        header = _validated_header(payload, width, role)
        digest = hashlib.sha256(payload).hexdigest()
        retained = _retain_private_artifact(directory, f"{role}.mjb", payload, role, digest)
    finally:
        payload.release()
        del buffer
    return SerializedArtifact(retained, directory / f"{role}.mjb", digest, size, header, width)


# Every decision-bearing consumer takes a retained subject. There is deliberately no pathname
# alternative: a `Path` overload would be an unverified way to reach bytes the receipt never
# measured, which is the exact defect this module exists to prevent.
ArtifactSubject = RetainedCompiledArtifact


def load_subject_model(subject: ArtifactSubject) -> mujoco.MjModel:
    """Compile-load one retained subject and bind the loaded model to its receipt-bound digest.

    Verifying the subject before and after the load proves only that the bytes were correct at two
    instants. It does not prove the loader read them: a process running as the same operating-system
    user can mutate the retained object after the first check and restore it before the second, and
    on Linux it can reach the nameless object through ``/proc/<pid>/fd`` to do so. That is the same
    reasoning that rejected hashing a pathname before and after its consumers.

    The binding that does hold is over the model itself. Serializing the exact loaded ``MjModel``
    must reproduce the receipt-bound complete-MJB size and SHA-256; a model built from transiently
    substituted bytes cannot, however the file is restored afterwards.
    """
    if not isinstance(subject, RetainedCompiledArtifact):
        raise TypeError("a decision-bearing model must be loaded from a retained subject")
    subject.verify()
    model = _load_model_from_retained_descriptor(subject)
    try:
        _require_loaded_model_matches_subject(model, subject)
        subject.verify()
    except BaseException:
        del model
        raise
    return model


def _load_model_from_binary_path(path: str) -> mujoco.MjModel:
    """Load one MJB path through MuJoCo; isolated so the bounded retry is testable."""
    import mujoco

    return mujoco.MjModel.from_binary_path(path)


def _load_model_from_retained_descriptor(subject: RetainedCompiledArtifact) -> mujoco.MjModel:
    """Load one retained MJB with one bounded retry for a transient descriptor-path open."""
    for attempt in range(1, _MODEL_LOAD_MAX_ATTEMPTS + 1):
        try:
            return _load_model_from_binary_path(subject.descriptor_path())
        except ValueError as exc:
            retryable = _RETRYABLE_MODEL_LOAD_MESSAGE in str(exc)
            if not retryable or attempt == _MODEL_LOAD_MAX_ATTEMPTS:
                raise
            subject.verify()
    raise AssertionError("bounded model-load loop did not return or raise")


def _require_loaded_model_matches_subject(
    model: mujoco.MjModel, subject: RetainedCompiledArtifact
) -> None:
    """Require this exact loaded model to serialize back to the receipt-bound artifact."""
    import mujoco
    import numpy as np

    observed_size = int(mujoco.mj_sizeModel(model))
    if observed_size > MAX_SERIALIZED_ARTIFACT_BYTES:
        raise refuse(
            OperationalReasonCode.COMPILED_ARTIFACT_SIZE_EXCEEDED,
            subject.role,
            mjb_size_bytes=observed_size,
            limit_bytes=MAX_SERIALIZED_ARTIFACT_BYTES,
        )
    buffer = np.empty(observed_size, dtype=np.uint8)
    payload = buffer.data
    try:
        mujoco.mj_saveModel(model, None, buffer)
        observed_digest = hashlib.sha256(payload).hexdigest()
    finally:
        payload.release()
        del buffer
    if observed_size == subject.mjb_size_bytes and observed_digest == subject.mjb_sha256:
        return
    raise subject._refuse(
        "loaded_model_digest_mismatch",
        expected_mjb_sha256=subject.mjb_sha256,
        observed_loaded_model_mjb_sha256=observed_digest,
        expected_mjb_size_bytes=subject.mjb_size_bytes,
        observed_loaded_model_mjb_size_bytes=observed_size,
    )


def read_header_words(path: Path, role: ModelRole) -> tuple[int, ...]:
    """Read the five native header integers back from a private artifact file."""
    with path.open("rb") as stream:
        prefix = stream.read(_HEADER_BYTES)
    if len(prefix) != _HEADER_BYTES:
        raise refuse(
            OperationalReasonCode.COMPILED_ARTIFACT_INVALID,
            role,
            issue="artifact_shorter_than_header",
            available_bytes=len(prefix),
        )
    return tuple(int(word) for word in struct.unpack(_HEADER_FORMAT, prefix))


def _validated_header(payload: memoryview, expected_width: int, role: ModelRole) -> tuple[int, ...]:
    """Validate the five-word header against the active build, then record it verbatim."""
    import mujoco

    if payload.nbytes < _HEADER_BYTES:
        raise refuse(
            OperationalReasonCode.COMPILED_ARTIFACT_INVALID,
            role,
            issue="artifact_shorter_than_header",
            mjb_size_bytes=payload.nbytes,
        )
    words = tuple(int(word) for word in struct.unpack_from(_HEADER_FORMAT, payload, 0))
    if words[0] != _MJB_MAGIC:
        raise refuse(
            OperationalReasonCode.COMPILED_ARTIFACT_INVALID,
            role,
            issue="magic_word_mismatch",
            observed_magic=words[0],
            expected_magic=_MJB_MAGIC,
        )
    if words[1] != expected_width:
        raise refuse(
            OperationalReasonCode.COMPILED_ARTIFACT_INVALID,
            role,
            issue="mjtnum_width_mismatch",
            observed_width=words[1],
            active_build_width=expected_width,
        )
    active_version = int(mujoco.mj_version())
    if words[3] != active_version:
        raise refuse(
            OperationalReasonCode.COMPILED_ARTIFACT_INVALID,
            role,
            issue="version_word_mismatch",
            observed_version=words[3],
            active_version=active_version,
        )
    return words


def _mjtnum_width(model: mujoco.MjModel) -> int:
    """Return the active build's mjtNum width, measured from a real model array."""
    import numpy as np

    return int(np.asarray(model.qpos0).dtype.itemsize)


def _retain_private_artifact(
    directory: Path, name: str, payload: memoryview, role: ModelRole, digest: str
) -> RetainedCompiledArtifact:
    """Write one private artifact, seal it, remove its name, and retain only its descriptor.

    The name exists only for the moment it takes to create and fsync the file. It is removed
    through the same directory descriptor that created it, and only after the entry is proven to
    still identify the object this descriptor holds, so nothing decision-bearing is ever reached
    by a pathname a same-user process could have redirected in between.
    """
    directory_fd = _open_private_directory(directory, role)
    descriptor = -1
    try:
        descriptor = _create_private_artifact(directory_fd, name, payload, role)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise refuse(
                OperationalReasonCode.COMPILED_ARTIFACT_INVALID,
                role,
                issue="private_artifact_is_not_a_regular_file",
            )
        _quarantine_private_name(directory_fd, name, metadata, role)
        retained = RetainedCompiledArtifact(
            descriptor,
            role=role,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mjb_size_bytes=metadata.st_size,
            mjb_sha256=digest,
        )
        # Prove the sealed, nameless object holds exactly the bytes that were just measured.
        # This runs inside the guard, because the case it exists to catch - a same-user writer
        # that reached the file during the brief window it had a name - is exactly the case where
        # the descriptor would otherwise be leaked by an escaping exception.
        retained.verify()
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        _unlink_quietly(directory_fd, name)
        raise
    finally:
        os.close(directory_fd)
    return retained


def _open_private_directory(directory: Path, role: ModelRole) -> int:
    """Open the private scratch directory itself, without following its final component."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(directory, flags)
    except OSError as exc:
        raise refuse(
            OperationalReasonCode.COMPILED_ARTIFACT_INVALID,
            role,
            issue="private_artifact_directory_unavailable",
            exception_type=type(exc).__name__,
        ) from exc


def _create_private_artifact(
    directory_fd: int, name: str, payload: memoryview, role: ModelRole
) -> int:
    """Exclusively create, completely write, and fsync one private artifact."""
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    except OSError as exc:
        raise refuse(
            OperationalReasonCode.COMPILED_ARTIFACT_INVALID,
            role,
            issue="private_artifact_create_failed",
            exception_type=type(exc).__name__,
        ) from exc
    try:
        written = 0
        while written < payload.nbytes:
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _quarantine_private_name(
    directory_fd: int, name: str, admitted: os.stat_result, role: ModelRole
) -> None:
    """Remove the artifact's only directory entry after proving it still names this object."""
    try:
        entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise refuse(
            OperationalReasonCode.COMPILED_ARTIFACT_INVALID,
            role,
            issue="private_artifact_name_unavailable",
            exception_type=type(exc).__name__,
        ) from exc
    if (entry.st_dev, entry.st_ino) != (admitted.st_dev, admitted.st_ino):
        raise refuse(
            OperationalReasonCode.COMPILED_ARTIFACT_INVALID,
            role,
            issue="private_artifact_name_replaced_before_quarantine",
        )
    try:
        os.unlink(name, dir_fd=directory_fd)
    except OSError as exc:
        raise refuse(
            OperationalReasonCode.COMPILED_ARTIFACT_INVALID,
            role,
            issue="private_artifact_name_removal_failed",
            exception_type=type(exc).__name__,
        ) from exc


def _unlink_quietly(directory_fd: int, name: str) -> None:
    """Best-effort removal of a partially created private artifact during failure cleanup."""
    try:
        os.unlink(name, dir_fd=directory_fd)
    except OSError:
        pass


__all__ = [
    "ARTIFACT_IDENTITY_SCHEMA",
    "COMPLETE_MJB_METHOD",
    "MAX_SERIALIZED_ARTIFACT_BYTES",
    "ArtifactSubject",
    "CompiledArtifactIdentity",
    "RetainedCompiledArtifact",
    "SerializedArtifact",
    "load_subject_model",
    "read_header_words",
    "serialize_complete_artifact",
]
