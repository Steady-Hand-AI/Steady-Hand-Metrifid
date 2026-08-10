"""Complete compiled-artifact identity over every serialized MJB byte.

The identity is SHA-256 over the complete ``mj_saveModel`` output. Nothing is projected,
normalized, rounded or tolerated. The buffer is streamed straight into a private file so a
role's bytes never have to stay resident while the other role compiles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mujoco  # type: ignore[import-untyped]

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

from .._model_closure import ModelRole, refuse
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


@dataclass(frozen=True, slots=True)
class SerializedArtifact:
    """A private complete-MJB file plus the facts measured while writing it."""

    path: Path
    mjb_sha256: str
    mjb_size_bytes: int
    header_words: tuple[int, ...]
    sizeof_mjtnum: int

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
        path = _write_private_file(directory / f"{role}.mjb", payload)
    finally:
        payload.release()
        del buffer
    return SerializedArtifact(path, digest, size, header, width)


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


def _write_private_file(path: Path, payload: memoryview) -> Path:
    """Create, write, and fsync one private artifact without following an existing path."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


__all__ = [
    "ARTIFACT_IDENTITY_SCHEMA",
    "COMPLETE_MJB_METHOD",
    "MAX_SERIALIZED_ARTIFACT_BYTES",
    "CompiledArtifactIdentity",
    "SerializedArtifact",
    "read_header_words",
    "serialize_complete_artifact",
]
