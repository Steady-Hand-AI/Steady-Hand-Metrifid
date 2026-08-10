"""Pure schema for the compiled-artifact certification runtime identity.

This module owns the ``CertifyRuntimeIdentity`` dataclass, its frozen member registry, strict
parsing, canonical serialization, and self-hash validation. It imports no native dependency, so a
reader can revalidate a certification receipt without MuJoCo or NumPy installed. Live measurement
of the running runtime stays in :mod:`metrifid._runtime_identity`, which imports this class.

Original module note follows.

Compact runtime identity for the compiled-artifact certification.

Certify makes a statement about bytes produced by one MuJoCo build inside one process. This
module records exactly the facts that determine those bytes, and nothing else. The hundreds of
per-file entries behind the two distribution hashes are deliberately not embedded here; the
hashes stand for them and the full manifests remain raw command evidence.

Both roles compile in the same process, so one runtime identity covers both artifacts. There
is no role-to-role runtime mismatch path and no public precompiled-MJB input.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Self, cast

from ..json_values import CanonicalValue, compute_self_hash, require_sha256, validate_self_hash
from ..operational import _require_exact_object_fields

RUNTIME_IDENTITY_SCHEMA = "metrifid.certify_runtime_identity"
RUNTIME_IDENTITY_SCHEMA_VERSION = 1

# Certify compiles and serializes. It never allocates mjData and never steps, so no result
# here depends on integrator state, and this constant is a claim the code structurally keeps.
EXECUTION_MODE_NO_MJDATA_EXECUTION = "NO_MJDATA_EXECUTION"

_MJB_HEADER_WORD_COUNT = 5

# The frozen member set. A reader that parses a receipt must see exactly these and no others.
_RUNTIME_TEXT_MEMBERS = (
    "schema",
    "metrifid_version",
    "metrifid_distribution_sha256",
    "mujoco_python_distribution_sha256",
    "mujoco_native_library_sha256",
    "mujoco_version",
    "mujoco_version_string",
    "python_implementation",
    "python_version",
    "numpy_version",
    "platform_system",
    "platform_machine",
    "platform_release",
    "libc",
    "byteorder",
    "execution_mode",
)
_RUNTIME_MEMBERS = (
    *_RUNTIME_TEXT_MEMBERS,
    "schema_version",
    "mujoco_version_integer",
    "mjb_header_words",
    "runtime_identity_sha256",
)


@dataclass(frozen=True, slots=True)
class CertifyRuntimeIdentity:
    """The measured runtime that produced a pair of complete compiled artifacts."""

    schema: str
    schema_version: int
    metrifid_version: str
    metrifid_distribution_sha256: str
    mujoco_python_distribution_sha256: str
    mujoco_native_library_sha256: str
    mujoco_version: str
    mujoco_version_string: str
    mujoco_version_integer: int
    python_implementation: str
    python_version: str
    numpy_version: str
    platform_system: str
    platform_machine: str
    platform_release: str
    libc: str
    byteorder: str
    mjb_header_words: tuple[int, ...]
    execution_mode: str
    runtime_identity_sha256: str | None

    _HASH_FIELD: ClassVar[str] = "runtime_identity_sha256"

    def __post_init__(self) -> None:
        """Validate Certify runtime versions, component hashes, and optional self-hash."""
        if self.schema != RUNTIME_IDENTITY_SCHEMA:
            raise ValueError("invalid certify runtime identity schema")
        if (
            type(self.schema_version) is not int
            or self.schema_version != RUNTIME_IDENTITY_SCHEMA_VERSION
        ):
            raise ValueError("invalid certify runtime identity schema_version")
        for field_name in (
            "metrifid_distribution_sha256",
            "mujoco_python_distribution_sha256",
            "mujoco_native_library_sha256",
        ):
            require_sha256(getattr(self, field_name), field_name)
        if len(self.mjb_header_words) != _MJB_HEADER_WORD_COUNT:
            raise ValueError("mjb_header_words must hold exactly five native integers")
        if any(type(word) is not int for word in self.mjb_header_words):
            raise TypeError("mjb_header_words must hold native integers")
        if self.execution_mode != EXECUTION_MODE_NO_MJDATA_EXECUTION:
            raise ValueError("certify records no mjData execution")
        if self.byteorder not in {"little", "big"}:
            raise ValueError("byteorder is outside the frozen registry")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Parse one runtime identity strictly, requiring its exact frozen member set."""
        obj = _require_exact_object_fields(value, set(_RUNTIME_MEMBERS), "CertifyRuntimeIdentity")
        words = obj["mjb_header_words"]
        if type(words) is not list or any(type(word) is not int for word in words):
            raise TypeError("mjb_header_words must be an array of native integers")
        text: dict[str, CanonicalValue] = {}
        for name in _RUNTIME_TEXT_MEMBERS:
            raw = obj[name]
            if type(raw) is not str or not raw:
                raise TypeError(f"{name} must be a nonempty string")
            text[name] = raw
        version_integer = obj["mujoco_version_integer"]
        if type(version_integer) is not int:
            raise TypeError("mujoco_version_integer must be an integer")
        schema_version = obj["schema_version"]
        if type(schema_version) is not int:
            raise TypeError("schema_version must be an integer")
        digest = obj["runtime_identity_sha256"]
        require_sha256(digest, "runtime_identity_sha256")
        parsed = cls(
            mujoco_version_integer=version_integer,
            schema_version=schema_version,
            mjb_header_words=tuple(int(word) for word in words),
            runtime_identity_sha256=str(digest),
            **cast("dict[str, str]", text),
        )
        parsed.validate_hash()
        return parsed

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the complete Certify runtime identity, including its optional aggregate hash."""
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "metrifid_version": self.metrifid_version,
            "metrifid_distribution_sha256": self.metrifid_distribution_sha256,
            "mujoco_python_distribution_sha256": self.mujoco_python_distribution_sha256,
            "mujoco_native_library_sha256": self.mujoco_native_library_sha256,
            "mujoco_version": self.mujoco_version,
            "mujoco_version_string": self.mujoco_version_string,
            "mujoco_version_integer": self.mujoco_version_integer,
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "platform_system": self.platform_system,
            "platform_machine": self.platform_machine,
            "platform_release": self.platform_release,
            "libc": self.libc,
            "byteorder": self.byteorder,
            "mjb_header_words": list(self.mjb_header_words),
            "execution_mode": self.execution_mode,
            "runtime_identity_sha256": self.runtime_identity_sha256,
        }

    def finalized(self) -> Self:
        """Return this Certify runtime identity with its canonical self-hash populated."""
        if self.runtime_identity_sha256 is not None:
            validate_self_hash(self.to_primitive(), self._HASH_FIELD)
            return self
        digest = compute_self_hash(self.to_primitive(), self._HASH_FIELD)
        return replace(self, runtime_identity_sha256=digest)

    def validate_hash(self) -> None:
        """Recompute and require the Certify runtime identity's aggregate hash."""
        validate_self_hash(self.to_primitive(), self._HASH_FIELD)


__all__ = [
    "EXECUTION_MODE_NO_MJDATA_EXECUTION",
    "RUNTIME_IDENTITY_SCHEMA",
    "RUNTIME_IDENTITY_SCHEMA_VERSION",
    "CertifyRuntimeIdentity",
]
