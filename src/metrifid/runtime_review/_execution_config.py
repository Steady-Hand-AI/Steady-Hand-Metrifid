"""Strict admission and filesystem identity binding for Runtime Review execution.

The execution journey accepts only two explicit Python launchers, one manifest, one fixture
identifier, and one absent output directory.  It never searches for an interpreter or accepts an
environment, command, callback, or code payload.  Every admitted filesystem object is retained as
an immutable identity that can be remeasured immediately before use and before a completed run is
returned.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Self

from .._json_admission import (
    JsonAdmissionLimits,
    bounded_strict_json_loads,
    read_bounded_regular_file,
)
from .._schema_primitives import _bounded_int, _fields, _object, _string
from ..json_values import CanonicalValue, canonical_sha256, require_sha256
from ._paths import (
    PathAdmissionError,
    admit_relative_portable_path,
    ensure_nonoverlapping,
    resolve_new_output_path,
)

RUN_CONFIG_SCHEMA: Final[str] = "metrifid.runtime_review_run_config"
RUN_CONFIG_SCHEMA_VERSION: Final[int] = 1
RUN_CONFIG_JSON_LIMITS: Final[JsonAdmissionLimits] = JsonAdmissionLimits(
    max_bytes=64 * 1024,
    max_depth=8,
    max_nodes=64,
    max_string_bytes=4096,
)
MANIFEST_MAX_BYTES: Final[int] = 1_048_576
_FIXTURE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z][a-z0-9_.-]{0,95}\Z")
_READ_FLAGS: Final[int] = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
_DIR_FLAGS: Final[int] = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


@dataclass(frozen=True, slots=True)
class RuntimeReviewRunConfig:
    """The closed data-only declaration for one installed execution journey."""

    schema: str
    schema_version: int
    baseline_python: str
    candidate_python: str
    manifest: str
    fixture_id: str
    output_dir: str

    def __post_init__(self) -> None:
        """Require the exact schema and reject every alternate field spelling or value kind."""
        if self.schema != RUN_CONFIG_SCHEMA:
            raise ValueError(f"schema must be {RUN_CONFIG_SCHEMA}")
        if type(self.schema_version) is not int or self.schema_version != RUN_CONFIG_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be the integer {RUN_CONFIG_SCHEMA_VERSION}")
        _admit_absolute_interpreter_spelling(self.baseline_python, "baseline_python")
        _admit_absolute_interpreter_spelling(self.candidate_python, "candidate_python")
        if self.baseline_python == self.candidate_python:
            raise ValueError("baseline_python and candidate_python must be lexically distinct")
        admit_relative_portable_path(self.manifest, "manifest")
        _admit_fixture_id(self.fixture_id)
        admit_relative_portable_path(self.output_dir, "output_dir")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode one strict run configuration with no optional or unknown fields."""
        obj = _object(value, "RuntimeReviewRunConfig")
        _fields(
            obj,
            {
                "schema",
                "schema_version",
                "baseline_python",
                "candidate_python",
                "manifest",
                "fixture_id",
                "output_dir",
            },
            "RuntimeReviewRunConfig",
        )
        return cls(
            schema=_string(obj["schema"], "schema"),
            schema_version=_bounded_int(obj["schema_version"], "schema_version", 1, 1),
            baseline_python=_admit_absolute_interpreter_spelling(
                obj["baseline_python"], "baseline_python"
            ),
            candidate_python=_admit_absolute_interpreter_spelling(
                obj["candidate_python"], "candidate_python"
            ),
            manifest=admit_relative_portable_path(obj["manifest"], "manifest"),
            fixture_id=_admit_fixture_id(obj["fixture_id"]),
            output_dir=admit_relative_portable_path(obj["output_dir"], "output_dir"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return the canonical semantic representation of this declaration."""
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "baseline_python": self.baseline_python,
            "candidate_python": self.candidate_python,
            "manifest": self.manifest,
            "fixture_id": self.fixture_id,
            "output_dir": self.output_dir,
        }


@dataclass(frozen=True, slots=True)
class InterpreterIdentity:
    """Lexical launcher identity plus the exact regular executable it resolves to."""

    lexical_path: Path
    lexical_kind: str
    lexical_device: int
    lexical_inode: int
    lexical_mode: int
    lexical_size_bytes: int
    lexical_mtime_ns: int
    lexical_link_target: str | None
    resolved_path: Path
    resolved_device: int
    resolved_inode: int
    resolved_mode: int
    resolved_size_bytes: int
    resolved_mtime_ns: int
    resolved_sha256: str

    def __post_init__(self) -> None:
        """Require a complete internally consistent launcher and executable identity."""
        if not self.lexical_path.is_absolute() or not self.resolved_path.is_absolute():
            raise ValueError("interpreter identity paths must be absolute")
        if self.lexical_kind not in {"regular_file", "symbolic_link"}:
            raise ValueError("lexical_kind must be regular_file or symbolic_link")
        if self.lexical_kind == "symbolic_link" and self.lexical_link_target is None:
            raise ValueError("symbolic-link launcher identity must retain its link target")
        if self.lexical_kind == "regular_file" and self.lexical_link_target is not None:
            raise ValueError("regular launcher identity must not retain a link target")
        for field in (
            "lexical_device",
            "lexical_inode",
            "lexical_mode",
            "lexical_size_bytes",
            "lexical_mtime_ns",
            "resolved_device",
            "resolved_inode",
            "resolved_mode",
            "resolved_size_bytes",
            "resolved_mtime_ns",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        require_sha256(self.resolved_sha256, "resolved_sha256")

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return a canonical execution-record projection of this identity."""
        return {
            "lexical_path": self.lexical_path.as_posix(),
            "lexical_kind": self.lexical_kind,
            "lexical_device": self.lexical_device,
            "lexical_inode": self.lexical_inode,
            "lexical_mode": self.lexical_mode,
            "lexical_size_bytes": self.lexical_size_bytes,
            "lexical_mtime_ns": self.lexical_mtime_ns,
            "lexical_link_target": self.lexical_link_target,
            "resolved_path": self.resolved_path.as_posix(),
            "resolved_device": self.resolved_device,
            "resolved_inode": self.resolved_inode,
            "resolved_mode": self.resolved_mode,
            "resolved_size_bytes": self.resolved_size_bytes,
            "resolved_mtime_ns": self.resolved_mtime_ns,
            "resolved_sha256": self.resolved_sha256,
        }


@dataclass(frozen=True, slots=True)
class ManifestIdentity:
    """One admitted bounded regular manifest and its mutation-sensitive identity."""

    path: Path
    device: int
    inode: int
    mode: int
    size_bytes: int
    mtime_ns: int
    sha256: str

    def __post_init__(self) -> None:
        """Require an absolute regular-file projection and one lowercase digest."""
        if not self.path.is_absolute():
            raise ValueError("manifest identity path must be absolute")
        for field in ("device", "inode", "mode", "size_bytes", "mtime_ns"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"manifest {field} must be a nonnegative integer")
        if not stat.S_ISREG(self.mode):
            raise ValueError("manifest identity mode must describe a regular file")
        if self.size_bytes > MANIFEST_MAX_BYTES:
            raise ValueError("manifest identity exceeds the worker admission limit")
        require_sha256(self.sha256, "manifest sha256")

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return the canonical retained projection of the admitted manifest."""
        return {
            "path": self.path.as_posix(),
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class AdmittedRuntimeReviewRunConfiguration:
    """Strict run semantics with raw bytes and every resolved filesystem binding."""

    config: RuntimeReviewRunConfig
    path: Path
    base_dir: Path
    raw_bytes: bytes
    raw_sha256: str
    semantic_sha256: str
    baseline_interpreter: InterpreterIdentity
    candidate_interpreter: InterpreterIdentity
    manifest_identity: ManifestIdentity
    manifest_path: Path
    output_dir: Path

    def __post_init__(self) -> None:
        """Require retained bytes, hashes, paths, and identities to agree exactly."""
        if not isinstance(self.config, RuntimeReviewRunConfig):
            raise TypeError("config must be a RuntimeReviewRunConfig")
        if type(self.raw_bytes) is not bytes:
            raise TypeError("raw_bytes must be bytes")
        if hashlib.sha256(self.raw_bytes).hexdigest() != self.raw_sha256:
            raise ValueError("raw_sha256 does not match raw_bytes")
        if canonical_sha256(self.config.to_primitive()) != self.semantic_sha256:
            raise ValueError("semantic_sha256 does not match canonical run semantics")
        if self.baseline_interpreter.lexical_path.as_posix() != self.config.baseline_python:
            raise ValueError("baseline interpreter does not bind baseline_python")
        if self.candidate_interpreter.lexical_path.as_posix() != self.config.candidate_python:
            raise ValueError("candidate interpreter does not bind candidate_python")
        if self.manifest_identity.path != self.manifest_path:
            raise ValueError("manifest identity does not bind manifest_path")
        if (
            self.manifest_path.parent != self.base_dir
            and self.base_dir not in self.manifest_path.parents
        ):
            raise ValueError("manifest_path must remain beneath the configuration directory")
        if self.output_dir.parent != self.base_dir and self.base_dir not in self.output_dir.parents:
            raise ValueError("output_dir must remain beneath the configuration directory")


def load_runtime_review_run_configuration(
    path: str | Path,
) -> AdmittedRuntimeReviewRunConfiguration:
    """Read, strictly decode, resolve, and identity-bind one execution declaration."""
    target = Path(path).absolute()
    raw = read_bounded_regular_file(target, RUN_CONFIG_JSON_LIMITS.max_bytes)
    primitive = bounded_strict_json_loads(raw, RUN_CONFIG_JSON_LIMITS)
    config = RuntimeReviewRunConfig.from_primitive(primitive)
    try:
        base_dir = target.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PathAdmissionError(f"configuration directory could not be resolved: {exc}") from exc
    if not base_dir.is_dir():
        raise PathAdmissionError("configuration directory must be a directory")

    baseline = admit_interpreter(config.baseline_python, "baseline_python")
    candidate = admit_interpreter(config.candidate_python, "candidate_python")
    manifest = _admit_manifest(base_dir, config.manifest)
    output = resolve_new_output_path(base_dir, config.output_dir)
    ensure_nonoverlapping(output, {"manifest": manifest.path})
    return AdmittedRuntimeReviewRunConfiguration(
        config=config,
        path=target,
        base_dir=base_dir,
        raw_bytes=raw,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        semantic_sha256=canonical_sha256(config.to_primitive()),
        baseline_interpreter=baseline,
        candidate_interpreter=candidate,
        manifest_identity=manifest,
        manifest_path=manifest.path,
        output_dir=output,
    )


def admit_interpreter(path: str | Path, field: str) -> InterpreterIdentity:
    """Admit one explicit absolute executable while permitting only a final launcher symlink."""
    spelling = _admit_absolute_interpreter_spelling(
        path.as_posix() if isinstance(path, Path) else path, field
    )
    lexical = Path(spelling)
    parent_fd = _open_absolute_parent(lexical, field)
    try:
        try:
            lexical_metadata = os.stat(lexical.name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise PathAdmissionError(f"{field} is not an existing executable path: {exc}") from exc
        if stat.S_ISLNK(lexical_metadata.st_mode):
            lexical_kind = "symbolic_link"
            try:
                link_target: str | None = os.readlink(lexical.name, dir_fd=parent_fd)
            except OSError as exc:
                raise PathAdmissionError(f"{field} symbolic link could not be read: {exc}") from exc
        elif stat.S_ISREG(lexical_metadata.st_mode):
            lexical_kind = "regular_file"
            link_target = None
        else:
            raise PathAdmissionError(
                f"{field} must name a regular executable or final symbolic link"
            )
    finally:
        os.close(parent_fd)
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PathAdmissionError(f"{field} could not resolve to an executable: {exc}") from exc
    resolved_metadata, resolved_sha256 = _measure_regular_file(resolved, field, maximum=None)
    lexical_after = os.lstat(lexical)
    if _stat_projection(lexical_after) != _stat_projection(lexical_metadata):
        raise PathAdmissionError(f"{field} launcher changed while its identity was measured")
    if lexical_kind == "symbolic_link" and lexical.readlink().as_posix() != link_target:
        raise PathAdmissionError(f"{field} symbolic link target changed during admission")
    if not os.access(resolved, os.X_OK):
        raise PathAdmissionError(f"{field} must resolve to an executable regular file")
    return InterpreterIdentity(
        lexical_path=lexical,
        lexical_kind=lexical_kind,
        lexical_device=lexical_metadata.st_dev,
        lexical_inode=lexical_metadata.st_ino,
        lexical_mode=lexical_metadata.st_mode,
        lexical_size_bytes=lexical_metadata.st_size,
        lexical_mtime_ns=lexical_metadata.st_mtime_ns,
        lexical_link_target=link_target,
        resolved_path=resolved,
        resolved_device=resolved_metadata.st_dev,
        resolved_inode=resolved_metadata.st_ino,
        resolved_mode=resolved_metadata.st_mode,
        resolved_size_bytes=resolved_metadata.st_size,
        resolved_mtime_ns=resolved_metadata.st_mtime_ns,
        resolved_sha256=resolved_sha256,
    )


def recheck_interpreter_identity(identity: InterpreterIdentity) -> InterpreterIdentity:
    """Remeasure one launcher and refuse any lexical or resolved executable substitution."""
    if not isinstance(identity, InterpreterIdentity):
        raise TypeError("identity must be an InterpreterIdentity")
    observed = admit_interpreter(identity.lexical_path, "interpreter")
    if observed != identity:
        raise PathAdmissionError("interpreter identity changed after run configuration admission")
    return observed


def recheck_manifest_identity(identity: ManifestIdentity) -> ManifestIdentity:
    """Remeasure one manifest and refuse any path, metadata, or byte mutation."""
    if not isinstance(identity, ManifestIdentity):
        raise TypeError("identity must be a ManifestIdentity")
    observed = _measure_manifest_path(identity.path, "manifest")
    if observed != identity:
        raise PathAdmissionError("manifest identity changed after run configuration admission")
    return observed


def recheck_runtime_review_run_configuration(
    admitted: AdmittedRuntimeReviewRunConfiguration,
    *,
    require_output_absent: bool = True,
) -> None:
    """Recheck every input and optionally require that output ownership has not begun."""
    if not isinstance(admitted, AdmittedRuntimeReviewRunConfiguration):
        raise TypeError("admitted must be an AdmittedRuntimeReviewRunConfiguration")
    if type(require_output_absent) is not bool:
        raise TypeError("require_output_absent must be a boolean")
    recheck_runtime_review_run_inputs(admitted)
    if not require_output_absent:
        return
    try:
        os.lstat(admitted.output_dir)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise PathAdmissionError(f"output_dir could not be rechecked: {exc}") from exc
    raise PathAdmissionError("output_dir appeared after run configuration admission")


def recheck_runtime_review_run_inputs(
    admitted: AdmittedRuntimeReviewRunConfiguration,
) -> None:
    """Replay the exact configuration bytes and all mutable input identities after ownership."""
    if not isinstance(admitted, AdmittedRuntimeReviewRunConfiguration):
        raise TypeError("admitted must be an AdmittedRuntimeReviewRunConfiguration")
    raw = read_bounded_regular_file(admitted.path, RUN_CONFIG_JSON_LIMITS.max_bytes)
    if raw != admitted.raw_bytes or hashlib.sha256(raw).hexdigest() != admitted.raw_sha256:
        raise PathAdmissionError("run configuration bytes changed after admission")
    primitive = bounded_strict_json_loads(raw, RUN_CONFIG_JSON_LIMITS)
    replayed = RuntimeReviewRunConfig.from_primitive(primitive)
    if (
        replayed != admitted.config
        or canonical_sha256(replayed.to_primitive()) != admitted.semantic_sha256
    ):
        raise PathAdmissionError("run configuration semantics changed after admission")
    recheck_interpreter_identity(admitted.baseline_interpreter)
    recheck_interpreter_identity(admitted.candidate_interpreter)
    recheck_manifest_identity(admitted.manifest_identity)


def _admit_absolute_interpreter_spelling(value: object, field: str) -> str:
    """Require one normalized absolute POSIX launcher spelling without discovery semantics."""
    raw = _string(value, field)
    if not raw or raw.startswith("//") or "\x00" in raw or "\\" in raw:
        raise PathAdmissionError(f"{field} must be a nonempty absolute POSIX path")
    path = PurePosixPath(raw)
    if (
        not path.is_absolute()
        or path.as_posix() != raw
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise PathAdmissionError(f"{field} must be a normalized absolute POSIX path")
    return raw


def _admit_fixture_id(value: object) -> str:
    """Require the frozen worker's lowercase bounded semantic identifier grammar."""
    fixture_id = _string(value, "fixture_id")
    if _FIXTURE_ID_PATTERN.fullmatch(fixture_id) is None:
        raise ValueError("fixture_id must use the evidence worker's semantic identifier grammar")
    return fixture_id


def _admit_manifest(base_dir: Path, relative: str) -> ManifestIdentity:
    """Resolve one confined relative manifest without following any declared component link."""
    admitted = admit_relative_portable_path(relative, "manifest")
    components = admitted.split("/")
    descriptor = _open_absolute_directory(base_dir, "configuration directory")
    try:
        for component in components[:-1]:
            try:
                child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            except OSError as exc:
                raise PathAdmissionError(f"manifest ancestor could not be opened: {exc}") from exc
            os.close(descriptor)
            descriptor = child
        metadata, digest = _measure_regular_file_at(
            descriptor,
            components[-1],
            "manifest",
            maximum=MANIFEST_MAX_BYTES,
        )
    finally:
        os.close(descriptor)
    return ManifestIdentity(
        path=base_dir.joinpath(*components),
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size_bytes=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        sha256=digest,
    )


def _open_absolute_parent(path: Path, field: str) -> int:
    """Open one absolute path's parent through no-follow retained descriptors."""
    if not path.is_absolute() or path.name == "":
        raise PathAdmissionError(f"{field} must name an absolute non-root path")
    return _open_absolute_directory(path.parent, f"{field} parent")


def _open_absolute_directory(path: Path, field: str) -> int:
    """Open one absolute directory a component at a time without following links."""
    absolute = path.absolute()
    descriptor = os.open(absolute.anchor, _DIR_FLAGS)
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            except OSError as exc:
                raise PathAdmissionError(f"{field} contains an unsafe component: {exc}") from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _measure_regular_file_at(
    parent_fd: int,
    name: str,
    field: str,
    *,
    maximum: int | None,
) -> tuple[os.stat_result, str]:
    """Hash one directory-relative regular file through a retained no-follow descriptor."""
    try:
        descriptor = os.open(name, _READ_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise PathAdmissionError(f"{field} could not be opened as a regular file: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise PathAdmissionError(f"{field} must name a regular file")
        if maximum is not None and metadata.st_size > maximum:
            raise PathAdmissionError(f"{field} exceeds the {maximum} byte admission limit")
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if maximum is not None and total > maximum:
                raise PathAdmissionError(f"{field} exceeds the {maximum} byte admission limit")
            digest.update(block)
        after = os.fstat(descriptor)
        if _stat_projection(after) != _stat_projection(metadata):
            raise PathAdmissionError(f"{field} changed while its identity was measured")
        return after, digest.hexdigest()
    finally:
        os.close(descriptor)


def _measure_manifest_path(path: Path, field: str) -> ManifestIdentity:
    """Measure one bounded no-follow regular manifest through a single descriptor."""
    metadata, digest = _measure_regular_file(path, field, maximum=MANIFEST_MAX_BYTES)
    return ManifestIdentity(
        path=path,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size_bytes=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        sha256=digest,
    )


def _measure_regular_file(
    path: Path, field: str, *, maximum: int | None
) -> tuple[os.stat_result, str]:
    """Hash one regular no-follow file and return metadata from that same descriptor."""
    parent_fd = _open_absolute_parent(path, field)
    try:
        return _measure_regular_file_at(parent_fd, path.name, field, maximum=maximum)
    finally:
        os.close(parent_fd)


def _stat_projection(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return the metadata fields that bind one measured filesystem object."""
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


__all__ = [
    "AdmittedRuntimeReviewRunConfiguration",
    "InterpreterIdentity",
    "MANIFEST_MAX_BYTES",
    "ManifestIdentity",
    "RUN_CONFIG_JSON_LIMITS",
    "RUN_CONFIG_SCHEMA",
    "RUN_CONFIG_SCHEMA_VERSION",
    "RuntimeReviewRunConfig",
    "admit_interpreter",
    "load_runtime_review_run_configuration",
    "recheck_interpreter_identity",
    "recheck_manifest_identity",
    "recheck_runtime_review_run_configuration",
    "recheck_runtime_review_run_inputs",
]
