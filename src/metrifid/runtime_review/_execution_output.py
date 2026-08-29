"""No-clobber ownership and completed-record verification for execution runs.

An execution output is diagnostic evidence even when a preflight or evidence cell refuses.  This
module therefore creates an absent root through retained no-follow directory descriptors, writes
every Metrifid-owned member with exclusive creation, and never removes a partial tree.  The
completed ``runtime_review_run.json`` member is canonical, self-hashed, reference-replayed, and
written last.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Self, cast

from .._json_admission import (
    CONFIG_JSON_LIMITS,
    RECEIPT_JSON_LIMITS,
    bounded_strict_json_loads,
    read_bounded_regular_file,
)
from ..json_values import (
    CanonicalValue,
    canonical_json_bytes,
    canonical_sha256,
    compute_self_hash,
    require_sha256,
    validate_self_hash,
)
from ._config import (
    BASELINE_PROFILE_ID,
    CANDIDATE_PROFILE_ID,
    PROFILE_ROLES,
    REPEAT_IDS,
    STEP_DTS,
)
from ._execution_config import (
    AdmittedRuntimeReviewRunConfiguration,
    recheck_runtime_review_run_inputs,
)
from ._native_profile_identity import (
    ProfileIdentityRefusal,
    load_native_profile_identity,
    load_native_profile_identity_v2,
)
from ._receipt import RUNTIME_REVIEW_LIMITATIONS
from ._receipt_validation import load_and_validate_runtime_review_receipt
from ._status import (
    RuntimeReviewReasonCode,
    RuntimeReviewStatus,
    runtime_review_exit_code,
)

RUN_RECORD_SCHEMA: Final[str] = "metrifid.runtime_review_run"
RUN_RECORD_SCHEMA_VERSION: Final[int] = 1
RUN_RECORD_SCHEMA_VERSION_V2: Final[int] = 2
LEGACY_FROZEN_EVIDENCE_WORKER_SHA256: Final[str] = (
    "941cc0cba66632901e89ee0a5be63575a2a5635dc98595e10271d9bed003dd6f"
)
FROZEN_EVIDENCE_WORKER_SHA256: Final[str] = (
    "b00e509a344593806c088c4e49783ed71bacd815466d74bce9e27c931535b4ff"
)
RUNTIME_REVIEW_RUN_CLAIM_LIMITATIONS: Final[tuple[str, ...]] = (
    *RUNTIME_REVIEW_LIMITATIONS,
    "Profile preflight makes no claim of environment correctness beyond the measured identities "
    "recorded in this run.",
    "Executing the packaged evidence worker adds no scientific claim beyond the independently "
    "validated Runtime Review receipt.",
)
ADMITTED_RUN_CONFIG_LOCATOR: Final[str] = "admitted_runtime_review_run_config.json"
GENERATED_REVIEW_CONFIG_LOCATOR: Final[str] = "generated_runtime_review_config.json"
COMPLETED_RUN_RECORD_LOCATOR: Final[str] = "runtime_review_run.json"
_STEP_DIRECTORY_TOKENS: Final[Mapping[str, str]] = {
    "0.004": "0p004",
    "0.002": "0p002",
    "0.001": "0p001",
}
_RUN_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        "reason_code",
        "exit_code",
        "input_configuration",
        "packaged_worker",
        "profile_identity_collector",
        "profile_preflights",
        "evidence_attempts",
        "generated_runtime_review_config",
        "runtime_review_receipt",
        "claim_limitations",
        "run_sha256",
    }
)
_NO_EXIT_STATES: Final[frozenset[str]] = frozenset({"TIMEOUT", "NOT_STARTED"})
_PROCESS_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "command_locator",
        "command_sha256",
        "stdout_locator",
        "stdout_sha256",
        "stderr_locator",
        "stderr_sha256",
        "exit_code_locator",
        "exit_code_sha256",
        "exit_code",
        "no_exit_status",
    }
)
_PREFLIGHT_RECORD_FIELDS: Final[frozenset[str]] = _PROCESS_RECORD_FIELDS | {
    "role",
    "profile_id",
    "lexical_interpreter",
    "resolved_interpreter",
    "resolved_executable_sha256",
    "identity_locator",
    "identity_file_sha256",
    "profile_identity_sha256",
}
_PREFLIGHT_RECORD_FIELDS_V2: Final[frozenset[str]] = _PROCESS_RECORD_FIELDS | {
    "role",
    "package_version",
    "native_version",
    "native_version_integer",
    "support_tier",
    "lexical_interpreter",
    "resolved_interpreter",
    "resolved_executable_sha256",
    "identity_locator",
    "identity_file_sha256",
    "profile_identity_sha256",
    "sentinel_identity_sha256",
}
_ATTEMPT_RECORD_FIELDS: Final[frozenset[str]] = _PROCESS_RECORD_FIELDS | {
    "role",
    "profile_id",
    "mujoco_version",
    "step_dt",
    "repeat_id",
    "lexical_interpreter",
    "resolved_interpreter",
    "resolved_executable_sha256",
    "cell_locator",
    "result_sha256",
    "checksum_manifest_sha256",
}
_ATTEMPT_RECORD_FIELDS_V2: Final[frozenset[str]] = _PROCESS_RECORD_FIELDS | {
    "role",
    "package_version",
    "native_version",
    "native_version_integer",
    "profile_identity_sha256",
    "runtime_identity_sha256",
    "sentinel_identity_sha256",
    "step_dt",
    "repeat_id",
    "lexical_interpreter",
    "resolved_interpreter",
    "resolved_executable_sha256",
    "cell_locator",
    "result_sha256",
    "checksum_manifest_sha256",
}
_DIR_FLAGS: Final[int] = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_READ_FLAGS: Final[int] = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
_WRITE_FLAGS: Final[int] = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
_MAX_RECORD_BYTES: Final[int] = 16 * 1024 * 1024
_MAX_REFERENCED_BYTES: Final[int] = 512 * 1024 * 1024
_CELL_MEMBER_LIMITS: Final[Mapping[str, int]] = {
    "fixture.xml": 16 * 1024 * 1024,
    "input_manifest.json": 4 * 1024 * 1024,
    "model.mjb": 256 * 1024 * 1024,
    "result.json": 64 * 1024 * 1024,
    "trace.npz": 512 * 1024 * 1024,
}
_CELL_MEMBERS: Final[tuple[str, ...]] = (
    "CHECKSUMS.sha256",
    "fixture.xml",
    "input_manifest.json",
    "model.mjb",
    "result.json",
    "trace.npz",
)
_CHECKSUM_LINE: Final[re.Pattern[str]] = re.compile(
    r"([0-9a-f]{64})  (fixture\.xml|input_manifest\.json|model\.mjb|result\.json|trace\.npz)\Z"
)


class RuntimeReviewRunOutputError(ValueError):
    """Raised when an execution output cannot be owned, written, or replayed safely."""


@dataclass(frozen=True, slots=True)
class RetainedFile:
    """One exclusively created regular output member and its exact byte identity."""

    locator: str
    path: Path
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        """Require a normalized relative locator, absolute path, digest, and bounded size."""
        _admit_locator(self.locator, "retained file locator")
        if not self.path.is_absolute():
            raise ValueError("retained file path must be absolute")
        require_sha256(self.sha256, "retained file sha256")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("retained file size_bytes must be a nonnegative integer")

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return a canonical locator and byte-identity projection."""
        return {
            "locator": self.locator,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class RetainedProcessOutput:
    """The exact command, streams, and process-exit observation for one attempt."""

    command: RetainedFile
    stdout: RetainedFile
    stderr: RetainedFile
    exit_code_file: RetainedFile
    exit_code: int | None
    no_exit_status: str | None

    def __post_init__(self) -> None:
        """Require either one strict process exit or one truthful no-exit state."""
        if self.exit_code is not None:
            if type(self.exit_code) is not int:
                raise TypeError("exit_code must be an integer and not a boolean")
            if self.no_exit_status is not None:
                raise ValueError("no_exit_status must be absent when a process exited")
        elif self.no_exit_status not in _NO_EXIT_STATES:
            raise ValueError("a missing exit_code requires TIMEOUT or NOT_STARTED")

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return locators and hashes suitable for one operational run record."""
        return {
            "command_locator": self.command.locator,
            "command_sha256": self.command.sha256,
            "stdout_locator": self.stdout.locator,
            "stdout_sha256": self.stdout.sha256,
            "stderr_locator": self.stderr.locator,
            "stderr_sha256": self.stderr.sha256,
            "exit_code_locator": self.exit_code_file.locator,
            "exit_code_sha256": self.exit_code_file.sha256,
            "exit_code": self.exit_code,
            "no_exit_status": self.no_exit_status,
        }


@dataclass(frozen=True, slots=True)
class VerifiedRuntimeReviewRunRecord:
    """One canonical completed run document and its independent file identity."""

    path: Path
    document: dict[str, CanonicalValue]
    file_sha256: str
    run_sha256: str

    def __post_init__(self) -> None:
        """Require both the file digest and canonical self-hash to be well formed."""
        if not self.path.is_absolute():
            raise ValueError("completed run record path must be absolute")
        require_sha256(self.file_sha256, "completed run record file_sha256")
        require_sha256(self.run_sha256, "completed run record run_sha256")
        if self.document.get("run_sha256") != self.run_sha256:
            raise ValueError("completed run record object does not bind run_sha256")


class OwnedRuntimeReviewRunOutput:
    """Descriptor-bound owner for one absent execution-output root."""

    __slots__ = (
        "_admitted",
        "_closed",
        "_directory_identities",
        "_published_document",
        "_root_device",
        "_root_fd",
        "_root_inode",
        "admitted_configuration",
        "root",
    )

    def __init__(self, admitted: AdmittedRuntimeReviewRunConfiguration) -> None:
        """Own the configured absent root and immediately retain exact admitted input bytes."""
        if not isinstance(admitted, AdmittedRuntimeReviewRunConfiguration):
            raise TypeError("admitted must be an AdmittedRuntimeReviewRunConfiguration")
        self._admitted = admitted
        self.root = admitted.output_dir.absolute()
        self._root_fd, created = _create_owned_output_root(self.root)
        root_metadata = os.fstat(self._root_fd)
        self._root_device = root_metadata.st_dev
        self._root_inode = root_metadata.st_ino
        self._directory_identities: dict[str, tuple[int, int]] = {
            ".": (
                self._root_device,
                self._root_inode,
            )
        }
        self._directory_identities.update(created)
        self._closed = False
        self._published_document: dict[str, CanonicalValue] | None = None
        try:
            self.admitted_configuration = self._write_new_file(
                PurePosixPath(ADMITTED_RUN_CONFIG_LOCATOR), admitted.raw_bytes
            )
        except BaseException:
            self.close()
            raise

    @property
    def generated_runtime_review_config(self) -> Path:
        """Return the fixed absolute generated-review configuration path."""
        return self.root / GENERATED_REVIEW_CONFIG_LOCATOR

    @property
    def completed_run_record(self) -> Path:
        """Return the fixed absolute completed operational record path."""
        return self.root / COMPLETED_RUN_RECORD_LOCATOR

    def new_profile_identity_path(self, role: str) -> Path:
        """Create the owned identity parent and return one absent collector output path."""
        admitted_role = _admit_role(role)
        parent = PurePosixPath("profile_identities")
        parent_fd = self._ensure_owned_directory(parent)
        try:
            name = f"{admitted_role}.json"
            _require_absent(parent_fd, name, f"profile identity {admitted_role}")
        finally:
            os.close(parent_fd)
        self._verify_root_binding()
        return self.root / parent / name

    def new_evidence_cell_path(self, role: str, step_dt: str, repeat_id: int) -> Path:
        """Create owned slot parents and return one absent worker-owned cell directory path."""
        locator = _cell_locator(role, step_dt, repeat_id)
        parent_fd = self._ensure_owned_directory(locator.parent)
        try:
            _require_absent(parent_fd, locator.name, f"evidence cell {locator.as_posix()}")
        finally:
            os.close(parent_fd)
        self._verify_root_binding()
        return self.root / locator

    def write_profile_preflight(
        self,
        role: str,
        *,
        command: Mapping[str, CanonicalValue],
        stdout: bytes,
        stderr: bytes,
        exit_code: int | None,
        no_exit_status: str | None = None,
    ) -> RetainedProcessOutput:
        """Retain one and only one ordered profile-preflight process observation."""
        admitted_role = _admit_role(role)
        return self._write_process_output(
            PurePosixPath("profile_preflight", admitted_role),
            command=command,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            no_exit_status=no_exit_status,
        )

    def write_evidence_attempt(
        self,
        role: str,
        step_dt: str,
        repeat_id: int,
        *,
        command: Mapping[str, CanonicalValue],
        stdout: bytes,
        stderr: bytes,
        exit_code: int | None,
        no_exit_status: str | None = None,
    ) -> RetainedProcessOutput:
        """Retain one exact evidence-cell attempt without permitting a retry collision."""
        cell = _cell_locator(role, step_dt, repeat_id)
        attempt = PurePosixPath("attempts", *cell.parts[1:])
        return self._write_process_output(
            attempt,
            command=command,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            no_exit_status=no_exit_status,
        )

    def write_generated_runtime_review_configuration(self, raw: bytes) -> RetainedFile:
        """Retain one strict canonical generated referee configuration exactly once."""
        if type(raw) is not bytes:
            raise TypeError("generated runtime-review configuration must be bytes")
        primitive = bounded_strict_json_loads(raw, CONFIG_JSON_LIMITS)
        if raw != canonical_json_bytes(primitive) + b"\n":
            raise RuntimeReviewRunOutputError(
                "generated runtime-review configuration must be canonical JSON plus one newline"
            )
        return self._write_new_file(PurePosixPath(GENERATED_REVIEW_CONFIG_LOCATOR), raw)

    def publish_completed_run_record(
        self, record: Mapping[str, CanonicalValue]
    ) -> VerifiedRuntimeReviewRunRecord:
        """Self-hash, publish last, and independently replay one completed run record."""
        if self._published_document is not None:
            raise RuntimeReviewRunOutputError("completed run record was already published")
        document = dict(record)
        if "run_sha256" not in document or document["run_sha256"] is None:
            document["run_sha256"] = ""
            document["run_sha256"] = compute_self_hash(document, "run_sha256")
        _validate_run_record_document(document)
        self._verify_root_binding()
        _verify_record_references(self.root, document)
        try:
            recheck_runtime_review_run_inputs(self._admitted)
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeReviewRunOutputError(
                "execution inputs changed before completed-record publication"
            ) from exc
        raw = canonical_json_bytes(document) + b"\n"
        retained = self._publish_completed_file(raw)
        self._published_document = document
        try:
            return verify_completed_run_record(
                retained.path,
                expected=document,
                output_root=self.root,
            )
        except BaseException:
            self._published_document = None
            raise

    def verify_completed_output(self) -> VerifiedRuntimeReviewRunRecord:
        """Rebind the root, exact input copy, completed record, and every referenced artifact."""
        if self._published_document is None:
            raise RuntimeReviewRunOutputError("partial run has no completed run record")
        self._verify_root_binding()
        admitted_raw = _read_relative_regular(
            self.root,
            ADMITTED_RUN_CONFIG_LOCATOR,
            len(self._admitted.raw_bytes),
        )
        if admitted_raw != self._admitted.raw_bytes:
            raise RuntimeReviewRunOutputError("admitted run configuration copy changed")
        return verify_completed_run_record(
            self.completed_run_record,
            expected=self._published_document,
            output_root=self.root,
        )

    def close(self) -> None:
        """Release the retained output descriptor without deleting any partial evidence."""
        if self._closed:
            return
        self._closed = True
        try:
            os.close(self._root_fd)
        except OSError:  # pragma: no cover - defensive descriptor cleanup
            pass

    def __enter__(self) -> Self:
        """Return this live descriptor-bound output owner."""
        if self._closed:
            raise RuntimeReviewRunOutputError("closed output owner cannot be re-entered")
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release descriptors while deliberately preserving complete or partial output."""
        self.close()

    def _write_process_output(
        self,
        locator: PurePosixPath,
        *,
        command: Mapping[str, CanonicalValue],
        stdout: bytes,
        stderr: bytes,
        exit_code: int | None,
        no_exit_status: str | None,
    ) -> RetainedProcessOutput:
        """Write one new four-member process directory through exclusive descriptors."""
        if type(stdout) is not bytes or type(stderr) is not bytes:
            raise TypeError("captured stdout and stderr must be bytes")
        if exit_code is not None and type(exit_code) is not int:
            raise TypeError("exit_code must be an integer and not a boolean")
        if exit_code is None:
            if no_exit_status not in _NO_EXIT_STATES:
                raise ValueError("missing exit_code requires TIMEOUT or NOT_STARTED")
            exit_bytes = f"{no_exit_status}\n".encode("ascii")
        else:
            if no_exit_status is not None:
                raise ValueError("no_exit_status must be absent when a process exited")
            exit_bytes = f"{exit_code}\n".encode("ascii")
        command_bytes = canonical_json_bytes(dict(command)) + b"\n"
        directory_fd = self._make_new_owned_directory(locator)
        try:
            command_file = self._write_new_file_at(
                directory_fd, locator / "command.json", command_bytes
            )
            stdout_file = self._write_new_file_at(directory_fd, locator / "stdout.txt", stdout)
            stderr_file = self._write_new_file_at(directory_fd, locator / "stderr.txt", stderr)
            exit_file = self._write_new_file_at(directory_fd, locator / "exit_code.txt", exit_bytes)
        finally:
            os.close(directory_fd)
        return RetainedProcessOutput(
            command=command_file,
            stdout=stdout_file,
            stderr=stderr_file,
            exit_code_file=exit_file,
            exit_code=exit_code,
            no_exit_status=no_exit_status,
        )

    def _publish_completed_file(self, payload: bytes) -> RetainedFile:
        """Seal complete staging bytes before one atomic no-clobber final-name publication."""
        staging_locator = PurePosixPath(f".{COMPLETED_RUN_RECORD_LOCATOR}.staging")
        final_locator = PurePosixPath(COMPLETED_RUN_RECORD_LOCATOR)
        root_fd = os.dup(self._root_fd)
        try:
            _require_absent(root_fd, final_locator.name, "completed run record")
            self._write_new_file_at(root_fd, staging_locator, payload)
            try:
                os.link(
                    staging_locator.name,
                    final_locator.name,
                    src_dir_fd=root_fd,
                    dst_dir_fd=root_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise RuntimeReviewRunOutputError(
                    "completed run record could not be published without clobbering"
                ) from exc
            os.fsync(root_fd)
            os.unlink(staging_locator.name, dir_fd=root_fd)
            os.fsync(root_fd)
            final_fd = os.open(final_locator.name, _READ_FLAGS, dir_fd=root_fd)
            try:
                metadata = os.fstat(final_fd)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise RuntimeReviewRunOutputError(
                        "completed run record is not one exclusive regular file"
                    )
                observed = _read_bounded_descriptor(
                    final_fd, len(payload), COMPLETED_RUN_RECORD_LOCATOR
                )
                if observed != payload:
                    raise RuntimeReviewRunOutputError(
                        "completed run record changed during atomic publication"
                    )
            finally:
                os.close(final_fd)
        finally:
            os.close(root_fd)
        return RetainedFile(
            locator=final_locator.as_posix(),
            path=self.root / final_locator,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )

    def _write_new_file(self, locator: PurePosixPath, payload: bytes) -> RetainedFile:
        """Write one new root-relative file after owning all of its parent directories."""
        parent_fd = self._ensure_owned_directory(locator.parent)
        try:
            return self._write_new_file_at(parent_fd, locator, payload)
        finally:
            os.close(parent_fd)

    def _write_new_file_at(
        self, parent_fd: int, locator: PurePosixPath, payload: bytes
    ) -> RetainedFile:
        """Write and fsync one new regular file beneath an already bound parent."""
        if type(payload) is not bytes:
            raise TypeError("owned output payload must be bytes")
        try:
            descriptor = os.open(locator.name, _WRITE_FLAGS, 0o600, dir_fd=parent_fd)
        except OSError as exc:
            raise RuntimeReviewRunOutputError(
                f"owned output member could not be created: {locator.as_posix()}"
            ) from exc
        try:
            view = memoryview(payload)
            written = 0
            while written < len(view):
                written += os.write(descriptor, view[written:])
            os.fsync(descriptor)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise RuntimeReviewRunOutputError(
                    f"owned output member is not an exclusive regular file: {locator.as_posix()}"
                )
        finally:
            os.close(descriptor)
        return RetainedFile(
            locator=locator.as_posix(),
            path=self.root / locator,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )

    def _ensure_owned_directory(self, locator: PurePosixPath) -> int:
        """Create or reopen an owner-created directory chain and return its descriptor."""
        _admit_locator(locator.as_posix(), "owned directory locator", allow_root=True)
        if locator == PurePosixPath("."):
            return os.dup(self._root_fd)
        current_fd = os.dup(self._root_fd)
        key_parts: list[str] = []
        try:
            for component in locator.parts:
                key_parts.append(component)
                key = "/".join(key_parts)
                try:
                    metadata = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
                except FileNotFoundError:
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                    metadata = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
                    self._directory_identities[key] = (metadata.st_dev, metadata.st_ino)
                except OSError as exc:
                    raise RuntimeReviewRunOutputError(
                        f"owned output directory could not be inspected: {key}"
                    ) from exc
                expected = self._directory_identities.get(key)
                if (
                    expected is None
                    or not stat.S_ISDIR(metadata.st_mode)
                    or (metadata.st_dev, metadata.st_ino) != expected
                ):
                    raise RuntimeReviewRunOutputError(
                        f"owned output directory was not created by this run: {key}"
                    )
                child = os.open(component, _DIR_FLAGS, dir_fd=current_fd)
                bound = os.fstat(child)
                if (bound.st_dev, bound.st_ino) != expected:
                    os.close(child)
                    raise RuntimeReviewRunOutputError(f"owned output directory was replaced: {key}")
                os.close(current_fd)
                current_fd = child
            return current_fd
        except BaseException:
            os.close(current_fd)
            raise

    def _make_new_owned_directory(self, locator: PurePosixPath) -> int:
        """Create one fresh leaf directory under owner-created parents without reuse."""
        parent_fd = self._ensure_owned_directory(locator.parent)
        try:
            _require_absent(parent_fd, locator.name, f"owned directory {locator.as_posix()}")
            os.mkdir(locator.name, mode=0o700, dir_fd=parent_fd)
            metadata = os.stat(locator.name, dir_fd=parent_fd, follow_symlinks=False)
            self._directory_identities[locator.as_posix()] = (
                metadata.st_dev,
                metadata.st_ino,
            )
            child = os.open(locator.name, _DIR_FLAGS, dir_fd=parent_fd)
            bound = os.fstat(child)
            if (bound.st_dev, bound.st_ino) != (metadata.st_dev, metadata.st_ino):
                os.close(child)
                raise RuntimeReviewRunOutputError(
                    f"new owned directory was replaced: {locator.as_posix()}"
                )
            return child
        except BaseException:
            raise
        finally:
            os.close(parent_fd)

    def _verify_root_binding(self) -> None:
        """Require the public root path to still name the retained owned directory."""
        try:
            metadata = os.lstat(self.root)
        except OSError as exc:
            raise RuntimeReviewRunOutputError("owned output root is no longer available") from exc
        if not stat.S_ISDIR(metadata.st_mode) or (metadata.st_dev, metadata.st_ino) != (
            self._root_device,
            self._root_inode,
        ):
            raise RuntimeReviewRunOutputError("owned output root was replaced")


def prepare_runtime_review_run_output(
    admitted: AdmittedRuntimeReviewRunConfiguration,
) -> OwnedRuntimeReviewRunOutput:
    """Own one absent run root and copy the exact admitted configuration into it."""
    return OwnedRuntimeReviewRunOutput(admitted)


def verify_completed_run_record(
    path: str | Path,
    *,
    expected: Mapping[str, CanonicalValue] | None = None,
    output_root: str | Path | None = None,
) -> VerifiedRuntimeReviewRunRecord:
    """Strictly replay one completed record and every file or resource identity it references."""
    target = Path(path).absolute()
    raw = read_bounded_regular_file(target, _MAX_RECORD_BYTES)
    primitive = bounded_strict_json_loads(raw, RECEIPT_JSON_LIMITS)
    if type(primitive) is not dict:
        raise RuntimeReviewRunOutputError("completed run record must be a JSON object")
    document = primitive
    if raw != canonical_json_bytes(document) + b"\n":
        raise RuntimeReviewRunOutputError(
            "completed run record must be canonical JSON plus newline"
        )
    _validate_run_record_document(document)
    if expected is not None and canonical_json_bytes(document) != canonical_json_bytes(
        dict(expected)
    ):
        raise RuntimeReviewRunOutputError("completed run record differs from the published record")
    root = Path(output_root).absolute() if output_root is not None else target.parent
    _verify_record_references(root, document)
    run_sha256 = cast(str, document["run_sha256"])
    return VerifiedRuntimeReviewRunRecord(
        path=target,
        document=document,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        run_sha256=run_sha256,
    )


def verify_owned_runtime_review_run_output(
    output: OwnedRuntimeReviewRunOutput,
) -> VerifiedRuntimeReviewRunRecord:
    """Verify one owner's completed boundary without accepting a partial run as completed."""
    if not isinstance(output, OwnedRuntimeReviewRunOutput):
        raise TypeError("output must be an OwnedRuntimeReviewRunOutput")
    return output.verify_completed_output()


def _validate_run_record_document(document: dict[str, CanonicalValue]) -> None:
    """Require the closed completed-record shape, status mapping, order, and self-hash."""
    actual = set(document)
    if actual != _RUN_RECORD_FIELDS:
        missing = sorted(_RUN_RECORD_FIELDS - actual)
        unknown = sorted(actual - _RUN_RECORD_FIELDS)
        raise RuntimeReviewRunOutputError(
            f"completed run record field set is not closed; missing={missing!r}, unknown={unknown!r}"
        )
    if document["schema"] != RUN_RECORD_SCHEMA:
        raise RuntimeReviewRunOutputError(f"completed run schema must be {RUN_RECORD_SCHEMA}")
    if type(document["schema_version"]) is not int or document["schema_version"] not in {
        RUN_RECORD_SCHEMA_VERSION,
        RUN_RECORD_SCHEMA_VERSION_V2,
    }:
        raise RuntimeReviewRunOutputError("completed run schema_version must be 1 or 2")
    _validate_completed_status(document)
    _require_object(document["input_configuration"], "input_configuration")
    _require_object(document["packaged_worker"], "packaged_worker")
    _require_object(document["profile_identity_collector"], "profile_identity_collector")
    _require_object(document["generated_runtime_review_config"], "generated config")
    _require_object(document["runtime_review_receipt"], "runtime review receipt")
    preflights = _require_array(document["profile_preflights"], "profile_preflights")
    if len(preflights) != 2:
        raise RuntimeReviewRunOutputError("completed run must contain exactly two preflights")
    attempts = _require_array(document["evidence_attempts"], "evidence_attempts")
    if len(attempts) != 12:
        raise RuntimeReviewRunOutputError("completed run must contain exactly twelve attempts")
    limitations = _require_array(document["claim_limitations"], "claim_limitations")
    if limitations != list(RUNTIME_REVIEW_RUN_CLAIM_LIMITATIONS):
        raise RuntimeReviewRunOutputError("claim_limitations must equal the frozen claim boundary")
    validate_self_hash(document, "run_sha256")


def _validate_completed_status(document: Mapping[str, object]) -> None:
    """Require one supported completed status, reason, and exact frozen exit mapping."""
    status_value = document["status"]
    if type(status_value) is not str:
        raise RuntimeReviewRunOutputError("completed run status must be a string")
    try:
        status = RuntimeReviewStatus(status_value)
    except ValueError as exc:
        raise RuntimeReviewRunOutputError("completed run status is unsupported") from exc
    exit_code = document["exit_code"]
    if type(exit_code) is not int or exit_code != int(runtime_review_exit_code(status)):
        raise RuntimeReviewRunOutputError("completed run exit_code does not match status")
    reason = document["reason_code"]
    if reason is not None:
        if type(reason) is not str:
            raise RuntimeReviewRunOutputError("completed run reason_code must be null or a string")
        try:
            RuntimeReviewReasonCode(reason)
        except ValueError as exc:
            raise RuntimeReviewRunOutputError("completed run reason_code is unsupported") from exc
    if (status is RuntimeReviewStatus.INSUFFICIENT_EVIDENCE) != (reason is not None):
        raise RuntimeReviewRunOutputError(
            "completed run reason_code must be present exactly for insufficient evidence"
        )


def _verify_record_references(root: Path, document: dict[str, CanonicalValue]) -> None:
    """Rehash every referenced retained byte and validate canonical slot locators."""
    input_record = _require_object(document["input_configuration"], "input_configuration")
    _require_fields(
        input_record,
        {"locator", "raw_sha256", "semantic_sha256"},
        "input_configuration",
    )
    input_locator = _require_locator_field(input_record, "locator", "input_configuration")
    if input_locator != ADMITTED_RUN_CONFIG_LOCATOR:
        raise RuntimeReviewRunOutputError("input configuration locator is not canonical")
    input_raw = _verify_relative_hash(root, input_locator, input_record, "raw_sha256")
    input_primitive = bounded_strict_json_loads(input_raw, CONFIG_JSON_LIMITS)
    if canonical_sha256(input_primitive) != _require_hash_field(
        input_record, "semantic_sha256", "input_configuration"
    ):
        raise RuntimeReviewRunOutputError("input configuration semantic SHA-256 changed")

    record_version = cast(int, document["schema_version"])
    expected_worker_sha256 = (
        LEGACY_FROZEN_EVIDENCE_WORKER_SHA256
        if record_version == RUN_RECORD_SCHEMA_VERSION
        else FROZEN_EVIDENCE_WORKER_SHA256
    )
    _verify_external_resource(
        document["packaged_worker"],
        "packaged_worker",
        expected_sha256=expected_worker_sha256,
    )
    _verify_external_resource(document["profile_identity_collector"], "profile_identity_collector")
    _verify_preflight_references(
        root, document["profile_preflights"], input_primitive, record_version
    )
    _verify_attempt_references(root, document["evidence_attempts"], input_primitive, record_version)

    generated = _require_object(
        document["generated_runtime_review_config"], "generated_runtime_review_config"
    )
    _require_fields(generated, {"locator", "sha256"}, "generated_runtime_review_config")
    generated_locator = _require_locator_field(
        generated, "locator", "generated_runtime_review_config"
    )
    if generated_locator != GENERATED_REVIEW_CONFIG_LOCATOR:
        raise RuntimeReviewRunOutputError("generated configuration locator is not canonical")
    generated_raw = _verify_relative_hash(root, generated_locator, generated, "sha256")
    generated_primitive = bounded_strict_json_loads(generated_raw, CONFIG_JSON_LIMITS)
    if generated_raw != canonical_json_bytes(generated_primitive) + b"\n":
        raise RuntimeReviewRunOutputError("generated configuration is not canonical JSON")

    receipt = _require_object(document["runtime_review_receipt"], "runtime_review_receipt")
    _require_fields(
        receipt,
        {"locator", "file_sha256", "receipt_sha256"},
        "runtime_review_receipt",
    )
    receipt_locator = _require_locator_field(receipt, "locator", "runtime_review_receipt")
    if receipt_locator != "decision/runtime_review/runtime_review.json":
        raise RuntimeReviewRunOutputError("runtime-review receipt locator is not canonical")
    receipt_raw = _verify_relative_hash(root, receipt_locator, receipt, "file_sha256")
    receipt_path = root / receipt_locator
    loaded = load_and_validate_runtime_review_receipt(receipt_path)
    if loaded.get("receipt_sha256") != _require_hash_field(
        receipt, "receipt_sha256", "runtime_review_receipt"
    ):
        raise RuntimeReviewRunOutputError("runtime-review receipt self-hash changed")
    if (
        loaded.get("status") != document["status"]
        or loaded.get("reason_code") != document["reason_code"]
    ):
        raise RuntimeReviewRunOutputError("operational record does not propagate receipt decision")
    if receipt_raw != canonical_json_bytes(loaded) + b"\n":
        raise RuntimeReviewRunOutputError("runtime-review receipt bytes are not canonical")
    receipt_configuration = _require_object(
        loaded.get("configuration"), "runtime_review_receipt.configuration"
    )
    if receipt_configuration.get("raw_sha256") != hashlib.sha256(
        generated_raw
    ).hexdigest() or receipt_configuration.get("semantic_sha256") != canonical_sha256(
        generated_primitive
    ):
        raise RuntimeReviewRunOutputError(
            "runtime-review receipt does not bind the generated configuration"
        )


def _verify_external_resource(
    value: object, field: str, *, expected_sha256: str | None = None
) -> None:
    """Rehash one absolute packaged source resource named by the run record."""
    record = _require_object(value, field)
    _require_fields(record, {"locator", "sha256"}, field)
    locator = record.get("locator")
    if type(locator) is not str or not Path(locator).is_absolute():
        raise RuntimeReviewRunOutputError(f"{field}.locator must be an absolute path")
    declared_sha256 = _require_hash_field(record, "sha256", field)
    if expected_sha256 is not None and declared_sha256 != expected_sha256:
        raise RuntimeReviewRunOutputError("packaged worker does not use the frozen worker SHA-256")
    payload = _read_absolute_regular(Path(locator), _MAX_REFERENCED_BYTES, field)
    if hashlib.sha256(payload).hexdigest() != declared_sha256:
        raise RuntimeReviewRunOutputError(f"{field} bytes changed")


def _verify_preflight_references(
    root: Path,
    value: object,
    input_configuration: CanonicalValue,
    record_version: int = RUN_RECORD_SCHEMA_VERSION,
) -> None:
    """Require baseline-then-candidate preflight order and replay all retained files."""
    records = _require_array(value, "profile_preflights")
    input_record = _require_object(input_configuration, "input configuration")
    for role, profile_id, raw_record in zip(
        PROFILE_ROLES,
        (BASELINE_PROFILE_ID, CANDIDATE_PROFILE_ID),
        records,
        strict=True,
    ):
        field = f"profile_preflights[{role}]"
        record = _require_object(raw_record, field)
        expected_fields = (
            _PREFLIGHT_RECORD_FIELDS
            if record_version == RUN_RECORD_SCHEMA_VERSION
            else _PREFLIGHT_RECORD_FIELDS_V2
        )
        _require_fields(record, expected_fields, field)
        if record.get("role") != role:
            raise RuntimeReviewRunOutputError(f"{field} role identity is not canonical")
        if record_version == RUN_RECORD_SCHEMA_VERSION and record.get("profile_id") != profile_id:
            raise RuntimeReviewRunOutputError(f"{field} legacy profile identity is not canonical")
        _verify_lexical_interpreter(record, input_record, role, field)
        _verify_process_record(root, record, f"profile_preflight/{role}", field)
        identity_locator = _require_locator_field(record, "identity_locator", field)
        if identity_locator != f"profile_identities/{role}.json":
            raise RuntimeReviewRunOutputError(f"{field} identity locator is not canonical")
        identity_raw = _read_relative_regular(root, identity_locator, CONFIG_JSON_LIMITS.max_bytes)
        if hashlib.sha256(identity_raw).hexdigest() != _require_hash_field(
            record, "identity_file_sha256", field
        ):
            raise RuntimeReviewRunOutputError(f"{field} identity file SHA-256 changed")
        try:
            if record_version == RUN_RECORD_SCHEMA_VERSION:
                identity = load_native_profile_identity(
                    root / identity_locator,
                    expected_profile_id=profile_id,
                    expected_worker_sha256=LEGACY_FROZEN_EVIDENCE_WORKER_SHA256,
                )
            else:
                identity = load_native_profile_identity_v2(
                    root / identity_locator,
                    expected_profile_role=role,
                    expected_profile_identity_sha256=_require_hash_field(
                        record, "profile_identity_sha256", field
                    ),
                    expected_worker_sha256=FROZEN_EVIDENCE_WORKER_SHA256,
                )
        except ProfileIdentityRefusal as exc:
            raise RuntimeReviewRunOutputError(f"{field} identity failed strict replay") from exc
        if identity_raw != canonical_json_bytes(cast(CanonicalValue, identity)) + b"\n":
            raise RuntimeReviewRunOutputError(f"{field} identity is not canonical JSON")
        if identity.get("profile_identity_sha256") != _require_hash_field(
            record, "profile_identity_sha256", field
        ):
            raise RuntimeReviewRunOutputError(f"{field} admitted profile identity changed")
        if record_version == RUN_RECORD_SCHEMA_VERSION_V2:
            sentinel = _require_object(identity.get("sentinel"), f"{field}.sentinel")
            if (
                identity.get("package_version") != record.get("package_version")
                or identity.get("native_version") != record.get("native_version")
                or identity.get("native_version_integer") != record.get("native_version_integer")
                or identity.get("support_tier") != record.get("support_tier")
                or sentinel.get("status") != "PASS"
                or sentinel.get("sentinel_identity_sha256")
                != _require_hash_field(record, "sentinel_identity_sha256", field)
            ):
                raise RuntimeReviewRunOutputError(f"{field} v2 profile binding changed")
        _verify_resolved_executable(record, field)


def _verify_attempt_references(
    root: Path,
    value: object,
    input_configuration: CanonicalValue,
    record_version: int = RUN_RECORD_SCHEMA_VERSION,
) -> None:
    """Require canonical twelve-slot order and replay attempt, result, and checksum bytes."""
    records = _require_array(value, "evidence_attempts")
    input_record = _require_object(input_configuration, "input configuration")
    expected_slots = tuple(
        (role, step_dt, repeat_id)
        for role in PROFILE_ROLES
        for step_dt in STEP_DTS
        for repeat_id in REPEAT_IDS
    )
    for (role, step_dt, repeat_id), raw_record in zip(expected_slots, records, strict=True):
        field = f"evidence_attempts[{role}/{step_dt}/repeat_{repeat_id}]"
        record = _require_object(raw_record, field)
        expected_fields = (
            _ATTEMPT_RECORD_FIELDS
            if record_version == RUN_RECORD_SCHEMA_VERSION
            else _ATTEMPT_RECORD_FIELDS_V2
        )
        _require_fields(record, expected_fields, field)
        if (
            record.get("role") != role
            or record.get("step_dt") != step_dt
            or record.get("repeat_id") != repeat_id
        ):
            raise RuntimeReviewRunOutputError(f"{field} slot identity is not canonical")
        if record_version == RUN_RECORD_SCHEMA_VERSION:
            if record.get("profile_id") != (
                BASELINE_PROFILE_ID if role == "baseline" else CANDIDATE_PROFILE_ID
            ):
                raise RuntimeReviewRunOutputError(f"{field} profile_id is not canonical")
            expected_version = "3.10.0" if role == "baseline" else "3.11.0"
            if record.get("mujoco_version") != expected_version:
                raise RuntimeReviewRunOutputError(f"{field} MuJoCo version is not canonical")
        else:
            for name in (
                "package_version",
                "native_version",
                "profile_identity_sha256",
                "runtime_identity_sha256",
                "sentinel_identity_sha256",
            ):
                if name.endswith("sha256"):
                    _require_hash_field(record, name, field)
                elif type(record.get(name)) is not str or not record.get(name):
                    raise RuntimeReviewRunOutputError(f"{field}.{name} must be nonempty")
            if type(record.get("native_version_integer")) is not int:
                raise RuntimeReviewRunOutputError(
                    f"{field}.native_version_integer must be an integer"
                )
        _verify_lexical_interpreter(record, input_record, role, field)
        cell = _cell_locator(role, step_dt, repeat_id).as_posix()
        attempt = f"attempts/{'/'.join(PurePosixPath(cell).parts[1:])}"
        _verify_process_record(root, record, attempt, field)
        cell_locator = _require_locator_field(record, "cell_locator", field)
        if cell_locator != cell:
            raise RuntimeReviewRunOutputError(f"{field} cell locator is not canonical")
        checksums_locator = f"{cell_locator}/CHECKSUMS.sha256"
        result_sha256 = _require_hash_field(record, "result_sha256", field)
        _verify_relative_digest(
            root,
            checksums_locator,
            _require_hash_field(record, "checksum_manifest_sha256", field),
            field,
        )
        _verify_cell_checksums(root, cell_locator, result_sha256, field)
        if record_version == RUN_RECORD_SCHEMA_VERSION_V2:
            _verify_v2_attempt_result(root, cell_locator, record, field)
        _verify_resolved_executable(record, field)


def _verify_v2_attempt_result(
    root: Path,
    cell_locator: str,
    record: Mapping[str, object],
    field: str,
) -> None:
    """Bind one retained v2 result's redundant role/profile/runtime/sentinel identities."""
    raw = _read_relative_regular(
        root,
        f"{cell_locator}/result.json",
        _CELL_MEMBER_LIMITS["result.json"],
    )
    from ._evidence import _strict_json_bytes

    primitive = _strict_json_bytes(raw, "result.json")
    result = _require_object(primitive, f"{field}.result")
    expected = {
        "profile_role": record.get("role"),
        "package_version": record.get("package_version"),
        "native_version": record.get("native_version"),
        "native_version_integer": record.get("native_version_integer"),
        "profile_identity_sha256": record.get("profile_identity_sha256"),
        "runtime_identity_sha256": record.get("runtime_identity_sha256"),
        "sentinel_identity_sha256": record.get("sentinel_identity_sha256"),
        "step_dt": record.get("step_dt"),
        "repeat_id": record.get("repeat_id"),
    }
    if (
        result.get("schema") != "metrifid.native_upgrade_worker_result"
        or result.get("schema_version") != 2
        or result.get("status") != "COMPLETED"
        or any(result.get(name) != value for name, value in expected.items())
    ):
        raise RuntimeReviewRunOutputError(f"{field} v2 result binding changed")


def _verify_cell_checksums(
    root: Path,
    cell_locator: str,
    result_sha256: str,
    field: str,
) -> None:
    """Require the exact six-member cell and replay all five manifest-bound payloads."""
    actual_members = _list_relative_directory(root, cell_locator)
    if actual_members != _CELL_MEMBERS:
        raise RuntimeReviewRunOutputError(f"{field} captured cell has an invalid closed shape")
    manifest_locator = f"{cell_locator}/CHECKSUMS.sha256"
    manifest_raw = _read_relative_regular(root, manifest_locator, 16 * 1024)
    try:
        manifest_text = manifest_raw.decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise RuntimeReviewRunOutputError(f"{field} checksum manifest is not ASCII") from exc
    if not manifest_text.endswith("\n"):
        raise RuntimeReviewRunOutputError(f"{field} checksum manifest lacks its final newline")
    lines = manifest_text.splitlines()
    expected_names = tuple(_CELL_MEMBER_LIMITS)
    if len(lines) != len(expected_names):
        raise RuntimeReviewRunOutputError(f"{field} checksum manifest must contain five lines")
    declared: dict[str, str] = {}
    for line in lines:
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise RuntimeReviewRunOutputError(f"{field} checksum manifest line is malformed")
        digest, name = match.groups()
        if name in declared:
            raise RuntimeReviewRunOutputError(f"{field} checksum manifest repeats a member")
        declared[name] = digest
    if tuple(declared) != expected_names:
        raise RuntimeReviewRunOutputError(
            f"{field} checksum manifest member order or field set is not canonical"
        )
    for name, maximum in _CELL_MEMBER_LIMITS.items():
        payload = _read_relative_regular(root, f"{cell_locator}/{name}", maximum)
        if hashlib.sha256(payload).hexdigest() != declared[name]:
            raise RuntimeReviewRunOutputError(f"{field} checksum mismatch for {name}")
    if declared["result.json"] != result_sha256:
        raise RuntimeReviewRunOutputError(f"{field} result hash disagrees with checksum manifest")


def _verify_process_record(
    root: Path, record: Mapping[str, object], directory: str, field: str
) -> None:
    """Rehash the exact four retained files for one process observation."""
    expected = {
        "command_locator": f"{directory}/command.json",
        "stdout_locator": f"{directory}/stdout.txt",
        "stderr_locator": f"{directory}/stderr.txt",
        "exit_code_locator": f"{directory}/exit_code.txt",
    }
    for locator_field, expected_locator in expected.items():
        locator = _require_locator_field(record, locator_field, field)
        if locator != expected_locator:
            raise RuntimeReviewRunOutputError(f"{field}.{locator_field} is not canonical")
        hash_field = locator_field.removesuffix("_locator") + "_sha256"
        _verify_relative_digest(
            root,
            locator,
            _require_hash_field(record, hash_field, field),
            field,
        )
    exit_code = record.get("exit_code")
    if type(exit_code) is not int or exit_code != 0:
        raise RuntimeReviewRunOutputError(f"{field}.exit_code must be the successful integer zero")
    if record.get("no_exit_status") is not None:
        raise RuntimeReviewRunOutputError(
            f"{field}.no_exit_status must be null when the process exited successfully"
        )
    exit_raw = _read_relative_regular(root, expected["exit_code_locator"], 128)
    if exit_raw != f"{exit_code}\n".encode("ascii"):
        raise RuntimeReviewRunOutputError(f"{field} retained exit code disagrees with record")


def _verify_resolved_executable(record: Mapping[str, object], field: str) -> None:
    """Rehash the canonical resolved executable retained by a process record."""
    resolved = record.get("resolved_interpreter")
    if type(resolved) is not str or not Path(resolved).is_absolute():
        raise RuntimeReviewRunOutputError(f"{field}.resolved_interpreter must be absolute")
    payload = _read_absolute_regular(Path(resolved), _MAX_REFERENCED_BYTES, field)
    if hashlib.sha256(payload).hexdigest() != _require_hash_field(
        record, "resolved_executable_sha256", field
    ):
        raise RuntimeReviewRunOutputError(f"{field} resolved executable changed")


def _verify_lexical_interpreter(
    record: Mapping[str, object],
    input_configuration: Mapping[str, object],
    role: str,
    field: str,
) -> None:
    """Cross-bind one process record to its exact explicitly configured launcher."""
    lexical = record.get("lexical_interpreter")
    configured = input_configuration.get(f"{role}_python")
    if type(lexical) is not str or lexical != configured or not Path(lexical).is_absolute():
        raise RuntimeReviewRunOutputError(f"{field} lexical interpreter is not configured")


def _verify_relative_hash(
    root: Path,
    locator: str,
    record: Mapping[str, object],
    hash_field: str,
) -> bytes:
    """Read one retained relative file and compare it with a hash field."""
    payload = _read_relative_regular(root, locator, _MAX_REFERENCED_BYTES)
    if hashlib.sha256(payload).hexdigest() != _require_hash_field(record, hash_field, locator):
        raise RuntimeReviewRunOutputError(f"referenced output bytes changed: {locator}")
    return payload


def _verify_relative_digest(root: Path, locator: str, expected: str, field: str) -> None:
    """Require one retained relative file to match its declared SHA-256."""
    payload = _read_relative_regular(root, locator, _MAX_REFERENCED_BYTES)
    if hashlib.sha256(payload).hexdigest() != expected:
        raise RuntimeReviewRunOutputError(f"{field} referenced bytes changed: {locator}")


def _list_relative_directory(root: Path, locator: str) -> tuple[str, ...]:
    """List one root-relative no-follow directory through a retained descriptor."""
    admitted = PurePosixPath(_admit_locator(locator, "directory locator"))
    descriptor = _open_absolute_directory(root)
    try:
        for component in admitted.parts:
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        try:
            return tuple(
                sorted(os.listdir(descriptor))  # noqa: PTH208 - descriptor confinement
            )
        except OSError as exc:
            raise RuntimeReviewRunOutputError(
                f"referenced output directory could not be listed: {locator}"
            ) from exc
    finally:
        os.close(descriptor)


def _read_relative_regular(root: Path, locator: str, maximum: int) -> bytes:
    """Read one bounded root-relative regular file through no-follow descriptors."""
    admitted = PurePosixPath(_admit_locator(locator, "record locator"))
    root_fd = _open_absolute_directory(root)
    current_fd = root_fd
    try:
        for component in admitted.parts[:-1]:
            child = os.open(component, _DIR_FLAGS, dir_fd=current_fd)
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = child
        try:
            descriptor = os.open(admitted.name, _READ_FLAGS, dir_fd=current_fd)
        except OSError as exc:
            raise RuntimeReviewRunOutputError(
                f"referenced output member is unavailable: {locator}"
            ) from exc
        try:
            return _read_bounded_descriptor(descriptor, maximum, locator)
        finally:
            os.close(descriptor)
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)


def _read_absolute_regular(path: Path, maximum: int, field: str) -> bytes:
    """Read one bounded absolute no-follow regular resource."""
    try:
        descriptor = os.open(path, _READ_FLAGS)
    except OSError as exc:
        raise RuntimeReviewRunOutputError(f"{field} resource is unavailable: {path}") from exc
    try:
        return _read_bounded_descriptor(descriptor, maximum, field)
    finally:
        os.close(descriptor)


def _read_bounded_descriptor(descriptor: int, maximum: int, field: str) -> bytes:
    """Read one regular descriptor up to an inclusive byte bound."""
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeReviewRunOutputError(f"{field} must reference a regular file")
    if metadata.st_size > maximum:
        raise RuntimeReviewRunOutputError(f"{field} exceeds its verification byte bound")
    chunks: list[bytes] = []
    remaining = maximum + 1
    while remaining:
        block = os.read(descriptor, min(remaining, 1024 * 1024))
        if not block:
            break
        chunks.append(block)
        remaining -= len(block)
    payload = b"".join(chunks)
    if len(payload) > maximum:
        raise RuntimeReviewRunOutputError(f"{field} exceeds its verification byte bound")
    return payload


def _create_owned_output_root(path: Path) -> tuple[int, dict[str, tuple[int, int]]]:
    """Create an absent possibly nested root through no-follow directory descriptors."""
    absolute = path.absolute()
    descriptor = os.open(absolute.anchor, _DIR_FLAGS)
    created: dict[str, tuple[int, int]] = {}
    relative_parts: list[str] = []
    try:
        for index, component in enumerate(absolute.parts[1:]):
            final = index == len(absolute.parts[1:]) - 1
            relative_parts.append(component)
            key = "/".join(relative_parts)
            try:
                metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
                metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
                created[key] = (metadata.st_dev, metadata.st_ino)
            else:
                if final:
                    raise RuntimeReviewRunOutputError(
                        "runtime-review run output already exists; output is never overwritten"
                    )
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise RuntimeReviewRunOutputError("run output path contains an unsafe ancestor")
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            bound = os.fstat(child)
            if (bound.st_dev, bound.st_ino) != (metadata.st_dev, metadata.st_ino):
                os.close(child)
                raise RuntimeReviewRunOutputError("run output ancestor was replaced")
            os.close(descriptor)
            descriptor = child
        if not created or key not in created:
            raise RuntimeReviewRunOutputError("run output root was not newly owned")
        return descriptor, created
    except BaseException:
        os.close(descriptor)
        raise


def _open_absolute_directory(path: Path) -> int:
    """Open one absolute directory a component at a time without following links."""
    absolute = path.absolute()
    descriptor = os.open(absolute.anchor, _DIR_FLAGS)
    try:
        for component in absolute.parts[1:]:
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _require_absent(parent_fd: int, name: str, field: str) -> None:
    """Require one directory-relative member name to be absent without following it."""
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeReviewRunOutputError(f"{field} could not be inspected") from exc
    raise RuntimeReviewRunOutputError(f"{field} already exists; output is never overwritten")


def _cell_locator(role: str, step_dt: str, repeat_id: int) -> PurePosixPath:
    """Return the frozen retained-evidence locator for one canonical execution slot."""
    admitted_role = _admit_role(role)
    if step_dt not in STEP_DTS:
        raise ValueError(f"step_dt must be one of {list(STEP_DTS)}")
    if type(repeat_id) is not int or repeat_id not in REPEAT_IDS:
        raise ValueError(f"repeat_id must be one of {list(REPEAT_IDS)}")
    return PurePosixPath(
        "captured_evidence",
        admitted_role,
        _STEP_DIRECTORY_TOKENS[step_dt],
        f"repeat_{repeat_id}",
    )


def _admit_role(role: str) -> str:
    """Require the closed baseline/candidate role vocabulary."""
    if type(role) is not str or role not in PROFILE_ROLES:
        raise ValueError("role must be baseline or candidate")
    return role


def _admit_locator(value: object, field: str, *, allow_root: bool = False) -> str:
    """Require one normalized root-relative POSIX locator without traversal."""
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise RuntimeReviewRunOutputError(f"{field} must be a nonempty relative POSIX path")
    path = PurePosixPath(value)
    if allow_root and value == ".":
        return value
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(component in {"", ".", ".."} for component in path.parts)
    ):
        raise RuntimeReviewRunOutputError(f"{field} must be a normalized relative POSIX path")
    return value


def _require_object(value: object, field: str) -> dict[str, object]:
    """Require one concrete JSON object with string keys."""
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise RuntimeReviewRunOutputError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _require_array(value: object, field: str) -> list[object]:
    """Require one concrete JSON array."""
    if type(value) is not list:
        raise RuntimeReviewRunOutputError(f"{field} must be an array")
    return cast(list[object], value)


def _require_fields(
    record: Mapping[str, object], expected: set[str] | frozenset[str], field: str
) -> None:
    """Require one nested operational record to use its exact closed field set."""
    actual = set(record)
    if actual != expected:
        raise RuntimeReviewRunOutputError(f"{field} field set is not closed")


def _require_locator_field(record: Mapping[str, object], name: str, field: str) -> str:
    """Read and admit one normalized relative locator field."""
    value = record.get(name)
    return _admit_locator(value, f"{field}.{name}")


def _require_hash_field(record: Mapping[str, object], name: str, field: str) -> str:
    """Read and admit one lowercase SHA-256 field."""
    try:
        return require_sha256(record.get(name), f"{field}.{name}")
    except (TypeError, ValueError) as exc:
        raise RuntimeReviewRunOutputError(str(exc)) from exc


__all__ = [
    "ADMITTED_RUN_CONFIG_LOCATOR",
    "COMPLETED_RUN_RECORD_LOCATOR",
    "FROZEN_EVIDENCE_WORKER_SHA256",
    "LEGACY_FROZEN_EVIDENCE_WORKER_SHA256",
    "GENERATED_REVIEW_CONFIG_LOCATOR",
    "OwnedRuntimeReviewRunOutput",
    "RUN_RECORD_SCHEMA",
    "RUN_RECORD_SCHEMA_VERSION",
    "RUNTIME_REVIEW_RUN_CLAIM_LIMITATIONS",
    "RetainedFile",
    "RetainedProcessOutput",
    "RuntimeReviewRunOutputError",
    "VerifiedRuntimeReviewRunRecord",
    "prepare_runtime_review_run_output",
    "verify_completed_run_record",
    "verify_owned_runtime_review_run_output",
]
