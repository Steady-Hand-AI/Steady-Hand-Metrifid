"""Run, retain, read and independently replay bounded repair records.

Returned paths name verified files in a descriptor-bound directory. A read
confirms internal consistency, not authorship or fresh simulation. Publication
failures may leave committed files; only owned private temporaries are cleaned.
"""

from __future__ import annotations

import errno
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from metrifid import _owned_artifacts as artifacts
from metrifid.workload_qualification._owned_output import OwnedOutputError, OwnedOutputRoot

from . import _codec, _engine
from ._request import RepairInputError, load_request
from ._types import EngineResult, ExecutionCosts, ReplayResult

_RECORD_LIMIT = _codec.MAX_RESULT_BYTES
_MEMBER_LIMITS = _codec.MEMBER_LIMITS
_EMPTY = PurePosixPath()
PathInput = str | os.PathLike[str]


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """Command outcome, recorded outcome and only verified public artifact paths."""

    operation: str
    command_status: str
    command_exit_code: int
    reason: str
    recorded_status: str | None = None
    recorded_exit_code: int | None = None
    qualified: bool = False
    result_path: Path | None = None
    artifact_paths: dict[str, Path] = field(default_factory=dict)
    document: dict[str, Any] | None = None
    actual_work: ExecutionCosts | None = None
    diagnostic: str | None = None
    partial_output: bool = False


@dataclass(frozen=True)
class _ReadRecord:
    """One admitted document and the exact bounded bytes from its bound root."""

    document: dict[str, Any]
    result: EngineResult | ReplayResult
    raw: bytes
    members: dict[str, bytes]
    filename: str


class _Refused(ValueError):
    """Carry a file-workflow refusal without starting an evaluator."""

    def __init__(self, reason: str, message: str) -> None:
        """Retain the stable refusal reason separately from its diagnostic."""
        self.reason = reason
        super().__init__(message)


def _cause_oserror(exc: BaseException) -> OSError | None:
    """Find an actual OS cause without interpreting human diagnostic strings."""
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, OSError):
            return current
        current = current.__cause__
    return None


def _failure(
    operation: str,
    exc: Exception,
    phase: str,
    *,
    work: ExecutionCosts | None = None,
    output_attempted: bool = False,
) -> WorkflowResult:
    """Keep malformed data, OS failures and partial publication distinct."""
    refused, reason = False, "INTERNAL_ERROR"
    if phase == "publication":
        reason = "OUTPUT_PUBLICATION_FAILED"
    elif isinstance(exc, _Refused):
        refused, reason = True, exc.reason
    elif isinstance(exc, RepairInputError):
        refused = exc.reason != "INPUT_READ_FAILED"
        reason = exc.reason
    elif isinstance(exc, OwnedOutputError):
        cause = _cause_oserror(exc)
        refused = (
            cause is None
            or (
                phase == "destination" and cause.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP)
            )
            or (phase == "input" and cause.errno in (errno.ELOOP, errno.ENOTDIR))
        )
        reason = "INVALID_DESTINATION" if phase == "destination" else "INVALID_RESULT"
        if not refused:
            reason = "FILE_IO_FAILED"
    elif isinstance(exc, (OSError, artifacts.OwnedArtifactError)):
        reason = "FILE_IO_FAILED"
    return WorkflowResult(
        operation,
        "REFUSED" if refused else "ERROR",
        64 if refused else 70,
        reason,
        actual_work=work,
        diagnostic=f"{type(exc).__name__}: {str(exc)[:500]}",
        partial_output=output_attempted,
    )


def _absolute(path: PathInput) -> Path:
    """Keep the supplied components intact for the subsequent no-follow walk."""
    try:
        raw = os.fspath(path)
        if not isinstance(raw, str) or not raw or "\x00" in raw:
            raise ValueError("Expected a nonempty text path without NUL")
        return Path(raw).absolute()
    except (TypeError, ValueError) as exc:
        raise _Refused("INVALID_PATH", "Expected a valid file-system path") from exc


def _own_destination(path: PathInput, source: Path | None = None) -> OwnedOutputRoot:
    """Bind a new directory under an existing real parent, without recursive creation."""
    output = _absolute(path)
    if output.parent == output:
        raise _Refused("INVALID_DESTINATION", "Output must be a new child directory")
    with OwnedOutputRoot.bind_existing(output.parent):
        # Only compare lexical canonical forms after proving every parent component
        # is a real directory. Do not resolve away symlinks before that proof.
        normalized = Path(os.path.normpath(output))
        if source is not None:
            original = Path(os.path.normpath(source))
            if (
                normalized == original
                or normalized in original.parents
                or original in normalized.parents
            ):
                raise _Refused(
                    "INVALID_DESTINATION", "Replay output must be outside its input bundle"
                )
        try:
            output.lstat()
        except FileNotFoundError:
            pass
        else:
            raise _Refused(
                "INVALID_DESTINATION", "Output already exists and will not be overwritten"
            )
    return OwnedOutputRoot(output)


def _verify_public_root(owner: OwnedOutputRoot) -> None:
    """Require the displayed pathname to still name the retained directory."""
    retained = owner.open_owned_directory(_EMPTY)
    try:
        with OwnedOutputRoot.bind_existing(owner.path) as named:
            current = named.open_owned_directory(_EMPTY)
            try:
                left, right = os.fstat(retained), os.fstat(current)
                if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
                    raise OwnedOutputError(
                        "Public output pathname no longer names the bound directory"
                    )
            finally:
                os.close(current)
    finally:
        os.close(retained)


def _read_record(owner: OwnedOutputRoot, filename: str) -> _ReadRecord:
    """Read declared fixed members; unrelated directory entries are never consumed."""
    if filename not in ("repair.json", "replay.json"):
        raise _Refused("INVALID_RESULT", "Expected repair.json or replay.json")
    raw = owner.read_bytes(PurePosixPath(filename), _RECORD_LIMIT)
    document = _codec.decode_document(raw)
    operation = document.get("operation")
    expected_name = (
        "repair.json" if operation == "run" else "replay.json" if operation == "replay" else None
    )
    if expected_name != filename:
        raise _Refused("INVALID_RESULT", "Record operation does not match its fixed filename")
    records = _codec.artifact_records(document)
    members: dict[str, bytes] = {}
    for name, record in records.items():
        if name not in _MEMBER_LIMITS or record.get("filename") != name:
            raise _Refused("INVALID_RESULT", "Artifact must use a fixed version-1 member filename")
        payload = owner.read_bytes(PurePosixPath(name), _MEMBER_LIMITS[name])
        if (
            record.get("size_bytes") != len(payload)
            or record.get("sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise _Refused("INVALID_RESULT", "Retained artifact length or digest does not match")
        members[name] = payload
    result = _codec.admit_record(document, members)
    retained = _ReadRecord(document, result, raw, members, filename)
    _verify_record(owner, retained)
    return retained


def _verify_record(owner: OwnedOutputRoot, retained: _ReadRecord) -> None:
    """Recheck exact retained bytes and the public root before returning any path."""
    for name, payload in retained.members.items():
        if owner.read_bytes(PurePosixPath(name), _MEMBER_LIMITS[name]) != payload:
            raise OwnedOutputError("Retained member changed after admission")
    if owner.read_bytes(PurePosixPath(retained.filename), _RECORD_LIMIT) != retained.raw:
        raise OwnedOutputError("Result changed after admission")
    _verify_public_root(owner)


def _publish(
    owner: OwnedOutputRoot, filename: str, payload: bytes, members: dict[str, bytes]
) -> _ReadRecord:
    """Seal and commit members first, the result last, preserving owned finals on error."""
    document = _codec.decode_document(payload)
    _codec.admit_record(document, members)
    if len(payload) > _RECORD_LIMIT or any(
        name not in _MEMBER_LIMITS or len(value) > _MEMBER_LIMITS[name]
        for name, value in members.items()
    ):
        raise ValueError("Serialized output exceeds the declared record/member bounds")
    if filename == "replay.json" and members:
        raise ValueError("Replay publication cannot contain a second run bundle")
    retained: list[artifacts.OwnedArtifact] = []
    parent = owner.open_owned_directory(_EMPTY)
    try:
        for name, value in [*sorted(members.items()), (filename, payload)]:
            member = artifacts.create_owned_artifact(parent, ".repair-")
            retained.append(member)
            artifacts.write_owned_bytes(member, value)
            artifacts.commit_owned_artifact(member, name)
        for member in retained:
            member.verify()
        return _read_record(owner, filename)
    finally:
        for member in reversed(retained):
            member.cleanup()
            member.close()
        os.close(parent)


def _verified_workflow(
    operation: str, owner: OwnedOutputRoot, record: _ReadRecord
) -> WorkflowResult:
    """Expose the admitted outcome separately from a successful native-free read."""
    result = record.result
    names = [*record.members, record.filename]
    paths = {name: owner.display_path(PurePosixPath(name)) for name in names}
    return WorkflowResult(
        operation,
        "READ" if operation == "show" else result.status,
        0 if operation == "show" else result.exit_code,
        result.reason,
        recorded_status=result.status,
        recorded_exit_code=result.exit_code,
        qualified=result.qualified,
        result_path=paths[record.filename],
        artifact_paths=paths,
        document=record.document,
        actual_work=result.costs,
        diagnostic=result.diagnostic,
    )


def _unpublished(operation: str, result: EngineResult | ReplayResult) -> WorkflowResult:
    """Return an execution refusal without asserting that a saved record exists."""
    return WorkflowResult(
        operation,
        result.status,
        result.exit_code,
        result.reason,
        actual_work=result.costs,
        diagnostic=result.diagnostic,
    )


def run_request_file(request_path: PathInput, output_directory: PathInput) -> WorkflowResult:
    """Admit an explicit request and new destination, execute repair, then retain its result."""
    owner: OwnedOutputRoot | None = None
    result: EngineResult | None = None
    phase, output_attempted = "input", False
    try:
        request = load_request(request_path)
        phase, output_attempted = "destination", True
        owner = _own_destination(output_directory)
        phase = "execution"
        result = _engine.repair_controller(request)
        if result.status == "REFUSED":
            return _unpublished("run", result)
        phase = "publication"
        payload, members = _codec.serialize_run(result)
        record = _publish(owner, "repair.json", payload, members)
        return _verified_workflow("run", owner, record)
    except Exception as exc:
        return _failure(
            "run",
            exc,
            phase,
            work=result.costs if result else None,
            output_attempted=output_attempted,
        )
    finally:
        if owner is not None:
            owner.close()


def read_result_file(result_path: PathInput) -> WorkflowResult:
    """Read and verify one run or replay record without importing native libraries."""
    try:
        path = _absolute(result_path)
        with OwnedOutputRoot.bind_existing(path.parent) as owner:
            record = _read_record(owner, path.name)
            return _verified_workflow("show", owner, record)
    except Exception as exc:
        return _failure("show", exc, "input")


def replay_run_directory(run_directory: PathInput, output_directory: PathInput) -> WorkflowResult:
    """Re-admit a retained run and independently evaluate its selected controller."""
    owner: OwnedOutputRoot | None = None
    result: ReplayResult | None = None
    phase, output_attempted = "input", False
    try:
        with OwnedOutputRoot.bind_existing(_absolute(run_directory)) as source:
            record = _read_record(source, "repair.json")
            if not isinstance(record.result, EngineResult):
                raise _Refused("INVALID_RESULT", "Replay requires a run bundle")
            if record.result.selected_parameters is None:
                raise _Refused(
                    "NO_SELECTED_CONTROLLER", "The retained run has no selected controller"
                )
            if set(record.members) != set(_MEMBER_LIMITS):
                raise _Refused(
                    "INVALID_RESULT", "Replay requires both scenes, reset and selected policy"
                )
            phase, output_attempted = "destination", True
            owner = _own_destination(output_directory, source.path)
            phase = "input"
            _verify_record(source, record)
            phase = "execution"
            result = _engine.replay_controller(record.result)
        if (
            result.status == "REFUSED"
            and result.costs.known_completed_steps == 0
            and result.costs.unknown_step_calls == 0
        ):
            return _unpublished("replay", result)
        phase = "publication"
        payload = _codec.serialize_replay(result, hashlib.sha256(record.raw).hexdigest())
        retained = _publish(owner, "replay.json", payload, {})
        return _verified_workflow("replay", owner, retained)
    except Exception as exc:
        return _failure(
            "replay",
            exc,
            phase,
            work=result.costs if result else None,
            output_attempted=output_attempted,
        )
    finally:
        if owner is not None:
            owner.close()
