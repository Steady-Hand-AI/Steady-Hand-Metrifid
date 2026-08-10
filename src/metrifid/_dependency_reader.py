"""MuJoCo-reported dependency discovery and snapshot containment.

MuJoCo 3.10.0 resolves contained ``mesh``, ``hfield`` and ``texture`` assets through the effective
compiler asset directories, but ``mju_getXMLDependencies`` reports some of those assets resolved
against the main MJCF directory instead. A reported path can therefore name a file the compiler
never opens. This module keeps ``mju_getXMLDependencies`` as the dependency inventory and, when a
reported asset does not bind to a contained measured member, re-derives the path the compiler
actually uses from the declared asset element and the effective compiler directories.

The declarations that feed that fallback come from the actual include-expanded composite MJCF
document, never from a lexical scan of the measured snapshot. The snapshot deliberately retains
every regular file below the model root, so an unrelated XML file that MuJoCo never includes must
not influence compiler settings, asset declarations, dependencies, or identity. Repeated global
compiler sections are legal in MuJoCo and the last explicit setting encountered in composite
document order wins.

Resolution never searches the tree, never matches by basename or suffix, and never accepts a member
merely because compilation later succeeds: a dependency must bind to exactly one justified contained
measured member or the existing refusal is raised.
"""

from __future__ import annotations

import hashlib
import os
import posixpath
import stat
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import cast

import mujoco as mujoco  # type: ignore[import-untyped]

from ._model_closure import ModelClosureSnapshot, ModelRole, refuse
from .operational import OperationalReasonCode
from .schemas import ModelClosureMember


def _compile_error_reason(role: ModelRole) -> OperationalReasonCode:
    """Select the role-specific model-compilation refusal code."""
    if role == "baseline":
        return OperationalReasonCode.BASELINE_MODEL_COMPILE_ERROR
    if role == "candidate":
        return OperationalReasonCode.CANDIDATE_MODEL_COMPILE_ERROR
    raise ValueError("dependency discovery requires a baseline or candidate role")


def _discovery_error(
    snapshot: ModelClosureSnapshot,
    role: ModelRole,
    exc: BaseException,
) -> Exception:
    """Wrap dependency-reporter failure with snapshot and role evidence."""
    return refuse(
        _compile_error_reason(role),
        role,
        stage="dependency_discovery",
        snapshot_entrypoint=snapshot.entrypoint,
        exception_type=type(exc).__name__,
        message=str(exc),
    )


def _raw_dependencies(
    snapshot: ModelClosureSnapshot,
    role: ModelRole,
) -> list[str]:
    """Run MuJoCo's dependency reporter against the measured snapshot entrypoint."""
    try:
        values = mujoco.mju_getXMLDependencies(str(snapshot.snapshot_entrypoint))
    except (OSError, RuntimeError, TypeError, ValueError, mujoco.FatalError) as exc:
        raise _discovery_error(snapshot, role, exc) from exc
    if type(values) is not list or any(type(item) is not str for item in values):
        problem = TypeError("mju_getXMLDependencies returned a non-string dependency list")
        raise _discovery_error(snapshot, role, problem)
    return cast(list[str], values)


_ASSET_DIRECTORY_ATTRIBUTE = {
    "mesh": "meshdir",
    "hfield": "meshdir",
    "texture": "texturedir",
}
_COMPOSITE_STAGE = "composite_mjcf_expansion"
_PRECHECK_STAGE = "model_root_precheck"


class _AssetResolutionError(Exception):
    """One declared asset could not be interpreted without guessing."""


def _member_map(snapshot: ModelClosureSnapshot) -> dict[str, ModelClosureMember]:
    """Index measured closure members by unique relative path."""
    members = {member.path: member for member in snapshot.identity.members}
    if len(members) != snapshot.identity.member_count:
        raise ValueError("measured closure member paths are not unique")
    return members


def _descriptor_matches_measured_member(
    before: os.stat_result, path_metadata: os.stat_result, member: ModelClosureMember
) -> bool:
    """True when the opened descriptor is the same regular file that was measured."""
    return (
        not stat.S_ISLNK(path_metadata.st_mode)
        and stat.S_ISREG(path_metadata.st_mode)
        and before.st_dev == path_metadata.st_dev
        and before.st_ino == path_metadata.st_ino
        and before.st_mode == path_metadata.st_mode
        and before.st_size == member.size_bytes
    )


def _read_bounded(descriptor: int, limit: int) -> list[bytes]:
    """Read at most ``limit`` bytes.

    Callers pass one more than the measured member size, so a file that grew between measurement
    and read is detectable rather than silently truncated.
    """
    chunks: list[bytes] = []
    remaining = limit
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return chunks


def _read_dependency_member(
    candidate: Path,
    member: ModelClosureMember,
    role: ModelRole,
    snapshot: ModelClosureSnapshot,
    dependency: str,
    relative: str,
) -> bytes:
    """Load dependency member from candidate, member and role for dependency reader."""
    descriptor, path_metadata = _open_dependency_member(
        candidate, role, snapshot, dependency, relative
    )
    before, after, chunks, stable = _read_open_dependency(
        descriptor, path_metadata, member, role, snapshot, dependency, relative
    )
    content = b"".join(chunks)
    return _validate_dependency_content(
        content, before, after, stable, member, role, snapshot, dependency, relative
    )


def _dependency_read_error(
    role: ModelRole,
    snapshot: ModelClosureSnapshot,
    dependency: str,
    relative: str,
    exc: OSError,
) -> Exception:
    """Return the closed operational refusal for a dependency read error."""
    return refuse(
        OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID,
        role,
        stage="dependency_discovery",
        snapshot_entrypoint=snapshot.entrypoint,
        offending_dependency=dependency,
        relative_dependency=relative,
        exception_type=type(exc).__name__,
    )


def _open_dependency_member(
    candidate: Path,
    role: ModelRole,
    snapshot: ModelClosureSnapshot,
    dependency: str,
    relative: str,
) -> tuple[int, os.stat_result]:
    """Open a dependency without following links and return its path metadata."""
    try:
        path_metadata = candidate.lstat()
        if not stat.S_ISREG(path_metadata.st_mode):
            # Never open a non-regular member. Opening a FIFO for reading blocks until a writer
            # appears, so a special file swapped in after measurement must fail closed here rather
            # than stall the comparison. O_NONBLOCK additionally bounds the race between the lstat
            # above and the open below.
            raise OSError(
                f"measured member is not a regular file: {stat.S_IFMT(path_metadata.st_mode):#o}"
            )
        descriptor = os.open(
            candidate,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
    except OSError as exc:
        raise _dependency_read_error(role, snapshot, dependency, relative, exc) from exc
    return descriptor, path_metadata


def _read_open_dependency(
    descriptor: int,
    path_metadata: os.stat_result,
    member: ModelClosureMember,
    role: ModelRole,
    snapshot: ModelClosureSnapshot,
    dependency: str,
    relative: str,
) -> tuple[os.stat_result, os.stat_result, list[bytes], bool]:
    """Read and restat an opened dependency, always closing its descriptor."""
    try:
        before = os.fstat(descriptor)
        stable = _descriptor_matches_measured_member(before, path_metadata, member)
        # One byte more than measured: a grown file must be detected, not truncated.
        chunks = _read_bounded(descriptor, member.size_bytes + 1) if stable else []
        after = os.fstat(descriptor)
    except OSError as exc:
        raise _dependency_read_error(role, snapshot, dependency, relative, exc) from exc
    finally:
        os.close(descriptor)
    return before, after, chunks, stable


def _validate_dependency_content(
    content: bytes,
    before: os.stat_result,
    after: os.stat_result,
    stable: bool,
    member: ModelClosureMember,
    role: ModelRole,
    snapshot: ModelClosureSnapshot,
    dependency: str,
    relative: str,
) -> bytes:
    """Require read bytes and descriptor metadata to match the measured member."""
    measured_hash = hashlib.sha256(content).hexdigest()
    unchanged = (
        stable
        and before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_mode == after.st_mode
        and before.st_size == after.st_size
        and stat.S_ISREG(after.st_mode)
    )
    if not unchanged or len(content) != member.size_bytes or measured_hash != member.sha256:
        raise refuse(
            OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID,
            role,
            stage="dependency_discovery",
            snapshot_entrypoint=snapshot.entrypoint,
            offending_dependency=dependency,
            relative_dependency=relative,
            issue="dependency_does_not_match_measured_member",
            measured_size_bytes=len(content),
            expected_size_bytes=member.size_bytes,
            measured_sha256=measured_hash,
            expected_sha256=member.sha256,
        )
    return content


def _member_bytes(
    snapshot: ModelClosureSnapshot,
    role: ModelRole,
    relative: str,
    members: dict[str, ModelClosureMember],
    stage: str,
) -> bytes:
    """Read one measured member through the bounded, no-follow, hash-verified path."""
    member = members.get(relative)
    if member is None:
        raise refuse(
            OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID,
            role,
            stage=stage,
            snapshot_entrypoint=snapshot.entrypoint,
            relative_dependency=relative,
            issue="member_not_in_measured_closure",
        )
    candidate = snapshot.snapshot_root.joinpath(*PurePosixPath(relative).parts)
    return _read_dependency_member(candidate, member, role, snapshot, relative, relative)


def read_measured_entrypoint_bytes(
    snapshot: ModelClosureSnapshot,
    role: ModelRole,
) -> bytes:
    """Return the measured main entrypoint bytes for the bounded model-root precheck."""
    members = _member_map(snapshot)
    return _member_bytes(snapshot, role, snapshot.entrypoint, members, _PRECHECK_STAGE)


def first_complete_root_element(data: bytes) -> ElementTree.Element | None:
    """Return the first complete top-level element, ignoring any trailing bytes.

    Official MJCF documents are not always well-formed XML documents. The pinned Unitree G1 scene
    carries a duplicated closing tag that MuJoCo tolerates, so only the first complete top-level
    element is read here and MuJoCo 3.10.0 remains the final syntax authority. This is deliberately
    not a general recovering parser: the bytes up to the first complete element must still parse,
    and a document that never closes a top-level element yields None.
    """
    parser = ElementTree.XMLPullParser(("start", "end"))
    depth = 0
    root_element: ElementTree.Element | None = None
    try:
        parser.feed(data)
    except ElementTree.ParseError:
        pass
    try:
        # ElementTree ships read_events() loosely typed; the pairs are exact.
        for event, element in cast(
            "Iterator[tuple[str, ElementTree.Element[str]]]", parser.read_events()
        ):
            if event == "start":
                depth += 1
            else:
                depth -= 1
                if depth == 0 and root_element is None:
                    root_element = element
    except ElementTree.ParseError:
        pass
    return root_element


def _include_relative(
    snapshot: ModelClosureSnapshot,
    role: ModelRole,
    token: str | None,
) -> str:
    """Resolve one include token against the main MJCF entrypoint directory."""
    text = (token or "").replace("\\", "/")
    if not text or "\x00" in text:
        raise refuse(
            OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID,
            role,
            stage=_COMPOSITE_STAGE,
            snapshot_entrypoint=snapshot.entrypoint,
            issue="include_without_usable_file_attribute",
        )
    if posixpath.isabs(text) or Path(text).is_absolute():
        try:
            root = snapshot.snapshot_root.resolve(strict=True)
            return Path(text).resolve(strict=False).relative_to(root).as_posix()
        except (OSError, RuntimeError, ValueError) as exc:
            raise refuse(
                OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE,
                role,
                stage=_COMPOSITE_STAGE,
                snapshot_entrypoint=snapshot.entrypoint,
                offending_include=text,
            ) from exc
    parent = PurePosixPath(snapshot.entrypoint).parent
    prefix = "" if str(parent) == "." else str(parent)
    joined = posixpath.normpath(posixpath.join(prefix, text))
    if joined == ".." or joined.startswith("../") or posixpath.isabs(joined):
        raise refuse(
            OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE,
            role,
            stage=_COMPOSITE_STAGE,
            snapshot_entrypoint=snapshot.entrypoint,
            offending_include=text,
        )
    return joined


def _included_root(
    snapshot: ModelClosureSnapshot,
    role: ModelRole,
    relative: str,
    members: dict[str, ModelClosureMember],
) -> ElementTree.Element:
    """Return the unique top-level element of one included measured member."""
    data = _member_bytes(snapshot, role, relative, members, _COMPOSITE_STAGE)
    try:
        return ElementTree.fromstring(data)
    except ElementTree.ParseError as exc:
        raise refuse(
            _compile_error_reason(role),
            role,
            stage=_COMPOSITE_STAGE,
            snapshot_entrypoint=snapshot.entrypoint,
            included_file=relative,
            issue="included_file_has_no_unique_top_level_element",
            exception_type=type(exc).__name__,
            message=str(exc),
        ) from exc
