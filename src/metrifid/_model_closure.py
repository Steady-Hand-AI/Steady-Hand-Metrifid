"""Conservative full-root model closure measurement and immutable snapshotting."""

# Keep the baseline import statement order for rename-normalized AST identity.
from __future__ import annotations  # noqa: I001

import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Self, cast

from ._alignment import (
    _ACTIVATION_LAYOUT_WIDTHS as _ACTIVATION_LAYOUT_WIDTHS,
)
from ._alignment import (
    _JOINT_WIDTHS as _JOINT_WIDTHS,
)
from ._alignment import (
    ActivationFamily as ActivationFamily,
)
from ._alignment import (
    AlignedActuator as AlignedActuator,
)
from ._alignment import (
    AlignedJoint as AlignedJoint,
)
from ._alignment import (
    JointType as JointType,
)
from ._alignment import (
    _canonical_targets as _canonical_targets,
)
from ._alignment import (
    _completed_hash as _completed_hash,
)
from ._alignment import (
    _require_exact_object_fields as _require_exact_object_fields,
)
from ._alignment import (
    _nonnegative_int as _nonnegative_int,
)
from ._alignment import (
    _slice as _slice,
)
from ._alignment import (
    _strict_array as _strict_array,
)
from ._alignment import (
    _strict_name as _strict_name,
)
from ._alignment import (
    _unique_sorted_names as _unique_sorted_names,
)
from ._alignment import (
    _valid_target_shape as _valid_target_shape,
)
from ._alignment import (
    _validate_activation_layout as _validate_activation_layout,
)
from ._alignment import (
    _validate_slice as _validate_slice,
)
from ._model_refusal import (
    ModelAdmissionRefusal as ModelAdmissionRefusal,
)
from ._model_refusal import (
    ModelRole as ModelRole,
)
from ._model_refusal import (
    refuse as refuse,
)
from .json_values import CanonicalValue
from .operational import OperationalReasonCode
from .schemas import ModelClosureIdentity, ModelClosureMember

MAX_MODEL_CLOSURE_BYTES: Final[int] = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _EnumeratedMember:
    """Carry one confined source member's path and race-detection metadata."""

    relative_path: str
    absolute_path: Path
    size_bytes: int
    device: int
    inode: int
    mode: int
    _root_binding: _RootBinding | None = None
    _parent_descriptor: int | None = None
    _directory_chain: tuple[_BoundDirectory, ...] = ()


@dataclass(frozen=True, slots=True)
class _BoundDirectory:
    """Bind one admitted child directory to its retained parent descriptor."""

    relative_path: str
    name: str
    descriptor: int
    parent_descriptor: int
    device: int
    inode: int


@dataclass(slots=True)
class _RootBinding:
    """Own the admitted root descriptor and all retained child descriptors."""

    source_root: Path
    descriptor: int
    device: int
    inode: int
    _child_descriptors: list[int]
    _bound_directories: list[_BoundDirectory]
    _closed: bool = False

    def retain_child(self, descriptor: int) -> None:
        """Retain one child descriptor until this source binding closes."""
        self._child_descriptors.append(descriptor)

    def retain_directory(self, directory: _BoundDirectory) -> None:
        """Remember one child name binding for end-to-end revalidation."""
        self._bound_directories.append(directory)

    def close(self) -> None:
        """Close every retained descriptor exactly once."""
        if self._closed:
            return
        self._closed = True
        for descriptor in reversed(self._child_descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            os.close(self.descriptor)
        except OSError:
            pass

    def __del__(self) -> None:
        """Release descriptors if an internal caller drops an enumerated batch."""
        self.close()


@dataclass(frozen=True, slots=True)
class _PendingDirectory:
    """Carry one already-open directory through descriptor-confined traversal."""

    relative_path: str
    descriptor: int
    chain: tuple[_BoundDirectory, ...]
    bound: _BoundDirectory | None
    root_binding: _RootBinding


@dataclass(slots=True)
class ModelClosureSnapshot:
    """One measured source closure and same-byte temporary snapshot tree."""

    source_root: Path
    entrypoint: str
    identity: ModelClosureIdentity
    snapshot_root: Path
    _temporary_directory: tempfile.TemporaryDirectory[str]
    _source_binding: _RootBinding

    @property
    def snapshot_entrypoint(self) -> Path:
        """Return the copied entrypoint inside this immutable snapshot tree."""
        return self.snapshot_root.joinpath(*PurePosixPath(self.entrypoint).parts)

    def close(self) -> None:
        """Remove the private temporary snapshot tree once, if still owned."""
        try:
            self._temporary_directory.cleanup()
        finally:
            self._source_binding.close()

    def __enter__(self) -> Self:
        """Enter the managed resource context and return its bound value."""
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Release resources owned by the managed context."""
        self.close()


def _path_evidence(path: Path | str) -> str:
    """Render a path for refusal evidence without resolving or admitting it."""
    return os.fspath(path)


def _validate_root(model_root: Path, role: ModelRole) -> Path:
    """Admit a real nonsymlink model-root directory for one comparison role."""
    if not model_root.is_absolute():
        raise refuse(
            OperationalReasonCode.MODEL_ROOT_INVALID,
            role,
            model_root=_path_evidence(model_root),
            issue="root_not_absolute",
        )
    try:
        metadata = model_root.lstat()
    except OSError as exc:
        raise refuse(
            OperationalReasonCode.MODEL_ROOT_INVALID,
            role,
            model_root=_path_evidence(model_root),
            issue="root_unavailable",
            exception_type=type(exc).__name__,
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise refuse(
            OperationalReasonCode.MODEL_ROOT_INVALID,
            role,
            model_root=_path_evidence(model_root),
            issue="root_not_real_directory",
        )
    return model_root


def _validate_component(component: str, role: ModelRole, path: str) -> None:
    """Validate one confined UTF-8 model-path component and its filesystem metadata."""
    if component in {"", ".", ".."} or "\\" in component or "\x00" in component:
        raise refuse(
            OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID,
            role,
            path=path,
            issue="invalid_path_component",
        )
    try:
        if component.encode("utf-8", errors="strict").decode("utf-8") != component:
            raise UnicodeError
    except UnicodeError as exc:
        raise refuse(
            OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID,
            role,
            path=path,
            issue="path_not_strict_utf8",
        ) from exc


def _validate_entrypoint(entrypoint: str, role: ModelRole) -> str:
    """Admit a normalized relative MJCF entrypoint path within the model root."""
    if not isinstance(entrypoint, str):
        raise refuse(
            OperationalReasonCode.MODEL_ENTRYPOINT_INVALID,
            role,
            issue="entrypoint_not_string",
        )
    if "\\" in entrypoint or "\x00" in entrypoint:
        raise refuse(
            OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE,
            role,
            entrypoint=entrypoint,
            issue="forbidden_path_syntax",
        )
    path = PurePosixPath(entrypoint)
    invalid = (
        path.is_absolute()
        or entrypoint in {"", "."}
        or any(part in {"", ".", ".."} for part in path.parts)
    )
    if invalid or path.as_posix() != entrypoint:
        raise refuse(
            OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE,
            role,
            entrypoint=entrypoint,
            issue="entrypoint_not_normal_relative_posix",
        )
    for component in path.parts:
        _validate_component(component, role, entrypoint)
    if path.suffix != ".xml":
        raise refuse(
            OperationalReasonCode.MODEL_ENTRYPOINT_INVALID,
            role,
            entrypoint=entrypoint,
            issue="entrypoint_not_xml",
        )
    return entrypoint


def _metadata(entry: os.DirEntry[str], role: ModelRole, relative: str) -> os.stat_result:
    """Capture no-follow metadata for one regular source member."""
    try:
        return entry.stat(follow_symlinks=False)
    except OSError as exc:
        raise refuse(
            OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID,
            role,
            path=relative,
            exception_type=type(exc).__name__,
        ) from exc


def _directory_flags() -> int:
    """Return nonblocking, no-follow flags for a directory descriptor."""
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, int]:
    """Return the stable object identity used to bind a directory."""
    return metadata.st_dev, metadata.st_ino


def _open_root_binding(model_root: Path, role: ModelRole) -> _RootBinding:
    """Open and bind the declared root without following a replacement link."""
    root = _validate_root(model_root, role)
    try:
        descriptor = os.open(root, _directory_flags())
    except OSError as exc:
        raise refuse(
            OperationalReasonCode.MODEL_ROOT_INVALID,
            role,
            model_root=_path_evidence(root),
            issue="root_unavailable",
            exception_type=type(exc).__name__,
        ) from exc
    try:
        opened = os.fstat(descriptor)
        named = root.lstat()
    except OSError as exc:
        os.close(descriptor)
        raise refuse(
            OperationalReasonCode.MODEL_ROOT_INVALID,
            role,
            model_root=_path_evidence(root),
            issue="root_changed_during_open",
            exception_type=type(exc).__name__,
        ) from exc
    valid = (
        stat.S_ISDIR(opened.st_mode)
        and not stat.S_ISLNK(named.st_mode)
        and stat.S_ISDIR(named.st_mode)
        and _directory_identity(opened) == _directory_identity(named)
    )
    if not valid:
        os.close(descriptor)
        raise refuse(
            OperationalReasonCode.MODEL_ROOT_INVALID,
            role,
            model_root=_path_evidence(root),
            issue="root_changed_during_open",
        )
    return _RootBinding(root, descriptor, opened.st_dev, opened.st_ino, [], [])


def _directory_still_named(
    parent_descriptor: int,
    name: str,
    expected: tuple[int, int],
) -> bool:
    """Return whether a child name still identifies the admitted directory."""
    try:
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISDIR(current.st_mode) and _directory_identity(current) == expected


def _directory_mutated(
    role: ModelRole,
    relative: str,
    issue: str,
    **evidence: CanonicalValue,
) -> ModelAdmissionRefusal:
    """Build a typed refusal for a changed directory binding."""
    return refuse(
        OperationalReasonCode.MODEL_CLOSURE_MUTATED,
        role,
        path=relative,
        issue=issue,
        **evidence,
    )


def _open_child_directory(
    directory: _PendingDirectory,
    entry: os.DirEntry[str],
    metadata: os.stat_result,
    role: ModelRole,
    relative: str,
) -> _BoundDirectory:
    """Open one child relative to its retained parent and bind its identity."""
    expected = _directory_identity(metadata)
    try:
        descriptor = os.open(entry.name, _directory_flags(), dir_fd=directory.descriptor)
    except OSError as exc:
        if not _directory_still_named(directory.descriptor, entry.name, expected):
            raise _directory_mutated(role, relative, "directory_changed_before_open") from exc
        raise refuse(
            OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID,
            role,
            path=relative,
            exception_type=type(exc).__name__,
        ) from exc
    try:
        opened = os.fstat(descriptor)
    except OSError as exc:
        os.close(descriptor)
        raise _directory_mutated(role, relative, "directory_stat_failed_after_open") from exc
    if not stat.S_ISDIR(opened.st_mode) or _directory_identity(opened) != expected:
        os.close(descriptor)
        raise _directory_mutated(role, relative, "directory_changed_during_open")
    bound = _BoundDirectory(
        relative,
        entry.name,
        descriptor,
        directory.descriptor,
        opened.st_dev,
        opened.st_ino,
    )
    directory.root_binding.retain_child(descriptor)
    directory.root_binding.retain_directory(bound)
    return bound


def _verify_bound_directory(directory: _BoundDirectory, role: ModelRole) -> None:
    """Refuse when a retained child descriptor is no longer named by its parent."""
    expected = directory.device, directory.inode
    if not _directory_still_named(directory.parent_descriptor, directory.name, expected):
        raise _directory_mutated(
            role,
            directory.relative_path,
            "directory_changed_after_open",
        )


def _verify_public_root_binding(binding: _RootBinding, role: ModelRole) -> None:
    """Refuse unless the declared root still names the retained root object."""
    try:
        opened = os.fstat(binding.descriptor)
        named = binding.source_root.lstat()
    except OSError as exc:
        raise _directory_mutated(
            role,
            _path_evidence(binding.source_root),
            "root_changed_after_open",
            exception_type=type(exc).__name__,
        ) from exc
    valid = (
        stat.S_ISDIR(opened.st_mode)
        and not stat.S_ISLNK(named.st_mode)
        and stat.S_ISDIR(named.st_mode)
        and _directory_identity(opened) == _directory_identity(named)
        and _directory_identity(opened) == (binding.device, binding.inode)
    )
    if not valid:
        raise _directory_mutated(
            role,
            _path_evidence(binding.source_root),
            "root_changed_after_open",
        )


def _verify_bound_tree(binding: _RootBinding, role: ModelRole) -> None:
    """Revalidate the public root and every admitted child directory name."""
    _verify_public_root_binding(binding, role)
    for directory in binding._bound_directories:
        _verify_bound_directory(directory, role)


def _enumerate_members(
    root: Path,
    role: ModelRole,
    max_bytes: int,
    _binding: _RootBinding | None = None,
) -> tuple[_EnumeratedMember, ...]:
    """Enumerate confined regular model files in path order under byte limits."""
    binding = _binding if _binding is not None else _open_root_binding(root, role)
    owns_binding = _binding is None
    pending = [_PendingDirectory("", binding.descriptor, (), None, binding)]
    members: list[_EnumeratedMember] = []
    total_bytes = 0
    try:
        while pending:
            directory = pending.pop()
            if directory.bound is not None:
                _verify_bound_directory(directory.bound, role)
            try:
                entries = sorted(os.scandir(directory.descriptor), key=lambda item: item.name)
            except OSError as exc:
                raise refuse(
                    OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID,
                    role,
                    path=directory.relative_path or _path_evidence(binding.source_root),
                    exception_type=type(exc).__name__,
                ) from exc
            for entry in entries:
                queued_before = len(pending)
                member = _enumerated_entry(directory, role, entry, pending)
                if member is None:
                    if len(pending) > queued_before and pending[-1].bound is not None:
                        _verify_bound_directory(pending[-1].bound, role)
                    continue
                total_bytes += member.size_bytes
                if total_bytes > max_bytes:
                    over = {
                        "measured_metadata_bytes": total_bytes,
                        "first_over_budget_path": member.relative_path,
                    }
                    reason = OperationalReasonCode.MODEL_CLOSURE_BUDGET_EXCEEDED
                    raise refuse(
                        reason,
                        role,
                        max_bytes=max_bytes,
                        **cast("dict[str, CanonicalValue]", over),
                    )
                members.append(member)
    except Exception:
        if owns_binding:
            binding.close()
        raise
    return tuple(sorted(members, key=lambda item: item.relative_path))


def _enumerated_entry(
    directory: _PendingDirectory,
    role: ModelRole,
    entry: os.DirEntry[str],
    pending: list[_PendingDirectory],
) -> _EnumeratedMember | None:
    """Validate one directory entry and return a regular measured member."""
    relative = f"{directory.relative_path}/{entry.name}".lstrip("/")
    for component in PurePosixPath(relative).parts:
        _validate_component(component, role, relative)
    metadata = _metadata(entry, role, relative)
    if stat.S_ISLNK(metadata.st_mode):
        raise refuse(
            OperationalReasonCode.MODEL_CLOSURE_SYMLINK_REFUSED,
            role,
            path=relative,
        )
    if stat.S_ISDIR(metadata.st_mode):
        bound = _open_child_directory(directory, entry, metadata, role, relative)
        pending.append(
            _PendingDirectory(
                relative,
                bound.descriptor,
                (*directory.chain, bound),
                bound,
                directory.root_binding,
            )
        )
        return None
    if not stat.S_ISREG(metadata.st_mode):
        bad = OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID
        raise refuse(bad, role, path=relative, mode=metadata.st_mode)
    return _EnumeratedMember(
        relative,
        directory.root_binding.source_root.joinpath(*PurePosixPath(relative).parts),
        metadata.st_size,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        directory.root_binding,
        directory.descriptor,
        directory.chain,
    )


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    """Return the device and inode identity used for race detection."""
    return metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size


def _mutated(
    member: _EnumeratedMember, role: ModelRole, issue: str, **evidence: CanonicalValue
) -> ModelAdmissionRefusal:
    """Build a typed refusal for a source member changed during measurement."""
    reason = OperationalReasonCode.MODEL_CLOSURE_MUTATED
    return refuse(reason, role, path=member.relative_path, issue=issue, **evidence)


def _swapped_before_open(member: _EnumeratedMember, expected: tuple[int, int, int, int]) -> bool:
    """True when the enumerated member is no longer that exact regular file on disk.

    A FIFO swapped in between enumeration and open would block os.open until a writer appeared, so
    the no-follow identity is rechecked before any descriptor exists. An unstattable path is left to
    the bounded open, which refuses it: only a special file can block, and lstat catches that.
    """
    try:
        metadata = member.absolute_path.lstat()
    except OSError:
        return False
    return not stat.S_ISREG(metadata.st_mode) or _file_identity(metadata) != expected


def _verify_member_directories(member: _EnumeratedMember, role: ModelRole) -> None:
    """Verify every retained directory name leading to one member."""
    for directory in member._directory_chain:
        _verify_bound_directory(directory, role)


def _open_member_descriptor(
    member: _EnumeratedMember,
    role: ModelRole,
    expected: tuple[int, int, int, int],
) -> int:
    """Open one file relative to its retained parent without following links."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    parent = member._parent_descriptor
    if member._root_binding is None or parent is None:
        if _swapped_before_open(member, expected):
            raise _mutated(member, role, "member_changed_before_open")
        target: Path | str = member.absolute_path
        kwargs: dict[str, int] = {}
    else:
        _verify_member_directories(member, role)
        target = PurePosixPath(member.relative_path).name
        kwargs = {"dir_fd": parent}
        try:
            named = os.stat(target, dir_fd=parent, follow_symlinks=False)
        except OSError as exc:
            raise _mutated(
                member,
                role,
                "open_failed_after_enumeration",
                exception_type=type(exc).__name__,
            ) from exc
        if not stat.S_ISREG(named.st_mode) or _file_identity(named) != expected:
            raise _mutated(member, role, "member_changed_before_open")
    try:
        return os.open(target, flags, **kwargs)
    except OSError as exc:
        raise _mutated(
            member, role, "open_failed_after_enumeration", exception_type=type(exc).__name__
        ) from exc


def _read_member(member: _EnumeratedMember, role: ModelRole) -> bytes:
    """Load member from member and role for model closure, rejecting invalid input with _mutated."""
    expected = member.device, member.inode, member.mode, member.size_bytes
    descriptor = _open_member_descriptor(member, role, expected)
    try:
        before = os.fstat(descriptor)
        if _file_identity(before) != expected or not stat.S_ISREG(before.st_mode):
            raise _mutated(member, role, "member_changed_before_read")
        remaining = member.size_bytes + 1
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    except ModelAdmissionRefusal:
        raise
    except OSError as exc:
        raise _mutated(
            member, role, "member_read_failed_after_enumeration", exception_type=type(exc).__name__
        ) from exc
    finally:
        os.close(descriptor)
    content = b"".join(chunks)
    if _file_identity(before) != _file_identity(after) or not stat.S_ISREG(after.st_mode):
        raise _mutated(member, role, "member_changed_during_read")
    if len(content) != member.size_bytes:
        raise _mutated(member, role, "member_size_changed_during_read")
    if member._root_binding is not None:
        _verify_member_directories(member, role)
    return content


def _measure_bound(
    model_root: Path,
    entrypoint: str,
    role: ModelRole,
    max_bytes: int,
    binding: _RootBinding | None = None,
) -> tuple[ModelClosureIdentity, dict[str, bytes], _RootBinding]:
    """Measure through one root binding and return ownership to the caller."""
    source_binding = binding if binding is not None else _open_root_binding(model_root, role)
    owns_binding = binding is None
    try:
        _verify_public_root_binding(source_binding, role)
        normalized_entrypoint = _validate_entrypoint(entrypoint, role)
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        contents: dict[str, bytes] = {}
        members: list[ModelClosureMember] = []
        enumerated = _enumerate_members(model_root, role, max_bytes, source_binding)
        for member in enumerated:
            content = _read_member(member, role)
            contents[member.relative_path] = content
            members.append(
                ModelClosureMember(
                    member.relative_path,
                    len(content),
                    hashlib.sha256(content).hexdigest(),
                )
            )
        _verify_bound_tree(source_binding, role)
        if normalized_entrypoint not in contents:
            raise refuse(
                OperationalReasonCode.MODEL_ENTRYPOINT_INVALID,
                role,
                entrypoint=normalized_entrypoint,
                issue="entrypoint_missing_or_not_regular",
            )
        identity = ModelClosureIdentity(normalized_entrypoint, len(members), tuple(members))
        return identity, contents, source_binding
    except Exception:
        if owns_binding:
            source_binding.close()
        raise


def _measure(
    model_root: Path,
    entrypoint: str,
    role: ModelRole,
    max_bytes: int,
) -> tuple[ModelClosureIdentity, dict[str, bytes]]:
    """Measure and copy a complete immutable role-local model closure."""
    identity, contents, binding = _measure_bound(model_root, entrypoint, role, max_bytes)
    binding.close()
    return identity, contents


def measure_model_closure(
    model_root: Path,
    entrypoint: str,
    role: ModelRole,
    *,
    max_bytes: int = MAX_MODEL_CLOSURE_BYTES,
    _root_binding: _RootBinding | None = None,
) -> ModelClosureIdentity:
    """Measure every regular file beneath one model root without following symlinks."""
    if _root_binding is None:
        identity, _ = _measure(model_root, entrypoint, role, max_bytes)
        return identity
    identity, _, _ = _measure_bound(
        model_root,
        entrypoint,
        role,
        max_bytes,
        _root_binding,
    )
    return identity


def create_model_closure_snapshot(
    model_root: Path,
    entrypoint: str,
    role: ModelRole,
    *,
    max_bytes: int = MAX_MODEL_CLOSURE_BYTES,
) -> ModelClosureSnapshot:
    """Measure a closure and copy the exact measured bytes into an isolated tree."""
    identity, contents, binding = _measure_bound(model_root, entrypoint, role, max_bytes)
    try:
        temporary = tempfile.TemporaryDirectory(prefix=f"metrifid_{role}_model_")
    except Exception:
        binding.close()
        raise
    snapshot_root = Path(temporary.name)
    try:
        for relative, content in contents.items():
            destination = snapshot_root.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
    except Exception:
        try:
            temporary.cleanup()
        finally:
            binding.close()
        raise
    return ModelClosureSnapshot(
        model_root,
        identity.entrypoint,
        identity,
        snapshot_root,
        temporary,
        binding,
    )


def verify_model_closure_unchanged(
    snapshot: ModelClosureSnapshot,
    role: ModelRole,
    *,
    max_bytes: int = MAX_MODEL_CLOSURE_BYTES,
) -> None:
    """Re-measure a source closure and refuse any post-snapshot mutation."""
    _verify_public_root_binding(snapshot._source_binding, role)
    try:
        current = measure_model_closure(
            snapshot.source_root,
            snapshot.entrypoint,
            role,
            max_bytes=max_bytes,
            _root_binding=snapshot._source_binding,
        )
    except ModelAdmissionRefusal as exc:
        if exc.reason is OperationalReasonCode.MODEL_CLOSURE_MUTATED:
            raise
        raise refuse(
            OperationalReasonCode.MODEL_CLOSURE_MUTATED,
            role,
            prior_reason=exc.reason.value,
            prior_evidence=exc.to_primitive()["evidence"],
        ) from exc
    _verify_public_root_binding(snapshot._source_binding, role)
    if current != snapshot.identity:
        raise refuse(
            OperationalReasonCode.MODEL_CLOSURE_MUTATED,
            role,
            expected_closure_sha256=snapshot.identity.sha256(),
            actual_closure_sha256=current.sha256(),
        )
