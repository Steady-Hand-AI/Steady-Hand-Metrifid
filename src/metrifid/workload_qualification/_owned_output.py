"""Own the qualification output root before anything is written into it.

Three defects motivate this module. Evidence directories were created from semantic labels, so an
absolute label wrote outside the intended root. A configuration whose output overlapped a model root
created staging directories inside that model root before the nested comparison refused, which
mutates the very tree the campaign is supposed to observe. And ownership was decided against one
path spelling but taken against another, so an intermediate component could be retargeted between
the decision and the creation.

Ownership is therefore decided once and carried. The proposed root is admitted in canonical form,
compared against every canonically resolved model root, and required not to exist; that same
canonical result is what creation binds. Binding never opens a whole path at once: every absolute
component is opened from the filesystem anchor through a retained descriptor with
``O_DIRECTORY | O_NOFOLLOW``, the final root is created exclusively relative to its bound parent,
and the retained descriptor's device and inode become this root's identity.

Binding a component is not opening it. A named component is inspected with a no-follow ``stat``,
required to be a real directory, opened ``O_DIRECTORY | O_NOFOLLOW`` relative to its retained parent,
and only accepted when the named stat and the opened ``fstat`` agree on device and inode. A directory
replaced between those two observations is refused rather than owned. The same agreement is required
of the final root this owner creates, and every child it creates keeps its recorded identity, so a
directory another process inserts under the owned root never becomes owned merely by being a real
directory.

Nothing outside this module is ever handed a pathname as authority. An external subsystem that
publishes into an owned directory receives the retained descriptor itself, so the write cannot be
redirected between a check and its use. A public path remains available as display metadata only.
Reads are bounded by an explicit maximum the caller declares.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path, PurePosixPath
from typing import Final

_DIR_FLAGS: Final[int] = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_CREATE_FLAGS: Final[int] = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
# A retained member is opened without blocking, so a FIFO or device substituted for a regular
# file is refused at the open rather than waiting there for a writer that may never arrive.
_FILE_READ_FLAGS: Final[int] = (
    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
)
_MODE: Final[int] = 0o700
_READ_CHUNK: Final[int] = 1024 * 1024


class OwnedOutputError(ValueError):
    """Raised when the proposed output root cannot be owned safely."""


def canonical_root(path: Path) -> Path:
    """Resolve one existing directory canonically, following intermediate links deliberately.

    Model roots are inputs the user already controls; resolving them fully is what makes an overlap
    test meaningful, because ``baseline`` and a symlink pointing at it must compare equal.
    """
    return path.resolve(strict=True)


def _overlaps(output: Path, root: Path) -> bool:
    """Return whether the proposed output is the model root or lies beneath it."""
    return output == root or root in output.parents


def preflight(proposed_output: Path, model_roots: dict[str, Path]) -> Path:
    """Check ownership of the proposed output root and return the identity creation must use.

    The output's own final component may not exist yet, so its nearest existing ancestor is
    canonicalized and the remaining components are appended. That gives one canonical form for the
    output that can be compared against every canonically resolved model root, which catches the
    symlink-alias spelling of the same overlap as well as the literal one. That canonical form is
    returned because it is the only spelling the caller may create: creating the original spelling
    instead would let an intermediate component be retargeted between this decision and that write.
    """
    absolute = proposed_output.absolute()
    existing = absolute
    trailing: list[str] = []
    while not existing.exists():
        if existing.parent == existing:
            raise OwnedOutputError("the proposed output has no existing ancestor")
        trailing.append(existing.name)
        existing = existing.parent
    canonical_output = existing.resolve(strict=True).joinpath(*reversed(trailing))

    for label, root in model_roots.items():
        try:
            canonical = canonical_root(root)
        except (OSError, RuntimeError) as exc:
            raise OwnedOutputError(f"model root {label} could not be resolved: {exc}") from exc
        if _overlaps(canonical_output, canonical):
            raise OwnedOutputError(
                f"the qualification output root would be inside the {label} model root; "
                "an output that lives under an observed model would modify the tree the "
                "campaign is measuring"
            )
    if absolute.exists() or canonical_output.exists():
        raise OwnedOutputError(
            "the qualification output root already exists; published evidence is never overwritten"
        )
    return canonical_output


def _object_identity(info: os.stat_result) -> tuple[int, int]:
    """Return the exact filesystem object a stat observation names."""
    return (info.st_dev, info.st_ino)


def _read_observation(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """Return the exact file observation a bounded read must find unchanged on both sides.

    A read is only bounded if the object it started on is the object it finished on. Device and
    inode catch replacement, the file type catches substitution, and size with both timestamps
    catches truncation, growth, and a same-size rewrite performed while the read was in progress.
    """
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _bind_named_directory(parent_fd: int, name: str, field: str) -> tuple[int, tuple[int, int]]:
    """Bind one named child directory so the object named and the object opened are the same.

    A no-follow ``stat`` names the object, the ``O_DIRECTORY | O_NOFOLLOW`` open takes it, and the
    opened ``fstat`` says which object was actually taken. Requiring those two observations to agree
    is what refuses a directory replaced in between; opening alone cannot detect that.
    """
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise OwnedOutputError(
            f"{field} could not be inspected without following a link: {exc}"
        ) from exc
    if not stat.S_ISDIR(named.st_mode):
        raise OwnedOutputError(f"{field} is not a real directory")
    try:
        descriptor = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise OwnedOutputError(
            f"{field} could not be opened without following a link: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode):  # pragma: no cover - guarded by O_DIRECTORY
            raise OwnedOutputError(f"{field} is not a directory")
        if _object_identity(named) != _object_identity(opened):
            raise OwnedOutputError(f"{field} was replaced between being named and being opened")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, _object_identity(opened)


def _open_absolute_directory(path: Path, field: str) -> tuple[int, tuple[int, int]]:
    """Bind one absolute directory from its anchor, one identity-checked component at a time.

    Opening a whole path at once leaves ``O_NOFOLLOW`` protecting only the final component, so an
    intermediate link decides where the descriptor lands.
    """
    absolute = path.absolute()
    descriptor = os.open(absolute.anchor, _DIR_FLAGS)
    try:
        identity = _object_identity(os.fstat(descriptor))
        for component in absolute.parts[1:]:
            child, identity = _bind_named_directory(descriptor, component, f"{field} component")
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, identity


class OwnedOutputRoot:
    """One qualification output root, created and used only through retained descriptors."""

    __slots__ = ("_adopt_unknown", "_children", "_closed", "_root_fd", "path")

    def __init__(self, path: Path) -> None:
        """Create the root under its component-bound parent and retain a descriptor on it."""
        absolute = path.absolute()
        if absolute.parent == absolute:
            raise OwnedOutputError("the owned output root must have a parent directory")
        parent_fd, _ = _open_absolute_directory(absolute.parent, "output root parent")
        try:
            try:
                os.mkdir(absolute.name, mode=_MODE, dir_fd=parent_fd)
            except FileExistsError as exc:
                raise OwnedOutputError(
                    "the qualification output root already exists; published evidence is never "
                    "overwritten"
                ) from exc
            root_fd, _ = _bind_named_directory(parent_fd, absolute.name, "the created output root")
        finally:
            os.close(parent_fd)
        self._bind(root_fd, absolute, adopt_unknown=False)

    @classmethod
    def bind_existing(cls, path: Path) -> OwnedOutputRoot:
        """Bind an already created owned root through a component-wise no-follow walk.

        This is the authority public receipt replay binds through, so it never opens a whole path at
        once and never creates anything: an intermediate linked directory refuses here rather than
        redirecting a later read.
        """
        absolute = path.absolute()
        instance = cls.__new__(cls)
        root_fd, _ = _open_absolute_directory(absolute, "owned output root")
        instance._bind(root_fd, absolute, adopt_unknown=True)
        return instance

    def _bind(self, root_fd: int, absolute: Path, *, adopt_unknown: bool) -> None:
        """Retain one open root descriptor, its public path, and its child identity register.

        The root's own identity was already proved where the descriptor was taken, by requiring the
        object named and the object opened to agree; nothing afterwards re-derives it, so it is not
        retained. What is retained is the descriptor every operation goes through and the identity
        of each child this owner binds.

        ``adopt_unknown`` separates the two lifecycles. A creating owner made every directory
        beneath its root, so anything it has not recorded is foreign and is refused. A read-only
        replay binding has recorded nothing, so it records each child the first time it descends and
        requires the same object on every later descent within that one replay.
        """
        self._root_fd = root_fd
        self.path = absolute
        self._children: dict[str, tuple[int, int]] = {}
        self._adopt_unknown = adopt_unknown
        self._closed = False

    def close(self) -> None:
        """Release the retained root descriptor exactly once."""
        if self._closed:
            return
        self._closed = True
        try:
            os.close(self._root_fd)
        except OSError:  # pragma: no cover - defensive
            pass

    def _bind_child(self, parent_fd: int, name: str, key: str, *, create: bool) -> int:
        """Bind one child directory and require it to be the object this owner already knows.

        Ownership of a child comes from having created it, not from the entry path used to reach
        it. A creating owner therefore refuses an unrecorded child whether it is met while creating
        the chain or while opening it later; only a read-only replay binding may adopt one.
        """
        created = False
        if create:
            try:
                os.mkdir(name, mode=_MODE, dir_fd=parent_fd)
                created = True
            except FileExistsError:
                created = False
        if not created and key not in self._children and not self._adopt_unknown:
            raise OwnedOutputError(
                f"{key} exists inside the owned root and was not created by this run, so it is "
                "not this run's object to own"
            )
        descriptor, identity = _bind_named_directory(parent_fd, name, key)
        recorded = self._children.get(key)
        if recorded is None:
            self._children[key] = identity
            return descriptor
        if recorded != identity:
            os.close(descriptor)
            raise OwnedOutputError(f"{key} is no longer the directory this run bound")
        return descriptor

    def _walk(self, locator: PurePosixPath, *, create: bool) -> int:
        """Bind the directory a locator names, descending one identity-checked component at a time.

        A component that is a link, that is not a directory, that was replaced between being named
        and being opened, or that is not the object this owner already recorded, refuses here. That
        is what makes a recorded locator safe to reuse: it is resolved against this owner's retained
        objects rather than against whatever the pathname happens to traverse at use time.
        """
        current = os.dup(self._root_fd)
        prefix = PurePosixPath()
        try:
            for part in locator.parts:
                prefix = prefix / part
                nxt = self._bind_child(current, part, prefix.as_posix(), create=create)
                os.close(current)
                current = nxt
            return current
        except OSError as exc:
            os.close(current)
            raise OwnedOutputError(
                f"{locator} could not be reached inside the owned root without following a link: "
                f"{exc}"
            ) from exc
        except BaseException:
            os.close(current)
            raise

    def display_path(self, locator: PurePosixPath) -> Path:
        """Return the public pathname of one owned locator, as display metadata only.

        Nothing writes through this value. It is what a receipt records and what a report prints;
        the objects themselves are always reached through :meth:`open_owned_directory`.
        """
        return self.path / locator

    def make_directory(self, locator: PurePosixPath) -> None:
        """Create one directory tree under the owned root, recording each object it creates."""
        os.close(self._walk(locator, create=True))

    def open_owned_directory(self, locator: PurePosixPath, *, create: bool = False) -> int:
        """Return a retained descriptor on one owned directory, for a descriptor-bound write.

        The caller owns the returned descriptor and must close it. Handing the descriptor rather
        than a pathname is what keeps an external publisher writing into this exact object: there is
        no second traversal for another process to redirect.
        """
        return self._walk(locator, create=create)

    def write_bytes(self, locator: PurePosixPath, payload: bytes) -> None:
        """Write one new file under the owned root, refusing to clobber or follow a link."""
        parent = (
            self._walk(locator.parent, create=True)
            if locator.parent.parts
            else os.dup(self._root_fd)
        )
        try:
            descriptor = os.open(locator.name, _FILE_CREATE_FLAGS, 0o600, dir_fd=parent)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            os.close(parent)

    def read_bytes(self, locator: PurePosixPath, maximum: int) -> bytes:
        """Read one bounded regular file under the owned root through retained descriptors.

        The caller declares the maximum, because the caller is the one that knows which admission
        profile the member belongs to. The open does not block, so a FIFO or device substituted for
        a regular file is refused rather than waited on. The declared size is checked before
        anything is allocated, at most one byte beyond the maximum is ever read, and the file is
        observed again afterwards: a member replaced, truncated, grown, or rewritten at the same
        size while the read was in progress is refused rather than returned.
        """
        if type(maximum) is not int or maximum <= 0:
            raise OwnedOutputError("a retained read requires an explicit positive maximum")
        parent = (
            self._walk(locator.parent, create=False)
            if locator.parent.parts
            else os.dup(self._root_fd)
        )
        try:
            descriptor = os.open(locator.name, _FILE_READ_FLAGS, dir_fd=parent)
        except OSError as exc:
            raise OwnedOutputError(f"{locator} is not a readable regular file: {exc}") from exc
        finally:
            os.close(parent)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise OwnedOutputError(f"{locator} is not a regular file")
            if info.st_size > maximum:
                raise OwnedOutputError(
                    f"{locator} declares {info.st_size} bytes above its {maximum} byte bound"
                )
            chunks: list[bytes] = []
            remaining = maximum + 1
            while remaining > 0:
                try:
                    block = os.read(descriptor, min(remaining, _READ_CHUNK))
                except BlockingIOError as exc:
                    raise OwnedOutputError(f"{locator} would block while being read") from exc
                if not block:
                    break
                chunks.append(block)
                remaining -= len(block)
            payload = b"".join(chunks)
            if len(payload) > maximum:
                raise OwnedOutputError(f"{locator} exceeds its {maximum} byte bound")
            if _read_observation(os.fstat(descriptor)) != _read_observation(info):
                raise OwnedOutputError(f"{locator} changed while it was being read")
            if len(payload) != info.st_size:
                raise OwnedOutputError(
                    f"{locator} returned {len(payload)} bytes for an admitted size of "
                    f"{info.st_size}"
                )
            return payload
        finally:
            os.close(descriptor)

    def __enter__(self) -> OwnedOutputRoot:
        """Enter the owned-root context."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release the retained descriptor on exit."""
        self.close()
