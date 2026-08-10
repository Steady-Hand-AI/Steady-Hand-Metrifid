"""Expand composite MJCF dependencies inside an immutable snapshot."""

from __future__ import annotations

import os
import posixpath
from pathlib import Path
from typing import cast
from xml.etree import ElementTree

from ._dependency_reader import (
    _ASSET_DIRECTORY_ATTRIBUTE,
    _COMPOSITE_STAGE,
    _AssetResolutionError,
    _compile_error_reason,
    _include_relative,
    _included_root,
    _member_bytes,
    _member_map,
    _raw_dependencies,
    _read_dependency_member,
    first_complete_root_element,
)
from ._model_closure import ModelClosureSnapshot, ModelRole, refuse
from .json_values import CanonicalValue
from .operational import OperationalReasonCode
from .schemas import ModelClosureMember


class _CompositeModel:
    """The include-expanded MJCF document of one measured snapshot.

    Only ``include`` elements reachable from the measured main entrypoint are followed. Each
    included file contributes its top-level element's children at the location of the include, in
    source order, and may appear at most once in the whole composite model.
    """

    def __init__(
        self,
        snapshot: ModelClosureSnapshot,
        role: ModelRole,
        members: dict[str, ModelClosureMember],
    ) -> None:
        """Start include expansion for one role's measured snapshot and member index."""
        self._snapshot = snapshot
        self._role = role
        self._members = members
        self._seen: set[str] = {snapshot.entrypoint}
        self._stack: list[str] = [snapshot.entrypoint]
        self.meshdir = ""
        self.texturedir = ""
        self.strippath = False
        self.assets: list[tuple[str, str]] = []

    def expand(self) -> None:
        """Parse the measured entrypoint and traverse its complete reachable MJCF tree."""
        data = _member_bytes(
            self._snapshot,
            self._role,
            self._snapshot.entrypoint,
            self._members,
            _COMPOSITE_STAGE,
        )
        root = first_complete_root_element(data)
        if root is None:
            raise refuse(
                _compile_error_reason(self._role),
                self._role,
                stage=_COMPOSITE_STAGE,
                snapshot_entrypoint=self._snapshot.entrypoint,
                issue="no_complete_top_level_element",
            )
        self._walk(root)

    def _walk(self, element: ElementTree.Element) -> None:
        """Collect compiler settings and asset tokens while expanding nested includes."""
        for child in element:
            if child.tag == "include":
                self._expand_include(child.get("file"))
                continue
            if child.tag == "compiler":
                self._apply_compiler(child)
                continue
            if child.tag in _ASSET_DIRECTORY_ATTRIBUTE:
                declared = child.get("file")
                if declared:
                    self.assets.append((child.tag, declared))
            self._walk(child)

    def _expand_include(self, token: str | None) -> None:
        """Expand one measured include while rejecting cycles and duplicate inclusion."""
        relative = _include_relative(self._snapshot, self._role, token)
        if relative in self._stack:
            self._refuse_include(relative, "include_cycle")
        if relative in self._seen:
            self._refuse_include(relative, "duplicate_include")
        self._seen.add(relative)
        self._stack.append(relative)
        self._walk(_included_root(self._snapshot, self._role, relative, self._members))
        self._stack.pop()

    def _refuse_include(self, relative: str, issue: str) -> None:
        """Raise the closure-member refusal for a cyclic or duplicate include."""
        raise refuse(
            OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID,
            self._role,
            stage=_COMPOSITE_STAGE,
            snapshot_entrypoint=self._snapshot.entrypoint,
            included_file=relative,
            issue=issue,
        )

    def _apply_compiler(self, element: ElementTree.Element) -> None:
        """Apply one compiler element; the last explicit setting wins."""
        assetdir = element.get("assetdir")
        if assetdir is not None:
            self.meshdir = assetdir
            self.texturedir = assetdir
        meshdir = element.get("meshdir")
        if meshdir is not None:
            self.meshdir = meshdir
        texturedir = element.get("texturedir")
        if texturedir is not None:
            self.texturedir = texturedir
        strippath = element.get("strippath")
        if strippath is not None:
            self.strippath = strippath.strip().lower() == "true"

    def directory(self, kind: str) -> str:
        """Return the effective compiler directory for a mesh or texture asset kind."""
        if _ASSET_DIRECTORY_ATTRIBUTE[kind] == "meshdir":
            return self.meshdir
        return self.texturedir


def _compiler_relative_join(base: Path, directory: str, token: str, strippath: bool) -> Path:
    """Join one asset token the way the MuJoCo 3.10.0 compiler does."""
    name = posixpath.basename(token.replace("\\", "/")) if strippath else token
    if not name:
        raise _AssetResolutionError("empty asset file token")
    name_path = Path(name)
    if name_path.is_absolute():
        return name_path
    directory_path = Path(directory) if directory else Path()
    if directory_path.is_absolute():
        return directory_path / name_path
    return base / directory_path / name_path


def _contained_member(
    snapshot: ModelClosureSnapshot,
    members: dict[str, ModelClosureMember],
    candidate: Path,
) -> tuple[Path, str] | None:
    """Return the contained measured member a path names, or None."""
    try:
        root = snapshot.snapshot_root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
    except (OSError, RuntimeError, ValueError):
        return None
    if relative not in members:
        return None
    return candidate, relative


def _compiler_directory_binding(
    snapshot: ModelClosureSnapshot,
    role: ModelRole,
    members: dict[str, ModelClosureMember],
    reported: Path,
) -> tuple[Path, str] | None:
    """Bind one reported asset dependency through effective compiler-directory semantics.

    A declared asset of the actual composite document justifies the reported dependency only when
    the main-MJCF-relative resolution of its own file token is exactly the reported path. The
    compiler-directory resolution of that same token must then name exactly one contained measured
    member. Zero or several justified members return None so the caller raises the existing refusal.
    """
    composite = _CompositeModel(snapshot, role, members)
    composite.expand()
    base = snapshot.snapshot_entrypoint.parent
    justified: dict[str, Path] = {}
    for kind, token in composite.assets:
        try:
            reporter_path = _compiler_relative_join(base, "", token, composite.strippath)
        except _AssetResolutionError:
            continue
        if os.path.normpath(str(reporter_path)) != os.path.normpath(str(reported)):
            continue
        try:
            compiler_path = _compiler_relative_join(
                base, composite.directory(kind), token, composite.strippath
            )
        except _AssetResolutionError:
            continue
        found = _contained_member(snapshot, members, compiler_path)
        if found is not None:
            justified[found[1]] = found[0]
    if len(justified) != 1:
        return None
    relative, candidate = next(iter(justified.items()))
    return candidate, relative


def _candidate_path(snapshot: ModelClosureSnapshot, dependency: str) -> Path:
    """Resolve a reported dependency relative to the snapshot entrypoint when needed."""
    if not dependency or "\x00" in dependency:
        raise ValueError("dependency path must be a nonempty filesystem string")
    path = Path(dependency)
    if path.is_absolute():
        return path
    return snapshot.snapshot_entrypoint.parent / path


def _resolve_dependency(
    snapshot: ModelClosureSnapshot,
    role: ModelRole,
    dependency: str,
    members: dict[str, ModelClosureMember],
) -> str:
    """Locate, verify, and return the measured relative path for one dependency."""
    candidate, resolved, relative, member = _located_dependency(snapshot, role, dependency, members)
    if member is None:
        _refuse_missing_dependency(snapshot, role, dependency, resolved, relative)
    _read_dependency_member(
        candidate, cast(ModelClosureMember, member), role, snapshot, dependency, cast(str, relative)
    )
    return cast(str, relative)


def _located_dependency(
    snapshot: ModelClosureSnapshot,
    role: ModelRole,
    dependency: str,
    members: dict[str, ModelClosureMember],
) -> tuple[Path, Path, str | None, ModelClosureMember | None]:
    """Resolve a reported dependency, including the compiler-directory fallback."""
    candidate = _candidate_path(snapshot, dependency)
    try:
        root = snapshot.snapshot_root.resolve(strict=True)
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise refuse(
            OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID,
            role,
            stage="dependency_discovery",
            snapshot_entrypoint=snapshot.entrypoint,
            offending_dependency=dependency,
            exception_type=type(exc).__name__,
        ) from exc
    try:
        relative: str | None = resolved.relative_to(root).as_posix()
    except ValueError:
        relative = None
    member = members.get(relative) if relative is not None else None
    if member is None:
        # MuJoCo reports some contained mesh, height-field and texture assets resolved against the
        # main MJCF directory rather than through the effective compiler asset directory. Re-derive
        # the path the compiler actually opens from the actual composite document, binding only to
        # exactly one justified member.
        try:
            binding = _compiler_directory_binding(snapshot, role, members, candidate)
        except _AssetResolutionError:
            binding = None
        if binding is not None:
            candidate, relative = binding
            member = members[relative]
    return candidate, resolved, relative, member


def _refuse_missing_dependency(
    snapshot: ModelClosureSnapshot,
    role: ModelRole,
    dependency: str,
    resolved: Path,
    relative: str | None,
) -> None:
    """Raise the exact escape or unmeasured-member refusal for a missing dependency."""
    if relative is None:
        raise refuse(
            OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE,
            role,
            stage="dependency_discovery",
            snapshot_entrypoint=snapshot.entrypoint,
            offending_dependency=dependency,
            resolved_dependency=str(resolved),
        )
    raise refuse(
        OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID,
        role,
        stage="dependency_discovery",
        snapshot_entrypoint=snapshot.entrypoint,
        offending_dependency=dependency,
        relative_dependency=relative,
        issue="dependency_not_in_measured_closure",
    )


def discover_snapshot_dependencies(
    snapshot: ModelClosureSnapshot,
    role: ModelRole,
) -> tuple[str, ...]:
    """Return the deterministic contained compilation dependency set."""
    if role not in {"baseline", "candidate"}:
        raise ValueError("dependency discovery requires a baseline or candidate role")
    members = _member_map(snapshot)
    dependencies = {
        _resolve_dependency(snapshot, role, dependency, members)
        for dependency in _raw_dependencies(snapshot, role)
    }
    if snapshot.entrypoint not in dependencies:
        raise refuse(
            OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID,
            role,
            stage="dependency_discovery",
            snapshot_entrypoint=snapshot.entrypoint,
            issue="entrypoint_missing_from_dependency_set",
            dependencies=cast(CanonicalValue, sorted(dependencies)),
        )
    return tuple(sorted(dependencies))
