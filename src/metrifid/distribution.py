"""Bind the executing :mod:`metrifid` code to one installed wheel."""

from __future__ import annotations

import importlib.metadata as metadata
import sys
import sysconfig as sysconfig
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

from ._distribution_constants import (
    _CONSOLE_SCRIPT_NAME as _CONSOLE_SCRIPT_NAME,
)
from ._distribution_constants import (
    _CONSOLE_SCRIPT_VALUE as _CONSOLE_SCRIPT_VALUE,
)
from ._distribution_constants import (
    _DISTRIBUTION_NAME,
    _PACKAGE_NAME,
)
from ._distribution_manifest import (
    _dist_info_directory as _dist_info_directory,
)
from ._distribution_manifest import (
    _distribution_install_root as _distribution_install_root,
)
from ._distribution_manifest import (
    _expected_console_launcher as _expected_console_launcher,
)
from ._distribution_manifest import (
    _expected_package_root as _expected_package_root,
)
from ._distribution_manifest import (
    _normalized_manifest as _normalized_manifest,
)
from ._distribution_manifest import (
    _reject_editable as _reject_editable,
)
from ._distribution_manifest import (
    _verified_external_launcher as _verified_external_launcher,
)
from ._distribution_manifest import (
    _verify_external_console_launcher as _verify_external_console_launcher,
)
from ._distribution_members import (
    _license_member_paths as _license_member_paths,
)
from ._distribution_members import (
    _normalize_distribution_path as _normalize_distribution_path,
)
from ._distribution_members import (
    _path_digest,
)
from ._distribution_members import (
    _resolved_regular_member as _resolved_regular_member,
)
from ._distribution_members import (
    _selected_member_paths as _selected_member_paths,
)
from ._distribution_members import (
    _urlsafe_b64decode as _urlsafe_b64decode,
)
from ._distribution_members import (
    _validate_loaded_modules as _validate_loaded_modules,
)
from ._distribution_members import (
    _verified_member_record as _verified_member_record,
)
from ._distribution_members import (
    _verify_record_payload as _verify_record_payload,
)
from ._distribution_types import (
    DistributionIdentityError as DistributionIdentityError,
)
from ._distribution_types import (
    _InstalledManifest as _InstalledManifest,
)
from ._distribution_types import (
    _ManifestRecord as _ManifestRecord,
)
from .json_values import CanonicalValue, canonical_sha256
from .operational import OperationalReasonCode
from .version import __version__


def installed_distribution_identity() -> dict[str, CanonicalValue]:
    """Return the identity of the installed wheel whose code is executing."""
    imported_root = _imported_package_root()
    distribution = _single_distribution()
    return _bound_distribution_identity(distribution, imported_root, sys.modules)


def installed_distribution_sha256() -> str:
    """Return SHA-256 over the bound installed distribution identity."""
    return canonical_sha256(installed_distribution_identity())


def _single_distribution() -> metadata.Distribution:
    """Require exactly one installed distribution matching the Metrifid package name."""
    try:
        candidates = list(metadata.distributions(name=_DISTRIBUTION_NAME))
    except Exception as exc:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "installed metrifid distribution metadata cannot be enumerated",
            field="distribution",
        ) from exc
    if len(candidates) != 1:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "exactly one installed metrifid distribution is required",
            field="distribution",
            evidence={"distribution_count": len(candidates)},
        )
    return candidates[0]


def _imported_package_root() -> Path:
    """Resolve the executing package's real import root without accepting a symlink."""
    module = sys.modules.get(_PACKAGE_NAME)
    if not isinstance(module, ModuleType):
        raise DistributionIdentityError(
            OperationalReasonCode.EXECUTING_CODE_NOT_INSTALLED_DISTRIBUTION,
            "metrifid is not an imported package module",
            field="metrifid",
        )
    spec = module.__spec__
    locations = None if spec is None else spec.submodule_search_locations
    if locations is None:
        raise DistributionIdentityError(
            OperationalReasonCode.EXECUTING_CODE_NOT_INSTALLED_DISTRIBUTION,
            "metrifid has no package search location",
            field="metrifid.__spec__",
        )
    roots = tuple(str(item) for item in locations)
    if len(roots) != 1:
        raise DistributionIdentityError(
            OperationalReasonCode.MIXED_METRIFID_MODULE_ROOTS,
            "metrifid must resolve to exactly one imported package root",
            field="metrifid.__path__",
            evidence={"root_count": len(roots)},
        )
    raw_root = Path(roots[0])
    try:
        if raw_root.is_symlink():
            raise OSError("package root is a symlink")
        root = raw_root.resolve(strict=True)
    except OSError as exc:
        raise DistributionIdentityError(
            OperationalReasonCode.EXECUTING_CODE_NOT_INSTALLED_DISTRIBUTION,
            "the imported metrifid package root is unavailable or symlinked",
            field="metrifid.__path__",
        ) from exc
    if not root.is_dir():
        raise DistributionIdentityError(
            OperationalReasonCode.EXECUTING_CODE_NOT_INSTALLED_DISTRIBUTION,
            "the imported metrifid package root is not a directory",
            field="metrifid.__path__",
        )
    return root


def _bound_distribution_identity(
    distribution: metadata.Distribution,
    imported_root: Path,
    loaded_modules: Mapping[str, object],
) -> dict[str, CanonicalValue]:
    """Verify manifest members and hash the installed distribution identity surface."""
    manifest = _normalized_manifest(distribution)
    records = manifest.records
    dist_info = _dist_info_directory(records)
    install_root = _distribution_install_root(distribution, records, dist_info)
    _reject_editable(distribution, records, dist_info, install_root)
    _verify_external_console_launcher(
        distribution,
        records,
        manifest.external_records,
        dist_info,
        install_root,
    )
    expected_root = _expected_package_root(distribution, records)
    _validate_distribution_roots(imported_root, expected_root, install_root)
    _validate_loaded_modules(distribution, records, expected_root, install_root, loaded_modules)
    selected = _selected_member_paths(distribution, records, dist_info)
    members: list[CanonicalValue] = [
        _verified_member_record(distribution, records[path], install_root)
        for path in sorted(selected)
    ]
    version = distribution.version
    if version != __version__:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "installed metadata version does not match metrifid.__version__",
            field="distribution_version",
        )
    return {
        "schema": "metrifid.installed_distribution_identity",
        "schema_version": 1,
        "distribution_name": _DISTRIBUTION_NAME,
        "distribution_version": version,
        "members": members,
    }


def _validate_distribution_roots(
    imported_root: Path, expected_root: Path, install_root: Path
) -> None:
    """Require imported code, wheel members, and metadata to share one installation root."""
    if expected_root.parent != install_root:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "installed package and dist-info roots do not share one installation root",
            field="distribution",
        )
    if imported_root != expected_root:
        raise DistributionIdentityError(
            OperationalReasonCode.EXECUTING_CODE_NOT_INSTALLED_DISTRIBUTION,
            "the imported metrifid root does not match the installed wheel root",
            field="metrifid.__path__",
            evidence={
                "imported_root_sha256": _path_digest(imported_root),
                "installed_root_sha256": _path_digest(expected_root),
            },
        )
