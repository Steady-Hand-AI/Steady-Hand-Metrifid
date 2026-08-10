"""Verify installed wheel members and imported module roots."""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib.metadata as metadata
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from ._distribution_constants import _PACKAGE_NAME
from ._distribution_types import DistributionIdentityError, _ManifestRecord
from .json_values import CanonicalValue
from .operational import OperationalReasonCode


def _validate_loaded_modules(
    distribution: metadata.Distribution,
    records: Mapping[str, _ManifestRecord],
    expected_root: Path,
    install_root: Path,
    loaded_modules: Mapping[str, object],
) -> None:
    """Bind every loaded Metrifid module to its verified installed manifest member."""
    for name, candidate in loaded_modules.items():
        if name != _PACKAGE_NAME and not name.startswith(f"{_PACKAGE_NAME}."):
            continue
        module_file = getattr(candidate, "__file__", None)
        if module_file is None:
            continue
        _validate_loaded_module(
            distribution, records, expected_root, install_root, name, module_file
        )


def _validate_loaded_module(
    distribution: metadata.Distribution,
    records: Mapping[str, _ManifestRecord],
    expected_root: Path,
    install_root: Path,
    name: str,
    module_file: object,
) -> None:
    """Validate one loaded package module against its installed manifest member."""
    if type(module_file) is not str or not module_file:
        raise DistributionIdentityError(
            OperationalReasonCode.MIXED_METRIFID_MODULE_ROOTS,
            "a loaded metrifid module has an invalid __file__",
            field=name,
        )
    raw = Path(module_file)
    try:
        if raw.is_symlink():
            raise OSError("module file is a symlink")
        resolved = raw.resolve(strict=True)
        mode = resolved.lstat().st_mode
    except OSError as exc:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "a loaded metrifid module file is missing, unreadable, or symlinked",
            field=name,
        ) from exc
    if not stat.S_ISREG(mode):
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "a loaded metrifid module file is not regular",
            field=name,
        )
    try:
        relative = resolved.relative_to(expected_root)
    except ValueError as exc:
        raise DistributionIdentityError(
            OperationalReasonCode.MIXED_METRIFID_MODULE_ROOTS,
            "loaded metrifid modules resolve from multiple roots",
            field=name,
        ) from exc
    normalized = f"{_PACKAGE_NAME}/{relative.as_posix()}"
    record = records.get(normalized)
    if record is None:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "a loaded metrifid module is absent from the installed manifest",
            field=name,
            evidence={"module_path": normalized},
        )
    _verified_member_record(distribution, record, install_root)


def _selected_member_paths(
    distribution: metadata.Distribution,
    records: Mapping[str, _ManifestRecord],
    dist_info: str,
) -> set[str]:
    """Select package, metadata, entry-point, and license members that define identity."""
    selected = {
        path
        for path in records
        if path.startswith(f"{_PACKAGE_NAME}/")
        and "/__pycache__/" not in f"/{path}/"
        and not path.endswith((".pyc", ".pyo"))
    }
    selected.update({f"{dist_info}/METADATA", f"{dist_info}/WHEEL"})
    entry_points = f"{dist_info}/entry_points.txt"
    if entry_points in records:
        selected.add(entry_points)
    selected.update(_license_member_paths(distribution, records, dist_info))
    return selected


def _license_member_paths(
    distribution: metadata.Distribution,
    records: Mapping[str, _ManifestRecord],
    dist_info: str,
) -> set[str]:
    """Select all declared wheel license files and require at least one."""
    selected: set[str] = set()
    for license_name in distribution.metadata.get_all("License-File") or []:
        normalized_name = _normalize_distribution_path(license_name)
        candidates = {
            f"{dist_info}/{normalized_name}",
            f"{dist_info}/licenses/{normalized_name}",
        }
        matches = candidates.intersection(records)
        if not matches:
            suffix = f"/{normalized_name}"
            matches = {
                path
                for path in records
                if path.startswith(f"{dist_info}/") and path.endswith(suffix)
            }
        if not matches:
            raise DistributionIdentityError(
                OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
                f"recorded license file is missing: {license_name}",
                field="License-File",
            )
        selected.update(matches)
    return selected


def _verified_member_record(
    distribution: metadata.Distribution,
    record: _ManifestRecord,
    install_root: Path,
) -> dict[str, CanonicalValue]:
    """Read one confined regular member and verify its RECORD hash and size."""
    member_path = _resolved_regular_member(distribution, record, install_root)
    try:
        payload = member_path.read_bytes()
    except OSError as exc:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            f"distribution member is unreadable: {record.path}",
            field=record.path,
        ) from exc
    actual = _verify_record_payload(record, payload, "distribution member")
    return {"path": record.path, "size_bytes": len(payload), "sha256": actual.hex()}


def _verify_record_payload(
    record: _ManifestRecord,
    payload: bytes,
    label: str,
) -> bytes:
    """Match payload byte count and SHA-256 to one wheel RECORD row."""
    if record.size is None or record.size != len(payload):
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            f"{label} size differs from the installed manifest: {record.path}",
            field=record.path,
        )
    if record.hash_mode != "sha256" or record.hash_value is None:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            f"{label} lacks a SHA-256 manifest hash: {record.path}",
            field=record.path,
        )
    actual = hashlib.sha256(payload).digest()
    expected = _urlsafe_b64decode(record.hash_value)
    if actual != expected:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            f"{label} bytes differ from the installed manifest: {record.path}",
            field=record.path,
        )
    return actual


def _resolved_regular_member(
    distribution: metadata.Distribution,
    record: _ManifestRecord,
    install_root: Path,
) -> Path:
    """Resolve a manifest member beneath the install root without following a final symlink."""
    raw = Path(str(distribution.locate_file(metadata.PackagePath(record.path))))
    try:
        if raw.is_symlink():
            raise OSError("member is a symlink")
        resolved = raw.resolve(strict=True)
        mode = resolved.lstat().st_mode
        resolved.relative_to(install_root)
    except (OSError, ValueError) as exc:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            f"distribution member is missing, escaped, unreadable, or symlinked: {record.path}",
            field=record.path,
        ) from exc
    if not stat.S_ISREG(mode):
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            f"distribution member is not a regular file: {record.path}",
            field=record.path,
        )
    return resolved


def _normalize_distribution_path(raw: str) -> str:
    """Normalize and confine a wheel RECORD path in POSIX form."""
    if not raw or "\\" in raw:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "distribution path must be nonempty POSIX form",
            field="RECORD",
        )
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            f"invalid distribution path: {raw}",
            field="RECORD",
        )
    normalized = path.as_posix()
    if normalized != raw:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            f"distribution path is not normalized: {raw}",
            field="RECORD",
        )
    return normalized


def _urlsafe_b64decode(value: str) -> bytes:
    """Decode an unpadded URL-safe RECORD digest with strict validation."""
    try:
        return base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError, TypeError) as exc:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "distribution RECORD contains an invalid SHA-256 encoding",
            field="RECORD",
        ) from exc


def _path_digest(path: Path) -> str:
    """Hash exact bytes from one verified installed member path."""
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()
