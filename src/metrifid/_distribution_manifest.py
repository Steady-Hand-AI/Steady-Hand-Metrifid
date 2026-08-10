"""Parse wheel metadata and verify console-script installation paths."""

from __future__ import annotations

import csv
import importlib.metadata as metadata
import json
import os
import stat
import sysconfig
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from ._distribution_constants import _CONSOLE_SCRIPT_NAME, _CONSOLE_SCRIPT_VALUE, _PACKAGE_NAME
from ._distribution_members import (
    _normalize_distribution_path,
    _resolved_regular_member,
    _verified_member_record,
    _verify_record_payload,
)
from ._distribution_types import DistributionIdentityError, _InstalledManifest, _ManifestRecord
from .operational import OperationalReasonCode


def _normalized_manifest(distribution: metadata.Distribution) -> _InstalledManifest:
    """Parse wheel RECORD into unique normalized in-root and external rows."""
    try:
        text = distribution.read_text("RECORD")
    except Exception as exc:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "installed distribution RECORD cannot be read",
            field="RECORD",
        ) from exc
    if text is None:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "installed distribution does not expose a wheel RECORD manifest",
            field="RECORD",
        )
    result: dict[str, _ManifestRecord] = {}
    external_records: list[_ManifestRecord] = []
    try:
        rows = csv.reader(text.splitlines())
        for row in rows:
            record = _manifest_record(row)
            try:
                normalized = _normalize_distribution_path(record.path)
            except DistributionIdentityError:
                external_records.append(record)
                continue
            if normalized in result:
                raise ValueError(f"duplicate normalized distribution path: {normalized}")
            result[normalized] = _ManifestRecord(
                normalized,
                record.hash_mode,
                record.hash_value,
                record.size,
            )
    except (csv.Error, UnicodeError, ValueError) as exc:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "installed distribution RECORD is malformed",
            field="RECORD",
        ) from exc
    return _InstalledManifest(result, tuple(external_records))


def _manifest_record(row: list[str]) -> _ManifestRecord:
    """Parse one three-column wheel RECORD row without normalizing its path."""
    if len(row) != 3:
        raise ValueError("RECORD rows must contain exactly three fields")
    raw_path, raw_hash, raw_size = row
    hash_mode: str | None = None
    hash_value: str | None = None
    if raw_hash:
        parts = raw_hash.split("=", 1)
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"invalid RECORD hash field: {raw_path}")
        hash_mode, hash_value = parts
    size: int | None = None
    if raw_size:
        size = int(raw_size)
        if size < 0:
            raise ValueError(f"negative RECORD size: {raw_path}")
    return _ManifestRecord(raw_path, hash_mode, hash_value, size)


def _verify_external_console_launcher(
    distribution: metadata.Distribution,
    records: Mapping[str, _ManifestRecord],
    external_records: tuple[_ManifestRecord, ...],
    dist_info: str,
    install_root: Path,
) -> None:
    """Verify that the sole permitted out-of-root RECORD row is the declared launcher."""
    entry_points_path = f"{dist_info}/entry_points.txt"
    entry_points_record = records.get(entry_points_path)
    has_exact_binding = _has_exact_console_binding(
        distribution, entry_points_record, install_root, entry_points_path
    )

    if not has_exact_binding:
        if external_records:
            raise DistributionIdentityError(
                OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
                "installed distribution RECORD contains an unauthorized external path",
                field="RECORD",
            )
        return

    launcher_path, expected_record_path = _expected_console_launcher(install_root)
    record = _external_launcher_record(external_records, expected_record_path)
    _verified_external_launcher(record, launcher_path)


def _has_exact_console_binding(
    distribution: metadata.Distribution,
    record: _ManifestRecord | None,
    install_root: Path,
    entry_points_path: str,
) -> bool:
    """Return whether installed metadata declares only the supported console binding."""
    if record is None:
        return False
    _verified_member_record(distribution, record, install_root)
    try:
        bindings = tuple(
            (entry_point.name, entry_point.value)
            for entry_point in distribution.entry_points
            if entry_point.group == "console_scripts"
        )
    except Exception as exc:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "installed distribution entry_points.txt is malformed",
            field=entry_points_path,
        ) from exc
    return bindings == ((_CONSOLE_SCRIPT_NAME, _CONSOLE_SCRIPT_VALUE),)


def _external_launcher_record(
    records: tuple[_ManifestRecord, ...], expected_path: str
) -> _ManifestRecord:
    """Select the sole authorized external launcher manifest row."""
    matching = tuple(record for record in records if record.path == expected_path)
    if len(matching) == 1 and len(records) == 1:
        return matching[0]
    if not matching:
        message = "installed distribution RECORD is missing the expected console launcher row"
    elif len(matching) > 1:
        message = "installed distribution RECORD contains duplicate console launcher rows"
    else:
        message = "installed distribution RECORD contains an unauthorized external path"
    raise DistributionIdentityError(
        OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
        message,
        field="RECORD",
    )


def _expected_console_launcher(install_root: Path) -> tuple[Path, str]:
    """Compute the interpreter's launcher path and its POSIX RECORD-relative spelling."""
    scripts = sysconfig.get_path("scripts")
    if type(scripts) is not str or not scripts:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "the current interpreter does not expose an expected scripts directory",
            field="sysconfig.scripts",
        )
    launcher = Path(scripts) / _CONSOLE_SCRIPT_NAME
    if os.name != "posix" or not launcher.is_absolute():
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "the current interpreter console launcher path is not absolute POSIX form",
            field="sysconfig.scripts",
        )
    try:
        relative = os.path.relpath(launcher, start=install_root)
    except ValueError as exc:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "the expected console launcher cannot be related to the installation root",
            field="sysconfig.scripts",
        ) from exc
    record_path = Path(relative).as_posix()
    if not record_path or "\\" in record_path or PurePosixPath(record_path).is_absolute():
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "the expected console launcher RECORD path is not POSIX relative form",
            field="sysconfig.scripts",
        )
    return launcher, record_path


def _verified_external_launcher(record: _ManifestRecord, launcher: Path) -> None:
    """Require a real regular console launcher whose bytes match its RECORD row."""
    try:
        mode = launcher.lstat().st_mode
    except OSError as exc:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "the expected console launcher is missing or unreadable",
            field=record.path,
        ) from exc
    if stat.S_ISLNK(mode):
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "the expected console launcher must not be a symlink",
            field=record.path,
        )
    if not stat.S_ISREG(mode):
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "the expected console launcher is not a regular file",
            field=record.path,
        )
    try:
        payload = launcher.read_bytes()
    except OSError as exc:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "the expected console launcher is unreadable",
            field=record.path,
        ) from exc
    _verify_record_payload(record, payload, "console launcher")


def _dist_info_directory(records: Mapping[str, _ManifestRecord]) -> str:
    """Identify the sole wheel ``dist-info`` directory and require WHEEL and RECORD rows."""
    metadata_paths = [path for path in records if path.endswith(".dist-info/METADATA")]
    if len(metadata_paths) != 1:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "installed wheel must contain exactly one dist-info/METADATA",
            field="RECORD",
        )
    dist_info = metadata_paths[0].rsplit("/", 1)[0]
    if f"{dist_info}/WHEEL" not in records:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "source-only or non-wheel distribution is unsupported",
            field="RECORD",
        )
    if f"{dist_info}/RECORD" not in records:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "wheel RECORD must list itself",
            field="RECORD",
        )
    return dist_info


def _distribution_install_root(
    distribution: metadata.Distribution,
    records: Mapping[str, _ManifestRecord],
    dist_info: str,
) -> Path:
    """Anchor the installation root through the measured real dist-info METADATA file."""
    metadata_path = f"{dist_info}/METADATA"
    record = records[metadata_path]
    raw = Path(str(distribution.locate_file(metadata.PackagePath(record.path))))
    try:
        if raw.is_symlink():
            raise OSError("METADATA is a symlink")
        resolved = raw.resolve(strict=True)
        mode = resolved.lstat().st_mode
    except OSError as exc:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "installed dist-info METADATA is missing, unreadable, or symlinked",
            field=metadata_path,
        ) from exc
    if not stat.S_ISREG(mode):
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "installed dist-info METADATA is not a regular file",
            field=metadata_path,
        )
    return resolved.parent.parent


def _reject_editable(
    distribution: metadata.Distribution,
    records: Mapping[str, _ManifestRecord],
    dist_info: str,
    install_root: Path,
) -> None:
    """Reject editable finder markers and editable ``direct_url.json`` metadata."""
    if any(
        PurePosixPath(path).name.startswith(("__editable__.", "__editable___")) for path in records
    ):
        raise DistributionIdentityError(
            OperationalReasonCode.EDITABLE_INSTALL_UNSUPPORTED,
            "editable-install finder markers are unsupported",
            field="RECORD",
        )
    direct_url = f"{dist_info}/direct_url.json"
    record = records.get(direct_url)
    if record is None:
        return
    path = _resolved_regular_member(distribution, record, install_root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "installed direct_url.json is unreadable or malformed",
            field="direct_url.json",
        ) from exc
    if isinstance(raw, dict):
        dir_info = raw.get("dir_info")
        if isinstance(dir_info, dict) and dir_info.get("editable") is True:
            raise DistributionIdentityError(
                OperationalReasonCode.EDITABLE_INSTALL_UNSUPPORTED,
                "editable installations are unsupported",
                field="direct_url.json",
            )


def _expected_package_root(
    distribution: metadata.Distribution,
    records: Mapping[str, _ManifestRecord],
) -> Path:
    """Resolve the real installed package root through its manifest-bound initializer."""
    normalized = f"{_PACKAGE_NAME}/__init__.py"
    record = records.get(normalized)
    if record is None:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "installed package payload is incomplete",
            field="RECORD",
        )
    raw = Path(str(distribution.locate_file(metadata.PackagePath(record.path))))
    try:
        if raw.is_symlink():
            raise OSError("package initializer is a symlink")
        resolved = raw.resolve(strict=True)
        mode = resolved.lstat().st_mode
    except OSError as exc:
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "installed package initializer is missing, unreadable, or symlinked",
            field=normalized,
        ) from exc
    if not stat.S_ISREG(mode):
        raise DistributionIdentityError(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            "installed package initializer is not a regular file",
            field=normalized,
        )
    return resolved.parent
