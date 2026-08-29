"""Pure versioned identities for characterized MuJoCo model-review surfaces.

The entries in this module are measurement results, not runtime-admission policy.  A release is
added only after two clean processes running that exact upstream distribution emit byte-identical
surface and comparable-registry observations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

PUBLIC_FIELD_REGISTRY_SCHEMA: Final = "metrifid.mujoco_public_field_registry"
PUBLIC_FIELD_REGISTRY_SCHEMA_VERSION: Final = 1

_STABLE_PACKAGE_VERSION: Final = re.compile(
    r"(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)"
    r"(?:\.post(?:0|[1-9][0-9]*))?"
    r"(?:\+[0-9A-Za-z]+(?:[._-][0-9A-Za-z]+)*)?"
)
_STABLE_NATIVE_VERSION: Final = re.compile(
    r"(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)"
)


@dataclass(frozen=True, slots=True)
class PublicFieldRegistryCatalogEntry:
    """One exact runtime's independently repeated public-model surface observation."""

    base_version: str
    full_public_surface_sha256: str
    full_public_surface_count: int
    comparable_registry_sha256: str
    comparable_registry_count: int
    measurement_process_count: int
    measurements_identical: bool


_MEASURED_ENTRIES: Final = (
    PublicFieldRegistryCatalogEntry(
        base_version="3.9.0",
        full_public_surface_sha256=(
            "a270838d7ea9371f99fbf831a495ae2a2320d8dfd39e469c06d06bb543d02aab"
        ),
        full_public_surface_count=721,
        comparable_registry_sha256=(
            "d2659517d9515588122c086f801bd1ed96984f31437a9c1d225ac359c344be02"
        ),
        comparable_registry_count=675,
        measurement_process_count=2,
        measurements_identical=True,
    ),
    PublicFieldRegistryCatalogEntry(
        base_version="3.10.0",
        full_public_surface_sha256=(
            "a270838d7ea9371f99fbf831a495ae2a2320d8dfd39e469c06d06bb543d02aab"
        ),
        full_public_surface_count=721,
        comparable_registry_sha256=(
            "d2659517d9515588122c086f801bd1ed96984f31437a9c1d225ac359c344be02"
        ),
        comparable_registry_count=675,
        measurement_process_count=2,
        measurements_identical=True,
    ),
    PublicFieldRegistryCatalogEntry(
        base_version="3.11.0",
        full_public_surface_sha256=(
            "db4b643579ddd9f57f3ec01b65a8de2789819b1157c92db0c3465a663306c2f6"
        ),
        full_public_surface_count=743,
        comparable_registry_sha256=(
            "7fdcfeb200258a05cac058a98d1926590dcd266046f188a4c65cc26fa747b656"
        ),
        comparable_registry_count=697,
        measurement_process_count=2,
        measurements_identical=True,
    ),
    PublicFieldRegistryCatalogEntry(
        base_version="3.12.0",
        full_public_surface_sha256=(
            "0b56690844f789b9a68dc18e4243f6549d01f5059a18922d574469b463633440"
        ),
        full_public_surface_count=745,
        comparable_registry_sha256=(
            "23bb9b012fe1b838e33ba50848c35afa6d9ac5e76a15bc8e679ad46233c783cd"
        ),
        comparable_registry_count=699,
        measurement_process_count=2,
        measurements_identical=True,
    ),
)

PUBLIC_FIELD_REGISTRY_CATALOG: Final = MappingProxyType(
    {entry.base_version: entry for entry in _MEASURED_ENTRIES}
)


def characterized_registry(base_version: str) -> PublicFieldRegistryCatalogEntry | None:
    """Return an exact measured entry without treating the catalog as an allowlist."""
    return PUBLIC_FIELD_REGISTRY_CATALOG.get(base_version)


def coherent_runtime_base_version(
    package_version: str,
    native_version_string: str,
    native_version_integer: int,
) -> str:
    """Derive one coherent stable base version using only pure receipt facts."""
    if type(package_version) is not str or type(native_version_string) is not str:
        raise TypeError("MuJoCo package and native versions must be strings")
    if type(native_version_integer) is not int:
        raise TypeError("MuJoCo native version integer must be an integer")
    package_match = _STABLE_PACKAGE_VERSION.fullmatch(package_version)
    if package_match is None:
        raise ValueError("MuJoCo package version is not an admitted stable release")
    native_match = _STABLE_NATIVE_VERSION.fullmatch(native_version_string)
    if native_match is None:
        raise ValueError("MuJoCo native version string is not an exact stable triplet")
    package_triplet = tuple(int(package_match.group(name)) for name in ("major", "minor", "patch"))
    native_triplet = tuple(int(native_match.group(name)) for name in ("major", "minor", "patch"))
    if package_triplet != native_triplet:
        raise ValueError("MuJoCo package and native base versions differ")
    if any(component >= 1000 for component in native_triplet):
        raise ValueError("MuJoCo native version component cannot be encoded unambiguously")
    major, minor, patch = native_triplet
    expected_integer = major * 1_000_000 + minor * 1_000 + patch
    if native_version_integer != expected_integer:
        raise ValueError("MuJoCo native version integer contradicts the native triplet")
    return f"{major}.{minor}.{patch}"


__all__ = [
    "PUBLIC_FIELD_REGISTRY_CATALOG",
    "PUBLIC_FIELD_REGISTRY_SCHEMA",
    "PUBLIC_FIELD_REGISTRY_SCHEMA_VERSION",
    "PublicFieldRegistryCatalogEntry",
    "characterized_registry",
    "coherent_runtime_base_version",
]
