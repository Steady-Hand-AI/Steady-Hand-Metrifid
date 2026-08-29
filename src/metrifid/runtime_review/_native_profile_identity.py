"""Collect one exact native Runtime Review profile identity.

This private standalone program deliberately imports no Metrifid package module.  It loads the
frozen native evidence worker by its admitted absolute path only after :func:`main` is called, then
reuses the worker's measurement and manifest helpers to produce the execution identity consumed by
the Runtime Review runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import struct
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Protocol, cast

_LEGACY_FROZEN_WORKER_SHA256: Final = (
    "941cc0cba66632901e89ee0a5be63575a2a5635dc98595e10271d9bed003dd6f"
)
# Updated after the standalone worker bytes are finalized.  The legacy digest above remains
# immutable so retained schema-v1 identities can still be replayed exactly.
_FROZEN_WORKER_SHA256: Final = "b00e509a344593806c088c4e49783ed71bacd815466d74bce9e27c931535b4ff"
_PROFILE_IDENTITY_SCHEMA: Final = "metrifid.runtime_review.native_profile_identity"
_LEGACY_SCHEMA_VERSION: Final = 1
_PRODUCTION_SCHEMA_VERSION: Final = 2
_MAX_IDENTITY_BYTES: Final = 4 * 1024 * 1024
_MAX_WORKER_BYTES: Final = 1024 * 1024
_SMOKE_STEP: Final = Decimal("0.004")
_IDENTIFIER_PATTERN: Final = re.compile(r"[a-z][a-z0-9_.-]{0,95}\Z")
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}\Z")
_PROFILE_VERSIONS: Final = {"A_3.10.0": "3.10.0", "B_3.11.0": "3.11.0"}
_PROFILE_ROLES: Final = ("baseline", "candidate")
_STABLE_PACKAGE_PATTERN: Final = re.compile(
    r"\A(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)"
    r"(?:\.post(?:0|[1-9][0-9]*))?"
    r"(?:\+[0-9A-Za-z]+(?:[._-][0-9A-Za-z]+)*)?\Z"
)
_STABLE_NATIVE_PATTERN: Final = re.compile(
    r"\A(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)\Z"
)
_MINIMUM_MUJOCO_VERSION: Final = (3, 9, 0)
_DETERMINISTIC_ENVIRONMENT: Final = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_DISTRIBUTION_FIELDS: Final = {
    "name",
    "version",
    "member_count",
    "payload_identity_algorithm",
    "payload_sha256",
    "record_bound_identity_algorithm",
    "record_bound_member_count",
    "record_bound_payload_sha256",
    "record_declared_sha256_member_count",
    "record_unhashed_member_count",
}
_LEGACY_PROFILE_IDENTITY_FIELDS: Final = {
    "schema",
    "schema_version",
    "profile_id",
    "profile_contract",
    "python",
    "host",
    "environment",
    "mujoco",
    "numpy",
    "installation",
    "metrifid_installed",
    "pip_check",
    "native_smoke",
    "profile_identity_sha256",
}
_PRODUCTION_PROFILE_IDENTITY_FIELDS: Final = {
    "schema",
    "schema_version",
    "profile_role",
    "package_version",
    "native_version",
    "native_version_integer",
    "support_tier",
    "profile_contract",
    "python",
    "host",
    "environment",
    "mujoco",
    "numpy",
    "installation",
    "metrifid_installed",
    "pip_check",
    "native_smoke",
    "sentinel",
    "profile_identity_sha256",
}
_SENTINEL_FIELDS: Final = {
    "schema",
    "schema_version",
    "profile_role",
    "fixture_id",
    "step_dt",
    "state_signature",
    "state_size",
    "warmup_step_count",
    "pre_restore_integration_state",
    "pre_restore_integration_state_sha256",
    "post_forward_projection",
    "post_forward_projection_sha256",
    "post_step_integration_state",
    "post_step_integration_state_sha256",
    "post_step_projection",
    "post_step_projection_sha256",
    "finite_values",
    "warnings_passed",
    "solver_converged",
    "status",
    "failure_reason",
    "limitations",
    "sentinel_identity_sha256",
}


class ProfileIdentityRefusal(ValueError):
    """Represent one bounded collector or profile-identity admission refusal."""


class _FrozenWorker(Protocol):
    """Describe only the frozen worker helpers reused by this collector."""

    def _runtime_identity(
        self, profile_role: str | None = None, *, allow_unbound_profile: bool = False
    ) -> tuple[str, str, dict[str, Any]]:
        """Measure the active worker runtime identity."""

    def _load_manifest(self, path: Path, fixture_id: str) -> tuple[bytes, dict[str, Any], Any]:
        """Admit one manifest and selected fixture."""

    def _resolve_fixture_source(self, manifest_path: Path, relative: str) -> tuple[Path, bytes]:
        """Resolve one self-contained fixture source."""

    def _admit_self_contained_xml(self, raw: bytes) -> str:
        """Admit one self-contained XML document."""

    def _compile_model(self, xml_text: str, spec: Any, step_dt: Decimal) -> Any:
        """Compile one admitted model at the smoke timestep."""

    def _compiled_mjb(self, model: Any) -> bytes:
        """Serialize one compiled model to nonempty native bytes."""

    def _distribution_payload(self, name: str) -> dict[str, Any]:
        """Measure one installed distribution and all RECORD-bound members."""

    def _same_profile_sentinel(self, model: Any, spec: Any, profile_role: str) -> dict[str, Any]:
        """Execute one public complete-integration-state sentinel."""


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize one JSON value with the product's canonical representation."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    """Hash one canonical JSON value."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    """Hash one regular nonsymlink file using bounded-memory reads."""
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ProfileIdentityRefusal("identity input is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ProfileIdentityRefusal("identity input must be a regular nonsymlink file")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ProfileIdentityRefusal("identity input could not be read") from exc
    return digest.hexdigest()


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one object while refusing duplicate JSON member names."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProfileIdentityRefusal(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    """Refuse JSON parser extensions for nonfinite numbers."""
    raise ProfileIdentityRefusal(f"non-finite JSON number is forbidden: {value}")


def _strict_json_document(payload: bytes) -> dict[str, Any]:
    """Decode one strict UTF-8 JSON object with duplicate-name rejection."""
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ProfileIdentityRefusal("profile identity is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ProfileIdentityRefusal("profile identity root must be an object")
    document = cast(dict[str, Any], value)
    _require_bounded_json(document)
    return document


def _require_bounded_json(value: Any) -> None:
    """Refuse a deeply nested or excessively broad decoded JSON value iteratively."""
    pending: list[tuple[Any, int]] = [(value, 0)]
    visited = 0
    while pending:
        current, depth = pending.pop()
        visited += 1
        if depth > 32 or visited > 20_000:
            raise ProfileIdentityRefusal("profile identity JSON structure exceeds its bounds")
        if type(current) is dict:
            pending.extend((item, depth + 1) for item in cast(dict[str, Any], current).values())
        elif type(current) is list:
            pending.extend((item, depth + 1) for item in current)


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    """Require an object's exact closed field set."""
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ProfileIdentityRefusal(
            f"{context} fields differ; missing={missing}, unknown={unknown}"
        )


def _object(value: Any, context: str) -> dict[str, Any]:
    """Require one exact JSON object."""
    if type(value) is not dict:
        raise ProfileIdentityRefusal(f"{context} must be an object")
    return cast(dict[str, Any], value)


def _string(value: Any, context: str) -> str:
    """Require one nonempty JSON string."""
    if type(value) is not str or not value:
        raise ProfileIdentityRefusal(f"{context} must be a nonempty string")
    return value


def _boolean(value: Any, context: str) -> bool:
    """Require one exact JSON boolean."""
    if type(value) is not bool:
        raise ProfileIdentityRefusal(f"{context} must be a boolean")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    """Require one bounded nonnegative JSON integer."""
    if type(value) is not int or value < minimum:
        raise ProfileIdentityRefusal(f"{context} must be an integer of at least {minimum}")
    return value


def _sha256(value: Any, context: str) -> str:
    """Require one lowercase SHA-256 hexadecimal token."""
    token = _string(value, context)
    if _SHA256_PATTERN.fullmatch(token) is None:
        raise ProfileIdentityRefusal(f"{context} must be a lowercase SHA-256")
    return token


def _absolute_path(value: Any, context: str) -> str:
    """Require one explicit absolute filesystem path string."""
    token = _string(value, context)
    if not Path(token).is_absolute():
        raise ProfileIdentityRefusal(f"{context} must be an absolute path")
    return token


def _compact_distribution(value: Any) -> dict[str, Any]:
    """Project the frozen worker's distribution measurement without its member expansion."""
    distribution = _object(value, "worker distribution")
    if not _DISTRIBUTION_FIELDS.issubset(distribution):
        raise ProfileIdentityRefusal("worker distribution measurement is incomplete")
    return {key: distribution[key] for key in sorted(_DISTRIBUTION_FIELDS)}


def _record_bound_distribution_identity(value: Any) -> dict[str, Any]:
    """Project one install to its exact wheel-stable RECORD-bound distribution identity."""
    distribution = _object(value, "distribution identity")
    if not _DISTRIBUTION_FIELDS.issubset(distribution):
        raise ProfileIdentityRefusal("distribution identity is incomplete")
    return {key: distribution[key] for key in sorted(_DISTRIBUTION_FIELDS - {"payload_sha256"})}


def _validate_distribution(value: Any, context: str, name: str, version: str) -> dict[str, Any]:
    """Validate one exact compact worker distribution projection."""
    distribution = _object(value, context)
    _exact_keys(distribution, _DISTRIBUTION_FIELDS, context)
    if distribution["name"] != name or distribution["version"] != version:
        raise ProfileIdentityRefusal(f"{context} has the wrong package identity")
    _string(distribution["payload_identity_algorithm"], f"{context}.payload_identity_algorithm")
    _sha256(distribution["payload_sha256"], f"{context}.payload_sha256")
    _string(
        distribution["record_bound_identity_algorithm"],
        f"{context}.record_bound_identity_algorithm",
    )
    _sha256(
        distribution["record_bound_payload_sha256"],
        f"{context}.record_bound_payload_sha256",
    )
    member_count = _integer(distribution["member_count"], f"{context}.member_count", minimum=1)
    bound_count = _integer(
        distribution["record_bound_member_count"],
        f"{context}.record_bound_member_count",
        minimum=1,
    )
    declared_count = _integer(
        distribution["record_declared_sha256_member_count"],
        f"{context}.record_declared_sha256_member_count",
        minimum=1,
    )
    unhashed_count = _integer(
        distribution["record_unhashed_member_count"],
        f"{context}.record_unhashed_member_count",
    )
    if member_count != declared_count + unhashed_count or bound_count > declared_count:
        raise ProfileIdentityRefusal(f"{context} member counts are inconsistent")
    return distribution


def _validate_profile_contract(
    value: Any, expected_profile_id: str, expected_worker_sha256: str
) -> dict[str, Any]:
    """Validate the frozen runtime, headless, and worker contract for one role."""
    contract = _object(value, "profile_contract")
    _exact_keys(
        contract,
        {
            "headless",
            "mujoco_version",
            "numpy_version",
            "viewer_or_renderer_used",
            "worker_sha256",
        },
        "profile_contract",
    )
    expected_version = _PROFILE_VERSIONS[expected_profile_id]
    if (
        _boolean(contract["headless"], "profile_contract.headless") is not True
        or _boolean(contract["viewer_or_renderer_used"], "profile_contract.viewer_or_renderer_used")
        is not False
        or contract["mujoco_version"] != expected_version
        or contract["numpy_version"] != "2.3.5"
        or _sha256(contract["worker_sha256"], "profile_contract.worker_sha256")
        != expected_worker_sha256
    ):
        raise ProfileIdentityRefusal("profile contract does not match the frozen runtime role")
    return contract


def _validate_python(value: Any) -> dict[str, Any]:
    """Validate exact CPython executable, build, and resolved-binary identity."""
    python = _object(value, "python")
    _exact_keys(
        python,
        {
            "build",
            "cache_tag",
            "compiler",
            "executable",
            "implementation",
            "implementation_name",
            "resolved_executable",
            "resolved_executable_sha256",
            "version",
            "version_full",
        },
        "python",
    )
    build = python["build"]
    if type(build) is not list or len(build) != 2 or any(type(item) is not str for item in build):
        raise ProfileIdentityRefusal("python.build must contain exactly two strings")
    _absolute_path(python["executable"], "python.executable")
    _absolute_path(python["resolved_executable"], "python.resolved_executable")
    _sha256(python["resolved_executable_sha256"], "python.resolved_executable_sha256")
    for field in ("cache_tag", "compiler", "version", "version_full"):
        _string(python[field], f"python.{field}")
    if python["implementation"] != "CPython" or python["implementation_name"] != "cpython":
        raise ProfileIdentityRefusal("python identity must describe CPython")
    return python


def _validate_host(value: Any) -> dict[str, Any]:
    """Validate the worker's exact cross-platform host projection."""
    host = _object(value, "host")
    _exact_keys(
        host,
        {
            "architecture",
            "cpu_model",
            "cpu_model_source",
            "hardware_model",
            "hardware_profile",
            "hyper_threading_technology",
            "logical_cpu_count",
            "machine",
            "physical_cpu_count",
            "platform",
            "release",
            "system",
            "version",
        },
        "host",
    )
    architecture = host["architecture"]
    if (
        type(architecture) is not list
        or len(architecture) != 2
        or any(type(item) is not str for item in architecture)
    ):
        raise ProfileIdentityRefusal("host.architecture must contain exactly two strings")
    for field in (
        "cpu_model",
        "cpu_model_source",
        "hardware_model",
        "machine",
        "platform",
        "release",
        "system",
        "version",
    ):
        if type(host[field]) is not str:
            raise ProfileIdentityRefusal(f"host.{field} must be a string")
    _integer(host["logical_cpu_count"], "host.logical_cpu_count", minimum=1)
    physical_count = host["physical_cpu_count"]
    if physical_count is not None:
        _integer(physical_count, "host.physical_cpu_count", minimum=1)
    hyper_threading = host["hyper_threading_technology"]
    if hyper_threading is not None and type(hyper_threading) is not str:
        raise ProfileIdentityRefusal("host.hyper_threading_technology must be a string or null")
    hardware_profile = _object(host["hardware_profile"], "host.hardware_profile")
    if any(type(key) is not str or type(item) is not str for key, item in hardware_profile.items()):
        raise ProfileIdentityRefusal("host.hardware_profile must contain only string values")
    return host


def _validate_environment(value: Any) -> dict[str, Any]:
    """Validate the exact deterministic worker environment."""
    environment = _object(value, "environment")
    if environment != _DETERMINISTIC_ENVIRONMENT:
        raise ProfileIdentityRefusal(
            "profile environment does not match the deterministic contract"
        )
    return environment


def _validate_native_library(value: Any) -> dict[str, Any]:
    """Validate the worker's exact compact loaded-native-library projection."""
    library = _object(value, "mujoco.loaded_native_library")
    _exact_keys(
        library,
        {"filename", "loaded_path", "resolved_path", "sha256", "size_bytes"},
        "mujoco.loaded_native_library",
    )
    filename = _string(library["filename"], "mujoco.loaded_native_library.filename")
    if Path(filename).name != filename:
        raise ProfileIdentityRefusal("native library filename must be a bare name")
    _absolute_path(library["loaded_path"], "mujoco.loaded_native_library.loaded_path")
    _absolute_path(library["resolved_path"], "mujoco.loaded_native_library.resolved_path")
    _sha256(library["sha256"], "mujoco.loaded_native_library.sha256")
    _integer(library["size_bytes"], "mujoco.loaded_native_library.size_bytes", minimum=1)
    return library


def _validate_mujoco(value: Any, expected_version: str) -> dict[str, Any]:
    """Validate exact MuJoCo Python, native, distribution, and library identity."""
    mujoco = _object(value, "mujoco")
    _exact_keys(
        mujoco,
        {
            "distribution",
            "loaded_native_library",
            "native_version",
            "python_version",
            "version_integer",
        },
        "mujoco",
    )
    if mujoco["python_version"] != expected_version or mujoco["native_version"] != expected_version:
        raise ProfileIdentityRefusal("MuJoCo Python and native versions do not match the role")
    major, minor, patch = (int(token) for token in expected_version.split("."))
    expected_integer = major * 1_000_000 + minor * 1_000 + patch
    if _integer(mujoco["version_integer"], "mujoco.version_integer", minimum=1) != expected_integer:
        raise ProfileIdentityRefusal("MuJoCo version integer does not match the role")
    _validate_distribution(
        mujoco["distribution"], "mujoco.distribution", "mujoco", expected_version
    )
    _validate_native_library(mujoco["loaded_native_library"])
    return mujoco


def _validate_numpy(value: Any) -> dict[str, Any]:
    """Validate exact NumPy Python and distribution identity."""
    numpy = _object(value, "numpy")
    _exact_keys(numpy, {"distribution", "python_version"}, "numpy")
    if numpy["python_version"] != "2.3.5":
        raise ProfileIdentityRefusal("NumPy Python version does not match the frozen contract")
    _validate_distribution(numpy["distribution"], "numpy.distribution", "numpy", "2.3.5")
    return numpy


def _validate_installation(value: Any) -> dict[str, Any]:
    """Validate an explicit unavailable installation-provenance projection."""
    installation = _object(value, "installation")
    _exact_keys(installation, {"artifacts", "available", "install_command"}, "installation")
    if (
        _boolean(installation["available"], "installation.available") is not False
        or installation["install_command"] is not None
        or installation["artifacts"] != []
    ):
        raise ProfileIdentityRefusal("installation provenance must be explicitly unavailable")
    return installation


def _validate_pip_check(value: Any, python_executable: str) -> dict[str, Any]:
    """Validate one successful bounded pip consistency measurement."""
    pip_check = _object(value, "pip_check")
    _exact_keys(pip_check, {"argv", "exit_code", "stderr", "stdout"}, "pip_check")
    if pip_check["argv"] != [python_executable, "-m", "pip", "check"]:
        raise ProfileIdentityRefusal("pip-check argv does not bind the measured interpreter")
    if _integer(pip_check["exit_code"], "pip_check.exit_code") != 0:
        raise ProfileIdentityRefusal("pip check did not pass")
    if type(pip_check["stdout"]) is not str or type(pip_check["stderr"]) is not str:
        raise ProfileIdentityRefusal("pip-check output fields must be strings")
    return pip_check


def _validate_native_smoke(value: Any) -> dict[str, Any]:
    """Validate one successful headless compile-only smoke measurement."""
    smoke = _object(value, "native_smoke")
    _exact_keys(
        smoke,
        {
            "compiled_model_nbody",
            "compiled_model_ngeom",
            "compiled_model_nq",
            "compiled_model_nu",
            "compiled_model_nv",
            "compiled_model_size_bytes",
            "fixture_id",
            "manifest_raw_sha256",
            "passed",
            "step_dt",
        },
        "native_smoke",
    )
    fixture_id = _string(smoke["fixture_id"], "native_smoke.fixture_id")
    if _IDENTIFIER_PATTERN.fullmatch(fixture_id) is None:
        raise ProfileIdentityRefusal("native-smoke fixture ID is invalid")
    if smoke["step_dt"] != "0.004" or _boolean(smoke["passed"], "native_smoke.passed") is not True:
        raise ProfileIdentityRefusal("native smoke did not pass the exact compile contract")
    _sha256(smoke["manifest_raw_sha256"], "native_smoke.manifest_raw_sha256")
    _integer(smoke["compiled_model_nq"], "native_smoke.compiled_model_nq")
    for field in (
        "compiled_model_nv",
        "compiled_model_nu",
        "compiled_model_nbody",
        "compiled_model_ngeom",
    ):
        _integer(smoke[field], f"native_smoke.{field}")
    _integer(
        smoke["compiled_model_size_bytes"],
        "native_smoke.compiled_model_size_bytes",
        minimum=1,
    )
    return smoke


def _validate_native_smoke_v2(value: Any) -> dict[str, Any]:
    """Validate the production smoke plus dimensions used by sentinel projections."""
    smoke = _object(value, "native_smoke")
    _exact_keys(
        smoke,
        {
            "compiled_model_nbody",
            "compiled_model_ngeom",
            "compiled_model_nq",
            "compiled_model_nsensordata",
            "compiled_model_nu",
            "compiled_model_nv",
            "compiled_model_size_bytes",
            "fixture_id",
            "manifest_raw_sha256",
            "passed",
            "step_dt",
        },
        "native_smoke",
    )
    projected = dict(smoke)
    sensor_width = projected.pop("compiled_model_nsensordata")
    _validate_native_smoke(projected)
    _integer(sensor_width, "native_smoke.compiled_model_nsensordata")
    return smoke


def _validate_profile_identity_v1(
    document: dict[str, Any], *, expected_profile_id: str, expected_worker_sha256: str
) -> dict[str, Any]:
    """Validate one immutable historical Runtime Review profile identity."""
    if expected_profile_id not in _PROFILE_VERSIONS:
        raise ProfileIdentityRefusal("expected profile ID is unsupported")
    _sha256(expected_worker_sha256, "expected worker SHA-256")
    _exact_keys(document, _LEGACY_PROFILE_IDENTITY_FIELDS, "profile identity")
    if document["schema"] != _PROFILE_IDENTITY_SCHEMA or (
        _integer(document["schema_version"], "schema_version", minimum=1) != _LEGACY_SCHEMA_VERSION
    ):
        raise ProfileIdentityRefusal("profile identity schema is unsupported")
    if document["profile_id"] != expected_profile_id:
        raise ProfileIdentityRefusal("profile identity has the wrong runtime role")
    claimed_hash = _sha256(document["profile_identity_sha256"], "profile_identity_sha256")
    projection = dict(document)
    projection.pop("profile_identity_sha256")
    if _canonical_sha256(projection) != claimed_hash:
        raise ProfileIdentityRefusal("profile identity self-hash is invalid")
    _validate_profile_contract(
        document["profile_contract"], expected_profile_id, expected_worker_sha256
    )
    python = _validate_python(document["python"])
    _validate_host(document["host"])
    _validate_environment(document["environment"])
    _validate_mujoco(document["mujoco"], _PROFILE_VERSIONS[expected_profile_id])
    _validate_numpy(document["numpy"])
    _validate_installation(document["installation"])
    _boolean(document["metrifid_installed"], "metrifid_installed")
    _validate_pip_check(document["pip_check"], cast(str, python["executable"]))
    _validate_native_smoke(document["native_smoke"])
    return document


def _stable_version_triplet(value: Any, *, package: bool) -> tuple[int, int, int] | None:
    """Parse a bounded stable package token or exact native version triplet."""
    if type(value) is not str or len(value) > 128:
        return None
    pattern = _STABLE_PACKAGE_PATTERN if package else _STABLE_NATIVE_PATTERN
    match = pattern.fullmatch(value)
    if match is None:
        return None
    triplet = tuple(int(match.group(name)) for name in ("major", "minor", "patch"))
    if any(component >= 1000 for component in triplet):
        return None
    return cast(tuple[int, int, int], triplet)


def _validate_profile_contract_v2(value: Any, expected_worker_sha256: str) -> dict[str, Any]:
    """Validate the generic headless production worker contract."""
    contract = _object(value, "profile_contract")
    _exact_keys(
        contract,
        {"headless", "viewer_or_renderer_used", "worker_sha256"},
        "profile_contract",
    )
    if (
        _boolean(contract["headless"], "profile_contract.headless") is not True
        or _boolean(contract["viewer_or_renderer_used"], "profile_contract.viewer_or_renderer_used")
        is not False
        or _sha256(contract["worker_sha256"], "profile_contract.worker_sha256")
        != expected_worker_sha256
    ):
        raise ProfileIdentityRefusal("profile contract does not match the production worker")
    return contract


def _validate_host_v2(value: Any) -> dict[str, Any]:
    """Validate the production host projection including its explicit libc identity."""
    host = _object(value, "host")
    legacy_projection = dict(host)
    try:
        libc = legacy_projection.pop("libc")
    except KeyError as exc:
        raise ProfileIdentityRefusal("host.libc is required") from exc
    if type(libc) is not list or len(libc) != 2 or any(type(item) is not str for item in libc):
        raise ProfileIdentityRefusal("host.libc must contain exactly two strings")
    _validate_host(legacy_projection)
    return host


def _validate_mujoco_v2(
    value: Any, package_version: str, native_version: str, native_integer: int
) -> dict[str, Any]:
    """Validate generic exact package/native/distribution/library identity."""
    measured = _object(value, "mujoco")
    _exact_keys(
        measured,
        {
            "distribution",
            "loaded_native_library",
            "package_version",
            "native_version",
            "native_version_integer",
        },
        "mujoco",
    )
    if (
        measured["package_version"] != package_version
        or measured["native_version"] != native_version
        or measured["native_version_integer"] != native_integer
    ):
        raise ProfileIdentityRefusal("MuJoCo nested identity differs from the profile root")
    _validate_distribution(
        measured["distribution"], "mujoco.distribution", "mujoco", package_version
    )
    _validate_native_library(measured["loaded_native_library"])
    return measured


def _validate_numpy_v2(value: Any) -> dict[str, Any]:
    """Validate one exact but version-generic NumPy distribution identity."""
    measured = _object(value, "numpy")
    _exact_keys(measured, {"distribution", "python_version"}, "numpy")
    version = _string(measured["python_version"], "numpy.python_version")
    _validate_distribution(measured["distribution"], "numpy.distribution", "numpy", version)
    return measured


def _validate_installation_v2(value: Any) -> dict[str, Any]:
    """Validate optional measured Metrifid installation identity without install authority."""
    installation = _object(value, "installation")
    _exact_keys(installation, {"available", "distribution"}, "installation")
    available = _boolean(installation["available"], "installation.available")
    distribution = installation["distribution"]
    if not available:
        if distribution is not None:
            raise ProfileIdentityRefusal("unavailable installation must have null distribution")
        return installation
    measured = _object(distribution, "installation.distribution")
    version = _string(measured.get("version"), "installation.distribution.version")
    _validate_distribution(measured, "installation.distribution", "metrifid", version)
    return installation


def _finite_float_token(value: Any, field: str) -> float:
    """Decode one retained round-trippable finite binary64 token."""
    token = _string(value, field)
    try:
        result = float(token)
    except ValueError as exc:
        raise ProfileIdentityRefusal(f"{field} is not a binary64 token") from exc
    if not math.isfinite(result):
        raise ProfileIdentityRefusal(f"{field} must be finite")
    canonical = (
        ("-0.0" if math.copysign(1.0, result) < 0.0 else "0") if result == 0.0 else repr(result)
    )
    if token != canonical:
        raise ProfileIdentityRefusal(f"{field} is not a canonical binary64 token")
    return result


def _validate_state_evidence(value: Any, state_size: int, field: str) -> dict[str, Any]:
    """Validate and independently rehash one retained complete integration-state array."""
    state = _object(value, field)
    _exact_keys(state, {"dtype", "shape", "values", "sha256"}, field)
    if state["dtype"] != "<f8" or state["shape"] != [state_size]:
        raise ProfileIdentityRefusal(f"{field} has the wrong dtype or shape")
    values = state["values"]
    if type(values) is not list or len(values) != state_size:
        raise ProfileIdentityRefusal(f"{field}.values has the wrong width")
    payload = b"".join(
        struct.pack("<d", _finite_float_token(item, f"{field}.values")) for item in values
    )
    if hashlib.sha256(payload).hexdigest() != _sha256(state["sha256"], f"{field}.sha256"):
        raise ProfileIdentityRefusal(f"{field} byte hash is invalid")
    return state


def _validate_optional_projection(value: Any, field: str) -> dict[str, Any]:
    """Validate one exact available-or-absent finite array projection."""
    projection = _object(value, field)
    _exact_keys(projection, {"available", "shape", "values"}, field)
    available = _boolean(projection["available"], f"{field}.available")
    if not available:
        if projection["shape"] is not None or projection["values"] is not None:
            raise ProfileIdentityRefusal(f"{field} absent projection must retain null details")
        return projection
    shape = projection["shape"]
    values = projection["values"]
    if type(shape) is not list or any(type(item) is not int or item < 0 for item in shape):
        raise ProfileIdentityRefusal(f"{field}.shape must contain nonnegative integers")
    if type(values) is not list:
        raise ProfileIdentityRefusal(f"{field}.values must be an array")
    width = math.prod(cast(list[int], shape))
    if len(values) != width:
        raise ProfileIdentityRefusal(f"{field}.values has the wrong width")
    for item in values:
        _finite_float_token(item, f"{field}.values")
    return projection


def _validate_warning_projection(value: Any, field: str) -> dict[str, Any]:
    """Validate every warning counter and its exact passing predicate."""
    projection = _object(value, field)
    _exact_keys(projection, {"available", "passed", "records"}, field)
    available = _boolean(projection["available"], f"{field}.available")
    passed = _boolean(projection["passed"], f"{field}.passed")
    if not available:
        if projection["records"] is not None or not passed:
            raise ProfileIdentityRefusal(f"{field} absent warning projection is inconsistent")
        return projection
    records = projection["records"]
    if type(records) is not list:
        raise ProfileIdentityRefusal(f"{field}.records must be an array")
    expected_ids = list(range(len(records)))
    observed_ids: list[int] = []
    numbers: list[int] = []
    for index, item in enumerate(records):
        record = _object(item, f"{field}.records[{index}]")
        _exact_keys(record, {"warning_id", "number", "lastinfo"}, f"{field}.records")
        observed_ids.append(_integer(record["warning_id"], f"{field}.warning_id"))
        numbers.append(_integer(record["number"], f"{field}.number"))
        _integer(record["lastinfo"], f"{field}.lastinfo")
    if observed_ids != expected_ids or passed is not all(number == 0 for number in numbers):
        raise ProfileIdentityRefusal(f"{field} warning counters or pass fact are inconsistent")
    return projection


def _validate_solver_projection(value: Any, field: str) -> dict[str, Any]:
    """Validate available solver diagnostics and their strict iteration gate."""
    projection = _object(value, field)
    _exact_keys(
        projection,
        {"iterations", "residuals", "iteration_limit", "max_iterations", "passed"},
        field,
    )
    iterations = _object(projection["iterations"], f"{field}.iterations")
    residuals = _object(projection["residuals"], f"{field}.residuals")
    _exact_keys(iterations, {"available", "values"}, f"{field}.iterations")
    _exact_keys(residuals, {"available", "values"}, f"{field}.residuals")
    iteration_values: list[int]
    if _boolean(iterations["available"], f"{field}.iterations.available"):
        raw_iterations = iterations["values"]
        if type(raw_iterations) is not list:
            raise ProfileIdentityRefusal(f"{field}.iterations.values must be an array")
        iteration_values = [_integer(item, f"{field}.iterations.values") for item in raw_iterations]
    else:
        if iterations["values"] is not None:
            raise ProfileIdentityRefusal(f"{field}.iterations absent values must be null")
        iteration_values = []
    if _boolean(residuals["available"], f"{field}.residuals.available"):
        raw_residuals = residuals["values"]
        if type(raw_residuals) is not list:
            raise ProfileIdentityRefusal(f"{field}.residuals.values must be an array")
        for item in raw_residuals:
            _finite_float_token(item, f"{field}.residuals.values")
    elif residuals["values"] is not None:
        raise ProfileIdentityRefusal(f"{field}.residuals absent values must be null")
    iteration_limit = _integer(projection["iteration_limit"], f"{field}.iteration_limit", minimum=1)
    maximum = _integer(projection["max_iterations"], f"{field}.max_iterations")
    passed = _boolean(projection["passed"], f"{field}.passed")
    if maximum != max(iteration_values, default=0) or passed is not (maximum < iteration_limit):
        raise ProfileIdentityRefusal(f"{field} solver maximum or pass fact is inconsistent")
    return projection


def _validate_contact_projection(value: Any, field: str) -> dict[str, Any]:
    """Validate canonical active semantic contacts without raw-slot ordering."""
    projection = _object(value, field)
    _exact_keys(projection, {"ncon", "contacts"}, field)
    count = _integer(projection["ncon"], f"{field}.ncon")
    contacts = projection["contacts"]
    if type(contacts) is not list or len(contacts) != count:
        raise ProfileIdentityRefusal(f"{field}.contacts has the wrong active count")
    for index, item in enumerate(contacts):
        contact = _object(item, f"{field}.contacts[{index}]")
        _exact_keys(
            contact,
            {"geom_names", "dim", "distance", "position", "frame", "friction", "force"},
            f"{field}.contacts[{index}]",
        )
        names = contact["geom_names"]
        if (
            type(names) is not list
            or len(names) != 2
            or any(type(name) is not str or not name for name in names)
            or names != sorted(names)
        ):
            raise ProfileIdentityRefusal(f"{field} geom names are not canonical")
        dimension = _integer(contact["dim"], f"{field}.dim", minimum=1)
        if dimension > 6:
            raise ProfileIdentityRefusal(f"{field}.dim exceeds the public contact bound")
        _finite_float_token(contact["distance"], f"{field}.distance")
        widths = {"position": 3, "frame": 9, "friction": 5, "force": 6}
        for member, width in widths.items():
            values = contact[member]
            if type(values) is not list or len(values) != width:
                raise ProfileIdentityRefusal(f"{field}.{member} has the wrong fixed width")
            for token in values:
                _finite_float_token(token, f"{field}.{member}")
    if contacts != sorted(contacts, key=_canonical_json_bytes):
        raise ProfileIdentityRefusal(f"{field}.contacts are not in canonical semantic order")
    return projection


def _validate_constraint_projection(value: Any, field: str) -> dict[str, Any]:
    """Validate canonical active constraint type and finite-force observations."""
    projection = _object(value, field)
    _exact_keys(projection, {"available", "active_count", "rows"}, field)
    available = _boolean(projection["available"], f"{field}.available")
    active_count = _integer(projection["active_count"], f"{field}.active_count")
    if not available:
        if active_count != 0 or projection["rows"] is not None:
            raise ProfileIdentityRefusal(f"{field} absent constraint projection is inconsistent")
        return projection
    rows = projection["rows"]
    if type(rows) is not list or len(rows) != active_count:
        raise ProfileIdentityRefusal(f"{field}.rows has the wrong active count")
    for index, item in enumerate(rows):
        row = _object(item, f"{field}.rows[{index}]")
        _exact_keys(row, {"type", "force"}, f"{field}.rows[{index}]")
        _integer(row["type"], f"{field}.type")
        _finite_float_token(row["force"], f"{field}.force")
    if rows != sorted(rows, key=_canonical_json_bytes):
        raise ProfileIdentityRefusal(f"{field}.rows are not in canonical order")
    return projection


def _validate_sentinel_projection(value: Any, field: str) -> dict[str, Any]:
    """Validate the complete closed deterministic output projection and self-hash."""
    projection = _object(value, field)
    _exact_keys(
        projection,
        {
            "qacc",
            "qfrc_actuator",
            "qfrc_constraint",
            "sensordata",
            "warnings",
            "solver",
            "contacts",
            "constraints",
            "projection_sha256",
        },
        field,
    )
    for member in ("qacc", "qfrc_actuator", "qfrc_constraint", "sensordata"):
        _validate_optional_projection(projection[member], f"{field}.{member}")
    _validate_warning_projection(projection["warnings"], f"{field}.warnings")
    _validate_solver_projection(projection["solver"], f"{field}.solver")
    _validate_contact_projection(projection["contacts"], f"{field}.contacts")
    _validate_constraint_projection(projection["constraints"], f"{field}.constraints")
    claimed = _sha256(projection["projection_sha256"], f"{field}.projection_sha256")
    unhashed = dict(projection)
    unhashed.pop("projection_sha256")
    if _canonical_sha256(unhashed) != claimed:
        raise ProfileIdentityRefusal(f"{field} self-hash is invalid")
    return projection


def _validate_projection_pair(value: Any, claimed_hash: Any, field: str) -> dict[str, Any]:
    """Validate equal left/right canonical projections and their retained comparison hash."""
    pair = _object(value, field)
    _exact_keys(pair, {"left", "right"}, field)
    if pair["left"] != pair["right"]:
        raise ProfileIdentityRefusal(f"{field} left and right projections differ")
    for side in ("left", "right"):
        _validate_sentinel_projection(pair[side], f"{field}.{side}")
    if _canonical_sha256(pair) != _sha256(claimed_hash, f"{field}_sha256"):
        raise ProfileIdentityRefusal(f"{field} comparison hash is invalid")
    return pair


def _validate_sentinel_v2(
    value: Any,
    expected_profile_role: str,
    *,
    require_pass: bool,
    expected_nv: int | None = None,
    expected_sensor_width: int | None = None,
) -> dict[str, Any]:
    """Validate complete-state sentinel structure, hashes, role, and success status."""
    sentinel = _object(value, "sentinel")
    _validate_sentinel_envelope(sentinel, expected_profile_role)
    status = sentinel["status"]
    if require_pass and status != "PASS":
        raise ProfileIdentityRefusal("complete integration-state sentinel did not pass")
    if status == "FAIL":
        _string(sentinel["failure_reason"], "sentinel.failure_reason")
        return sentinel
    _validate_passing_sentinel(
        sentinel,
        expected_nv=expected_nv,
        expected_sensor_width=expected_sensor_width,
    )
    return sentinel


def _validate_sentinel_envelope(sentinel: dict[str, Any], expected_profile_role: str) -> None:
    """Validate fields and hashes shared by passing and retained failed sentinels."""
    _exact_keys(sentinel, _SENTINEL_FIELDS, "sentinel")
    if (
        sentinel["schema"] != "metrifid.runtime_review.integration_state_sentinel"
        or _integer(sentinel["schema_version"], "sentinel.schema_version", minimum=1) != 1
        or sentinel["profile_role"] != expected_profile_role
    ):
        raise ProfileIdentityRefusal("sentinel schema or profile role is inconsistent")
    claimed_self_hash = _sha256(
        sentinel["sentinel_identity_sha256"], "sentinel.sentinel_identity_sha256"
    )
    unhashed = dict(sentinel)
    unhashed.pop("sentinel_identity_sha256")
    if _canonical_sha256(unhashed) != claimed_self_hash:
        raise ProfileIdentityRefusal("sentinel self-hash is invalid")
    _string(sentinel["fixture_id"], "sentinel.fixture_id")
    if (
        sentinel["step_dt"] != "0.004"
        or _integer(sentinel["warmup_step_count"], "sentinel.warmup_step_count", minimum=0) != 2
    ):
        raise ProfileIdentityRefusal("sentinel procedure metadata is inconsistent")
    for field in ("finite_values", "warnings_passed", "solver_converged"):
        _boolean(sentinel[field], f"sentinel.{field}")
    limitations = sentinel["limitations"]
    if type(limitations) is not list or limitations != [
        "EXACT_PROFILE_FIXTURE_AND_COARSEST_TIMESTEP_ONLY",
        "PUBLIC_MJSTATE_INTEGRATION_ONLY",
        "EXACT_BINARY64_EQUALITY_NO_TOLERANCE",
        "NO_CROSS_PROFILE_EQUIVALENCE_CLAIM",
        "SENTINEL_PROVES_SAME_PROFILE_RESTORE_PIPELINE_ONLY",
    ]:
        raise ProfileIdentityRefusal("sentinel limitations differ from the fixed claim boundary")
    status = sentinel["status"]
    if status not in {"PASS", "FAIL"}:
        raise ProfileIdentityRefusal("sentinel status is unsupported")


def _validate_passing_sentinel(
    sentinel: dict[str, Any], *, expected_nv: int | None, expected_sensor_width: int | None
) -> None:
    """Validate exact state, projections, and pass facts for one successful sentinel."""
    if sentinel["failure_reason"] is not None:
        raise ProfileIdentityRefusal("passing sentinel must not retain a failure reason")
    _integer(sentinel["state_signature"], "sentinel.state_signature", minimum=1)
    if (
        sentinel["finite_values"] is not True
        or sentinel["warnings_passed"] is not True
        or sentinel["solver_converged"] is not True
    ):
        raise ProfileIdentityRefusal("passing sentinel does not satisfy the exact procedure")
    state_size = _integer(sentinel["state_size"], "sentinel.state_size", minimum=1)
    before = _validate_state_evidence(
        sentinel["pre_restore_integration_state"], state_size, "pre_restore_integration_state"
    )
    if before["sha256"] != _sha256(
        sentinel["pre_restore_integration_state_sha256"],
        "sentinel.pre_restore_integration_state_sha256",
    ):
        raise ProfileIdentityRefusal("pre-restore state hashes differ")
    forward_pair = _validate_projection_pair(
        sentinel["post_forward_projection"],
        sentinel["post_forward_projection_sha256"],
        "post_forward_projection",
    )
    step_state = _object(sentinel["post_step_integration_state"], "post_step_integration_state")
    _exact_keys(step_state, {"left", "right"}, "post_step_integration_state")
    left = _validate_state_evidence(step_state["left"], state_size, "post_step_state.left")
    right = _validate_state_evidence(step_state["right"], state_size, "post_step_state.right")
    if left != right or _canonical_sha256(step_state) != _sha256(
        sentinel["post_step_integration_state_sha256"],
        "sentinel.post_step_integration_state_sha256",
    ):
        raise ProfileIdentityRefusal("post-step integration-state comparison is invalid")
    step_pair = _validate_projection_pair(
        sentinel["post_step_projection"],
        sentinel["post_step_projection_sha256"],
        "post_step_projection",
    )
    projections = [
        cast(dict[str, Any], pair[side])
        for pair in (forward_pair, step_pair)
        for side in ("left", "right")
    ]
    if any(
        cast(dict[str, Any], projection["warnings"])["passed"] is not True
        or cast(dict[str, Any], projection["solver"])["passed"] is not True
        for projection in projections
    ):
        raise ProfileIdentityRefusal("passing sentinel contains a failed output projection")
    if expected_nv is not None and expected_sensor_width is not None:
        expected_shapes = {
            "qacc": [expected_nv],
            "qfrc_actuator": [expected_nv],
            "qfrc_constraint": [expected_nv],
            "sensordata": [expected_sensor_width],
        }
        for projection in projections:
            for field, shape in expected_shapes.items():
                measured = cast(dict[str, Any], projection[field])
                if measured["available"] is not True or measured["shape"] != shape:
                    raise ProfileIdentityRefusal(
                        f"passing sentinel {field} shape differs from compiled smoke"
                    )


def _validate_profile_identity_v2(
    document: dict[str, Any],
    *,
    expected_profile_role: str,
    expected_worker_sha256: str,
    require_sentinel_pass: bool = True,
) -> dict[str, Any]:
    """Validate one role-based exact production profile and its sentinel binding."""
    if expected_profile_role not in _PROFILE_ROLES:
        raise ProfileIdentityRefusal("expected profile role is unsupported")
    _sha256(expected_worker_sha256, "expected worker SHA-256")
    _exact_keys(document, _PRODUCTION_PROFILE_IDENTITY_FIELDS, "profile identity")
    if (
        document["schema"] != _PROFILE_IDENTITY_SCHEMA
        or _integer(document["schema_version"], "schema_version", minimum=1)
        != _PRODUCTION_SCHEMA_VERSION
        or document["profile_role"] != expected_profile_role
    ):
        raise ProfileIdentityRefusal("profile identity schema or role is unsupported")
    claimed = _sha256(document["profile_identity_sha256"], "profile_identity_sha256")
    unhashed = dict(document)
    unhashed.pop("profile_identity_sha256")
    if _canonical_sha256(unhashed) != claimed:
        raise ProfileIdentityRefusal("profile identity self-hash is invalid")
    package_version = _string(document["package_version"], "package_version")
    native_version = _string(document["native_version"], "native_version")
    package_triplet = _stable_version_triplet(package_version, package=True)
    native_triplet = _stable_version_triplet(native_version, package=False)
    native_integer = _integer(
        document["native_version_integer"], "native_version_integer", minimum=1
    )
    if (
        package_triplet is None
        or package_triplet < _MINIMUM_MUJOCO_VERSION
        or native_triplet != package_triplet
        or native_integer
        != package_triplet[0] * 1_000_000 + package_triplet[1] * 1_000 + package_triplet[2]
    ):
        raise ProfileIdentityRefusal("profile MuJoCo package/native identity is incoherent")
    if document["support_tier"] != "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE":
        raise ProfileIdentityRefusal("live profile support tier is not capability-compatible")
    _validate_profile_contract_v2(document["profile_contract"], expected_worker_sha256)
    python = _validate_python(document["python"])
    _validate_host_v2(document["host"])
    _validate_environment(document["environment"])
    _validate_mujoco_v2(document["mujoco"], package_version, native_version, native_integer)
    _validate_numpy_v2(document["numpy"])
    installation = _validate_installation_v2(document["installation"])
    metrifid_installed = _boolean(document["metrifid_installed"], "metrifid_installed")
    if metrifid_installed is not installation["available"]:
        raise ProfileIdentityRefusal("Metrifid installation facts are inconsistent")
    _validate_pip_check(document["pip_check"], cast(str, python["executable"]))
    smoke = _validate_native_smoke_v2(document["native_smoke"])
    sentinel = _validate_sentinel_v2(
        document["sentinel"],
        expected_profile_role,
        require_pass=require_sentinel_pass,
        expected_nv=cast(int, smoke["compiled_model_nv"]),
        expected_sensor_width=cast(int, smoke["compiled_model_nsensordata"]),
    )
    if sentinel["fixture_id"] != smoke["fixture_id"]:
        raise ProfileIdentityRefusal("sentinel fixture differs from compiled smoke fixture")
    return document


def load_native_profile_identity_v2(
    path: Path,
    *,
    expected_profile_role: str,
    expected_profile_identity_sha256: str | None = None,
    expected_worker_sha256: str | None = None,
) -> dict[str, Any]:
    """Load one production profile identity and require its exact optional outer binding."""
    worker_sha256 = (
        _FROZEN_WORKER_SHA256 if expected_worker_sha256 is None else expected_worker_sha256
    )
    document = _validate_profile_identity_v2(
        _read_profile_identity_document(path),
        expected_profile_role=expected_profile_role,
        expected_worker_sha256=worker_sha256,
    )
    if expected_profile_identity_sha256 is not None and document[
        "profile_identity_sha256"
    ] != _sha256(expected_profile_identity_sha256, "expected profile identity SHA-256"):
        raise ProfileIdentityRefusal("profile identity does not match its expected self-hash")
    return document


def _read_profile_identity_document(path: Path) -> dict[str, Any]:
    """Read one bounded stable regular profile-identity document exactly once."""
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
    except OSError as exc:
        raise ProfileIdentityRefusal("profile identity is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ProfileIdentityRefusal("profile identity must be a regular nonsymlink file")
        if before.st_size > _MAX_IDENTITY_BYTES:
            raise ProfileIdentityRefusal("profile identity exceeds the byte limit")
        chunks: list[bytes] = []
        remaining = _MAX_IDENTITY_BYTES + 1
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ProfileIdentityRefusal("profile identity could not be read") from exc
    finally:
        os.close(descriptor)
    before_projection = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    after_projection = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if len(raw) > _MAX_IDENTITY_BYTES:
        raise ProfileIdentityRefusal("profile identity exceeds the byte limit")
    if len(raw) != before.st_size or before_projection != after_projection:
        raise ProfileIdentityRefusal("profile identity changed while it was read")
    return _strict_json_document(raw)


def load_native_profile_identity(
    path: Path,
    *,
    expected_profile_id: str,
    expected_worker_sha256: str = _LEGACY_FROZEN_WORKER_SHA256,
) -> dict[str, Any]:
    """Load one immutable schema-v1 collector identity through the legacy route."""
    return _validate_profile_identity_v1(
        _read_profile_identity_document(path),
        expected_profile_id=expected_profile_id,
        expected_worker_sha256=expected_worker_sha256,
    )


def require_compatible_profile_identities_v2(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> None:
    """Require roles to share one Python build and wheel-stable NumPy identity."""
    _validate_profile_identity_v2(
        baseline,
        expected_profile_role="baseline",
        expected_worker_sha256=_FROZEN_WORKER_SHA256,
    )
    _validate_profile_identity_v2(
        candidate,
        expected_profile_role="candidate",
        expected_worker_sha256=_FROZEN_WORKER_SHA256,
    )
    if baseline["host"] != candidate["host"]:
        raise ProfileIdentityRefusal("native profiles do not share the same admitted host")
    if baseline["environment"] != candidate["environment"]:
        raise ProfileIdentityRefusal("native profiles do not share the deterministic environment")
    python_fields = {
        "build",
        "cache_tag",
        "compiler",
        "implementation",
        "implementation_name",
        "resolved_executable_sha256",
        "version",
        "version_full",
    }
    baseline_python = cast(dict[str, Any], baseline["python"])
    candidate_python = cast(dict[str, Any], candidate["python"])
    if any(baseline_python[field] != candidate_python[field] for field in python_fields):
        raise ProfileIdentityRefusal(
            "native profiles do not share one CPython build and executable"
        )
    baseline_numpy = cast(dict[str, Any], baseline["numpy"])
    candidate_numpy = cast(dict[str, Any], candidate["numpy"])
    if baseline_numpy["python_version"] != candidate_numpy[
        "python_version"
    ] or _record_bound_distribution_identity(
        baseline_numpy["distribution"]
    ) != _record_bound_distribution_identity(candidate_numpy["distribution"]):
        raise ProfileIdentityRefusal(
            "native profiles do not share one exact NumPy package and distribution identity"
        )


def profile_identity_receipt_projection_v2(identity: dict[str, Any]) -> dict[str, Any]:
    """Return the exact closed profile projection consumed by a version-2 receipt."""
    role = identity.get("profile_role")
    if role not in _PROFILE_ROLES:
        raise ProfileIdentityRefusal("profile identity lacks a supported semantic role")
    admitted = _validate_profile_identity_v2(
        identity,
        expected_profile_role=cast(str, role),
        expected_worker_sha256=_FROZEN_WORKER_SHA256,
    )
    mujoco = _object(admitted["mujoco"], "mujoco")
    sentinel = _object(admitted["sentinel"], "sentinel")
    return {
        "profile_role": admitted["profile_role"],
        "package_version": admitted["package_version"],
        "native_version": admitted["native_version"],
        "native_version_integer": admitted["native_version_integer"],
        "profile_identity_sha256": admitted["profile_identity_sha256"],
        "mujoco_distribution": mujoco["distribution"],
        "loaded_native_library": mujoco["loaded_native_library"],
        "numpy": admitted["numpy"],
        "sentinel": {
            "status": sentinel["status"],
            "sentinel_identity_sha256": sentinel["sentinel_identity_sha256"],
        },
        "support_tier": admitted["support_tier"],
    }


def _read_verified_worker_source(
    path: Path, expected_worker_sha256: str
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    """Read and hash the frozen worker through one retained no-follow descriptor."""
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
    except OSError as exc:
        raise ProfileIdentityRefusal("packaged native evidence worker is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ProfileIdentityRefusal(
                "packaged native evidence worker must be a regular nonsymlink file"
            )
        if before.st_size > _MAX_WORKER_BYTES:
            raise ProfileIdentityRefusal("packaged native evidence worker exceeds its byte limit")
        chunks: list[bytes] = []
        remaining = _MAX_WORKER_BYTES + 1
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        source = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ProfileIdentityRefusal("packaged native evidence worker could not be read") from exc
    finally:
        os.close(descriptor)
    before_projection = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    after_projection = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if (
        len(source) > _MAX_WORKER_BYTES
        or len(source) != before.st_size
        or before_projection != after_projection
    ):
        raise ProfileIdentityRefusal("packaged native evidence worker changed while it was read")
    if hashlib.sha256(source).hexdigest() != expected_worker_sha256:
        raise ProfileIdentityRefusal("packaged native evidence worker SHA-256 is invalid")
    return source, after_projection


def _load_frozen_worker(path: Path, *, expected_worker_sha256: str | None = None) -> _FrozenWorker:
    """Compile and execute only the descriptor-read, hash-verified frozen worker bytes."""
    if not path.is_absolute():
        raise ProfileIdentityRefusal("--worker must be an absolute path")
    expected_worker_sha256 = (
        _FROZEN_WORKER_SHA256 if expected_worker_sha256 is None else expected_worker_sha256
    )
    _sha256(expected_worker_sha256, "expected worker SHA-256")
    source, after_projection = _read_verified_worker_source(path, expected_worker_sha256)
    module_name = "_metrifid_runtime_review_native_evidence_worker"
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        code = compile(source, str(path), "exec", dont_inherit=True, optimize=0)
        exec(code, module.__dict__)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise ProfileIdentityRefusal("packaged native evidence worker cannot be loaded") from exc
    try:
        current = os.lstat(path)
    except OSError as exc:
        sys.modules.pop(module_name, None)
        raise ProfileIdentityRefusal(
            "packaged native evidence worker changed while loading"
        ) from exc
    current_projection = (
        current.st_dev,
        current.st_ino,
        current.st_mode,
        current.st_size,
        current.st_mtime_ns,
    )
    if current_projection != after_projection:
        sys.modules.pop(module_name, None)
        raise ProfileIdentityRefusal("packaged native evidence worker changed while loading")
    return cast(_FrozenWorker, module)


def _metrifid_installation_is_measured() -> bool:
    """Measure whether the external profile contains a Metrifid distribution."""
    try:
        importlib.metadata.distribution("metrifid")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _metrifid_installation_identity(worker: _FrozenWorker) -> dict[str, Any]:
    """Measure the installed Metrifid distribution when present without changing it."""
    if not _metrifid_installation_is_measured():
        return {"available": False, "distribution": None}
    return {
        "available": True,
        "distribution": _compact_distribution(worker._distribution_payload("metrifid")),
    }


def _smoke_identity(
    worker: _FrozenWorker,
    manifest_path: Path,
    fixture_id: str,
    *,
    include_sentinel_dimensions: bool = False,
) -> dict[str, Any]:
    """Compile one admitted fixture headlessly at the coarsest frozen step without simulating."""
    manifest_raw, _, fixture = worker._load_manifest(manifest_path, fixture_id)
    relative_xml = cast(str, fixture.xml_path)
    _, fixture_raw = worker._resolve_fixture_source(manifest_path, relative_xml)
    xml_text = worker._admit_self_contained_xml(fixture_raw)
    model = worker._compile_model(xml_text, fixture, _SMOKE_STEP)
    compiled_model = worker._compiled_mjb(model)
    model_nq = int(model.nq)
    if not compiled_model:
        raise ProfileIdentityRefusal("headless compile smoke produced no compiled state")
    smoke = {
        "compiled_model_nbody": int(model.nbody),
        "compiled_model_ngeom": int(model.ngeom),
        "compiled_model_nq": model_nq,
        "compiled_model_nu": int(model.nu),
        "compiled_model_nv": int(model.nv),
        "compiled_model_size_bytes": len(compiled_model),
        "fixture_id": fixture_id,
        "manifest_raw_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "passed": True,
        "step_dt": "0.004",
    }
    if include_sentinel_dimensions:
        smoke["compiled_model_nsensordata"] = int(model.nsensordata)
    return smoke


def _collect_identity(
    worker_path: Path, manifest_path: Path, fixture_id: str, declared_profile_id: str
) -> dict[str, Any]:
    """Measure and construct one exact canonical profile identity."""
    if declared_profile_id not in _PROFILE_VERSIONS:
        raise ProfileIdentityRefusal("--profile-id is unsupported")
    if _IDENTIFIER_PATTERN.fullmatch(fixture_id) is None:
        raise ProfileIdentityRefusal("--fixture-id is invalid")
    if not manifest_path.is_absolute():
        raise ProfileIdentityRefusal("--manifest must be an absolute path")
    worker = _load_frozen_worker(worker_path, expected_worker_sha256=_LEGACY_FROZEN_WORKER_SHA256)
    measured_profile_id, measured_version, runtime = worker._runtime_identity()
    if measured_profile_id != declared_profile_id:
        raise ProfileIdentityRefusal("declared profile ID does not match the measured runtime")
    if measured_version != _PROFILE_VERSIONS[declared_profile_id]:
        raise ProfileIdentityRefusal("measured MuJoCo version does not match the declared profile")
    runtime_python = _object(runtime.get("python"), "worker runtime python")
    python_identity = dict(runtime_python)
    python_identity["build"] = list(platform.python_build())
    runtime_mujoco = _object(runtime.get("mujoco"), "worker runtime mujoco")
    runtime_numpy = _object(runtime.get("numpy"), "worker runtime numpy")
    mujoco_identity = dict(runtime_mujoco)
    mujoco_identity["distribution"] = _compact_distribution(runtime_mujoco.get("distribution"))
    numpy_identity = dict(runtime_numpy)
    numpy_identity["distribution"] = _compact_distribution(runtime_numpy.get("distribution"))
    document: dict[str, Any] = {
        "schema": _PROFILE_IDENTITY_SCHEMA,
        "schema_version": _LEGACY_SCHEMA_VERSION,
        "profile_id": declared_profile_id,
        "profile_contract": {
            "headless": True,
            "mujoco_version": measured_version,
            "numpy_version": "2.3.5",
            "viewer_or_renderer_used": False,
            "worker_sha256": _LEGACY_FROZEN_WORKER_SHA256,
        },
        "python": python_identity,
        "host": runtime.get("host"),
        "environment": runtime.get("thread_environment"),
        "mujoco": mujoco_identity,
        "numpy": numpy_identity,
        "installation": runtime.get("installation"),
        "metrifid_installed": _metrifid_installation_is_measured(),
        "pip_check": runtime.get("pip_check"),
        "native_smoke": _smoke_identity(worker, manifest_path, fixture_id),
    }
    if _file_sha256(worker_path) != _LEGACY_FROZEN_WORKER_SHA256:
        raise ProfileIdentityRefusal("packaged native evidence worker changed during collection")
    document["profile_identity_sha256"] = _canonical_sha256(document)
    return _validate_profile_identity_v1(
        document,
        expected_profile_id=declared_profile_id,
        expected_worker_sha256=_LEGACY_FROZEN_WORKER_SHA256,
    )


def _sentinel_identity(
    worker: _FrozenWorker, manifest_path: Path, fixture_id: str, profile_role: str
) -> dict[str, Any]:
    """Compile the admitted fixture and execute one role-bound same-profile sentinel."""
    _manifest_raw, _manifest, fixture = worker._load_manifest(manifest_path, fixture_id)
    relative_xml = cast(str, fixture.xml_path)
    _, fixture_raw = worker._resolve_fixture_source(manifest_path, relative_xml)
    xml_text = worker._admit_self_contained_xml(fixture_raw)
    model = worker._compile_model(xml_text, fixture, _SMOKE_STEP)
    sentinel = worker._same_profile_sentinel(model, fixture, profile_role)
    return _validate_sentinel_v2(sentinel, profile_role, require_pass=False)


def _collect_identity_v2(
    worker_path: Path, manifest_path: Path, fixture_id: str, declared_profile_role: str
) -> dict[str, Any]:
    """Measure one version-generic production profile and complete-state sentinel."""
    if declared_profile_role not in _PROFILE_ROLES:
        raise ProfileIdentityRefusal("--profile-role is unsupported")
    if _IDENTIFIER_PATTERN.fullmatch(fixture_id) is None:
        raise ProfileIdentityRefusal("--fixture-id is invalid")
    if not manifest_path.is_absolute():
        raise ProfileIdentityRefusal("--manifest must be an absolute path")
    worker = _load_frozen_worker(worker_path)
    measured_role, package_version, runtime = worker._runtime_identity(
        declared_profile_role, allow_unbound_profile=True
    )
    if measured_role != declared_profile_role:
        raise ProfileIdentityRefusal("declared profile role differs from the measured runtime")
    runtime_python = _object(runtime.get("python"), "worker runtime python")
    python_identity = dict(runtime_python)
    python_identity["build"] = list(platform.python_build())
    runtime_mujoco = _object(runtime.get("mujoco"), "worker runtime mujoco")
    runtime_numpy = _object(runtime.get("numpy"), "worker runtime numpy")
    native_version = _string(runtime.get("native_version"), "worker native_version")
    native_integer = _integer(
        runtime.get("native_version_integer"), "worker native_version_integer", minimum=1
    )
    mujoco_identity = {
        "package_version": package_version,
        "native_version": native_version,
        "native_version_integer": native_integer,
        "distribution": _compact_distribution(runtime_mujoco.get("distribution")),
        "loaded_native_library": runtime_mujoco.get("loaded_native_library"),
    }
    numpy_identity = dict(runtime_numpy)
    numpy_identity["distribution"] = _compact_distribution(runtime_numpy.get("distribution"))
    sentinel = _sentinel_identity(worker, manifest_path, fixture_id, declared_profile_role)
    installation = _metrifid_installation_identity(worker)
    document: dict[str, Any] = {
        "schema": _PROFILE_IDENTITY_SCHEMA,
        "schema_version": _PRODUCTION_SCHEMA_VERSION,
        "profile_role": declared_profile_role,
        "package_version": package_version,
        "native_version": native_version,
        "native_version_integer": native_integer,
        "support_tier": "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE",
        "profile_contract": {
            "headless": True,
            "viewer_or_renderer_used": False,
            "worker_sha256": _FROZEN_WORKER_SHA256,
        },
        "python": python_identity,
        "host": runtime.get("host"),
        "environment": runtime.get("thread_environment"),
        "mujoco": mujoco_identity,
        "numpy": numpy_identity,
        "installation": installation,
        "metrifid_installed": installation["available"],
        "pip_check": runtime.get("pip_check"),
        "native_smoke": _smoke_identity(
            worker,
            manifest_path,
            fixture_id,
            include_sentinel_dimensions=True,
        ),
        "sentinel": sentinel,
    }
    if _file_sha256(worker_path) != _FROZEN_WORKER_SHA256:
        raise ProfileIdentityRefusal("packaged native evidence worker changed during collection")
    document["profile_identity_sha256"] = _canonical_sha256(document)
    return _validate_profile_identity_v2(
        document,
        expected_profile_role=declared_profile_role,
        expected_worker_sha256=_FROZEN_WORKER_SHA256,
        require_sentinel_pass=False,
    )


def _write_exclusive_json(path: Path, document: dict[str, Any]) -> None:
    """Write one canonical identity exclusively beneath an existing nonsymlink directory."""
    if not path.is_absolute():
        raise ProfileIdentityRefusal("--output must be an absolute path")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ProfileIdentityRefusal("--output parent must be an existing nonsymlink directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ProfileIdentityRefusal(
            "refusing to overwrite or create the profile identity"
        ) from exc
    try:
        payload = _canonical_json_bytes(document) + b"\n"
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    except OSError as exc:
        raise ProfileIdentityRefusal("profile identity could not be written completely") from exc
    finally:
        os.close(fd)
    directory_fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the exact private profile-identity collector invocation surface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixture-id", required=True)
    profile = parser.add_mutually_exclusive_group(required=True)
    profile.add_argument("--profile-id")
    profile.add_argument("--profile-role", choices=_PROFILE_ROLES)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Collect one identity or return one bounded refusal without exposing a traceback."""
    args = _arguments(argv)
    try:
        if args.profile_role is None:
            document = _collect_identity(
                cast(Path, args.worker),
                cast(Path, args.manifest),
                str(args.fixture_id),
                str(args.profile_id),
            )
        else:
            document = _collect_identity_v2(
                cast(Path, args.worker),
                cast(Path, args.manifest),
                str(args.fixture_id),
                str(args.profile_role),
            )
        _write_exclusive_json(cast(Path, args.output), document)
        sentinel = document.get("sentinel")
        if isinstance(sentinel, dict) and sentinel.get("status") != "PASS":
            raise ProfileIdentityRefusal("complete integration-state sentinel did not pass")
    except Exception as exc:
        sys.stderr.write(f"REFUSED: {type(exc).__name__}: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
