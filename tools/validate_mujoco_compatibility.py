#!/usr/bin/env python3
"""Measure one installed MuJoCo compatibility profile in independent clean processes.

The validator is intentionally an external release harness rather than an installed command. Run it
with the Python interpreter that already contains the exact upstream MuJoCo distribution and the
noneditable candidate wheel. It verifies the executing Metrifid distribution, admits every owned
native claim surface, measures the complete public ``MjModel`` surface, and requires repeated
clean-process observations to be byte-identical. Live product admission remains capability-
compatible; the result separately records the exact tuple validated by this retained evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

_SCHEMA: Final = "metrifid.validation.mujoco_compatibility_profile"
_SCHEMA_VERSION: Final = 2
_OBSERVATION_SCHEMA: Final = "metrifid.validation.mujoco_compatibility_observation"
_OBSERVATION_SCHEMA_VERSION: Final = 2
_EXACT_VALIDATION_SCHEMA: Final = "metrifid.validation.retained_exact_mujoco_profile"
_EXACT_VALIDATION_SCHEMA_VERSION: Final = 1
_OUTPUT_NAME: Final = "compatibility_validation.json"
_PROFILE_VERSIONS: Final = {
    "oldest_validated": "3.9.0",
    "prior_validated": "3.10.0",
    "current_validated": "3.11.0",
    "latest_validated": "3.12.0",
}
_PROFILE_ROLES: Final = tuple(_PROFILE_VERSIONS)
_EXACT_TUPLE_FIELDS: Final = frozenset(
    {
        "package_version",
        "package_base_version",
        "mujoco_python_distribution_sha256",
        "mujoco_record_sha256",
        "native_version_string",
        "native_version_integer",
        "native_library_sha256",
        "python",
        "platform",
    }
)
_EXACT_PYTHON_FIELDS: Final = frozenset(
    {
        "executable",
        "resolved_executable",
        "implementation",
        "version",
        "build",
        "cache_tag",
        "compiler",
    }
)
_EXACT_PLATFORM_FIELDS: Final = frozenset({"system", "machine", "release", "libc"})
_SURFACE_MODEL_XML: Final = """<mujoco model="compatibility_surface">
  <option timestep="0.001"/>
  <worldbody>
    <body name="link" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="shape" type="capsule" size="0.04 0.2" mass="1.5"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="hinge" gear="1"/>
  </actuator>
</mujoco>
"""


class CompatibilityValidationError(RuntimeError):
    """Report one deterministic validation failure suitable for retained matrix evidence."""


def _canonical_bytes(value: object) -> bytes:
    """Encode one JSON-compatible value using stable release-evidence settings."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    """Return the canonical JSON digest of one validation value."""
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _capability_inventory() -> dict[str, list[dict[str, str]]]:
    """Project the private operation inventory into deterministic machine evidence."""
    from metrifid._mujoco_runtime import MUJOCO_CAPABILITY_INVENTORY

    return {
        operation.value: [
            {"kind": requirement.kind, "name": requirement.name} for requirement in requirements
        ]
        for operation, requirements in sorted(
            MUJOCO_CAPABILITY_INVENTORY.items(), key=lambda item: item[0].value
        )
    }


def _distribution_record_sha256(distribution_name: str) -> str:
    """Hash the exact installed wheel RECORD bytes for one measured distribution."""
    distribution = metadata.distribution(distribution_name)
    files = distribution.files
    if files is None:
        raise CompatibilityValidationError(
            f"{distribution_name} distribution has no installed member manifest"
        )
    records = [item for item in files if str(item).endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise CompatibilityValidationError(
            f"{distribution_name} distribution must expose exactly one RECORD member"
        )
    record = Path(str(distribution.locate_file(records[0])))
    if record.is_symlink() or not record.is_file():
        raise CompatibilityValidationError(
            f"{distribution_name} distribution RECORD must be a regular nonsymlink file"
        )
    try:
        payload = record.read_bytes()
    except OSError as exc:
        raise CompatibilityValidationError(
            f"{distribution_name} distribution RECORD could not be read"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _runtime_observation() -> dict[str, object]:
    """Measure runtime, capability, feature, and public-surface facts in this process."""
    import mujoco  # type: ignore[import-untyped]

    import metrifid
    from metrifid._mujoco_runtime import (
        MujocoClaimSurface,
        admit_model_feature_coverage,
        admit_mujoco_runtime,
    )
    from metrifid.compare._environment import (
        mujoco_distribution_payload_sha256,
        native_mujoco_library_sha256,
    )
    from metrifid.distribution import installed_distribution_sha256
    from metrifid.model_release._public_field_registry_catalog import characterized_registry
    from metrifid.model_release._snapshot import measure_public_field_surface

    inventory = _capability_inventory()
    admissions = {
        operation.value: admit_mujoco_runtime(operation) for operation in MujocoClaimSurface
    }
    identities = {
        (
            admission.package_version,
            admission.package_base_version,
            admission.native_version_string,
            admission.native_version_integer,
            admission.support_tier.value,
        )
        for admission in admissions.values()
    }
    if len(identities) != 1:
        raise CompatibilityValidationError(
            "operation-specific admission returned inconsistent runtime identities"
        )
    package_version, base_version, native_string, native_integer, support_tier = identities.pop()
    model = mujoco.MjModel.from_xml_string(_SURFACE_MODEL_XML)
    static_facts = admit_model_feature_coverage(
        model, admissions[MujocoClaimSurface.STATIC_MODEL_REVIEW.value], "comparison"
    )
    dynamic_facts = admit_model_feature_coverage(
        model, admissions[MujocoClaimSurface.DYNAMIC_REPLAY.value], "comparison"
    )
    surface = measure_public_field_surface(model)
    expected = characterized_registry(base_version)
    catalog_match = expected is not None and (
        surface.full_public_surface_sha256 == expected.full_public_surface_sha256
        and surface.full_public_surface_count == expected.full_public_surface_count
        and surface.comparable_registry_sha256 == expected.comparable_registry_sha256
        and surface.comparable_registry_count == expected.comparable_registry_count
    )
    exact_profile_tuple = {
        "package_version": package_version,
        "package_base_version": base_version,
        "mujoco_python_distribution_sha256": mujoco_distribution_payload_sha256(),
        "mujoco_record_sha256": _distribution_record_sha256("mujoco"),
        "native_version_string": native_string,
        "native_version_integer": native_integer,
        "native_library_sha256": native_mujoco_library_sha256(),
        "python": {
            "executable": sys.executable,
            "resolved_executable": str(Path(sys.executable).resolve()),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "build": list(platform.python_build()),
            "cache_tag": sys.implementation.cache_tag,
            "compiler": platform.python_compiler(),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
            "libc": list(platform.libc_ver()),
        },
    }
    return {
        "schema": _OBSERVATION_SCHEMA,
        "schema_version": _OBSERVATION_SCHEMA_VERSION,
        "python": {
            "executable": sys.executable,
            "resolved_executable": str(Path(sys.executable).resolve()),
            "implementation": sys.implementation.name,
            "version": ".".join(str(component) for component in sys.version_info[:3]),
        },
        "metrifid": {
            "distribution_sha256": installed_distribution_sha256(),
            "import_path": str(Path(metrifid.__file__).resolve()),
            "version": metrifid.__version__,
        },
        "runtime": {
            "package_version": package_version,
            "package_base_version": base_version,
            "native_version_string": native_string,
            "native_version_integer": native_integer,
            "support_tier": support_tier,
        },
        "exact_profile_tuple": exact_profile_tuple,
        "exact_profile_tuple_sha256": _sha256(exact_profile_tuple),
        "operation_admissions": {
            operation: admission.to_evidence()
            for operation, admission in sorted(admissions.items())
        },
        "capability_inventory": inventory,
        "capability_inventory_sha256": _sha256(inventory),
        "siso_feature_facts": {
            "static_model_review": static_facts.to_evidence(),
            "dynamic_replay": dynamic_facts.to_evidence(),
        },
        "public_surface": surface.identity_primitive(),
        "catalog": {
            "entry_present": expected is not None,
            "matches_observation": catalog_match,
            "expected": (
                None
                if expected is None
                else {
                    "runtime_base_version": expected.base_version,
                    "full_public_surface_sha256": expected.full_public_surface_sha256,
                    "full_public_surface_count": expected.full_public_surface_count,
                    "comparable_registry_sha256": expected.comparable_registry_sha256,
                    "comparable_registry_count": expected.comparable_registry_count,
                    "measurement_process_count": expected.measurement_process_count,
                    "measurements_identical": expected.measurements_identical,
                }
            ),
        },
    }


def _child_command() -> list[str]:
    """Return the exact clean-process observation command for this source file."""
    return [sys.executable, str(Path(__file__).resolve()), "--measurement-child"]


def _clean_process_observation() -> dict[str, object]:
    """Run one isolated observation and decode its single canonical JSON document."""
    completed = subprocess.run(
        _child_command(),
        cwd=Path(os.devnull).parent,
        env=os.environ.copy(),
        capture_output=True,
        check=False,
        timeout=180,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise CompatibilityValidationError(
            f"clean-process measurement exited {completed.returncode}: {stderr}"
        )
    try:
        decoded = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompatibilityValidationError(
            "clean-process measurement did not emit one UTF-8 JSON document"
        ) from exc
    if type(decoded) is not dict:
        raise CompatibilityValidationError("clean-process measurement is not a JSON object")
    return decoded


def _validate_observation(
    observation: Mapping[str, object], expected_version: str | None
) -> dict[str, object]:
    """Validate one measured tuple without upgrading the live product support tier."""
    runtime = observation.get("runtime")
    catalog = observation.get("catalog")
    exact_tuple = observation.get("exact_profile_tuple")
    exact_tuple_sha256 = observation.get("exact_profile_tuple_sha256")
    if (
        not isinstance(runtime, Mapping)
        or not isinstance(catalog, Mapping)
        or not isinstance(exact_tuple, Mapping)
    ):
        raise CompatibilityValidationError(
            "measurement omitted runtime, exact-tuple, or catalog evidence"
        )
    package_version = runtime.get("package_version")
    if expected_version is not None and package_version != expected_version:
        raise CompatibilityValidationError(
            f"expected MuJoCo {expected_version}, observed {package_version!r}"
        )
    if runtime.get("support_tier") != "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE":
        raise CompatibilityValidationError(
            "live product admission did not report ADMITTED_CAPABILITY_COMPATIBLE_PROFILE"
        )
    if exact_tuple.get("package_version") != package_version:
        raise CompatibilityValidationError(
            "exact profile tuple does not bind the admitted package version"
        )
    if set(exact_tuple) != _EXACT_TUPLE_FIELDS:
        raise CompatibilityValidationError("exact profile tuple has an incomplete field set")
    exact_python = exact_tuple.get("python")
    exact_platform = exact_tuple.get("platform")
    if (
        not isinstance(exact_python, Mapping)
        or set(exact_python) != _EXACT_PYTHON_FIELDS
        or not isinstance(exact_platform, Mapping)
        or set(exact_platform) != _EXACT_PLATFORM_FIELDS
    ):
        raise CompatibilityValidationError(
            "exact profile tuple has an incomplete Python or platform field set"
        )
    for field in (
        "mujoco_python_distribution_sha256",
        "mujoco_record_sha256",
        "native_library_sha256",
    ):
        digest = exact_tuple.get(field)
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise CompatibilityValidationError(f"exact profile tuple has invalid {field}")
    measured_tuple_sha256 = _sha256(dict(exact_tuple))
    if exact_tuple_sha256 != measured_tuple_sha256:
        raise CompatibilityValidationError("exact profile tuple digest does not match its bytes")
    if catalog.get("entry_present") is not True or catalog.get("matches_observation") is not True:
        raise CompatibilityValidationError(
            "installed exact matrix profile does not match its characterized public-field catalog"
        )
    return {
        "schema": _EXACT_VALIDATION_SCHEMA,
        "schema_version": _EXACT_VALIDATION_SCHEMA_VERSION,
        "validation_tier": "VALIDATED_EXACT_PROFILE",
        "exact_profile_tuple": dict(exact_tuple),
        "exact_profile_tuple_sha256": measured_tuple_sha256,
        "expected_package_version": expected_version,
        "expected_package_version_matches": True,
        "characterized_public_field_catalog_matches": True,
        "live_product_support_tier": runtime["support_tier"],
    }


def validate_profile(
    *,
    profile_role: str,
    expected_version: str | None,
    measurement_count: int,
) -> dict[str, object]:
    """Return one complete compatibility result after repeated clean-process measurement."""
    if profile_role not in _PROFILE_ROLES:
        raise ValueError("profile_role is not a recognized semantic matrix role")
    if expected_version != _PROFILE_VERSIONS[profile_role]:
        raise ValueError("profile_role does not match the exact preregistered MuJoCo version")
    if measurement_count < 2:
        raise ValueError("measurement_count must be at least two")
    observations = [_clean_process_observation() for _index in range(measurement_count)]
    encoded = [_canonical_bytes(observation) for observation in observations]
    measurements_identical = all(item == encoded[0] for item in encoded[1:])
    if not measurements_identical:
        raise CompatibilityValidationError(
            "independent clean-process compatibility measurements were not byte-identical"
        )
    observation = observations[0]
    retained_exact_validation = _validate_observation(observation, expected_version)
    return {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "profile_role": profile_role,
        "expected_mujoco_package_version": expected_version,
        "measurement_process_count": measurement_count,
        "measurements_identical": True,
        "observation_sha256": hashlib.sha256(encoded[0]).hexdigest(),
        "observation": observation,
        "retained_exact_profile_validation": retained_exact_validation,
        "passed": True,
    }


def _write_result(output: Path, result: Mapping[str, object]) -> Path:
    """Publish one fresh canonical result without replacing prior matrix evidence."""
    output.mkdir(parents=True, exist_ok=True)
    destination = output / _OUTPUT_NAME
    if destination.exists():
        raise CompatibilityValidationError(f"refusing to replace existing {destination}")
    payload = _canonical_bytes(result) + b"\n"
    temporary = output / f".{_OUTPUT_NAME}.{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _parser() -> argparse.ArgumentParser:
    """Build the external validation harness argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-role", choices=_PROFILE_ROLES)
    parser.add_argument("--expected-mujoco-version")
    parser.add_argument("--measurement-count", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--measurement-child", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one child observation or publish one repeated installed-profile result."""
    arguments = _parser().parse_args(None if argv is None else list(argv))
    if arguments.measurement_child:
        sys.stdout.buffer.write(_canonical_bytes(_runtime_observation()) + b"\n")
        return 0
    if (
        arguments.profile_role is None
        or arguments.expected_mujoco_version is None
        or arguments.output is None
    ):
        _parser().error("--profile-role, --expected-mujoco-version, and --output are required")
    output = arguments.output.expanduser().resolve()
    try:
        result = validate_profile(
            profile_role=arguments.profile_role,
            expected_version=arguments.expected_mujoco_version,
            measurement_count=arguments.measurement_count,
        )
        destination = _write_result(output, result)
    except (CompatibilityValidationError, OSError, subprocess.SubprocessError, ValueError) as exc:
        failure = {
            "schema": _SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "profile_role": arguments.profile_role,
            "expected_mujoco_package_version": arguments.expected_mujoco_version,
            "measurement_process_count": arguments.measurement_count,
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        try:
            _write_result(output, failure)
        except (CompatibilityValidationError, OSError):
            pass
        sys.stderr.buffer.write(_canonical_bytes(failure) + b"\n")
        return 1
    sys.stdout.buffer.write(_canonical_bytes({"passed": True, "result": str(destination)}) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
