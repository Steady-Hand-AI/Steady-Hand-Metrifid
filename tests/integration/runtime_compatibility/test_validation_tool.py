"""Integration tests for the external installed-profile compatibility validator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import mujoco  # type: ignore[import-untyped]
import pytest

from metrifid.model_release._public_field_registry_catalog import characterized_registry
from metrifid.model_release._snapshot import measure_public_field_surface

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TOOL_PATH = _REPOSITORY_ROOT / "tools" / "validate_mujoco_compatibility.py"


@pytest.fixture(scope="module")
def compatibility_tool() -> ModuleType:
    """Load the tracked external validator without changing the package import root."""
    specification = importlib.util.spec_from_file_location(
        "metrifid_compatibility_validation_tool", _TOOL_PATH
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _observation(
    *,
    support_tier: str = "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE",
    catalog_match: bool = True,
) -> dict[str, object]:
    """Build the smallest clean-process observation accepted by the result aggregator."""
    exact_tuple = {
        "package_version": "3.10.0",
        "package_base_version": "3.10.0",
        "mujoco_python_distribution_sha256": "a" * 64,
        "mujoco_record_sha256": "c" * 64,
        "native_version_string": "3.10.0",
        "native_version_integer": 3_010_000,
        "native_library_sha256": "b" * 64,
        "python": {
            "executable": "/profile/bin/python",
            "resolved_executable": "/usr/bin/python3.12",
            "implementation": "CPython",
            "version": "3.12.0",
            "build": ["main", "Aug 23 2026"],
            "cache_tag": "cpython-312",
            "compiler": "GCC",
        },
        "platform": {
            "system": "Linux",
            "machine": "x86_64",
            "release": "test-kernel",
            "libc": ["glibc", "2.39"],
        },
    }
    encoded = json.dumps(
        exact_tuple,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "runtime": {
            "package_version": "3.10.0",
            "support_tier": support_tier,
        },
        "exact_profile_tuple": exact_tuple,
        "exact_profile_tuple_sha256": hashlib.sha256(encoded).hexdigest(),
        "catalog": {
            "entry_present": True,
            "matches_observation": catalog_match,
        },
    }


def test_validator_help_is_callable_without_importing_the_checkout() -> None:
    """Keep the external tool discoverable before it imports an installed native package."""
    completed = subprocess.run(
        [sys.executable, str(_TOOL_PATH), "--help"],
        cwd=_TOOL_PATH.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0
    assert "--profile-role" in completed.stdout
    assert "--measurement-count" in completed.stdout
    assert "--expected-mujoco-version" in completed.stdout


def test_operation_inventory_is_complete_distinct_and_digestible(
    compatibility_tool: ModuleType,
) -> None:
    """Retain one deterministic inventory row for every owned native claim surface."""
    inventory = compatibility_tool._capability_inventory()
    assert set(inventory) == {"COMPILED_ARTIFACT", "STATIC_MODEL_REVIEW", "DYNAMIC_REPLAY"}
    for requirements in inventory.values():
        names = [row["name"] for row in requirements]
        assert names
        assert len(names) == len(set(names))
        assert {row["kind"] for row in requirements} <= {"ATTRIBUTE", "CALLABLE"}
    assert len(compatibility_tool._sha256(inventory)) == 64


def test_current_public_surface_reconstructs_its_exact_catalog_entry(
    compatibility_tool: ModuleType,
) -> None:
    """Measure the executing upstream model holder and match all four catalog coordinates."""
    model = mujoco.MjModel.from_xml_string(compatibility_tool._SURFACE_MODEL_XML)
    observed = measure_public_field_surface(model)
    expected = characterized_registry(mujoco.mj_versionString())
    assert expected is not None
    assert observed.opaque_member_paths == ()
    assert observed.full_public_surface_sha256 == expected.full_public_surface_sha256
    assert observed.full_public_surface_count == expected.full_public_surface_count
    assert observed.comparable_registry_sha256 == expected.comparable_registry_sha256
    assert observed.comparable_registry_count == expected.comparable_registry_count


def test_profile_result_requires_two_identical_clean_processes(
    compatibility_tool: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject single-process catalog claims and bind repeated bytes into one observation digest."""
    observation = _observation()

    def repeat_observation() -> dict[str, object]:
        """Return the same isolated observation for the aggregation seam."""
        return observation

    monkeypatch.setattr(compatibility_tool, "_clean_process_observation", repeat_observation)
    with pytest.raises(ValueError, match="at least two"):
        compatibility_tool.validate_profile(
            profile_role="prior_validated",
            expected_version="3.10.0",
            measurement_count=1,
        )
    result = compatibility_tool.validate_profile(
        profile_role="prior_validated",
        expected_version="3.10.0",
        measurement_count=2,
    )
    assert result["measurements_identical"] is True
    assert result["measurement_process_count"] == 2
    assert result["observation"] == observation
    assert len(str(result["observation_sha256"])) == 64
    retained = result["retained_exact_profile_validation"]
    assert isinstance(retained, Mapping)
    assert retained["validation_tier"] == "VALIDATED_EXACT_PROFILE"
    assert retained["live_product_support_tier"] == "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE"
    assert retained["exact_profile_tuple"] == observation["exact_profile_tuple"]


def test_exact_evidence_never_requires_live_exact_support_tier(
    compatibility_tool: ModuleType,
) -> None:
    """Keep exact retained tuple validation separate from live capability admission."""
    retained = compatibility_tool._validate_observation(_observation(), "3.10.0")
    assert retained["validation_tier"] == "VALIDATED_EXACT_PROFILE"
    assert retained["live_product_support_tier"] == "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE"
    with pytest.raises(
        compatibility_tool.CompatibilityValidationError,
        match="live product admission",
    ):
        compatibility_tool._validate_observation(
            _observation(support_tier="VALIDATED_EXACT_PROFILE"), "3.10.0"
        )


def test_semantic_profile_role_cannot_be_bound_to_another_exact_version(
    compatibility_tool: ModuleType,
) -> None:
    """Reject matrix evidence whose semantic profile role names another exact release."""
    with pytest.raises(ValueError, match="profile_role does not match"):
        compatibility_tool.validate_profile(
            profile_role="latest_validated",
            expected_version="3.10.0",
            measurement_count=2,
        )


def test_profile_result_rejects_divergent_or_uncharacterized_observations(
    compatibility_tool: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed when process observations drift or an exact matrix surface lacks a catalog."""
    observations = iter((_observation(), {**_observation(), "drift": True}))

    def next_observation() -> Mapping[str, object]:
        """Return the next deliberately divergent clean-process observation."""
        return next(observations)

    monkeypatch.setattr(compatibility_tool, "_clean_process_observation", next_observation)
    with pytest.raises(compatibility_tool.CompatibilityValidationError, match="not byte-identical"):
        compatibility_tool.validate_profile(
            profile_role="prior_validated",
            expected_version="3.10.0",
            measurement_count=2,
        )

    with pytest.raises(
        compatibility_tool.CompatibilityValidationError, match="characterized public-field"
    ):
        compatibility_tool._validate_observation(_observation(catalog_match=False), "3.10.0")

    changed_digest = _observation()
    changed_digest["exact_profile_tuple_sha256"] = "0" * 64
    with pytest.raises(
        compatibility_tool.CompatibilityValidationError, match="digest does not match"
    ):
        compatibility_tool._validate_observation(changed_digest, "3.10.0")

    incomplete_tuple = _observation()
    exact_tuple = incomplete_tuple["exact_profile_tuple"]
    assert isinstance(exact_tuple, dict)
    exact_tuple.pop("native_library_sha256")
    incomplete_tuple["exact_profile_tuple_sha256"] = compatibility_tool._sha256(exact_tuple)
    with pytest.raises(
        compatibility_tool.CompatibilityValidationError, match="incomplete field set"
    ):
        compatibility_tool._validate_observation(incomplete_tuple, "3.10.0")


def test_validator_publishes_fresh_canonical_json_without_clobbering(
    compatibility_tool: ModuleType, tmp_path: Path
) -> None:
    """Write one newline-terminated result and preserve it against a second publication."""
    result = {"schema": "test", "passed": True}
    destination = compatibility_tool._write_result(tmp_path, result)
    assert destination == tmp_path / "compatibility_validation.json"
    assert destination.read_bytes() == compatibility_tool._canonical_bytes(result) + b"\n"
    with pytest.raises(compatibility_tool.CompatibilityValidationError, match="replace"):
        compatibility_tool._write_result(tmp_path, result)
