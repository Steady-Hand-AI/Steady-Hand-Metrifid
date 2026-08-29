"""Behavioral tests for role-based Runtime Review configuration admission."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from metrifid.json_values import canonical_sha256
from metrifid.runtime_review._config import (
    CONFIG_SCHEMA,
    AdmittedRuntimeReviewConfigurationV2,
    RuntimeReviewConfigV2,
    load_runtime_review_configuration,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
SHA_0 = "0" * 64


def _role_configuration_primitive() -> dict[str, Any]:
    """Return one complete role-based declaration with noncanonical campaign order."""
    cells = [
        {
            "profile_role": role,
            "step_dt": step,
            "repeat_id": repeat,
            "directory": f"evidence/{role}_{step.replace('.', 'p')}_repeat_{repeat}",
        }
        for role in ("candidate", "baseline")
        for step in ("0.001", "0.002", "0.004")
        for repeat in (1, 0)
    ]
    return {
        "schema": CONFIG_SCHEMA,
        "schema_version": 2,
        "baseline_profile": {
            "profile_role": "baseline",
            "package_version": "3.12.0.post2+vendor.1",
            "native_version": "3.12.0",
            "native_version_integer": 3_012_000,
            "profile_identity_sha256": SHA_A,
            "identity_file": "profile_identities/baseline.json",
        },
        "candidate_profile": {
            "profile_role": "candidate",
            "package_version": "3.12.0+candidate.1",
            "native_version": "3.12.0",
            "native_version_integer": 3_012_000,
            "profile_identity_sha256": SHA_B,
            "identity_file": "profile_identities/candidate.json",
        },
        "expected_subject": {
            "fixture_id": "declared_fixture",
            "source_closure_sha256": SHA_C,
            "fixture_manifest_sha256": SHA_D,
        },
        "expected_workload": {
            "semantic_sha256": SHA_E,
            "initial_state_semantic_sha256": SHA_F,
            "action_program_semantic_sha256": SHA_0,
        },
        "required_horizon": "1",
        "step_dts": ["0.001", "0.004", "0.002"],
        "repeat_ids": [1, 0],
        "cells": cells,
        "output_dir": "runtime_review_output",
    }


def _materialize_inputs(base: Path, primitive: dict[str, Any]) -> None:
    """Create every declared cell directory and exact profile identity file."""
    for cell in primitive["cells"]:
        (base / cell["directory"]).mkdir(parents=True)
    identities = base / "profile_identities"
    identities.mkdir()
    (identities / "baseline.json").write_bytes(b"{}\n")
    (identities / "candidate.json").write_bytes(b"{}\n")


def _write_configuration(path: Path, primitive: dict[str, Any]) -> bytes:
    """Write one strict JSON declaration and return its exact bytes."""
    payload = json.dumps(primitive, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    path.write_bytes(payload)
    return payload


def test_role_configuration_dispatch_binds_exact_identity_files(tmp_path: Path) -> None:
    """Dispatch the role route and retain bytes, semantics, slots, and profile files."""
    primitive = _role_configuration_primitive()
    _materialize_inputs(tmp_path, primitive)
    path = tmp_path / "runtime_review.json"
    raw = _write_configuration(path, primitive)

    admitted = load_runtime_review_configuration(path)

    assert isinstance(admitted, AdmittedRuntimeReviewConfigurationV2)
    assert isinstance(admitted.config, RuntimeReviewConfigV2)
    assert admitted.raw_bytes == raw
    assert admitted.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert admitted.semantic_sha256 == canonical_sha256(admitted.config.to_primitive())
    assert admitted.profile_identity_paths == (
        tmp_path / "profile_identities" / "baseline.json",
        tmp_path / "profile_identities" / "candidate.json",
    )
    identity_file_sha256 = hashlib.sha256(b"{}\n").hexdigest()
    assert admitted.profile_identity_file_sha256 == (
        identity_file_sha256,
        identity_file_sha256,
    )
    assert admitted.profile_identity_path("baseline").name == "baseline.json"
    assert admitted.profile_identity_file_hash("candidate") == identity_file_sha256
    assert [cell.slot for cell in admitted.config.cells] == [
        (role, step, repeat)
        for role in ("baseline", "candidate")
        for step in ("0.004", "0.002", "0.001")
        for repeat in (0, 1)
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("profile_role", "candidate", "identity_file for candidate"),
        ("package_version", "3.12.0rc1", "bounded stable version grammar"),
        ("native_version", "3.11.0", "base triplet"),
        ("native_version_integer", 3_011_000, "encode the exact"),
        ("profile_identity_sha256", "not-a-hash", "lowercase hexadecimal"),
        ("identity_file", "profile_identities/other.json", "identity_file for baseline"),
    ],
)
def test_role_profile_refuses_incoherent_identity_declarations(
    field: str, value: object, message: str
) -> None:
    """Reject role, version, hash, and locator substitutions before filesystem use."""
    primitive = _role_configuration_primitive()
    primitive["baseline_profile"][field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        RuntimeReviewConfigV2.from_primitive(primitive)


@pytest.mark.parametrize("schema_version", [0, 3, True, "2"])
def test_configuration_dispatch_refuses_unknown_or_ambiguous_versions(
    tmp_path: Path, schema_version: object
) -> None:
    """Never guess a route for unsupported or wrongly typed schema-version tokens."""
    primitive = _role_configuration_primitive()
    primitive["schema_version"] = schema_version
    path = tmp_path / "runtime_review.json"
    _write_configuration(path, primitive)

    with pytest.raises((TypeError, ValueError), match="schema_version"):
        load_runtime_review_configuration(path)


def test_role_identity_file_must_be_regular_confined_and_unlinked(tmp_path: Path) -> None:
    """Reject a missing or linked profile identity before review configuration admission."""
    primitive = _role_configuration_primitive()
    _materialize_inputs(tmp_path, primitive)
    baseline = tmp_path / "profile_identities" / "baseline.json"
    baseline.unlink()
    baseline.symlink_to(tmp_path / "profile_identities" / "candidate.json")
    path = tmp_path / "runtime_review.json"
    _write_configuration(path, primitive)

    with pytest.raises(ValueError, match="regular no-follow file"):
        load_runtime_review_configuration(path)

    baseline.unlink()
    with pytest.raises(ValueError, match="regular no-follow file"):
        load_runtime_review_configuration(path)


def test_role_and_campaign_fields_remain_closed(tmp_path: Path) -> None:
    """Reject unknown role fields and duplicate campaign slots on the new route."""
    primitive = _role_configuration_primitive()
    unknown = copy.deepcopy(primitive)
    unknown["candidate_profile"]["optional"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        RuntimeReviewConfigV2.from_primitive(unknown)

    duplicate = copy.deepcopy(primitive)
    duplicate["cells"][-1] = copy.deepcopy(duplicate["cells"][0])
    duplicate["cells"][-1]["directory"] = "evidence/otherwise_distinct"
    with pytest.raises(ValueError, match="duplicate slot"):
        RuntimeReviewConfigV2.from_primitive(duplicate)
