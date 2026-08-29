"""Behavioral tests for strict runtime-review configuration admission."""

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
    RuntimeReviewConfig,
    load_runtime_review_configuration,
)
from metrifid.runtime_review._paths import PathAdmissionError
from metrifid.runtime_review._status import (
    RuntimeReviewExitCode,
    RuntimeReviewReasonCode,
    RuntimeReviewStatus,
    runtime_review_exit_code,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _configuration_primitive() -> dict[str, Any]:
    """Return one complete valid declaration in deliberately noncanonical input order."""
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
        "schema_version": 1,
        "baseline_profile": {"profile_id": "A_3.10.0", "mujoco_version": "3.10.0"},
        "candidate_profile": {"profile_id": "B_3.11.0", "mujoco_version": "3.11.0"},
        "expected_subject": {
            "fixture_id": "declared_fixture",
            "source_closure_sha256": SHA_A,
            "fixture_manifest_sha256": SHA_B,
        },
        "expected_workload": {
            "semantic_sha256": SHA_C,
            "initial_state_semantic_sha256": SHA_D,
            "action_program_semantic_sha256": SHA_E,
        },
        "required_horizon": "1",
        "step_dts": ["0.001", "0.004", "0.002"],
        "repeat_ids": [1, 0],
        "cells": cells,
        "output_dir": "runtime_review_output",
    }


def _create_cell_directories(base: Path, primitive: dict[str, Any]) -> None:
    """Create each evidence directory declared by one test configuration."""
    for cell in primitive["cells"]:
        (base / cell["directory"]).mkdir(parents=True)


def _write_configuration(path: Path, primitive: dict[str, Any], *, indent: int | None = 2) -> bytes:
    """Write one JSON configuration and return its exact bytes."""
    payload = json.dumps(primitive, indent=indent, ensure_ascii=False).encode("utf-8") + b"\n"
    path.write_bytes(payload)
    return payload


def test_completed_statuses_map_to_their_frozen_exit_codes() -> None:
    """Every and only completed statuses and reasons belong to their closed registries."""
    assert tuple(RuntimeReviewStatus) == (
        RuntimeReviewStatus.WITHIN_DECLARED_MIGRATION_ENVELOPE,
        RuntimeReviewStatus.INSUFFICIENT_EVIDENCE,
        RuntimeReviewStatus.UNRESOLVED_NEAR_BOUNDARY,
        RuntimeReviewStatus.OUTSIDE_DECLARED_MIGRATION_ENVELOPE,
    )
    assert tuple(RuntimeReviewReasonCode) == (
        RuntimeReviewReasonCode.REPEATABILITY_FAILED,
        RuntimeReviewReasonCode.SOLVER_NOT_CONVERGED,
        RuntimeReviewReasonCode.CONTACT_EVENT_TOPOLOGY_CHANGED,
        RuntimeReviewReasonCode.NON_ASYMPTOTIC_REGIME,
        RuntimeReviewReasonCode.PREFIX_TOO_SHORT,
    )
    assert [runtime_review_exit_code(status) for status in RuntimeReviewStatus] == [
        RuntimeReviewExitCode.WITHIN_DECLARED_MIGRATION_ENVELOPE,
        RuntimeReviewExitCode.INSUFFICIENT_EVIDENCE,
        RuntimeReviewExitCode.UNRESOLVED_NEAR_BOUNDARY,
        RuntimeReviewExitCode.OUTSIDE_DECLARED_MIGRATION_ENVELOPE,
    ]
    with pytest.raises(TypeError, match="RuntimeReviewStatus"):
        runtime_review_exit_code("INSUFFICIENT_EVIDENCE")  # type: ignore[arg-type]


def test_loaded_configuration_retains_bytes_hashes_and_canonical_bindings(tmp_path: Path) -> None:
    """Admission retains raw identity while sorting every semantic campaign slot."""
    primitive = _configuration_primitive()
    _create_cell_directories(tmp_path, primitive)
    path = tmp_path / "runtime_review.json"
    raw = _write_configuration(path, primitive)

    admitted = load_runtime_review_configuration(path)

    assert admitted.raw_bytes == raw
    assert admitted.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert admitted.semantic_sha256 == canonical_sha256(admitted.config.to_primitive())
    assert [cell.slot for cell in admitted.config.cells] == [
        (role, step, repeat)
        for role in ("baseline", "candidate")
        for step in ("0.004", "0.002", "0.001")
        for repeat in (0, 1)
    ]
    assert admitted.config.step_dts == ("0.004", "0.002", "0.001")
    assert admitted.config.repeat_ids == (0, 1)
    assert tuple(path.name for path in admitted.cell_directories) == tuple(
        Path(cell.directory).name for cell in admitted.config.cells
    )
    assert admitted.output_dir == tmp_path / "runtime_review_output"


def test_input_order_and_format_do_not_change_configuration_semantics(tmp_path: Path) -> None:
    """Raw identities vary while semantically equal unordered declarations hash equally."""
    first = _configuration_primitive()
    _create_cell_directories(tmp_path, first)
    second = copy.deepcopy(first)
    second["cells"].reverse()
    second["step_dts"].reverse()
    second["repeat_ids"].reverse()
    first_raw = _write_configuration(tmp_path / "first.json", first, indent=2)
    second_raw = _write_configuration(tmp_path / "second.json", second, indent=None)

    first_admitted = load_runtime_review_configuration(tmp_path / "first.json")
    second_admitted = load_runtime_review_configuration(tmp_path / "second.json")

    assert first_raw != second_raw
    assert first_admitted.raw_sha256 != second_admitted.raw_sha256
    assert first_admitted.semantic_sha256 == second_admitted.semantic_sha256
    assert first_admitted.config.to_primitive() == second_admitted.config.to_primitive()


def test_unknown_missing_and_wrongly_typed_configuration_fields_refuse() -> None:
    """No field may be silently added, omitted, or coerced to the required type."""
    unknown = _configuration_primitive()
    unknown["optional"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        RuntimeReviewConfig.from_primitive(unknown)

    missing = _configuration_primitive()
    del missing["required_horizon"]
    with pytest.raises(ValueError, match="missing fields"):
        RuntimeReviewConfig.from_primitive(missing)

    boolean_integer = _configuration_primitive()
    boolean_integer["schema_version"] = True
    with pytest.raises(TypeError, match="integer and not a boolean"):
        RuntimeReviewConfig.from_primitive(boolean_integer)


def test_declared_profiles_horizon_steps_and_repeats_are_closed() -> None:
    """The first referee accepts only its exact profiles and complete sampling shape."""
    wrong_baseline = _configuration_primitive()
    wrong_baseline["baseline_profile"]["profile_id"] = "another_profile"
    with pytest.raises(ValueError, match="baseline_profile must be profile"):
        RuntimeReviewConfig.from_primitive(wrong_baseline)

    wrong_candidate = _configuration_primitive()
    wrong_candidate["candidate_profile"]["mujoco_version"] = "9.9.9"
    with pytest.raises(ValueError, match="candidate_profile must be profile"):
        RuntimeReviewConfig.from_primitive(wrong_candidate)

    wrong_horizon = _configuration_primitive()
    wrong_horizon["required_horizon"] = "1.0"
    with pytest.raises(ValueError, match="exact token"):
        RuntimeReviewConfig.from_primitive(wrong_horizon)

    missing_step = _configuration_primitive()
    missing_step["step_dts"] = ["0.004", "0.002"]
    with pytest.raises(ValueError, match="step_dts must contain exactly"):
        RuntimeReviewConfig.from_primitive(missing_step)

    duplicate_repeat = _configuration_primitive()
    duplicate_repeat["repeat_ids"] = [0, 0]
    with pytest.raises(ValueError, match="repeat_ids must contain exactly"):
        RuntimeReviewConfig.from_primitive(duplicate_repeat)


def test_campaign_requires_every_unique_slot_and_directory() -> None:
    """A campaign cannot omit, duplicate, or alias any role/step/repeat slot."""
    duplicate_slot = _configuration_primitive()
    duplicate_slot["cells"][-1] = copy.deepcopy(duplicate_slot["cells"][0])
    duplicate_slot["cells"][-1]["directory"] = "evidence/otherwise_distinct"
    with pytest.raises(ValueError, match="duplicate slot"):
        RuntimeReviewConfig.from_primitive(duplicate_slot)

    duplicate_directory = _configuration_primitive()
    duplicate_directory["cells"][-1]["directory"] = duplicate_directory["cells"][0]["directory"]
    with pytest.raises(ValueError, match="distinct directory"):
        RuntimeReviewConfig.from_primitive(duplicate_directory)


def test_configuration_json_rejects_duplicate_names_and_nonfinite_numbers(tmp_path: Path) -> None:
    """The file boundary keeps strict duplicate-key and nonfinite-number semantics."""
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema": "first", "schema": "second"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_runtime_review_configuration(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON numeric token"):
        load_runtime_review_configuration(nonfinite)


def test_evidence_directories_must_be_real_unlinked_and_confined(tmp_path: Path) -> None:
    """Traversal, symbolic links, and regular files cannot stand in for evidence directories."""
    traversal = _configuration_primitive()
    traversal["cells"][0]["directory"] = "../outside"
    with pytest.raises(PathAdmissionError, match="relative POSIX path"):
        RuntimeReviewConfig.from_primitive(traversal)

    primitive = _configuration_primitive()
    _create_cell_directories(tmp_path, primitive)
    symlink_cell = primitive["cells"][0]
    symlink_path = tmp_path / symlink_cell["directory"]
    symlink_path.rmdir()
    symlink_path.symlink_to(tmp_path / primitive["cells"][1]["directory"], target_is_directory=True)
    _write_configuration(tmp_path / "linked.json", primitive)
    with pytest.raises(PathAdmissionError, match="symbolic links"):
        load_runtime_review_configuration(tmp_path / "linked.json")

    symlink_path.unlink()
    symlink_path.write_text("not a directory", encoding="utf-8")
    with pytest.raises(PathAdmissionError, match="must name a directory"):
        load_runtime_review_configuration(tmp_path / "linked.json")


def test_output_must_be_new_and_disjoint_from_evidence(tmp_path: Path) -> None:
    """Publication cannot reuse or nest inside any declared evidence input."""
    primitive = _configuration_primitive()
    _create_cell_directories(tmp_path, primitive)
    primitive["output_dir"] = f"{primitive['cells'][0]['directory']}/review"
    _write_configuration(tmp_path / "overlap.json", primitive)
    with pytest.raises(PathAdmissionError, match="overlaps cell"):
        load_runtime_review_configuration(tmp_path / "overlap.json")

    primitive["output_dir"] = "already_present"
    (tmp_path / "already_present").mkdir()
    _write_configuration(tmp_path / "existing.json", primitive)
    with pytest.raises(PathAdmissionError, match="already exists"):
        load_runtime_review_configuration(tmp_path / "existing.json")
