"""Hostile boundary tests for strict Runtime Review execution configuration admission."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from metrifid._json_admission import JsonAdmissionError
from metrifid.runtime_review._execution_config import (
    MANIFEST_MAX_BYTES,
    RUN_CONFIG_SCHEMA,
    load_runtime_review_run_configuration,
    recheck_runtime_review_run_configuration,
    recheck_runtime_review_run_inputs,
)
from metrifid.runtime_review._paths import PathAdmissionError


def _make_executable(path: Path, marker: str) -> Path:
    """Create one test-local executable regular file with distinct stable bytes."""
    path.write_text(f"#!/bin/sh\n# {marker}\nexit 0\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def _valid_primitive(tmp_path: Path) -> dict[str, Any]:
    """Create filesystem inputs and return one complete valid declaration."""
    baseline = _make_executable(tmp_path / "baseline-python", "baseline")
    candidate = _make_executable(tmp_path / "candidate-python", "candidate")
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    return {
        "schema": RUN_CONFIG_SCHEMA,
        "schema_version": 1,
        "baseline_python": baseline.as_posix(),
        "candidate_python": candidate.as_posix(),
        "manifest": "manifest.json",
        "fixture_id": "smooth_pendulum",
        "output_dir": "runtime_review_run_output",
    }


def _write_config(path: Path, primitive: dict[str, Any]) -> bytes:
    """Write one JSON declaration and return its exact noncanonical input bytes."""
    raw = json.dumps(primitive, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    path.write_bytes(raw)
    return raw


def test_valid_run_configuration_is_admitted(tmp_path: Path) -> None:
    """The exact closed schema retains raw, semantic, interpreter, and manifest identities."""
    primitive = _valid_primitive(tmp_path)
    raw = _write_config(tmp_path / "run.json", primitive)

    admitted = load_runtime_review_run_configuration(tmp_path / "run.json")

    assert admitted.raw_bytes == raw
    assert admitted.config.to_primitive() == primitive
    assert admitted.baseline_interpreter.lexical_path == Path(primitive["baseline_python"])
    assert admitted.candidate_interpreter.lexical_path == Path(primitive["candidate_python"])
    assert admitted.manifest_path == tmp_path / "manifest.json"
    assert admitted.output_dir == tmp_path / "runtime_review_run_output"
    recheck_runtime_review_run_configuration(admitted)


def test_unknown_run_configuration_field_is_refused(tmp_path: Path) -> None:
    """A command, environment, or any other undeclared field cannot enter execution."""
    primitive = _valid_primitive(tmp_path)
    primitive["command"] = "python user_script.py"
    _write_config(tmp_path / "run.json", primitive)

    with pytest.raises(ValueError, match="unknown fields"):
        load_runtime_review_run_configuration(tmp_path / "run.json")


def test_duplicate_run_configuration_field_is_refused(tmp_path: Path) -> None:
    """Duplicate JSON object names refuse rather than choosing one interpreter spelling."""
    _valid_primitive(tmp_path)
    raw = (
        b'{"schema":"metrifid.runtime_review_run_config","schema_version":1,'
        b'"baseline_python":"/first","baseline_python":"/second"}'
    )
    (tmp_path / "run.json").write_bytes(raw)

    with pytest.raises(JsonAdmissionError, match="duplicate JSON object key"):
        load_runtime_review_run_configuration(tmp_path / "run.json")


def test_relative_profile_interpreter_is_refused(tmp_path: Path) -> None:
    """Neither role may search PATH or interpret a relative launcher declaration."""
    primitive = _valid_primitive(tmp_path)
    primitive["baseline_python"] = "python"
    _write_config(tmp_path / "run.json", primitive)

    with pytest.raises(PathAdmissionError, match="absolute POSIX path"):
        load_runtime_review_run_configuration(tmp_path / "run.json")


def test_existing_output_directory_is_refused(tmp_path: Path) -> None:
    """An existing output root is caller data and is never reused or overwritten."""
    primitive = _valid_primitive(tmp_path)
    (tmp_path / primitive["output_dir"]).mkdir()
    _write_config(tmp_path / "run.json", primitive)

    with pytest.raises(PathAdmissionError, match="already exists"):
        load_runtime_review_run_configuration(tmp_path / "run.json")


def test_manifest_traversal_is_refused(tmp_path: Path) -> None:
    """A manifest locator cannot escape above its configuration directory."""
    primitive = _valid_primitive(tmp_path)
    primitive["manifest"] = "../manifest.json"
    _write_config(tmp_path / "run.json", primitive)

    with pytest.raises(PathAdmissionError, match="relative POSIX path"):
        load_runtime_review_run_configuration(tmp_path / "run.json")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing fields"),
        ("wrong_type", "must be a string"),
        ("same_launcher", "lexically distinct"),
        ("bad_fixture", "identifier grammar"),
        ("double_slash", "absolute POSIX path"),
    ],
    ids=(
        "missing-required-field",
        "wrong-field-type",
        "identical-launcher-spelling",
        "invalid-fixture-identifier",
        "noncanonical-absolute-launcher",
    ),
)
def test_closed_configuration_rejects_hostile_semantics(
    tmp_path: Path, mutation: str, message: str
) -> None:
    """Missing, typed, aliasing, identifier, and noncanonical path variants all refuse."""
    primitive = _valid_primitive(tmp_path)
    if mutation == "missing":
        del primitive["fixture_id"]
    elif mutation == "wrong_type":
        primitive["fixture_id"] = 7
    elif mutation == "same_launcher":
        primitive["candidate_python"] = primitive["baseline_python"]
    elif mutation == "bad_fixture":
        primitive["fixture_id"] = "Invalid Fixture"
    else:
        primitive["baseline_python"] = f"/{primitive['baseline_python']}"
    _write_config(tmp_path / "run.json", primitive)

    with pytest.raises((TypeError, ValueError), match=message):
        load_runtime_review_run_configuration(tmp_path / "run.json")


def test_strict_json_boundary_refuses_non_utf8_nonfinite_and_deep_input(tmp_path: Path) -> None:
    """Text decoding, numeric grammar, and nesting bounds fail before schema admission."""
    hostile = (
        b"\xff",
        b'{"value":NaN}',
        (b'{"nested":' * 10) + b"null" + (b"}" * 10),
    )
    for index, raw in enumerate(hostile):
        path = tmp_path / f"hostile-{index}.json"
        path.write_bytes(raw)
        with pytest.raises(JsonAdmissionError):
            load_runtime_review_run_configuration(path)


def test_missing_and_nonexecutable_interpreters_are_refused(tmp_path: Path) -> None:
    """Explicit launchers must already exist and resolve to executable regular files."""
    primitive = _valid_primitive(tmp_path)
    primitive["baseline_python"] = (tmp_path / "missing-python").as_posix()
    _write_config(tmp_path / "missing.json", primitive)
    with pytest.raises(PathAdmissionError, match="existing executable"):
        load_runtime_review_run_configuration(tmp_path / "missing.json")

    primitive = _valid_primitive(tmp_path)
    Path(primitive["baseline_python"]).chmod(0o600)
    _write_config(tmp_path / "nonexecutable.json", primitive)
    with pytest.raises(PathAdmissionError, match="executable regular file"):
        load_runtime_review_run_configuration(tmp_path / "nonexecutable.json")


def test_final_virtual_environment_launcher_symlink_is_identity_bound(tmp_path: Path) -> None:
    """A normal final launcher symlink is accepted while its lexical target is recorded."""
    primitive = _valid_primitive(tmp_path)
    target = Path(primitive["baseline_python"])
    launcher = tmp_path / "baseline-link"
    launcher.symlink_to(target.name)
    primitive["baseline_python"] = launcher.as_posix()
    _write_config(tmp_path / "run.json", primitive)

    admitted = load_runtime_review_run_configuration(tmp_path / "run.json")

    assert admitted.baseline_interpreter.lexical_kind == "symbolic_link"
    assert admitted.baseline_interpreter.lexical_link_target == target.name
    assert admitted.baseline_interpreter.resolved_path == target


def test_manifest_symlinks_nonregular_files_and_oversize_bytes_are_refused(
    tmp_path: Path,
) -> None:
    """Manifest admission never follows links, opens devices, or exceeds worker bounds."""
    primitive = _valid_primitive(tmp_path)
    manifest = tmp_path / "manifest.json"
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    manifest.unlink()
    manifest.symlink_to(target.name)
    _write_config(tmp_path / "linked.json", primitive)
    with pytest.raises(PathAdmissionError, match="regular file"):
        load_runtime_review_run_configuration(tmp_path / "linked.json")

    manifest.unlink()
    manifest.mkdir()
    with pytest.raises(PathAdmissionError, match="regular file"):
        load_runtime_review_run_configuration(tmp_path / "linked.json")

    manifest.rmdir()
    with manifest.open("wb") as stream:
        stream.truncate(MANIFEST_MAX_BYTES + 1)
    with pytest.raises(PathAdmissionError, match="byte admission limit"):
        load_runtime_review_run_configuration(tmp_path / "linked.json")


def test_manifest_and_configuration_mutations_are_detected_after_admission(
    tmp_path: Path,
) -> None:
    """Final input replay refuses changes to either retained manifest or run declaration."""
    primitive = _valid_primitive(tmp_path)
    config_path = tmp_path / "run.json"
    original = _write_config(config_path, primitive)
    admitted = load_runtime_review_run_configuration(config_path)

    (tmp_path / "manifest.json").write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(PathAdmissionError, match="manifest identity changed"):
        recheck_runtime_review_run_inputs(admitted)

    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    config_path.write_bytes(original + b" ")
    with pytest.raises(PathAdmissionError, match="configuration bytes changed"):
        recheck_runtime_review_run_inputs(admitted)


def test_output_traversal_symlink_and_existing_file_are_refused(tmp_path: Path) -> None:
    """Every ambiguous output spelling or preexisting caller-owned member refuses."""
    primitive = _valid_primitive(tmp_path)
    primitive["output_dir"] = "../outside"
    _write_config(tmp_path / "traversal.json", primitive)
    with pytest.raises(PathAdmissionError, match="relative POSIX path"):
        load_runtime_review_run_configuration(tmp_path / "traversal.json")

    primitive = _valid_primitive(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "linked-output").symlink_to(outside, target_is_directory=True)
    primitive["output_dir"] = "linked-output"
    _write_config(tmp_path / "linked.json", primitive)
    with pytest.raises(PathAdmissionError, match="symbolic links"):
        load_runtime_review_run_configuration(tmp_path / "linked.json")

    (tmp_path / "linked-output").unlink()
    (tmp_path / "linked-output").write_text("caller data", encoding="utf-8")
    with pytest.raises(PathAdmissionError, match="ancestors must be directories"):
        load_runtime_review_run_configuration(tmp_path / "linked.json")


def test_output_manifest_overlap_is_refused(tmp_path: Path) -> None:
    """The output root cannot reuse the admitted manifest's caller-owned path."""
    primitive = _valid_primitive(tmp_path)
    primitive["output_dir"] = primitive["manifest"]
    _write_config(tmp_path / "run.json", primitive)

    with pytest.raises(PathAdmissionError, match="ancestors must be directories"):
        load_runtime_review_run_configuration(tmp_path / "run.json")


def test_output_appearance_is_refused_before_ownership_but_allowed_for_final_input_replay(
    tmp_path: Path,
) -> None:
    """The preownership gate detects collisions while final input replay permits the owned root."""
    primitive = _valid_primitive(tmp_path)
    _write_config(tmp_path / "run.json", primitive)
    admitted = load_runtime_review_run_configuration(tmp_path / "run.json")
    admitted.output_dir.mkdir()

    with pytest.raises(PathAdmissionError, match="appeared"):
        recheck_runtime_review_run_configuration(admitted)
    recheck_runtime_review_run_configuration(admitted, require_output_absent=False)


def test_interpreter_launcher_mutation_is_detected(tmp_path: Path) -> None:
    """Changing launcher bytes after admission invalidates the final execution identity gate."""
    primitive = _valid_primitive(tmp_path)
    _write_config(tmp_path / "run.json", primitive)
    admitted = load_runtime_review_run_configuration(tmp_path / "run.json")
    launcher = Path(primitive["baseline_python"])
    launcher.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    launcher.chmod(0o700)

    with pytest.raises(PathAdmissionError, match="interpreter identity changed"):
        recheck_runtime_review_run_inputs(admitted)


def test_configuration_file_symlink_is_refused(tmp_path: Path) -> None:
    """The run declaration itself must be a no-follow regular file."""
    primitive = _valid_primitive(tmp_path)
    target = tmp_path / "target.json"
    _write_config(target, primitive)
    (tmp_path / "run.json").symlink_to(target.name)

    with pytest.raises(JsonAdmissionError, match="no-follow"):
        load_runtime_review_run_configuration(tmp_path / "run.json")


def test_manifest_parent_component_symlink_is_refused(tmp_path: Path) -> None:
    """Descriptor-confined manifest walking cannot escape through an ancestor link."""
    primitive = _valid_primitive(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "manifest.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    primitive["manifest"] = "linked/manifest.json"
    _write_config(tmp_path / "run.json", primitive)

    with pytest.raises(PathAdmissionError, match="ancestor could not be opened"):
        load_runtime_review_run_configuration(tmp_path / "run.json")


def test_launcher_parent_component_symlink_is_refused(tmp_path: Path) -> None:
    """Only the declared launcher's final component may be a virtual-environment symlink."""
    primitive = _valid_primitive(tmp_path)
    real_parent = tmp_path / "real-bin"
    real_parent.mkdir()
    executable = _make_executable(real_parent / "python", "linked-parent")
    (tmp_path / "linked-bin").symlink_to(real_parent, target_is_directory=True)
    primitive["baseline_python"] = (tmp_path / "linked-bin" / executable.name).as_posix()
    _write_config(tmp_path / "run.json", primitive)

    with pytest.raises(PathAdmissionError, match="unsafe component"):
        load_runtime_review_run_configuration(tmp_path / "run.json")


def test_manifest_fifo_is_refused_without_blocking(tmp_path: Path) -> None:
    """A FIFO cannot block descriptor-bound manifest admission."""
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable")
    primitive = _valid_primitive(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.unlink()
    os.mkfifo(manifest)
    _write_config(tmp_path / "run.json", primitive)

    with pytest.raises(PathAdmissionError, match="regular file"):
        load_runtime_review_run_configuration(tmp_path / "run.json")
