"""Actual repair file workflows with controlled engine results and owned file faults."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO

import pytest

from metrifid import _owned_artifacts as artifacts
from metrifid.repair import _codec, _engine, _files, _profile
from metrifid.repair._physical_types import Measurement, Snapshot
from metrifid.repair._request import RepairController, RepairRequest, RepairTask
from metrifid.repair._types import EngineResult, ExecutionCosts, PairedMeasurement, ReplayResult

_RUNTIME = {
    "implementation": "CPython",
    "python": "3.13.5",
    "system": "Linux",
    "machine": "x86_64",
    "mujoco": "3.16.0",
    "numpy": "2.5.4",
}


def _request() -> RepairRequest:
    """Use ordinary controlled numerical inputs, without a physical success fixture."""
    return RepairRequest(
        RepairTask(-0.06, (0.08, 0.06), math.pi / 2), RepairController((0.5,) * 15)
    )


def _pair(parameters: tuple[float, ...], *, qualified: bool) -> PairedMeasurement:
    """Create complete controlled measurements using the accepted objective calculation."""
    measurements = []
    for dt in (0.002, 0.001):
        maxima = (0.004 if qualified else 0.00625, 0.01, 0.001, 0.01)
        measurements.append(
            Measurement(
                "COMPLETE",
                qualified,
                "COMPLETE" if qualified else "TARGET_HOLD_NOT_QUALIFIED",
                maxima,
                completed_steps=round(4 / dt),
                expected_steps=round(4 / dt),
                dt_s=dt,
                hold_samples=round(0.2 / dt) + 1,
                expected_hold_samples=round(0.2 / dt) + 1,
            )
        )
    nominal, finer = measurements
    return PairedMeasurement(
        parameters,
        nominal,
        finer,
        _engine._objective(nominal, finer),
        "COMPLETE",
        "COMPLETE_PAIRED_EVALUATION" if qualified else "TARGET_HOLD_NOT_QUALIFIED",
        True,
        qualified,
        6000,
        0,
    )


def _snapshot(identity: str) -> Snapshot:
    """Supply a shape/hash-valid controlled state, never evidence of actual settling."""
    shapes = {"qpos": (9,), "qvel": (8,), "act": (0,), "mocap_pos": (0, 3), "mocap_quat": (0, 4)}
    arrays = tuple((name, shape, (0.0,) * math.prod(shape)) for name, shape in shapes.items())
    raw = b"".join(struct.pack("<" + "d" * len(values), *values) for _, _, values in sorted(arrays))
    return Snapshot(arrays, (0.0, 0.0, 0.01), 0.0, identity, hashlib.sha256(raw).hexdigest())


def _result(*, qualified: bool = False, selected: bool = True) -> EngineResult:
    """Return a consistent typed result while all physical execution remains controlled."""
    request = _request()
    original = request.controller.parameters_normalized
    result = EngineResult(
        "ERROR",
        70,
        "EVALUATOR_EXCEPTION",
        request=request,
        original_parameters=original,
        costs=ExecutionCosts(limits=request.limits, search_cpu_fraction=0.75),
        diagnostic="controlled software fixture",
    )
    if not selected:
        return result
    profile, nominal, finer, identities = _profile.generate_expected(request)
    chosen = (0.51, *original[1:]) if qualified else original
    initial, chosen_pair = _pair(original, qualified=False), _pair(chosen, qualified=qualified)
    steps = 18200 if qualified else 12200
    return replace(
        result,
        status="REPAIRED" if qualified else "NO_QUALIFIED_POLICY_FOUND",
        exit_code=0 if qualified else 10,
        reason="COMPLETE_PAIRED_CONFIRMATION" if qualified else "ATTEMPT_LIMIT",
        selected_parameters=chosen,
        profile=profile,
        nominal_xml=nominal,
        finer_xml=finer,
        timestep_identities=identities,
        reset=_snapshot(str(identities["nominal_scene_sha256"])),
        initialization=Measurement(
            "COMPLETE", False, "SETTLED", completed_steps=200, expected_steps=200, dt_s=0.002
        ),
        initial_screening=initial,
        selected_screening=chosen_pair,
        confirmation=chosen_pair if qualified else None,
        confirmation_complete=qualified,
        qualified=qualified,
        runtime=_RUNTIME.copy(),
        costs=ExecutionCosts(2, steps, steps, 0, 0.125, request.limits, 0.75),
        diagnostic=None,
    )


def _input(tmp_path: Path) -> Path:
    """Create the explicit request in this test's own directory."""
    path = tmp_path / "request.json"
    path.write_bytes(_codec.encode_json(asdict(_request())))
    return path


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    qualified: bool = False,
    selected: bool = True,
) -> tuple[Path, _files.WorkflowResult]:
    """Exercise real admission/publication, replacing only expensive engine execution."""
    expected = _result(qualified=qualified, selected=selected)
    calls = []

    def execute(request: RepairRequest) -> EngineResult:
        calls.append(request)
        assert request == expected.request
        return expected

    monkeypatch.setattr(_engine, "repair_controller", execute)
    directory = tmp_path / "run"
    result = _files.run_request_file(_input(tmp_path), directory)
    assert calls == [expected.request]
    assert result.command_exit_code == expected.exit_code, result.diagnostic
    assert result.result_path == directory / "repair.json"
    return directory, result


def test_run_publishes_real_members_then_commit_marker_and_show_preserves_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sibling run is allowed; successful reading preserves recorded nonqualification."""
    committed = []
    original = artifacts.commit_owned_artifact

    def commit(artifact: artifacts.OwnedArtifact, name: str) -> artifacts.OwnedArtifact:
        committed.append(name)
        return original(artifact, name)

    monkeypatch.setattr(artifacts, "commit_owned_artifact", commit)
    directory, outcome = _run(tmp_path, monkeypatch)
    assert committed[-1] == "repair.json"
    assert set(committed) == {
        "nominal.xml",
        "finer.xml",
        "reset.json",
        "policy.json",
        "repair.json",
    }
    assert set(outcome.artifact_paths) == set(committed)
    assert all(path.is_file() and path.is_absolute() for path in outcome.artifact_paths.values())
    assert not list(directory.glob(".repair-*.tmp"))
    shown = _files.read_result_file(directory / "repair.json")
    assert (shown.command_status, shown.command_exit_code) == ("READ", 0)
    assert (shown.recorded_status, shown.recorded_exit_code, shown.qualified) == (
        "NO_QUALIFIED_POLICY_FOUND",
        10,
        False,
    )
    assert shown.actual_work.integration_steps == 12200
    assert shown.document["request"]["task"]["target_xy_m"] == [0.08, 0.06]


@pytest.mark.parametrize(
    "kind", ["existing", "missing_parent", "linked_parent", "linked_output", "nul"]
)
def test_destination_refusal_precedes_engine_and_never_overwrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    """Invalid destinations cannot trigger repair or alter unrelated existing bytes."""
    supplied = _input(tmp_path)
    marker = tmp_path / "keep"
    marker.write_bytes(b"unrelated")
    output = tmp_path / "run"
    if kind == "existing":
        output.mkdir()
        (output / "repair.json").write_bytes(b"prior run")
    elif kind == "missing_parent":
        output = tmp_path / "missing/run"
    elif kind == "linked_parent":
        (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
        output = tmp_path / "alias/run"
    elif kind == "linked_output":
        output.symlink_to(marker)
    else:
        output = tmp_path / "bad\x00name"
    monkeypatch.setattr(
        _engine, "repair_controller", lambda _request: pytest.fail("engine invoked")
    )
    result = _files.run_request_file(supplied, output)
    assert (result.command_status, result.command_exit_code) == ("REFUSED", 64)
    assert result.result_path is None
    assert not result.qualified
    assert marker.read_bytes() == b"unrelated"
    if kind == "existing":
        assert (output / "repair.json").read_bytes() == b"prior run"
    if kind == "missing_parent":
        assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize(("contents", "expected"), [(b"{", 64), (None, 70)])
def test_request_error_classification_precedes_output_and_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: bytes | None,
    expected: int,
) -> None:
    """Malformed input differs from an OS read failure, with no destination created."""
    path = tmp_path / "request.json"
    if contents is not None:
        path.write_bytes(contents)
    monkeypatch.setattr(
        _engine, "repair_controller", lambda _request: pytest.fail("engine invoked")
    )
    result = _files.run_request_file(path, tmp_path / "run")
    assert result.command_exit_code == expected
    assert result.result_path is None
    assert not (tmp_path / "run").exists()


def test_actual_runtime_refusal_creates_no_bundle_and_no_physics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the accepted runtime gate through the real file and engine route."""

    def refuse() -> dict[str, str]:
        raise _engine._RuntimeRefused("controlled unsupported runtime")

    monkeypatch.setattr(_engine, "_runtime", refuse)
    monkeypatch.setattr(_engine, "_create_backend", lambda *_args: pytest.fail("physics invoked"))
    result = _files.run_request_file(_input(tmp_path), tmp_path / "run")
    assert (result.command_status, result.command_exit_code, result.reason) == (
        "REFUSED",
        64,
        "UNSUPPORTED_RUNTIME",
    )
    assert result.result_path is None
    assert not result.qualified
    assert (tmp_path / "run").is_dir()
    assert list((tmp_path / "run").iterdir()) == []


def test_partial_write_preserves_committed_owned_finals_and_unrelated_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real partial sealed write cleans its temporary but keeps the earlier final."""
    original = artifacts.write_owned_bytes
    count = 0

    def fail_second(artifact: artifacts.OwnedArtifact, payload: bytes) -> None:
        nonlocal count
        count += 1
        if count == 2:

            def partial(stream: BinaryIO) -> None:
                stream.write(payload[:7])
                raise OSError("controlled partial write")

            artifacts.write_owned_stream(artifact, partial)
        else:
            original(artifact, payload)

    monkeypatch.setattr(artifacts, "write_owned_bytes", fail_second)
    monkeypatch.setattr(_engine, "repair_controller", lambda _request: _result(qualified=True))
    keep = tmp_path / "keep"
    keep.write_bytes(b"untouched")
    result = _files.run_request_file(_input(tmp_path), tmp_path / "run")
    assert (result.command_status, result.command_exit_code) == ("ERROR", 70)
    assert result.result_path is None
    assert not result.qualified
    assert result.artifact_paths == {}
    assert result.partial_output
    assert result.actual_work.integration_steps == 18200
    assert {path.name for path in (tmp_path / "run").iterdir()} == {"finer.xml"}
    assert keep.read_bytes() == b"untouched"


def test_output_replacement_after_commit_has_no_false_success_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the actual public-root identity check while preserving all owned finals."""
    original = _files._verify_public_root
    output, moved = tmp_path / "run", tmp_path / "moved-owned"

    def replace_root(owner: _files.OwnedOutputRoot) -> None:
        owner.path.rename(moved)
        output.mkdir()
        (output / "foreign").write_bytes(b"must survive")
        original(owner)

    monkeypatch.setattr(_files, "_verify_public_root", replace_root)
    monkeypatch.setattr(_engine, "repair_controller", lambda _request: _result(qualified=True))
    result = _files.run_request_file(_input(tmp_path), output)
    assert (result.command_status, result.command_exit_code) == ("ERROR", 70)
    assert result.result_path is None
    assert not result.qualified
    assert result.partial_output
    assert (moved / "repair.json").is_file()
    assert (output / "foreign").read_bytes() == b"must survive"
    assert not list(moved.glob(".repair-*.tmp"))


def test_no_clobber_collision_during_member_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real exclusive link refuses a newly inserted foreign final."""
    original = artifacts.commit_owned_artifact
    output = tmp_path / "run"

    def collide(artifact: artifacts.OwnedArtifact, name: str) -> artifacts.OwnedArtifact:
        if name == "nominal.xml":
            (output / name).write_bytes(b"foreign final")
        return original(artifact, name)

    monkeypatch.setattr(artifacts, "commit_owned_artifact", collide)
    monkeypatch.setattr(_engine, "repair_controller", lambda _request: _result())
    result = _files.run_request_file(_input(tmp_path), output)
    assert result.command_exit_code == 70
    assert result.result_path is None
    assert (output / "nominal.xml").read_bytes() == b"foreign final"
    assert (output / "finer.xml").is_file()
    assert not (output / "repair.json").exists()
    assert not list(output.glob(".repair-*.tmp"))


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("bytes", 64), ("link", 64), ("fifo", 64), ("oversize", 64), ("missing", 70)],
)
def test_read_rejects_inconsistent_or_nonregular_declared_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    expected: int,
) -> None:
    """Real bound reads cannot follow a member link, block on a FIFO or trust changed bytes."""
    directory, _ = _run(tmp_path, monkeypatch)
    member = directory / "nominal.xml"
    member.unlink()
    if kind == "bytes":
        member.write_bytes(b"changed")
    elif kind == "link":
        member.symlink_to(directory / "finer.xml")
    elif kind == "fifo":
        os.mkfifo(member)
    elif kind == "oversize":
        member.write_bytes(b"x" * (65536 + 1))
    result = _files.read_result_file(directory / "repair.json")
    assert result.command_exit_code == expected
    assert result.result_path is None
    assert not result.qualified


def test_read_allows_unreferenced_files_without_claiming_them_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only declared fixed artifacts enter the returned verified-path mapping."""
    directory, _ = _run(tmp_path, monkeypatch)
    (directory / "notes.txt").write_text("ordinary side note", encoding="utf-8")
    result = _files.read_result_file(directory / "repair.json")
    assert result.command_exit_code == 0
    assert "notes.txt" not in result.artifact_paths


def test_no_selection_error_is_readable_but_replay_refuses_without_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid failed run retains its outcome without inventing a selected controller."""
    directory, _ = _run(tmp_path, monkeypatch, selected=False)
    assert {path.name for path in directory.iterdir()} == {"repair.json"}
    shown = _files.read_result_file(directory / "repair.json")
    assert (shown.command_status, shown.command_exit_code, shown.recorded_exit_code) == (
        "READ",
        0,
        70,
    )
    monkeypatch.setattr(_engine, "replay_controller", lambda _result: pytest.fail("replay invoked"))
    result = _files.replay_run_directory(directory, tmp_path / "replay")
    assert (result.command_status, result.command_exit_code, result.reason) == (
        "REFUSED",
        64,
        "NO_SELECTED_CONTROLLER",
    )
    assert not (tmp_path / "replay").exists()


def test_moved_bundle_replay_uses_retained_data_and_writes_only_fresh_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual file route re-admits a moved bundle and never reopens its old request."""
    directory, _ = _run(tmp_path, monkeypatch, qualified=True)
    moved = tmp_path / "moved"
    directory.rename(moved)
    (tmp_path / "request.json").unlink()
    before = {path.name: path.read_bytes() for path in moved.iterdir()}
    calls = []

    def evaluate(saved: EngineResult) -> ReplayResult:
        calls.append(saved)
        assert saved.qualified
        assert saved.request == _request()
        return ReplayResult(
            "QUALIFIED",
            0,
            "COMPLETE_PAIRED_EVALUATION",
            saved.request,
            saved.selected_parameters,
            _pair(saved.selected_parameters, qualified=True),
            True,
            _RUNTIME.copy(),
            ExecutionCosts(0, 6200, 6200, 0, 0.125, saved.request.limits, None),
        )

    monkeypatch.setattr(_engine, "replay_controller", evaluate)
    replay = tmp_path / "replay"
    result = _files.replay_run_directory(moved, replay)
    assert len(calls) == 1
    assert (result.command_status, result.command_exit_code, result.qualified) == (
        "QUALIFIED",
        0,
        True,
    )
    assert result.result_path == replay / "replay.json"
    assert {path.name for path in replay.iterdir()} == {"replay.json"}
    assert {path.name: path.read_bytes() for path in moved.iterdir()} == before
    shown = _files.read_result_file(replay / "replay.json")
    assert shown.command_status == "READ"
    assert shown.recorded_status == "QUALIFIED"
    assert shown.actual_work.integration_steps == 6200


@pytest.mark.parametrize("where", ["inside", "same", "ancestor", "existing"])
def test_replay_output_preflight_precedes_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    where: str,
) -> None:
    """Replay never mutates its source or evaluates for an invalid destination."""
    directory, _ = _run(tmp_path, monkeypatch)
    outputs = {
        "inside": directory / "replay",
        "same": directory,
        "ancestor": tmp_path,
        "existing": tmp_path / "existing",
    }
    outputs["existing"].mkdir()
    monkeypatch.setattr(_engine, "replay_controller", lambda _source: pytest.fail("replay invoked"))
    result = _files.replay_run_directory(directory, outputs[where])
    assert result.command_exit_code == 64
    assert result.result_path is None
    assert {path.name for path in directory.iterdir()} == {
        "repair.json",
        "nominal.xml",
        "finer.xml",
        "reset.json",
        "policy.json",
    }


def test_public_read_imports_no_native_libraries_in_fresh_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read a real decimal record with NumPy and MuJoCo imports explicitly forbidden."""
    directory, _ = _run(tmp_path, monkeypatch)
    code = """
import importlib.abc, json, sys
class DenyNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'mujoco'}:
            raise RuntimeError('native import forbidden: '+fullname)
sys.meta_path.insert(0, DenyNative())
from metrifid.repair import read_result_file
result = read_result_file(sys.argv[1])
assert result.command_status == 'READ'
assert result.recorded_exit_code == 10
assert result.actual_work.integration_steps == 12200
assert not any(name.split('.')[0] in {'numpy', 'mujoco'} for name in sys.modules)
print(json.dumps({'status':result.command_status,'recorded_exit':result.recorded_exit_code}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(directory / "repair.json")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"status": "READ", "recorded_exit": 10}


def test_inconsistent_success_cannot_publish_a_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual codec admission rejects a success claim without its confirmation."""
    invalid = replace(_result(qualified=True), confirmation=None)
    monkeypatch.setattr(_engine, "repair_controller", lambda _request: invalid)
    result = _files.run_request_file(_input(tmp_path), tmp_path / "run")
    assert (result.command_status, result.command_exit_code) == ("ERROR", 70)
    assert result.result_path is None
    assert not result.qualified
    assert list((tmp_path / "run").iterdir()) == []


def test_self_consistent_selected_replacement_is_evaluated_as_supplied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsigned internal consistency admits a supported replacement for fresh evaluation."""
    directory, _ = _run(tmp_path, monkeypatch)
    changed = (0.6,) + (0.5,) * 14
    replaced = replace(
        _result(), selected_parameters=changed, selected_screening=_pair(changed, qualified=False)
    )
    marker, members = _codec.serialize_run(replaced)
    for name, payload in members.items():
        (directory / name).write_bytes(payload)
    (directory / "repair.json").write_bytes(marker)
    received = []

    def evaluate(saved: EngineResult) -> ReplayResult:
        received.append(saved.selected_parameters)
        return ReplayResult(
            "NOT_QUALIFIED",
            10,
            "TARGET_HOLD_NOT_QUALIFIED",
            saved.request,
            saved.selected_parameters,
            _pair(saved.selected_parameters, qualified=False),
            False,
            _RUNTIME.copy(),
            ExecutionCosts(0, 6200, 6200, 0, 0.125, saved.request.limits, None),
        )

    monkeypatch.setattr(_engine, "replay_controller", evaluate)
    result = _files.replay_run_directory(directory, tmp_path / "replay")
    assert received == [changed]
    assert (result.command_status, result.command_exit_code, result.qualified) == (
        "NOT_QUALIFIED",
        10,
        False,
    )
    assert result.result_path == tmp_path / "replay/replay.json"


def test_committed_member_mismatch_is_publication_error_not_input_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A final content check can fail after the marker exists; no saved path is claimed."""
    original = _files._read_record

    def corrupt_before_read(owner: _files.OwnedOutputRoot, filename: str) -> _files._ReadRecord:
        (owner.path / "policy.json").write_bytes(b"changed after commit")
        return original(owner, filename)

    monkeypatch.setattr(_files, "_read_record", corrupt_before_read)
    monkeypatch.setattr(_engine, "repair_controller", lambda _request: _result(qualified=True))
    result = _files.run_request_file(_input(tmp_path), tmp_path / "run")
    assert (result.command_status, result.command_exit_code) == ("ERROR", 70)
    assert result.result_path is None
    assert not result.qualified
    assert result.partial_output
    assert (tmp_path / "run/repair.json").is_file()


def test_replay_reset_refusal_after_setup_is_retained_and_readable_without_native(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh setup work survives a reset refusal in the standalone replay record."""
    directory, _ = _run(tmp_path, monkeypatch)

    def refused(saved: EngineResult) -> ReplayResult:
        return ReplayResult(
            "REFUSED",
            64,
            "RESET_MISMATCH",
            saved.request,
            saved.selected_parameters,
            runtime=_RUNTIME.copy(),
            costs=ExecutionCosts(0, 200, 200, 0, 0.125, saved.request.limits, None),
        )

    monkeypatch.setattr(_engine, "replay_controller", refused)
    output = tmp_path / "replay"
    result = _files.replay_run_directory(directory, output)
    assert (result.command_status, result.command_exit_code, result.reason) == (
        "REFUSED",
        64,
        "RESET_MISMATCH",
    )
    assert not result.qualified
    assert result.result_path == output / "replay.json"
    assert {path.name for path in output.iterdir()} == {"replay.json"}
    code = """
import importlib.abc, sys
class DenyNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'mujoco'}:
            raise RuntimeError('native import forbidden: '+fullname)
sys.meta_path.insert(0, DenyNative())
from metrifid.repair import read_result_file
result = read_result_file(sys.argv[1])
assert result.command_status == 'READ'
assert result.command_exit_code == 0
assert result.recorded_status == 'REFUSED'
assert result.recorded_exit_code == 64
assert result.reason == 'RESET_MISMATCH'
assert not result.qualified
assert result.actual_work.integration_steps == 200
assert result.actual_work.known_completed_steps == 200
assert result.actual_work.unknown_step_calls == 0
assert not any(name.split('.')[0] in {'numpy', 'mujoco'} for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(output / "replay.json")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _nonfinite_settling_reset(scene_hash: str, location: str) -> Snapshot:
    """Preserve raw failed-state values and their accepted physical-array digest."""
    reset = replace(_snapshot(scene_hash), settling_failures=("NONFINITE_STATE",))
    if location == "com":
        return replace(reset, settled_block_com=(math.nan, 0.0, 0.01))
    if location == "yaw":
        return replace(reset, settled_block_yaw=math.nan)
    arrays = list(reset.arrays)
    name, shape, values = arrays[0]
    arrays[0] = (name, shape, (math.nan, *values[1:]))
    raw = b"".join(struct.pack("<" + "d" * len(values), *values) for _, _, values in sorted(arrays))
    return replace(reset, arrays=tuple(arrays), state_sha256=hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize("location", ["arrays", "com", "yaw"])
def test_nonfinite_initialization_retains_incomplete_file_run_and_native_free_show(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
) -> None:
    """The real engine and file route preserve failed setup without an unusable reset."""
    setup_calls = 0
    created = []

    def setup_step() -> None:
        nonlocal setup_calls
        setup_calls += 1

    def controlled_backend(request: RepairRequest, budget: _engine._Budget) -> SimpleNamespace:
        assert request == _request()
        profile, nominal, finer, identities = _profile.generate_expected(request)
        reset = _nonfinite_settling_reset(str(identities["nominal_scene_sha256"]), location)
        initialization = Measurement(
            "INCOMPLETE",
            False,
            "INITIAL_STATE_NOT_ADMISSIBLE",
            metrics={
                "settling_failures": ["NONFINITE_STATE"],
                "physical_state_sha256": reset.state_sha256,
            },
            hard_failures=("NONFINITE_STATE",),
            completed_steps=200,
            expected_steps=200,
            dt_s=0.002,
        )
        for _ in range(200):
            budget.step(setup_step)
        backend = SimpleNamespace(
            profile=profile,
            nominal_xml=nominal,
            finer_xml=finer,
            identities=identities,
            snapshot=reset,
            initialization=initialization,
            evaluate=lambda *_args, **_kwargs: pytest.fail("Failed setup reached evaluation"),
        )
        created.append(backend)
        return backend

    monkeypatch.setattr(_engine, "_runtime", lambda: _RUNTIME.copy())
    monkeypatch.setattr(_engine.time, "process_time", lambda: 0.0)
    monkeypatch.setattr(_engine, "_create_backend", controlled_backend)
    output = tmp_path / "run"
    result = _files.run_request_file(_input(tmp_path), output)
    assert (result.command_status, result.command_exit_code, result.reason) == (
        "INCOMPLETE",
        20,
        "INITIAL_STATE_NOT_ADMISSIBLE",
    ), result.diagnostic
    assert setup_calls == 200
    assert len(created) == 1
    assert result.result_path == output / "repair.json"
    assert not result.qualified
    assert result.recorded_status == "INCOMPLETE"
    assert result.recorded_exit_code == 20
    assert result.actual_work.integration_steps == 200
    assert result.actual_work.known_completed_steps == 200
    assert result.actual_work.unknown_step_calls == 0
    assert result.actual_work.attempts == 0
    assert result.actual_work.cpu_seconds == 0.0
    assert result.actual_work.limits == _request().limits
    assert {path.name for path in output.iterdir()} == {"repair.json", "nominal.xml", "finer.xml"}
    assert set(result.artifact_paths) == {"repair.json", "nominal.xml", "finer.xml"}
    assert set(result.document["artifacts"]) == {"nominal.xml", "finer.xml"}
    assert result.document["selected_parameters"] is None
    assert result.document["initial_screening"] is None
    assert result.document["selected_screening"] is None
    assert result.document["confirmation"] is None
    assert not result.document["confirmation_complete"]
    initialization = result.document["initialization"]
    assert initialization["classification"] == "INCOMPLETE"
    assert initialization["reason"] == "INITIAL_STATE_NOT_ADMISSIBLE"
    assert initialization["hard_failures"] == ["NONFINITE_STATE"]
    assert initialization["metrics"]["settling_failures"] == ["NONFINITE_STATE"]
    assert initialization["completed_steps"] == initialization["expected_steps"] == 200
    assert initialization["dt_s"] == 0.002
    assert initialization["metrics"]["physical_state_sha256"] == created[0].snapshot.state_sha256
    assert (output / "nominal.xml").read_bytes() == created[0].nominal_xml.encode("utf-8")
    assert (output / "finer.xml").read_bytes() == created[0].finer_xml.encode("utf-8")
    assert created[0].snapshot.settling_failures == ("NONFINITE_STATE",)
    code = """
import importlib.abc, sys
class DenyNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'mujoco'}:
            raise AssertionError('Native import forbidden: '+fullname)
sys.meta_path.insert(0, DenyNative())
from pathlib import Path
from metrifid.repair import _engine, read_result_file, replay_run_directory
path = Path(sys.argv[1])
shown = read_result_file(path)
assert shown.command_status == 'READ'
assert shown.command_exit_code == 0
assert shown.recorded_status == 'INCOMPLETE'
assert shown.recorded_exit_code == 20
assert shown.reason == 'INITIAL_STATE_NOT_ADMISSIBLE'
assert not shown.qualified
assert shown.actual_work.integration_steps == 200
assert shown.actual_work.known_completed_steps == 200
assert shown.actual_work.unknown_step_calls == 0
assert shown.actual_work.attempts == 0
assert shown.document['initialization']['hard_failures'] == ['NONFINITE_STATE']
assert shown.document['initialization']['metrics']['settling_failures'] == ['NONFINITE_STATE']
assert set(shown.document['artifacts']) == {'nominal.xml', 'finer.xml'}
assert shown.document['selected_parameters'] is None
invoked = []
def forbidden_replay(source):
    invoked.append(source)
    raise AssertionError('No-selection bundle invoked replay engine')
_engine.replay_controller = forbidden_replay
output = Path(sys.argv[2])
replayed = replay_run_directory(path.parent, output)
assert replayed.command_status == 'REFUSED'
assert replayed.command_exit_code == 64
assert replayed.reason == 'NO_SELECTED_CONTROLLER'
assert not replayed.qualified
assert replayed.result_path is None
assert not invoked
assert not output.exists()
from metrifid.cli import main
assert main(['repair', 'show', str(path), '--json']) == 0
assert not any(name.split('.')[0] in {'numpy', 'mujoco'} for name in sys.modules)
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            code,
            str(output / "repair.json"),
            str(tmp_path / "replay"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert len(completed.stdout.splitlines()) == 1
    summary = json.loads(completed.stdout)
    assert summary["schema"] == "metrifid.repair_summary"
    assert summary["operation"] == "show"
    assert summary["command_status"] == "READ"
    assert summary["command_exit_code"] == 0
    assert summary["recorded_status"] == "INCOMPLETE"
    assert summary["recorded_exit_code"] == 20
    assert summary["reason"] == "INITIAL_STATE_NOT_ADMISSIBLE"
    assert not summary["qualified"]
    assert summary["result_path"] == str(output / "repair.json")
