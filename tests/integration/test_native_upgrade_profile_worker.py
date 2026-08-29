"""Integration coverage for the standalone native-upgrade profile worker."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import struct
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import mujoco  # type: ignore[import-untyped]
import numpy as np
import pytest

_ROOT = Path(__file__).parents[2]
_WORKER = _ROOT / "tools" / "native_upgrade_profile_worker.py"
_MANIFEST = _ROOT / "tests" / "fixtures" / "native_upgrade" / "manifest.json"
_THREAD_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_RETAINED_WORKER_PROFILES = {
    "3.10.0": "A_3.10.0",
    "3.11.0": "B_3.11.0",
}
_COMPLETED_CHECKSUM_MEMBERS = (
    "fixture.xml",
    "input_manifest.json",
    "model.mjb",
    "result.json",
    "trace.npz",
)
_REFUSED_CHECKSUM_MEMBERS = (
    "fixture.xml",
    "input_manifest.json",
    "model.mjb",
    "result.json",
)
_EXACT_RUNTIME_REFUSAL = "worker requires exact native MuJoCo 3.10.0 or 3.11.0"
_EXACT_NUMPY_REFUSAL = "worker requires exact NumPy 2.3.5"
_EXPECTED_LIMITATIONS = [
    "ONE_EXACT_SELF_CONTAINED_MJCF_CLOSURE_ONLY",
    "ONE_EXACT_INITIAL_STATE_AND_ACTION_PROGRAM_ONLY",
    "EXACT_NATIVE_CPU_RUNTIME_PROFILES_ONLY",
    "NO_UNIVERSAL_MUJOCO_VERSION_EQUIVALENCE_CLAIM",
    "NO_POLICY_OR_HARDWARE_SAFETY_CLAIM",
    "NO_TASK_SUCCESS_OR_REAL_WORLD_TRANSFER_CLAIM",
    "NO_BACKEND_PARITY_CLAIM",
    "WORKER_EMITS_EVIDENCE_ONLY_NO_MIGRATION_DECISION",
]


def _canonical(value: Any) -> bytes:
    """Serialize one JSON value using the independently reproduced worker convention."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _canonical_sha(value: Any) -> str:
    """Hash one canonical JSON projection."""
    return hashlib.sha256(_canonical(value)).hexdigest()


def _frame(digest: Any, label: str, payload: bytes) -> None:
    """Append one independently implemented trace-hash frame."""
    encoded = label.encode()
    digest.update(struct.pack(">Q", len(encoded)))
    digest.update(encoded)
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def _load_worker_module() -> ModuleType:
    """Load the standalone worker as a test-local module without executing its CLI."""
    spec = importlib.util.spec_from_file_location("native_upgrade_profile_worker_test", _WORKER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_binary64_state_evidence_preserves_signed_zero(
    worker_module: ModuleType,
) -> None:
    """State tokens round-trip signed zero and byte equality observes its sign bit."""
    state = np.asarray([0.0, -0.0, 1.0, -1.0], dtype="<f8")

    evidence = worker_module._integration_state_evidence(state)

    assert evidence["values"] == ["0", "-0.0", "1.0", "-1.0"]
    assert evidence["sha256"] == hashlib.sha256(state.tobytes(order="C")).hexdigest()
    assert worker_module._binary64_arrays_equal(state, state.copy())

    changed_sign = state.copy()
    changed_sign[1] = 0.0
    assert not worker_module._binary64_arrays_equal(state, changed_sign)


def test_exact_binary64_state_equality_detects_nonzero_bit_changes(
    worker_module: ModuleType,
) -> None:
    """The exact-state predicate rejects a real one-ULP state change."""
    reference = np.asarray([0.0, 1.0], dtype="<f8")
    changed = reference.copy()
    changed[1] = np.nextafter(changed[1], math.inf)

    assert not worker_module._binary64_arrays_equal(reference, changed)


def test_exact_binary64_state_evidence_refuses_non_finite_values(
    worker_module: ModuleType,
) -> None:
    """Signed-zero support must not admit a non-finite complete integration state."""
    assert worker_module._float_token(math.nan) is None
    assert worker_module._float_token(math.inf) is None
    assert worker_module._float_token(-math.inf) is None

    for corrupt in (math.nan, math.inf, -math.inf):
        state = np.asarray([0.0, corrupt], dtype="<f8")
        with pytest.raises(worker_module.WorkerRefusal, match="non-finite value"):
            worker_module._integration_state_evidence(state)


@pytest.fixture(scope="module")
def worker_module() -> ModuleType:
    """Provide one imported worker module for pure admission and simulation checks."""
    return _load_worker_module()


def _worker_environment() -> dict[str, str]:
    """Build the exact bounded worker environment without an external profile link."""
    environment = os.environ.copy()
    environment.update(_THREAD_ENVIRONMENT)
    environment.pop("METRIFID_NATIVE_UPGRADE_PROFILE_IDENTITY", None)
    return environment


def _run_worker(
    *,
    manifest: Path,
    fixture_id: str,
    step_dt: str,
    repeat_id: int,
    output: Path,
) -> subprocess.CompletedProcess[str]:
    """Invoke one fresh standalone worker process with the frozen CLI shape."""
    return subprocess.run(
        [
            sys.executable,
            str(_WORKER),
            "--manifest",
            str(manifest),
            "--fixture-id",
            fixture_id,
            "--step-dt",
            step_dt,
            "--repeat-id",
            str(repeat_id),
            "--output",
            str(output),
        ],
        capture_output=True,
        check=False,
        env=_worker_environment(),
        text=True,
        timeout=90,
    )


def _load_json(path: Path) -> dict[str, Any]:
    """Load one emitted JSON object for contract assertions."""
    value = json.loads(path.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _verify_checksums(
    output: Path,
    expected_members: tuple[str, ...] = _COMPLETED_CHECKSUM_MEMBERS,
) -> None:
    """Independently verify exact checksum grammar, membership, and digests."""
    lines = (output / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines()
    assert [line.split("  ", 1)[1] for line in lines] == list(expected_members)
    for line in lines:
        digest, name = line.split("  ", 1)
        assert len(digest) == 64
        assert digest == digest.lower()
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def _retained_worker_profile() -> tuple[str, str] | None:
    """Return the frozen worker profile for the exact live retained runtime, if any."""
    package_version = str(getattr(mujoco, "__version__", "UNKNOWN"))
    profile_id = _RETAINED_WORKER_PROFILES.get(package_version)
    if (
        profile_id is None
        or str(mujoco.mj_versionString()) != package_version
        or str(np.__version__) != "2.3.5"
    ):
        return None
    return profile_id, package_version


def _assert_nonretained_runtime_refusal(
    completed: subprocess.CompletedProcess[str],
    output: Path,
    *,
    fixture_id: str,
    step_dt: str,
    repeat_id: int,
) -> None:
    """Require the frozen worker's exact bounded result outside its retained profiles."""
    package_version = str(getattr(mujoco, "__version__", "UNKNOWN"))
    profile_id = _RETAINED_WORKER_PROFILES.get(package_version)
    if profile_id is None or str(mujoco.mj_versionString()) != package_version:
        reason = f"WorkerRefusal: {_EXACT_RUNTIME_REFUSAL}"
    else:
        assert str(np.__version__) != "2.3.5"
        reason = f"WorkerRefusal: {_EXACT_NUMPY_REFUSAL}"
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == f"REFUSED: {reason}\n"
    assert sorted(path.name for path in output.iterdir()) == [
        "CHECKSUMS.sha256",
        *_REFUSED_CHECKSUM_MEMBERS,
    ]
    _verify_checksums(output, _REFUSED_CHECKSUM_MEMBERS)
    assert not (output / "trace.npz").exists()
    assert _load_json(output / "result.json") == {
        "schema": "metrifid.native_upgrade_worker_result",
        "schema_version": 1,
        "status": "REFUSED",
        "fixture_id": fixture_id,
        "profile_id": profile_id or "UNKNOWN",
        "profile_version": package_version,
        "step_dt": step_dt,
        "repeat_id": repeat_id,
        "manifest_raw_sha256": None,
        "manifest_echo": None,
        "subject": None,
        "workload": None,
        "runtime": None,
        "trace": None,
        "contacts": [],
        "diagnostics": {"refusal_reason": reason},
        "limitations": _EXPECTED_LIMITATIONS,
    }


def _verify_self_hash(value: dict[str, Any], field: str) -> None:
    """Independently recompute one append-last canonical self-hash."""
    projection = dict(value)
    claimed = projection.pop(field)
    assert claimed == _canonical_sha(projection)


def _verify_trace_hash(output: Path, result: dict[str, Any]) -> None:
    """Independently reproduce the scientific hash without hashing NPZ container bytes."""
    trace = result["trace"]
    descriptors = trace["channels"]
    with np.load(output / "trace.npz", allow_pickle=False) as archive:
        expected = {"observation_times", *(item["array_key"] for item in descriptors)}
        assert set(archive.files) == expected
        times = np.ascontiguousarray(archive["observation_times"], dtype="<f8")
        assert times.shape == (trace["observation_count"],)
        header = {
            "schema": "metrifid.native_upgrade_canonical_trace",
            "schema_version": 1,
            "observation_times": {
                "array_key": "observation_times",
                "shape": list(times.shape),
                "dtype": "<f8",
                "unit": "seconds",
            },
            "channels": descriptors,
            "semantic_contacts": result["contacts"],
        }
        digest = hashlib.sha256()
        _frame(digest, "metadata", _canonical(header))
        _frame(digest, "observation_times", times.tobytes(order="C"))
        for descriptor in descriptors:
            array = np.ascontiguousarray(archive[descriptor["array_key"]])
            assert list(array.shape) == descriptor["shape"]
            assert array.dtype.str == descriptor["dtype"]
            _frame(digest, descriptor["array_key"], array.tobytes(order="C"))
    assert digest.hexdigest() == trace["canonical_trace_sha256"]


def test_completed_worker_evidence_is_owned_replayable_and_self_bound(tmp_path: Path) -> None:
    """A retained worker completes fully; every other live runtime refuses exactly."""
    output = tmp_path / "peg"
    completed = _run_worker(
        manifest=_MANIFEST,
        fixture_id="peg_contact",
        step_dt="0.004",
        repeat_id=0,
        output=output,
    )
    profile = _retained_worker_profile()
    if profile is None:
        _assert_nonretained_runtime_refusal(
            completed,
            output,
            fixture_id="peg_contact",
            step_dt="0.004",
            repeat_id=0,
        )
        return
    profile_id, profile_version = profile
    assert completed.returncode == 0, completed.stderr
    assert sorted(path.name for path in output.iterdir()) == [
        "CHECKSUMS.sha256",
        "fixture.xml",
        "input_manifest.json",
        "model.mjb",
        "result.json",
        "trace.npz",
    ]
    _verify_checksums(output)
    result = _load_json(output / "result.json")
    assert result["status"] == "COMPLETED"
    assert result["fixture_id"] == "peg_contact"
    assert result["profile_id"] == profile_id
    assert result["profile_version"] == profile_version
    assert result["step_dt"] == "0.004"
    assert result["repeat_id"] == 0
    assert (
        result["manifest_raw_sha256"]
        == hashlib.sha256((output / "input_manifest.json").read_bytes()).hexdigest()
    )
    assert result["manifest_echo"] == json.loads(
        (output / "input_manifest.json").read_text(encoding="utf-8")
    )
    fixture_echo = next(
        item for item in result["manifest_echo"]["fixtures"] if item["fixture_id"] == "peg_contact"
    )
    subject = result["subject"]
    assert subject["fixture_manifest_sha256"] == _canonical_sha(fixture_echo)
    assert (
        subject["fixture_raw_sha256"]
        == hashlib.sha256((output / "fixture.xml").read_bytes()).hexdigest()
    )
    assert subject["compiled_mjb_size_bytes"] == (output / "model.mjb").stat().st_size
    assert (
        subject["compiled_mjb_sha256"]
        == hashlib.sha256((output / "model.mjb").read_bytes()).hexdigest()
    )
    _verify_self_hash(subject["source_closure"], "closure_sha256")
    _verify_self_hash(result["workload"]["initial_state"], "semantic_sha256")
    _verify_self_hash(result["workload"]["action_program"], "semantic_sha256")
    _verify_self_hash(result["workload"], "semantic_sha256")
    _verify_self_hash(result["runtime"], "runtime_identity_sha256")
    assert result["runtime"]["mujoco"]["loaded_native_library"]["sha256"]
    assert result["runtime"]["pip_check"]["exit_code"] == 0
    assert result["runtime"]["host"]["system"] == platform.system()
    assert result["runtime"]["host"]["release"] == platform.release()
    assert result["runtime"]["host"]["machine"] == platform.machine()
    assert result["runtime"]["host"]["architecture"] == list(platform.architecture())
    assert result["runtime"]["host"]["logical_cpu_count"] == os.cpu_count()
    if platform.system() == "Darwin":
        assert result["runtime"]["host"]["hardware_profile"]["Model Identifier"]
        assert result["runtime"]["host"]["hardware_profile"]["Processor Name"]
        assert result["runtime"]["host"]["hardware_profile"]["Total Number of Cores"]
        assert result["runtime"]["host"]["hardware_profile"]["Hyper-Threading Technology"]
    assert result["diagnostics"]["finite_values"] is True
    assert result["diagnostics"]["warnings_passed"] is True
    assert result["diagnostics"]["solver_converged"] is True
    assert len(result["diagnostics"]["samples"]) == 51
    assert result["contacts"][0]["events"] == [
        {"event": "ONSET", "time": "0.232"},
        {"event": "RELEASE", "time": "0.608"},
    ]
    _verify_trace_hash(output, result)


def test_fresh_process_repeats_have_identical_scientific_trace(tmp_path: Path) -> None:
    """Retained repeats match scientifically; other live runtimes refuse exactly."""
    profile = _retained_worker_profile()
    hashes: list[str] = []
    for repeat_id in (0, 1):
        output = tmp_path / f"repeat_{repeat_id}"
        completed = _run_worker(
            manifest=_MANIFEST,
            fixture_id="smooth_pendulum",
            step_dt="0.002",
            repeat_id=repeat_id,
            output=output,
        )
        if profile is None:
            _assert_nonretained_runtime_refusal(
                completed,
                output,
                fixture_id="smooth_pendulum",
                step_dt="0.002",
                repeat_id=repeat_id,
            )
        else:
            assert completed.returncode == 0, completed.stderr
            hashes.append(_load_json(output / "result.json")["trace"]["canonical_trace_sha256"])
    if profile is None:
        assert hashes == []
        return
    assert hashes[0] == hashes[1]


def test_existing_output_is_never_clobbered(tmp_path: Path) -> None:
    """An existing path is refused before any existing member can change."""
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_bytes(b"owned by caller")
    before = sentinel.read_bytes()
    completed = _run_worker(
        manifest=_MANIFEST,
        fixture_id="smooth_pendulum",
        step_dt="0.004",
        repeat_id=0,
        output=output,
    )
    assert completed.returncode == 2
    assert sentinel.read_bytes() == before
    assert [path.name for path in output.iterdir()] == ["sentinel"]


def _mutated_manifest(mutation: str) -> str:
    """Construct one deterministic invalid manifest mutation."""
    document = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    fixture = document["fixtures"][0]
    if mutation == "unknown_root":
        document["unknown"] = True
    elif mutation == "missing_key":
        del fixture["horizon"]
    elif mutation == "boolean_number":
        fixture["initial_qpos"][0] = False
    elif mutation == "numeric_string":
        fixture["control_dt"] = "0.02"
    elif mutation == "nonfinite":
        fixture["initial_qpos"][0] = float("nan")
    elif mutation == "duplicate_channel":
        fixture["channels"][1]["channel_id"] = fixture["channels"][0]["channel_id"]
    elif mutation == "path_escape":
        fixture["xml_path"] = "../smooth_pendulum.xml"
    elif mutation == "bad_grid":
        fixture["step_dts"][1] = 0.003
    elif mutation == "unaligned_boundary":
        fixture["controls"][0]["end"] = 0.31
        fixture["controls"][1]["start"] = 0.31
    elif mutation == "duplicate_fixture":
        document["fixtures"].append(copy.deepcopy(fixture))
    else:
        raise AssertionError(mutation)
    return json.dumps(document, allow_nan=True, separators=(",", ":"))


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_root",
        "missing_key",
        "boolean_number",
        "numeric_string",
        "nonfinite",
        "duplicate_channel",
        "path_escape",
        "bad_grid",
        "unaligned_boundary",
        "duplicate_fixture",
    ],
)
def test_strict_manifest_admission_refuses_invalid_shapes(tmp_path: Path, mutation: str) -> None:
    """Unknown, missing, unsafe, duplicate, nonnumeric, and misaligned inputs fail closed."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(_mutated_manifest(mutation), encoding="utf-8")
    output = tmp_path / "output"
    completed = _run_worker(
        manifest=manifest,
        fixture_id="smooth_pendulum",
        step_dt="0.004",
        repeat_id=0,
        output=output,
    )
    assert completed.returncode == 2
    result = _load_json(output / "result.json")
    assert result["status"] == "REFUSED"
    assert result["diagnostics"]["refusal_reason"]


def test_duplicate_json_member_is_refused(tmp_path: Path) -> None:
    """A duplicate JSON object member is never admitted with last-write-wins semantics."""
    raw = _MANIFEST.read_text(encoding="utf-8")
    duplicate = raw.replace(
        '"schema": "metrifid.native_upgrade_manifest",',
        '"schema": "metrifid.native_upgrade_manifest",\n  "schema": "duplicate",',
        1,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(duplicate, encoding="utf-8")
    completed = _run_worker(
        manifest=manifest,
        fixture_id="smooth_pendulum",
        step_dt="0.004",
        repeat_id=0,
        output=tmp_path / "output",
    )
    assert completed.returncode == 2
    assert "duplicate JSON member" in completed.stderr


def test_lexically_distinct_decimal_cannot_round_into_exact_grid(tmp_path: Path) -> None:
    """A near-equal JSON token is compared as Decimal, never rounded to the frozen value."""
    raw = _MANIFEST.read_text(encoding="utf-8")
    mutated = raw.replace('"control_dt": 0.02', '"control_dt": 0.020000000000000001', 1)
    assert mutated != raw
    manifest = tmp_path / "manifest.json"
    manifest.write_text(mutated, encoding="utf-8")
    completed = _run_worker(
        manifest=manifest,
        fixture_id="smooth_pendulum",
        step_dt="0.004",
        repeat_id=0,
        output=tmp_path / "output",
    )
    assert completed.returncode == 2
    assert "control_dt and horizon must be exactly" in completed.stderr


@pytest.mark.parametrize(
    "xml",
    [
        '<mujoco><include file="outside.xml"/></mujoco>',
        '<mujoco><asset><mesh name="external" file="outside.stl"/></asset></mujoco>',
        '<!DOCTYPE mujoco [<!ENTITY x "bad">]><mujoco/>',
        '<mujoco><extension><plugin plugin="external"/></extension></mujoco>',
    ],
)
def test_external_xml_mechanics_are_refused(worker_module: ModuleType, xml: str) -> None:
    """Includes, assets, entities, and plugins cannot widen the declared source closure."""
    with pytest.raises(worker_module.WorkerRefusal):
        worker_module._admit_self_contained_xml(xml.encode())


def test_symlinked_fixture_source_is_confined(tmp_path: Path) -> None:
    """A syntactically confined XML path cannot cross the source boundary through a symlink."""
    document = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    fixture = document["fixtures"][0]
    fixture["xml_path"] = "linked.xml"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    (tmp_path / "linked.xml").symlink_to(_MANIFEST.parent / "smooth_pendulum.xml")
    completed = _run_worker(
        manifest=manifest,
        fixture_id="smooth_pendulum",
        step_dt="0.004",
        repeat_id=0,
        output=tmp_path / "output",
    )
    assert completed.returncode == 2
    assert "symlink" in completed.stderr


def test_fixed_fixture_contracts_compile_and_simulate(
    worker_module: ModuleType,
) -> None:
    """All five fixtures bind their exact dimensions/channels and required semantic mechanics."""
    requirements = {
        "smooth_pendulum": (1, 1),
        "planar_arm": (2, 2),
        "contact_hopper": (3, 2),
        "articulated_chain": (12, 12),
        "peg_contact": (1, 1),
    }
    for fixture_id, (minimum_joints, actuator_count) in requirements.items():
        _, _, fixture = worker_module._load_manifest(_MANIFEST, fixture_id)
        _, raw = worker_module._resolve_fixture_source(_MANIFEST, fixture.xml_path)
        model = worker_module._compile_model(
            worker_module._admit_self_contained_xml(raw), fixture, worker_module.Decimal("0.004")
        )
        assert model.njnt >= minimum_joints
        assert model.nu == actuator_count
        evidence = worker_module._simulate(model, fixture, worker_module.Decimal("0.004"))
        assert evidence.observation_times.shape == (51,)
        assert evidence.diagnostics["finite_values"] is True
        assert evidence.diagnostics["warnings_passed"] is True
        assert evidence.diagnostics["solver_converged"] is True
        assert [item["channel_id"] for item in evidence.descriptors] == sorted(
            item["channel_id"] for item in evidence.descriptors
        )
    assert requirements["contact_hopper"][0] >= 3


@pytest.mark.parametrize("step_dt", ["0.004", "0.002", "0.001"])
def test_simulation_projects_declared_sensor_at_current_observation_state(
    worker_module: ModuleType, step_dt: str
) -> None:
    """Every grid projects the joint-position sensor from its exact current boundary state."""
    _, _, fixture = worker_module._load_manifest(_MANIFEST, "smooth_pendulum")
    _, raw = worker_module._resolve_fixture_source(_MANIFEST, fixture.xml_path)
    model = worker_module._compile_model(
        worker_module._admit_self_contained_xml(raw), fixture, worker_module.Decimal(step_dt)
    )

    evidence = worker_module._simulate(model, fixture, worker_module.Decimal(step_dt))
    descriptor_by_channel = {item["channel_id"]: item for item in evidence.descriptors}
    sensor = evidence.arrays[descriptor_by_channel["pendulum.theta.sensor"]["array_key"]]
    position = evidence.arrays[descriptor_by_channel["pendulum.theta.position"]["array_key"]]
    expected_times = np.asarray(
        [float(worker_module.Decimal("0.02") * index) for index in range(51)], dtype="<f8"
    )

    assert evidence.observation_times.shape == (51,)
    assert np.array_equal(evidence.observation_times, expected_times)
    assert bool(np.all(np.isfinite(sensor)))
    assert bool(np.all(np.isfinite(position)))
    assert np.array_equal(sensor, position)


def test_observation_projection_does_not_mutate_live_trajectory(
    worker_module: ModuleType,
) -> None:
    """Observer forwarding leaves live state and its next identical control step unchanged."""
    _, _, fixture = worker_module._load_manifest(_MANIFEST, "smooth_pendulum")
    _, raw = worker_module._resolve_fixture_source(_MANIFEST, fixture.xml_path)
    step_dt = worker_module.Decimal("0.004")
    model = worker_module._compile_model(
        worker_module._admit_self_contained_xml(raw), fixture, step_dt
    )
    left = worker_module._initial_data(model, fixture)
    right = worker_module._initial_data(model, fixture)
    for step_index in range(1, 6):
        start_time = step_dt * (step_index - 1)
        control = np.asarray(
            [float(value) for value in worker_module._control_at(fixture, start_time)],
            dtype=np.float64,
        )
        left.ctrl[:] = control
        right.ctrl[:] = control
        mujoco.mj_step(model, left)
        mujoco.mj_step(model, right)
    state_signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state_size = int(mujoco.mj_stateSize(model, state_signature))
    before = worker_module._integration_state(model, left, state_signature, state_size)
    before_bytes = before.tobytes(order="C")
    observer = worker_module._project_observation_data(
        model, left, mujoco.MjData(model), state_signature, state_size
    )
    after = worker_module._integration_state(model, left, state_signature, state_size)
    sensor_id = worker_module._name_id(model, mujoco.mjtObj.mjOBJ_SENSOR, "theta_position")
    joint_id = worker_module._name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "theta")
    sensor_value = observer.sensordata[int(model.sensor_adr[sensor_id])]
    position_value = observer.qpos[int(model.jnt_qposadr[joint_id])]

    assert np.array_equal(after, before)
    assert after.tobytes(order="C") == before_bytes
    assert sensor_value == position_value

    next_control = np.asarray(
        [
            float(value)
            for value in worker_module._control_at(fixture, worker_module.Decimal("0.02"))
        ],
        dtype=np.float64,
    )
    left.ctrl[:] = next_control
    right.ctrl[:] = next_control
    mujoco.mj_step(model, left)
    mujoco.mj_step(model, right)
    left_state = worker_module._integration_state(model, left, state_signature, state_size)
    right_state = worker_module._integration_state(model, right, state_signature, state_size)
    assert np.array_equal(left_state, right_state)
    assert left_state.tobytes(order="C") == right_state.tobytes(order="C")


def test_observation_projection_rejects_zero_sign_substitution(
    worker_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restored zero whose sign bit was flipped refuses before any forward projection.

    Numerically the substituted state is identical, so a comparison built on numerical equality
    accepts it. The retained bytes and their SHA-256 identity differ, so the exact-state boundary
    must reject it, and must do so before the observer pipeline runs or a receipt is produced.
    """
    _, _, fixture = worker_module._load_manifest(_MANIFEST, "smooth_pendulum")
    _, raw = worker_module._resolve_fixture_source(_MANIFEST, fixture.xml_path)
    model = worker_module._compile_model(
        worker_module._admit_self_contained_xml(raw), fixture, worker_module.Decimal("0.004")
    )
    live = worker_module._initial_data(model, fixture)
    state_signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state_size = int(mujoco.mj_stateSize(model, state_signature))
    live_before = worker_module._integration_state(model, live, state_signature, state_size)
    assert float(live.qvel[0]) == 0.0, "the fixture must start with a zero velocity component"
    assert math.copysign(1.0, float(live.qvel[0])) > 0.0, "that component must be positive zero"

    original_set_state = worker_module.mujoco.mj_setState
    forward_calls = 0

    def sign_substituted_set_state(
        subject_model: Any,
        subject_data: Any,
        state: Any,
        signature: int,
    ) -> None:
        """Restore through MuJoCo, then flip one observer-only zero to negative zero."""
        original_set_state(subject_model, subject_data, state, signature)
        subject_data.qvel[0] = -0.0

    def counted_forward(subject_model: Any, subject_data: Any) -> None:
        """Record any forbidden forward call after a sign-substituted restoration."""
        nonlocal forward_calls
        forward_calls += 1

    monkeypatch.setattr(worker_module.mujoco, "mj_setState", sign_substituted_set_state)
    monkeypatch.setattr(worker_module.mujoco, "mj_forward", counted_forward)

    with pytest.raises(worker_module.WorkerRefusal, match="exact integration state"):
        worker_module._project_observation_data(
            model, live, mujoco.MjData(model), state_signature, state_size
        )

    live_after = worker_module._integration_state(model, live, state_signature, state_size)
    assert forward_calls == 0, "the observer pipeline ran despite an inexact restoration"
    assert live_after.tobytes(order="C") == live_before.tobytes(order="C")


def test_observation_projection_rejects_inexact_restoration(
    worker_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-bit observer restoration change refuses before any forward projection."""
    _, _, fixture = worker_module._load_manifest(_MANIFEST, "smooth_pendulum")
    _, raw = worker_module._resolve_fixture_source(_MANIFEST, fixture.xml_path)
    model = worker_module._compile_model(
        worker_module._admit_self_contained_xml(raw), fixture, worker_module.Decimal("0.004")
    )
    live = worker_module._initial_data(model, fixture)
    state_signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state_size = int(mujoco.mj_stateSize(model, state_signature))
    live_before = worker_module._integration_state(model, live, state_signature, state_size)
    original_set_state = worker_module.mujoco.mj_setState
    forward_calls = 0

    def inexact_set_state(
        subject_model: Any,
        subject_data: Any,
        state: Any,
        signature: int,
    ) -> None:
        """Restore through MuJoCo, then alter one observer-only state member by one ULP."""
        original_set_state(subject_model, subject_data, state, signature)
        subject_data.qpos[0] = np.nextafter(subject_data.qpos[0], math.inf)

    def counted_forward(subject_model: Any, subject_data: Any) -> None:
        """Record any forbidden forward call after an inexact restoration."""
        nonlocal forward_calls
        forward_calls += 1

    monkeypatch.setattr(worker_module.mujoco, "mj_setState", inexact_set_state)
    monkeypatch.setattr(worker_module.mujoco, "mj_forward", counted_forward)

    with pytest.raises(worker_module.WorkerRefusal, match="exact integration state"):
        worker_module._project_observation_data(
            model, live, mujoco.MjData(model), state_signature, state_size
        )

    live_after = worker_module._integration_state(model, live, state_signature, state_size)
    assert forward_calls == 0
    assert np.array_equal(live_after, live_before)


def test_observation_projection_rejects_nonfinite_forward_projection(
    worker_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-finite observer sensor refuses without changing the live integration state."""
    _, _, fixture = worker_module._load_manifest(_MANIFEST, "smooth_pendulum")
    _, raw = worker_module._resolve_fixture_source(_MANIFEST, fixture.xml_path)
    model = worker_module._compile_model(
        worker_module._admit_self_contained_xml(raw), fixture, worker_module.Decimal("0.004")
    )
    live = worker_module._initial_data(model, fixture)
    state_signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state_size = int(mujoco.mj_stateSize(model, state_signature))
    live_before = worker_module._integration_state(model, live, state_signature, state_size)
    original_forward = worker_module.mujoco.mj_forward

    def nonfinite_forward(subject_model: Any, subject_data: Any) -> None:
        """Run the real observer pipeline, then corrupt one derived observer-only sensor."""
        original_forward(subject_model, subject_data)
        subject_data.sensordata[0] = math.nan

    monkeypatch.setattr(worker_module.mujoco, "mj_forward", nonfinite_forward)

    with pytest.raises(worker_module.WorkerRefusal, match="non-finite value"):
        worker_module._project_observation_data(
            model, live, mujoco.MjData(model), state_signature, state_size
        )

    live_after = worker_module._integration_state(model, live, state_signature, state_size)
    assert np.array_equal(live_after, live_before)


def test_nonfinite_solver_residual_is_a_time_local_gate_failure(
    worker_module: ModuleType,
) -> None:
    """A NaN solver residual is never filtered into a finite passing aggregate."""
    _, _, fixture = worker_module._load_manifest(_MANIFEST, "smooth_pendulum")
    _, raw = worker_module._resolve_fixture_source(_MANIFEST, fixture.xml_path)
    model = worker_module._compile_model(
        worker_module._admit_self_contained_xml(raw), fixture, worker_module.Decimal("0.004")
    )
    data = mujoco.MjData(model)
    data.solver_fwdinv[:] = np.nan
    _, residual = worker_module._solver_sample(model, data)
    assert math.isinf(residual)
    sample = worker_module._diagnostic_values(
        model, data, True, True, 0, residual, worker_module.Decimal("0")
    )
    assert sample["max_solver_residual"] is None
    assert sample["solver_converged"] is False


def test_external_host_projection_must_match_live_measurement(worker_module: ModuleType) -> None:
    """A retained host identity cannot replace or contradict the live worker measurement."""
    live = {
        "system": "Darwin",
        "release": "24.6.0",
        "machine": "x86_64",
        "logical_cpu_count": 12,
    }
    worker_module._admit_external_host({"host": dict(live)}, live)
    forged = dict(live)
    forged["machine"] = "arm64"
    with pytest.raises(worker_module.WorkerRefusal, match="does not match live"):
        worker_module._admit_external_host({"host": forged}, live)


@pytest.mark.parametrize("field", ["distance", "force_0", "force_5"])
def test_nonfinite_contact_numerics_refuse_before_aggregation(
    worker_module: ModuleType, field: str
) -> None:
    """No non-finite manifold value can be hidden by normal-force clipping or summation."""
    distance = 0.0
    force = np.zeros(6, dtype=np.float64)
    if field == "distance":
        distance = math.nan
    else:
        force[int(field.removeprefix("force_"))] = math.nan
    with pytest.raises(worker_module.WorkerRefusal, match="full force vector must be finite"):
        worker_module._admit_contact_numerics(distance, force)


def test_contact_force_aggregate_overflow_refuses(worker_module: ModuleType) -> None:
    """Two individually finite manifold forces cannot overflow the semantic pair aggregate."""
    maximum = float(np.finfo(np.float64).max)
    with pytest.raises(worker_module.WorkerRefusal, match="normal-force aggregate overflowed"):
        worker_module._finite_contact_sum(
            maximum, maximum, "semantic contact normal-force aggregate"
        )


def test_cumulative_contact_impulse_overflow_refuses(worker_module: ModuleType) -> None:
    """Finite force and accumulated impulse cannot overflow during timestep integration."""
    maximum = float(np.finfo(np.float64).max)
    with pytest.raises(worker_module.WorkerRefusal, match="cumulative impulse overflowed"):
        worker_module._accumulate_contact_impulse(maximum, maximum, worker_module.Decimal("0.004"))


def test_open_contact_episode_refuses_publication(worker_module: ModuleType) -> None:
    """A horizon-censored contact cannot masquerade as a complete event topology."""
    pair = worker_module.ContactPairSpec(
        "test.contact",
        ("geom_a", "geom_b"),
        worker_module.Decimal("1"),
        worker_module.Decimal("1"),
    )
    state = worker_module.ContactState(
        spec=pair,
        occupied=True,
        onset=worker_module.Decimal("0.5"),
        events=[{"event": "ONSET", "time": "0.5"}],
        segments=[],
        persistence_steps=10,
        aggregate_normal_impulse=1.0,
        current_normal_force=1.0,
    )
    with pytest.raises(worker_module.WorkerRefusal, match="remains open"):
        worker_module._finalize_contacts(
            {"test.contact": state}, worker_module.Decimal("1"), worker_module.Decimal("0.004")
        )


def _sentinel_subject(worker_module: ModuleType) -> tuple[Any, Any]:
    """Compile one fixture exposing public integration-state member families."""
    _, _, fixture = worker_module._load_manifest(_MANIFEST, "smooth_pendulum")
    xml = """
<mujoco model="sentinel_state">
  <compiler angle="radian"/>
  <size nuserdata="2"/>
  <option timestep="0.004" gravity="0 0 0" integrator="Euler"/>
  <worldbody>
    <body name="pendulum">
      <joint name="theta" type="hinge" axis="0 1 0"/>
      <geom name="pendulum_rod" type="capsule" fromto="0 0 0 0 0 -0.5"
            size="0.035" density="500" contype="0" conaffinity="0"/>
    </body>
    <body name="target" mocap="true" pos="0 0 0">
      <geom name="target_geom" size="0.01" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <equality>
    <weld name="tracking" body1="pendulum" body2="target"/>
  </equality>
  <actuator>
    <general name="drive" joint="theta" gear="1" dyntype="filter" dynprm="0.1"
             ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
  <sensor>
    <jointpos name="theta_position" joint="theta"/>
  </sensor>
</mujoco>
"""
    model = worker_module._compile_model(xml, fixture, worker_module.Decimal("0.004"))
    return model, fixture


def test_complete_public_integration_state_sentinel_passes(
    worker_module: ModuleType,
) -> None:
    """Two fresh data instances reproduce one complete public integration state exactly."""
    model, fixture = _sentinel_subject(worker_module)

    sentinel = worker_module._same_profile_sentinel(model, fixture, "baseline")

    assert sentinel["status"] == "PASS"
    assert sentinel["failure_reason"] is None
    assert sentinel["state_signature"] == int(mujoco.mjtState.mjSTATE_INTEGRATION)
    assert sentinel["state_size"] == mujoco.mj_stateSize(
        model, int(mujoco.mjtState.mjSTATE_INTEGRATION)
    )
    assert sentinel["warmup_step_count"] == 2
    _verify_self_hash(sentinel, "sentinel_identity_sha256")
    assert (
        sentinel["post_forward_projection"]["left"] == sentinel["post_forward_projection"]["right"]
    )
    assert (
        sentinel["post_step_integration_state"]["left"]
        == sentinel["post_step_integration_state"]["right"]
    )


@pytest.mark.parametrize(
    "member",
    [
        "qpos",
        "qvel",
        "ctrl",
        "qacc_warmstart",
        "act",
        "qfrc_applied",
        "xfrc_applied",
        "mocap_pos",
        "mocap_quat",
        "eq_active",
        "userdata",
    ],
    ids=(
        "position",
        "velocity",
        "control",
        "warm-start",
        "history",
        "generalized-force",
        "body-force",
        "mocap-position",
        "mocap-orientation",
        "equality-active",
        "user-state",
    ),
)
def test_public_integration_state_mutation_fails_before_forward(
    worker_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    member: str,
) -> None:
    """Every present public integration-state family is bound to the retrieved source."""
    model, fixture = _sentinel_subject(worker_module)
    original = worker_module.mujoco.mj_setState

    def mutate_restored_state(
        model_value: Any, data_value: Any, state: Any, signature: int
    ) -> None:
        """Apply the public restore, then alter one decision-bearing state member."""
        original(model_value, data_value, state, signature)
        target = np.asarray(getattr(data_value, member))
        assert target.size > 0
        flattened = target.reshape(-1)
        if np.issubdtype(flattened.dtype, np.integer) or np.issubdtype(flattened.dtype, np.bool_):
            flattened[0] = 0 if int(flattened[0]) else 1
        else:
            flattened[0] += 0.25

    monkeypatch.setattr(worker_module.mujoco, "mj_setState", mutate_restored_state)

    sentinel = worker_module._same_profile_sentinel(model, fixture, "baseline")

    assert sentinel["status"] == "FAIL"
    assert "retrieved source" in sentinel["failure_reason"]
    assert sentinel["post_forward_projection"] is None


def test_genuinely_absent_plugin_state_is_not_synthesized(worker_module: ModuleType) -> None:
    """A runtime without fixture plugin state records genuine absence through state size only."""
    model, fixture = _sentinel_subject(worker_module)
    data = worker_module._initial_data(model, fixture)

    assert np.asarray(data.plugin_state).size == 0
    assert worker_module._same_profile_sentinel(model, fixture, "baseline")["status"] == "PASS"


def test_nonfinite_restored_state_fails_closed(
    worker_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nonfinite complete-state mutation cannot reach a passing projection."""
    model, fixture = _sentinel_subject(worker_module)
    original = worker_module.mujoco.mj_setState

    def inject_nonfinite(model_value: Any, data_value: Any, state: Any, signature: int) -> None:
        """Restore through the public API and inject one nonfinite coordinate."""
        original(model_value, data_value, state, signature)
        data_value.qpos[0] = math.nan

    monkeypatch.setattr(worker_module.mujoco, "mj_setState", inject_nonfinite)

    sentinel = worker_module._same_profile_sentinel(model, fixture, "baseline")

    assert sentinel["status"] == "FAIL"
    assert "non-finite" in sentinel["failure_reason"]


@pytest.mark.parametrize(
    ("projection_helper", "failed_fact", "reason"),
    [
        ("_sentinel_warning_projection", "warnings_passed", "warning counter"),
        ("_sentinel_solver_projection", "solver_converged", "solver diagnostics"),
    ],
    ids=("warning", "solver"),
)
def test_failed_runtime_diagnostic_fails_sentinel(
    worker_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    projection_helper: str,
    failed_fact: str,
    reason: str,
) -> None:
    """A warning or solver gate failure cannot produce a passing sentinel."""
    model, fixture = _sentinel_subject(worker_module)
    original = getattr(worker_module, projection_helper)

    def fail_projection(*args: Any) -> dict[str, Any]:
        """Preserve observed details while forcing the selected gate fact false."""
        projection = original(*args)
        projection["passed"] = False
        return projection

    monkeypatch.setattr(worker_module, projection_helper, fail_projection)

    sentinel = worker_module._same_profile_sentinel(model, fixture, "baseline")

    assert sentinel["status"] == "FAIL"
    assert sentinel[failed_fact] is False
    assert reason in sentinel["failure_reason"]


def test_output_projection_mutation_fails_exact_comparison(
    worker_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-hashed output mutation in one fresh data instance is still detected."""
    model, fixture = _sentinel_subject(worker_module)
    original = worker_module._sentinel_projection
    calls = 0

    def mutate_one_projection(model_value: Any, data_value: Any) -> dict[str, Any]:
        """Alter and self-hash only the second deterministic projection."""
        nonlocal calls
        calls += 1
        projection = original(model_value, data_value)
        if calls == 2:
            projection["qacc"]["values"][0] = "123"
            projection["projection_sha256"] = worker_module._canonical_sha256(
                {key: value for key, value in projection.items() if key != "projection_sha256"}
            )
        return projection

    monkeypatch.setattr(worker_module, "_sentinel_projection", mutate_one_projection)

    sentinel = worker_module._same_profile_sentinel(model, fixture, "baseline")

    assert sentinel["status"] == "FAIL"
    assert "post-forward projections differ" in sentinel["failure_reason"]


def test_callback_authority_refuses_sentinel(
    worker_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any active process-wide MuJoCo callback blocks deterministic sentinel authority."""
    model, fixture = _sentinel_subject(worker_module)
    monkeypatch.setattr(worker_module.mujoco, "get_mjcb_control", lambda: object())

    sentinel = worker_module._same_profile_sentinel(model, fixture, "baseline")

    assert sentinel["status"] == "FAIL"
    assert "callback authority is active" in sentinel["failure_reason"]
    assert sentinel["pre_restore_integration_state"] is None


def test_contact_slot_reorder_preserves_semantic_projection(
    worker_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw contact-slot order cannot affect the canonical semantic contact projection."""
    contacts = [
        SimpleNamespace(
            geom1=1,
            geom2=0,
            dim=3,
            dist=-0.01,
            pos=np.asarray([1.0, 0.0, 0.0]),
            frame=np.arange(9, dtype=np.float64),
            friction=np.arange(5, dtype=np.float64),
            force=np.asarray([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
        SimpleNamespace(
            geom1=2,
            geom2=3,
            dim=1,
            dist=-0.02,
            pos=np.asarray([0.0, 1.0, 0.0]),
            frame=np.arange(9, dtype=np.float64) + 1,
            friction=np.arange(5, dtype=np.float64) + 1,
            force=np.asarray([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
    ]
    names = {0: "alpha", 1: "beta", 2: "gamma", 3: "delta"}

    def semantic_name(_model: Any, _kind: Any, identifier: int) -> str:
        """Return stable semantic names for synthetic contacts."""
        return names[identifier]

    def contact_force(_model: Any, data: Any, index: int, output: Any) -> None:
        """Return force associated with the contact object, independent of its slot."""
        output[:] = data.contact[index].force

    monkeypatch.setattr(worker_module.mujoco, "mj_id2name", semantic_name)
    monkeypatch.setattr(worker_module.mujoco, "mj_contactForce", contact_force)
    forward = worker_module._sentinel_contact_projection(
        object(), SimpleNamespace(ncon=2, contact=list(contacts))
    )
    reordered = worker_module._sentinel_contact_projection(
        object(), SimpleNamespace(ncon=2, contact=list(reversed(contacts)))
    )

    assert reordered == forward


@pytest.mark.parametrize(
    ("package_version", "native_version", "native_integer"),
    [
        ("3.12.0.post2", "3.12.0", 3_012_000),
        ("3.12.0+vendor.1", "3.12.0", 3_012_000),
        ("4.2.1", "4.2.1", 4_002_001),
    ],
    ids=("post-release", "vendor-build", "future-stable"),
)
def test_semantic_worker_role_admits_coherent_stable_runtime(
    worker_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    package_version: str,
    native_version: str,
    native_integer: int,
) -> None:
    """The generic worker preserves exact stable package tokens without a minor allowlist."""
    for name, value in _THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(worker_module.mujoco, "__version__", package_version)
    monkeypatch.setattr(worker_module.mujoco, "mj_versionString", lambda: native_version)
    monkeypatch.setattr(worker_module.mujoco, "mj_version", lambda: native_integer)

    measured = worker_module._production_profile_contract("candidate")

    assert measured == (package_version, native_version, native_integer)


@pytest.mark.parametrize(
    ("capability", "replacement"),
    [("mj_step", None), ("mj_getState", object())],
    ids=("missing", "noncallable"),
)
def test_incomplete_worker_call_graph_is_refused(
    worker_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capability: str,
    replacement: object,
) -> None:
    """A missing or noncallable cell capability blocks profile admission."""
    original = worker_module._resolve_capability

    def substitute(name: str) -> object | None:
        """Replace exactly one required callable capability."""
        return replacement if name == capability else original(name)

    monkeypatch.setattr(worker_module, "_resolve_capability", substitute)

    with pytest.raises(worker_module.WorkerRefusal, match="lacks required"):
        worker_module._require_production_capabilities()


def test_missing_worker_enum_authority_is_refused(
    worker_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing semantic enum member blocks capability-complete profile admission."""
    original = worker_module._resolve_capability

    def omit_semantic_enum(name: str) -> object | None:
        """Remove one exact semantic enum from the required worker call graph."""
        if name == "mjtObj.mjOBJ_SENSOR":
            return None
        return original(name)

    monkeypatch.setattr(worker_module, "_resolve_capability", omit_semantic_enum)

    with pytest.raises(worker_module.WorkerRefusal, match="mjOBJ_SENSOR"):
        worker_module._require_production_capabilities()
