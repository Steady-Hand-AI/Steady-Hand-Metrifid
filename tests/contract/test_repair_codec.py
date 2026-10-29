"""Bounded decimal repair records, exercised without physical evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from metrifid.repair import _codec, _engine, _profile
from metrifid.repair._physical_types import Measurement, Snapshot
from metrifid.repair._request import RepairController, RepairInputError, RepairRequest, RepairTask
from metrifid.repair._types import EngineResult, ExecutionCosts, PairedMeasurement, ReplayResult

_RUNTIME = {
    "implementation": "CPython",
    "python": "3.13.5",
    "system": "Linux",
    "machine": "x86_64",
    "mujoco": "3.16.0",
    "numpy": "2.5.4",
}

_LEGACY_LIMITATIONS = [
    "Simulation only.",
    "Only planar_push_timed15_v1 on Linux x86_64, CPython 3.13, MuJoCo 3.16.0 and NumPy 2.5.4.",
    "No global-optimum, robustness, authorship or cross-platform bitwise replay guarantee.",
]
_CURRENT_LIMITATIONS = [
    "Simulation only.",
    "Only planar_push_timed15_v1 on Linux x86_64, CPython 3.13, MuJoCo 3.16.0 or 3.17.0 and NumPy 2.5.4.",
    "No global-optimum, robustness, authorship or cross-platform bitwise replay guarantee.",
]


def _pair(vector: tuple[float, ...], *, qualified: bool) -> PairedMeasurement:
    cases = []
    for dt in (0.002, 0.001):
        hold = (0.004 if qualified else 0.006, 0.01, 0.001, 0.01)
        metrics = {
            "dt_s": dt,
            "episode_time_s": 4.0,
            "physics_steps": round(4 / dt),
            "requested_physics_steps": round(4 / dt),
            "hold_samples": round(0.2 / dt) + 1,
            "expected_hold_samples": round(0.2 / dt) + 1,
            "endpoint_errors": list(hold),
            "warnings": [],
            "hold_max_position_error_m": hold[0],
            "hold_max_yaw_error_rad": hold[1],
            "hold_max_linear_speed_m_s": hold[2],
            "hold_max_angular_speed_rad_s": hold[3],
            "schedule": {
                "boundaries_s": [0.0, 1.0, 2.0, 3.0],
                "decoded_boundaries_s": [0.0, 1.0, 2.0, 3.0],
                "motion_deadlines_s": [1.0, 2.0, 3.0],
                "scheduled_starts_s": [0.0, 1.0, 2.0],
                "stationary_tail_start_s": 3.0,
                "stroke_windows_s": [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
                "timing_parameters_normalized": [0.5, 0.5, 0.5],
            },
        }
        cases.append(
            Measurement(
                "COMPLETE",
                qualified,
                "COMPLETE",
                hold,
                metrics,
                completed_steps=round(4 / dt),
                expected_steps=round(4 / dt),
                dt_s=dt,
                hold_samples=round(0.2 / dt) + 1,
                expected_hold_samples=round(0.2 / dt) + 1,
            )
        )
    pair = PairedMeasurement(vector, *cases, known_completed_steps=6000)
    _engine._classify_pair(pair)
    return pair


def _result(*, qualified: bool = True) -> EngineResult:
    supplied = RepairRequest(
        RepairTask(-0.06, (0.08, 0.06), math.pi / 2), RepairController((0.5,) * 15)
    )
    profile, nominal, finer, identities = _profile.generate_expected(supplied)
    shapes = {"qpos": (9,), "qvel": (8,), "act": (0,), "mocap_pos": (0, 3), "mocap_quat": (0, 4)}
    arrays = tuple((name, shape, (0.0,) * math.prod(shape)) for name, shape in shapes.items())
    content = b"".join(
        struct.pack("<" + "d" * len(values), *values) for _, _, values in sorted(arrays)
    )
    reset = Snapshot(
        arrays,
        (0.0, 0.0, 0.01),
        0.0,
        identities["nominal_scene_sha256"],
        hashlib.sha256(content).hexdigest(),
    )
    original = supplied.controller.parameters_normalized
    selected = (math.nextafter(0.5, 1.0), *original[1:]) if qualified else original
    first = _pair(original, qualified=False)
    chosen = _pair(selected, qualified=True) if qualified else first
    count = 18200 if qualified else 6200
    return EngineResult(
        "REPAIRED" if qualified else "NO_QUALIFIED_POLICY_FOUND",
        0 if qualified else 10,
        "COMPLETE_PAIRED_CONFIRMATION" if qualified else "INTEGRATION_STEP_RESERVE",
        request=supplied,
        original_parameters=original,
        selected_parameters=selected,
        profile=profile,
        nominal_xml=nominal,
        finer_xml=finer,
        timestep_identities=identities,
        reset=reset,
        initialization=Measurement(
            "COMPLETE", True, "ADMITTED", completed_steps=200, expected_steps=200, dt_s=0.002
        ),
        runtime=_RUNTIME.copy(),
        initial_screening=first,
        selected_screening=chosen,
        screening_measurements=[first, chosen],
        confirmation=copy.deepcopy(chosen) if qualified else None,
        confirmation_complete=qualified,
        qualified=qualified,
        costs=ExecutionCosts(2 if qualified else 1, count, count, 0, 0.125, supplied.limits, 0.75),
    )


def _document(result: EngineResult | None = None):
    raw, members = _codec.serialize_run(_result() if result is None else result)
    return _codec.decode_document(raw), members


def _refresh_member(document, members, name, value):
    raw = value if isinstance(value, bytes) else _codec.encode_json(value)
    members[name] = raw
    document["artifacts"][name] = {
        "filename": name,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def test_encoding_and_parameter_identity_preserve_binary64() -> None:
    """Encoding has one specified byte representation, including signed zero."""
    assert _codec.encode_json({"b": -0.0, "a": 0.1}) == b'{"a":0.1,"b":-0.0}\n'
    result = _result()
    raw, members = _codec.serialize_run(result)
    document = _codec.decode_document(raw)
    selected_raw = (
        json.dumps(
            list(result.selected_parameters),
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    assert document["selected_parameters_sha256"] == hashlib.sha256(selected_raw).hexdigest()
    admitted = _codec.admit_record(document, members)
    assert admitted.selected_parameters == result.selected_parameters
    assert admitted.selected_parameters[0].hex() == result.selected_parameters[0].hex()
    assert "screening_measurements" not in document
    assert admitted.screening_measurements == []


@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize("mujoco", ["3.16.0", "3.17.0"])
def test_run_round_trip_reconstitutes_typed_evidence(qualified: bool, mujoco: str) -> None:
    """Both confirmed recovery and finite nonqualification retain their typed evidence."""
    result = _result(qualified=qualified)
    result.runtime["mujoco"] = mujoco
    raw, members = _codec.serialize_run(result)
    document = _codec.decode_document(raw)
    assert document["limitations"] == _CURRENT_LIMITATIONS
    admitted = _codec.admit_record(document, members)
    assert isinstance(admitted, EngineResult)
    assert admitted == replace(result, screening_measurements=[])
    repeated, repeated_members = _codec.serialize_run(admitted)
    assert repeated == raw
    assert repeated_members == members


def test_missing_selection_error_is_readable_without_invented_artifacts() -> None:
    """A failed setup can retain a useful error without fake state or measurements."""
    request = _result().request
    failed = EngineResult(
        "ERROR",
        70,
        "EVALUATOR_EXCEPTION",
        request=request,
        original_parameters=request.controller.parameters_normalized,
        diagnostic="controlled setup exception",
        costs=ExecutionCosts(limits=request.limits, search_cpu_fraction=0.75),
    )
    raw, members = _codec.serialize_run(failed)
    admitted = _codec.admit_record(_codec.decode_document(raw), members)
    assert admitted == failed
    assert members == {}
    assert _codec.decode_document(raw)["unavailable_measurements"] == [
        "confirmation",
        "initial_screening",
        "initialization",
        "selected_screening",
    ]


def test_incomplete_nonfinite_metrics_are_explicit_nulls_with_paths() -> None:
    """Unknown measured values stay unavailable and cannot become qualification."""
    result = _result(qualified=False)
    incomplete = Measurement(
        "INCOMPLETE",
        False,
        "NONFINITE_STATE",
        (math.nan, 0.01, 0.0, 0.0),
        {"hold_max_position_error_m": math.inf, "endpoint_errors": [math.nan, 0.01, 0.0, 0.0]},
        completed_steps=7,
        expected_steps=2000,
        dt_s=0.002,
    )
    confirmation = PairedMeasurement(
        result.selected_parameters, incomplete, reason="NONFINITE_STATE", known_completed_steps=7
    )
    result.status, result.exit_code, result.reason = "INCOMPLETE", 20, "NONFINITE_STATE"
    result.confirmation = confirmation
    result.costs.integration_steps = result.costs.known_completed_steps = 6207
    raw, members = _codec.serialize_run(result)
    document = _codec.decode_document(raw)
    assert b"NaN" not in raw
    assert b"Infinity" not in raw
    assert document["confirmation"]["nominal"]["hold_maxima"][0] is None
    assert "confirmation.nominal.hold_maxima[0]" in document["unavailable_measurements"]
    assert "confirmation.nominal.metrics.endpoint_errors[0]" in document["unavailable_measurements"]
    assert (
        "confirmation.nominal.metrics.hold_max_position_error_m"
        in document["unavailable_measurements"]
    )
    admitted = _codec.admit_record(document, members)
    assert admitted.status == "INCOMPLETE"
    assert not admitted.qualified


def test_nonfinite_request_or_selected_controller_is_never_sanitized() -> None:
    """Unavailable-metric normalization cannot repair invalid input vectors."""
    result = _result()
    result.selected_parameters = (math.nan, *result.selected_parameters[1:])
    with pytest.raises(RepairInputError):
        _codec.serialize_run(result)


def _replay_result() -> ReplayResult:
    source = _result()
    return ReplayResult(
        "QUALIFIED",
        0,
        "COMPLETE_PAIRED_EVALUATION",
        source.request,
        source.selected_parameters,
        source.confirmation,
        True,
        _RUNTIME.copy(),
        ExecutionCosts(0, 6200, 6200, 0, 0.2, source.request.limits, None),
    )


@pytest.mark.parametrize("mujoco", ["3.16.0", "3.17.0"])
def test_standalone_replay_round_trip_needs_no_members(mujoco: str) -> None:
    """Fresh replay remains a small standalone result with source linkage."""
    result = _replay_result()
    result.runtime["mujoco"] = mujoco
    raw = _codec.serialize_replay(result, "a" * 64)
    document = _codec.decode_document(raw)
    assert document["limitations"] == _CURRENT_LIMITATIONS
    assert document["source_repair_sha256"] == "a" * 64
    assert _codec.artifact_records(document) == {}
    assert _codec.admit_record(document, {}) == result
    with pytest.raises(RepairInputError):
        _codec.admit_record(document, {"nominal.xml": b"not a replay member"})


def _record(result: EngineResult | ReplayResult):
    if isinstance(result, EngineResult):
        raw, members = _codec.serialize_run(result)
    else:
        raw, members = _codec.serialize_replay(result, "a" * 64), {}
    return _codec.decode_document(raw), members


@pytest.mark.parametrize("operation", ["run", "replay"])
def test_exact_legacy_limitations_preserve_recorded_316_success(operation: str) -> None:
    """Pre-extension records retain their measured outcome and runtime identity."""
    result = _result() if operation == "run" else _replay_result()
    document, members = _record(result)
    document["limitations"] = _LEGACY_LIMITATIONS.copy()
    admitted = _codec.admit_record(document, members)
    assert admitted.status == result.status
    assert admitted.runtime == _RUNTIME
    assert admitted.qualified
    assert admitted.costs == result.costs


@pytest.mark.parametrize("operation", ["run", "replay"])
@pytest.mark.parametrize(
    "runtime", [{}, {"implementation": "CPython"}, {"mujoco": "3.16.0"}, {"mujoco": "3.17.0"}]
)
def test_legacy_refusal_does_not_claim_runtime_execution(operation: str, runtime) -> None:
    """Absent or partial discovery is valid when a legacy invocation never executed."""
    result = (
        EngineResult("REFUSED", 64, "UNSUPPORTED_RUNTIME", runtime=runtime)
        if operation == "run"
        else ReplayResult("REFUSED", 64, "UNSUPPORTED_RUNTIME", runtime=runtime)
    )
    document, members = _record(result)
    document["limitations"] = _LEGACY_LIMITATIONS.copy()
    admitted = _codec.admit_record(document, members)
    assert admitted == result
    assert not admitted.qualified
    assert admitted.costs.known_completed_steps == 0


@pytest.mark.parametrize("kind", ["run", "negative_run", "replay", "setup_refusal", "unknown_call"])
def test_317_execution_cannot_claim_legacy_316_limitations(kind: str) -> None:
    """New-runtime work must carry the exact extended support statement."""
    result = _result(qualified=kind != "negative_run") if "run" in kind else _replay_result()
    if kind in ("setup_refusal", "unknown_call"):
        result = replace(
            result,
            status="REFUSED",
            exit_code=64,
            reason="RESET_MISMATCH",
            measurement=None,
            qualified=False,
            costs=ExecutionCosts(
                integration_steps=200 if kind == "setup_refusal" else None,
                known_completed_steps=200 if kind == "setup_refusal" else 0,
                unknown_step_calls=int(kind == "unknown_call"),
                limits=result.request.limits,
            ),
        )
    result.runtime["mujoco"] = "3.17.0"
    document, members = _record(result)
    assert _codec.admit_record(document, members).runtime["mujoco"] == "3.17.0"
    document["limitations"] = _LEGACY_LIMITATIONS.copy()
    with pytest.raises(RepairInputError, match="limitations"):
        _codec.admit_record(document, members)


@pytest.mark.parametrize(
    "limitations",
    [
        list(reversed(_LEGACY_LIMITATIONS)),
        _CURRENT_LIMITATIONS[:-1],
        ["Simulation only.", "Any installed runtime.", _CURRENT_LIMITATIONS[-1]],
    ],
)
def test_limitations_compatibility_is_not_arbitrary_text(limitations) -> None:
    """Only the two complete documented representations are admitted."""
    document, members = _document()
    document["limitations"] = limitations
    with pytest.raises(RepairInputError, match="limitations"):
        _codec.admit_record(document, members)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"a":1,"a":2}',
        b'{"a":{"b":1,"b":2}}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":1e309}',
        b"\xef\xbb\xbf{}",
        b'{"a":"\xff"}',
        b'{"a":"\\ud800"}',
    ],
)
def test_decoder_rejects_nondecimal_or_ambiguous_json(payload: bytes) -> None:
    """Duplicate fields, invalid Unicode and nonfinite tokens never reach admission."""
    with pytest.raises(RepairInputError):
        _codec.decode_document(payload)


def test_exact_document_structure_bounds() -> None:
    """Byte, node, depth and UTF-8 string limits are inclusive and independent."""
    assert _codec.decode_document(b"{}" + b" " * (_codec.MAX_RESULT_BYTES - 2)) == {}
    with pytest.raises(RepairInputError):
        _codec.decode_document(b"{}" + b" " * (_codec.MAX_RESULT_BYTES - 1))
    assert _codec.decode_document(json.dumps({"x": [0] * 15997}).encode())["x"][-1] == 0
    with pytest.raises(RepairInputError):
        _codec.decode_document(json.dumps({"x": [0] * 15998}).encode())
    value = 0
    for _ in range(16):
        value = {"x": value}
    _codec.decode_document(json.dumps(value).encode())
    with pytest.raises(RepairInputError):
        _codec.decode_document(json.dumps({"x": value}).encode())
    assert _codec.decode_document(json.dumps({"x": "é" * 32768}, ensure_ascii=False).encode())["x"]
    with pytest.raises(RepairInputError):
        _codec.decode_document(json.dumps({"x": "é" * 32769}, ensure_ascii=False).encode())


@pytest.mark.parametrize(
    "domain", ["root", "measurement", "metrics", "schedule", "costs", "policy", "reset"]
)
def test_unknown_keys_are_rejected_at_each_public_domain(domain: str) -> None:
    """Bounded objects are explicit schema domains, not extensible payload bags."""
    document, members = _document()
    if domain == "root":
        document["extra"] = 1
    elif domain == "measurement":
        document["confirmation"]["nominal"]["extra"] = 1
    elif domain == "metrics":
        document["confirmation"]["nominal"]["metrics"]["extra"] = 1
    elif domain == "schedule":
        document["confirmation"]["nominal"]["metrics"]["schedule"]["extra"] = 1
    elif domain == "costs":
        document["costs"]["extra"] = 1
    else:
        filename = domain + ".json"
        value = _codec.decode_document(members[filename])
        value["extra"] = 1
        _refresh_member(document, members, filename, value)
    with pytest.raises(RepairInputError):
        _codec.admit_record(document, members)


@pytest.mark.parametrize(
    "domain", ["version", "cost", "count", "profile", "method", "identity", "hold"]
)
def test_boolean_numeric_and_decimal_integer_substitutions_refuse(domain: str) -> None:
    """Python's True-equals-one shortcut cannot admit numeric record fields."""
    document, members = _document()
    if domain == "version":
        document["version"] = True
    elif domain == "cost":
        document["costs"]["cpu_seconds"] = True
    elif domain == "count":
        document["costs"]["attempts"] = 2.0
    elif domain == "profile":
        document["profile"]["scene"]["floor"]["surface_z"] = False
    elif domain == "method":
        document["method"]["settings"]["initial_preferred_sign"] = True
    elif domain == "identity":
        document["timestep_identities"]["nominal_dt_s"] = False
    else:
        document["confirmation"]["nominal"]["hold_maxima"][0] = False
    with pytest.raises(RepairInputError):
        _codec.admit_record(document, members)


@pytest.mark.parametrize(
    "change",
    [
        "exit",
        "qualification",
        "missing_confirmation",
        "changed_status",
        "zero_cost",
        "unknown_work",
        "wrong_vector",
        "incomplete_case",
    ],
)
def test_success_requires_consistent_fresh_measured_witnesses(change: str) -> None:
    """A status string and hashes cannot replace qualifying paired evidence."""
    document, members = _document()
    if change == "exit":
        document["exit_code"] = 10
    elif change == "qualification":
        document["qualified"] = False
    elif change == "missing_confirmation":
        document["confirmation"] = None
        document["confirmation_complete"] = False
        document["unavailable_measurements"].append("confirmation")
        document["unavailable_measurements"].sort()
    elif change == "changed_status":
        document["status"] = "ALREADY_QUALIFIED"
    elif change == "zero_cost":
        document["costs"]["integration_steps"] = 0
        document["costs"]["known_completed_steps"] = 0
    elif change == "unknown_work":
        document["costs"]["integration_steps"] = None
        document["costs"]["unknown_step_calls"] = 1
    elif change == "wrong_vector":
        document["confirmation"]["parameters_normalized"][0] = 0.7
    else:
        document["confirmation"]["finer"]["completed_steps"] = 3999
    with pytest.raises(RepairInputError):
        _codec.admit_record(document, members)


def test_recomputed_scene_digest_cannot_admit_arbitrary_xml() -> None:
    """Only independently generated scene bytes are accepted, even with new hashes."""
    document, members = _document()
    _refresh_member(document, members, "nominal.xml", members["nominal.xml"] + b" ")
    document["timestep_identities"]["nominal_scene_sha256"] = document["artifacts"]["nominal.xml"][
        "sha256"
    ]
    with pytest.raises(RepairInputError):
        _codec.admit_record(document, members)


def test_reset_layout_state_digest_and_policy_binding_are_verified() -> None:
    """A recomputed member hash does not excuse an invalid reset or policy identity."""
    for target in ("reset.json", "policy.json"):
        document, members = _document()
        value = _codec.decode_document(members[target])
        if target == "reset.json":
            value["arrays"][0]["values"][0] = 0.01
        else:
            value["parameters_normalized"][0] = 0.8
        _refresh_member(document, members, target, value)
        with pytest.raises(RepairInputError):
            _codec.admit_record(document, members)


def test_valid_consistent_controller_substitution_is_not_authenticated() -> None:
    """An internally consistent supported replacement is available for fresh replay."""
    document, members = _document()
    vector = [0.75, *document["selected_parameters"][1:]]
    identity = hashlib.sha256(_codec.encode_json(vector)).hexdigest()
    document["selected_parameters"], document["selected_parameters_sha256"] = vector, identity
    document["selected_screening"]["parameters_normalized"] = vector.copy()
    document["confirmation"]["parameters_normalized"] = vector.copy()
    policy = _codec.decode_document(members["policy.json"])
    policy["parameters_normalized"], policy["parameters_sha256"] = vector, identity
    _refresh_member(document, members, "policy.json", policy)
    admitted = _codec.admit_record(document, members)
    assert admitted.selected_parameters == tuple(vector)


def test_member_allowlist_bounds_hashes_and_unavailable_paths() -> None:
    """Artifact lookup is fixed-name and measured nulls cannot be hidden."""
    document, members = _document()
    document["artifacts"]["../escape"] = {
        "filename": "../escape",
        "size_bytes": 0,
        "sha256": "a" * 64,
    }
    with pytest.raises(RepairInputError):
        _codec.artifact_records(document)
    document, members = _document()
    document["artifacts"]["policy.json"]["size_bytes"] = 16385
    with pytest.raises(RepairInputError):
        _codec.artifact_records(document)
    document, members = _document()
    members["policy.json"] += b" "
    with pytest.raises(RepairInputError):
        _codec.admit_record(document, members)
    document, members = _document()
    document["unavailable_measurements"] = []
    with pytest.raises(RepairInputError):
        _codec.admit_record(document, members)


@pytest.mark.parametrize(
    ("mujoco", "legacy"), [("3.16.0", True), ("3.16.0", False), ("3.17.0", False)]
)
def test_native_free_public_codec_in_fresh_process(
    tmp_path: Path, mujoco: str, legacy: bool
) -> None:
    """Legacy and current bundles remain readable with both native imports forbidden."""
    result = _result()
    result.runtime["mujoco"] = mujoco
    document, members = _record(result)
    if legacy:
        document["limitations"] = _LEGACY_LIMITATIONS.copy()
    (tmp_path / "repair.json").write_bytes(_codec.encode_json(document))
    for name, contents in members.items():
        (tmp_path / name).write_bytes(contents)
    script = r"""
import importlib.abc, json, sys
from pathlib import Path
from types import SimpleNamespace
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'mujoco'):
            raise AssertionError('native import forbidden: ' + fullname)
sys.meta_path.insert(0, Deny())
from metrifid.repair import _codec, read_result_file
root = Path(sys.argv[1])
document = _codec.decode_document((root/'repair.json').read_bytes())
members = {name:(root/name).read_bytes() for name in _codec.artifact_records(document)}
result = _codec.admit_record(document,members)
assert result.status == 'REPAIRED' and result.qualified
assert result.runtime['mujoco'] == sys.argv[2]
read = read_result_file(root/'repair.json')
assert read.command_status == 'READ' and read.command_exit_code == 0
assert read.recorded_status == 'REPAIRED' and read.recorded_exit_code == 0
assert read.document['runtime']['mujoco'] == sys.argv[2]
assert 'numpy' not in sys.modules and 'mujoco' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(tmp_path), mujoco],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_post_evaluation_cpu_cutoff_retains_unadjudicated_complete_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The accepted engine can interrupt after complete cases but before pair adjudication."""
    seed = _result(qualified=False)
    clock = [0.0]

    def evaluate(parameters, *, finer, check, step):
        measured = seed.initial_screening.finer if finer else seed.initial_screening.nominal
        for _ in range(measured.completed_steps):
            step(lambda: None)
        if finer:
            clock[0] = 90.0
        return measured

    def create(request, budget):
        for _ in range(200):
            budget.step(lambda: None)
        return SimpleNamespace(
            profile=seed.profile,
            nominal_xml=seed.nominal_xml,
            finer_xml=seed.finer_xml,
            identities=seed.timestep_identities,
            snapshot=seed.reset,
            initialization=seed.initialization,
            evaluate=evaluate,
        )

    monkeypatch.setattr(_engine, "_runtime", lambda: _RUNTIME.copy())
    monkeypatch.setattr(_engine, "_create_backend", create)
    monkeypatch.setattr(_engine.time, "process_time", lambda: clock[0])
    result = _engine.repair_controller(seed.request)
    assert result.status == "INCOMPLETE"
    assert result.reason == "CPU_LIMIT"
    assert result.initial_screening.finer.classification == "COMPLETE"
    assert result.initial_screening.classification == "INCOMPLETE"
    assert result.initial_screening.objective is None
    assert result.costs.integration_steps == 6200
    raw, members = _codec.serialize_run(result)
    admitted = _codec.admit_record(_codec.decode_document(raw), members)
    assert admitted == replace(result, screening_measurements=[])
    assert admitted.screening_measurements == []
    assert not admitted.qualified


@pytest.mark.parametrize(
    "fault",
    [
        "pending_initial",
        "pending_confirmation",
        "qualified_confirmation",
        "search_cpu_cutoff",
        "confirmation_cpu_cutoff",
    ],
)
def test_finite_negative_rejects_unknown_confirmation_and_cpu_exhaustion(fault: str) -> None:
    """Interrupted/unadjudicated work cannot masquerade as a finite exit ten."""
    result = _result(qualified=False)
    if fault == "pending_initial":
        result.initial_screening = PairedMeasurement(result.original_parameters, reason="CPU_LIMIT")
        result.selected_screening = None
    elif fault in ("pending_confirmation", "confirmation_cpu_cutoff"):
        result.confirmation = PairedMeasurement(result.selected_parameters, reason="CPU_LIMIT")
        if fault == "confirmation_cpu_cutoff":
            result.confirmation = copy.deepcopy(result.selected_screening)
            result.confirmation_complete = True
            result.costs.cpu_seconds = 120.0
            result.costs.integration_steps = result.costs.known_completed_steps = 12200
    elif fault == "qualified_confirmation":
        result.confirmation = _pair(result.selected_parameters, qualified=True)
        result.confirmation_complete = True
        result.costs.integration_steps = result.costs.known_completed_steps = 12200
    else:
        result.costs.cpu_seconds = 90.0
    with pytest.raises(RepairInputError):
        _codec.serialize_run(result)


@pytest.mark.parametrize(
    "fault", ["warning", "penetration", "force", "reference_speed", "false_qualification"]
)
def test_complete_case_cannot_contradict_supplied_physical_measurements(fault: str) -> None:
    """Optional supplied diagnostics cannot contradict a recorded complete case."""
    document, members = _document()
    measured = document["confirmation"]["nominal"]
    if fault == "warning":
        measured["metrics"]["warnings"] = [{"index": 0, "number": 1, "lastinfo": 0}]
    elif fault == "penetration":
        measured["metrics"]["maximum_penetration_m"] = 0.01
    elif fault == "force":
        measured["metrics"]["max_command_norm_N"] = 3.0
    elif fault == "reference_speed":
        measured["metrics"]["max_reference_speed_m_s"] = 0.13
    else:
        measured["qualified"] = False
    with pytest.raises(RepairInputError):
        _codec.admit_record(document, members)


def test_known_hard_initial_rejection_remains_finite_readable() -> None:
    """A known physical initial rejection is distinct from unknown measurement loss."""
    result = _result(qualified=False)
    hard = Measurement(
        "HARD_REJECTION",
        False,
        "BLOCK_TILT",
        hard_failures=("BLOCK_TILT",),
        completed_steps=7,
        expected_steps=2000,
        dt_s=0.002,
    )
    first = PairedMeasurement(
        result.original_parameters,
        hard,
        classification="HARD_REJECTION",
        reason="PHYSICAL_HARD_REJECTION",
        complete=True,
        known_completed_steps=7,
    )
    result.initial_screening, result.selected_screening = first, None
    result.selected_parameters = None
    result.reason = "INITIAL_POLICY_NOT_SEARCHABLE"
    result.costs.integration_steps = result.costs.known_completed_steps = 207
    raw, members = _codec.serialize_run(result)
    admitted = _codec.admit_record(_codec.decode_document(raw), members)
    assert admitted.status == "NO_QUALIFIED_POLICY_FOUND"
    assert admitted.costs.integration_steps == 207


@pytest.mark.parametrize("failure", ["NONFINITE_STATE", "MUJOCO_WARNING_REQUIRES_ADJUDICATION"])
@pytest.mark.parametrize("location", ["initialization", "nominal"])
def test_unknown_numerical_failure_cannot_become_known_physical_rejection(
    failure: str, location: str
) -> None:
    """The accepted physical classifier keeps numerical faults unresolved."""
    result = _result(qualified=False)
    hard = Measurement(
        "HARD_REJECTION",
        False,
        failure,
        hard_failures=(failure,),
        completed_steps=7,
        expected_steps=2000,
        dt_s=0.002,
    )
    result.selected_parameters, result.selected_screening = None, None
    if location == "initialization":
        result.initialization, result.initial_screening = hard, None
    else:
        result.initial_screening = PairedMeasurement(
            result.original_parameters,
            hard,
            classification="HARD_REJECTION",
            reason="PHYSICAL_HARD_REJECTION",
            complete=True,
            known_completed_steps=7,
        )
    with pytest.raises(RepairInputError):
        _codec.serialize_run(result)


def test_zero_warning_count_and_accepted_control_roundoff_remain_admitted() -> None:
    """Diagnostics use the accepted tolerances and do not invent warning failures."""
    document, members = _document()
    metrics = document["confirmation"]["nominal"]["metrics"]
    metrics["warnings"] = [{"index": 0, "number": 0, "lastinfo": 0}]
    metrics["max_reference_speed_m_s"] = math.nextafter(0.12, 1.0)
    metrics["max_command_norm_N"] = math.nextafter(2.0, 3.0)
    admitted = _codec.admit_record(document, members)
    assert admitted.qualified


def test_replay_finite_negative_cannot_disclaim_complete_qualified_pair() -> None:
    """A saved replay must report the outcome of its own complete physical pair."""
    seed = _result()
    result = ReplayResult(
        "NOT_QUALIFIED",
        10,
        "REPLAY_NOT_QUALIFIED",
        seed.request,
        seed.selected_parameters,
        seed.confirmation,
        False,
        seed.runtime,
        ExecutionCosts(0, 6200, 6200, 0, 0.1, seed.request.limits, None),
    )
    with pytest.raises(RepairInputError):
        _codec.serialize_replay(result, "a" * 64)


def _failed_initialization(*, nonfinite: bool = True) -> EngineResult:
    seed = _result(qualified=False)
    arrays = list(seed.reset.arrays)
    name, shape, values = arrays[0]
    arrays[0] = (name, shape, (math.nan if nonfinite else 0.0, *values[1:]))
    payload = b"".join(
        struct.pack("<" + "d" * len(values), *values) for _, _, values in sorted(arrays)
    )
    snapshot = replace(
        seed.reset,
        arrays=tuple(arrays),
        state_sha256=hashlib.sha256(payload).hexdigest(),
        settling_failures=("NONFINITE_STATE",),
    )
    initialization = Measurement(
        "INCOMPLETE",
        False,
        "INITIAL_STATE_NOT_ADMISSIBLE",
        metrics={
            "settling_failures": ["NONFINITE_STATE"],
            "physical_state_sha256": snapshot.state_sha256,
        },
        hard_failures=("NONFINITE_STATE",),
        completed_steps=200,
        expected_steps=200,
        dt_s=0.002,
    )
    return replace(
        seed,
        status="INCOMPLETE",
        exit_code=20,
        reason=initialization.reason,
        selected_parameters=None,
        reset=snapshot,
        initialization=initialization,
        initial_screening=None,
        selected_screening=None,
        screening_measurements=[],
        costs=ExecutionCosts(0, 200, 200, 0, 0.125, seed.request.limits, 0.75),
    )


@pytest.mark.parametrize("nonfinite", [False, True])
def test_failed_initialization_retains_only_usable_reset_without_mutating_result(nonfinite):
    """A nonfinite settling failure remains incomplete; valid finite reset evidence survives."""
    result = _failed_initialization(nonfinite=nonfinite)
    original_reset, original_initialization = result.reset, result.initialization
    original_costs = replace(result.costs)
    raw, members = _codec.serialize_run(result)
    document = _codec.decode_document(raw)
    admitted = _codec.admit_record(document, members)
    assert (admitted.status, admitted.exit_code, admitted.reason) == (
        "INCOMPLETE",
        20,
        "INITIAL_STATE_NOT_ADMISSIBLE",
    )
    assert not admitted.qualified
    assert admitted.selected_parameters is None
    assert admitted.initialization == original_initialization
    assert admitted.costs == original_costs
    expected = {"nominal.xml", "finer.xml"} | (set() if nonfinite else {"reset.json"})
    assert set(members) == set(document["artifacts"]) == expected
    assert admitted.reset == (None if nonfinite else original_reset)
    assert result.reset is original_reset
    assert result.initialization is original_initialization
    assert result.costs == original_costs
    assert result.selected_parameters is None


@pytest.mark.parametrize(
    "fault", ["selected", "complete", "missing_failure", "missing_reset_failure"]
)
def test_nonfinite_reset_omission_requires_failed_unselected_initialization(fault):
    """Other nonfinite resets remain invalid instead of being silently dropped."""
    result = _failed_initialization()
    if fault == "selected":
        result.selected_parameters = result.original_parameters
    elif fault == "complete":
        result.initialization = replace(result.initialization, classification="COMPLETE")
    elif fault == "missing_failure":
        result.initialization = replace(result.initialization, hard_failures=())
    else:
        result.reset = replace(result.reset, settling_failures=())
    with pytest.raises(RepairInputError):
        _codec.serialize_run(result)


def test_unrelated_invalid_reset_is_not_hidden_by_failed_initialization():
    """A finite reset with a bad identity still fails strict admission."""
    result = _failed_initialization(nonfinite=False)
    result.reset = replace(result.reset, state_sha256="0" * 64)
    with pytest.raises(RepairInputError, match="reset layout, finiteness, or identity"):
        _codec.serialize_run(result)


def test_success_cannot_discard_nonfinite_reset_to_claim_qualification():
    """The incomplete-outcome exception never admits a contradictory success."""
    result = _result()
    result.reset = _failed_initialization().reset
    with pytest.raises(RepairInputError):
        _codec.serialize_run(result)
