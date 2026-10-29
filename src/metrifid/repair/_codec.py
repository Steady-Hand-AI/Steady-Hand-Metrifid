"""Version-one decimal JSON records for retained repair and fresh replay.

Records establish bounded internal consistency, not authorship or new physical
qualification. Reading requires neither NumPy nor MuJoCo. JSON identities use
sorted keys, compact separators, ASCII escapes and one trailing newline.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, NoReturn, cast

from .._json_admission import JsonAdmissionLimits, enforce_json_structure
from ..version import __version__
from . import _engine, _profile
from ._physical_types import Classification, Measurement, Snapshot, snapshot_well_formed
from ._request import (
    RepairInputError,
    RepairLimits,
    RepairRequest,
    _decimal_token,
    _integer_token,
    _nonfinite_token,
    _request_from_document,
    _unique_object,
    validate_selected_parameters,
)
from ._types import EngineResult, ExecutionCosts, PairedMeasurement, ReplayResult

RUN_SCHEMA = "metrifid.repair_result"
REPLAY_SCHEMA = "metrifid.repair_replay"
MAX_RESULT_BYTES = 1_048_576
MEMBER_LIMITS = {
    "nominal.xml": 65_536,
    "finer.xml": 65_536,
    "reset.json": 65_536,
    "policy.json": 16_384,
}
JSON_LIMITS = JsonAdmissionLimits(MAX_RESULT_BYTES, 16, 16_000, 65_536)
METHOD = {
    "name": "coordinate_poll",
    "settings": {
        "initial_step": 0.01,
        "minimum_step": 0.00125,
        "actual_improvement_tolerance": 1e-10,
        "max_sweeps": 100,
        "initial_preferred_sign": 1,
        "stagnant_step_multiplier": 0.5,
    },
}
LIMITATIONS = [
    "Simulation only.",
    "Only planar_push_timed15_v1 on Linux x86_64, CPython 3.13, MuJoCo 3.16.0 and NumPy 2.5.4.",
    "No global-optimum, robustness, authorship or cross-platform bitwise replay guarantee.",
]
_COMMON = {
    "schema",
    "version",
    "operation",
    "tool",
    "request",
    "status",
    "exit_code",
    "reason",
    "qualified",
    "diagnostic",
    "selected_parameters",
    "selected_parameters_sha256",
    "costs",
    "runtime",
    "limitations",
    "unavailable_measurements",
}
_RUN_KEYS = _COMMON | {
    "profile",
    "timestep_identities",
    "confirmation_complete",
    "original_parameters",
    "original_parameters_sha256",
    "initialization",
    "initial_screening",
    "selected_screening",
    "confirmation",
    "method",
    "artifacts",
}
_REPLAY_KEYS = _COMMON | {"source_repair_sha256", "measurement"}
_MEASUREMENT_KEYS = {
    "classification",
    "qualified",
    "reason",
    "hold_maxima",
    "metrics",
    "hard_failures",
    "completed_steps",
    "expected_steps",
    "dt_s",
    "hold_samples",
    "expected_hold_samples",
}
_PAIR_KEYS = {
    "parameters_normalized",
    "nominal",
    "finer",
    "objective",
    "classification",
    "reason",
    "complete",
    "qualified",
    "known_completed_steps",
    "unknown_step_calls",
}
_COST_KEYS = {
    "attempts",
    "integration_steps",
    "known_completed_steps",
    "unknown_step_calls",
    "cpu_seconds",
    "limits",
    "search_cpu_fraction",
}
_RUNTIME_KEYS = {"implementation", "machine", "mujoco", "numpy", "python", "system"}
_METRIC_KEYS = {
    "dt_s",
    "episode_time_s",
    "physics_steps",
    "requested_physics_steps",
    "hold_samples",
    "expected_hold_samples",
    "settled_state_sha256",
    "endpoint_errors",
    "schedule",
    "warnings",
    "hold_max_position_error_m",
    "hold_max_yaw_error_rad",
    "hold_max_linear_speed_m_s",
    "hold_max_angular_speed_rad_s",
    "maximum_penetration_m",
    "max_command_norm_N",
    "max_finger_speed_m_s",
    "max_block_speed_m_s",
    "maximum_lift_m",
    "maximum_tilt_rad",
    "max_reference_speed_m_s",
    "physical_state_sha256",
    "settling_failures",
}
_HOLD_METRICS = (
    "hold_max_position_error_m",
    "hold_max_yaw_error_rad",
    "hold_max_linear_speed_m_s",
    "hold_max_angular_speed_rad_s",
)
_SCHEDULE_KEYS = {
    "boundaries_s",
    "decoded_boundaries_s",
    "motion_deadlines_s",
    "scheduled_starts_s",
    "stationary_tail_start_s",
    "stroke_windows_s",
    "timing_parameters_normalized",
}
_RUN_EXITS = {
    "REPAIRED": 0,
    "ALREADY_QUALIFIED": 0,
    "NO_QUALIFIED_POLICY_FOUND": 10,
    "INCOMPLETE": 20,
    "REFUSED": 64,
    "ERROR": 70,
}
_REPLAY_EXITS = {"QUALIFIED": 0, "NOT_QUALIFIED": 10, "INCOMPLETE": 20, "REFUSED": 64, "ERROR": 70}


def _refuse(message: str) -> NoReturn:
    raise RepairInputError(message)


def _object(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise RepairInputError(f"{name} has missing or unknown fields")
    return cast(dict[str, Any], value)


def _text(value: object, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value):
        raise RepairInputError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise RepairInputError(f"{name} must be a nonnegative integer")
    return value


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise RepairInputError(f"{name} must be a finite number")
    try:
        number = float(cast("int | float", value))
    except (ValueError, OverflowError) as exc:
        raise RepairInputError(f"{name} is outside binary64") from exc
    if not math.isfinite(number):
        raise RepairInputError(f"{name} must be finite")
    return number


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise RepairInputError(f"{name} must be boolean")
    return value


def _digest(value: object, name: str) -> str:
    text = _text(value, name)
    if re.fullmatch("[0-9a-f]{64}", text) is None:
        raise RepairInputError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise RepairInputError(f"{name} must be a string array")
    return tuple(_text(item, name) for item in value)


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encode_json(value: object) -> bytes:
    """Encode the single deterministic repair-domain JSON identity representation."""
    try:
        return (
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
            )
            + "\n"
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise RepairInputError(f"Repair data cannot be encoded as finite JSON: {exc}") from exc


def decode_document(raw: bytes, *, max_bytes: int = MAX_RESULT_BYTES) -> dict[str, Any]:
    """Admit bounded UTF-8 decimal JSON, refusing duplicates, BOM and nonfinite tokens."""
    if not isinstance(raw, bytes) or len(raw) > max_bytes:
        raise RepairInputError("Repair document exceeds its byte bound")
    try:
        text = raw.decode("utf-8", errors="strict")
        if text.startswith("\ufeff"):
            raise RepairInputError("A UTF-8 byte-order mark is not supported")
        value: object = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_int=_integer_token,
            parse_float=_decimal_token,
            parse_constant=_nonfinite_token,
        )
        enforce_json_structure(value, JSON_LIMITS)
    except RepairInputError:
        raise
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise RepairInputError(f"Invalid repair JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RepairInputError("Repair document must be an object")
    return cast(dict[str, Any], value)


def _dump(document: dict[str, Any], max_bytes: int = MAX_RESULT_BYTES) -> bytes:
    raw = encode_json(document)
    decode_document(raw, max_bytes=max_bytes)
    return raw


def _same(value: object, expected: object, name: str) -> None:
    if encode_json(value) != encode_json(expected):
        raise RepairInputError(f"{name} differs from the supported resolved value")


def _measurement_value(value: object, path: str, unavailable: list[str]) -> object:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        unavailable.append(path)
        return None
    if isinstance(value, dict):
        return {
            key: _measurement_value(item, f"{path}.{key}", unavailable)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _measurement_value(item, f"{path}[{index}]", unavailable)
            for index, item in enumerate(value)
        ]
    return value


def _measured_document(
    value: Measurement | PairedMeasurement | None, path: str, unavailable: list[str]
) -> object:
    return _measurement_value(asdict(value) if value is not None else None, path, unavailable)


def _null_paths(value: object, path: str) -> list[str]:
    if value is None:
        return [path]
    if isinstance(value, dict):
        return [
            entry for key, child in value.items() for entry in _null_paths(child, f"{path}.{key}")
        ]
    if isinstance(value, list):
        return [
            entry
            for index, child in enumerate(value)
            for entry in _null_paths(child, f"{path}[{index}]")
        ]
    return []


def _request(value: object) -> RepairRequest | None:
    if value is None:
        return None
    request = _request_from_document(value)
    document = cast(dict[str, Any], value)
    _object(
        document["limits"],
        {"max_attempts", "max_integration_steps", "max_cpu_seconds"},
        "resolved limits",
    )
    return request


def _vector(value: object, identity: object, name: str) -> tuple[float, ...] | None:
    if value is None:
        if identity is not None:
            _refuse(f"{name} identity is present without a vector")
        return None
    if not isinstance(value, list):
        _refuse(f"{name} must be an array")
    vector = validate_selected_parameters(value)
    if _digest(identity, name + " identity") != _hash(encode_json(value)):
        _refuse(f"{name} identity does not match its encoding")
    return vector


def _nullable_values(value: object, length: int, name: str) -> list[float | None]:
    if not isinstance(value, list) or len(value) != length:
        raise RepairInputError(f"{name} must have {length} elements")
    return [None if item is None else _number(item, name) for item in value]


def _schedule(value: object) -> None:
    row = _object(value, _SCHEDULE_KEYS, "schedule")
    for key, length in (
        ("boundaries_s", 4),
        ("decoded_boundaries_s", 4),
        ("motion_deadlines_s", 3),
        ("scheduled_starts_s", 3),
        ("timing_parameters_normalized", 3),
    ):
        _nullable_values(row[key], length, key)
    if row["stationary_tail_start_s"] is not None:
        _number(row["stationary_tail_start_s"], "stationary tail")
    windows = row["stroke_windows_s"]
    if not isinstance(windows, list) or len(windows) != 3:
        _refuse("Stroke windows must contain three pairs")
    for window in windows:
        _nullable_values(window, 2, "stroke window")


def _metric_value(key: str, value: object) -> None:
    if value is None:
        return
    if key == "schedule":
        _schedule(value)
    elif key == "endpoint_errors":
        _nullable_values(value, 4, key)
    elif key == "warnings":
        if not isinstance(value, list):
            _refuse("warnings must be an array")
        for warning in cast(list[object], value):
            row = _object(warning, {"index", "number", "lastinfo"}, "warning")
            for name, count in row.items():
                _integer(count, name)
    elif key == "settling_failures":
        _strings(value, key)
    elif key.endswith("sha256"):
        _digest(value, key)
    elif key in {
        "physics_steps",
        "requested_physics_steps",
        "hold_samples",
        "expected_hold_samples",
    }:
        _integer(value, key)
    else:
        _number(value, key)


def _measurement(value: object, name: str, profile: dict[str, Any] | None) -> Measurement | None:
    if value is None:
        return None
    row = _object(value, _MEASUREMENT_KEYS, name)
    classification = row["classification"]
    if classification not in ("COMPLETE", "HARD_REJECTION", "INCOMPLETE"):
        _refuse("Unknown measurement classification")
    metrics = row["metrics"]
    if not isinstance(metrics, dict) or not set(metrics) <= _METRIC_KEYS:
        _refuse("Measurement metrics contain unknown fields")
    for key, metric in metrics.items():
        _metric_value(key, metric)
    maxima = row["hold_maxima"]
    hold = None
    if maxima is not None:
        values = _nullable_values(maxima, 4, name + ".hold_maxima")
        hold = tuple(math.nan if item is None else item for item in values)
    dt = math.nan if row["dt_s"] is None else _number(row["dt_s"], name + ".dt_s")
    measured = Measurement(
        cast(Classification, classification),
        _boolean(row["qualified"], "qualified"),
        _text(row["reason"], "measurement reason"),
        cast("tuple[float, float, float, float] | None", hold),
        metrics,
        _strings(row["hard_failures"], "hard failures"),
        _integer(row["completed_steps"], "completed steps"),
        _integer(row["expected_steps"], "expected steps"),
        dt,
        _integer(row["hold_samples"], "hold samples"),
        _integer(row["expected_hold_samples"], "expected hold samples"),
    )
    _measurement_consistency(measured, name)
    _known_rejection(measured)
    _complete_case(measured, name, profile)
    return measured


def _measurement_consistency(measured: Measurement, name: str) -> None:
    if (
        measured.completed_steps > measured.expected_steps
        or measured.hold_samples > measured.expected_hold_samples
    ):
        _refuse(f"{name} exceeds its declared measurement counts")
    if measured.classification != "COMPLETE" and measured.qualified:
        _refuse(f"{name} claims qualification without complete evidence")
    if measured.classification == "COMPLETE":
        missing_hold = name != "initialization" and (
            measured.hold_maxima is None
            or any(not math.isfinite(value) or value < 0 for value in measured.hold_maxima)
        )
        if (
            measured.hard_failures
            or not math.isfinite(measured.dt_s)
            or missing_hold
            or _null_paths(measured.metrics, name + ".metrics")
        ):
            _refuse(f"{name} complete flag contradicts its physical evidence")
    for key, expected in (
        ("physics_steps", measured.completed_steps),
        ("requested_physics_steps", measured.expected_steps),
        ("hold_samples", measured.hold_samples),
        ("expected_hold_samples", measured.expected_hold_samples),
        ("dt_s", measured.dt_s),
    ):
        if (
            key in measured.metrics
            and measured.metrics[key] is not None
            and measured.metrics[key] != expected
        ):
            _refuse(f"{name}.{key} contradicts its explicit field")
    if measured.hold_maxima is not None:
        for key, maximum in zip(_HOLD_METRICS, measured.hold_maxima, strict=True):
            if (
                key in measured.metrics
                and measured.metrics[key] is not None
                and measured.metrics[key] != maximum
            ):
                _refuse(f"{name}.{key} contradicts hold_maxima")


def _known_rejection(measured: Measurement) -> None:
    if measured.classification != "HARD_REJECTION":
        return
    unknown = any(
        reason.startswith("NONFINITE_") or reason == "MUJOCO_WARNING_REQUIRES_ADJUDICATION"
        for reason in measured.hard_failures
    )
    warnings = cast(list[dict[str, int]], measured.metrics.get("warnings", []))
    if unknown or any(warning["number"] > 0 for warning in warnings):
        _refuse("Unknown numerical evidence cannot substantiate a known physical rejection")


def _complete_case(measured: Measurement, name: str, profile: dict[str, Any] | None) -> None:
    if profile is None:
        _refuse("Measurements require a resolved supported task")
    if measured.classification != "COMPLETE":
        return
    if name == "initialization":
        if measured.metrics.get("settling_failures"):
            _refuse("Complete initialization contradicts its settling failures")
        return
    dt = _profile.NOMINAL_DT_S if name.endswith(".nominal") else _profile.FINER_DT_S
    if not _engine._complete_measurement(measured, dt) or measured.hold_maxima is None:
        _refuse("Complete case requires explicit complete timestep and hold counters")
    qualified = all(
        value <= limit
        for value, limit in zip(measured.hold_maxima, _profile.HOLD_LIMITS, strict=True)
    )
    if measured.qualified != qualified:
        _refuse("Complete case qualification contradicts its hold maxima")
    _supplied_gate_metrics(measured.metrics, profile)


def _supplied_gate_metrics(metrics: dict[str, object], profile: dict[str, Any]) -> None:
    warnings = metrics.get("warnings", [])
    if any(warning["number"] > 0 for warning in cast(list[dict[str, int]], warnings)):
        _refuse("Complete case contains a simulator warning requiring adjudication")
    scene, controller = profile["scene"], profile["controller"]
    gates = scene["whole_episode_gates"]
    limits = {
        "maximum_penetration_m": scene["contact"]["penetration_gate_m"],
        "maximum_lift_m": gates["lift_above_settled_com_m"],
        "maximum_tilt_rad": gates["tilt_max_rad"],
        "max_block_speed_m_s": gates["block_speed_max_m_s"],
        "max_finger_speed_m_s": gates["finger_speed_max_m_s"],
        "max_command_norm_N": controller["force_norm_bound_N"] + 1e-9,
        "max_reference_speed_m_s": controller["reference_speed_bound_m_s"] + 1e-9,
    }
    for name, limit in limits.items():
        if name in metrics:
            measured = _number(metrics[name], name)
            if measured < 0 or measured > limit:
                _refuse(f"Complete case contradicts the supplied physical gate metric {name}")


def _pair(value: object, name: str, profile: dict[str, Any] | None) -> PairedMeasurement | None:
    if value is None:
        return None
    row = _object(value, _PAIR_KEYS, name)
    parameters = validate_selected_parameters(row["parameters_normalized"])
    pair = PairedMeasurement(
        parameters,
        _measurement(row["nominal"], name + ".nominal", profile),
        _measurement(row["finer"], name + ".finer", profile),
        None if row["objective"] is None else _number(row["objective"], "objective"),
        _text(row["classification"], "pair classification"),
        _text(row["reason"], "pair reason"),
        _boolean(row["complete"], "complete"),
        _boolean(row["qualified"], "qualified"),
        _integer(row["known_completed_steps"], "pair completed steps"),
        _integer(row["unknown_step_calls"], "pair unknown calls"),
    )
    actual = PairedMeasurement(parameters, pair.nominal, pair.finer)
    _engine._classify_pair(actual)
    pending = (
        pair.classification == "INCOMPLETE"
        and pair.objective is None
        and not pair.complete
        and not pair.qualified
    )
    if not pending and (pair.classification, pair.objective, pair.complete, pair.qualified) != (
        actual.classification,
        actual.objective,
        actual.complete,
        actual.qualified,
    ):
        _refuse(f"{name} contradicts its paired physical measurements")
    observed = sum(item.completed_steps for item in (pair.nominal, pair.finer) if item is not None)
    if observed > pair.known_completed_steps or (
        pair.complete and observed != pair.known_completed_steps
    ):
        _refuse(f"{name} contradicts its explicit step receipts")
    if pair.complete and pair.unknown_step_calls:
        _refuse(f"{name} cannot be complete with unknown physical work")
    return pair


def _limits(value: object) -> RepairLimits | None:
    if value is None:
        return None
    row = _object(
        value, {"max_attempts", "max_integration_steps", "max_cpu_seconds"}, "cost limits"
    )
    return RepairLimits(
        _integer(row["max_attempts"], "attempt limit"),
        _integer(row["max_integration_steps"], "integration limit"),
        _number(row["max_cpu_seconds"], "CPU limit"),
    )


def _costs(value: object, request: RepairRequest | None, *, replay: bool) -> ExecutionCosts:
    row = _object(value, _COST_KEYS, "costs")
    costs = ExecutionCosts(
        _integer(row["attempts"], "attempts"),
        None
        if row["integration_steps"] is None
        else _integer(row["integration_steps"], "integration steps"),
        _integer(row["known_completed_steps"], "known completed steps"),
        _integer(row["unknown_step_calls"], "unknown step calls"),
        _number(row["cpu_seconds"], "CPU seconds"),
        _limits(row["limits"]),
        None
        if row["search_cpu_fraction"] is None
        else _number(row["search_cpu_fraction"], "CPU fraction"),
    )
    if costs.cpu_seconds < 0 or costs.limits != (request.limits if request is not None else None):
        _refuse("Cost limits or CPU duration contradict the resolved request")
    expected_steps = None if costs.unknown_step_calls else costs.known_completed_steps
    if costs.integration_steps != expected_steps:
        _refuse("Unknown integration work or completed counters are inconsistent")
    expected_fraction = None if replay or costs.limits is None else 0.75
    if costs.search_cpu_fraction != expected_fraction or (replay and costs.attempts):
        _refuse("Cost record contradicts the operation's accounting policy")
    if costs.limits is not None and (
        costs.attempts > costs.limits.max_attempts
        or costs.known_completed_steps > costs.limits.max_integration_steps
    ):
        _refuse("Actual work exceeds declared finite bounds")
    return costs


def _runtime(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not set(value) <= _RUNTIME_KEYS:
        raise RepairInputError("Runtime contains unknown fields")
    return {key: _text(item, "runtime value") for key, item in value.items()}


def _supported_runtime(runtime: dict[str, str]) -> bool:
    expected = {
        "implementation": "CPython",
        "system": "Linux",
        "machine": "x86_64",
        "mujoco": "3.16.0",
        "numpy": "2.5.4",
    }
    return (
        all(runtime.get(key) == value for key, value in expected.items())
        and re.fullmatch(r"3\.13\.\d+", runtime.get("python", "")) is not None
    )


def _outcome(document: dict[str, Any], *, replay: bool) -> None:
    exits = _REPLAY_EXITS if replay else _RUN_EXITS
    status = _text(document["status"], "status")
    code = _integer(document["exit_code"], "exit code")
    if status not in exits or exits[status] != code:
        _refuse("Status and exit code do not describe a supported outcome")
    qualified = _boolean(document["qualified"], "qualification")
    if qualified != (code == 0):
        _refuse("Recorded qualification contradicts the status")
    _text(document["reason"], "reason")
    if document["diagnostic"] is not None:
        _text(document["diagnostic"], "diagnostic", empty=True)


def _common(document: dict[str, Any], *, replay: bool) -> RepairRequest | None:
    _object(document, _REPLAY_KEYS if replay else _RUN_KEYS, "repair record")
    expected_schema, operation = (REPLAY_SCHEMA, "replay") if replay else (RUN_SCHEMA, "run")
    if (
        document["schema"] != expected_schema
        or type(document["version"]) is not int
        or document["version"] != 1
        or document["operation"] != operation
    ):
        _refuse("Unsupported repair record schema/version/operation")
    tool = _object(document["tool"], {"name", "version"}, "tool")
    if tool["name"] != "metrifid":
        _refuse("Unsupported record tool")
    _text(tool["version"], "tool version")
    _same(document["limitations"], LIMITATIONS, "limitations")
    _outcome(document, replay=replay)
    paths = _strings(document["unavailable_measurements"], "unavailable measurements")
    names = (
        ("measurement",)
        if replay
        else ("initialization", "initial_screening", "selected_screening", "confirmation")
    )
    expected_paths = sorted(path for name in names for path in _null_paths(document[name], name))
    if list(paths) != expected_paths:
        _refuse("Unavailable measurement paths must enumerate exactly the null measurement fields")
    return _request(document["request"])


def artifact_records(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Admit fixed member names and bounded byte identities before file reads."""
    if document.get("schema") == REPLAY_SCHEMA:
        if "artifacts" in document:
            _refuse("Replay records do not declare bundle members")
        return {}
    if document.get("schema") != RUN_SCHEMA:
        _refuse("Unsupported repair result schema")
    records = document.get("artifacts")
    if not isinstance(records, dict) or not set(records) <= set(MEMBER_LIMITS):
        _refuse("Artifact names must be the fixed version-one filenames")
    admitted: dict[str, dict[str, Any]] = {}
    for name, value in cast(dict[str, Any], records).items():
        row = _object(value, {"filename", "size_bytes", "sha256"}, "artifact")
        size = _integer(row["size_bytes"], "artifact size")
        if row["filename"] != name or size > MEMBER_LIMITS[name]:
            _refuse("Artifact filename or size is outside the supported bundle")
        _digest(row["sha256"], "artifact digest")
        admitted[name] = row
    return admitted


def _members(document: dict[str, Any], members: Mapping[str, bytes]) -> None:
    records = artifact_records(document)
    if set(members) != set(records):
        _refuse("Declared and supplied artifact members differ")
    for name, row in records.items():
        raw = members[name]
        if (
            not isinstance(raw, bytes)
            or len(raw) != row["size_bytes"]
            or _hash(raw) != row["sha256"]
        ):
            _refuse(f"Artifact bytes do not match their identity: {name}")


def _snapshot_document(snapshot: Snapshot) -> dict[str, Any]:
    return {
        "schema": "metrifid.repair_reset",
        "version": 1,
        "arrays": [
            {"name": name, "shape": list(shape), "values": list(values)}
            for name, shape, values in snapshot.arrays
        ],
        "settled_block_com": list(snapshot.settled_block_com),
        "settled_block_yaw": snapshot.settled_block_yaw,
        "nominal_scene_sha256": snapshot.nominal_scene_sha256,
        "state_sha256": snapshot.state_sha256,
        "settling_failures": list(snapshot.settling_failures),
    }


def _snapshot(raw: bytes, scene_hash: str) -> Snapshot:
    row = _object(
        decode_document(raw, max_bytes=MEMBER_LIMITS["reset.json"]),
        {
            "schema",
            "version",
            "arrays",
            "settled_block_com",
            "settled_block_yaw",
            "nominal_scene_sha256",
            "state_sha256",
            "settling_failures",
        },
        "reset",
    )
    if (
        row["schema"] != "metrifid.repair_reset"
        or type(row["version"]) is not int
        or row["version"] != 1
    ):
        _refuse("Unsupported reset schema")
    if not isinstance(row["arrays"], list):
        _refuse("Physical arrays must be an array of named records")
    arrays = []
    for member in row["arrays"]:
        item = _object(member, {"name", "shape", "values"}, "physical array")
        if not isinstance(item["shape"], list) or not isinstance(item["values"], list):
            _refuse("Physical shape and values must be arrays")
        arrays.append(
            (
                _text(item["name"], "array name"),
                tuple(_integer(v, "array dimension") for v in item["shape"]),
                tuple(_number(v, "physical array value") for v in item["values"]),
            )
        )
    pose = _nullable_values(row["settled_block_com"], 3, "settled COM")
    if any(value is None for value in pose):
        _refuse("Settled COM must be finite")
    snapshot = Snapshot(
        tuple(arrays),
        cast("tuple[float, float, float]", tuple(pose)),
        _number(row["settled_block_yaw"], "settled yaw"),
        _digest(row["nominal_scene_sha256"], "nominal model digest"),
        _digest(row["state_sha256"], "state digest"),
        _strings(row["settling_failures"], "settling failures"),
    )
    if not snapshot_well_formed(snapshot, scene_hash):
        _refuse("Retained reset layout, finiteness, or identity is invalid")
    return snapshot


def _policy(
    raw: bytes, selected: tuple[float, ...] | None, identity: object, qualified: bool
) -> None:
    row = _object(
        decode_document(raw, max_bytes=MEMBER_LIMITS["policy.json"]),
        {"schema", "version", "parameters_normalized", "parameters_sha256", "qualified"},
        "policy",
    )
    if (
        row["schema"] != "metrifid.repair_policy"
        or type(row["version"]) is not int
        or row["version"] != 1
    ):
        _refuse("Unsupported policy schema")
    vector = _vector(row["parameters_normalized"], row["parameters_sha256"], "policy vector")
    if (
        vector is None
        or vector != selected
        or row["parameters_sha256"] != identity
        or _boolean(row["qualified"], "policy qualification") != qualified
    ):
        _refuse("Policy contradicts the selected vector binding or qualification")


def _scene_members(
    result: EngineResult, members: Mapping[str, bytes], nominal: str, finer: str
) -> None:
    for name, expected in (("nominal.xml", nominal), ("finer.xml", finer)):
        if name in members:
            if members[name] != expected.encode("utf-8"):
                _refuse(f"Generated scene differs from the admitted request: {name}")
            if name == "nominal.xml":
                result.nominal_xml = expected
            else:
                result.finer_xml = expected


def _run_members(
    result: EngineResult, document: dict[str, Any], members: Mapping[str, bytes]
) -> None:
    if result.request is None:
        if members or result.profile is not None or result.timestep_identities is not None:
            _refuse("Physical evidence cannot be bound without a resolved request")
        return
    profile, nominal, finer, identities = _profile.generate_expected(result.request)
    if result.profile is not None:
        _same(result.profile, profile, "profile")
    if result.timestep_identities is not None:
        _same(result.timestep_identities, identities, "timestep identities")
    if members and (result.profile is None or result.timestep_identities is None):
        _refuse("Artifacts require resolved profile and timestep identities")
    _scene_members(result, members, nominal, finer)
    if "reset.json" in members:
        result.reset = _snapshot(members["reset.json"], str(identities["nominal_scene_sha256"]))
    if "policy.json" in members:
        _policy(
            members["policy.json"],
            result.selected_parameters,
            document["selected_parameters_sha256"],
            result.qualified,
        )
    if (result.selected_parameters is not None) != ("policy.json" in members):
        _refuse("Selected policy availability contradicts its artifact record")


def _bindings(result: EngineResult) -> None:
    original = (
        result.request.controller.parameters_normalized if result.request is not None else None
    )
    if result.original_parameters != original:
        _refuse("Original vector differs from the resolved request")
    if (
        result.initial_screening is not None
        and result.initial_screening.parameters_normalized != original
    ):
        _refuse("Initial measurement differs from the supplied controller")
    for measured in (result.selected_screening, result.confirmation):
        if measured is not None and measured.parameters_normalized != result.selected_parameters:
            _refuse("Selected measurement/controller bindings disagree")
    if result.confirmation_complete != (
        result.confirmation is not None and result.confirmation.complete
    ):
        _refuse("Confirmation completeness contradicts the retained measurement")


def _qualified_pair(pair: PairedMeasurement | None) -> bool:
    return bool(
        pair is not None
        and pair.classification == "COMPLETE"
        and pair.complete
        and pair.objective == 0.0
        and pair.qualified
        and pair.unknown_step_calls == 0
    )


def _witness_costs(result: EngineResult) -> None:
    steps = result.initialization.completed_steps if result.initialization is not None else 0
    unknown = 0
    seen: set[tuple[float, ...]] = set()
    for pair in (result.initial_screening, result.selected_screening):
        if pair is not None and pair.parameters_normalized not in seen:
            seen.add(pair.parameters_normalized)
            steps += pair.known_completed_steps
            unknown += pair.unknown_step_calls
    if result.confirmation is not None:
        steps += result.confirmation.known_completed_steps
        unknown += result.confirmation.unknown_step_calls
    if (
        result.costs.known_completed_steps < steps
        or result.costs.unknown_step_calls < unknown
        or result.costs.attempts < len(seen)
    ):
        _refuse("Costs omit work demonstrated by retained measurements")
    _finite_negative(result)
    if (
        result.initial_screening is not None
        and result.selected_screening is not None
        and result.initial_screening.parameters_normalized
        == result.selected_screening.parameters_normalized
    ):
        _same(
            asdict(result.selected_screening),
            asdict(result.initial_screening),
            "unchanged selected screening",
        )


def _finite_negative(result: EngineResult) -> None:
    if result.status != "NO_QUALIFIED_POLICY_FOUND":
        return
    initial = result.initial_screening
    known_initial = (
        initial is not None
        and initial.complete
        and initial.classification in ("COMPLETE", "HARD_REJECTION")
    )
    rejected_setup = (
        result.initialization is not None
        and result.initialization.classification == "HARD_REJECTION"
    )
    if not known_initial and not rejected_setup:
        _refuse("Finite nonqualification requires a fully adjudicated initial observation")
    confirmation = result.confirmation
    if confirmation is not None and (not confirmation.complete or confirmation.qualified):
        _refuse("Finite nonqualification contradicts the confirmation outcome")
    fraction = 1.0 if confirmation is not None else 0.75
    _finite_cost(result.costs, fraction)


def _finite_cost(costs: ExecutionCosts, fraction: float) -> None:
    if (
        costs.limits is None
        or costs.unknown_step_calls
        or costs.cpu_seconds >= fraction * costs.limits.max_cpu_seconds
    ):
        _refuse("Finite outcome contradicts unknown work or its applicable CPU cutoff")


def _successful_run(result: EngineResult, members: Mapping[str, bytes]) -> None:
    if not result.qualified:
        return
    if (
        set(members) != set(MEMBER_LIMITS)
        or result.initialization is None
        or result.initialization.classification != "COMPLETE"
        or result.initialization.hard_failures
        or result.initialization.completed_steps != _profile.setup_steps()
        or result.initial_screening is None
        or result.initial_screening.classification != "COMPLETE"
        or not _qualified_pair(result.selected_screening)
        or not _qualified_pair(result.confirmation)
    ):
        _refuse("Recorded success lacks complete initialization/screening/confirmation evidence")
    unchanged = result.selected_parameters == result.original_parameters
    if unchanged != (result.status == "ALREADY_QUALIFIED"):
        _refuse("Success status contradicts the unchanged/changed controller relationship")
    if (result.initial_screening.objective == 0.0) != unchanged:
        _refuse("Success status contradicts the initial screening objective")
    if result.reset is None or result.reset.settling_failures:
        _refuse("Recorded success requires an admitted settled reset")
    if not _supported_runtime(result.runtime) or result.costs.unknown_step_calls:
        _refuse("Recorded success requires the supported runtime and known physical work")
    if (
        result.costs.limits is None
        or result.costs.cpu_seconds >= result.costs.limits.max_cpu_seconds
    ):
        _refuse("Recorded success exceeds its invocation CPU allowance")


def _run(document: dict[str, Any], members: Mapping[str, bytes]) -> EngineResult:
    request = _common(document, replay=False)
    expected_profile = _profile.resolve_profile(request) if request is not None else None
    _same(document["method"], METHOD, "method")
    _members(document, members)
    original = _vector(
        document["original_parameters"],
        document["original_parameters_sha256"],
        "original parameters",
    )
    selected = _vector(
        document["selected_parameters"],
        document["selected_parameters_sha256"],
        "selected parameters",
    )
    result = EngineResult(
        document["status"],
        document["exit_code"],
        document["reason"],
        request=request,
        original_parameters=original,
        selected_parameters=selected,
        profile=document["profile"],
        timestep_identities=document["timestep_identities"],
        runtime=_runtime(document["runtime"]),
        initialization=_measurement(document["initialization"], "initialization", expected_profile),
        initial_screening=_pair(
            document["initial_screening"], "initial_screening", expected_profile
        ),
        selected_screening=_pair(
            document["selected_screening"], "selected_screening", expected_profile
        ),
        confirmation=_pair(document["confirmation"], "confirmation", expected_profile),
        confirmation_complete=_boolean(
            document["confirmation_complete"], "confirmation completeness"
        ),
        qualified=document["qualified"],
        costs=_costs(document["costs"], request, replay=False),
        diagnostic=document["diagnostic"],
    )
    _run_members(result, document, members)
    _bindings(result)
    _witness_costs(result)
    _successful_run(result, members)
    return result


def _replay(document: dict[str, Any], members: Mapping[str, bytes]) -> ReplayResult:
    request = _common(document, replay=True)
    expected_profile = _profile.resolve_profile(request) if request is not None else None
    if members:
        _refuse("Replay is a standalone record, not another run bundle")
    _digest(document["source_repair_sha256"], "source repair digest")
    selected = _vector(
        document["selected_parameters"],
        document["selected_parameters_sha256"],
        "selected parameters",
    )
    result = ReplayResult(
        document["status"],
        document["exit_code"],
        document["reason"],
        request,
        selected,
        _pair(document["measurement"], "measurement", expected_profile),
        document["qualified"],
        _runtime(document["runtime"]),
        _costs(document["costs"], request, replay=True),
        document["diagnostic"],
    )
    if result.measurement is not None:
        if result.measurement.parameters_normalized != selected:
            _refuse("Replay measurement differs from selected controller")
        if (
            result.costs.known_completed_steps
            != _profile.setup_steps() + result.measurement.known_completed_steps
        ):
            _refuse("Replay costs must include its fresh setup and measured pair")
        if result.costs.unknown_step_calls != result.measurement.unknown_step_calls:
            _refuse("Replay unknown-work counters disagree")
    if result.status == "NOT_QUALIFIED" and (
        result.measurement is None
        or not result.measurement.complete
        or result.measurement.qualified
    ):
        _refuse("Replay nonqualification requires complete physical evidence")
    if result.status == "NOT_QUALIFIED":
        _finite_cost(result.costs, 1.0)
    if result.qualified and (
        request is None
        or not _qualified_pair(result.measurement)
        or not _supported_runtime(result.runtime)
        or result.costs.unknown_step_calls
        or result.costs.cpu_seconds >= request.limits.max_cpu_seconds
    ):
        _refuse("Replay qualification lacks complete supported-runtime evidence")
    return result


def admit_record(
    document: dict[str, Any], members: Mapping[str, bytes]
) -> EngineResult | ReplayResult:
    """Verify an explicit version-one record and exact supplied artifact bytes."""
    try:
        # Recheck JSON types/bounds for programmatic callers too.
        document = decode_document(encode_json(document))
        if document.get("schema") == RUN_SCHEMA:
            return _run(document, members)
        if document.get("schema") == REPLAY_SCHEMA:
            return _replay(document, members)
        raise RepairInputError("Unsupported repair result schema")
    except RepairInputError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise RepairInputError(f"Malformed repair record: {exc}") from exc


def _base(result: EngineResult | ReplayResult, *, replay: bool) -> dict[str, Any]:
    selected = list(result.selected_parameters) if result.selected_parameters is not None else None
    return {
        "schema": REPLAY_SCHEMA if replay else RUN_SCHEMA,
        "version": 1,
        "operation": "replay" if replay else "run",
        "tool": {"name": "metrifid", "version": __version__},
        "request": asdict(result.request) if result.request is not None else None,
        "status": result.status,
        "exit_code": result.exit_code,
        "reason": result.reason,
        "qualified": result.qualified,
        "diagnostic": result.diagnostic,
        "selected_parameters": selected,
        "selected_parameters_sha256": _hash(encode_json(selected))
        if selected is not None
        else None,
        "costs": asdict(result.costs),
        "runtime": result.runtime,
        "limitations": LIMITATIONS,
    }


def _nonfinite_initial_reset(result: EngineResult) -> bool:
    """Recognize unusable raw state retained by a failed settling observation."""
    snapshot, initialization = result.reset, result.initialization
    if (
        snapshot is None
        or result.status != "INCOMPLETE"
        or result.selected_parameters is not None
        or result.qualified
        or initialization is None
        or initialization.classification != "INCOMPLETE"
        or "NONFINITE_STATE" not in initialization.hard_failures
        or "NONFINITE_STATE" not in snapshot.settling_failures
    ):
        return False
    return (
        any(not math.isfinite(value) for _, _, values in snapshot.arrays for value in values)
        or any(not math.isfinite(value) for value in snapshot.settled_block_com)
        or not math.isfinite(snapshot.settled_block_yaw)
    )


def serialize_run(result: EngineResult) -> tuple[bytes, dict[str, bytes]]:
    """Serialize one consistent run, returning its marker and available members."""
    document = _base(result, replay=False)
    unavailable: list[str] = []
    original = list(result.original_parameters) if result.original_parameters is not None else None
    document.update(
        profile=result.profile,
        timestep_identities=result.timestep_identities,
        original_parameters=original,
        original_parameters_sha256=_hash(encode_json(original)) if original is not None else None,
        confirmation_complete=result.confirmation_complete,
        method=METHOD,
    )
    for name in ("initialization", "initial_screening", "selected_screening", "confirmation"):
        document[name] = _measured_document(getattr(result, name), name, unavailable)
    document["unavailable_measurements"] = sorted(unavailable)
    members: dict[str, bytes] = {}
    for name, xml in (("nominal.xml", result.nominal_xml), ("finer.xml", result.finer_xml)):
        if xml is not None:
            members[name] = xml.encode("utf-8", errors="strict")
    if result.reset is not None and not _nonfinite_initial_reset(result):
        members["reset.json"] = _dump(_snapshot_document(result.reset), MEMBER_LIMITS["reset.json"])
    if result.selected_parameters is not None:
        members["policy.json"] = _dump(
            {
                "schema": "metrifid.repair_policy",
                "version": 1,
                "parameters_normalized": document["selected_parameters"],
                "parameters_sha256": document["selected_parameters_sha256"],
                "qualified": result.qualified,
            },
            MEMBER_LIMITS["policy.json"],
        )
    document["artifacts"] = {
        name: {"filename": name, "size_bytes": len(raw), "sha256": _hash(raw)}
        for name, raw in members.items()
    }
    raw = _dump(document)
    admit_record(decode_document(raw), members)
    return raw, members


def serialize_replay(result: ReplayResult, source_digest: str) -> bytes:
    """Serialize a fresh replay outcome with a source-record digest linkage."""
    document = _base(result, replay=True)
    unavailable: list[str] = []
    document.update(
        source_repair_sha256=_digest(source_digest, "source repair digest"),
        measurement=_measured_document(result.measurement, "measurement", unavailable),
        unavailable_measurements=sorted(unavailable),
    )
    raw = _dump(document)
    admit_record(decode_document(raw), {})
    return raw
