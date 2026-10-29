"""Immutable requests and bounded, native-free admission for controller repair.

Request numbers may be decimal JSON values. Integer-valued schema and count
fields still require integer tokens. This decoder deliberately leaves the
package's separate float-free receipt format unchanged.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal, NoReturn, cast

from metrifid._json_admission import (
    JsonAdmissionError,
    JsonAdmissionLimits,
    enforce_json_structure,
    read_bounded_regular_file,
)

from ._profile import (
    BLOCK_CENTER_BOUNDS_M,
    BLOCK_SIZE_M,
    FAMILY,
    INITIAL_FINGER_Y_M,
    PARAMETER_DIMENSION,
    RAIL_Y_LOWER_MIN_M,
    SUPPORT_BOUNDS_M,
    minimum_step_budget,
)

_REQUEST_SCHEMA = "metrifid.repair_request"
_INITIAL_MARGIN = 1e-9
_REQUEST_JSON_LIMITS = JsonAdmissionLimits(
    max_bytes=65_536,
    max_depth=8,
    max_nodes=128,
    max_string_bytes=128,
)


class RepairInputError(ValueError):
    """A request could not be read or does not belong to the supported family.

    ``reason`` distinguishes an operating-system read failure from invalid
    content. Neither condition requires native simulation libraries.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: Literal["INVALID_REQUEST", "INPUT_READ_FAILED"] = "INVALID_REQUEST",
    ) -> None:
        """Retain the concise input diagnostic and its stable reason."""
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class RepairTask:
    """Admitted rail and block target for the supported fixed physical family."""

    rail_y_lower_m: float
    target_xy_m: tuple[float, float]
    target_yaw_rad: float

    def __post_init__(self) -> None:
        """Normalize numeric inputs and validate the actual rotated footprint."""
        rail = _number(self.rail_y_lower_m, "task.rail_y_lower_m")
        if not RAIL_Y_LOWER_MIN_M <= rail <= INITIAL_FINGER_Y_M:
            raise RepairInputError("task.rail_y_lower_m is outside the supported rail range")
        target = _vector(self.target_xy_m, 2, "task.target_xy_m")
        yaw = _number(self.target_yaw_rad, "task.target_yaw_rad")
        if not -math.pi <= yaw <= math.pi:
            raise RepairInputError("task.target_yaw_rad must be within [-pi, pi]")
        _check_target_footprint(target, yaw)
        object.__setattr__(self, "rail_y_lower_m", rail)
        object.__setattr__(self, "target_xy_m", (target[0], target[1]))
        object.__setattr__(self, "target_yaw_rad", yaw)


@dataclass(frozen=True, slots=True)
class RepairController:
    """Explicit starting vector, strictly inside the normalized local box."""

    parameters_normalized: tuple[float, ...]

    def __post_init__(self) -> None:
        """Copy and normalize the caller's starting vector into immutable storage."""
        parameters = _vector(
            self.parameters_normalized, PARAMETER_DIMENSION, "controller.parameters_normalized"
        )
        if any(not _INITIAL_MARGIN < value < 1 - _INITIAL_MARGIN for value in parameters):
            raise RepairInputError("Starting parameters must be strictly inside (1e-9, 1 - 1e-9)")
        object.__setattr__(self, "parameters_normalized", parameters)


@dataclass(frozen=True, slots=True)
class RepairLimits:
    """Invocation limits, including actual settling and confirmation work."""

    max_attempts: int = 128
    max_integration_steps: int = 96_000
    max_cpu_seconds: float = 120.0

    def __post_init__(self) -> None:
        """Admit counts and a finite positive process-CPU limit."""
        attempts = _integer(self.max_attempts, "limits.max_attempts")
        if not 2 <= attempts <= 1_000:
            raise RepairInputError("limits.max_attempts must be an integer from 2 through 1000")
        steps = _integer(self.max_integration_steps, "limits.max_integration_steps")
        minimum = minimum_step_budget()
        if steps < minimum:
            raise RepairInputError(f"limits.max_integration_steps must be at least {minimum}")
        cpu = _number(self.max_cpu_seconds, "limits.max_cpu_seconds")
        if cpu <= 0:
            raise RepairInputError("limits.max_cpu_seconds must be positive")
        object.__setattr__(self, "max_cpu_seconds", cpu)


@dataclass(frozen=True, slots=True)
class RepairRequest:
    """Validated, immutable task, starting controller and resource limits.

    Construction validates the supplied values. Execution independently calls
    :func:`validate_request` so altered or manually assembled Python objects
    cannot bypass the same admission rules.
    """

    task: RepairTask
    controller: RepairController
    limits: RepairLimits = field(default_factory=RepairLimits)
    schema: str = _REQUEST_SCHEMA
    version: int = 1
    family: str = FAMILY

    def __post_init__(self) -> None:
        """Validate metadata and make fresh immutable copies of nested values."""
        if type(self.schema) is not str or self.schema != _REQUEST_SCHEMA:
            raise RepairInputError("schema must be metrifid.repair_request")
        if type(self.version) is not int or self.version != 1:
            raise RepairInputError("version must be the integer 1")
        if type(self.family) is not str or self.family != FAMILY:
            raise RepairInputError(f"family must be {FAMILY}")
        if type(self.task) is not RepairTask:
            raise RepairInputError("task must be a RepairTask")
        if type(self.controller) is not RepairController:
            raise RepairInputError("controller must be a RepairController")
        if type(self.limits) is not RepairLimits:
            raise RepairInputError("limits must be RepairLimits")
        object.__setattr__(
            self,
            "task",
            RepairTask(self.task.rail_y_lower_m, self.task.target_xy_m, self.task.target_yaw_rad),
        )
        object.__setattr__(
            self, "controller", RepairController(self.controller.parameters_normalized)
        )
        object.__setattr__(
            self,
            "limits",
            RepairLimits(
                self.limits.max_attempts,
                self.limits.max_integration_steps,
                self.limits.max_cpu_seconds,
            ),
        )


def validate_request(request: object) -> RepairRequest:
    """Re-admit a Python request at an execution or replay boundary.

    Returns a fresh immutable request, including revalidated nested values.
    This does not import NumPy, MuJoCo or an execution backend.
    """
    if type(request) is not RepairRequest:
        raise RepairInputError("Expected a RepairRequest produced by supported request admission")
    try:
        return RepairRequest(
            task=request.task,
            controller=request.controller,
            limits=request.limits,
            schema=request.schema,
            version=request.version,
            family=request.family,
        )
    except RepairInputError:
        raise
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise RepairInputError("RepairRequest has missing or invalid fields") from exc


def validate_selected_parameters(parameters: object) -> tuple[float, ...]:
    """Admit a generated or retained controller on the closed normalized box.

    Starting requests have stricter interior bounds. A selected controller on
    an actual box boundary remains eligible for independent replay.
    """
    values = _vector(parameters, PARAMETER_DIMENSION, "selected parameters")
    if any(value < 0 or value > 1 for value in values):
        raise RepairInputError("Selected parameters must be within [0, 1]")
    return values


def load_request(path: str | os.PathLike[str]) -> RepairRequest:
    """Read one bounded decimal-valued UTF-8 request from a regular file.

    Duplicate members, unknown fields, unsupported values, nonfinite numbers,
    a byte-order mark and all declared size-bound violations are refused.

    Args:
        path: Request file, opened once without following a final symbolic link.

    Returns:
        An immutable request containing normalized binary64 task/controller values.

    Raises:
        RepairInputError: A read failed or the request was invalid. Its ``reason``
            is ``INPUT_READ_FAILED`` for an operating-system reader failure and
            ``INVALID_REQUEST`` for structural or semantic invalidity.
    """
    try:
        payload = read_bounded_regular_file(path, _REQUEST_JSON_LIMITS.max_bytes)
    except (OSError, JsonAdmissionError) as exc:
        reason: Literal["INVALID_REQUEST", "INPUT_READ_FAILED"] = (
            "INPUT_READ_FAILED" if _reader_os_failure(exc) else "INVALID_REQUEST"
        )
        raise RepairInputError(str(exc), reason=reason) from exc
    except (TypeError, ValueError) as exc:
        raise RepairInputError("Request path is invalid") from exc
    return _request_from_document(_decode_request_document(payload))


def _number(value: object, name: str) -> float:
    """Normalize only finite built-in integer or decimal values to binary64."""
    if type(value) not in (int, float):
        raise RepairInputError(f"{name} must be a finite number, not a boolean or numeric string")
    try:
        number = float(cast(int | float, value))
    except (ValueError, OverflowError) as exc:
        raise RepairInputError(f"{name} does not fit finite binary64") from exc
    if not math.isfinite(number):
        raise RepairInputError(f"{name} must be finite")
    return number


def _integer(value: object, name: str) -> int:
    """Require an actual integer, excluding booleans and decimal tokens."""
    if type(value) is not int:
        raise RepairInputError(f"{name} must be an integer")
    return value


def _vector(value: object, length: int, name: str) -> tuple[float, ...]:
    """Admit a finite list or tuple without executing an arbitrary iterator."""
    if type(value) not in (list, tuple):
        raise RepairInputError(f"{name} must be an array of {length} numbers")
    sequence = cast(list[object] | tuple[object, ...], value)
    if len(sequence) != length:
        raise RepairInputError(f"{name} must contain exactly {length} numbers")
    return tuple(_number(member, name) for member in sequence)


def _check_target_footprint(target: tuple[float, ...], yaw: float) -> None:
    """Require the center and each rotated corner to fit the shipped geometry."""
    for coordinate, bounds in zip(target, BLOCK_CENTER_BOUNDS_M, strict=True):
        if not bounds[0] <= coordinate <= bounds[1]:
            raise RepairInputError("Target center is outside the supported block-center bounds")
    cosine, sine = math.cos(yaw), math.sin(yaw)
    half_x, half_y = BLOCK_SIZE_M[0] / 2, BLOCK_SIZE_M[1] / 2
    for local_x in (-half_x, half_x):
        for local_y in (-half_y, half_y):
            corner = (
                target[0] + local_x * cosine - local_y * sine,
                target[1] + local_x * sine + local_y * cosine,
            )
            for coordinate, bounds in zip(corner, SUPPORT_BOUNDS_M, strict=True):
                if not bounds[0] <= coordinate <= bounds[1]:
                    raise RepairInputError("The yaw-rotated target footprint exceeds the support")


def _object(
    value: object, *, required: set[str], optional: set[str], name: str
) -> dict[str, object]:
    """Admit only the named JSON object members at one schema level."""
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise RepairInputError(f"{name} must be an object")
    result = cast(dict[str, object], value)
    names = set(result)
    if not required <= names:
        raise RepairInputError(f"{name} is missing required members")
    if names - required - optional:
        raise RepairInputError(f"{name} contains unsupported members")
    return result


def _request_from_document(value: object) -> RepairRequest:
    """Resolve the exact request schema and independent optional defaults."""
    root = _object(
        value,
        required={"schema", "version", "family", "task", "controller"},
        optional={"limits"},
        name="request",
    )
    task = _object(
        root["task"],
        required={"rail_y_lower_m", "target_xy_m", "target_yaw_rad"},
        optional=set(),
        name="task",
    )
    controller = _object(
        root["controller"], required={"parameters_normalized"}, optional=set(), name="controller"
    )
    limits = _object(
        root.get("limits", {}),
        required=set(),
        optional={"max_attempts", "max_integration_steps", "max_cpu_seconds"},
        name="limits",
    )
    defaults = RepairLimits()
    target = _vector(task["target_xy_m"], 2, "task.target_xy_m")
    request = RepairRequest(
        schema=_string(root["schema"], "schema"),
        version=_integer(root["version"], "version"),
        family=_string(root["family"], "family"),
        task=RepairTask(
            _number(task["rail_y_lower_m"], "task.rail_y_lower_m"),
            (target[0], target[1]),
            _number(task["target_yaw_rad"], "task.target_yaw_rad"),
        ),
        controller=RepairController(
            _vector(
                controller["parameters_normalized"], PARAMETER_DIMENSION, "parameters_normalized"
            )
        ),
        limits=RepairLimits(
            _integer(limits.get("max_attempts", defaults.max_attempts), "limits.max_attempts"),
            _integer(
                limits.get("max_integration_steps", defaults.max_integration_steps),
                "limits.max_integration_steps",
            ),
            _number(
                limits.get("max_cpu_seconds", defaults.max_cpu_seconds), "limits.max_cpu_seconds"
            ),
        ),
    )
    return request


def _string(value: object, name: str) -> str:
    """Require a schema string without coercing other JSON types."""
    if type(value) is not str:
        raise RepairInputError(f"{name} must be a string")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject repeated member names at every object depth during decoding."""
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise RepairInputError("Duplicate JSON object member")
        result[name] = value
    return result


def _integer_token(token: str) -> int:
    """Retain integer-token identity without adding a decimal-digit quota."""
    return int(Decimal(token))


def _decimal_token(token: str) -> float:
    """Normalize a decimal token and refuse overflow instead of producing infinity."""
    value = float(token)
    if not math.isfinite(value):
        raise RepairInputError("A decimal JSON number does not fit finite binary64")
    return value


def _nonfinite_token(token: str) -> NoReturn:
    """Reject the nonstandard constants accepted by Python's default JSON decoder."""
    raise RepairInputError(f"Nonfinite JSON constant is not supported: {token}")


def _decode_request_document(payload: bytes) -> object:
    """Decode only the bounded decimal JSON domain used by repair requests."""
    if len(payload) > _REQUEST_JSON_LIMITS.max_bytes:
        raise RepairInputError("Request exceeds the 65536-byte limit")
    try:
        text = payload.decode("utf-8", errors="strict")
        if text.startswith("\ufeff"):
            raise RepairInputError("A UTF-8 byte-order mark is not allowed")
        parsed: object = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_int=_integer_token,
            parse_float=_decimal_token,
            parse_constant=_nonfinite_token,
        )
        enforce_json_structure(parsed, _REQUEST_JSON_LIMITS)
    except RepairInputError:
        raise
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise RepairInputError(f"Invalid bounded request JSON: {exc}") from exc
    return parsed


def _reader_os_failure(error: BaseException) -> bool:
    """Preserve the shared reader's underlying operating-system failure distinction."""
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, OSError):
            return True
        cause = cause.__cause__
    return False
