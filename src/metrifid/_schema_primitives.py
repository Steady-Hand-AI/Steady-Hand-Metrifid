"""Primitive validators shared by strict schema modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import PurePosixPath
from typing import cast

from .errors import EngineThreadpoolState
from .json_values import ExactRational, require_sha256


def _object(value: object, context: str) -> dict[str, object]:
    """Admit only a concrete JSON object with string keys."""
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    obj = cast(dict[object, object], value)
    for key in obj:
        if type(key) is not str:
            raise TypeError(f"{context} keys must be strings")
    return cast(dict[str, object], obj)


def _string_keyed_mapping(value: object, field: str) -> Mapping[str, object]:
    """Admit a mapping only when every key is a string."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    for key in value:
        if type(key) is not str:
            raise TypeError(f"{field} keys must be strings")
    return cast(Mapping[str, object], value)


def _fields(obj: Mapping[str, object], expected: set[str], context: str) -> None:
    """Require an object's keys to exactly equal its frozen schema field set."""
    actual = set(obj)
    missing = expected - actual
    unknown = actual - expected
    if unknown:
        raise ValueError(f"{context} unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{context} missing fields: {sorted(missing)}")


def _string(value: object, field: str) -> str:
    """Admit only a strict UTF-8 JSON string for the named field."""
    if type(value) is not str:
        raise TypeError(f"{field} must be a string")
    result = value
    result.encode("utf-8", errors="strict")
    return result


def _nonempty_string(value: object, field: str) -> str:
    """Admit a string only when it contains at least one character."""
    result = _string(value, field)
    if not result:
        raise ValueError(f"{field} must be nonempty")
    return result


def _name(value: object, field: str) -> str:
    """Admit a nonempty UTF-8 semantic name without NUL characters."""
    result = _nonempty_string(value, field)
    if "\x00" in result:
        raise ValueError(f"{field} must not contain U+0000")
    return result


def _exact_int(value: object, field: str) -> int:
    """Admit a built-in integer while explicitly excluding booleans."""
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer and not a boolean")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    """Admit a strict integer greater than or equal to zero."""
    result = _exact_int(value, field)
    if result < 0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def _bounded_int(value: object, field: str, minimum: int, maximum: int) -> int:
    """Admit a strict integer within the declared inclusive bounds."""
    result = _exact_int(value, field)
    if not minimum <= result <= maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return result


def _sequence(value: object, field: str) -> list[object]:
    """Admit only a concrete JSON array for the named field."""
    if type(value) is not list:
        raise TypeError(f"{field} must be an array")
    return cast(list[object], value)


def _string_sequence(value: object, field: str) -> tuple[str, ...]:
    """Decode a JSON array whose every element is a nonempty string."""
    return tuple(_nonempty_string(item, field) for item in _sequence(value, field))


def _name_sequence(value: object, field: str) -> tuple[str, ...]:
    """Decode a JSON array of nonempty NUL-free UTF-8 semantic names."""
    return tuple(_name(item, field) for item in _sequence(value, field))


def _int_sequence(value: object, field: str) -> tuple[int, ...]:
    """Decode a JSON array of nonnegative strict integers."""
    return tuple(_nonnegative_int(item, field) for item in _sequence(value, field))


def _positive_decimal(value: object, field: str) -> ExactRational:
    """Parse a frozen decimal token and require a value greater than zero."""
    token = _string(value, field)
    rational = ExactRational.from_decimal_token(token)
    return _positive_rational(rational, field)


def _positive_rational_primitive(value: object, field: str) -> ExactRational:
    """Decode a normalized exact-rational object and require positive magnitude."""
    return _positive_rational(ExactRational.from_primitive(value), field)


def _positive_rational(value: object, field: str) -> ExactRational:
    """Admit an ``ExactRational`` instance only when it is strictly positive."""
    if not isinstance(value, ExactRational):
        raise TypeError(f"{field} must be an ExactRational")
    if value.numerator <= 0:
        raise ValueError(f"{field} must be strictly positive")
    return value


def _optional_hash(value: object, field: str) -> str | None:
    """Admit either ``None`` or one lowercase SHA-256 digest."""
    if value is None:
        return None
    return require_sha256(value, field)


def _relative_posix_path(value: object, field: str) -> str:
    """Admit a normalized relative POSIX path with no parent traversal."""
    raw = _nonempty_string(value, field)
    if "\\" in raw:
        raise ValueError(f"{field} must use POSIX separators")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a normalized relative POSIX path")
    normalized = path.as_posix()
    if normalized != raw:
        raise ValueError(f"{field} must already be normalized")
    return normalized


def _unique_names(names: Sequence[str], field: str) -> None:
    """Reject invalid or duplicate semantic names in a schema sequence."""
    for name in names:
        _name(name, field)
    if len(names) != len(set(names)):
        raise ValueError(f"{field} must contain unique names")


def _sorted_unique_names(names: Sequence[str], field: str) -> None:
    """Require semantic names to be unique and Unicode-code-point ordered."""
    _unique_names(names, field)
    if tuple(names) != tuple(sorted(names)):
        raise ValueError(f"{field} must be sorted by exact Unicode code points")


def _unique_strings(values: Sequence[str], field: str) -> None:
    """Reject empty or duplicate string values in a schema sequence."""
    for value in values:
        _nonempty_string(value, field)
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must not contain duplicates")


def _validate_offsets(
    offsets: Sequence[int], item_count: int, value_count: int, field: str
) -> None:
    """Validate segmented-array offsets against item and flattened-value counts."""
    if len(offsets) != item_count + 1:
        raise ValueError(f"{field} length must equal item count + 1")
    if not offsets or offsets[0] != 0:
        raise ValueError(f"{field} must start at zero")
    if any(left > right for left, right in pairwise(offsets)):
        raise ValueError(f"{field} must be nondecreasing")
    if offsets[-1] != value_count:
        raise ValueError(f"{field} final offset must equal the value count")


def _require_integral_ratio(control_dt: ExactRational, step_dt: ExactRational, field: str) -> None:
    """Require a control interval to contain an exact positive integer number of steps."""
    numerator = control_dt.numerator * step_dt.denominator
    denominator = control_dt.denominator * step_dt.numerator
    if numerator % denominator != 0:
        raise ValueError(f"control_dt / {field} must be a positive exact integer")


def _require_instance(value: object, expected_type: type[object], field: str) -> None:
    """Require a schema field to have its declared runtime type."""
    if not isinstance(value, expected_type):
        raise TypeError(f"{field} must be a {expected_type.__name__}")


def _require_typed_tuple(value: object, expected_type: type[object], field: str) -> None:
    """Require a tuple whose every member has the declared runtime type."""
    if type(value) is not tuple:
        raise TypeError(f"{field} must be a tuple")
    if any(not isinstance(item, expected_type) for item in value):
        raise TypeError(f"{field} must contain only {expected_type.__name__} values")


def _require_string_tuple(value: object, field: str, *, names: bool = False) -> None:
    """Require a string tuple and optionally validate every member as a semantic name."""
    if type(value) is not tuple:
        raise TypeError(f"{field} must be a tuple")
    validator = _name if names else _nonempty_string
    for item in value:
        validator(item, field)


def _require_int_tuple(value: object, field: str) -> None:
    """Require a tuple of nonnegative strict integers, excluding booleans."""
    if type(value) is not tuple:
        raise TypeError(f"{field} must be a tuple")
    for item in value:
        _nonnegative_int(item, field)


def _require_mapping_tuple(value: object, field: str) -> None:
    """Require a tuple whose every member is a canonical-object mapping."""
    if type(value) is not tuple:
        raise TypeError(f"{field} must be a tuple")
    if any(not isinstance(item, Mapping) for item in value):
        raise TypeError(f"{field} must contain only canonical objects")


def _engine_threadpool_state(value: object) -> EngineThreadpoolState:
    """Parse a token through the closed MuJoCo threadpool-state registry."""
    token = _nonempty_string(value, "engine_threadpool_state")
    try:
        return EngineThreadpoolState(token)
    except ValueError as exc:
        raise ValueError(f"unknown engine_threadpool_state: {token}") from exc
