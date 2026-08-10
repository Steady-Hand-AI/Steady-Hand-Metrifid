"""Deterministic canonical values, JSON bytes, and installed payload identity."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import TypeAlias, cast

CanonicalScalar: TypeAlias = bool | int | str | None
CanonicalValue: TypeAlias = CanonicalScalar | list["CanonicalValue"] | dict[str, "CanonicalValue"]
FrozenCanonicalValue: TypeAlias = (
    CanonicalScalar | tuple["FrozenCanonicalValue", ...] | Mapping[str, "FrozenCanonicalValue"]
)
FrozenCanonicalObject: TypeAlias = Mapping[str, FrozenCanonicalValue]

_BINARY64_RE = re.compile(r"[0-9a-f]{16}\Z")
_DECIMAL_RE = re.compile(r"(?:0|[1-9][0-9]*|0\.[0-9]+|[1-9][0-9]*\.[0-9]+)\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class Binary64:
    """One IEEE-754 binary64 value represented by its exact 64 bits."""

    bits: str

    def __post_init__(self) -> None:
        """Require exactly 64 bits encoded as lowercase hexadecimal text."""
        if not isinstance(self.bits, str):
            raise TypeError("bits must be a string")
        if _BINARY64_RE.fullmatch(self.bits) is None:
            raise ValueError("bits must be exactly 16 lowercase hexadecimal characters")

    @classmethod
    def from_float(cls, value: float) -> Binary64:
        """Capture the exact bits of a Python binary64 float."""
        if type(value) is not float:
            raise TypeError("value must be a Python float")
        return cls(struct.pack(">d", value).hex())

    @classmethod
    def from_primitive(cls, value: object) -> Binary64:
        """Parse the exact tagged primitive representation."""
        obj = _expect_exact_object(value, {"kind", "bits"}, "Binary64")
        if obj["kind"] != "ieee754_binary64":
            raise ValueError("Binary64 kind must be ieee754_binary64")
        bits = obj["bits"]
        if not isinstance(bits, str):
            raise TypeError("Binary64 bits must be a string")
        return cls(bits)

    def to_float(self) -> float:
        """Restore the represented binary64 value without numeric conversion."""
        return cast(float, struct.unpack(">d", bytes.fromhex(self.bits))[0])

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return the canonical tagged primitive object."""
        return {"kind": "ieee754_binary64", "bits": self.bits}

    @property
    def classification(self) -> str:
        """Return the frozen semantic classification of the bit pattern."""
        raw = int(self.bits, 16)
        sign = raw >> 63
        exponent = (raw >> 52) & 0x7FF
        fraction = raw & ((1 << 52) - 1)
        if exponent == 0x7FF:
            if fraction == 0:
                return "negative_infinity" if sign else "positive_infinity"
            return "nan"
        if exponent == 0 and fraction == 0:
            return "negative_zero" if sign else "positive_zero"
        return "finite"

    @property
    def is_finite(self) -> bool:
        """Whether the bit pattern is finite, including signed zero."""
        return self.classification in {"finite", "positive_zero", "negative_zero"}

    def to_exact_rational(self) -> ExactRational:
        """Convert a finite binary64 value to its exact dyadic rational."""
        raw = int(self.bits, 16)
        sign = -1 if raw >> 63 else 1
        exponent = (raw >> 52) & 0x7FF
        fraction = raw & ((1 << 52) - 1)
        if exponent == 0x7FF:
            raise ValueError("nonfinite binary64 values have no rational representation")
        if exponent == 0:
            significand = fraction
            exponent_two = -1074
        else:
            significand = (1 << 52) | fraction
            exponent_two = exponent - 1023 - 52
        numerator = sign * significand
        if exponent_two >= 0:
            numerator *= 1 << exponent_two
            denominator = 1
        else:
            denominator = 1 << (-exponent_two)
        return ExactRational(numerator, denominator)


@dataclass(frozen=True, slots=True)
class ExactRational:
    """A normalized exact rational with a strictly positive denominator."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        """Normalize the fraction and require a strictly positive denominator."""
        numerator = _strict_int(self.numerator, "numerator")
        denominator = _strict_int(self.denominator, "denominator")
        if denominator <= 0:
            raise ValueError("denominator must be strictly positive")
        fraction = Fraction(numerator, denominator)
        object.__setattr__(self, "numerator", fraction.numerator)
        object.__setattr__(self, "denominator", fraction.denominator)

    @classmethod
    def from_primitive(cls, value: object) -> ExactRational:
        """Parse and normalize the exact two-field primitive object."""
        obj = _expect_exact_object(value, {"numerator", "denominator"}, "ExactRational")
        return cls(
            _strict_int(obj["numerator"], "numerator"),
            _strict_int(obj["denominator"], "denominator"),
        )

    @classmethod
    def from_decimal_token(cls, token: str) -> ExactRational:
        """Parse the frozen nonnegative decimal-token grammar exactly."""
        if type(token) is not str:
            raise TypeError("decimal token must be a string")
        if _DECIMAL_RE.fullmatch(token) is None:
            raise ValueError("invalid decimal token")
        if "." not in token:
            return cls(int(token), 1)
        integer_text, fraction_text = token.split(".", 1)
        denominator = 10 ** len(fraction_text)
        numerator = int(integer_text) * denominator + int(fraction_text)
        return cls(numerator, denominator)

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return the normalized exact-rational primitive object."""
        return {"numerator": self.numerator, "denominator": self.denominator}

    def to_decimal_token(self) -> str:
        """Return the shortest exact ordinary-decimal token when one exists."""
        denominator = self.denominator
        twos = 0
        fives = 0
        while denominator % 2 == 0:
            denominator //= 2
            twos += 1
        while denominator % 5 == 0:
            denominator //= 5
            fives += 1
        if denominator != 1:
            raise ValueError("rational does not have a finite decimal expansion")
        scale = max(twos, fives)
        scaled = self.numerator * (2 ** (scale - twos)) * (5 ** (scale - fives))
        sign = "-" if scaled < 0 else ""
        digits = str(abs(scaled))
        if scale == 0:
            return f"{sign}{digits}"
        digits = digits.rjust(scale + 1, "0")
        whole = digits[:-scale]
        fractional = digits[-scale:].rstrip("0")
        return f"{sign}{whole}.{fractional}"

    def multiplied_by_int(self, multiplier: int) -> ExactRational:
        """Multiply by one strict integer without converting through float."""
        return ExactRational(
            self.numerator * _strict_int(multiplier, "multiplier"), self.denominator
        )


def strict_json_loads(data: str | bytes) -> CanonicalValue:
    """Load strict JSON while rejecting duplicate keys and every raw float token."""
    if isinstance(data, bytes):
        text = data.decode("utf-8", errors="strict")
    elif isinstance(data, str):
        text = data
    else:
        raise TypeError("strict_json_loads accepts UTF-8 text or bytes only")
    parsed = json.loads(
        text,
        object_pairs_hook=_object_from_pairs,
        parse_float=_reject_float_token,
        parse_constant=_reject_constant_token,
    )
    validated = _validate_canonical(parsed)
    return cast(CanonicalValue, validated)


def canonical_json_bytes(value: CanonicalValue) -> bytes:
    """Encode one validated canonical semantic object to frozen JSON bytes."""
    validated = _validate_canonical(value)
    text = json.dumps(
        validated,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return text.encode("utf-8", errors="strict")


def canonical_sha256(value: CanonicalValue) -> str:
    """Return lowercase SHA-256 over canonical JSON bytes."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def compute_self_hash(value: dict[str, CanonicalValue], field: str) -> str:
    """Compute a canonical self-hash after removing the named field entirely."""
    if type(value) is not dict:
        raise TypeError("self-hashed value must be an object")
    if type(field) is not str or not field:
        raise TypeError("self-hash field must be a nonempty string")
    if field not in value:
        raise ValueError(f"self-hash field {field!r} is missing")
    unhashed = dict(value)
    del unhashed[field]
    return canonical_sha256(unhashed)


def validate_self_hash(value: dict[str, CanonicalValue], field: str) -> None:
    """Raise when a self-hash is absent, malformed, or does not recompute."""
    actual = value.get(field)
    if not isinstance(actual, str) or _SHA256_RE.fullmatch(actual) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 string")
    expected = compute_self_hash(value, field)
    if actual != expected:
        raise ValueError(f"{field} does not match canonical content")


def freeze_canonical(value: CanonicalValue) -> FrozenCanonicalValue:
    """Recursively freeze a validated canonical value for immutable schemas."""
    validated = _validate_canonical(value)
    return _freeze_validated(cast(CanonicalValue, validated))


def thaw_canonical(value: FrozenCanonicalValue) -> CanonicalValue:
    """Return a fresh mutable canonical primitive from a frozen value."""
    if value is None or type(value) in {bool, int, str}:
        return cast(CanonicalScalar, value)
    if isinstance(value, tuple):
        return [thaw_canonical(item) for item in value]
    if isinstance(value, Mapping):
        return {key: thaw_canonical(item) for key, item in value.items()}
    raise TypeError("invalid frozen canonical value")


def require_sha256(value: object, field: str) -> str:
    """Validate and return one frozen lowercase SHA-256 string."""
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be exactly 64 lowercase hexadecimal characters")
    return value


def installed_distribution_identity() -> dict[str, CanonicalValue]:
    """Return the identity of the installed wheel whose code is executing."""
    from .distribution import installed_distribution_identity as implementation

    return implementation()


def installed_distribution_sha256() -> str:
    """Return SHA-256 over the bound installed distribution identity."""
    from .distribution import installed_distribution_sha256 as implementation

    return implementation()


def _expect_exact_object(value: object, fields: set[str], context: str) -> dict[str, object]:
    """Require a concrete JSON object with exactly the expected canonical fields."""
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    obj = cast(dict[str, object], value)
    actual = set(obj)
    missing = fields - actual
    unknown = actual - fields
    if missing:
        raise ValueError(f"{context} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{context} unknown fields: {sorted(unknown)}")
    return obj


def _strict_int(value: object, field: str) -> int:
    """Admit a built-in JSON integer while excluding booleans."""
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer and not a boolean")
    return value


def _object_from_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    """Build a strict JSON object while rejecting duplicate member names."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_float_token(token: str) -> float:
    """Reject raw JSON decimals so numbers cannot bypass exact tagged forms."""
    raise ValueError(f"raw JSON floating-point token is forbidden: {token}")


def _reject_constant_token(token: str) -> float:
    """Reject nonstandard JSON constants such as NaN and Infinity."""
    raise ValueError(f"non-standard JSON numeric token is forbidden: {token}")


def _validate_canonical(value: object, active: set[int] | None = None) -> object:
    """Recursively validate one canonical value while detecting container cycles."""
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is str:
        value.encode("utf-8", errors="strict")
        return value
    if type(value) is list:
        return _validate_canonical_array(cast(list[object], value), active or set())
    if type(value) is dict:
        return _validate_canonical_object(cast(dict[object, object], value), active or set())
    raise TypeError(f"unsupported canonical primitive type: {type(value).__name__}")


def _validate_canonical_array(value: list[object], active: set[int]) -> list[object]:
    """Validate one canonical array, including recursive-cycle protection."""
    identity = id(value)
    if identity in active:
        raise ValueError("canonical arrays must not contain cycles")
    active.add(identity)
    try:
        for item in value:
            _validate_canonical(item, active)
    finally:
        active.remove(identity)
    return value


def _validate_canonical_object(
    value: dict[object, object], active: set[int]
) -> dict[object, object]:
    """Validate one canonical object, including keys, tags, and cycle protection."""
    identity = id(value)
    if identity in active:
        raise ValueError("canonical objects must not contain cycles")
    active.add(identity)
    try:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("canonical object keys must be strings")
            key.encode("utf-8", errors="strict")
            _validate_canonical(item, active)
        _validate_tagged_object(cast(dict[str, object], value))
    finally:
        active.remove(identity)
    return value


def _validate_tagged_object(obj: dict[str, object]) -> None:
    """Validate binary64 and exact-rational objects when their tag shape is present."""
    if obj.get("kind") == "ieee754_binary64":
        Binary64.from_primitive(obj)
    if set(obj) == {"numerator", "denominator"}:
        rational = ExactRational.from_primitive(obj)
        if rational.to_primitive() != obj:
            raise ValueError("ExactRational primitive must already be normalized")


def _freeze_validated(value: CanonicalValue) -> FrozenCanonicalValue:
    """Recursively convert validated arrays and objects to immutable canonical containers."""
    if value is None or type(value) in {bool, int, str}:
        return cast(CanonicalScalar, value)
    if type(value) is list:
        return tuple(_freeze_validated(item) for item in value)
    obj = cast(dict[str, CanonicalValue], value)
    return MappingProxyType({key: _freeze_validated(item) for key, item in obj.items()})


__all__ = [
    "Binary64",
    "ExactRational",
    "canonical_json_bytes",
    "canonical_sha256",
    "strict_json_loads",
]
