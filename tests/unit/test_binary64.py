"""Binary64 bit-preservation and classification tests."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from metrifid import Binary64, ExactRational


@pytest.mark.parametrize(
    ("bits", "classification"),
    [
        ("0000000000000000", "positive_zero"),
        ("8000000000000000", "negative_zero"),
        ("3ff0000000000000", "finite"),
        ("7fefffffffffffff", "finite"),
        ("7ff0000000000000", "positive_infinity"),
        ("fff0000000000000", "negative_infinity"),
        ("7ff8000000000001", "nan"),
        ("7ff0000000000001", "nan"),
    ],
)
def test_classification_and_primitive_round_trip(bits: str, classification: str) -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises classification and primitive round trip; equivalent values must
    serialize identically while malformed representations fail explicitly.
    """
    value = Binary64(bits)
    assert value.classification == classification
    assert Binary64.from_primitive(value.to_primitive()) == value
    assert value.is_finite is (classification in {"finite", "positive_zero", "negative_zero"})


@pytest.mark.parametrize(
    "bits",
    [
        "0000000000000000",
        "8000000000000000",
        "0000000000000001",
        "0010000000000000",
        "7fefffffffffffff",
        "7ff0000000000000",
        "fff0000000000000",
        "7ff8000000001234",
        "7ff0000000000001",
    ],
)
def test_float_bit_round_trip_preserves_payload(bits: str) -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises float bit round trip preserves payload; equivalent values must
    serialize identically while malformed representations fail explicitly.
    """
    original = Binary64(bits)
    restored = Binary64.from_float(original.to_float())
    assert restored.bits == bits


def test_signed_zero_is_preserved() -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises signed zero is preserved; equivalent values must serialize
    identically while malformed representations fail explicitly.
    """
    positive = Binary64.from_float(0.0)
    negative = Binary64.from_float(-0.0)
    assert positive.bits == "0000000000000000"
    assert negative.bits == "8000000000000000"
    assert math.copysign(1.0, negative.to_float()) == -1.0


@pytest.mark.parametrize(
    "bits",
    ["", "0" * 15, "0" * 17, "3FF0000000000000", "g" * 16, 1],
)
def test_invalid_bits_refuse(bits: object) -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises invalid bits refuse; equivalent values must serialize identically
    while malformed representations fail explicitly.
    """
    with pytest.raises((TypeError, ValueError)):
        Binary64(bits)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "primitive",
    [
        None,
        [],
        {"kind": "wrong", "bits": "3ff0000000000000"},
        {"kind": "ieee754_binary64"},
        {"kind": "ieee754_binary64", "bits": "3ff0000000000000", "extra": 1},
        {"kind": "ieee754_binary64", "bits": 1},
    ],
)
def test_invalid_primitive_refuses(primitive: object) -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises invalid primitive refuses; equivalent values must serialize
    identically while malformed representations fail explicitly.
    """
    with pytest.raises((TypeError, ValueError)):
        Binary64.from_primitive(primitive)


def test_from_float_rejects_non_float() -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises from float rejects non float; equivalent values must serialize
    identically while malformed representations fail explicitly.
    """
    with pytest.raises(TypeError):
        Binary64.from_float(1)


def test_exact_rational_conversion() -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises exact rational conversion; equivalent values must serialize
    identically while malformed representations fail explicitly.
    """
    assert Binary64("3ff0000000000000").to_exact_rational() == ExactRational(1, 1)
    assert Binary64("bfe0000000000000").to_exact_rational() == ExactRational(-1, 2)
    assert Binary64("0000000000000001").to_exact_rational() == ExactRational(1, 1 << 1074)
    assert Binary64("8000000000000000").to_exact_rational() == ExactRational(0, 1)
    assert Binary64.from_float(float(1 << 60)).to_exact_rational() == ExactRational(1 << 60, 1)
    with pytest.raises(ValueError):
        Binary64("7ff0000000000000").to_exact_rational()
    with pytest.raises(ValueError):
        Binary64("7ff8000000000001").to_exact_rational()


@given(st.integers(min_value=0, max_value=(1 << 64) - 1))
def test_property_arbitrary_64_bit_round_trip(raw: int) -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises property arbitrary 64 bit round trip; equivalent values must
    serialize identically while malformed representations fail explicitly.
    """
    bits = f"{raw:016x}"
    value = Binary64(bits)
    assert Binary64.from_primitive(value.to_primitive()) == value
    assert Binary64.from_float(value.to_float()).bits == bits
