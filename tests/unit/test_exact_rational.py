"""Exact rational and decimal-token grammar tests."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from metrifid import ExactRational


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [
        (2, 4, ExactRational(1, 2)),
        (-2, 4, ExactRational(-1, 2)),
        (0, 99, ExactRational(0, 1)),
        (123, 1, ExactRational(123, 1)),
    ],
)
def test_normalization(numerator: int, denominator: int, expected: ExactRational) -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises normalization; equivalent values must serialize identically while
    malformed representations fail explicitly.
    """
    assert ExactRational(numerator, denominator) == expected


@pytest.mark.parametrize("denominator", [0, -1, -99])
def test_nonpositive_denominator_refuses_before_reduction(denominator: int) -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises nonpositive denominator refuses before reduction; equivalent values
    must serialize identically while malformed representations fail explicitly.
    """
    with pytest.raises(ValueError, match="strictly positive"):
        ExactRational(0, denominator)
    with pytest.raises(ValueError, match="strictly positive"):
        ExactRational(1, denominator)
    with pytest.raises(ValueError, match="strictly positive"):
        ExactRational.from_primitive({"numerator": 1, "denominator": denominator})


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("0", ExactRational(0, 1)),
        ("7", ExactRational(7, 1)),
        ("0.0", ExactRational(0, 1)),
        ("0.002", ExactRational(1, 500)),
        ("12.3400", ExactRational(617, 50)),
    ],
)
def test_accepted_decimal_grammar(token: str, expected: ExactRational) -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises accepted decimal grammar; equivalent values must serialize
    identically while malformed representations fail explicitly.
    """
    assert ExactRational.from_decimal_token(token) == expected


@pytest.mark.parametrize(
    "token",
    ["", "+1", "-1", "01", "01.2", ".1", "1.", "1e2", "1_000", " 1", "1 ", "NaN", "Infinity"],
)
def test_rejected_decimal_grammar(token: str) -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises rejected decimal grammar; equivalent values must serialize
    identically while malformed representations fail explicitly.
    """
    with pytest.raises(ValueError):
        ExactRational.from_decimal_token(token)


def test_decimal_parser_rejects_bare_float() -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises decimal parser rejects bare float; equivalent values must serialize
    identically while malformed representations fail explicitly.
    """
    with pytest.raises(TypeError):
        ExactRational.from_decimal_token(0.1)  # type: ignore[arg-type]


@pytest.mark.parametrize("args", [(True, 1), (1, False)])
def test_constructor_refuses_bool(args: tuple[object, object]) -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises constructor refuses bool; equivalent values must serialize
    identically while malformed representations fail explicitly.
    """
    with pytest.raises(TypeError):
        ExactRational(*args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "primitive",
    [
        None,
        [],
        {"numerator": 1},
        {"numerator": 1, "denominator": 2, "extra": 3},
        {"numerator": True, "denominator": 1},
        {"numerator": 1, "denominator": False},
    ],
)
def test_invalid_primitive_refuses(primitive: object) -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises invalid primitive refuses; equivalent values must serialize
    identically while malformed representations fail explicitly.
    """
    with pytest.raises((TypeError, ValueError)):
        ExactRational.from_primitive(primitive)


def test_primitive_and_decimal_rendering() -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises primitive and decimal rendering; equivalent values must serialize
    identically while malformed representations fail explicitly.
    """
    assert ExactRational(2, 4).to_primitive() == {"numerator": 1, "denominator": 2}
    assert ExactRational(1, 500).to_decimal_token() == "0.002"
    assert ExactRational(20, 10).to_decimal_token() == "2"
    assert ExactRational(-1, 8).to_decimal_token() == "-0.125"
    assert ExactRational(1234, 100).to_decimal_token() == "12.34"
    with pytest.raises(ValueError):
        ExactRational(1, 3).to_decimal_token()


def test_integer_multiplication() -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises integer multiplication; equivalent values must serialize identically
    while malformed representations fail explicitly.
    """
    assert ExactRational(1, 500).multiplied_by_int(5) == ExactRational(1, 100)
    assert ExactRational(1, 2).multiplied_by_int(-3) == ExactRational(-3, 2)
    with pytest.raises(TypeError):
        ExactRational(1, 2).multiplied_by_int(True)


@given(
    st.integers(min_value=-(10**12), max_value=10**12),
    st.integers(min_value=1, max_value=10**9),
)
def test_property_normalization_is_idempotent(numerator: int, denominator: int) -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises property normalization is idempotent; equivalent values must
    serialize identically while malformed representations fail explicitly.
    """
    first = ExactRational(numerator, denominator)
    second = ExactRational.from_primitive(first.to_primitive())
    assert second == first
    assert second.denominator > 0
