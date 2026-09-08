"""Human-readable text for retained receipt values.

Display is never the numeric authority.  Every rendered number is decoded from the exact bits the
producer retained, and the retained representation itself is carried through unchanged beside it.
"""

from __future__ import annotations

from typing import Final

from ..json_values import Binary64, CanonicalValue

__all__ = [
    "NEGATIVE_INFINITY_TEXT",
    "NOT_A_NUMBER_TEXT",
    "POSITIVE_INFINITY_TEXT",
    "describe_value",
    "value_text",
]

NOT_A_NUMBER_TEXT: Final = "NaN"
POSITIVE_INFINITY_TEXT: Final = "+Inf"
NEGATIVE_INFINITY_TEXT: Final = "-Inf"
_NONFINITE_TEXT: Final = {
    "nan": NOT_A_NUMBER_TEXT,
    "positive_infinity": POSITIVE_INFINITY_TEXT,
    "negative_infinity": NEGATIVE_INFINITY_TEXT,
}
_ABSENT_TEXT: Final = "not retained"


def value_text(value: CanonicalValue) -> str:
    """Render one retained canonical value as display text.

    Tagged binary64 objects are decoded from their exact bits; a finite value is rendered as its
    shortest round-trip decimal, and the three nonfinite classes are named rather than printed as
    a numeral.  Any other canonical value is rendered structurally, and a sequence is rendered
    element by element so a vector reads as a vector.

    Args:
        value: One canonical value taken from a validated receipt.

    Returns:
        Display text for that value.
    """
    if value is None:
        return _ABSENT_TEXT
    if isinstance(value, dict):
        return _tagged_text(value)
    if isinstance(value, list):
        return "[" + ", ".join(value_text(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _tagged_text(value: dict[str, CanonicalValue]) -> str:
    """Render one canonical object, decoding the tagged binary64 representation."""
    if value.get("kind") != "ieee754_binary64":
        return "{" + ", ".join(f"{name}: {value_text(value[name])}" for name in sorted(value)) + "}"
    binary = Binary64.from_primitive(value)
    classification = binary.classification
    nonfinite = _NONFINITE_TEXT.get(classification)
    if nonfinite is not None:
        return nonfinite
    return repr(binary.to_float())


def describe_value(before: CanonicalValue, after: CanonicalValue) -> str:
    """Render one before/after transition as display text.

    Args:
        before: The retained baseline representation, or None when none was retained.
        after: The retained candidate representation, or None when none was retained.

    Returns:
        Display text naming both sides of the change.
    """
    return f"{value_text(before)} -> {value_text(after)}"
