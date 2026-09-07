"""Display text for retained receipt values must be exact, signed, and native-free.

The display helpers are the only place where a retained bit pattern becomes prose, so these tests
pin the rendering of finite doubles to the shortest round-trip decimal, hold the sign of a signed
zero, name the three nonfinite classes through the module constants rather than through numerals,
and prove that reading a receipt never pulls MuJoCo or NumPy into the interpreter.
"""

from __future__ import annotations

import math
import subprocess
import sys
import textwrap

import pytest

from metrifid.json_values import Binary64, CanonicalValue
from metrifid.result._display import (
    NEGATIVE_INFINITY_TEXT,
    NOT_A_NUMBER_TEXT,
    POSITIVE_INFINITY_TEXT,
    describe_value,
    value_text,
)

_AWKWARD_DOUBLES: tuple[float, ...] = (
    0.1,
    0.2,
    0.1 + 0.2,
    1.0 / 3.0,
    231.46257114070207,
    0.026407058823529415,
    1.2345678901234567e-05,
    9007199254740992.0,
    5e-324,
    2.2250738585072014e-308,
    1.7976931348623157e308,
    -1.7976931348623157e308,
    -0.026407058823529415,
)

_LAST_BIT_NEIGHBOURS: tuple[float, ...] = (
    0.1,
    1.0,
    1e-07,
    231.46257114070207,
    0.026407058823529415,
)

_BLOCK_NATIVE_IMPORTS = textwrap.dedent(
    """
    import sys

    class _BlockNative:
        def find_spec(self, name, path=None, target=None):
            if name.split(".", 1)[0] in {"mujoco", "numpy"}:
                raise ModuleNotFoundError(f"{name} is unavailable in this environment")
            return None

    sys.meta_path.insert(0, _BlockNative())

    import importlib

    module = importlib.import_module("metrifid.result._display")
    native = sorted(
        name for name in sys.modules if name.split(".", 1)[0] in {"mujoco", "numpy"}
    )
    assert native == [], f"native modules imported: {native}"
    rendered = module.value_text({"kind": "ieee754_binary64", "bits": "406ceecd61fe2c71"})
    assert rendered == "231.46257114070207", rendered
    assert module.value_text(None) != "None", "absence rendered as None"
    print("PURE_DISPLAY_OK")
    """
)


def _tagged(bits: str) -> dict[str, CanonicalValue]:
    """Build the canonical tagged binary64 object for one hexadecimal bit pattern."""
    return Binary64(bits).to_primitive()


def _tagged_float(value: float) -> dict[str, CanonicalValue]:
    """Build the canonical tagged binary64 object for one Python float."""
    return Binary64.from_float(value).to_primitive()


@pytest.mark.parametrize(
    ("bits", "expected"),
    [
        ("3ff8000000000000", "1.5"),
        ("4000000000000000", "2.0"),
        ("406ceecd61fe2c71", "231.46257114070207"),
        ("3f9b0a73b81f5776", "0.026407058823529415"),
    ],
)
def test_a_finite_binary64_renders_as_its_shortest_round_trip_decimal(
    bits: str, expected: str
) -> None:
    """Render each finite bit pattern as the exact shortest decimal that reproduces it."""
    assert value_text(_tagged(bits)) == expected


@pytest.mark.parametrize("value", _AWKWARD_DOUBLES)
def test_an_awkward_double_survives_a_round_trip_through_its_rendered_text(value: float) -> None:
    """Recover the identical double by parsing the text rendered for its retained bits."""
    rendered = value_text(_tagged_float(value))
    assert float(rendered) == value
    assert Binary64.from_float(float(rendered)) == Binary64.from_float(value)


@pytest.mark.parametrize("value", _LAST_BIT_NEIGHBOURS)
def test_two_doubles_differing_only_in_the_final_bit_render_as_different_text(
    value: float,
) -> None:
    """Distinguish adjacent doubles that a twelve-significant-digit format would collapse."""
    successor = math.nextafter(value, math.inf)
    assert Binary64.from_float(value) != Binary64.from_float(successor)
    assert f"{value:.12g}" == f"{successor:.12g}"
    lower = value_text(_tagged_float(value))
    upper = value_text(_tagged_float(successor))
    assert lower != upper
    assert float(lower) == value
    assert float(upper) == successor


@pytest.mark.parametrize(
    ("bits", "expected"),
    [
        ("0000000000000000", "0.0"),
        ("8000000000000000", "-0.0"),
    ],
)
def test_a_signed_zero_keeps_its_sign_in_the_rendered_text(bits: str, expected: str) -> None:
    """Preserve the sign of a signed zero through rendering and back through parsing."""
    rendered = value_text(_tagged(bits))
    assert rendered == expected
    assert math.copysign(1.0, float(rendered)) == math.copysign(1.0, Binary64(bits).to_float())


def test_the_two_signed_zeroes_render_as_distinct_text() -> None:
    """Keep positive and negative zero apart in display even though they compare equal."""
    positive = value_text(_tagged("0000000000000000"))
    negative = value_text(_tagged("8000000000000000"))
    assert positive != negative
    assert float(positive) == float(negative)


def test_the_nonfinite_display_constants_hold_their_frozen_text() -> None:
    """Freeze the three nonfinite display names as published text."""
    assert NOT_A_NUMBER_TEXT == "NaN"
    assert POSITIVE_INFINITY_TEXT == "+Inf"
    assert NEGATIVE_INFINITY_TEXT == "-Inf"


@pytest.mark.parametrize(
    ("bits", "expected"),
    [
        ("7ff8000000000000", NOT_A_NUMBER_TEXT),
        ("7ff0000000000000", POSITIVE_INFINITY_TEXT),
        ("fff0000000000000", NEGATIVE_INFINITY_TEXT),
    ],
)
def test_a_nonfinite_binary64_renders_as_its_named_class_rather_than_a_numeral(
    bits: str, expected: str
) -> None:
    """Name each nonfinite class instead of printing a numeral for it."""
    assert value_text(_tagged(bits)) == expected


@pytest.mark.parametrize(
    "bits",
    [
        "7ff8000000000000",
        "fff8000000000000",
        "7ff4000000000000",
        "7ff0000000000001",
        "fff0000000000001",
    ],
)
def test_every_quiet_and_signalling_nan_payload_renders_as_the_not_a_number_text(
    bits: str,
) -> None:
    """Render quiet and signalling NaN payloads alike as the single not-a-number name."""
    assert Binary64(bits).classification == "nan"
    assert value_text(_tagged(bits)) == NOT_A_NUMBER_TEXT


def test_a_list_of_tagged_values_renders_as_a_bracketed_comma_joined_vector() -> None:
    """Render a retained vector as its bracketed, comma-joined element text."""
    vector: list[CanonicalValue] = [_tagged_float(item) for item in (1.5, -0.0, 231.46257114070207)]
    assert value_text(vector) == "[1.5, -0.0, 231.46257114070207]"


def test_a_vector_renders_nonfinite_absent_and_nested_elements_element_by_element() -> None:
    """Render every element class inside a vector, including nesting and an empty vector."""
    assert value_text([]) == "[]"
    mixed: list[CanonicalValue] = [_tagged("7ff0000000000000"), None, 3, True]
    assert value_text(mixed) == "[+Inf, not retained, 3, true]"
    nested: list[CanonicalValue] = [[_tagged_float(1.5)], [_tagged_float(2.0)]]
    assert value_text(nested) == "[[1.5], [2.0]]"


def test_a_value_that_was_not_retained_renders_as_prose_rather_than_none_or_zero() -> None:
    """Say that nothing was retained instead of printing None or a fabricated zero."""
    rendered = value_text(None)
    assert rendered == "not retained"
    assert rendered != "None"
    assert rendered != "0"
    assert rendered != "0.0"
    assert rendered != ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (7, "7"),
        (-3, "-3"),
        (0, "0"),
        (1180591620717411303424, "1180591620717411303424"),
        (True, "true"),
        (False, "false"),
        ("free_joint", "free_joint"),
        ("", ""),
        ("1.5", "1.5"),
    ],
)
def test_a_plain_scalar_renders_without_decoration_and_a_boolean_renders_lowercase(
    value: CanonicalValue, expected: str
) -> None:
    """Render plain integers and strings verbatim and booleans as lowercase JSON words."""
    assert value_text(value) == expected


def test_describe_value_renders_both_sides_separated_by_an_arrow() -> None:
    """Join the retained baseline and candidate text with the frozen arrow separator."""
    assert describe_value(_tagged("3ff8000000000000"), _tagged("4000000000000000")) == "1.5 -> 2.0"
    assert describe_value(None, _tagged("406ceecd61fe2c71")) == "not retained -> 231.46257114070207"
    assert describe_value(_tagged("7ff0000000000000"), None) == "+Inf -> not retained"
    assert describe_value([_tagged_float(1.5)], "off") == "[1.5] -> off"


def test_an_untagged_object_renders_structurally_instead_of_raising() -> None:
    """Render an object without the binary64 tag as sorted structural text."""
    assert value_text({}) == "{}"
    assert value_text({"numerator": 1, "denominator": 3}) == "{denominator: 3, numerator: 1}"
    other_tag: dict[str, CanonicalValue] = {"kind": "exact_rational", "numerator": 1}
    assert value_text(other_tag) == "{kind: exact_rational, numerator: 1}"


def test_an_untagged_object_decodes_a_tagged_binary64_nested_inside_it() -> None:
    """Decode a tagged binary64 that is nested inside an otherwise structural object."""
    nested: dict[str, CanonicalValue] = {"count": 2, "stat": _tagged("3ff8000000000000")}
    assert value_text(nested) == "{count: 2, stat: 1.5}"


def test_the_display_module_imports_and_renders_with_numpy_and_mujoco_blocked() -> None:
    """Import and exercise the display module in an interpreter where natives cannot load."""
    completed = subprocess.run(
        [sys.executable, "-c", _BLOCK_NATIVE_IMPORTS],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PURE_DISPLAY_OK" in completed.stdout
