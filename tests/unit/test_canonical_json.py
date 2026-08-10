"""Strict JSON, canonical bytes, and self-hash tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from metrifid import (
    Binary64,
    ComparisonReceipt,
    ExactRational,
    ReasonRecord,
    canonical_json_bytes,
    canonical_sha256,
    strict_json_loads,
    validate_receipt,
)
from metrifid.errors import ordered_reasons, projected_reason_codes
from metrifid.json_values import (
    CanonicalValue,
    compute_self_hash,
    freeze_canonical,
    require_sha256,
    thaw_canonical,
    validate_self_hash,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"b": 2, "a": 1}, b'{"a":1,"b":2}'),
        ({"z": "é", "a": "λ"}, '{"a":"λ","z":"é"}'.encode()),
        ({"line": "a\nb", "quote": '"'}, b'{"line":"a\\nb","quote":"\\""}'),
        ([None, True, False, 3, "x"], b'[null,true,false,3,"x"]'),
    ],
)
def test_canonical_bytes(value: CanonicalValue, expected: bytes) -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises canonical bytes; equivalent values must serialize identically while
    malformed representations fail explicitly.
    """
    actual = canonical_json_bytes(value)
    assert actual == expected
    assert not actual.startswith(b"\xef\xbb\xbf")
    assert not actual.endswith(b"\n")
    assert canonical_sha256(value) == hashlib.sha256(expected).hexdigest()


def test_key_order_uses_unicode_code_points() -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises key order uses unicode code points; equivalent values must serialize
    identically while malformed representations fail explicitly.
    """
    value: CanonicalValue = {"é": 1, "z": 2, "a": 3, "λ": 4}
    assert canonical_json_bytes(value) == '{"a":3,"z":2,"é":1,"λ":4}'.encode()


@pytest.mark.parametrize(
    "text",
    [
        '{"a":1,"a":2}',
        '{"outer":{"x":1,"x":2}}',
        "[1.0]",
        '{"x":1e2}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":-Infinity}',
    ],
)
def test_strict_loader_rejects_duplicate_and_raw_float_tokens(text: str) -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises strict loader rejects duplicate and raw float tokens; equivalent
    values must serialize identically while malformed representations fail explicitly.
    """
    with pytest.raises(ValueError):
        strict_json_loads(text)


def test_strict_loader_accepts_text_and_bytes() -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises strict loader accepts text and bytes; equivalent values must
    serialize identically while malformed representations fail explicitly.
    """
    expected: CanonicalValue = {"x": [1, "é", True, None]}
    assert strict_json_loads('{"x":[1,"é",true,null]}') == expected
    assert strict_json_loads('{"x":[1,"é",true,null]}'.encode()) == expected


def test_strict_loader_rejects_invalid_input_and_utf8() -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises strict loader rejects invalid input and utf8; equivalent values must
    serialize identically while malformed representations fail explicitly.
    """
    with pytest.raises(TypeError):
        strict_json_loads(bytearray(b"{}"))  # type: ignore[arg-type]
    with pytest.raises(UnicodeDecodeError):
        strict_json_loads(b'"\xff"')
    with pytest.raises(UnicodeEncodeError):
        strict_json_loads('"\\ud800"')


@pytest.mark.parametrize(
    "value",
    [
        1.0,
        {"x": 1.0},
        (1, 2),
        b"bytes",
        {1: "not-string"},
        {"kind": "ieee754_binary64", "bits": "BAD"},
        {"kind": "ieee754_binary64", "bits": "3ff0000000000000", "extra": 1},
        {"numerator": 2, "denominator": 4},
    ],
)
def test_noncanonical_values_refuse(value: object) -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises noncanonical values refuse; equivalent values must serialize
    identically while malformed representations fail explicitly.
    """
    with pytest.raises((TypeError, ValueError)):
        canonical_json_bytes(value)  # type: ignore[arg-type]


def test_cycles_refuse() -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises cycles refuse; equivalent values must serialize identically while
    malformed representations fail explicitly.
    """
    array: list[CanonicalValue] = []
    array.append(array)
    with pytest.raises(ValueError):
        canonical_json_bytes(array)
    obj: dict[str, CanonicalValue] = {}
    obj["self"] = obj
    with pytest.raises(ValueError):
        canonical_json_bytes(obj)


def test_arbitrary_mapping_is_not_a_canonical_primitive() -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises arbitrary mapping is not a canonical primitive; equivalent values
    must serialize identically while malformed representations fail explicitly.
    """
    frozen = freeze_canonical({"a": [1, 2]})
    assert isinstance(frozen, Mapping)
    with pytest.raises(TypeError):
        canonical_json_bytes(frozen)  # type: ignore[arg-type]
    assert thaw_canonical(frozen) == {"a": [1, 2]}
    with pytest.raises(TypeError):
        thaw_canonical(object())  # type: ignore[arg-type]


def test_self_hash_removes_field_instead_of_nulling() -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises self hash removes field instead of nulling; equivalent values must
    serialize identically while malformed representations fail explicitly.
    """
    value: dict[str, CanonicalValue] = {"payload": {"x": 1}, "sha256": None}
    digest = compute_self_hash(value, "sha256")
    assert digest == canonical_sha256({"payload": {"x": 1}})
    assert digest != canonical_sha256({"payload": {"x": 1}, "sha256": None})
    value["sha256"] = digest
    validate_self_hash(value, "sha256")
    value["payload"] = {"x": 2}
    with pytest.raises(ValueError):
        validate_self_hash(value, "sha256")


@pytest.mark.parametrize(
    ("value", "field"),
    [
        ({"x": 1}, "sha256"),
        ({"sha256": None}, "sha256"),
        ({"sha256": "A" * 64}, "sha256"),
    ],
)
def test_self_hash_invalid_cases(value: dict[str, CanonicalValue], field: str) -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises self hash invalid cases; equivalent values must serialize
    identically while malformed representations fail explicitly.
    """
    with pytest.raises(ValueError):
        if field not in value:
            compute_self_hash(value, field)
        else:
            validate_self_hash(value, field)


def test_self_hash_argument_type_checks() -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises self hash argument type checks; equivalent values must serialize
    identically while malformed representations fail explicitly.
    """
    with pytest.raises(TypeError):
        compute_self_hash([], "sha256")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        compute_self_hash({"sha256": None}, "")


@pytest.mark.parametrize("value", ["0" * 64, "a" * 64])
def test_require_sha256_accepts(value: str) -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises require sha256 accepts; equivalent values must serialize identically
    while malformed representations fail explicitly.
    """
    assert require_sha256(value, "field") == value


@pytest.mark.parametrize("value", [None, "", "A" * 64, "g" * 64, "0" * 63])
def test_require_sha256_rejects(value: object) -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises require sha256 rejects; equivalent values must serialize identically
    while malformed representations fail explicitly.
    """
    with pytest.raises(ValueError):
        require_sha256(value, "field")


SURROGATE_CATEGORIES: tuple[Literal["Cs"], ...] = ("Cs",)

canonical_scalars = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.text(alphabet=st.characters(blacklist_categories=SURROGATE_CATEGORIES))
)
canonical_values = st.recursive(
    canonical_scalars,
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(
            st.text(alphabet=st.characters(blacklist_categories=SURROGATE_CATEGORIES), max_size=8),
            children,
            max_size=4,
        )
    ),
    max_leaves=20,
)


@given(canonical_values)
def test_property_canonicalization_idempotent(value: CanonicalValue) -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises property canonicalization idempotent; equivalent values must
    serialize identically while malformed representations fail explicitly.
    """
    encoded = canonical_json_bytes(value)
    decoded = strict_json_loads(encoded)
    assert canonical_json_bytes(decoded) == encoded


def test_golden_vectors() -> None:
    """Preserve canonical decision bytes across independent implementations.

    This scenario exercises golden vectors; equivalent values must serialize identically while
    malformed representations fail explicitly.
    """
    fixture_path = Path(__file__).parents[1] / "fixtures" / "canonical_vectors.json"
    vectors = cast(list[dict[str, Any]], json.loads(fixture_path.read_text(encoding="utf-8")))
    assert len(vectors) >= 25
    assert {vector["kind"] for vector in vectors} == {
        "binary64",
        "exact_rational",
        "canonical_json",
        "self_hash",
        "reason_ordering",
        "receipt_finalization",
    }
    for vector in vectors:
        _GOLDEN_ASSERTIONS[cast(str, vector["kind"])](vector)


def _assert_binary_vector(vector: dict[str, Any]) -> None:
    """Assert one exact binary64 representation vector."""
    value = Binary64(vector["bits"])
    assert value.classification == vector["classification"]
    assert value.to_primitive() == vector["primitive"]


def _assert_rational_vector(vector: dict[str, Any]) -> None:
    """Assert normalization and optional decimal rendering for one rational vector."""
    source = vector["input"]
    value = ExactRational(source["numerator"], source["denominator"])
    assert value.to_primitive() == vector["normalized"]
    if vector["decimal_token"] is not None:
        assert value.to_decimal_token() == vector["decimal_token"]


def _assert_canonical_json_vector(vector: dict[str, Any]) -> None:
    """Assert canonical UTF-8 bytes and SHA-256 for one semantic value."""
    value = vector["value"]
    assert canonical_json_bytes(value).decode("utf-8") == vector["canonical_utf8"]
    assert canonical_sha256(value) == vector["sha256"]


def _assert_self_hash_vector(vector: dict[str, Any]) -> None:
    """Assert self-hash computation and validation for one final object."""
    assert compute_self_hash(vector["unhashed"], vector["field"]) == vector["sha256"]
    validate_self_hash(vector["final"], vector["field"])


def _assert_reason_order_vector(vector: dict[str, Any]) -> None:
    """Assert canonical reason ordering and first-occurrence code projection."""
    reasons = tuple(ReasonRecord.from_primitive(item) for item in vector["input"])
    assert [item.to_primitive() for item in ordered_reasons(reasons)] == vector["ordered"]
    assert [code.value for code in projected_reason_codes(reasons)] == vector["reason_codes"]


def _assert_receipt_vector(vector: dict[str, Any]) -> None:
    """Assert strict receipt parsing, hash identity, and canonical serialized bytes."""
    finalized = ComparisonReceipt.from_primitive(vector["final"])
    assert validate_receipt(finalized) is finalized
    assert finalized.to_primitive() == vector["final"]
    assert finalized.receipt_sha256 == vector["receipt_sha256"]
    assert (
        canonical_json_bytes(finalized.to_primitive()).decode("utf-8") == vector["canonical_utf8"]
    )


_GOLDEN_ASSERTIONS: dict[str, Callable[[dict[str, Any]], None]] = {
    "binary64": _assert_binary_vector,
    "exact_rational": _assert_rational_vector,
    "canonical_json": _assert_canonical_json_vector,
    "self_hash": _assert_self_hash_vector,
    "reason_ordering": _assert_reason_order_vector,
    "receipt_finalization": _assert_receipt_vector,
}
