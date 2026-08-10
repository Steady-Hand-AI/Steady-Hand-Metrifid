"""Bounded strict JSON admission at the file-based trust boundary.

Every assertion here is about admission behavior a caller can observe: which documents are accepted,
which are refused, and exactly where each declared bound sits. The limits are checked at the
boundary and at boundary-plus-one so a future change to either side is caught.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from metrifid._json_admission import (
    CONFIG_JSON_LIMITS,
    RECEIPT_JSON_LIMITS,
    JsonAdmissionError,
    JsonAdmissionLimits,
    bounded_strict_json_loads,
    read_bounded_regular_file,
    read_bounded_strict_json,
)

_ROOMY = JsonAdmissionLimits(
    max_bytes=1 << 20, max_depth=64, max_nodes=100_000, max_string_bytes=1 << 16
)


def test_declared_limits_match_the_frozen_admission_policy() -> None:
    """Pin the exact byte, depth, node, and string bounds for configs and receipts."""
    assert (
        CONFIG_JSON_LIMITS.max_bytes,
        CONFIG_JSON_LIMITS.max_depth,
        CONFIG_JSON_LIMITS.max_nodes,
        CONFIG_JSON_LIMITS.max_string_bytes,
    ) == (4 * 1024 * 1024, 64, 100_000, 1024 * 1024)
    assert (
        RECEIPT_JSON_LIMITS.max_bytes,
        RECEIPT_JSON_LIMITS.max_depth,
        RECEIPT_JSON_LIMITS.max_nodes,
        RECEIPT_JSON_LIMITS.max_string_bytes,
    ) == (64 * 1024 * 1024, 128, 1_000_000, 8 * 1024 * 1024)


def test_a_valid_document_is_admitted_unchanged() -> None:
    """Admit an ordinary strict document and return its canonical value."""
    assert bounded_strict_json_loads('{"a": 1, "b": [true, null, "x"]}', _ROOMY) == {
        "a": 1,
        "b": [True, None, "x"],
    }


@pytest.mark.parametrize(
    "document",
    [
        '{"a": 1, "a": 2}',
        '{"outer": {"a": 1, "a": 2}}',
        '{"joint_tolerances": {"h": {"angle_rad": "1", "angle_rad": "2"}}}',
        '[{"a": 1, "a": 2}]',
    ],
)
def test_duplicate_member_names_refuse_at_every_level(document: str) -> None:
    """Refuse duplicate object names at the root and at nested tolerance levels."""
    with pytest.raises(JsonAdmissionError):
        bounded_strict_json_loads(document, _ROOMY)


@pytest.mark.parametrize(
    "document",
    ['{"a": 1.5}', '{"a": 1e3}', '{"a": -2.0E-4}', "[0.1]", '{"a": 1E5}'],
)
def test_raw_float_and_exponent_tokens_refuse(document: str) -> None:
    """Refuse every raw JSON decimal or exponent token."""
    with pytest.raises(JsonAdmissionError):
        bounded_strict_json_loads(document, _ROOMY)


@pytest.mark.parametrize("document", ['{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}', "[NaN]"])
def test_nonstandard_json_constants_refuse(document: str) -> None:
    """Refuse NaN, Infinity, and -Infinity."""
    with pytest.raises(JsonAdmissionError):
        bounded_strict_json_loads(document, _ROOMY)


def test_malformed_utf8_refuses() -> None:
    """Refuse bytes that are not strict UTF-8."""
    with pytest.raises(JsonAdmissionError):
        bounded_strict_json_loads(b'{"a": "\xff\xfe"}', _ROOMY)


def test_byte_limit_boundary_and_boundary_plus_one() -> None:
    """Admit a document at exactly max_bytes and refuse one byte more."""
    filler = "x" * 10
    document = '{"a":"' + filler + '"}'
    exact = len(document.encode("utf-8"))
    at_limit = JsonAdmissionLimits(
        max_bytes=exact, max_depth=8, max_nodes=100, max_string_bytes=1000
    )
    assert bounded_strict_json_loads(document, at_limit) == {"a": filler}
    one_short = JsonAdmissionLimits(
        max_bytes=exact - 1, max_depth=8, max_nodes=100, max_string_bytes=1000
    )
    with pytest.raises(JsonAdmissionError):
        bounded_strict_json_loads(document, one_short)


def test_depth_limit_boundary_and_boundary_plus_one() -> None:
    """Treat the root as depth zero and refuse one level beyond max_depth."""
    document = '{"a": {"b": {"c": 1}}}'  # deepest scalar sits at depth three
    ok = JsonAdmissionLimits(max_bytes=1000, max_depth=3, max_nodes=100, max_string_bytes=100)
    assert bounded_strict_json_loads(document, ok) == {"a": {"b": {"c": 1}}}
    too_deep = JsonAdmissionLimits(max_bytes=1000, max_depth=2, max_nodes=100, max_string_bytes=100)
    with pytest.raises(JsonAdmissionError):
        bounded_strict_json_loads(document, too_deep)


@pytest.mark.parametrize(
    ("document", "nodes"), [('{"a":1,"b":2}', 5), ("[1,2,3]", 4), ('{"a":1}', 3)]
)
def test_node_count_boundary_and_boundary_plus_one(document: str, nodes: int) -> None:
    """Count every value plus every member name, per the declared node rule."""
    ok = JsonAdmissionLimits(max_bytes=1000, max_depth=16, max_nodes=nodes, max_string_bytes=100)
    bounded_strict_json_loads(document, ok)
    too_many = JsonAdmissionLimits(
        max_bytes=1000, max_depth=16, max_nodes=nodes - 1, max_string_bytes=100
    )
    with pytest.raises(JsonAdmissionError):
        bounded_strict_json_loads(document, too_many)


def test_a_single_scalar_document_counts_as_one_node() -> None:
    """Count a bare scalar root as exactly one node."""
    ok = JsonAdmissionLimits(max_bytes=100, max_depth=4, max_nodes=1, max_string_bytes=100)
    assert bounded_strict_json_loads("1", ok) == 1


def test_string_limit_applies_to_values_and_keys() -> None:
    """Bound both string values and object member names by strict UTF-8 byte length."""
    value_doc = '{"a": "' + "y" * 8 + '"}'
    ok = JsonAdmissionLimits(max_bytes=1000, max_depth=8, max_nodes=100, max_string_bytes=8)
    bounded_strict_json_loads(value_doc, ok)
    too_long = JsonAdmissionLimits(max_bytes=1000, max_depth=8, max_nodes=100, max_string_bytes=7)
    with pytest.raises(JsonAdmissionError):
        bounded_strict_json_loads(value_doc, too_long)
    key_doc = '{"' + "k" * 8 + '": 1}'
    bounded_strict_json_loads(key_doc, ok)
    with pytest.raises(JsonAdmissionError):
        bounded_strict_json_loads(key_doc, too_long)


def test_multibyte_strings_are_measured_in_utf8_bytes() -> None:
    """Measure string bounds in encoded bytes, not characters."""
    document = '{"a": "éé"}'  # two characters, four UTF-8 bytes
    ok = JsonAdmissionLimits(max_bytes=1000, max_depth=8, max_nodes=100, max_string_bytes=4)
    bounded_strict_json_loads(document, ok)
    too_long = JsonAdmissionLimits(max_bytes=1000, max_depth=8, max_nodes=100, max_string_bytes=3)
    with pytest.raises(JsonAdmissionError):
        bounded_strict_json_loads(document, too_long)


def test_deeply_nested_document_refuses_without_crashing() -> None:
    """Refuse pathological nesting by the declared bound rather than by a crash."""
    document = "[" * 5000 + "]" * 5000
    with pytest.raises(JsonAdmissionError):
        bounded_strict_json_loads(document, CONFIG_JSON_LIMITS)


def test_regular_file_round_trip(tmp_path: Path) -> None:
    """Read and admit an ordinary regular file."""
    path = tmp_path / "config.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    assert read_bounded_strict_json(path, _ROOMY) == {"a": 1}
    assert read_bounded_regular_file(path, 1000) == b'{"a": 1}'


def test_symlink_refuses(tmp_path: Path) -> None:
    """Refuse a symbolic link rather than following it."""
    target = tmp_path / "real.json"
    target.write_text('{"a": 1}', encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(JsonAdmissionError):
        read_bounded_regular_file(link, 1000)


def test_directory_refuses(tmp_path: Path) -> None:
    """Refuse a directory as a configuration input."""
    with pytest.raises(JsonAdmissionError):
        read_bounded_regular_file(tmp_path, 1000)


def test_fifo_refuses(tmp_path: Path) -> None:
    """Refuse a FIFO without blocking on it."""
    fifo = tmp_path / "pipe.json"
    os.mkfifo(fifo)
    with pytest.raises(JsonAdmissionError):
        read_bounded_regular_file(fifo, 1000)


def test_oversized_file_refuses_before_parsing(tmp_path: Path) -> None:
    """Refuse a file past the byte ceiling without attempting to parse it."""
    path = tmp_path / "big.json"
    path.write_bytes(b"[" + b"1," * 4000 + b"1]")
    with pytest.raises(JsonAdmissionError):
        read_bounded_regular_file(path, 64)


def test_missing_file_refuses(tmp_path: Path) -> None:
    """Refuse an absent path with the admission error type."""
    with pytest.raises(JsonAdmissionError):
        read_bounded_regular_file(tmp_path / "absent.json", 1000)
