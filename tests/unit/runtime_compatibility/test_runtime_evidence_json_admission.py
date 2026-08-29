"""Runtime-evidence JSON admission must not depend on the running interpreter.

The three runtime-review evidence readers parse untrusted worker output with their own strict hooks,
so none can route through :func:`metrifid._json_admission.bounded_strict_json_loads`. Every document
here is derived from the declared limit profile its reader is bound to rather than from an
interpreter's stack allowance, so these assertions hold identically on every supported CPython.

Two failure modes are covered that a parser alone gets wrong. Escaped lone surrogates parse happily
and then have no strict UTF-8 encoding, which used to leak a raw ``UnicodeEncodeError`` through the
boundary; and a semantically valid campaign manifest is dense in nodes and sparse in bytes, so a
generic configuration node ceiling refused documents the byte ceiling was designed to allow.

This module lives under ``runtime_compatibility`` deliberately: it is exactly the cross-version
admission contract the boundary-smoke CI tier collects.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

import pytest

from metrifid._json_admission import (
    RECEIPT_JSON_LIMITS,
    JsonAdmissionError,
    JsonAdmissionLimits,
)
from metrifid.runtime_review._evidence import (
    _MANIFEST_JSON_LIMITS,
    RuntimeEvidenceAdmissionError,
    _strict_json_bytes,
    _strict_manifest_echo_json_bytes,
    _strict_manifest_json_bytes,
)

_Loader = Callable[[bytes], dict[str, object]]

_MIB = 1024 * 1024


def _admit_result(payload: bytes) -> dict[str, object]:
    """Admit one worker result exactly as evidence admission does."""
    return _strict_json_bytes(payload, "result.json")


def _admit_manifest_echo(payload: bytes) -> dict[str, object]:
    """Admit the binary64 reading of one input manifest exactly as evidence admission does."""
    return _strict_manifest_echo_json_bytes(payload)


def _admit_manifest_exact(payload: bytes) -> dict[str, object]:
    """Admit the lexical-decimal reading of one input manifest exactly as admission does."""
    return _strict_manifest_json_bytes(payload)


# Each reader is paired with the profile it is declared to enforce. Swapping any binding in the
# product makes the boundary assertions below fail, because the profiles differ in depth.
_LOADERS = (
    pytest.param(_admit_result, RECEIPT_JSON_LIMITS, id="worker_result_receipt_profile"),
    pytest.param(_admit_manifest_echo, _MANIFEST_JSON_LIMITS, id="manifest_echo_manifest_profile"),
    pytest.param(
        _admit_manifest_exact, _MANIFEST_JSON_LIMITS, id="manifest_exact_manifest_profile"
    ),
)
_LOADER_IDS = [param.id for param in _LOADERS]
_ALL_LOADERS = [param.values[0] for param in _LOADERS]
_MANIFEST_LOADERS = [_admit_manifest_echo, _admit_manifest_exact]


def _nested_list(nesting: int) -> object:
    """Return ``nesting`` nested arrays wrapping the scalar zero."""
    value: object = 0
    for _ in range(nesting):
        value = [value]
    return value


def _document_of_depth(depth: int) -> bytes:
    """Return one strict JSON object whose deepest leaf sits exactly at ``depth``.

    The object root is depth zero, so ``depth - 1`` nested arrays place the leaf at ``depth``.
    """
    if depth < 1:
        raise ValueError("depth must place the leaf below the object root")
    nesting = depth - 1
    return b'{"x":' + (b"[" * nesting) + b"0" + (b"]" * nesting) + b"}"


def _document_of_depth_beside_a_scalar(depth: int) -> bytes:
    """Return one object whose deepest leaf sits at ``depth`` beside a shallow sibling scalar.

    The deep member is written first, so the sibling scalar is the first leaf an iterative
    depth-first walk pops. A walk that stopped at its first non-container leaf would admit this
    document, so the sibling is load-bearing: do not reorder these two members.
    """
    if depth < 2:
        raise ValueError("depth must leave room for a sibling member")
    nesting = depth - 1
    return b'{"d":' + (b"[" * nesting) + b"0" + (b"]" * nesting) + b',"z":0}'


def _document_of_node_count(nodes: int) -> bytes:
    """Return one strict JSON object counting exactly ``nodes`` values and member names."""
    if nodes < 3:
        raise ValueError("a root object with one array member counts at least three nodes")
    return b'{"n":[' + b",".join(b"0" for _ in range(nodes - 3)) + b"]}"


def _document_with_member_name_of(size: int) -> bytes:
    """Return one strict JSON object whose only member name measures ``size`` UTF-8 bytes."""
    return b'{"' + (b"k" * size) + b'":0}'


# --------------------------------------------------------------------------------------------
# Declared depth policy, identical on every supported interpreter
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("admit", "limits"), _LOADERS)
def test_each_reader_admits_a_document_at_its_declared_depth_boundary(
    admit: _Loader, limits: JsonAdmissionLimits
) -> None:
    """Keep the exact declared depth admissible, so the bound is a policy and not a margin."""
    admitted = admit(_document_of_depth(limits.max_depth))

    assert admitted == {"x": _nested_list(limits.max_depth - 1)}


@pytest.mark.parametrize(("admit", "limits"), _LOADERS)
def test_each_reader_refuses_the_first_document_beyond_its_declared_depth(
    admit: _Loader, limits: JsonAdmissionLimits
) -> None:
    """Refuse one level past the bound with the stable typed refusal and a causal admission error."""
    with pytest.raises(RuntimeEvidenceAdmissionError, match="strict JSON admission") as refusal:
        admit(_document_of_depth(limits.max_depth + 1))

    assert isinstance(refusal.value.__cause__, JsonAdmissionError)


@pytest.mark.parametrize(("admit", "limits"), _LOADERS)
def test_each_reader_walks_past_a_shallow_sibling_to_the_deep_member(
    admit: _Loader, limits: JsonAdmissionLimits
) -> None:
    """Refuse an over-deep member even when a shallower leaf is reached first."""
    admit(_document_of_depth_beside_a_scalar(limits.max_depth))

    with pytest.raises(RuntimeEvidenceAdmissionError) as refusal:
        admit(_document_of_depth_beside_a_scalar(limits.max_depth + 1))

    assert isinstance(refusal.value.__cause__, JsonAdmissionError)


@pytest.mark.parametrize(("admit", "limits"), _LOADERS)
def test_the_structural_bound_runs_before_the_root_type_check(
    admit: _Loader, limits: JsonAdmissionLimits
) -> None:
    """Bound the document as parsed, so a deep non-object root is a bounds refusal, not a shape one."""
    nesting = limits.max_depth + 1
    deep_array_root = (b"[" * nesting) + b"0" + (b"]" * nesting)

    with pytest.raises(RuntimeEvidenceAdmissionError) as refusal:
        admit(deep_array_root)

    assert isinstance(refusal.value.__cause__, JsonAdmissionError)


def test_the_result_and_manifest_profiles_are_not_interchangeable() -> None:
    """Bind worker results and input manifests to different declared depths, and prove it."""
    assert _MANIFEST_JSON_LIMITS.max_depth < RECEIPT_JSON_LIMITS.max_depth
    between = _document_of_depth(_MANIFEST_JSON_LIMITS.max_depth + 1)

    assert _admit_result(between) == {"x": _nested_list(_MANIFEST_JSON_LIMITS.max_depth)}
    for admit in _MANIFEST_LOADERS:
        with pytest.raises(RuntimeEvidenceAdmissionError, match="strict JSON admission"):
            admit(between)


def test_both_manifest_readings_share_one_declared_profile() -> None:
    """One evidence member must have exactly one structural answer, whichever reading is taken."""
    at_bound = _document_of_depth(_MANIFEST_JSON_LIMITS.max_depth)
    past_bound = _document_of_depth(_MANIFEST_JSON_LIMITS.max_depth + 1)

    for admit in _MANIFEST_LOADERS:
        admit(at_bound)
        with pytest.raises(RuntimeEvidenceAdmissionError):
            admit(past_bound)


# --------------------------------------------------------------------------------------------
# Text that parses but has no strict UTF-8 encoding
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("admit", _ALL_LOADERS, ids=_LOADER_IDS)
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b'{"x":"\\ud800"}', id="lone_surrogate_value"),
        pytest.param(b'{"\\ud800":0}', id="lone_surrogate_key"),
    ],
)
def test_each_reader_refuses_unencodable_text_with_a_typed_causal_chain(
    admit: _Loader, payload: bytes
) -> None:
    """Escaped lone surrogates parse, then have no strict UTF-8 encoding.

    That used to leak a raw ``UnicodeEncodeError`` out of the admission boundary. The boundary now
    speaks one refusal vocabulary while preserving the whole causal chain.
    """
    with pytest.raises(RuntimeEvidenceAdmissionError) as refusal:
        admit(payload)

    admission = refusal.value.__cause__
    assert isinstance(admission, JsonAdmissionError)
    assert isinstance(admission.__cause__, UnicodeEncodeError)


@pytest.mark.parametrize("admit", _ALL_LOADERS, ids=_LOADER_IDS)
@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param('{"e":"\U0001f600"}'.encode(), "\U0001f600", id="literal_emoji"),
        pytest.param(b'{"e":"\\ud83d\\ude00"}', "\U0001f600", id="escaped_surrogate_pair_emoji"),
        pytest.param('{"e":"é中"}'.encode(), "é中", id="accented_and_han"),
    ],
)
def test_valid_non_ascii_text_survives_admission_unchanged(
    admit: _Loader, payload: bytes, expected: str
) -> None:
    """Refusing unencodable text must not disturb text that does have a UTF-8 encoding."""
    assert admit(payload) == {"e": expected}


# --------------------------------------------------------------------------------------------
# A valid campaign manifest is dense in nodes and sparse in bytes
# --------------------------------------------------------------------------------------------


def test_the_manifest_node_ceiling_cannot_refuse_a_document_that_fits_its_byte_ceiling() -> None:
    """Derive the manifest node ceiling from its byte ceiling, not from a generic profile.

    The densest strict JSON is an array of single-digit numbers, where each further node costs a
    digit and a separator. A document that fits the byte ceiling therefore cannot reach this node
    ceiling, so node count can never be the sole reason a valid manifest is refused.
    """
    assert _MANIFEST_JSON_LIMITS.max_nodes >= _MANIFEST_JSON_LIMITS.max_bytes // 2

    densest = _document_of_node_count(_MANIFEST_JSON_LIMITS.max_nodes)
    assert len(densest) > _MANIFEST_JSON_LIMITS.max_bytes


@pytest.mark.parametrize("admit", _MANIFEST_LOADERS, ids=["manifest_echo", "manifest_exact"])
@pytest.mark.parametrize(
    "nodes", [124_497, 1_027_207], ids=["five_fixtures", "forty_eight_fixtures"]
)
def test_both_manifest_readings_admit_a_node_dense_document_inside_the_byte_ceiling(
    admit: _Loader, nodes: int
) -> None:
    """Admit the node counts a semantically valid campaign manifest actually reaches.

    These two counts are the measured node counts of a five-fixture and a forty-eight-fixture
    manifest whose fixtures each carry 4,096 qpos and qvel values and fifty contiguous control
    segments of 256 values. Both stay inside the 4 MiB member ceiling.
    """
    document = _document_of_node_count(nodes)
    assert len(document) < _MANIFEST_JSON_LIMITS.max_bytes

    assert admit(document) == {"n": [0] * (nodes - 3)}


@pytest.mark.parametrize(("admit", "limits"), _LOADERS)
def test_each_reader_enforces_the_whole_profile_not_only_its_depth(
    admit: _Loader, limits: JsonAdmissionLimits
) -> None:
    """Reach the member-name bound through each reader, proving one shared authority.

    A hand-rolled depth-only counter substituted for the shared walk satisfies every depth case
    above, so depth coverage alone leaves a reader's single-authority binding unprotected.
    """
    admit(_document_with_member_name_of(limits.max_string_bytes))

    with pytest.raises(RuntimeEvidenceAdmissionError) as refusal:
        admit(_document_with_member_name_of(limits.max_string_bytes + 1))
    assert isinstance(refusal.value.__cause__, JsonAdmissionError)


def test_the_result_reader_still_enforces_its_node_ceiling() -> None:
    """The receipt profile keeps a node ceiling a result document can actually reach."""
    assert RECEIPT_JSON_LIMITS.max_nodes < _MANIFEST_JSON_LIMITS.max_nodes

    _admit_result(_document_of_node_count(RECEIPT_JSON_LIMITS.max_nodes))
    with pytest.raises(RuntimeEvidenceAdmissionError) as refusal:
        _admit_result(_document_of_node_count(RECEIPT_JSON_LIMITS.max_nodes + 1))
    assert isinstance(refusal.value.__cause__, JsonAdmissionError)


# --------------------------------------------------------------------------------------------
# Representations and existing strictness
# --------------------------------------------------------------------------------------------


def test_worker_results_keep_binary64_numbers() -> None:
    """Preserve the finite binary64 representation the worker-result reader exists to produce."""
    admitted = _admit_result(b'{"a": 0.25, "b": [1.5, 2]}')

    assert admitted == {"a": 0.25, "b": [1.5, 2]}
    assert type(admitted["a"]) is float


def test_the_manifest_echo_keeps_binary64_numbers() -> None:
    """The echo reading is compared against the worker's own echo, so it stays binary64."""
    admitted = _admit_manifest_echo(b'{"control_dt": 0.02}')

    assert admitted == {"control_dt": 0.02}
    assert type(admitted["control_dt"]) is float


def test_input_manifests_keep_lexical_decimals() -> None:
    """Preserve the exact lexical decimal the manifest reader exists to produce."""
    admitted = _admit_manifest_exact(b'{"control_dt": 0.020000000000000001}')

    assert admitted == {"control_dt": Decimal("0.020000000000000001")}
    assert type(admitted["control_dt"]) is Decimal


def test_the_result_reader_still_refuses_a_number_that_overflows_binary64() -> None:
    """Keep the finiteness guard that makes every admitted worker number a finite binary64.

    The lexical-decimal reader is deliberately excluded: it parses the same token to a valid finite
    ``Decimal``, which is the exact representation it exists to preserve.
    """
    with pytest.raises(RuntimeEvidenceAdmissionError):
        _admit_result(b'{"a": 1e400}')
    with pytest.raises(RuntimeEvidenceAdmissionError):
        _admit_manifest_echo(b'{"a": 1e400}')


@pytest.mark.parametrize("admit", _ALL_LOADERS, ids=_LOADER_IDS)
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b'{"a": 1, "a": 2}', id="duplicate_member"),
        pytest.param(b'{"a": {"b": 1, "b": 2}}', id="nested_duplicate_member"),
        pytest.param(b'{"a": NaN}', id="nonstandard_constant"),
        pytest.param(b'{"a": "\xff\xfe"}', id="invalid_utf8"),
        pytest.param(b"[1, 2]", id="wrong_root_type"),
    ],
)
def test_each_reader_keeps_its_existing_strictness(admit: _Loader, payload: bytes) -> None:
    """Adding bounds must not relax duplicate, constant, UTF-8, or root strictness."""
    with pytest.raises(RuntimeEvidenceAdmissionError):
        admit(payload)
