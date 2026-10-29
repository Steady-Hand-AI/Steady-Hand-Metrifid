"""Behavioral admission boundaries for the native-free controller-repair request."""

from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import cast

import pytest

from metrifid.repair._request import (
    RepairController,
    RepairInputError,
    RepairLimits,
    RepairRequest,
    RepairTask,
    _decode_request_document,
    load_request,
    validate_request,
    validate_selected_parameters,
)

_Key = str | int


def _document() -> dict[str, object]:
    """Build an explicit nonhistorical request using ordinary decimal inputs."""
    return {
        "schema": "metrifid.repair_request",
        "version": 1,
        "family": "planar_push_timed15_v1",
        "task": {"rail_y_lower_m": -0.06, "target_xy_m": [0.08, 0.06], "target_yaw_rad": 1.5},
        "controller": {"parameters_normalized": [0.5] * 15},
    }


def _write(tmp_path: Path, document: object) -> Path:
    """Write only this test's input without involving a native implementation."""
    path = tmp_path / "request.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _set(document: dict[str, object], path: tuple[_Key, ...], value: object) -> None:
    """Set one explicitly addressed fixture field, including array elements."""
    parent: object = document
    for key in path[:-1]:
        if isinstance(key, int):
            parent = cast(list[object], parent)[key]
        else:
            parent = cast(dict[str, object], parent)[key]
    last = path[-1]
    if isinstance(last, int):
        cast(list[object], parent)[last] = value
    else:
        cast(dict[str, object], parent)[last] = value


def _base_request() -> RepairRequest:
    """Construct the same supported request directly through immutable types."""
    return RepairRequest(RepairTask(-0.06, (0.08, 0.06), 1.5), RepairController((0.5,) * 15))


def test_decimal_values_are_normalized_and_all_nested_data_are_immutable(tmp_path: Path) -> None:
    """Load real decimal tokens into immutable, independent binary64 data."""
    document = _document()
    _set(document, ("task", "target_yaw_rad"), 0)
    request = load_request(_write(tmp_path, document))
    assert request.schema == "metrifid.repair_request"
    assert request.version == 1
    assert request.family == "planar_push_timed15_v1"
    assert request.task == RepairTask(-0.06, (0.08, 0.06), 0.0)
    assert type(request.task.target_yaw_rad) is float
    assert type(request.task.target_xy_m) is tuple
    assert request.controller.parameters_normalized == (0.5,) * 15
    assert all(type(value) is float for value in request.controller.parameters_normalized)
    assert request.limits == RepairLimits(128, 96_000, 120.0)
    for instance, name, value in (
        (request.task, "rail_y_lower_m", -0.08),
        (request.controller, "parameters_normalized", (0.4,) * 15),
        (request.limits, "max_attempts", 2),
        (request, "family", "other"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(instance, name, value)


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        ({}, RepairLimits(128, 96_000, 120.0)),
        ({"max_attempts": 2}, RepairLimits(2, 96_000, 120.0)),
        ({"max_integration_steps": 12_200}, RepairLimits(128, 12_200, 120.0)),
        ({"max_cpu_seconds": 0.25}, RepairLimits(128, 96_000, 0.25)),
        ({"max_attempts": 1_000, "max_cpu_seconds": 3}, RepairLimits(1_000, 96_000, 3.0)),
    ],
)
def test_limits_default_independently(
    tmp_path: Path, limits: dict[str, object], expected: RepairLimits
) -> None:
    """Treat absent members as defaults without treating null as omission."""
    document = _document()
    document["limits"] = limits
    assert load_request(_write(tmp_path, document)).limits == expected


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema",), "other"),
        (("schema",), None),
        (("version",), True),
        (("version",), 1.0),
        (("version",), "1"),
        (("version",), 2),
        (("family",), "unsupported"),
        (("family",), None),
        (("task",), None),
        (("task",), []),
        (("controller",), None),
        (("controller",), []),
        (("limits",), None),
        (("limits",), []),
        (("task", "rail_y_lower_m"), True),
        (("task", "rail_y_lower_m"), "-0.06"),
        (("task", "rail_y_lower_m"), None),
        (("task", "target_xy_m"), None),
        (("task", "target_xy_m"), "0.08,0.06"),
        (("task", "target_xy_m"), [0.08]),
        (("task", "target_xy_m"), [0.08, 0.06, 0]),
        (("task", "target_xy_m", 0), True),
        (("task", "target_xy_m", 1), "0.06"),
        (("task", "target_yaw_rad"), False),
        (("task", "target_yaw_rad"), "1.5"),
        (("task", "target_yaw_rad"), None),
        (("controller", "parameters_normalized"), None),
        (("controller", "parameters_normalized"), [0.5] * 14),
        (("controller", "parameters_normalized"), [0.5] * 16),
        (("controller", "parameters_normalized", 0), True),
        (("controller", "parameters_normalized", 1), "0.5"),
        (("controller", "parameters_normalized", 2), None),
    ],
)
def test_schema_types_and_required_value_shapes_refuse(
    tmp_path: Path, path: tuple[_Key, ...], value: object
) -> None:
    """Refuse unsupported schemas and coercions at the public file boundary."""
    document = _document()
    _set(document, path, value)
    with pytest.raises(RepairInputError) as caught:
        load_request(_write(tmp_path, document))
    assert caught.value.reason == "INVALID_REQUEST"


@pytest.mark.parametrize("root", [None, True, 1, 1.0, [], "request"])
def test_nonobject_roots_refuse(tmp_path: Path, root: object) -> None:
    """Require the request object rather than inferring a schema from another type."""
    with pytest.raises(RepairInputError):
        load_request(_write(tmp_path, root))


@pytest.mark.parametrize(
    ("level", "member"),
    [
        (None, "schema"),
        (None, "version"),
        (None, "family"),
        (None, "task"),
        (None, "controller"),
        ("task", "rail_y_lower_m"),
        ("task", "target_xy_m"),
        ("task", "target_yaw_rad"),
        ("controller", "parameters_normalized"),
    ],
)
def test_missing_required_members_refuse(tmp_path: Path, level: str | None, member: str) -> None:
    """Require every named task and controller value without hidden fallbacks."""
    document = _document()
    parent = document if level is None else cast(dict[str, object], document[level])
    del parent[member]
    with pytest.raises(RepairInputError):
        load_request(_write(tmp_path, document))


@pytest.mark.parametrize("level", [None, "task", "controller", "limits"])
def test_unknown_members_refuse_at_each_schema_level(tmp_path: Path, level: str | None) -> None:
    """Refuse every additional key, including those whose value would be harmless."""
    document = _document()
    document["limits"] = {}
    parent = document if level is None else cast(dict[str, object], document[level])
    parent["additional"] = 0
    with pytest.raises(RepairInputError):
        load_request(_write(tmp_path, document))


@pytest.mark.parametrize("member", ["max_attempts", "max_integration_steps", "max_cpu_seconds"])
@pytest.mark.parametrize("value", [None, True, "128"])
def test_limit_type_coercions_refuse(tmp_path: Path, member: str, value: object) -> None:
    """Refuse null, booleans and numeric strings in each resource limit."""
    document = _document()
    document["limits"] = {member: value}
    with pytest.raises(RepairInputError):
        load_request(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("member", "value"),
    [
        ("max_attempts", 1),
        ("max_attempts", 1_001),
        ("max_attempts", 2.0),
        ("max_integration_steps", 12_199),
        ("max_integration_steps", 12_200.0),
        ("max_integration_steps", 0),
        ("max_cpu_seconds", 0),
        ("max_cpu_seconds", -1),
    ],
)
def test_limit_boundaries_and_integer_tokens_refuse(
    tmp_path: Path, member: str, value: object
) -> None:
    """Require the derived minimum work budget and exact integer count tokens."""
    document = _document()
    document["limits"] = {member: value}
    with pytest.raises(RepairInputError):
        load_request(_write(tmp_path, document))


def test_any_finite_positive_cpu_limit_is_admitted(tmp_path: Path) -> None:
    """Do not invent a minimum positive CPU allowance beyond the request contract."""
    document = _document()
    minimum_positive = math.nextafter(0.0, math.inf)
    document["limits"] = {"max_cpu_seconds": minimum_positive}
    assert load_request(_write(tmp_path, document)).limits.max_cpu_seconds == minimum_positive


@pytest.mark.parametrize("rail", [-0.18, -0.04])
def test_rail_endpoints_are_inclusive(tmp_path: Path, rail: float) -> None:
    """Admit the original lower bound and the fixed initial pusher position."""
    document = _document()
    _set(document, ("task", "rail_y_lower_m"), rail)
    assert load_request(_write(tmp_path, document)).task.rail_y_lower_m == rail


@pytest.mark.parametrize(
    "rail", [math.nextafter(-0.18, -math.inf), math.nextafter(-0.04, math.inf)]
)
def test_rail_just_outside_either_endpoint_refuses(tmp_path: Path, rail: float) -> None:
    """Refuse a single binary64 step outside each supported rail endpoint."""
    document = _document()
    _set(document, ("task", "rail_y_lower_m"), rail)
    with pytest.raises(RepairInputError):
        load_request(_write(tmp_path, document))


@pytest.mark.parametrize("yaw", [-math.pi, math.pi])
def test_yaw_endpoints_are_inclusive_without_wrapping(tmp_path: Path, yaw: float) -> None:
    """Preserve either supported yaw endpoint exactly."""
    document = _document()
    _set(document, ("task", "target_yaw_rad"), yaw)
    assert load_request(_write(tmp_path, document)).task.target_yaw_rad == yaw


@pytest.mark.parametrize(
    "yaw", [math.nextafter(-math.pi, -math.inf), math.nextafter(math.pi, math.inf)]
)
def test_yaw_outside_interval_refuses_instead_of_wrapping(tmp_path: Path, yaw: float) -> None:
    """Reject a one-ULP angular overflow rather than silently changing the target."""
    document = _document()
    _set(document, ("task", "target_yaw_rad"), yaw)
    with pytest.raises(RepairInputError):
        load_request(_write(tmp_path, document))


@pytest.mark.parametrize("value", [0, 1, 1e-9, 1 - 1e-9, -0.01, 1.01])
def test_starting_controller_requires_strict_interior(tmp_path: Path, value: float) -> None:
    """Reject closed-box boundaries and both exact local-chart margins at load time."""
    document = _document()
    _set(document, ("controller", "parameters_normalized", 0), value)
    with pytest.raises(RepairInputError):
        load_request(_write(tmp_path, document))


@pytest.mark.parametrize(
    "value", [math.nextafter(1e-9, math.inf), math.nextafter(1 - 1e-9, -math.inf)]
)
def test_starting_controller_just_inside_either_margin_is_admitted(
    tmp_path: Path, value: float
) -> None:
    """Admit each adjacent representable interior starting coordinate unchanged."""
    document = _document()
    _set(document, ("controller", "parameters_normalized", 0), value)
    assert load_request(_write(tmp_path, document)).controller.parameters_normalized[0] == value


@pytest.mark.parametrize(
    ("target", "yaw"),
    [((0.16, 0.0), 0.0), ((0.0, 0.18), 0.0), ((0.18, 0.0), math.pi / 2)],
)
def test_actual_footprint_admits_contact_with_support_boundary(
    tmp_path: Path, target: tuple[float, float], yaw: float
) -> None:
    """Use the cuboid dimensions and orientation, not a uniform guessed margin."""
    document = _document()
    _set(document, ("task", "target_xy_m"), list(target))
    _set(document, ("task", "target_yaw_rad"), yaw)
    assert load_request(_write(tmp_path, document)).task.target_xy_m == target


@pytest.mark.parametrize(
    ("target", "yaw"),
    [((0.17, 0.0), 0.0), ((0.16, 0.0), math.pi / 4), ((0.0, -0.19), 0.0)],
)
def test_rotated_footprint_and_center_bounds_refuse_overhang(
    tmp_path: Path, target: tuple[float, float], yaw: float
) -> None:
    """Reject an overhanging rotated corner even when the center is in bounds."""
    document = _document()
    _set(document, ("task", "target_xy_m"), list(target))
    _set(document, ("task", "target_yaw_rad"), yaw)
    with pytest.raises(RepairInputError):
        load_request(_write(tmp_path, document))


@pytest.mark.parametrize("level", [None, "task", "controller", "limits"])
def test_duplicate_keys_are_refused_even_with_identical_values(
    tmp_path: Path, level: str | None
) -> None:
    """Exercise the object-pairs decoder at the root and each nested schema object."""
    document = _document()
    document["limits"] = {"max_attempts": 2}
    key = {
        None: "version",
        "task": "rail_y_lower_m",
        "controller": "parameters_normalized",
        "limits": "max_attempts",
    }[level]
    parent = document if level is None else cast(dict[str, object], document[level])
    member = json.dumps(key) + ": " + json.dumps(parent[key])
    text = json.dumps(document).replace(member, member + ", " + member, 1)
    path = tmp_path / "duplicate.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(RepairInputError, match="Duplicate"):
        load_request(path)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e309", "-1e309"])
def test_nonfinite_json_tokens_and_decimal_overflow_refuse(tmp_path: Path, token: str) -> None:
    """Do not normalize nonfinite constants or overflowing exponents into usable input."""
    document = _document()
    _set(document, ("task", "target_yaw_rad"), "TOKEN")
    path = tmp_path / "nonfinite.json"
    path.write_text(json.dumps(document).replace('"TOKEN"', token), encoding="utf-8")
    with pytest.raises(RepairInputError) as caught:
        load_request(path)
    assert caught.value.reason == "INVALID_REQUEST"


def test_integer_overflow_in_a_binary64_field_refuses(tmp_path: Path) -> None:
    """Integer tokens used as physical numbers must also fit finite binary64."""
    document = _document()
    _set(document, ("task", "target_yaw_rad"), 10**309)
    with pytest.raises(RepairInputError):
        load_request(_write(tmp_path, document))


def test_integer_budget_is_not_constrained_by_float_or_digit_limits(tmp_path: Path) -> None:
    """Keep actual count fields as integers within the overall bounded JSON document."""
    text = json.dumps(_document())
    text = text[:-1] + ', "limits": {"max_integration_steps": 1' + "0" * 4_301 + "}}"
    path = tmp_path / "integer_budget.json"
    path.write_text(text, encoding="utf-8")
    request = load_request(path)
    assert request.limits.max_integration_steps == 10**4_301


def test_exact_byte_ceiling_counts_whitespace(tmp_path: Path) -> None:
    """Admit exactly 65,536 bytes and refuse one extra byte before parsing."""
    content = json.dumps(_document()).encode("utf-8")
    at_limit = content + b" " * (65_536 - len(content))
    path = tmp_path / "request.json"
    path.write_bytes(at_limit)
    assert load_request(path) == _base_request()
    path.write_bytes(at_limit + b"\n")
    with pytest.raises(RepairInputError) as caught:
        load_request(path)
    assert caught.value.reason == "INVALID_REQUEST"


@pytest.mark.parametrize("prefix", [b"\xef\xbb\xbf", b"\xff"])
def test_bom_and_invalid_utf8_refuse(tmp_path: Path, prefix: bytes) -> None:
    """Require strict UTF-8 without a byte-order mark."""
    path = tmp_path / "request.json"
    path.write_bytes(prefix + json.dumps(_document()).encode("utf-8"))
    with pytest.raises(RepairInputError):
        load_request(path)


@pytest.mark.parametrize("payload", [b"", b"{", b"{} trailing", b'"\\ud800"'])
def test_malformed_json_and_unencodable_strings_refuse(tmp_path: Path, payload: bytes) -> None:
    """Translate parser and string-encoding failures into repair-specific invalid input."""
    path = tmp_path / "request.json"
    path.write_bytes(payload)
    with pytest.raises(RepairInputError) as caught:
        load_request(path)
    assert caught.value.reason == "INVALID_REQUEST"


def test_decoder_enforces_exact_depth_node_and_utf8_string_bounds() -> None:
    """Exercise the shared structural walk through the actual decimal decoder."""
    assert _decode_request_document(b"[" * 8 + b"0" + b"]" * 8) is not None
    assert len(cast(list[object], _decode_request_document(json.dumps([0] * 127).encode()))) == 127
    assert _decode_request_document(json.dumps("é" * 64).encode()) == "é" * 64
    assert _decode_request_document(json.dumps({"x" * 128: 0}).encode()) == {"x" * 128: 0}
    for payload in (
        b"[" * 9 + b"0" + b"]" * 9,
        json.dumps([0] * 128).encode(),
        json.dumps("é" * 64 + "x").encode(),
        json.dumps({"x" * 129: 0}).encode(),
        b"[" * 5_000 + b"0" + b"]" * 5_000,
    ):
        with pytest.raises(RepairInputError):
            _decode_request_document(payload)


def test_reader_os_failure_is_distinct_from_bad_content(tmp_path: Path) -> None:
    """A missing file retains the reader failure reason; a directory is invalid input."""
    with pytest.raises(RepairInputError) as missing:
        load_request(tmp_path / "absent.json")
    assert missing.value.reason == "INPUT_READ_FAILED"
    with pytest.raises(RepairInputError) as directory:
        load_request(tmp_path)
    assert directory.value.reason == "INVALID_REQUEST"


def test_final_symlink_is_never_followed(tmp_path: Path) -> None:
    """Use the existing regular-file reader rather than a convenience read_text path."""
    target = _write(tmp_path, _document())
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(RepairInputError):
        load_request(link)


def test_execution_readmission_copies_and_revalidates_nested_values() -> None:
    """A validated immutable copy is returned without trusting the original object's history."""
    request = _base_request()
    admitted = validate_request(request)
    assert admitted == request
    assert admitted is not request
    assert admitted.task is not request.task
    assert admitted.controller is not request.controller
    assert admitted.limits is not request.limits
    object.__setattr__(request.controller, "parameters_normalized", (0.0,) * 15)
    with pytest.raises(RepairInputError):
        validate_request(request)
    assert admitted.controller.parameters_normalized == (0.5,) * 15


@pytest.mark.parametrize("field_name", [field.name for field in fields(RepairRequest)])
def test_execution_readmission_refuses_missing_slots(field_name: str) -> None:
    """A manually assembled object cannot bypass required typed fields."""
    request = _base_request()
    object.__delattr__(request, field_name)
    with pytest.raises(RepairInputError):
        validate_request(request)


@pytest.mark.parametrize(
    ("nested", "attribute", "value"),
    [
        ("task", "rail_y_lower_m", -0.2),
        ("task", "target_xy_m", (0.18, 0.18)),
        ("task", "target_yaw_rad", math.nan),
        ("limits", "max_attempts", True),
        ("limits", "max_integration_steps", 12_199),
        ("limits", "max_cpu_seconds", math.inf),
    ],
)
def test_execution_readmission_rejects_tampered_nested_fields(
    nested: str, attribute: str, value: object
) -> None:
    """Frozen dataclasses are not an authentication substitute at execution boundaries."""
    request = _base_request()
    object.__setattr__(getattr(request, nested), attribute, value)
    with pytest.raises(RepairInputError):
        validate_request(request)


@pytest.mark.parametrize("candidate", [None, {}, object(), "request"])
def test_execution_readmission_refuses_unrecognized_objects(candidate: object) -> None:
    """Require the supported immutable request type before native setup."""
    with pytest.raises(RepairInputError):
        validate_request(candidate)


def test_selected_controller_uses_closed_bounds_and_copies_inputs() -> None:
    """Retained boundary coordinates are admissible for replay, unlike starting requests."""
    values = [0, 1] + [0.5] * 13
    admitted = validate_selected_parameters(values)
    assert admitted == (0.0, 1.0, *((0.5,) * 13))
    assert type(admitted) is tuple
    values[0] = 0.3
    assert admitted[0] == 0.0
    with pytest.raises(RepairInputError):
        RepairController(admitted)


@pytest.mark.parametrize(
    "value",
    [math.nextafter(0.0, -math.inf), math.nextafter(1.0, math.inf), math.nan, math.inf, True, "0"],
)
def test_selected_controller_rejects_invalid_or_outside_values(value: object) -> None:
    """Do not clip, coerce or wrap a retained selected controller."""
    values: list[object] = [0.5] * 15
    values[0] = value
    with pytest.raises(RepairInputError):
        validate_selected_parameters(values)


def test_public_import_and_loading_never_attempt_native_imports(tmp_path: Path) -> None:
    """Load through the SDK in an isolated subprocess whose native imports are forbidden."""
    path = _write(tmp_path, _document())
    package = importlib.import_module("metrifid")
    assert package.__file__ is not None
    package_parent = Path(package.__file__).resolve().parent.parent
    script = """
import importlib.abc
import sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'mujoco'}:
            raise AssertionError('native import attempted: ' + fullname)
sys.meta_path.insert(0, NoNative())
sys.path.insert(0, sys.argv[1])
from metrifid.repair import load_request, repair_controller, replay_controller
request = load_request(sys.argv[2])
assert request.controller.parameters_normalized == (0.5,) * 15
assert request.task.rail_y_lower_m == -0.06
assert callable(repair_controller) and callable(replay_controller)
assert not any(name.split('.')[0] in {'numpy', 'mujoco'} for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(package_parent), str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
