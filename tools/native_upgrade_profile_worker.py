"""Run one bounded native-upgrade fixture cell and retain replayable evidence."""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

import mujoco  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

_MANIFEST_SCHEMA: Final = "metrifid.native_upgrade_manifest"
_RESULT_SCHEMA: Final = "metrifid.native_upgrade_worker_result"
_TRACE_SCHEMA: Final = "metrifid.native_upgrade_trace"
_CANONICAL_TRACE_SCHEMA: Final = "metrifid.native_upgrade_canonical_trace"
_SENTINEL_SCHEMA: Final = "metrifid.runtime_review.integration_state_sentinel"
_SCHEMA_VERSION: Final = 1
_PRODUCTION_SCHEMA_VERSION: Final = 2
_MAX_MANIFEST_BYTES: Final = 1_048_576
_MAX_XML_BYTES: Final = 1_048_576
_FIXED_STEP_DTS: Final = (Decimal("0.004"), Decimal("0.002"), Decimal("0.001"))
_CONTROL_DT: Final = Decimal("0.02")
_HORIZON: Final = Decimal("1.00")
_THREAD_ENVIRONMENT: Final = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_CHANNEL_KINDS: Final = {
    "JOINT_POSITION",
    "JOINT_VELOCITY",
    "BODY_POSITION",
    "BODY_QUATERNION",
    "SENSOR",
}
_ID_PATTERN: Final = re.compile(r"[a-z][a-z0-9_.-]{0,95}\Z")
_DECIMAL_PATTERN: Final = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_STABLE_PACKAGE_PATTERN: Final = re.compile(
    r"\A(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)"
    r"(?:\.post(?:0|[1-9][0-9]*))?"
    r"(?:\+[0-9A-Za-z]+(?:[._-][0-9A-Za-z]+)*)?\Z"
)
_STABLE_NATIVE_PATTERN: Final = re.compile(
    r"\A(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)\Z"
)
_MINIMUM_MUJOCO_VERSION: Final = (3, 9, 0)
_PROFILE_ROLES: Final = ("baseline", "candidate")
_CALLBACK_ACCESSORS: Final = (
    "get_mjcb_control",
    "get_mjcb_sensor",
    "get_mjcb_passive",
    "get_mjcb_act_dyn",
    "get_mjcb_act_gain",
    "get_mjcb_act_bias",
    "get_mjcb_contactfilter",
    "get_mjcb_time",
    "get_mju_user_warning",
    "get_mju_user_malloc",
    "get_mju_user_free",
)
_PRODUCTION_CALLABLE_CAPABILITIES: Final = (
    "mj_versionString",
    "mj_version",
    "MjModel",
    "MjModel.from_xml_string",
    "MjData",
    "mj_forward",
    "mj_step",
    "mj_sizeModel",
    "mj_saveModel",
    "mj_name2id",
    "mj_id2name",
    "mj_contactForce",
    "mj_stateSize",
    "mj_getState",
    "mj_setState",
)
_PRODUCTION_ATTRIBUTE_CAPABILITIES: Final = (
    "mjtState.mjSTATE_INTEGRATION",
    "mjtObj.mjOBJ_BODY",
    "mjtObj.mjOBJ_GEOM",
    "mjtObj.mjOBJ_JOINT",
    "mjtObj.mjOBJ_SENSOR",
    "mjtJoint.mjJNT_HINGE",
    "mjtJoint.mjJNT_SLIDE",
)
_SENTINEL_LIMITATIONS: Final = (
    "EXACT_PROFILE_FIXTURE_AND_COARSEST_TIMESTEP_ONLY",
    "PUBLIC_MJSTATE_INTEGRATION_ONLY",
    "EXACT_BINARY64_EQUALITY_NO_TOLERANCE",
    "NO_CROSS_PROFILE_EQUIVALENCE_CLAIM",
    "SENTINEL_PROVES_SAME_PROFILE_RESTORE_PIPELINE_ONLY",
)
_LIMITATIONS: Final = [
    "ONE_EXACT_SELF_CONTAINED_MJCF_CLOSURE_ONLY",
    "ONE_EXACT_INITIAL_STATE_AND_ACTION_PROGRAM_ONLY",
    "EXACT_NATIVE_CPU_RUNTIME_PROFILES_ONLY",
    "NO_UNIVERSAL_MUJOCO_VERSION_EQUIVALENCE_CLAIM",
    "NO_POLICY_OR_HARDWARE_SAFETY_CLAIM",
    "NO_TASK_SUCCESS_OR_REAL_WORLD_TRANSFER_CLAIM",
    "NO_BACKEND_PARITY_CLAIM",
    "WORKER_EMITS_EVIDENCE_ONLY_NO_MIGRATION_DECISION",
]


class WorkerRefusal(RuntimeError):
    """Represent a bounded input, runtime, or execution refusal."""


@dataclass(frozen=True)
class ControlSegment:
    """Hold one admitted left-boundary zero-order-hold control segment."""

    start: Decimal
    end: Decimal
    values: tuple[Decimal, ...]


@dataclass(frozen=True)
class ChannelSpec:
    """Hold one admitted scalar or quaternion observation declaration."""

    channel_id: str
    kind: str
    object_name: str
    component: int | None
    scale: Decimal | None


@dataclass(frozen=True)
class ContactPairSpec:
    """Hold one admitted semantic geom-pair observation declaration."""

    channel_id: str
    geom_names: tuple[str, str]
    force_scale: Decimal
    impulse_scale: Decimal


@dataclass(frozen=True)
class FixtureSpec:
    """Hold one fully admitted native-upgrade fixture declaration."""

    fixture_id: str
    campaign_role: str
    baseline_fixture_id: str
    xml_path: str
    initial_qpos: tuple[Decimal, ...]
    initial_qvel: tuple[Decimal, ...]
    controls: tuple[ControlSegment, ...]
    control_dt: Decimal
    horizon: Decimal
    step_dts: tuple[Decimal, ...]
    continuous_tolerance: Decimal
    so3_tolerance: Decimal
    contact_event_time_tolerance: Decimal
    minimum_common_prefix: Decimal
    channels: tuple[ChannelSpec, ...]
    contact_pairs: tuple[ContactPairSpec, ...]
    echo: dict[str, Any]


@dataclass
class ContactState:
    """Accumulate one declared contact pair without using manifold order semantically."""

    spec: ContactPairSpec
    occupied: bool
    onset: Decimal | None
    events: list[dict[str, Any]]
    segments: list[dict[str, Any]]
    persistence_steps: int
    aggregate_normal_impulse: float
    current_normal_force: float


@dataclass
class SimulationEvidence:
    """Hold normalized arrays and semantic execution evidence for one run."""

    observation_times: NDArray[np.float64]
    arrays: dict[str, NDArray[Any]]
    descriptors: list[dict[str, Any]]
    contacts: list[dict[str, Any]]
    diagnostics: dict[str, Any]


class OwnedOutput:
    """Create one new output directory and write each owned member exactly once."""

    def __init__(self, path: Path) -> None:
        """Reserve an absolute new output directory without following a parent symlink."""
        if not path.is_absolute():
            raise WorkerRefusal("--output must be an absolute path")
        parent = path.parent
        if not parent.is_dir() or parent.is_symlink():
            raise WorkerRefusal("--output parent must be an existing nonsymlink directory")
        try:
            path.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise WorkerRefusal("refusing to overwrite existing output path") from exc
        self.path = path
        _fsync_directory(parent)

    def write_bytes(self, name: str, payload: bytes) -> None:
        """Write one bounded bare member with exclusive creation and fsync."""
        if PurePosixPath(name).name != name or name in {"", ".", ".."}:
            raise WorkerRefusal("owned member name is invalid")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.path / name, flags, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)

    def write_json(self, name: str, value: Any) -> None:
        """Write one canonical JSON document followed by one line feed."""
        self.write_bytes(name, _canonical_json_bytes(value) + b"\n")

    def write_npz(self, arrays: dict[str, NDArray[Any]]) -> None:
        """Write one data-only NPZ container without permitting overwrite."""
        target = self.path / "trace.npz"
        with target.open("xb") as stream:
            np.savez(stream, **cast(dict[str, Any], arrays))
            stream.flush()
            os.fsync(stream.fileno())

    def write_checksums(self) -> None:
        """Bind every already-written regular member with a sorted checksum manifest."""
        members = sorted(path for path in self.path.iterdir() if path.name != "CHECKSUMS.sha256")
        lines: list[str] = []
        for member in members:
            if member.is_symlink() or not member.is_file():
                raise WorkerRefusal("owned output contains a non-regular member")
            lines.append(f"{_file_sha256(member)}  {member.name}\n")
        self.write_bytes("CHECKSUMS.sha256", "".join(lines).encode("ascii"))
        _fsync_directory(self.path)


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value using the campaign's exact canonical representation."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    """Hash one canonical JSON value."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    """Hash one regular nonsymlink file with bounded-memory reads."""
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise WorkerRefusal(f"not a regular nonsymlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry update before treating evidence as complete."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_bounded(path: Path, maximum: int, label: str) -> bytes:
    """Read one regular nonsymlink input with an explicit byte limit."""
    if path.is_symlink() or not path.is_file():
        raise WorkerRefusal(f"{label} must be a regular nonsymlink file")
    size = path.stat().st_size
    if size > maximum:
        raise WorkerRefusal(f"{label} exceeds the byte limit")
    payload = path.read_bytes()
    if len(payload) != size:
        raise WorkerRefusal(f"{label} changed while it was read")
    return payload


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate member names."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WorkerRefusal(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    """Reject JSON parser extensions for NaN and infinities."""
    raise WorkerRefusal(f"non-finite JSON number is forbidden: {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    """Decode one strict UTF-8 JSON object with duplicate-member rejection."""
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerRefusal("manifest is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise WorkerRefusal("manifest root must be an object")
    return cast(dict[str, Any], value)


def _strict_manifest_json(payload: bytes) -> dict[str, Any]:
    """Decode manifest floats from their lexical tokens without binary64 rounding."""
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_no_duplicate_object,
            parse_float=Decimal,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, InvalidOperation) as exc:
        raise WorkerRefusal("manifest is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise WorkerRefusal("manifest root must be an object")
    return cast(dict[str, Any], value)


def _object(value: Any, context: str) -> dict[str, Any]:
    """Require an admitted JSON object value."""
    if type(value) is not dict:
        raise WorkerRefusal(f"{context} must be an object")
    return cast(dict[str, Any], value)


def _array(value: Any, context: str, *, maximum: int = 4096) -> list[Any]:
    """Require a bounded admitted JSON array value."""
    if type(value) is not list:
        raise WorkerRefusal(f"{context} must be an array")
    result = value
    if len(result) > maximum:
        raise WorkerRefusal(f"{context} exceeds its member limit")
    return result


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    """Require exactly the frozen keys for one manifest object."""
    actual = set(value)
    if actual != expected:
        raise WorkerRefusal(
            f"{context} members differ: missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )


def _string(value: Any, context: str) -> str:
    """Require a nonempty bounded JSON string."""
    if type(value) is not str or not value or len(value) > 256:
        raise WorkerRefusal(f"{context} must be a nonempty bounded string")
    return value


def _identifier(value: Any, context: str) -> str:
    """Require a lowercase bounded semantic identifier."""
    result = _string(value, context)
    if _ID_PATTERN.fullmatch(result) is None:
        raise WorkerRefusal(f"{context} is not a valid identifier")
    return result


def _number(value: Any, context: str) -> Decimal:
    """Require a finite JSON number, explicitly excluding booleans and strings."""
    if type(value) not in {Decimal, int, float}:
        raise WorkerRefusal(f"{context} must be a JSON number and not a boolean or string")
    if type(value) is float and not math.isfinite(value):
        raise WorkerRefusal(f"{context} must be finite")
    try:
        result = value if type(value) is Decimal else Decimal(str(value))
    except InvalidOperation as exc:
        raise WorkerRefusal(f"{context} is not a decimal number") from exc
    if not result.is_finite():
        raise WorkerRefusal(f"{context} must be finite")
    return result


def _integer(value: Any, context: str, minimum: int, maximum: int) -> int:
    """Require one bounded JSON integer, explicitly excluding booleans."""
    if type(value) is not int or not minimum <= value <= maximum:
        raise WorkerRefusal(f"{context} must be an integer in [{minimum}, {maximum}]")
    return value


def _decimal_token(value: Decimal) -> str:
    """Return a canonical non-exponent decimal token for evidence metadata."""
    if value == 0:
        return "0"
    token = format(value, "f")
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    if _DECIMAL_PATTERN.fullmatch(token) is None:
        raise WorkerRefusal("decimal token could not be canonicalized")
    return token


def _float_token(value: float) -> str | None:
    """Return an exact round-trippable finite float token, preserving signed zero."""
    if not math.isfinite(value):
        return None
    if value == 0.0:
        return "-0.0" if math.copysign(1.0, value) < 0.0 else "0"
    return repr(float(value))


def _binary64_arrays_equal(left: object, right: object) -> bool:
    """Compare finite binary64 arrays by shape and exact little-endian bytes."""
    left_array = np.ascontiguousarray(left, dtype="<f8")
    right_array = np.ascontiguousarray(right, dtype="<f8")
    return left_array.shape == right_array.shape and left_array.tobytes(
        order="C"
    ) == right_array.tobytes(order="C")


def _stable_version_triplet(value: object, *, package: bool) -> tuple[int, int, int] | None:
    """Parse one bounded stable package token or exact native triplet."""
    if type(value) is not str or len(value) > 128:
        return None
    pattern = _STABLE_PACKAGE_PATTERN if package else _STABLE_NATIVE_PATTERN
    match = pattern.fullmatch(value)
    if match is None:
        return None
    triplet = tuple(int(match.group(name)) for name in ("major", "minor", "patch"))
    if any(component >= 1000 for component in triplet):
        return None
    return cast(tuple[int, int, int], triplet)


def _native_version_integer(triplet: tuple[int, int, int]) -> int:
    """Encode one bounded native version triplet without a release allowlist."""
    major, minor, patch = triplet
    return major * 1_000_000 + minor * 1_000 + patch


def _resolve_capability(name: str) -> object | None:
    """Resolve one dotted MuJoCo capability without binding optional members at import."""
    value: object = mujoco
    for segment in name.split("."):
        try:
            value = getattr(value, segment)
        except (AttributeError, RuntimeError):
            return None
    return value


def _require_production_capabilities() -> None:
    """Require every public native capability used by production worker execution."""
    missing = [
        name
        for name in _PRODUCTION_CALLABLE_CAPABILITIES
        if not callable(_resolve_capability(name))
    ]
    for name in _PRODUCTION_ATTRIBUTE_CAPABILITIES:
        value = _resolve_capability(name)
        try:
            measured = int(value) if value is not None else 0
        except (TypeError, ValueError):
            measured = 0
        if measured <= 0:
            missing.append(name)
    missing.extend(name for name in _CALLBACK_ACCESSORS if not callable(_resolve_capability(name)))
    if missing:
        raise WorkerRefusal(f"worker runtime lacks required MuJoCo capabilities: {sorted(missing)}")


def _require_callback_authority_absent() -> None:
    """Refuse every process-wide callback or allocator authority before deterministic stepping."""
    active: list[str] = []
    for name in _CALLBACK_ACCESSORS:
        accessor = _resolve_capability(name)
        if not callable(accessor):
            raise WorkerRefusal(f"worker runtime lacks callback accessor: {name}")
        try:
            value = accessor()
        except Exception as exc:
            raise WorkerRefusal(f"worker callback authority could not be measured: {name}") from exc
        if value is not None:
            active.append(name)
    if active:
        raise WorkerRefusal(f"worker callback authority is active: {active}")


def _divides(numerator: Decimal, denominator: Decimal) -> bool:
    """Return whether one positive decimal interval divides another exactly."""
    if denominator <= 0:
        return False
    return numerator % denominator == 0


def _parse_control(value: Any, index: int) -> ControlSegment:
    """Admit one bounded control segment."""
    obj = _object(value, f"controls[{index}]")
    _exact_keys(obj, {"start", "end", "values"}, f"controls[{index}]")
    start = _number(obj["start"], f"controls[{index}].start")
    end = _number(obj["end"], f"controls[{index}].end")
    values = tuple(
        _number(item, f"controls[{index}].values[{position}]")
        for position, item in enumerate(
            _array(obj["values"], f"controls[{index}].values", maximum=256)
        )
    )
    if not values or any(abs(item) > Decimal("1000000") for item in values):
        raise WorkerRefusal("control values must be nonempty and bounded")
    if not Decimal("0") <= start < end <= _HORIZON:
        raise WorkerRefusal("control segment bounds are invalid")
    return ControlSegment(start, end, values)


def _parse_channel(value: Any, index: int) -> ChannelSpec:
    """Admit one scalar or quaternion observation channel."""
    obj = _object(value, f"channels[{index}]")
    _exact_keys(
        obj,
        {"channel_id", "kind", "object_name", "component", "scale"},
        f"channels[{index}]",
    )
    channel_id = _identifier(obj["channel_id"], f"channels[{index}].channel_id")
    kind = _string(obj["kind"], f"channels[{index}].kind")
    if kind not in _CHANNEL_KINDS:
        raise WorkerRefusal(f"channels[{index}].kind is unsupported")
    object_name = _string(obj["object_name"], f"channels[{index}].object_name")
    component_value = obj["component"]
    component = (
        None
        if component_value is None
        else _integer(component_value, f"channels[{index}].component", 0, 255)
    )
    scale_value = obj["scale"]
    scale = None if scale_value is None else _number(scale_value, f"channels[{index}].scale")
    if kind == "BODY_QUATERNION":
        if component is not None or scale is not None:
            raise WorkerRefusal("quaternion channels require null component and scale")
    elif scale is None or scale <= 0:
        raise WorkerRefusal("continuous scalar channels require a positive physical scale")
    if kind in {"JOINT_POSITION", "JOINT_VELOCITY"} and component is not None:
        raise WorkerRefusal("joint scalar channels require a null component")
    if kind == "BODY_POSITION" and component not in {0, 1, 2}:
        raise WorkerRefusal("body-position channels require component 0, 1, or 2")
    if kind == "SENSOR" and component is None:
        raise WorkerRefusal("sensor channels require a component")
    return ChannelSpec(channel_id, kind, object_name, component, scale)


def _parse_contact_pair(value: Any, index: int) -> ContactPairSpec:
    """Admit one semantic contact-pair declaration."""
    obj = _object(value, f"contact_pairs[{index}]")
    _exact_keys(
        obj,
        {"channel_id", "geom_a", "geom_b", "force_scale", "impulse_scale"},
        f"contact_pairs[{index}]",
    )
    channel_id = _identifier(obj["channel_id"], f"contact_pairs[{index}].channel_id")
    names = tuple(
        sorted(
            (
                _string(obj["geom_a"], f"contact_pairs[{index}].geom_a"),
                _string(obj["geom_b"], f"contact_pairs[{index}].geom_b"),
            )
        )
    )
    if names[0] == names[1]:
        raise WorkerRefusal("contact-pair geom names must be distinct")
    force_scale = _number(obj["force_scale"], f"contact_pairs[{index}].force_scale")
    impulse_scale = _number(obj["impulse_scale"], f"contact_pairs[{index}].impulse_scale")
    if force_scale <= 0 or impulse_scale <= 0:
        raise WorkerRefusal("contact force and impulse scales must be positive")
    return ContactPairSpec(channel_id, cast(tuple[str, str], names), force_scale, impulse_scale)


def _parse_fixture(value: Any, echo_value: Any, index: int) -> FixtureSpec:
    """Admit one complete fixture declaration and every exact physical invariant."""
    obj = _object(value, f"fixtures[{index}]")
    echo = _object(echo_value, f"manifest_echo.fixtures[{index}]")
    expected = {
        "fixture_id",
        "campaign_role",
        "baseline_fixture_id",
        "xml_path",
        "initial_qpos",
        "initial_qvel",
        "controls",
        "control_dt",
        "horizon",
        "step_dts",
        "continuous_tolerance",
        "so3_tolerance",
        "contact_event_time_tolerance",
        "minimum_common_prefix",
        "channels",
        "contact_pairs",
    }
    _exact_keys(obj, expected, f"fixtures[{index}]")
    fixture_id = _identifier(obj["fixture_id"], f"fixtures[{index}].fixture_id")
    role = _string(obj["campaign_role"], f"fixtures[{index}].campaign_role")
    if role not in {"MIGRATION_SUBJECT", "CONTROLLED_MUTATION"}:
        raise WorkerRefusal("campaign_role is unsupported")
    baseline_id = _identifier(obj["baseline_fixture_id"], f"fixtures[{index}].baseline_fixture_id")
    if role == "MIGRATION_SUBJECT" and baseline_id != fixture_id:
        raise WorkerRefusal("migration subjects must identify themselves as their baseline")
    if role == "CONTROLLED_MUTATION" and baseline_id == fixture_id:
        raise WorkerRefusal("controlled mutations require a distinct baseline fixture ID")
    xml_path = _safe_relative_xml_path(obj["xml_path"], f"fixtures[{index}].xml_path")
    initial_qpos = _number_array(obj["initial_qpos"], f"fixtures[{index}].initial_qpos")
    initial_qvel = _number_array(obj["initial_qvel"], f"fixtures[{index}].initial_qvel")
    controls = tuple(
        _parse_control(item, position)
        for position, item in enumerate(
            _array(obj["controls"], f"fixtures[{index}].controls", maximum=256)
        )
    )
    channels = tuple(
        _parse_channel(item, position)
        for position, item in enumerate(
            _array(obj["channels"], f"fixtures[{index}].channels", maximum=256)
        )
    )
    contact_pairs = tuple(
        _parse_contact_pair(item, position)
        for position, item in enumerate(
            _array(obj["contact_pairs"], f"fixtures[{index}].contact_pairs", maximum=64)
        )
    )
    spec = FixtureSpec(
        fixture_id=fixture_id,
        campaign_role=role,
        baseline_fixture_id=baseline_id,
        xml_path=xml_path,
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        controls=controls,
        control_dt=_number(obj["control_dt"], f"fixtures[{index}].control_dt"),
        horizon=_number(obj["horizon"], f"fixtures[{index}].horizon"),
        step_dts=tuple(
            _number(item, f"fixtures[{index}].step_dts[{position}]")
            for position, item in enumerate(
                _array(obj["step_dts"], f"fixtures[{index}].step_dts", maximum=8)
            )
        ),
        continuous_tolerance=_number(
            obj["continuous_tolerance"], f"fixtures[{index}].continuous_tolerance"
        ),
        so3_tolerance=_number(obj["so3_tolerance"], f"fixtures[{index}].so3_tolerance"),
        contact_event_time_tolerance=_number(
            obj["contact_event_time_tolerance"],
            f"fixtures[{index}].contact_event_time_tolerance",
        ),
        minimum_common_prefix=_number(
            obj["minimum_common_prefix"], f"fixtures[{index}].minimum_common_prefix"
        ),
        channels=channels,
        contact_pairs=contact_pairs,
        echo=echo,
    )
    _validate_fixture_invariants(spec)
    return spec


def _safe_relative_xml_path(value: Any, context: str) -> str:
    """Admit one normalized relative POSIX XML member path without traversal."""
    token = _string(value, context)
    path = PurePosixPath(token)
    if (
        path.is_absolute()
        or path.as_posix() != token
        or path.suffix != ".xml"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise WorkerRefusal(f"{context} must be a normalized confined relative XML path")
    return token


def _number_array(value: Any, context: str) -> tuple[Decimal, ...]:
    """Admit one nonempty bounded vector of finite JSON numbers."""
    items = _array(value, context, maximum=4096)
    result = tuple(_number(item, f"{context}[{index}]") for index, item in enumerate(items))
    if not result or any(abs(item) > Decimal("1000000") for item in result):
        raise WorkerRefusal(f"{context} must be nonempty and bounded")
    return result


def _validate_fixture_invariants(spec: FixtureSpec) -> None:
    """Enforce fixed campaign grids, timing, uniqueness, and bounded actions."""
    _validate_fixed_physical_contract(spec)
    _validate_control_grid(spec)
    _validate_channel_uniqueness(spec)


def _validate_fixed_physical_contract(spec: FixtureSpec) -> None:
    """Enforce the exact fixed timing and tolerance declaration."""
    if spec.control_dt != _CONTROL_DT or spec.horizon != _HORIZON:
        raise WorkerRefusal("control_dt and horizon must be exactly 0.02 and 1.00 seconds")
    if spec.step_dts != _FIXED_STEP_DTS:
        raise WorkerRefusal("step_dts must be exactly [0.004, 0.002, 0.001]")
    if spec.continuous_tolerance != Decimal("0.000001"):
        raise WorkerRefusal("continuous_tolerance must be exactly 0.000001")
    if spec.so3_tolerance != Decimal("0.0000001"):
        raise WorkerRefusal("so3_tolerance must be exactly 0.0000001")
    if spec.contact_event_time_tolerance != Decimal("0.02"):
        raise WorkerRefusal("contact_event_time_tolerance must be exactly 0.02")
    if spec.minimum_common_prefix != Decimal("0.50"):
        raise WorkerRefusal("minimum_common_prefix must be exactly 0.50")


def _validate_control_grid(spec: FixtureSpec) -> None:
    """Enforce exact grid divisibility and contiguous bounded action segments."""
    if not _divides(spec.horizon, spec.control_dt):
        raise WorkerRefusal("control grid must divide the horizon")
    if any(
        not _divides(spec.control_dt, step_dt) or not _divides(spec.horizon, step_dt)
        for step_dt in spec.step_dts
    ):
        raise WorkerRefusal("every step grid must divide control_dt and horizon")
    if not spec.controls or spec.controls[0].start != 0 or spec.controls[-1].end != spec.horizon:
        raise WorkerRefusal("control segments must cover the full horizon")
    for index, segment in enumerate(spec.controls):
        if index and spec.controls[index - 1].end != segment.start:
            raise WorkerRefusal("control segments must be contiguous and ordered")
        boundaries = (segment.start, segment.end)
        if any(not _divides(boundary, spec.control_dt) for boundary in boundaries):
            raise WorkerRefusal("every control boundary must land on the control grid")
        if any(
            not _divides(boundary, step_dt) for boundary in boundaries for step_dt in spec.step_dts
        ):
            raise WorkerRefusal("every control boundary must land on every simulation grid")
    lengths = {len(segment.values) for segment in spec.controls}
    if len(lengths) != 1:
        raise WorkerRefusal("every control segment must have the same actuator width")


def _validate_channel_uniqueness(spec: FixtureSpec) -> None:
    """Enforce unique projected IDs and unique sorted semantic geom pairs."""
    channel_ids = [item.channel_id for item in spec.channels]
    for pair in spec.contact_pairs:
        channel_ids.extend(
            (
                f"{pair.channel_id}.occupancy",
                f"{pair.channel_id}.normal_force",
                f"{pair.channel_id}.normal_impulse",
            )
        )
    if len(channel_ids) != len(set(channel_ids)):
        raise WorkerRefusal("channel IDs must be unique, including derived contact channels")
    pairs = [item.geom_names for item in spec.contact_pairs]
    if len(pairs) != len(set(pairs)):
        raise WorkerRefusal("declared semantic contact pairs must be unique")


def _load_manifest(path: Path, fixture_id: str) -> tuple[bytes, dict[str, Any], FixtureSpec]:
    """Load the strict manifest and select exactly one unique fixture."""
    raw = _read_bounded(path, _MAX_MANIFEST_BYTES, "manifest")
    document = _strict_manifest_json(raw)
    echo = _strict_json(raw)
    _exact_keys(document, {"schema", "schema_version", "fixtures"}, "manifest")
    if document["schema"] != _MANIFEST_SCHEMA or document["schema_version"] != _SCHEMA_VERSION:
        raise WorkerRefusal("manifest schema or version is unsupported")
    echo_fixtures = _array(echo.get("fixtures"), "manifest_echo.fixtures", maximum=64)
    admitted_fixtures = _array(document["fixtures"], "manifest.fixtures", maximum=64)
    if len(echo_fixtures) != len(admitted_fixtures):
        raise WorkerRefusal("manifest echo fixture count differs")
    fixtures = tuple(
        _parse_fixture(item, echo_fixtures[index], index)
        for index, item in enumerate(admitted_fixtures)
    )
    ids = [item.fixture_id for item in fixtures]
    if not fixtures or len(ids) != len(set(ids)):
        raise WorkerRefusal("fixture IDs must be nonempty and unique")
    selected = [item for item in fixtures if item.fixture_id == fixture_id]
    if len(selected) != 1:
        raise WorkerRefusal("--fixture-id must select exactly one manifest fixture")
    return raw, echo, selected[0]


def _resolve_fixture_source(manifest_path: Path, relative: str) -> tuple[Path, bytes]:
    """Resolve one XML member beneath the manifest root without following symlinks."""
    root = manifest_path.parent.resolve(strict=True)
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise WorkerRefusal("fixture source path must not contain symlinks")
    try:
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise WorkerRefusal("fixture source escapes the manifest directory") from exc
    raw = _read_bounded(resolved, _MAX_XML_BYTES, "fixture XML")
    return resolved, raw


def _admit_self_contained_xml(raw: bytes) -> str:
    """Reject external resources, includes, plugins, DTDs, and malformed XML."""
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise WorkerRefusal("fixture XML must be strict UTF-8") from exc
    upper = text.upper()
    if "<!DOCTYPE" in upper or "<!ENTITY" in upper:
        raise WorkerRefusal("fixture XML must not declare a DTD or entity")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise WorkerRefusal("fixture XML is malformed") from exc
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in {"include", "plugin"}:
            raise WorkerRefusal("fixture XML includes and plugins are forbidden")
        forbidden = {"file", "meshdir", "texturedir", "assetdir"}.intersection(element.attrib)
        if forbidden:
            raise WorkerRefusal(
                "fixture XML must not reference external files or asset directories"
            )
    return text


def _distribution_payload(name: str) -> dict[str, Any]:
    """Hash every installed RECORD member and retain both literal and wheel-stable projections."""
    distribution = importlib.metadata.distribution(name)
    members: list[dict[str, Any]] = []
    record_bound: list[dict[str, Any]] = []
    declared_count = 0
    unhashed_count = 0
    files = distribution.files
    if files is None:
        raise WorkerRefusal(f"distribution {name!r} has no RECORD member list")
    for member in sorted(files, key=str):
        logical_path = str(member)
        located = Path(str(distribution.locate_file(member)))
        if located.is_symlink() or not located.is_file():
            raise WorkerRefusal(f"distribution member is absent or symlinked: {logical_path}")
        sha256 = _file_sha256(located)
        size = located.stat().st_size
        record_hash = member.hash
        mode = record_hash.mode if record_hash is not None else None
        value = record_hash.value if record_hash is not None else None
        if mode == "sha256":
            actual = base64.urlsafe_b64encode(bytes.fromhex(sha256)).rstrip(b"=").decode("ascii")
            if actual != value:
                raise WorkerRefusal(f"distribution member differs from RECORD: {logical_path}")
            declared_count += 1
        else:
            unhashed_count += 1
        projection = {"logical_path": logical_path, "sha256": sha256, "size_bytes": size}
        members.append({**projection, "record_hash_mode": mode, "record_hash_value": value})
        if mode == "sha256" and not logical_path.startswith("../"):
            record_bound.append(projection)
    identity_projection = [
        {
            "logical_path": item["logical_path"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in members
    ]
    return {
        "name": distribution.metadata["Name"],
        "version": distribution.version,
        "member_count": len(members),
        "payload_identity_algorithm": (
            "sha256(canonical-json([{logical_path,sha256,size_bytes},...]))"
        ),
        "payload_sha256": _canonical_sha256(identity_projection),
        "record_bound_identity_algorithm": (
            "sha256(canonical-json(RECORD sha256 members excluding path-escaping installer "
            "entry points))"
        ),
        "record_bound_member_count": len(record_bound),
        "record_bound_payload_sha256": _canonical_sha256(record_bound),
        "record_declared_sha256_member_count": declared_count,
        "record_unhashed_member_count": unhashed_count,
        "members": members,
    }


def _native_library_identity() -> dict[str, Any]:
    """Identify and hash the single packaged library matching the active native triplet."""
    package_file = getattr(mujoco, "__file__", None)
    if type(package_file) is not str:
        raise WorkerRefusal("MuJoCo package location is unavailable")
    package_root = Path(package_file).resolve(strict=True).parent
    system = platform.system()
    pattern = {
        "Darwin": "libmujoco.*.dylib",
        "Linux": "libmujoco.so*",
    }.get(system)
    if pattern is None:
        raise WorkerRefusal("native MuJoCo library identity is unsupported on this OS")
    native_version = str(mujoco.mj_versionString())
    candidates = [
        path
        for path in package_root.rglob(pattern)
        if path.is_file() and not path.is_symlink() and native_version in path.name
    ]
    if len(candidates) != 1:
        raise WorkerRefusal("exactly one version-matching native MuJoCo library is required")
    selected = candidates[0].resolve(strict=True)
    loaded = _loaded_mujoco_libraries()
    if loaded != [selected]:
        raise WorkerRefusal("loaded native MuJoCo library does not match the packaged candidate")
    return {
        "filename": selected.name,
        "loaded_path": str(loaded[0]),
        "resolved_path": str(selected),
        "size_bytes": selected.stat().st_size,
        "sha256": _file_sha256(selected),
    }


def _loaded_mujoco_libraries() -> list[Path]:
    """Enumerate the actual versioned MuJoCo native images loaded by this process."""
    if platform.system() == "Darwin":
        system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        image_count = system._dyld_image_count
        image_count.argtypes = []
        image_count.restype = ctypes.c_uint32
        image_name = system._dyld_get_image_name
        image_name.argtypes = [ctypes.c_uint32]
        image_name.restype = ctypes.c_char_p
        paths = []
        for index in range(image_count()):
            raw = image_name(index)
            if raw:
                path = Path(os.fsdecode(raw))
                if path.name.startswith("libmujoco.") and path.suffix == ".dylib":
                    paths.append(path.resolve(strict=True))
        return sorted(set(paths))
    maps = Path("/proc/self/maps")
    if not maps.is_file() or maps.is_symlink():
        raise WorkerRefusal("loaded native library enumeration is unavailable")
    paths = []
    for line in maps.read_text(encoding="utf-8", errors="strict").splitlines():
        candidate = line.rsplit(maxsplit=1)[-1]
        if "/" in candidate and Path(candidate).name.startswith("libmujoco.so"):
            paths.append(Path(candidate).resolve(strict=True))
    return sorted(set(paths))


def _pip_check() -> dict[str, Any]:
    """Run the installed-requirement consistency gate without network access."""
    environment = os.environ.copy()
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    environment["PIP_NO_CACHE_DIR"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=60,
    )
    result = {
        "argv": [sys.executable, "-m", "pip", "check"],
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise WorkerRefusal("pip check failed in the native worker profile")
    return result


def _hardware_profile() -> dict[str, str]:
    """Measure the exact bounded system-profiler CPU and hardware projection."""
    completed = subprocess.run(
        ["/usr/sbin/system_profiler", "SPHardwareDataType"],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise WorkerRefusal("live system_profiler hardware measurement failed")
    allowed = {
        "Model Name",
        "Model Identifier",
        "Processor Name",
        "Processor Speed",
        "Number of Processors",
        "Total Number of Cores",
        "Hyper-Threading Technology",
    }
    result: dict[str, str] = {}
    for raw_line in completed.stdout.splitlines():
        key, separator, value = raw_line.strip().partition(":")
        if separator and key in allowed:
            result[key] = value.strip()
    required = {
        "Model Identifier",
        "Processor Name",
        "Total Number of Cores",
        "Hyper-Threading Technology",
    }
    if not required.issubset(result):
        raise WorkerRefusal("live system_profiler projection is incomplete")
    return result


def _live_host_identity() -> dict[str, Any]:
    """Measure the live OS, architecture, CPU, and bounded hardware projection."""
    host: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "architecture": list(platform.architecture()),
        "libc": list(platform.libc_ver()),
        "logical_cpu_count": os.cpu_count(),
    }
    if platform.system() != "Darwin":
        host.update(
            {
                "cpu_model": platform.processor(),
                "cpu_model_source": "platform.processor",
                "hardware_model": platform.machine(),
                "hardware_profile": {},
                "physical_cpu_count": None,
                "hyper_threading_technology": None,
            }
        )
        return host
    hardware_profile = _hardware_profile()
    try:
        physical_count = int(hardware_profile["Total Number of Cores"])
    except ValueError as exc:
        raise WorkerRefusal("live system_profiler physical CPU count is invalid") from exc
    host.update(
        {
            "cpu_model": hardware_profile["Processor Name"],
            "cpu_model_source": "live system_profiler Processor Name",
            "hardware_model": hardware_profile["Model Identifier"],
            "hardware_profile": hardware_profile,
            "physical_cpu_count": physical_count,
            "hyper_threading_technology": hardware_profile["Hyper-Threading Technology"],
        }
    )
    return host


def _admit_external_host(
    document: dict[str, Any], live_host: dict[str, Any], *, require_libc: bool = False
) -> None:
    """Require the retained profile host projection to equal a fresh live measurement."""
    external_host = _object(document.get("host"), "external.host")
    if require_libc:
        if external_host != live_host:
            raise WorkerRefusal("external profile host does not match live OS/architecture/CPU")
        return
    shared_keys = {
        "system",
        "release",
        "version",
        "platform",
        "machine",
        "architecture",
        "logical_cpu_count",
        "hardware_model",
        "hardware_profile",
        "physical_cpu_count",
    }
    if any(external_host.get(key) != live_host.get(key) for key in shared_keys):
        raise WorkerRefusal("external profile host does not match live OS/architecture/CPU")


def _live_python_identity(*, include_build: bool) -> dict[str, Any]:
    """Measure the exact live CPython executable and build projection."""
    identity: dict[str, Any] = {
        "executable": sys.executable,
        "resolved_executable": str(Path(sys.executable).resolve(strict=True)),
        "resolved_executable_sha256": _file_sha256(Path(sys.executable).resolve(strict=True)),
        "version": platform.python_version(),
        "version_full": sys.version,
        "implementation": platform.python_implementation(),
        "implementation_name": sys.implementation.name,
        "compiler": platform.python_compiler(),
        "cache_tag": sys.implementation.cache_tag,
    }
    if include_build:
        identity["build"] = list(platform.python_build())
    return identity


def _legacy_profile_contract() -> tuple[str, str]:
    """Map the active exact native version to its immutable historical profile."""
    version = str(mujoco.__version__)
    expected = {"3.10.0": "A_3.10.0", "3.11.0": "B_3.11.0"}
    profile_id = expected.get(version)
    if profile_id is None or str(mujoco.mj_versionString()) != version:
        raise WorkerRefusal("worker requires exact native MuJoCo 3.10.0 or 3.11.0")
    if str(np.__version__) != "2.3.5":
        raise WorkerRefusal("worker requires exact NumPy 2.3.5")
    observed = {key: os.environ.get(key) for key in _THREAD_ENVIRONMENT}
    if observed != _THREAD_ENVIRONMENT:
        raise WorkerRefusal("required deterministic thread environment is not exact")
    return profile_id, version


def _production_profile_contract(profile_role: str) -> tuple[str, str, int]:
    """Admit one semantic role and coherent stable MuJoCo package/native identity."""
    if profile_role not in _PROFILE_ROLES:
        raise WorkerRefusal("profile role must be baseline or candidate")
    _require_production_capabilities()
    package_version = getattr(mujoco, "__version__", None)
    package_triplet = _stable_version_triplet(package_version, package=True)
    if package_triplet is None or package_triplet < _MINIMUM_MUJOCO_VERSION:
        raise WorkerRefusal("worker requires a stable MuJoCo package at or above 3.9.0")
    native_version = mujoco.mj_versionString()
    native_integer = mujoco.mj_version()
    native_triplet = _stable_version_triplet(native_version, package=False)
    if (
        native_triplet is None
        or native_triplet != package_triplet
        or type(native_integer) is not int
        or native_integer != _native_version_integer(native_triplet)
    ):
        raise WorkerRefusal(
            "worker requires coherent MuJoCo package, native string, and native integer identity"
        )
    observed = {key: os.environ.get(key) for key in _THREAD_ENVIRONMENT}
    if observed != _THREAD_ENVIRONMENT:
        raise WorkerRefusal("required deterministic thread environment is not exact")
    return cast(str, package_version), cast(str, native_version), native_integer


def _external_profile_identity(
    profile_token: str,
    mujoco_distribution: dict[str, Any],
    numpy_distribution: dict[str, Any],
    native_library: dict[str, Any],
    live_host: dict[str, Any],
    live_python: dict[str, Any],
    *,
    schema_version: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Bind an optional separately retained profile identity to live worker measurements."""
    token = os.environ.get("METRIFID_NATIVE_UPGRADE_PROFILE_IDENTITY")
    if token is None:
        return {"available": False, "raw_sha256": None, "profile_identity_sha256": None}, None
    path = Path(token)
    raw = _read_bounded(path, 4 * 1024 * 1024, "external profile identity")
    document = _strict_json(raw)
    if schema_version == 1:
        if document.get("schema_version") != 1 or document.get("profile_id") != profile_token:
            raise WorkerRefusal("external profile identity has the wrong legacy profile ID")
    elif (
        document.get("schema_version") != _PRODUCTION_SCHEMA_VERSION
        or document.get("profile_role") != profile_token
    ):
        raise WorkerRefusal("external profile identity has the wrong profile role")
    projection = dict(document)
    claimed_self_hash = projection.pop("profile_identity_sha256", None)
    if type(claimed_self_hash) is not str or _canonical_sha256(projection) != claimed_self_hash:
        raise WorkerRefusal("external profile identity self-hash is invalid")
    _admit_external_host(
        document, live_host, require_libc=schema_version == _PRODUCTION_SCHEMA_VERSION
    )
    external_python = _object(document.get("python"), "external.python")
    expected_python = (
        {**live_python, "build": list(platform.python_build())}
        if schema_version == _PRODUCTION_SCHEMA_VERSION
        else live_python
    )
    if any(external_python.get(key) != value for key, value in expected_python.items()):
        raise WorkerRefusal("external profile Python identity does not match the live interpreter")
    try:
        external_mujoco = _object(document["mujoco"], "external.mujoco")
        external_numpy = _object(document["numpy"], "external.numpy")
        external_native = _object(
            external_mujoco["loaded_native_library"], "external.mujoco.loaded_native_library"
        )
        external_mujoco_distribution = _object(
            external_mujoco["distribution"], "external.mujoco.distribution"
        )
        external_numpy_distribution = _object(
            external_numpy["distribution"], "external.numpy.distribution"
        )
    except KeyError as exc:
        raise WorkerRefusal("external profile identity is incomplete") from exc
    checks = (
        external_mujoco_distribution.get("payload_sha256") == mujoco_distribution["payload_sha256"],
        external_mujoco_distribution.get("record_bound_payload_sha256")
        == mujoco_distribution["record_bound_payload_sha256"],
        external_numpy_distribution.get("payload_sha256") == numpy_distribution["payload_sha256"],
        external_numpy_distribution.get("record_bound_payload_sha256")
        == numpy_distribution["record_bound_payload_sha256"],
        external_native.get("sha256") == native_library["sha256"],
        document.get("environment") == _THREAD_ENVIRONMENT,
    )
    if not all(checks):
        raise WorkerRefusal("external profile identity does not match live worker measurements")
    if schema_version == _PRODUCTION_SCHEMA_VERSION:
        package_version, native_version, native_integer = _production_profile_contract(
            profile_token
        )
        sentinel = _object(document.get("sentinel"), "external.sentinel")
        contract = _object(document.get("profile_contract"), "external.profile_contract")
        if (
            document.get("package_version") != package_version
            or document.get("native_version") != native_version
            or document.get("native_version_integer") != native_integer
            or document.get("support_tier") != "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE"
            or external_mujoco.get("package_version") != package_version
            or external_mujoco.get("native_version") != native_version
            or external_mujoco.get("native_version_integer") != native_integer
            or sentinel.get("profile_role") != profile_token
            or sentinel.get("status") != "PASS"
            or contract.get("worker_sha256") != _file_sha256(Path(__file__))
        ):
            raise WorkerRefusal("external production profile identity is not admissible")
        sentinel_hash = sentinel.get("sentinel_identity_sha256")
        sentinel_projection = dict(sentinel)
        sentinel_projection.pop("sentinel_identity_sha256", None)
        if (
            type(sentinel_hash) is not str
            or _canonical_sha256(sentinel_projection) != sentinel_hash
        ):
            raise WorkerRefusal("external sentinel self-hash is invalid")
    return {
        "available": True,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "profile_identity_sha256": claimed_self_hash,
    }, document


def _runtime_identity(
    profile_role: str | None = None, *, allow_unbound_profile: bool = False
) -> tuple[str, str, dict[str, Any]]:
    """Measure and self-hash one legacy or semantic-role runtime profile."""
    if profile_role is None:
        profile_token, package_version = _legacy_profile_contract()
        native_version = package_version
        native_integer = int(mujoco.mj_version())
        schema_version = 1
    else:
        package_version, native_version, native_integer = _production_profile_contract(profile_role)
        profile_token = profile_role
        schema_version = _PRODUCTION_SCHEMA_VERSION
    mujoco_distribution = _distribution_payload("mujoco")
    numpy_distribution = _distribution_payload("numpy")
    native_library = _native_library_identity()
    host = _live_host_identity()
    python_identity = _live_python_identity(include_build=False)
    external_link, external = _external_profile_identity(
        profile_token,
        mujoco_distribution,
        numpy_distribution,
        native_library,
        host,
        python_identity,
        schema_version=schema_version,
    )
    installation = (
        external.get("installation")
        if external is not None
        else {
            "available": False,
            "install_command": None,
            "artifacts": [],
        }
    )
    runtime: dict[str, Any] = {
        "schema": "metrifid.native_upgrade_runtime_identity",
        "schema_version": schema_version,
        "python": python_identity,
        "host": host,
        "thread_environment": dict(_THREAD_ENVIRONMENT),
        "mujoco": {
            "distribution": mujoco_distribution,
            "loaded_native_library": native_library,
        },
        "numpy": {"python_version": str(np.__version__), "distribution": numpy_distribution},
        "installation": installation,
        "pip_check": _pip_check(),
        "external_profile_identity": external_link,
    }
    if schema_version == 1:
        runtime["profile_id"] = profile_token
        runtime["profile_version"] = package_version
        cast(dict[str, Any], runtime["host"]).pop("libc", None)
        cast(dict[str, Any], runtime["mujoco"]).update(
            {
                "python_version": package_version,
                "native_version": native_version,
                "version_integer": native_integer,
            }
        )
    else:
        if external is None and not allow_unbound_profile:
            raise WorkerRefusal("production worker requires a retained profile identity")
        runtime.update(
            {
                "profile_role": profile_token,
                "package_version": package_version,
                "native_version": native_version,
                "native_version_integer": native_integer,
                "support_tier": "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE",
            }
        )
        cast(dict[str, Any], runtime["mujoco"]).update(
            {
                "package_version": package_version,
                "native_version": native_version,
                "native_version_integer": native_integer,
            }
        )
        if external is not None:
            _require_callback_authority_absent()
            sentinel = _object(external.get("sentinel"), "external.sentinel")
            sentinel_identity_sha256 = sentinel.get("sentinel_identity_sha256")
            profile_identity_sha256 = external.get("profile_identity_sha256")
            contract = _object(external.get("profile_contract"), "external.profile_contract")
            worker_sha256 = contract.get("worker_sha256")
            if any(
                type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in (sentinel_identity_sha256, profile_identity_sha256, worker_sha256)
            ):
                raise WorkerRefusal("external production identity lacks canonical binding hashes")
            runtime.update(
                {
                    "worker_sha256": worker_sha256,
                    "profile_identity_sha256": profile_identity_sha256,
                    "sentinel_identity_sha256": sentinel_identity_sha256,
                }
            )
    runtime["runtime_identity_sha256"] = _canonical_sha256(runtime)
    return profile_token, package_version, runtime


def _source_identity(spec: FixtureSpec, raw: bytes) -> dict[str, Any]:
    """Build a self-hashed one-member self-contained source closure identity."""
    member = {
        "path": spec.xml_path,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    closure: dict[str, Any] = {
        "schema": "metrifid.native_upgrade_source_closure",
        "schema_version": _SCHEMA_VERSION,
        "entrypoint": spec.xml_path,
        "member_count": 1,
        "members": [member],
    }
    closure["closure_sha256"] = _canonical_sha256(closure)
    return closure


def _workload_identity(spec: FixtureSpec) -> dict[str, Any]:
    """Retain and self-hash exact initial-state and action-program semantics."""
    initial: dict[str, Any] = {
        "schema": "metrifid.native_upgrade_initial_state",
        "schema_version": _SCHEMA_VERSION,
        "qpos": [_decimal_token(value) for value in spec.initial_qpos],
        "qvel": [_decimal_token(value) for value in spec.initial_qvel],
    }
    initial["semantic_sha256"] = _canonical_sha256(initial)
    action: dict[str, Any] = {
        "schema": "metrifid.native_upgrade_action_program",
        "schema_version": _SCHEMA_VERSION,
        "action_semantics": "LEFT_BOUNDARY_ZERO_ORDER_HOLD",
        "control_dt": _decimal_token(spec.control_dt),
        "horizon": _decimal_token(spec.horizon),
        "segments": [
            {
                "start": _decimal_token(segment.start),
                "end": _decimal_token(segment.end),
                "values": [_decimal_token(value) for value in segment.values],
            }
            for segment in spec.controls
        ],
    }
    action["semantic_sha256"] = _canonical_sha256(action)
    workload: dict[str, Any] = {
        "schema": "metrifid.native_upgrade_workload_identity",
        "schema_version": _SCHEMA_VERSION,
        "initial_state": initial,
        "action_program": action,
    }
    workload["semantic_sha256"] = _canonical_sha256(workload)
    return workload


def _compile_model(xml_text: str, spec: FixtureSpec, step_dt: Decimal) -> mujoco.MjModel:
    """Compile one self-contained model and bind every declared model-side reference."""
    try:
        model = mujoco.MjModel.from_xml_string(xml_text)
    except Exception as exc:
        raise WorkerRefusal("MuJoCo refused the self-contained fixture XML") from exc
    model.opt.timestep = float(step_dt)
    if model.nq != len(spec.initial_qpos) or model.nv != len(spec.initial_qvel):
        raise WorkerRefusal("initial qpos/qvel widths do not match the compiled model")
    actuator_widths = {len(segment.values) for segment in spec.controls}
    if actuator_widths != {int(model.nu)}:
        raise WorkerRefusal("control width does not match the compiled model actuator count")
    _validate_actuator_ranges(model, spec)
    for channel in spec.channels:
        _validate_channel_binding(model, channel)
    for pair in spec.contact_pairs:
        for name in pair.geom_names:
            if _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0:
                raise WorkerRefusal(f"declared contact geom does not exist: {name}")
    return model


def _validate_actuator_ranges(model: mujoco.MjModel, spec: FixtureSpec) -> None:
    """Require every commanded value to remain inside each compiled control range."""
    for actuator_id in range(int(model.nu)):
        limited = bool(model.actuator_ctrllimited[actuator_id])
        lower = float(model.actuator_ctrlrange[actuator_id, 0])
        upper = float(model.actuator_ctrlrange[actuator_id, 1])
        for segment in spec.controls:
            value = float(segment.values[actuator_id])
            if limited and not lower <= value <= upper:
                raise WorkerRefusal("declared control value exceeds compiled actuator ctrlrange")


def _name_id(model: mujoco.MjModel, object_type: Any, name: str) -> int:
    """Resolve one exact MuJoCo object name to its compiled integer ID."""
    return int(mujoco.mj_name2id(model, object_type, name))


def _validate_channel_binding(model: mujoco.MjModel, channel: ChannelSpec) -> None:
    """Require one declared channel to resolve to exactly one compatible compiled object."""
    if channel.kind in {"JOINT_POSITION", "JOINT_VELOCITY"}:
        joint_id = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, channel.object_name)
        if joint_id < 0:
            raise WorkerRefusal(f"declared joint does not exist: {channel.object_name}")
        joint_type = int(model.jnt_type[joint_id])
        scalar_types = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
        if joint_type not in scalar_types:
            raise WorkerRefusal("declared joint projection must identify a hinge or slide")
        return
    if channel.kind in {"BODY_POSITION", "BODY_QUATERNION"}:
        if _name_id(model, mujoco.mjtObj.mjOBJ_BODY, channel.object_name) < 0:
            raise WorkerRefusal(f"declared body does not exist: {channel.object_name}")
        return
    sensor_id = _name_id(model, mujoco.mjtObj.mjOBJ_SENSOR, channel.object_name)
    if sensor_id < 0:
        raise WorkerRefusal(f"declared sensor does not exist: {channel.object_name}")
    assert channel.component is not None
    if channel.component >= int(model.sensor_dim[sensor_id]):
        raise WorkerRefusal("declared sensor component exceeds the compiled sensor dimension")


def _compiled_mjb(model: mujoco.MjModel) -> bytes:
    """Serialize the complete compiled model into its exact native MJB bytes."""
    buffer = np.zeros(int(mujoco.mj_sizeModel(model)), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return buffer.tobytes(order="C")


def _initial_data(model: mujoco.MjModel, spec: FixtureSpec) -> mujoco.MjData:
    """Create one deterministic state from the admitted exact initial vectors."""
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray([float(value) for value in spec.initial_qpos], dtype=np.float64)
    data.qvel[:] = np.asarray([float(value) for value in spec.initial_qvel], dtype=np.float64)
    if model.nu:
        data.ctrl[:] = np.asarray([float(value) for value in spec.controls[0].values])
    mujoco.mj_forward(model, data)
    return data


def _finite_float_tokens(value: object, context: str) -> list[str]:
    """Encode one numerical array as exact round-trippable finite binary64 tokens."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise WorkerRefusal(f"{context} could not be measured as binary64") from exc
    if not bool(np.all(np.isfinite(array))):
        raise WorkerRefusal(f"{context} contains a non-finite value")
    return [cast(str, _float_token(float(item))) for item in array.reshape(-1)]


def _integration_state(
    model: mujoco.MjModel, data: mujoco.MjData, state_signature: int, state_size: int
) -> NDArray[np.float64]:
    """Retrieve one complete public integration-state array with exact measured width."""
    state = np.empty(state_size, dtype=np.float64)
    mujoco.mj_getState(model, data, state, state_signature)
    if state.shape != (state_size,) or not bool(np.all(np.isfinite(state))):
        raise WorkerRefusal("complete integration state is malformed or non-finite")
    return np.ascontiguousarray(state, dtype="<f8")


def _integration_state_evidence(state: NDArray[np.float64]) -> dict[str, Any]:
    """Retain exact tokens and a byte-level digest for one complete integration state."""
    normalized = np.ascontiguousarray(state, dtype="<f8")
    return {
        "dtype": "<f8",
        "shape": [int(normalized.size)],
        "values": _finite_float_tokens(normalized, "complete integration state"),
        "sha256": hashlib.sha256(normalized.tobytes(order="C")).hexdigest(),
    }


def _optional_float_projection(data: mujoco.MjData, name: str) -> dict[str, Any]:
    """Record one genuinely available finite data array or an explicit absence."""
    try:
        value = getattr(data, name)
    except (AttributeError, RuntimeError):
        return {"available": False, "shape": None, "values": None}
    array = np.asarray(value, dtype=np.float64)
    return {
        "available": True,
        "shape": list(array.shape),
        "values": _finite_float_tokens(array, name),
    }


def _sentinel_warning_projection(data: mujoco.MjData) -> dict[str, Any]:
    """Retain every public warning counter, or record genuine field absence."""
    try:
        warnings = data.warning
    except (AttributeError, RuntimeError):
        return {"available": False, "passed": True, "records": None}
    records = [
        {"warning_id": index, "number": int(item.number), "lastinfo": int(item.lastinfo)}
        for index, item in enumerate(warnings)
    ]
    return {
        "available": True,
        "passed": all(record["number"] == 0 for record in records),
        "records": records,
    }


def _sentinel_solver_projection(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    """Retain available solver iteration/residual diagnostics and their strict gate."""
    iterations_projection: dict[str, Any]
    residual_projection: dict[str, Any]
    try:
        iterations = np.asarray(data.solver_niter)
    except (AttributeError, RuntimeError):
        iterations_projection = {"available": False, "values": None}
        maximum_iterations = 0
    else:
        if not np.issubdtype(iterations.dtype, np.integer):
            raise WorkerRefusal("solver iteration diagnostics are not integral")
        values = [int(item) for item in iterations.reshape(-1)]
        if any(item < 0 for item in values):
            raise WorkerRefusal("solver iteration diagnostics contain a negative value")
        iterations_projection = {"available": True, "values": values}
        maximum_iterations = max(values, default=0)
    try:
        residuals = np.asarray(data.solver_fwdinv, dtype=np.float64)
    except (AttributeError, RuntimeError):
        residual_projection = {"available": False, "values": None}
    else:
        residual_projection = {
            "available": True,
            "values": _finite_float_tokens(residuals, "solver residual diagnostics"),
        }
    iteration_limit = int(model.opt.iterations)
    passed = maximum_iterations < iteration_limit
    return {
        "iterations": iterations_projection,
        "residuals": residual_projection,
        "iteration_limit": iteration_limit,
        "max_iterations": maximum_iterations,
        "passed": passed,
    }


def _sentinel_contact_projection(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    """Canonicalize active contacts semantically without using raw slot order."""
    contacts: list[dict[str, Any]] = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        names = []
        for geom in (int(contact.geom1), int(contact.geom2)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            if type(name) is not str or not name:
                raise WorkerRefusal("sentinel contacts require stable semantic geom names")
            names.append(name)
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_index, force)
        row = {
            "geom_names": sorted(names),
            "dim": int(contact.dim),
            "distance": cast(str, _float_token(float(contact.dist))),
            "position": _finite_float_tokens(contact.pos, "contact position"),
            "frame": _finite_float_tokens(contact.frame, "contact frame"),
            "friction": _finite_float_tokens(contact.friction, "contact friction"),
            "force": _finite_float_tokens(force, "contact force"),
        }
        if row["distance"] is None:
            raise WorkerRefusal("sentinel contact distance is non-finite")
        contacts.append(row)
    contacts.sort(key=_canonical_json_bytes)
    return {"ncon": int(data.ncon), "contacts": contacts}


def _sentinel_constraint_projection(data: mujoco.MjData) -> dict[str, Any]:
    """Retain and canonicalize only active constraint types and finite forces."""
    active_count = int(data.nefc)
    try:
        types = np.asarray(data.efc_type).reshape(-1)[:active_count]
        forces = np.asarray(data.efc_force, dtype=np.float64).reshape(-1)[:active_count]
    except (AttributeError, RuntimeError) as exc:
        if active_count:
            raise WorkerRefusal(
                "active constraints lack public type or force observations"
            ) from exc
        return {"available": False, "active_count": 0, "rows": None}
    if types.size != active_count or forces.size != active_count:
        raise WorkerRefusal("active constraint projection has the wrong width")
    force_tokens = _finite_float_tokens(forces, "active constraint forces")
    rows = [
        {"type": int(types[index]), "force": force_tokens[index]} for index in range(active_count)
    ]
    rows.sort(key=_canonical_json_bytes)
    return {"available": True, "active_count": active_count, "rows": rows}


def _sentinel_projection(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    """Build the bounded deterministic post-pipeline projection for one data instance."""
    projection = {
        "qacc": _optional_float_projection(data, "qacc"),
        "qfrc_actuator": _optional_float_projection(data, "qfrc_actuator"),
        "qfrc_constraint": _optional_float_projection(data, "qfrc_constraint"),
        "sensordata": _optional_float_projection(data, "sensordata"),
        "warnings": _sentinel_warning_projection(data),
        "solver": _sentinel_solver_projection(model, data),
        "contacts": _sentinel_contact_projection(model, data),
        "constraints": _sentinel_constraint_projection(data),
    }
    projection["projection_sha256"] = _canonical_sha256(projection)
    return projection


def _same_profile_sentinel(
    model: mujoco.MjModel, spec: FixtureSpec, profile_role: str
) -> dict[str, Any]:
    """Reproduce one complete integration state through two fresh data instances."""
    if profile_role not in _PROFILE_ROLES:
        raise WorkerRefusal("sentinel profile role must be baseline or candidate")
    step_dt = max(spec.step_dts)
    document: dict[str, Any] = {
        "schema": _SENTINEL_SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "profile_role": profile_role,
        "fixture_id": spec.fixture_id,
        "step_dt": _decimal_token(step_dt),
        "state_signature": None,
        "state_size": None,
        "warmup_step_count": 2,
        "pre_restore_integration_state": None,
        "pre_restore_integration_state_sha256": None,
        "post_forward_projection": None,
        "post_forward_projection_sha256": None,
        "post_step_integration_state": None,
        "post_step_integration_state_sha256": None,
        "post_step_projection": None,
        "post_step_projection_sha256": None,
        "finite_values": False,
        "warnings_passed": False,
        "solver_converged": False,
        "status": "FAIL",
        "failure_reason": None,
        "limitations": list(_SENTINEL_LIMITATIONS),
    }
    try:
        _require_callback_authority_absent()
        model.opt.timestep = float(step_dt)
        source = _initial_data(model, spec)
        for _index in range(2):
            mujoco.mj_step(model, source)
        state_signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        state_size = int(mujoco.mj_stateSize(model, state_signature))
        if state_size <= 0:
            raise WorkerRefusal("public integration-state size must be positive")
        state = _integration_state(model, source, state_signature, state_size)
        state_evidence = _integration_state_evidence(state)
        document["state_signature"] = state_signature
        document["state_size"] = state_size
        document["pre_restore_integration_state"] = state_evidence
        document["pre_restore_integration_state_sha256"] = state_evidence["sha256"]

        left = mujoco.MjData(model)
        right = mujoco.MjData(model)
        mujoco.mj_setState(model, left, state, state_signature)
        mujoco.mj_setState(model, right, state, state_signature)
        left_restored = _integration_state(model, left, state_signature, state_size)
        right_restored = _integration_state(model, right, state_signature, state_size)
        if not bool(
            _binary64_arrays_equal(left_restored, state)
            and _binary64_arrays_equal(right_restored, state)
        ):
            raise WorkerRefusal("restored integration state differs from the retrieved source")
        mujoco.mj_forward(model, left)
        mujoco.mj_forward(model, right)
        left_forward = _sentinel_projection(model, left)
        right_forward = _sentinel_projection(model, right)
        forward_pair = {"left": left_forward, "right": right_forward}
        document["post_forward_projection"] = forward_pair
        document["post_forward_projection_sha256"] = _canonical_sha256(forward_pair)
        if left_forward != right_forward:
            raise WorkerRefusal("restored post-forward projections differ")

        mujoco.mj_step(model, left)
        mujoco.mj_step(model, right)
        left_state = _integration_state(model, left, state_signature, state_size)
        right_state = _integration_state(model, right, state_signature, state_size)
        left_state_evidence = _integration_state_evidence(left_state)
        right_state_evidence = _integration_state_evidence(right_state)
        state_pair = {"left": left_state_evidence, "right": right_state_evidence}
        document["post_step_integration_state"] = state_pair
        document["post_step_integration_state_sha256"] = _canonical_sha256(state_pair)
        if not bool(_binary64_arrays_equal(left_state, right_state)):
            raise WorkerRefusal("restored post-step integration states differ")

        left_step = _sentinel_projection(model, left)
        right_step = _sentinel_projection(model, right)
        step_pair = {"left": left_step, "right": right_step}
        document["post_step_projection"] = step_pair
        document["post_step_projection_sha256"] = _canonical_sha256(step_pair)
        if left_step != right_step:
            raise WorkerRefusal("restored post-step projections differ")
        projections = (left_forward, right_forward, left_step, right_step)
        document["finite_values"] = True
        document["warnings_passed"] = all(
            cast(dict[str, Any], item["warnings"])["passed"] is True for item in projections
        )
        document["solver_converged"] = all(
            cast(dict[str, Any], item["solver"])["passed"] is True for item in projections
        )
        if not document["warnings_passed"]:
            raise WorkerRefusal("sentinel observed a nonzero warning counter")
        if not document["solver_converged"]:
            raise WorkerRefusal("sentinel solver diagnostics failed")
        document["status"] = "PASS"
    except Exception as exc:
        document["failure_reason"] = f"{type(exc).__name__}: {exc}"[:500]
    document["sentinel_identity_sha256"] = _canonical_sha256(document)
    return document


def _control_at(spec: FixtureSpec, time: Decimal) -> tuple[Decimal, ...]:
    """Select the unique left-boundary zero-order-hold action at one step boundary."""
    for segment in spec.controls:
        if segment.start <= time < segment.end:
            return segment.values
    raise WorkerRefusal("action program does not cover a simulation step boundary")


def _semantic_contact_sample(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    declared_pairs: set[tuple[str, str]],
) -> tuple[dict[tuple[str, str], bool], dict[tuple[str, str], float], list[dict[str, Any]]]:
    """Aggregate semantic normal force while retaining raw manifold order diagnostically."""
    forces = dict.fromkeys(declared_pairs, 0.0)
    occupancy = dict.fromkeys(declared_pairs, False)
    raw_order: list[dict[str, Any]] = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom_1 = int(contact.geom1)
        geom_2 = int(contact.geom2)
        name_1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_1)
        name_2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_2)
        if type(name_1) is not str or type(name_2) is not str:
            raise WorkerRefusal("every contacting geom must have a stable name")
        pair = cast(tuple[str, str], tuple(sorted((name_1, name_2))))
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_index, force)
        distance = float(contact.dist)
        normal = _admit_contact_numerics(distance, force)
        if pair in forces:
            occupancy[pair] = True
            forces[pair] = _finite_contact_sum(
                forces[pair], normal, "semantic contact normal-force aggregate"
            )
        raw_order.append(
            {
                "index": contact_index,
                "geom_names": list(pair),
                "distance": _float_token(distance),
                "normal_force": _float_token(normal),
            }
        )
    return occupancy, forces, raw_order


def _admit_contact_numerics(distance: float, force: NDArray[np.float64]) -> float:
    """Reject a non-finite contact distance or any non-finite force-vector component."""
    if force.shape != (6,):
        raise WorkerRefusal("MuJoCo contact-force vector must have exactly six components")
    if not math.isfinite(distance) or not bool(np.all(np.isfinite(force))):
        raise WorkerRefusal("MuJoCo contact distance and full force vector must be finite")
    return max(0.0, float(force[0]))


def _finite_contact_sum(left: float, right: float, context: str) -> float:
    """Add two finite contact quantities and refuse any binary64 overflow."""
    result = left + right
    if not math.isfinite(result):
        raise WorkerRefusal(f"{context} overflowed to a non-finite value")
    return result


def _accumulate_contact_impulse(current: float, force: float, step_dt: Decimal) -> float:
    """Integrate one contact-force step while refusing product or cumulative overflow."""
    increment = force * float(step_dt)
    if not math.isfinite(increment):
        raise WorkerRefusal("semantic contact impulse increment overflowed")
    return _finite_contact_sum(current, increment, "semantic contact cumulative impulse")


def _new_contact_states(
    spec: FixtureSpec,
    initial_occupancy: dict[tuple[str, str], bool],
    initial_forces: dict[tuple[str, str], float],
) -> dict[str, ContactState]:
    """Initialize semantic contact episodes, including a possible physical-time-zero onset."""
    result: dict[str, ContactState] = {}
    for pair in spec.contact_pairs:
        force = initial_forces[pair.geom_names]
        occupied = initial_occupancy[pair.geom_names]
        events = [{"event": "ONSET", "time": "0"}] if occupied else []
        result[pair.channel_id] = ContactState(
            spec=pair,
            occupied=occupied,
            onset=Decimal("0") if occupied else None,
            events=events,
            segments=[],
            persistence_steps=0,
            aggregate_normal_impulse=0.0,
            current_normal_force=force,
        )
    return result


def _advance_contact_states(
    states: dict[str, ContactState],
    occupancy: dict[tuple[str, str], bool],
    forces: dict[tuple[str, str], float],
    time: Decimal,
    step_dt: Decimal,
) -> None:
    """Advance exact semantic contact events and per-step force integrals."""
    for state in states.values():
        force = forces[state.spec.geom_names]
        occupied = occupancy[state.spec.geom_names]
        state.current_normal_force = force
        state.aggregate_normal_impulse = _accumulate_contact_impulse(
            state.aggregate_normal_impulse, force, step_dt
        )
        if occupied:
            state.persistence_steps += 1
        if occupied and not state.occupied:
            state.events.append({"event": "ONSET", "time": _decimal_token(time)})
            state.onset = time
        elif state.occupied and not occupied:
            if state.onset is None:
                raise WorkerRefusal("contact release occurred without an onset")
            persistence = time - state.onset
            state.events.append({"event": "RELEASE", "time": _decimal_token(time)})
            state.segments.append(
                {
                    "onset": _decimal_token(state.onset),
                    "release": _decimal_token(time),
                    "persistence": _decimal_token(persistence),
                }
            )
            state.onset = None
        state.occupied = occupied


def _finalize_contacts(
    states: dict[str, ContactState], horizon: Decimal, step_dt: Decimal
) -> list[dict[str, Any]]:
    """Finalize ordered contact evidence, preserving an open horizon episode if present."""
    result: list[dict[str, Any]] = []
    for channel_id in sorted(states):
        state = states[channel_id]
        if state.onset is not None:
            raise WorkerRefusal("declared contact episode remains open at the horizon")
        result.append(
            {
                "channel_id": channel_id,
                "geom_names": list(state.spec.geom_names),
                "events": state.events,
                "segments": state.segments,
                "persistence": _decimal_token(step_dt * state.persistence_steps),
                "aggregate_normal_impulse": _float_token(state.aggregate_normal_impulse),
            }
        )
    return result


def _warning_records(data: mujoco.MjData) -> list[dict[str, int]]:
    """Return every nonzero MuJoCo warning counter in stable numeric order."""
    result: list[dict[str, int]] = []
    for warning_id, warning in enumerate(data.warning):
        number = int(warning.number)
        if number:
            result.append(
                {"warning_id": warning_id, "number": number, "lastinfo": int(warning.lastinfo)}
            )
    return result


def _solver_sample(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, float]:
    """Measure the current maximum solver iteration count and forward/inverse residual."""
    iterations = np.asarray(data.solver_niter)
    max_iterations = int(np.max(iterations)) if iterations.size else 0
    residuals = np.asarray(data.solver_fwdinv, dtype=np.float64)
    if not bool(np.all(np.isfinite(residuals))):
        max_residual = math.inf
    else:
        max_residual = float(np.max(np.abs(residuals))) if residuals.size else 0.0
    if max_iterations > int(model.opt.iterations):
        raise WorkerRefusal("MuJoCo reported solver iterations above its configured limit")
    return max_iterations, max_residual


def _finite_state(data: mujoco.MjData) -> bool:
    """Check every selected dynamic state family for finite values."""
    arrays = (
        data.qpos,
        data.qvel,
        data.qacc,
        data.ctrl,
        data.sensordata,
        data.xpos,
        data.xquat,
    )
    return all(bool(np.all(np.isfinite(array))) for array in arrays)


def _observe_channel(model: mujoco.MjModel, data: mujoco.MjData, spec: ChannelSpec) -> Any:
    """Read one already-validated declared semantic channel from live MuJoCo state."""
    if spec.kind in {"JOINT_POSITION", "JOINT_VELOCITY"}:
        joint_id = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.object_name)
        address = (
            int(model.jnt_qposadr[joint_id])
            if spec.kind == "JOINT_POSITION"
            else int(model.jnt_dofadr[joint_id])
        )
        source = data.qpos if spec.kind == "JOINT_POSITION" else data.qvel
        return float(source[address])
    if spec.kind == "BODY_POSITION":
        body_id = _name_id(model, mujoco.mjtObj.mjOBJ_BODY, spec.object_name)
        assert spec.component is not None
        return float(data.xpos[body_id, spec.component])
    if spec.kind == "BODY_QUATERNION":
        body_id = _name_id(model, mujoco.mjtObj.mjOBJ_BODY, spec.object_name)
        return [float(value) for value in data.xquat[body_id]]
    sensor_id = _name_id(model, mujoco.mjtObj.mjOBJ_SENSOR, spec.object_name)
    assert spec.component is not None
    address = int(model.sensor_adr[sensor_id]) + spec.component
    return float(data.sensordata[address])


def _base_descriptor(
    spec: ChannelSpec, count: int, array_key: str, fixture: FixtureSpec
) -> dict[str, Any]:
    """Build one exact replayable descriptor for a declared state channel."""
    quaternion = spec.kind == "BODY_QUATERNION"
    return {
        "array_key": array_key,
        "channel_id": spec.channel_id,
        "kind": spec.kind,
        "semantic_type": "UNIT_QUATERNION_WXYZ" if quaternion else "CONTINUOUS_SCALAR",
        "object_name": spec.object_name,
        "component": spec.component,
        "shape": [count, 4] if quaternion else [count],
        "dtype": "<f8",
        "scale": None if quaternion else _decimal_token(cast(Decimal, spec.scale)),
        "tolerance": _decimal_token(
            fixture.so3_tolerance if quaternion else fixture.continuous_tolerance
        ),
    }


def _contact_descriptors(
    pair: ContactPairSpec, count: int, fixture: FixtureSpec
) -> list[dict[str, Any]]:
    """Build exact occupancy, force, and cumulative-impulse contact descriptors."""
    object_name = pair.channel_id
    definitions = (
        ("occupancy", "CONTACT_OCCUPANCY", "BOOLEAN_OCCUPANCY", "|u1", None, None),
        (
            "normal_force",
            "CONTACT_NORMAL_FORCE",
            "CONTINUOUS_SCALAR",
            "<f8",
            pair.force_scale,
            fixture.continuous_tolerance,
        ),
        (
            "normal_impulse",
            "CONTACT_NORMAL_IMPULSE",
            "CONTINUOUS_SCALAR",
            "<f8",
            pair.impulse_scale,
            fixture.continuous_tolerance,
        ),
    )
    return [
        {
            "array_key": "",
            "channel_id": f"{pair.channel_id}.{suffix}",
            "kind": kind,
            "semantic_type": semantic_type,
            "object_name": object_name,
            "component": None,
            "shape": [count],
            "dtype": dtype,
            "scale": None if scale is None else _decimal_token(scale),
            "tolerance": None if tolerance is None else _decimal_token(tolerance),
        }
        for suffix, kind, semantic_type, dtype, scale, tolerance in definitions
    ]


def _diagnostic_values(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    running_finite: bool,
    running_warning_free: bool,
    running_iterations: int,
    running_residual: float,
    time: Decimal,
) -> dict[str, Any]:
    """Build one cumulative time-local gate sample for prefix evaluation."""
    solver_converged = (
        running_finite
        and running_warning_free
        and running_iterations < int(model.opt.iterations)
        and math.isfinite(running_residual)
    )
    return {
        "time": _decimal_token(time),
        "finite_values": running_finite,
        "warnings_passed": running_warning_free,
        "solver_converged": solver_converged,
        "max_solver_iterations": running_iterations,
        "max_solver_residual": _float_token(running_residual),
    }


def _append_observation(
    model: mujoco.MjModel,
    observation_data: mujoco.MjData,
    spec: FixtureSpec,
    contact_states: dict[str, ContactState],
    rows: dict[str, list[Any]],
) -> None:
    """Append every declared state and contact projection at one observation boundary."""
    for channel in spec.channels:
        rows[channel.channel_id].append(_observe_channel(model, observation_data, channel))
    for channel_id, state in contact_states.items():
        rows[f"{channel_id}.occupancy"].append(1 if state.occupied else 0)
        rows[f"{channel_id}.normal_force"].append(state.current_normal_force)
        rows[f"{channel_id}.normal_impulse"].append(state.aggregate_normal_impulse)


def _project_observation_data(
    model: mujoco.MjModel,
    live: mujoco.MjData,
    observer: mujoco.MjData,
    state_signature: int,
    state_size: int,
) -> mujoco.MjData:
    """Forward an isolated exact integration-state copy for same-time declared observations."""
    state = _integration_state(model, live, state_signature, state_size)
    mujoco.mj_setState(model, observer, state, state_signature)
    restored = _integration_state(model, observer, state_signature, state_size)
    if not bool(_binary64_arrays_equal(restored, state)):
        raise WorkerRefusal("observation projection did not restore the exact integration state")
    mujoco.mj_forward(model, observer)
    if not _finite_state(observer):
        raise WorkerRefusal("observation projection contains a non-finite value")
    return observer


def _normalize_arrays_and_descriptors(
    spec: FixtureSpec,
    rows: dict[str, list[Any]],
    count: int,
) -> tuple[dict[str, NDArray[Any]], list[dict[str, Any]]]:
    """Normalize channel arrays to frozen dtypes, shapes, keys, and lexical channel order."""
    descriptors: list[dict[str, Any]] = []
    for channel in spec.channels:
        descriptors.append(_base_descriptor(channel, count, "", spec))
    for pair in spec.contact_pairs:
        descriptors.extend(_contact_descriptors(pair, count, spec))
    descriptors.sort(key=lambda item: cast(str, item["channel_id"]))
    arrays: dict[str, NDArray[Any]] = {}
    for index, descriptor in enumerate(descriptors):
        array_key = f"channel_{index:04d}"
        descriptor["array_key"] = array_key
        channel_id = cast(str, descriptor["channel_id"])
        dtype = cast(str, descriptor["dtype"])
        array = np.asarray(rows[channel_id], dtype=np.uint8 if dtype == "|u1" else "<f8")
        expected_shape = tuple(cast(list[int], descriptor["shape"]))
        if array.shape != expected_shape:
            raise WorkerRefusal("observed channel has an unexpected normalized shape")
        arrays[array_key] = np.ascontiguousarray(array)
    return arrays, descriptors


def _update_hash_frame(digest: Any, label: str, payload: bytes) -> None:
    """Append one length-delimited label/payload frame to a SHA-256 preimage."""
    encoded = label.encode("utf-8")
    digest.update(struct.pack(">Q", len(encoded)))
    digest.update(encoded)
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def _canonical_trace_sha256(
    observation_times: NDArray[np.float64],
    descriptors: list[dict[str, Any]],
    arrays: dict[str, NDArray[Any]],
    contacts: list[dict[str, Any]],
) -> str:
    """Hash canonical metadata and explicit C-contiguous little-endian scientific arrays."""
    header = {
        "schema": _CANONICAL_TRACE_SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "observation_times": {
            "array_key": "observation_times",
            "shape": list(observation_times.shape),
            "dtype": "<f8",
            "unit": "seconds",
        },
        "channels": descriptors,
        "semantic_contacts": contacts,
    }
    digest = hashlib.sha256()
    _update_hash_frame(digest, "metadata", _canonical_json_bytes(header))
    normalized_times = np.ascontiguousarray(observation_times, dtype="<f8")
    _update_hash_frame(digest, "observation_times", normalized_times.tobytes(order="C"))
    for descriptor in descriptors:
        key = cast(str, descriptor["array_key"])
        _update_hash_frame(digest, key, arrays[key].tobytes(order="C"))
    return digest.hexdigest()


def _simulate(model: mujoco.MjModel, spec: FixtureSpec, step_dt: Decimal) -> SimulationEvidence:
    """Execute one exact grid and retain semantic channels, contacts, and time-local gates."""
    data = _initial_data(model, spec)
    observation_data = mujoco.MjData(model)
    state_signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state_size = int(mujoco.mj_stateSize(model, state_signature))
    if state_size <= 0:
        raise WorkerRefusal("public integration-state size must be positive")
    declared_pairs = {pair.geom_names for pair in spec.contact_pairs}
    initial_occupancy, initial_forces, initial_raw = _semantic_contact_sample(
        model, data, declared_pairs
    )
    contact_states = _new_contact_states(spec, initial_occupancy, initial_forces)
    rows: dict[str, list[Any]] = {channel.channel_id: [] for channel in spec.channels}
    for pair in spec.contact_pairs:
        rows[f"{pair.channel_id}.occupancy"] = []
        rows[f"{pair.channel_id}.normal_force"] = []
        rows[f"{pair.channel_id}.normal_impulse"] = []
    total_steps = int(spec.horizon / step_dt)
    observation_stride = int(spec.control_dt / step_dt)
    observation_count = int(spec.horizon / spec.control_dt) + 1
    observation_tokens = [
        _decimal_token(spec.control_dt * index) for index in range(observation_count)
    ]
    observation_times = np.asarray(
        [float(Decimal(token)) for token in observation_tokens], dtype="<f8"
    )
    raw_order_digest = hashlib.sha256()
    _update_hash_frame(raw_order_digest, "0", _canonical_json_bytes(initial_raw))
    running_finite = _finite_state(data)
    running_warning_free = not _warning_records(data)
    running_iterations, running_residual = _solver_sample(model, data)
    diagnostics_samples = [
        _diagnostic_values(
            model,
            data,
            running_finite,
            running_warning_free,
            running_iterations,
            running_residual,
            Decimal("0"),
        )
    ]
    _append_observation(
        model,
        _project_observation_data(model, data, observation_data, state_signature, state_size),
        spec,
        contact_states,
        rows,
    )
    for step_index in range(1, total_steps + 1):
        start_time = step_dt * (step_index - 1)
        if model.nu:
            data.ctrl[:] = np.asarray(
                [float(value) for value in _control_at(spec, start_time)], dtype=np.float64
            )
        mujoco.mj_step(model, data)
        physical_time = step_dt * step_index
        occupancy, forces, raw_order = _semantic_contact_sample(model, data, declared_pairs)
        _advance_contact_states(contact_states, occupancy, forces, physical_time, step_dt)
        _update_hash_frame(
            raw_order_digest, _decimal_token(physical_time), _canonical_json_bytes(raw_order)
        )
        running_finite = running_finite and _finite_state(data)
        running_warning_free = running_warning_free and not _warning_records(data)
        current_iterations, current_residual = _solver_sample(model, data)
        running_iterations = max(running_iterations, current_iterations)
        running_residual = max(running_residual, current_residual)
        if step_index % observation_stride == 0:
            _append_observation(
                model,
                _project_observation_data(
                    model, data, observation_data, state_signature, state_size
                ),
                spec,
                contact_states,
                rows,
            )
            diagnostics_samples.append(
                _diagnostic_values(
                    model,
                    data,
                    running_finite,
                    running_warning_free,
                    running_iterations,
                    running_residual,
                    physical_time,
                )
            )
    if len(diagnostics_samples) != observation_count:
        raise WorkerRefusal("simulation did not land on every exact observation boundary")
    contacts = _finalize_contacts(contact_states, spec.horizon, step_dt)
    arrays, descriptors = _normalize_arrays_and_descriptors(spec, rows, observation_count)
    warnings = _warning_records(data)
    diagnostics = {
        "finite_values": running_finite,
        "warnings_passed": running_warning_free,
        "solver_converged": cast(bool, diagnostics_samples[-1]["solver_converged"]),
        "warning_records": warnings,
        "max_solver_iterations": running_iterations,
        "solver_iteration_limit": int(model.opt.iterations),
        "max_solver_residual": _float_token(running_residual),
        "raw_contact_order_sha256": raw_order_digest.hexdigest(),
        "samples": diagnostics_samples,
    }
    return SimulationEvidence(observation_times, arrays, descriptors, contacts, diagnostics)


def _trace_identity(evidence: SimulationEvidence, spec: FixtureSpec) -> dict[str, Any]:
    """Build the replayable trace locator and scientific canonical identity."""
    count = int(evidence.observation_times.shape[0])
    return {
        "schema": _TRACE_SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "arrays_locator": "trace.npz",
        "observation_time_tokens": [
            _decimal_token(spec.control_dt * index) for index in range(count)
        ],
        "observation_count": count,
        "channels": evidence.descriptors,
        "canonical_trace_sha256": _canonical_trace_sha256(
            evidence.observation_times,
            evidence.descriptors,
            evidence.arrays,
            evidence.contacts,
        ),
    }


def _subject_identity(spec: FixtureSpec, fixture_raw: bytes, mjb: bytes) -> dict[str, Any]:
    """Bind campaign role, exact manifest declaration, source closure, and complete MJB."""
    closure = _source_identity(spec, fixture_raw)
    return {
        "fixture_id": spec.fixture_id,
        "campaign_role": spec.campaign_role,
        "baseline_fixture_id": spec.baseline_fixture_id,
        "fixture_manifest_sha256": _canonical_sha256(spec.echo),
        "fixture_raw_sha256": hashlib.sha256(fixture_raw).hexdigest(),
        "source_closure": closure,
        "compiled_mjb_sha256": hashlib.sha256(mjb).hexdigest(),
        "compiled_mjb_size_bytes": len(mjb),
    }


def _completed_result(
    *,
    manifest_raw: bytes,
    manifest_echo: dict[str, Any],
    spec: FixtureSpec,
    step_dt: Decimal,
    repeat_id: int,
    profile_token: str,
    package_version: str,
    runtime: dict[str, Any],
    fixture_raw: bytes,
    mjb: bytes,
    evidence: SimulationEvidence,
) -> dict[str, Any]:
    """Assemble one completed evidence result without making a migration decision."""
    result = {
        "schema": _RESULT_SCHEMA,
        "schema_version": runtime["schema_version"],
        "status": "COMPLETED",
        "fixture_id": spec.fixture_id,
        "step_dt": _decimal_token(step_dt),
        "repeat_id": repeat_id,
        "manifest_raw_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_echo": manifest_echo,
        "subject": _subject_identity(spec, fixture_raw, mjb),
        "workload": _workload_identity(spec),
        "runtime": runtime,
        "trace": _trace_identity(evidence, spec),
        "contacts": evidence.contacts,
        "diagnostics": evidence.diagnostics,
        "limitations": list(_LIMITATIONS),
    }
    if runtime["schema_version"] == 1:
        result.update({"profile_id": profile_token, "profile_version": package_version})
    else:
        result.update(
            {
                "profile_role": profile_token,
                "package_version": package_version,
                "native_version": runtime["native_version"],
                "native_version_integer": runtime["native_version_integer"],
                "profile_identity_sha256": runtime["profile_identity_sha256"],
                "runtime_identity_sha256": runtime["runtime_identity_sha256"],
                "sentinel_identity_sha256": runtime["sentinel_identity_sha256"],
            }
        )
    return result


def _refused_result(
    fixture_id: str,
    step_dt: str,
    repeat_id: int,
    reason: str,
    profile_role: str | None = None,
) -> dict[str, Any]:
    """Assemble a bounded refusal record for a failed standalone attempt."""
    version = str(getattr(mujoco, "__version__", "UNKNOWN"))
    schema_version = 1 if profile_role is None else _PRODUCTION_SCHEMA_VERSION
    result = {
        "schema": _RESULT_SCHEMA,
        "schema_version": schema_version,
        "status": "REFUSED",
        "fixture_id": fixture_id,
        "step_dt": step_dt,
        "repeat_id": repeat_id,
        "manifest_raw_sha256": None,
        "manifest_echo": None,
        "subject": None,
        "workload": None,
        "runtime": None,
        "trace": None,
        "contacts": [],
        "diagnostics": {"refusal_reason": reason},
        "limitations": list(_LIMITATIONS),
    }
    if profile_role is None:
        result.update(
            {
                "profile_id": {"3.10.0": "A_3.10.0", "3.11.0": "B_3.11.0"}.get(version, "UNKNOWN"),
                "profile_version": version,
            }
        )
    else:
        native_version = getattr(mujoco, "mj_versionString", lambda: "UNKNOWN")()
        native_integer = getattr(mujoco, "mj_version", lambda: None)()
        result.update(
            {
                "profile_role": profile_role,
                "package_version": version,
                "native_version": (native_version if type(native_version) is str else "UNKNOWN"),
                "native_version_integer": (native_integer if type(native_integer) is int else None),
                "profile_identity_sha256": None,
                "runtime_identity_sha256": None,
                "sentinel_identity_sha256": None,
            }
        )
    return result


def _run_cell(
    manifest_path: Path,
    fixture_id: str,
    step_dt: Decimal,
    repeat_id: int,
    output: OwnedOutput,
    profile_role: str | None = None,
) -> None:
    """Execute and atomically complete one admitted worker cell inside a new directory."""
    manifest_raw, manifest_echo, spec = _load_manifest(manifest_path, fixture_id)
    if step_dt not in spec.step_dts:
        raise WorkerRefusal("--step-dt is not declared by the selected fixture")
    output.write_bytes("input_manifest.json", manifest_raw)
    _, fixture_raw = _resolve_fixture_source(manifest_path, spec.xml_path)
    xml_text = _admit_self_contained_xml(fixture_raw)
    output.write_bytes("fixture.xml", fixture_raw)
    model = _compile_model(xml_text, spec, step_dt)
    mjb = _compiled_mjb(model)
    output.write_bytes("model.mjb", mjb)
    profile_token, package_version, runtime = _runtime_identity(profile_role)
    evidence = _simulate(model, spec, step_dt)
    npz_arrays = {"observation_times": evidence.observation_times, **evidence.arrays}
    output.write_npz(npz_arrays)
    result = _completed_result(
        manifest_raw=manifest_raw,
        manifest_echo=manifest_echo,
        spec=spec,
        step_dt=step_dt,
        repeat_id=repeat_id,
        profile_token=profile_token,
        package_version=package_version,
        runtime=runtime,
        fixture_raw=fixture_raw,
        mjb=mjb,
        evidence=evidence,
    )
    output.write_json("result.json", result)
    output.write_checksums()


def _parse_step_dt(token: str) -> Decimal:
    """Parse one exact supported CLI timestep token."""
    if _DECIMAL_PATTERN.fullmatch(token) is None:
        raise argparse.ArgumentTypeError("step dt must be a plain finite decimal")
    value = Decimal(token)
    if value not in _FIXED_STEP_DTS:
        raise argparse.ArgumentTypeError("step dt must be exactly 0.004, 0.002, or 0.001")
    return value


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the exact standalone worker invocation surface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--step-dt", type=_parse_step_dt, required=True)
    parser.add_argument("--repeat-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--profile-role", choices=_PROFILE_ROLES)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one native cell, preserving either completed evidence or a bounded refusal."""
    args = _arguments(argv)
    fixture_id = str(args.fixture_id)
    if _ID_PATTERN.fullmatch(fixture_id) is None:
        raise SystemExit("--fixture-id is invalid")
    step_dt = cast(Decimal, args.step_dt)
    repeat_id = cast(int, args.repeat_id)
    profile_role = cast(str | None, args.profile_role)
    try:
        output = OwnedOutput(cast(Path, args.output))
    except WorkerRefusal as exc:
        sys.stderr.write(f"REFUSED: {exc}\n")
        return 2
    try:
        _run_cell(
            cast(Path, args.manifest),
            fixture_id,
            step_dt,
            repeat_id,
            output,
            profile_role,
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if not (output.path / "result.json").exists():
            output.write_json(
                "result.json",
                _refused_result(
                    fixture_id,
                    _decimal_token(step_dt),
                    repeat_id,
                    reason,
                    profile_role,
                ),
            )
        if not (output.path / "CHECKSUMS.sha256").exists():
            output.write_checksums()
        sys.stderr.write(f"REFUSED: {reason}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
