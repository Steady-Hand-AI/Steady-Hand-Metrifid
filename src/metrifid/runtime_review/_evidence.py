"""Strict admission and private-case reconstruction for runtime-review evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from .._json_admission import (
    CONFIG_JSON_LIMITS,
    RECEIPT_JSON_LIMITS,
    JsonAdmissionError,
    JsonAdmissionLimits,
    enforce_json_structure,
    read_bounded_regular_file,
)
from .._native_upgrade import (
    CaseEvidence,
    GateEvent,
    OrientationObservation,
    Quaternion,
    ScalarObservation,
)
from .._npz import ArtifactAdmissionRefusal, load_npz_arrays
from ..json_values import CanonicalValue, canonical_json_bytes, canonical_sha256
from ..operational import OperationalReasonCode
from ._config import (
    AdmittedRuntimeReviewConfiguration,
    AdmittedRuntimeReviewConfigurationAny,
    AdmittedRuntimeReviewConfigurationV2,
    RuntimeReviewCellConfig,
)
from ._native_profile_identity import (
    _FROZEN_WORKER_SHA256,
    ProfileIdentityRefusal,
    _record_bound_distribution_identity,
    load_native_profile_identity_v2,
    profile_identity_receipt_projection_v2,
)

_MEMBERS: Final = frozenset(
    {
        "CHECKSUMS.sha256",
        "fixture.xml",
        "input_manifest.json",
        "model.mjb",
        "result.json",
        "trace.npz",
    }
)
_CHECKSUMMED_MEMBERS: Final = (
    "fixture.xml",
    "input_manifest.json",
    "model.mjb",
    "result.json",
    "trace.npz",
)
_MEMBER_LIMITS: Final = {
    "CHECKSUMS.sha256": 4 * 1024,
    "fixture.xml": 16 * 1024 * 1024,
    "input_manifest.json": 4 * 1024 * 1024,
    "model.mjb": 512 * 1024 * 1024,
    "result.json": 64 * 1024 * 1024,
    "trace.npz": 512 * 1024 * 1024,
}
# The runtime input manifest is caller-supplied configuration, so it keeps the configuration depth
# and string ceilings. Its node ceiling is derived from its own byte ceiling instead of the generic
# configuration node count, because a semantically valid campaign manifest -- 4,096 qpos and qvel
# values, fifty contiguous control segments of 256 values, and its declared channels -- is dense in
# nodes and sparse in bytes. The densest strict JSON is an array of single-digit numbers, where each
# further node costs a digit and a separator, so a document that fits the byte ceiling can never
# reach this node ceiling: it bounds memory without ever being the reason a valid manifest is
# refused.
_MANIFEST_JSON_LIMITS: Final = JsonAdmissionLimits(
    max_bytes=_MEMBER_LIMITS["input_manifest.json"],
    max_depth=CONFIG_JSON_LIMITS.max_depth,
    max_nodes=_MEMBER_LIMITS["input_manifest.json"] // 2 + 1,
    max_string_bytes=CONFIG_JSON_LIMITS.max_string_bytes,
)

_RESULT_KEYS_V1: Final = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        "fixture_id",
        "profile_id",
        "profile_version",
        "step_dt",
        "repeat_id",
        "manifest_raw_sha256",
        "manifest_echo",
        "subject",
        "workload",
        "runtime",
        "trace",
        "contacts",
        "diagnostics",
        "limitations",
    }
)
_RESULT_KEYS_V2: Final = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        "fixture_id",
        "profile_role",
        "package_version",
        "native_version",
        "native_version_integer",
        "profile_identity_sha256",
        "runtime_identity_sha256",
        "sentinel_identity_sha256",
        "step_dt",
        "repeat_id",
        "manifest_raw_sha256",
        "manifest_echo",
        "subject",
        "workload",
        "runtime",
        "trace",
        "contacts",
        "diagnostics",
        "limitations",
    }
)
_FIXTURE_KEYS: Final = frozenset(
    {
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
)
_THREAD_ENVIRONMENT: Final = {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_LIMITATIONS: Final = (
    "ONE_EXACT_SELF_CONTAINED_MJCF_CLOSURE_ONLY",
    "ONE_EXACT_INITIAL_STATE_AND_ACTION_PROGRAM_ONLY",
    "EXACT_NATIVE_CPU_RUNTIME_PROFILES_ONLY",
    "NO_UNIVERSAL_MUJOCO_VERSION_EQUIVALENCE_CLAIM",
    "NO_POLICY_OR_HARDWARE_SAFETY_CLAIM",
    "NO_TASK_SUCCESS_OR_REAL_WORLD_TRANSFER_CLAIM",
    "NO_BACKEND_PARITY_CLAIM",
    "WORKER_EMITS_EVIDENCE_ONLY_NO_MIGRATION_DECISION",
)
_FIXED_HORIZON: Final = Decimal("1")
_FIXED_CONTROL_DT: Final = Decimal("0.02")
_FIXED_STEP_DTS: Final = (Decimal("0.004"), Decimal("0.002"), Decimal("0.001"))
_FIXED_CONTINUOUS_TOLERANCE: Final = Decimal("0.000001")
_FIXED_SO3_TOLERANCE: Final = Decimal("0.0000001")
_FIXED_CONTACT_EVENT_TIME_TOLERANCE: Final = Decimal("0.02")
_FIXED_MINIMUM_COMMON_PREFIX: Final = Decimal("0.50")
_FIXED_NUMPY_VERSION: Final = "2.3.5"
_MAX_FIXTURE_XML_BYTES: Final = 1_048_576
_EXPECTED_OBSERVATION_TIME_TOKENS: Final = tuple(
    format(Decimal(index) * _FIXED_CONTROL_DT, "f").rstrip("0").rstrip(".") or "0"
    for index in range(51)
)
_IDENTIFIER_PATTERN: Final = re.compile(r"[a-z][a-z0-9_.-]{0,95}\Z")
_MANIFEST_CHANNEL_KINDS: Final = frozenset(
    {
        "JOINT_POSITION",
        "JOINT_VELOCITY",
        "BODY_POSITION",
        "BODY_QUATERNION",
        "SENSOR",
    }
)

FloatArray: TypeAlias = npt.NDArray[np.float64]
ByteArray: TypeAlias = npt.NDArray[np.uint8]


class RuntimeEvidenceAdmissionError(ValueError):
    """Raised when one evidence cell or cross-cell identity fails admission."""


@dataclass(frozen=True, slots=True)
class EvidenceMember:
    """One exact regular member of an admitted six-member evidence cell."""

    name: str
    sha256: str
    size_bytes: int

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return the member's stable byte identity."""
        return {"name": self.name, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True, slots=True)
class EvidenceChannel:
    """One admitted semantic channel and its immutable raw observation values."""

    array_key: str
    channel_id: str
    kind: str
    semantic_type: str
    object_name: str
    component: int | None
    scale_token: str | None
    tolerance_token: str | None
    dtype: str
    shape: tuple[int, ...]
    values: FloatArray | ByteArray

    @property
    def scale(self) -> float | None:
        """Return the declared physical scale as binary64 for the private evaluator."""
        return None if self.scale_token is None else float(Decimal(self.scale_token))

    @property
    def tolerance(self) -> float | None:
        """Return the declared tolerance as binary64 for the private evaluator."""
        return None if self.tolerance_token is None else float(Decimal(self.tolerance_token))

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return the stable channel-layout fields without raw array content."""
        return {
            "channel_id": self.channel_id,
            "kind": self.kind,
            "semantic_type": self.semantic_type,
            "object_name": self.object_name,
            "component": self.component,
            "scale": self.scale_token,
            "tolerance": self.tolerance_token,
            "dtype": self.dtype,
            "value_shape": list(self.shape[1:]),
        }


@dataclass(frozen=True, slots=True)
class EvidenceContact:
    """One admitted semantic contact topology and its aggregate diagnostics."""

    channel_id: str
    geom_names: tuple[str, str]
    events: tuple[tuple[str, str], ...]
    segments: tuple[tuple[str, str, str], ...]
    persistence: str
    aggregate_normal_impulse: str


@dataclass(frozen=True, slots=True)
class DiagnosticSample:
    """One time-local cumulative finite, warning, and solver gate sample."""

    time_token: str
    finite_values: bool
    warnings_passed: bool
    solver_converged: bool
    max_solver_iterations: int
    max_solver_residual: str | None

    @property
    def time(self) -> float:
        """Return the exact decimal clock token as binary64."""
        return float(Decimal(self.time_token))


@dataclass(frozen=True, slots=True)
class AdmittedEvidenceCell:
    """One checksum-bound runtime-review cell with reconstructed raw evidence."""

    slot: RuntimeReviewCellConfig
    source_directory: Path
    members: tuple[EvidenceMember, ...]
    schema_version: int
    profile_id: str
    profile_version: str
    profile_identity: dict[str, CanonicalValue] | None
    fixture_id: str
    contact_event_time_tolerance: str
    manifest_raw_sha256: str
    fixture_raw_sha256: str
    compiled_mjb_sha256: str
    runtime_identity_sha256: str
    canonical_trace_sha256: str
    observation_times: FloatArray
    observation_time_tokens: tuple[str, ...]
    channels: tuple[EvidenceChannel, ...]
    contacts: tuple[EvidenceContact, ...]
    diagnostics: tuple[DiagnosticSample, ...]
    finite_values: bool
    warnings_passed: bool
    solver_converged: bool
    subject: dict[str, CanonicalValue]
    workload: dict[str, CanonicalValue]
    runtime: dict[str, CanonicalValue]
    diagnostic_primitive: dict[str, CanonicalValue]

    @property
    def profile_role(self) -> str:
        """Return this cell's canonical baseline or candidate role."""
        return self.slot.profile_role

    @property
    def step_dt(self) -> str:
        """Return this cell's exact step-size token."""
        return self.slot.step_dt

    @property
    def repeat_id(self) -> int:
        """Return this cell's declared repeat identifier."""
        return self.slot.repeat_id

    @property
    def member_sha256(self) -> dict[str, str]:
        """Return all six member names mapped to their independently measured digests."""
        return {member.name: member.sha256 for member in self.members}

    def to_primitive(self, locator: str | None = None) -> dict[str, CanonicalValue]:
        """Return a receipt-ready slot and six-member identity projection."""
        if self.schema_version == 2:
            runtime = self.runtime
            return {
                "profile_role": self.profile_role,
                "package_version": runtime["package_version"],
                "native_version": runtime["native_version"],
                "native_version_integer": runtime["native_version_integer"],
                "profile_identity_sha256": runtime["profile_identity_sha256"],
                "runtime_identity_sha256": self.runtime_identity_sha256,
                "sentinel_identity_sha256": runtime["sentinel_identity_sha256"],
                "step_dt": self.step_dt,
                "repeat_id": self.repeat_id,
                "directory": locator if locator is not None else self.slot.directory,
                "members": [member.to_primitive() for member in self.members],
                "canonical_trace_sha256": self.canonical_trace_sha256,
                "compiled_mjb_sha256": self.compiled_mjb_sha256,
            }
        return {
            "profile_role": self.profile_role,
            "profile_id": self.profile_id,
            "mujoco_version": self.profile_version,
            "step_dt": self.step_dt,
            "repeat_id": self.repeat_id,
            "directory": locator if locator is not None else self.slot.directory,
            "members": [member.to_primitive() for member in self.members],
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "canonical_trace_sha256": self.canonical_trace_sha256,
            "compiled_mjb_sha256": self.compiled_mjb_sha256,
        }


@dataclass(frozen=True, slots=True)
class AdmittedRuntimeEvidence:
    """All twelve cells, stable identity projections, and reconstructed decision inputs."""

    configuration: AdmittedRuntimeReviewConfigurationAny
    cells: tuple[AdmittedEvidenceCell, ...]
    case_evidence: CaseEvidence
    profiles: dict[str, CanonicalValue]
    subject: dict[str, CanonicalValue]
    workload: dict[str, CanonicalValue]
    channel_layout: tuple[dict[str, CanonicalValue], ...]
    observation_time_tokens: tuple[str, ...]


def admit_runtime_evidence(
    configuration: AdmittedRuntimeReviewConfigurationAny,
    *,
    cell_directories: tuple[Path, ...] | None = None,
) -> AdmittedRuntimeEvidence:
    """Admit, bind, and reconstruct the twelve exact configured evidence cells."""
    if not isinstance(
        configuration,
        (AdmittedRuntimeReviewConfiguration, AdmittedRuntimeReviewConfigurationV2),
    ):
        raise TypeError("configuration must be an admitted Runtime Review configuration")
    directories = configuration.cell_directories if cell_directories is None else cell_directories
    if type(directories) is not tuple or len(directories) != len(configuration.config.cells):
        raise ValueError("cell_directories must contain exactly twelve canonical slot paths")
    cells = tuple(
        _admit_cell(configuration, slot, directory)
        for slot, directory in zip(configuration.config.cells, directories, strict=True)
    )
    _bind_cross_cell_evidence(configuration, cells)
    case = _build_case(configuration, cells)
    first = cells[0]
    profiles = _profiles_primitive(cells)
    subject = _subject_primitive(configuration, cells)
    workload = _workload_primitive(configuration, first)
    layout = tuple(channel.to_primitive() for channel in first.channels)
    return AdmittedRuntimeEvidence(
        configuration=configuration,
        cells=cells,
        case_evidence=case,
        profiles=profiles,
        subject=subject,
        workload=workload,
        channel_layout=layout,
        observation_time_tokens=first.observation_time_tokens,
    )


def _admit_cell(
    configuration: AdmittedRuntimeReviewConfigurationAny,
    slot: RuntimeReviewCellConfig,
    directory: Path,
) -> AdmittedEvidenceCell:
    """Admit one cell checksum-first and bind every decision-bearing field to its slot."""
    members, raw = _read_verified_members(directory)
    result_document = _strict_json_bytes(raw["result.json"], "result.json")
    result_version = result_document.get("schema_version")
    if type(result_version) is not int or result_version not in {1, 2}:
        raise RuntimeEvidenceAdmissionError("worker result schema version is unsupported")
    result = _exact_object(
        result_document,
        _RESULT_KEYS_V1 if result_version == 1 else _RESULT_KEYS_V2,
        "result",
    )
    manifest_echo = _exact_object(
        _strict_manifest_echo_json_bytes(raw["input_manifest.json"]),
        frozenset({"schema", "schema_version", "fixtures"}),
        "input manifest echo",
    )
    manifest_exact = _exact_object(
        _strict_manifest_json_bytes(raw["input_manifest.json"]),
        frozenset({"schema", "schema_version", "fixtures"}),
        "exact input manifest",
    )
    _validate_result_root(configuration, result, manifest_echo, raw, slot)
    fixture_id = _string(result["fixture_id"], "fixture_id")
    fixture_echo = _select_manifest_fixture(manifest_echo, fixture_id)
    fixture_exact = _select_manifest_fixture(manifest_exact, fixture_id)
    _validate_fixture_contract(fixture_exact)
    subject = _admit_subject(configuration, result["subject"], fixture_echo, raw)
    workload = _admit_workload(configuration, result["workload"], fixture_exact)
    profile_id, profile_version, runtime, profile_identity = _admit_runtime(
        configuration, slot, result["runtime"]
    )
    _validate_result_profile_identity(result, runtime)
    (
        observation_times,
        observation_tokens,
        channels,
        contacts,
        trace_sha256,
    ) = _admit_trace(
        configuration,
        slot.step_dt,
        result["trace"],
        result["contacts"],
        fixture_exact,
        directory,
        raw,
    )
    diagnostics, samples = _admit_diagnostics(result["diagnostics"], observation_tokens)
    return AdmittedEvidenceCell(
        slot=slot,
        source_directory=directory,
        members=members,
        schema_version=result_version,
        profile_id=profile_id,
        profile_version=profile_version,
        profile_identity=profile_identity,
        fixture_id=fixture_id,
        contact_event_time_tolerance=_manifest_number_token(
            fixture_exact["contact_event_time_tolerance"], "contact event time tolerance"
        ),
        manifest_raw_sha256=_sha256(result["manifest_raw_sha256"], "manifest_raw_sha256"),
        fixture_raw_sha256=_string(subject["fixture_raw_sha256"], "fixture_raw_sha256"),
        compiled_mjb_sha256=_string(subject["compiled_mjb_sha256"], "compiled_mjb_sha256"),
        runtime_identity_sha256=_string(
            runtime["runtime_identity_sha256"], "runtime_identity_sha256"
        ),
        canonical_trace_sha256=trace_sha256,
        observation_times=observation_times,
        observation_time_tokens=observation_tokens,
        channels=channels,
        contacts=contacts,
        diagnostics=samples,
        finite_values=_boolean(diagnostics["finite_values"], "diagnostics.finite_values"),
        warnings_passed=_boolean(diagnostics["warnings_passed"], "diagnostics.warnings_passed"),
        solver_converged=_boolean(diagnostics["solver_converged"], "diagnostics.solver_converged"),
        subject=subject,
        workload=workload,
        runtime=runtime,
        diagnostic_primitive=diagnostics,
    )


def _read_verified_members(
    directory: Path,
) -> tuple[tuple[EvidenceMember, ...], dict[str, bytes]]:
    """Require the exact regular member set and validate all checksums before parsing."""
    if not isinstance(directory, Path):
        raise TypeError("evidence cell directory must be a Path")
    try:
        directory_mode = os.lstat(directory).st_mode
        names = {entry.name for entry in directory.iterdir()}
    except OSError as exc:
        raise RuntimeEvidenceAdmissionError("evidence cell directory is unavailable") from exc
    if stat.S_ISLNK(directory_mode) or not stat.S_ISDIR(directory_mode):
        raise RuntimeEvidenceAdmissionError("evidence cell must be a nonsymlink directory")
    if names != _MEMBERS:
        raise RuntimeEvidenceAdmissionError(
            f"evidence cell member mismatch; missing={sorted(_MEMBERS - names)}, "
            f"extra={sorted(names - _MEMBERS)}"
        )
    raw: dict[str, bytes] = {}
    for name in sorted(_MEMBERS):
        try:
            raw[name] = read_bounded_regular_file(directory / name, _MEMBER_LIMITS[name])
        except (JsonAdmissionError, OSError) as exc:
            raise RuntimeEvidenceAdmissionError(f"evidence member is unsafe: {name}") from exc
    expected = _parse_checksum_manifest(raw["CHECKSUMS.sha256"])
    for name in _CHECKSUMMED_MEMBERS:
        if hashlib.sha256(raw[name]).hexdigest() != expected[name]:
            raise RuntimeEvidenceAdmissionError(f"evidence checksum mismatch for {name}")
    members = tuple(
        EvidenceMember(name, hashlib.sha256(raw[name]).hexdigest(), len(raw[name]))
        for name in sorted(_MEMBERS)
    )
    return members, raw


def _parse_checksum_manifest(payload: bytes) -> dict[str, str]:
    """Parse the exact ordered five-line lowercase SHA-256 manifest grammar."""
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeEvidenceAdmissionError("CHECKSUMS.sha256 must be ASCII") from exc
    pattern = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)\n")
    matches = pattern.findall(text)
    if "".join(f"{digest}  {name}\n" for digest, name in matches) != text:
        raise RuntimeEvidenceAdmissionError("CHECKSUMS.sha256 has invalid grammar")
    if tuple(name for _digest, name in matches) != _CHECKSUMMED_MEMBERS:
        raise RuntimeEvidenceAdmissionError("CHECKSUMS.sha256 must cover the exact five members")
    return {name: digest for digest, name in matches}


def _strict_json_core(
    payload: bytes,
    label: str,
    limits: JsonAdmissionLimits,
    parse_number: Callable[[str], object],
) -> dict[str, object]:
    """Decode one bounded, duplicate-free UTF-8 evidence object under an explicit profile.

    This is the single parsing authority for runtime-review evidence. Every caller reaches it
    through a fixed wrapper that names both its structural profile and its numeric representation,
    so no profile is ever selected from a default, a label, or an interpreter version.

    The structural bound is applied after parsing and before the root-type check, which is what
    makes the depth policy identical on every supported CPython rather than whatever nesting the
    running parser happens to tolerate.

    Args:
        payload: Exact member bytes, already bound by the per-member byte ceiling.
        label: The evidence member name, used to name the failing document in a refusal.
        limits: The structural profile this member is admitted under.
        parse_number: The exact numeric representation this member retains.

    Returns:
        The admitted object.

    Raises:
        RuntimeEvidenceAdmissionError: The document was not admissible under this profile.
    """

    def object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        """Construct one object while refusing duplicate member names."""
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeEvidenceAdmissionError(f"duplicate JSON member in {label}: {key}")
            result[key] = value
        return result

    def forbidden_constant(token: str) -> object:
        """Reject nonstandard NaN and infinity JSON constants."""
        raise RuntimeEvidenceAdmissionError(f"nonstandard JSON constant in {label}: {token}")

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=object_from_pairs,
            parse_float=parse_number,
            parse_constant=forbidden_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        InvalidOperation,
        ValueError,
        OverflowError,
        RecursionError,
    ) as exc:
        if isinstance(exc, RuntimeEvidenceAdmissionError):
            raise
        raise RuntimeEvidenceAdmissionError(f"strict JSON admission failed for {label}") from exc
    try:
        enforce_json_structure(value, limits)
    except JsonAdmissionError as exc:
        raise RuntimeEvidenceAdmissionError(f"strict JSON admission failed for {label}") from exc
    if type(value) is not dict:
        raise RuntimeEvidenceAdmissionError(f"{label} root must be an object")
    return cast(dict[str, object], value)


def _finite_binary64(label: str) -> Callable[[str], object]:
    """Return the binary64 number reader that refuses a token overflowing to infinity."""

    def finite_float(token: str) -> object:
        """Parse one binary64 JSON token only when it remains finite."""
        value = float(token)
        if not math.isfinite(value):
            raise RuntimeEvidenceAdmissionError(f"non-finite JSON number in {label}")
        return value

    return finite_float


def _strict_json_bytes(payload: bytes, label: str) -> dict[str, object]:
    """Decode one worker result at receipt scale with finite binary64 numbers.

    Worker results are receipt-scale evidence, matching the 64 MiB ``result.json`` byte ceiling in
    ``_MEMBER_LIMITS``. ``_execution`` and ``_execution_output`` also load worker results through
    this wrapper.
    """
    return _strict_json_core(payload, label, RECEIPT_JSON_LIMITS, _finite_binary64(label))


def _strict_manifest_echo_json_bytes(payload: bytes) -> dict[str, object]:
    """Decode the input manifest at manifest scale, retaining finite binary64 numbers.

    This is the reading compared byte-for-byte against the worker's own ``manifest_echo``, so it
    keeps the binary64 representation. It is admitted under the same manifest profile as the exact
    lexical reading below, so the one member has exactly one declared structural bound.
    """
    return _strict_json_core(
        payload,
        "input_manifest.json",
        _MANIFEST_JSON_LIMITS,
        _finite_binary64("input_manifest.json"),
    )


def _strict_manifest_json_bytes(payload: bytes) -> dict[str, object]:
    """Decode the input manifest at manifest scale, retaining lexical decimals.

    Manifest floats become :class:`~decimal.Decimal` so a near-tolerance spelling survives semantic
    checking exactly as written.
    """
    return _strict_json_core(payload, "input_manifest.json", _MANIFEST_JSON_LIMITS, Decimal)


def _validate_result_root(
    configuration: AdmittedRuntimeReviewConfigurationAny,
    result: dict[str, object],
    manifest: dict[str, object],
    raw: Mapping[str, bytes],
    slot: RuntimeReviewCellConfig,
) -> None:
    """Bind the worker result root to exact schema, manifest bytes, and configured slot."""
    if (
        result["schema"] != "metrifid.native_upgrade_worker_result"
        or type(result["schema_version"]) is not int
        or result["status"] != "COMPLETED"
    ):
        raise RuntimeEvidenceAdmissionError("worker result is not a completed supported result")
    result_version = result["schema_version"]
    expected_version = 2 if isinstance(configuration, AdmittedRuntimeReviewConfigurationV2) else 1
    if result_version != expected_version:
        raise RuntimeEvidenceAdmissionError(
            "worker result schema version differs from the explicit configuration route"
        )
    if (
        _sha256(result["manifest_raw_sha256"], "manifest_raw_sha256")
        != hashlib.sha256(raw["input_manifest.json"]).hexdigest()
    ):
        raise RuntimeEvidenceAdmissionError("result does not bind exact input manifest bytes")
    if result["manifest_echo"] != manifest:
        raise RuntimeEvidenceAdmissionError("manifest_echo differs from input_manifest.json")
    _validate_configured_result_profile(configuration, result, slot, result_version)
    if (
        result["step_dt"] != slot.step_dt
        or type(result["repeat_id"]) is not int
        or result["repeat_id"] != slot.repeat_id
    ):
        raise RuntimeEvidenceAdmissionError(
            "worker step/repeat identity differs from configured slot"
        )
    limitations = _sequence(result["limitations"], "limitations")
    if tuple(limitations) != _LIMITATIONS:
        raise RuntimeEvidenceAdmissionError("worker limitation vocabulary is not frozen")
    if (
        manifest["schema"] != "metrifid.native_upgrade_manifest"
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
    ):
        raise RuntimeEvidenceAdmissionError("input manifest schema/version is unsupported")


def _validate_configured_result_profile(
    configuration: AdmittedRuntimeReviewConfigurationAny,
    result: Mapping[str, object],
    slot: RuntimeReviewCellConfig,
    result_version: object,
) -> None:
    """Bind legacy or role-based result profile fields through its explicit route."""
    if result_version == 1:
        if isinstance(configuration, AdmittedRuntimeReviewConfigurationV2):
            raise RuntimeEvidenceAdmissionError("legacy worker result requires v1 configuration")
        declared = (
            configuration.config.baseline_profile
            if slot.profile_role == "baseline"
            else configuration.config.candidate_profile
        )
        if (
            result["profile_id"] != declared.profile_id
            or result["profile_version"] != declared.mujoco_version
        ):
            raise RuntimeEvidenceAdmissionError(
                "worker result profile identity differs from configured role"
            )
        return
    if not isinstance(configuration, AdmittedRuntimeReviewConfigurationV2):
        raise RuntimeEvidenceAdmissionError("role-based worker result requires v2 configuration")
    declared_v2 = (
        configuration.config.baseline_profile
        if slot.profile_role == "baseline"
        else configuration.config.candidate_profile
    )
    expected = {
        "profile_role": declared_v2.profile_role,
        "package_version": declared_v2.package_version,
        "native_version": declared_v2.native_version,
        "native_version_integer": declared_v2.native_version_integer,
        "profile_identity_sha256": declared_v2.profile_identity_sha256,
    }
    if result["profile_role"] != slot.profile_role or any(
        result[field] != value for field, value in expected.items()
    ):
        raise RuntimeEvidenceAdmissionError(
            "worker result role or exact native identity differs from configuration"
        )


def _validate_result_profile_identity(
    result: Mapping[str, object], runtime: Mapping[str, CanonicalValue]
) -> None:
    """Bind redundant result-root profile claims to the admitted runtime identity."""
    if runtime["schema_version"] == 1:
        consistent = (
            result["profile_id"] == runtime["profile_id"]
            and result["profile_version"] == runtime["profile_version"]
        )
    else:
        fields = (
            "profile_role",
            "package_version",
            "native_version",
            "native_version_integer",
            "profile_identity_sha256",
            "runtime_identity_sha256",
            "sentinel_identity_sha256",
        )
        consistent = all(result[field] == runtime[field] for field in fields)
    if not consistent:
        raise RuntimeEvidenceAdmissionError(
            "worker result profile identity differs from admitted runtime"
        )


def _select_manifest_fixture(manifest: dict[str, object], fixture_id: str) -> dict[str, object]:
    """Select one fixture only after bounding the manifest and proving unique IDs."""
    fixtures = [
        _exact_object(item, _FIXTURE_KEYS, "manifest fixture")
        for item in _sequence(manifest["fixtures"], "manifest.fixtures")
    ]
    if not 1 <= len(fixtures) <= 64:
        raise RuntimeEvidenceAdmissionError(
            "manifest fixtures must contain between one and 64 declarations"
        )
    fixture_ids = [
        _manifest_identifier(item["fixture_id"], "manifest fixture_id") for item in fixtures
    ]
    if len(fixture_ids) != len(set(fixture_ids)):
        raise RuntimeEvidenceAdmissionError("manifest fixture IDs must be unique")
    selected = [item for item in fixtures if item["fixture_id"] == fixture_id]
    if len(selected) != 1:
        raise RuntimeEvidenceAdmissionError(
            "result fixture_id does not select one manifest fixture"
        )
    return selected[0]


def _validate_fixture_contract(fixture: Mapping[str, object]) -> None:
    """Mirror the worker's frozen selected-fixture and physical campaign contract."""
    fixture_id = _manifest_identifier(fixture["fixture_id"], "manifest fixture_id")
    role = _manifest_bounded_string(fixture["campaign_role"], "manifest campaign_role")
    if role not in {"MIGRATION_SUBJECT", "CONTROLLED_MUTATION"}:
        raise RuntimeEvidenceAdmissionError("manifest campaign_role is unsupported")
    baseline_id = _manifest_identifier(
        fixture["baseline_fixture_id"], "manifest baseline_fixture_id"
    )
    if role == "MIGRATION_SUBJECT" and baseline_id != fixture_id:
        raise RuntimeEvidenceAdmissionError(
            "manifest migration subject must identify itself as its baseline"
        )
    if role == "CONTROLLED_MUTATION" and baseline_id == fixture_id:
        raise RuntimeEvidenceAdmissionError(
            "manifest controlled mutation requires a distinct baseline fixture ID"
        )
    _manifest_relative_xml_path(fixture["xml_path"])
    for key in ("initial_qpos", "initial_qvel"):
        values = _bounded_manifest_sequence(fixture[key], f"manifest {key}", maximum=4096)
        numbers = tuple(_manifest_decimal(item, f"manifest {key}[]") for item in values)
        if not numbers or any(abs(item) > Decimal("1000000") for item in numbers):
            raise RuntimeEvidenceAdmissionError(f"manifest {key} must be nonempty and bounded")

    control_dt = _manifest_decimal(fixture["control_dt"], "manifest control_dt")
    horizon = _manifest_decimal(fixture["horizon"], "manifest horizon")
    step_dts = tuple(
        _manifest_decimal(item, "manifest step_dts[]")
        for item in _bounded_manifest_sequence(fixture["step_dts"], "manifest step_dts", maximum=8)
    )
    fixed_values = (
        (control_dt, _FIXED_CONTROL_DT, "control_dt"),
        (horizon, _FIXED_HORIZON, "horizon"),
        (
            _manifest_decimal(fixture["continuous_tolerance"], "manifest continuous_tolerance"),
            _FIXED_CONTINUOUS_TOLERANCE,
            "continuous_tolerance",
        ),
        (
            _manifest_decimal(fixture["so3_tolerance"], "manifest so3_tolerance"),
            _FIXED_SO3_TOLERANCE,
            "so3_tolerance",
        ),
        (
            _manifest_decimal(
                fixture["contact_event_time_tolerance"],
                "manifest contact_event_time_tolerance",
            ),
            _FIXED_CONTACT_EVENT_TIME_TOLERANCE,
            "contact_event_time_tolerance",
        ),
        (
            _manifest_decimal(fixture["minimum_common_prefix"], "manifest minimum_common_prefix"),
            _FIXED_MINIMUM_COMMON_PREFIX,
            "minimum_common_prefix",
        ),
    )
    for actual, expected, field in fixed_values:
        if actual != expected:
            raise RuntimeEvidenceAdmissionError(
                f"manifest {field} differs from the frozen runtime-review contract"
            )
    if step_dts != _FIXED_STEP_DTS:
        raise RuntimeEvidenceAdmissionError(
            "manifest step_dts differ from the frozen runtime-review contract"
        )

    controls = tuple(
        _validate_manifest_control(item, index)
        for index, item in enumerate(
            _bounded_manifest_sequence(fixture["controls"], "manifest controls", maximum=256)
        )
    )
    _validate_manifest_control_grid(controls, control_dt, horizon, step_dts)
    channels = tuple(
        _validate_manifest_channel(item, index)
        for index, item in enumerate(
            _bounded_manifest_sequence(fixture["channels"], "manifest channels", maximum=256)
        )
    )
    contacts = tuple(
        _validate_manifest_contact_pair(item, index)
        for index, item in enumerate(
            _bounded_manifest_sequence(
                fixture["contact_pairs"], "manifest contact_pairs", maximum=64
            )
        )
    )
    projected_ids = [item[0] for item in channels]
    for contact_id, _geom_names in contacts:
        projected_ids.extend(
            (
                f"{contact_id}.occupancy",
                f"{contact_id}.normal_force",
                f"{contact_id}.normal_impulse",
            )
        )
    if len(projected_ids) != len(set(projected_ids)):
        raise RuntimeEvidenceAdmissionError(
            "manifest channel IDs must be unique, including derived contact channels"
        )
    geom_pairs = [item[1] for item in contacts]
    if len(geom_pairs) != len(set(geom_pairs)):
        raise RuntimeEvidenceAdmissionError(
            "manifest declared semantic contact pairs must be unique"
        )


ManifestControl = tuple[Decimal, Decimal, tuple[Decimal, ...]]


def _validate_manifest_control(value: object, index: int) -> ManifestControl:
    """Admit one bounded exact-decimal manifest control segment."""
    segment = _exact_object(
        value, frozenset({"start", "end", "values"}), f"manifest controls[{index}]"
    )
    start = _manifest_decimal(segment["start"], f"manifest controls[{index}].start")
    end = _manifest_decimal(segment["end"], f"manifest controls[{index}].end")
    values = tuple(
        _manifest_decimal(item, f"manifest controls[{index}].values[]")
        for item in _bounded_manifest_sequence(
            segment["values"], f"manifest controls[{index}].values", maximum=256
        )
    )
    if not values or any(abs(item) > Decimal("1000000") for item in values):
        raise RuntimeEvidenceAdmissionError("manifest control values must be nonempty and bounded")
    if not Decimal("0") <= start < end <= _FIXED_HORIZON:
        raise RuntimeEvidenceAdmissionError("manifest control segment bounds are invalid")
    return start, end, values


def _validate_manifest_control_grid(
    controls: tuple[ManifestControl, ...],
    control_dt: Decimal,
    horizon: Decimal,
    step_dts: tuple[Decimal, ...],
) -> None:
    """Require exact grid divisibility, full coverage, ordering, and actuator width."""
    if not _manifest_divides(horizon, control_dt):
        raise RuntimeEvidenceAdmissionError("manifest control grid must divide the horizon")
    if any(
        not _manifest_divides(control_dt, step_dt) or not _manifest_divides(horizon, step_dt)
        for step_dt in step_dts
    ):
        raise RuntimeEvidenceAdmissionError(
            "every manifest step grid must divide control_dt and horizon"
        )
    if not controls or controls[0][0] != 0 or controls[-1][1] != horizon:
        raise RuntimeEvidenceAdmissionError("manifest control segments must cover the full horizon")
    for index, (start, end, _values) in enumerate(controls):
        if index and controls[index - 1][1] != start:
            raise RuntimeEvidenceAdmissionError(
                "manifest control segments must be contiguous and ordered"
            )
        if any(not _manifest_divides(boundary, control_dt) for boundary in (start, end)):
            raise RuntimeEvidenceAdmissionError(
                "every manifest control boundary must land on the control grid"
            )
        if any(
            not _manifest_divides(boundary, step_dt)
            for boundary in (start, end)
            for step_dt in step_dts
        ):
            raise RuntimeEvidenceAdmissionError(
                "every manifest control boundary must land on every simulation grid"
            )
    if len({len(values) for _start, _end, values in controls}) != 1:
        raise RuntimeEvidenceAdmissionError(
            "every manifest control segment must have the same actuator width"
        )


def _validate_manifest_channel(value: object, index: int) -> tuple[str, str]:
    """Admit one bounded scalar or quaternion manifest channel declaration."""
    channel = _exact_object(
        value,
        frozenset({"channel_id", "kind", "object_name", "component", "scale"}),
        f"manifest channels[{index}]",
    )
    channel_id = _manifest_identifier(
        channel["channel_id"], f"manifest channels[{index}].channel_id"
    )
    kind = _manifest_bounded_string(channel["kind"], f"manifest channels[{index}].kind")
    if kind not in _MANIFEST_CHANNEL_KINDS:
        raise RuntimeEvidenceAdmissionError("manifest channel kind is unsupported")
    _manifest_bounded_string(channel["object_name"], f"manifest channels[{index}].object_name")
    component_value = channel["component"]
    component = (
        None
        if component_value is None
        else _bounded_manifest_integer(
            component_value, f"manifest channels[{index}].component", 0, 255
        )
    )
    scale_value = channel["scale"]
    scale = (
        None
        if scale_value is None
        else _manifest_decimal(scale_value, f"manifest channels[{index}].scale")
    )
    if kind == "BODY_QUATERNION":
        if component is not None or scale is not None:
            raise RuntimeEvidenceAdmissionError(
                "manifest quaternion channels require null component and scale"
            )
    elif scale is None or scale <= 0:
        raise RuntimeEvidenceAdmissionError(
            "manifest continuous scalar channels require a positive scale"
        )
    if kind in {"JOINT_POSITION", "JOINT_VELOCITY"} and component is not None:
        raise RuntimeEvidenceAdmissionError(
            "manifest joint scalar channels require a null component"
        )
    if kind == "BODY_POSITION" and component not in {0, 1, 2}:
        raise RuntimeEvidenceAdmissionError(
            "manifest body-position channels require component 0, 1, or 2"
        )
    if kind == "SENSOR" and component is None:
        raise RuntimeEvidenceAdmissionError("manifest sensor channels require a component")
    return channel_id, kind


def _validate_manifest_contact_pair(value: object, index: int) -> tuple[str, tuple[str, str]]:
    """Admit one bounded contact declaration and its sorted semantic geometry pair."""
    contact = _exact_object(
        value,
        frozenset({"channel_id", "geom_a", "geom_b", "force_scale", "impulse_scale"}),
        f"manifest contact_pairs[{index}]",
    )
    channel_id = _manifest_identifier(
        contact["channel_id"], f"manifest contact_pairs[{index}].channel_id"
    )
    geom_names = tuple(
        sorted(
            (
                _manifest_bounded_string(
                    contact["geom_a"], f"manifest contact_pairs[{index}].geom_a"
                ),
                _manifest_bounded_string(
                    contact["geom_b"], f"manifest contact_pairs[{index}].geom_b"
                ),
            )
        )
    )
    if geom_names[0] == geom_names[1]:
        raise RuntimeEvidenceAdmissionError("manifest contact geom names must be distinct")
    if (
        _manifest_decimal(contact["force_scale"], f"manifest contact_pairs[{index}].force_scale")
        <= 0
        or _manifest_decimal(
            contact["impulse_scale"], f"manifest contact_pairs[{index}].impulse_scale"
        )
        <= 0
    ):
        raise RuntimeEvidenceAdmissionError(
            "manifest contact force and impulse scales must be positive"
        )
    return channel_id, cast(tuple[str, str], geom_names)


def _admit_subject(
    configuration: AdmittedRuntimeReviewConfigurationAny,
    value: object,
    fixture: dict[str, object],
    raw: Mapping[str, bytes],
) -> dict[str, CanonicalValue]:
    """Bind expected subject identities to manifest, fixture, closure, and compiled bytes."""
    _validate_self_contained_fixture_xml(raw["fixture.xml"])
    subject = _exact_object(
        value,
        frozenset(
            {
                "fixture_id",
                "campaign_role",
                "baseline_fixture_id",
                "fixture_manifest_sha256",
                "fixture_raw_sha256",
                "source_closure",
                "compiled_mjb_sha256",
                "compiled_mjb_size_bytes",
            }
        ),
        "subject",
    )
    expected = configuration.config.expected_subject
    if subject["fixture_id"] != expected.fixture_id or fixture["fixture_id"] != expected.fixture_id:
        raise RuntimeEvidenceAdmissionError("cell fixture identity differs from expected_subject")
    fixture_manifest_sha = hashlib.sha256(_worker_json_bytes(fixture)).hexdigest()
    if (
        _sha256(subject["fixture_manifest_sha256"], "fixture_manifest_sha256")
        != expected.fixture_manifest_sha256
        or fixture_manifest_sha != expected.fixture_manifest_sha256
    ):
        raise RuntimeEvidenceAdmissionError(
            "fixture manifest identity differs from expected_subject"
        )
    fixture_sha = hashlib.sha256(raw["fixture.xml"]).hexdigest()
    compiled_sha = hashlib.sha256(raw["model.mjb"]).hexdigest()
    if (
        _sha256(subject["fixture_raw_sha256"], "fixture_raw_sha256") != fixture_sha
        or _sha256(subject["compiled_mjb_sha256"], "compiled_mjb_sha256") != compiled_sha
        or _integer(subject["compiled_mjb_size_bytes"], "compiled_mjb_size_bytes", minimum=1)
        != len(raw["model.mjb"])
    ):
        raise RuntimeEvidenceAdmissionError("subject does not bind copied fixture/MJB bytes")
    if (
        subject["campaign_role"] != fixture["campaign_role"]
        or subject["baseline_fixture_id"] != fixture["baseline_fixture_id"]
    ):
        raise RuntimeEvidenceAdmissionError("subject role/baseline differs from manifest")
    closure = _exact_object(
        subject["source_closure"],
        frozenset(
            {"schema", "schema_version", "entrypoint", "member_count", "members", "closure_sha256"}
        ),
        "source_closure",
    )
    _validate_source_closure(
        closure, fixture, fixture_sha, len(raw["fixture.xml"]), expected.source_closure_sha256
    )
    return _canonical_mapping(subject, "subject")


def _validate_source_closure(
    closure: dict[str, object],
    fixture: Mapping[str, object],
    fixture_sha256: str,
    fixture_size: int,
    expected_sha256: str,
) -> None:
    """Recompute the one-member self-contained source-closure identity."""
    members = _sequence(closure["members"], "source_closure.members")
    if len(members) != 1:
        raise RuntimeEvidenceAdmissionError("source closure must contain exactly one member")
    member = _exact_object(
        members[0], frozenset({"path", "sha256", "size_bytes"}), "source_closure member"
    )
    entrypoint = _string(closure["entrypoint"], "source_closure.entrypoint")
    if (
        closure["schema"] != "metrifid.native_upgrade_source_closure"
        or closure["schema_version"] != 1
        or closure["member_count"] != 1
        or entrypoint != fixture["xml_path"]
        or member["path"] != entrypoint
        or member["sha256"] != fixture_sha256
        or member["size_bytes"] != fixture_size
    ):
        raise RuntimeEvidenceAdmissionError("source closure does not bind exact fixture bytes")
    claimed = _sha256(closure["closure_sha256"], "source_closure.closure_sha256")
    if claimed != expected_sha256 or _self_hash(closure, "closure_sha256") != claimed:
        raise RuntimeEvidenceAdmissionError(
            "source closure self-hash differs from expected_subject"
        )


def _validate_self_contained_fixture_xml(raw: bytes) -> None:
    """Mirror the worker's bounded strict-UTF-8 self-contained MJCF admission."""
    if len(raw) > _MAX_FIXTURE_XML_BYTES:
        raise RuntimeEvidenceAdmissionError("fixture XML exceeds the one-MiB worker bound")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RuntimeEvidenceAdmissionError("fixture XML must be strict UTF-8") from exc
    upper = text.upper()
    if "<!DOCTYPE" in upper or "<!ENTITY" in upper:
        raise RuntimeEvidenceAdmissionError("fixture XML must not declare a DTD or entity")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise RuntimeEvidenceAdmissionError("fixture XML must be well-formed") from exc
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in {"include", "plugin"}:
            raise RuntimeEvidenceAdmissionError("fixture XML includes and plugins are forbidden")
        if {"file", "meshdir", "texturedir", "assetdir"}.intersection(element.attrib):
            raise RuntimeEvidenceAdmissionError(
                "fixture XML must not reference external files or asset directories"
            )


def _admit_workload(
    configuration: AdmittedRuntimeReviewConfigurationAny,
    value: object,
    fixture: Mapping[str, object],
) -> dict[str, CanonicalValue]:
    """Recompute and bind workload, initial-state, and action-program identities."""
    workload = _exact_object(
        value,
        frozenset(
            {"schema", "schema_version", "initial_state", "action_program", "semantic_sha256"}
        ),
        "workload",
    )
    if (
        workload["schema"] != "metrifid.native_upgrade_workload_identity"
        or workload["schema_version"] != 1
    ):
        raise RuntimeEvidenceAdmissionError("workload schema/version is unsupported")
    initial = _exact_object(
        workload["initial_state"],
        frozenset({"schema", "schema_version", "qpos", "qvel", "semantic_sha256"}),
        "initial_state",
    )
    action = _exact_object(
        workload["action_program"],
        frozenset(
            {
                "schema",
                "schema_version",
                "action_semantics",
                "control_dt",
                "horizon",
                "segments",
                "semantic_sha256",
            }
        ),
        "action_program",
    )
    _validate_initial_state(initial, fixture)
    _validate_action_program(action, fixture)
    expected = configuration.config.expected_workload
    if (
        _self_hash(initial, "semantic_sha256") != expected.initial_state_semantic_sha256
        or initial["semantic_sha256"] != expected.initial_state_semantic_sha256
        or _self_hash(action, "semantic_sha256") != expected.action_program_semantic_sha256
        or action["semantic_sha256"] != expected.action_program_semantic_sha256
        or _self_hash(workload, "semantic_sha256") != expected.semantic_sha256
        or workload["semantic_sha256"] != expected.semantic_sha256
    ):
        raise RuntimeEvidenceAdmissionError("workload identity differs from expected_workload")
    return _canonical_mapping(workload, "workload")


def _validate_initial_state(initial: Mapping[str, object], fixture: Mapping[str, object]) -> None:
    """Bind canonical initial-state tokens to the manifest's genuine JSON numbers."""
    if (
        initial["schema"] != "metrifid.native_upgrade_initial_state"
        or initial["schema_version"] != 1
    ):
        raise RuntimeEvidenceAdmissionError("initial-state schema/version is unsupported")
    for result_key, fixture_key in (("qpos", "initial_qpos"), ("qvel", "initial_qvel")):
        tokens = tuple(
            _decimal_token(item, f"initial_state.{result_key}[]")
            for item in _sequence(initial[result_key], f"initial_state.{result_key}")
        )
        manifest_tokens = tuple(
            _manifest_number_token(item, f"manifest.{fixture_key}[]")
            for item in _sequence(fixture[fixture_key], f"manifest.{fixture_key}")
        )
        if not tokens or tokens != manifest_tokens:
            raise RuntimeEvidenceAdmissionError("initial state differs from manifest")


def _validate_action_program(action: Mapping[str, object], fixture: Mapping[str, object]) -> None:
    """Bind exact left-boundary action semantics to the manifest controls and horizon."""
    if (
        action["schema"] != "metrifid.native_upgrade_action_program"
        or action["schema_version"] != 1
        or action["action_semantics"] != "LEFT_BOUNDARY_ZERO_ORDER_HOLD"
    ):
        raise RuntimeEvidenceAdmissionError("action-program semantics are unsupported")
    for key in ("control_dt", "horizon"):
        if _decimal_token(action[key], f"action_program.{key}") != _manifest_number_token(
            fixture[key], f"manifest.{key}"
        ):
            raise RuntimeEvidenceAdmissionError(f"action-program {key} differs from manifest")
    actual = [_action_segment(item) for item in _sequence(action["segments"], "action segments")]
    expected = [
        _manifest_action_segment(item)
        for item in _sequence(fixture["controls"], "manifest controls")
    ]
    if not actual or actual != expected:
        raise RuntimeEvidenceAdmissionError("action-program segments differ from manifest")


def _action_segment(value: object) -> dict[str, object]:
    """Admit one canonical result-side action segment."""
    segment = _exact_object(value, frozenset({"start", "end", "values"}), "action segment")
    return {
        "start": _decimal_token(segment["start"], "action segment start"),
        "end": _decimal_token(segment["end"], "action segment end"),
        "values": tuple(
            _decimal_token(item, "action segment value")
            for item in _sequence(segment["values"], "action segment values")
        ),
    }


def _manifest_action_segment(value: object) -> dict[str, object]:
    """Project one manifest action segment into canonical decimal tokens."""
    segment = _exact_object(value, frozenset({"start", "end", "values"}), "manifest segment")
    return {
        "start": _manifest_number_token(segment["start"], "manifest segment start"),
        "end": _manifest_number_token(segment["end"], "manifest segment end"),
        "values": tuple(
            _manifest_number_token(item, "manifest segment value")
            for item in _sequence(segment["values"], "manifest segment values")
        ),
    }


def _admit_runtime(
    configuration: AdmittedRuntimeReviewConfigurationAny,
    slot: RuntimeReviewCellConfig,
    value: object,
) -> tuple[
    str,
    str,
    dict[str, CanonicalValue],
    dict[str, CanonicalValue] | None,
]:
    """Self-hash and bind one exact live runtime identity to its declared profile role."""
    if type(value) is not dict:
        raise RuntimeEvidenceAdmissionError("runtime must be an object")
    runtime_version = cast(dict[str, object], value).get("schema_version")
    if type(runtime_version) is not int:
        raise RuntimeEvidenceAdmissionError("runtime schema version must be an integer")
    if runtime_version == 1:
        if not isinstance(configuration, AdmittedRuntimeReviewConfiguration):
            raise RuntimeEvidenceAdmissionError("legacy runtime requires a v1 configuration")
        profile_id, profile_version, runtime = _admit_runtime_v1(configuration, slot, value)
        return profile_id, profile_version, runtime, None
    if runtime_version == 2:
        if not isinstance(configuration, AdmittedRuntimeReviewConfigurationV2):
            raise RuntimeEvidenceAdmissionError("role-based runtime requires a v2 configuration")
        return _admit_runtime_v2(configuration, slot, value)
    raise RuntimeEvidenceAdmissionError("runtime schema version is unsupported")


def _admit_runtime_v1(
    configuration: AdmittedRuntimeReviewConfiguration,
    slot: RuntimeReviewCellConfig,
    value: object,
) -> tuple[str, str, dict[str, CanonicalValue]]:
    """Admit an immutable historical runtime identity without v2 normalization."""
    runtime = _exact_object(
        value,
        frozenset(
            {
                "schema",
                "schema_version",
                "profile_id",
                "profile_version",
                "python",
                "host",
                "thread_environment",
                "mujoco",
                "numpy",
                "installation",
                "pip_check",
                "external_profile_identity",
                "runtime_identity_sha256",
            }
        ),
        "runtime",
    )
    declared = (
        configuration.config.baseline_profile
        if slot.profile_role == "baseline"
        else configuration.config.candidate_profile
    )
    profile_id = _string(runtime["profile_id"], "runtime.profile_id")
    profile_version = _string(runtime["profile_version"], "runtime.profile_version")
    if (
        runtime["schema"] != "metrifid.native_upgrade_runtime_identity"
        or runtime["schema_version"] != 1
        or profile_id != declared.profile_id
        or profile_version != declared.mujoco_version
    ):
        raise RuntimeEvidenceAdmissionError("runtime profile identity differs from configured role")
    claimed = _sha256(runtime["runtime_identity_sha256"], "runtime_identity_sha256")
    canonical_runtime = _canonical_mapping(runtime, "runtime")
    if _self_hash(canonical_runtime, "runtime_identity_sha256") != claimed:
        raise RuntimeEvidenceAdmissionError("runtime identity self-hash is inconsistent")
    _validate_runtime_python(runtime["python"])
    _validate_runtime_host(runtime["host"])
    if (
        _exact_string_mapping(runtime["thread_environment"], "thread_environment")
        != _THREAD_ENVIRONMENT
    ):
        raise RuntimeEvidenceAdmissionError("runtime deterministic thread environment differs")
    _validate_runtime_mujoco(runtime["mujoco"], profile_version)
    _validate_runtime_numpy(runtime["numpy"])
    if type(runtime["installation"]) is not dict:
        raise RuntimeEvidenceAdmissionError("runtime installation evidence must be an object")
    _validate_pip_check(runtime["pip_check"])
    external = _exact_object(
        runtime["external_profile_identity"],
        frozenset({"available", "raw_sha256", "profile_identity_sha256"}),
        "external_profile_identity",
    )
    if external["available"] is not True:
        raise RuntimeEvidenceAdmissionError("runtime lacks retained external-profile identity")
    _sha256(external["raw_sha256"], "external_profile_identity.raw_sha256")
    _sha256(
        external["profile_identity_sha256"], "external_profile_identity.profile_identity_sha256"
    )
    return profile_id, profile_version, canonical_runtime


def _admit_runtime_v2(
    configuration: AdmittedRuntimeReviewConfigurationV2,
    slot: RuntimeReviewCellConfig,
    value: object,
) -> tuple[str, str, dict[str, CanonicalValue], dict[str, CanonicalValue]]:
    """Admit one role-based runtime and bind it to exact preflight identity evidence."""
    runtime = _exact_object(
        value,
        frozenset(
            {
                "schema",
                "schema_version",
                "profile_role",
                "package_version",
                "native_version",
                "native_version_integer",
                "support_tier",
                "python",
                "host",
                "thread_environment",
                "mujoco",
                "numpy",
                "installation",
                "pip_check",
                "external_profile_identity",
                "worker_sha256",
                "profile_identity_sha256",
                "sentinel_identity_sha256",
                "runtime_identity_sha256",
            }
        ),
        "runtime",
    )
    declared = (
        configuration.config.baseline_profile
        if slot.profile_role == "baseline"
        else configuration.config.candidate_profile
    )
    if (
        runtime["schema"] != "metrifid.native_upgrade_runtime_identity"
        or runtime["schema_version"] != 2
        or runtime["profile_role"] != declared.profile_role
        or runtime["package_version"] != declared.package_version
        or runtime["native_version"] != declared.native_version
        or runtime["native_version_integer"] != declared.native_version_integer
        or runtime["profile_identity_sha256"] != declared.profile_identity_sha256
        or runtime["support_tier"] != "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE"
        or runtime["worker_sha256"] != _FROZEN_WORKER_SHA256
    ):
        raise RuntimeEvidenceAdmissionError(
            "role-based runtime identity differs from its configured exact profile"
        )
    claimed = _sha256(runtime["runtime_identity_sha256"], "runtime_identity_sha256")
    canonical_runtime = _canonical_mapping(runtime, "runtime")
    if _self_hash(canonical_runtime, "runtime_identity_sha256") != claimed:
        raise RuntimeEvidenceAdmissionError("runtime identity self-hash is inconsistent")
    _validate_runtime_python(runtime["python"])
    _validate_runtime_host_v2(runtime["host"])
    if (
        _exact_string_mapping(runtime["thread_environment"], "thread_environment")
        != _THREAD_ENVIRONMENT
    ):
        raise RuntimeEvidenceAdmissionError("runtime deterministic thread environment differs")
    _validate_runtime_mujoco_v2(
        runtime["mujoco"],
        declared.package_version,
        declared.native_version,
        declared.native_version_integer,
    )
    _validate_runtime_numpy_v2(runtime["numpy"])
    _validate_runtime_installation_v2(runtime["installation"])
    _validate_pip_check(runtime["pip_check"])
    external = _exact_object(
        runtime["external_profile_identity"],
        frozenset({"available", "raw_sha256", "profile_identity_sha256"}),
        "external_profile_identity",
    )
    if (
        external["available"] is not True
        or _sha256(external["raw_sha256"], "external_profile_identity.raw_sha256")
        != configuration.profile_identity_file_hash(slot.profile_role)
        or _sha256(
            external["profile_identity_sha256"],
            "external_profile_identity.profile_identity_sha256",
        )
        != declared.profile_identity_sha256
    ):
        raise RuntimeEvidenceAdmissionError(
            "runtime external-profile binding differs from the admitted identity file"
        )
    try:
        admitted_identity = load_native_profile_identity_v2(
            configuration.profile_identity_path(slot.profile_role),
            expected_profile_role=slot.profile_role,
            expected_profile_identity_sha256=declared.profile_identity_sha256,
            expected_worker_sha256=_FROZEN_WORKER_SHA256,
        )
    except ProfileIdentityRefusal as exc:
        raise RuntimeEvidenceAdmissionError(
            "runtime profile identity evidence is not admissible"
        ) from exc
    canonical_identity = _canonical_mapping(admitted_identity, "profile identity")
    _bind_runtime_to_profile_identity_v2(canonical_runtime, canonical_identity)
    return slot.profile_role, declared.package_version, canonical_runtime, canonical_identity


def _bind_runtime_to_profile_identity_v2(
    runtime: Mapping[str, CanonicalValue], identity: Mapping[str, CanonicalValue]
) -> None:
    """Require every decision-bearing runtime fact to equal its preflight identity binding."""
    direct_fields = (
        "profile_role",
        "package_version",
        "native_version",
        "native_version_integer",
        "support_tier",
        "profile_identity_sha256",
    )
    if any(runtime[field] != identity[field] for field in direct_fields):
        raise RuntimeEvidenceAdmissionError("runtime differs from its preflight profile identity")
    runtime_python = cast(Mapping[str, CanonicalValue], runtime["python"])
    identity_python = cast(Mapping[str, CanonicalValue], identity["python"])
    if any(runtime_python[field] != identity_python[field] for field in runtime_python):
        raise RuntimeEvidenceAdmissionError("runtime Python differs from preflight identity")
    runtime_mujoco = cast(Mapping[str, CanonicalValue], runtime["mujoco"])
    identity_mujoco = cast(Mapping[str, CanonicalValue], identity["mujoco"])
    if (
        _compact_distribution(runtime_mujoco["distribution"]) != identity_mujoco["distribution"]
        or runtime_mujoco["loaded_native_library"] != identity_mujoco["loaded_native_library"]
    ):
        raise RuntimeEvidenceAdmissionError("runtime MuJoCo differs from preflight identity")
    runtime_numpy = cast(Mapping[str, CanonicalValue], runtime["numpy"])
    identity_numpy = cast(Mapping[str, CanonicalValue], identity["numpy"])
    if (
        runtime_numpy["python_version"] != identity_numpy["python_version"]
        or _compact_distribution(runtime_numpy["distribution"]) != identity_numpy["distribution"]
    ):
        raise RuntimeEvidenceAdmissionError("runtime NumPy differs from preflight identity")
    sentinel = cast(Mapping[str, CanonicalValue], identity["sentinel"])
    if (
        runtime["host"] != identity["host"]
        or runtime["thread_environment"] != identity["environment"]
        or runtime["installation"] != identity["installation"]
        or runtime["pip_check"] != identity["pip_check"]
        or runtime["worker_sha256"]
        != cast(Mapping[str, CanonicalValue], identity["profile_contract"])["worker_sha256"]
        or runtime["sentinel_identity_sha256"] != sentinel["sentinel_identity_sha256"]
        or sentinel["status"] != "PASS"
    ):
        raise RuntimeEvidenceAdmissionError("runtime bindings differ from preflight identity")


def _compact_distribution(value: CanonicalValue) -> dict[str, CanonicalValue]:
    """Project a full worker distribution to the exact compact preflight identity shape."""
    distribution = cast(Mapping[str, CanonicalValue], value)
    fields = (
        "name",
        "version",
        "member_count",
        "payload_identity_algorithm",
        "payload_sha256",
        "record_bound_identity_algorithm",
        "record_bound_member_count",
        "record_bound_payload_sha256",
        "record_declared_sha256_member_count",
        "record_unhashed_member_count",
    )
    return {field: distribution[field] for field in fields}


def _validate_runtime_python(value: object) -> None:
    """Admit the exact worker-side Python build projection."""
    python = _exact_object(
        value,
        frozenset(
            {
                "executable",
                "resolved_executable",
                "resolved_executable_sha256",
                "version",
                "version_full",
                "implementation",
                "implementation_name",
                "compiler",
                "cache_tag",
            }
        ),
        "runtime.python",
    )
    for key in (
        "executable",
        "resolved_executable",
        "version",
        "version_full",
        "implementation",
        "implementation_name",
        "compiler",
        "cache_tag",
    ):
        _string(python[key], f"runtime.python.{key}")
    if (
        not Path(cast(str, python["executable"])).is_absolute()
        or not Path(cast(str, python["resolved_executable"])).is_absolute()
    ):
        raise RuntimeEvidenceAdmissionError("runtime Python locators must be absolute")
    _sha256(python["resolved_executable_sha256"], "resolved_executable_sha256")


def _validate_runtime_host(value: object) -> None:
    """Admit the exact live host/OS/architecture/CPU projection."""
    _exact_object(
        value,
        frozenset(
            {
                "system",
                "release",
                "version",
                "platform",
                "machine",
                "architecture",
                "logical_cpu_count",
                "cpu_model",
                "cpu_model_source",
                "hardware_model",
                "hardware_profile",
                "physical_cpu_count",
                "hyper_threading_technology",
            }
        ),
        "runtime.host",
    )


def _validate_runtime_host_v2(value: object) -> None:
    """Admit the live host projection with exact libc identity."""
    host = _exact_object(
        value,
        frozenset(
            {
                "system",
                "release",
                "version",
                "platform",
                "machine",
                "architecture",
                "libc",
                "logical_cpu_count",
                "cpu_model",
                "cpu_model_source",
                "hardware_model",
                "hardware_profile",
                "physical_cpu_count",
                "hyper_threading_technology",
            }
        ),
        "runtime.host",
    )
    libc = _sequence(host["libc"], "runtime.host.libc")
    if len(libc) != 2 or any(type(item) is not str for item in libc):
        raise RuntimeEvidenceAdmissionError("runtime host libc identity must contain two strings")


def _validate_runtime_mujoco(value: object, profile_version: str) -> None:
    """Admit the intentional MuJoCo distribution/native-library profile difference."""
    mujoco = _exact_object(
        value,
        frozenset(
            {
                "python_version",
                "native_version",
                "version_integer",
                "distribution",
                "loaded_native_library",
            }
        ),
        "runtime.mujoco",
    )
    expected_integer = {"3.10.0": 3_010_000, "3.11.0": 3_011_000}[profile_version]
    if (
        mujoco["python_version"] != profile_version
        or mujoco["native_version"] != profile_version
        or mujoco["version_integer"] != expected_integer
    ):
        raise RuntimeEvidenceAdmissionError("runtime MuJoCo identity differs from profile version")
    distribution = _validate_distribution(mujoco["distribution"], "runtime.mujoco.distribution")
    if distribution["name"] != "mujoco" or distribution["version"] != profile_version:
        raise RuntimeEvidenceAdmissionError("MuJoCo distribution identity differs from profile")
    native = _exact_object(
        mujoco["loaded_native_library"],
        frozenset({"filename", "loaded_path", "resolved_path", "size_bytes", "sha256"}),
        "loaded native library",
    )
    for key in ("loaded_path", "resolved_path"):
        if not Path(_string(native[key], f"native.{key}")).is_absolute():
            raise RuntimeEvidenceAdmissionError("native library locators must be absolute")
    _sha256(native["sha256"], "native library sha256")
    _integer(native["size_bytes"], "native library size", minimum=1)


def _validate_runtime_mujoco_v2(
    value: object,
    package_version: str,
    native_version: str,
    native_version_integer: int,
) -> None:
    """Admit one version-generic exact MuJoCo distribution and loaded library."""
    mujoco = _exact_object(
        value,
        frozenset(
            {
                "package_version",
                "native_version",
                "native_version_integer",
                "distribution",
                "loaded_native_library",
            }
        ),
        "runtime.mujoco",
    )
    if (
        mujoco["package_version"] != package_version
        or mujoco["native_version"] != native_version
        or mujoco["native_version_integer"] != native_version_integer
    ):
        raise RuntimeEvidenceAdmissionError("runtime MuJoCo identity differs from profile")
    distribution = _validate_distribution(mujoco["distribution"], "runtime.mujoco.distribution")
    if distribution["name"] != "mujoco" or distribution["version"] != package_version:
        raise RuntimeEvidenceAdmissionError("MuJoCo distribution identity differs from profile")
    native = _exact_object(
        mujoco["loaded_native_library"],
        frozenset({"filename", "loaded_path", "resolved_path", "size_bytes", "sha256"}),
        "loaded native library",
    )
    _string(native["filename"], "native.filename")
    for key in ("loaded_path", "resolved_path"):
        if not Path(_string(native[key], f"native.{key}")).is_absolute():
            raise RuntimeEvidenceAdmissionError("native library locators must be absolute")
    _sha256(native["sha256"], "native library sha256")
    _integer(native["size_bytes"], "native library size", minimum=1)


def _validate_runtime_numpy(value: object) -> None:
    """Admit the common NumPy distribution identity used by the scientific profiles."""
    numpy = _exact_object(value, frozenset({"python_version", "distribution"}), "runtime.numpy")
    if numpy["python_version"] != _FIXED_NUMPY_VERSION:
        raise RuntimeEvidenceAdmissionError("runtime NumPy version differs from 2.3.5")
    distribution = _validate_distribution(numpy["distribution"], "runtime.numpy.distribution")
    if distribution["version"] != _FIXED_NUMPY_VERSION or distribution["name"] != "numpy":
        raise RuntimeEvidenceAdmissionError("runtime NumPy identity is inconsistent")


def _validate_runtime_numpy_v2(value: object) -> None:
    """Admit an exact but version-generic NumPy distribution identity."""
    numpy = _exact_object(value, frozenset({"python_version", "distribution"}), "runtime.numpy")
    version = _string(numpy["python_version"], "runtime.numpy.python_version")
    distribution = _validate_distribution(numpy["distribution"], "runtime.numpy.distribution")
    if distribution["version"] != version or distribution["name"] != "numpy":
        raise RuntimeEvidenceAdmissionError("runtime NumPy identity is inconsistent")


def _validate_runtime_installation_v2(value: object) -> None:
    """Admit optional measured Metrifid distribution identity without install authority."""
    installation = _exact_object(
        value, frozenset({"available", "distribution"}), "runtime.installation"
    )
    if type(installation["available"]) is not bool:
        raise RuntimeEvidenceAdmissionError("runtime installation availability must be boolean")
    if installation["available"] is False:
        if installation["distribution"] is not None:
            raise RuntimeEvidenceAdmissionError(
                "unavailable runtime installation must have null distribution"
            )
        return
    distribution = _exact_object(
        installation["distribution"],
        frozenset(
            {
                "name",
                "version",
                "member_count",
                "payload_identity_algorithm",
                "payload_sha256",
                "record_bound_identity_algorithm",
                "record_bound_member_count",
                "record_bound_payload_sha256",
                "record_declared_sha256_member_count",
                "record_unhashed_member_count",
            }
        ),
        "runtime.installation.distribution",
    )
    if distribution["name"] != "metrifid":
        raise RuntimeEvidenceAdmissionError("runtime installation is not Metrifid")
    _string(distribution["version"], "runtime.installation.distribution.version")
    for field in ("payload_sha256", "record_bound_payload_sha256"):
        _sha256(distribution[field], f"runtime.installation.distribution.{field}")
    for field in (
        "member_count",
        "record_bound_member_count",
        "record_declared_sha256_member_count",
        "record_unhashed_member_count",
    ):
        _integer(distribution[field], f"runtime.installation.distribution.{field}")


def _validate_distribution(value: object, field: str) -> dict[str, object]:
    """Recompute full and RECORD-bound hashes for one installed distribution projection."""
    distribution = _exact_object(
        value,
        frozenset(
            {
                "member_count",
                "members",
                "name",
                "payload_identity_algorithm",
                "payload_sha256",
                "record_bound_identity_algorithm",
                "record_bound_member_count",
                "record_bound_payload_sha256",
                "record_declared_sha256_member_count",
                "record_unhashed_member_count",
                "version",
            }
        ),
        field,
    )
    raw_members = _sequence(distribution["members"], f"{field}.members")
    if not 1 <= len(raw_members) <= 4096:
        raise RuntimeEvidenceAdmissionError(f"{field} member count is outside bounds")
    payload: list[dict[str, CanonicalValue]] = []
    record_bound: list[dict[str, CanonicalValue]] = []
    declared = 0
    logical_paths: list[str] = []
    for raw_member in raw_members:
        member = _exact_object(
            raw_member,
            frozenset(
                {"logical_path", "sha256", "size_bytes", "record_hash_mode", "record_hash_value"}
            ),
            f"{field} member",
        )
        logical_path = _string(member["logical_path"], f"{field}.logical_path")
        projected: dict[str, CanonicalValue] = {
            "logical_path": logical_path,
            "sha256": _sha256(member["sha256"], f"{field}.sha256"),
            "size_bytes": _integer(member["size_bytes"], f"{field}.size_bytes"),
        }
        mode = member["record_hash_mode"]
        if mode is None:
            if member["record_hash_value"] is not None:
                raise RuntimeEvidenceAdmissionError(f"{field} has RECORD value without mode")
        elif mode == "sha256":
            _string(member["record_hash_value"], f"{field}.record_hash_value")
            declared += 1
            if not logical_path.startswith("../"):
                record_bound.append(projected)
        else:
            raise RuntimeEvidenceAdmissionError(f"{field} uses unsupported RECORD hash mode")
        logical_paths.append(logical_path)
        payload.append(projected)
    if logical_paths != sorted(logical_paths) or len(logical_paths) != len(set(logical_paths)):
        raise RuntimeEvidenceAdmissionError(f"{field} logical paths are not unique and lexical")
    counts = (
        distribution["member_count"],
        distribution["record_bound_member_count"],
        distribution["record_declared_sha256_member_count"],
        distribution["record_unhashed_member_count"],
    )
    if counts != (len(payload), len(record_bound), declared, len(payload) - declared):
        raise RuntimeEvidenceAdmissionError(f"{field} member counts are inconsistent")
    if distribution["payload_sha256"] != canonical_sha256(
        cast(CanonicalValue, payload)
    ) or distribution["record_bound_payload_sha256"] != canonical_sha256(
        cast(CanonicalValue, record_bound)
    ):
        raise RuntimeEvidenceAdmissionError(f"{field} distribution hashes are inconsistent")
    return distribution


def _validate_pip_check(value: object) -> None:
    """Require the scientific profile's retained package-consistency check to pass."""
    check = _exact_object(value, frozenset({"argv", "exit_code", "stdout", "stderr"}), "pip_check")
    if (
        check["exit_code"] != 0
        or not isinstance(check["stdout"], str)
        or not isinstance(check["stderr"], str)
    ):
        raise RuntimeEvidenceAdmissionError("runtime pip check did not pass")
    if not _sequence(check["argv"], "pip_check.argv"):
        raise RuntimeEvidenceAdmissionError("runtime pip check argv is empty")


def _admit_trace(
    configuration: AdmittedRuntimeReviewConfigurationAny,
    step_dt: str,
    trace_value: object,
    contacts_value: object,
    fixture: Mapping[str, object],
    directory: Path,
    raw: Mapping[str, bytes],
) -> tuple[
    FloatArray,
    tuple[str, ...],
    tuple[EvidenceChannel, ...],
    tuple[EvidenceContact, ...],
    str,
]:
    """Admit bounded NPZ arrays and independently reproduce the canonical trace identity."""
    trace = _exact_object(
        trace_value,
        frozenset(
            {
                "schema",
                "schema_version",
                "arrays_locator",
                "observation_count",
                "observation_time_tokens",
                "channels",
                "canonical_trace_sha256",
            }
        ),
        "trace",
    )
    if (
        trace["schema"] != "metrifid.native_upgrade_trace"
        or trace["schema_version"] != 1
        or trace["arrays_locator"] != "trace.npz"
    ):
        raise RuntimeEvidenceAdmissionError("trace schema/version/locator is unsupported")
    count = _integer(trace["observation_count"], "trace.observation_count", minimum=1)
    if count > 10_001:
        raise RuntimeEvidenceAdmissionError("trace observation count exceeds the product bound")
    tokens = tuple(
        _decimal_token(item, "trace observation time")
        for item in _sequence(trace["observation_time_tokens"], "trace observation times")
    )
    _validate_observation_clock(tokens, count, configuration.config.required_horizon)
    descriptors = _sequence(trace["channels"], "trace.channels")
    if not 1 <= len(descriptors) <= 256:
        raise RuntimeEvidenceAdmissionError("trace channel count is outside [1, 256]")
    channel_metadata = tuple(
        _admit_channel_descriptor(item, f"channel_{index:04d}", count)
        for index, item in enumerate(descriptors)
    )
    channel_ids = [item[1] for item in channel_metadata]
    if channel_ids != sorted(channel_ids) or len(channel_ids) != len(set(channel_ids)):
        raise RuntimeEvidenceAdmissionError("trace channel IDs must be unique and lexical")
    contacts = _admit_contacts(contacts_value, configuration.config.required_horizon, step_dt)
    _validate_layout_against_manifest(channel_metadata, contacts, fixture)
    expected_members = frozenset(
        {"observation_times.npy", *(f"{item[0]}.npy" for item in channel_metadata)}
    )
    try:
        loaded = load_npz_arrays(
            directory / "trace.npz",
            expected_members=expected_members,
            invalid_reason=OperationalReasonCode.NPZ_DTYPE_MISMATCH,
        )
    except ArtifactAdmissionRefusal as exc:
        raise RuntimeEvidenceAdmissionError(
            f"trace.npz admission failed: {exc.reason.value}"
        ) from exc
    if loaded.raw_file_sha256 != hashlib.sha256(raw["trace.npz"]).hexdigest():
        raise RuntimeEvidenceAdmissionError("trace.npz changed after checksum validation")
    observation_times = _admit_observation_time_array(loaded.arrays, tokens)
    channels = tuple(_copy_channel_array(loaded.arrays, metadata) for metadata in channel_metadata)
    _validate_contact_trace_semantics(channels, contacts, observation_times)
    actual_trace_sha = _canonical_trace_sha256(
        descriptors, _sequence(contacts_value, "contacts"), observation_times, loaded.arrays
    )
    claimed = _sha256(trace["canonical_trace_sha256"], "canonical_trace_sha256")
    if actual_trace_sha != claimed:
        raise RuntimeEvidenceAdmissionError(
            "canonical trace hash does not bind admitted raw arrays"
        )
    return observation_times, tokens, channels, contacts, claimed


ChannelMetadata = tuple[
    str,
    str,
    str,
    str,
    str,
    int | None,
    str | None,
    str | None,
    tuple[int, ...],
    str,
]


def _admit_channel_descriptor(
    value: object,
    expected_key: str,
    observation_count: int,
) -> ChannelMetadata:
    """Admit one exact scalar, quaternion, or contact trace descriptor."""
    descriptor = _exact_object(
        value,
        frozenset(
            {
                "array_key",
                "channel_id",
                "kind",
                "semantic_type",
                "object_name",
                "component",
                "shape",
                "dtype",
                "scale",
                "tolerance",
            }
        ),
        "channel descriptor",
    )
    array_key = _string(descriptor["array_key"], "channel array_key")
    if array_key != expected_key:
        raise RuntimeEvidenceAdmissionError("channel array_key differs from deterministic order")
    channel_id = _string(descriptor["channel_id"], "channel_id")
    kind = _string(descriptor["kind"], "channel kind")
    semantic_type = _string(descriptor["semantic_type"], "channel semantic_type")
    object_name = _string(descriptor["object_name"], "channel object_name")
    component_value = descriptor["component"]
    component = None if component_value is None else _integer(component_value, "component")
    shape = tuple(
        _integer(item, "channel shape", minimum=1)
        for item in _sequence(descriptor["shape"], "channel shape")
    )
    dtype = _string(descriptor["dtype"], "channel dtype")
    if kind == "BODY_QUATERNION":
        if (
            semantic_type != "UNIT_QUATERNION_WXYZ"
            or shape != (observation_count, 4)
            or dtype != "<f8"
            or descriptor["scale"] is not None
        ):
            raise RuntimeEvidenceAdmissionError("quaternion descriptor semantics are invalid")
        tolerance = _positive_decimal_token(descriptor["tolerance"], "quaternion tolerance")
        return (
            array_key,
            channel_id,
            kind,
            semantic_type,
            object_name,
            component,
            None,
            tolerance,
            shape,
            dtype,
        )
    if kind == "CONTACT_OCCUPANCY":
        if (
            semantic_type != "BOOLEAN_OCCUPANCY"
            or shape != (observation_count,)
            or dtype != "|u1"
            or descriptor["scale"] is not None
            or descriptor["tolerance"] is not None
        ):
            raise RuntimeEvidenceAdmissionError("occupancy descriptor semantics are invalid")
        return (
            array_key,
            channel_id,
            kind,
            semantic_type,
            object_name,
            component,
            None,
            None,
            shape,
            dtype,
        )
    allowed_scalars = {
        "JOINT_POSITION",
        "JOINT_VELOCITY",
        "BODY_POSITION",
        "SENSOR",
        "CONTACT_NORMAL_FORCE",
        "CONTACT_NORMAL_IMPULSE",
    }
    if (
        kind not in allowed_scalars
        or semantic_type != "CONTINUOUS_SCALAR"
        or shape != (observation_count,)
        or dtype != "<f8"
    ):
        raise RuntimeEvidenceAdmissionError("scalar descriptor semantics are invalid")
    return (
        array_key,
        channel_id,
        kind,
        semantic_type,
        object_name,
        component,
        _positive_decimal_token(descriptor["scale"], "scalar scale"),
        _positive_decimal_token(descriptor["tolerance"], "scalar tolerance"),
        shape,
        dtype,
    )


def _validate_observation_clock(tokens: tuple[str, ...], count: int, horizon: str) -> None:
    """Require the frozen 51-sample 0.02-second grid across the one-second horizon."""
    if (
        horizon != _render_decimal(_FIXED_HORIZON)
        or count != len(_EXPECTED_OBSERVATION_TIME_TOKENS)
        or tokens != _EXPECTED_OBSERVATION_TIME_TOKENS
    ):
        raise RuntimeEvidenceAdmissionError(
            "observation clock must contain exactly t=0..1 in 0.02-second steps"
        )


def _admit_observation_time_array(
    arrays: Mapping[str, npt.NDArray[np.generic]],
    tokens: tuple[str, ...],
) -> FloatArray:
    """Bind the exact little-endian binary64 clock array to its decimal metadata tokens."""
    value = arrays.get("observation_times")
    if value is None or value.dtype.str != "<f8" or value.shape != (len(tokens),):
        raise RuntimeEvidenceAdmissionError("observation_times has invalid dtype or shape")
    expected = np.asarray([float(Decimal(token)) for token in tokens], dtype="<f8")
    if not np.array_equal(value, expected):
        raise RuntimeEvidenceAdmissionError("observation_times differs from exact metadata clock")
    copied = np.ascontiguousarray(value, dtype="<f8").copy()
    copied.setflags(write=False)
    return cast(FloatArray, copied)


def _copy_channel_array(
    arrays: Mapping[str, npt.NDArray[np.generic]], metadata: ChannelMetadata
) -> EvidenceChannel:
    """Copy one preflighted channel into immutable exact-dtype C-contiguous storage."""
    (
        array_key,
        channel_id,
        kind,
        semantic_type,
        object_name,
        component,
        scale,
        tolerance,
        shape,
        dtype,
    ) = metadata
    try:
        value = arrays[array_key]
    except KeyError as exc:  # pragma: no cover - the NPZ member preflight catches this first
        raise RuntimeEvidenceAdmissionError("trace channel array is missing") from exc
    if value.dtype.str != dtype or value.shape != shape or not value.flags.c_contiguous:
        raise RuntimeEvidenceAdmissionError(
            "trace channel dtype/shape/layout differs from descriptor"
        )
    copied = np.ascontiguousarray(value, dtype=np.dtype(dtype)).copy()
    if kind == "CONTACT_OCCUPANCY" and not np.all((copied == 0) | (copied == 1)):
        raise RuntimeEvidenceAdmissionError("contact occupancy array contains values outside {0,1}")
    copied.setflags(write=False)
    return EvidenceChannel(
        array_key=array_key,
        channel_id=channel_id,
        kind=kind,
        semantic_type=semantic_type,
        object_name=object_name,
        component=component,
        scale_token=scale,
        tolerance_token=tolerance,
        dtype=dtype,
        shape=shape,
        values=cast(FloatArray | ByteArray, copied),
    )


def _validate_layout_against_manifest(
    actual: tuple[ChannelMetadata, ...],
    contacts: tuple[EvidenceContact, ...],
    fixture: Mapping[str, object],
) -> None:
    """Require exact channels, semantic types, scales, tolerances, and contact pairs."""
    expected_layout, expected_contacts = _manifest_layout(fixture)
    actual_layout = tuple(
        (
            channel_id,
            kind,
            semantic_type,
            object_name,
            component,
            scale,
            tolerance,
            dtype,
            shape[1:],
        )
        for (
            _array_key,
            channel_id,
            kind,
            semantic_type,
            object_name,
            component,
            scale,
            tolerance,
            shape,
            dtype,
        ) in actual
    )
    actual_contacts = tuple((contact.channel_id, contact.geom_names) for contact in contacts)
    if actual_layout != expected_layout or actual_contacts != expected_contacts:
        raise RuntimeEvidenceAdmissionError("trace channel/contact layout differs from manifest")


def _manifest_layout(
    fixture: Mapping[str, object],
) -> tuple[tuple[tuple[object, ...], ...], tuple[tuple[str, tuple[str, str]], ...]]:
    """Reproduce the worker's lexical channel and contact layout from manifest declarations."""
    continuous = _manifest_number_token(fixture["continuous_tolerance"], "continuous_tolerance")
    so3 = _manifest_number_token(fixture["so3_tolerance"], "so3_tolerance")
    rows: list[tuple[object, ...]] = []
    for raw_channel in _sequence(fixture["channels"], "manifest channels"):
        channel = _exact_object(
            raw_channel,
            frozenset({"channel_id", "kind", "object_name", "component", "scale"}),
            "manifest channel",
        )
        kind = _string(channel["kind"], "manifest channel kind")
        quaternion = kind == "BODY_QUATERNION"
        scale = (
            None
            if channel["scale"] is None
            else _manifest_number_token(channel["scale"], "manifest channel scale")
        )
        rows.append(
            (
                _string(channel["channel_id"], "manifest channel_id"),
                kind,
                "UNIT_QUATERNION_WXYZ" if quaternion else "CONTINUOUS_SCALAR",
                _string(channel["object_name"], "manifest object_name"),
                channel["component"],
                scale,
                so3 if quaternion else continuous,
                "<f8",
                (4,) if quaternion else (),
            )
        )
    contact_rows: list[tuple[str, tuple[str, str]]] = []
    for raw_pair in _sequence(fixture["contact_pairs"], "manifest contact pairs"):
        pair = _exact_object(
            raw_pair,
            frozenset({"channel_id", "geom_a", "geom_b", "force_scale", "impulse_scale"}),
            "manifest contact pair",
        )
        contact_id = _string(pair["channel_id"], "contact pair channel_id")
        geom_names = tuple(
            sorted(
                (
                    _string(pair["geom_a"], "contact geom_a"),
                    _string(pair["geom_b"], "contact geom_b"),
                )
            )
        )
        contact_rows.append((contact_id, cast(tuple[str, str], geom_names)))
        rows.extend(
            (
                (
                    f"{contact_id}.occupancy",
                    "CONTACT_OCCUPANCY",
                    "BOOLEAN_OCCUPANCY",
                    contact_id,
                    None,
                    None,
                    None,
                    "|u1",
                    (),
                ),
                (
                    f"{contact_id}.normal_force",
                    "CONTACT_NORMAL_FORCE",
                    "CONTINUOUS_SCALAR",
                    contact_id,
                    None,
                    _manifest_number_token(pair["force_scale"], "contact force scale"),
                    continuous,
                    "<f8",
                    (),
                ),
                (
                    f"{contact_id}.normal_impulse",
                    "CONTACT_NORMAL_IMPULSE",
                    "CONTINUOUS_SCALAR",
                    contact_id,
                    None,
                    _manifest_number_token(pair["impulse_scale"], "contact impulse scale"),
                    continuous,
                    "<f8",
                    (),
                ),
            )
        )
    rows.sort(key=lambda row: cast(str, row[0]))
    contact_rows.sort(key=lambda row: row[0])
    if len({row[0] for row in rows}) != len(rows):
        raise RuntimeEvidenceAdmissionError("manifest channel IDs are not unique")
    return tuple(rows), tuple(contact_rows)


def _admit_contacts(value: object, horizon: str, step_dt: str) -> tuple[EvidenceContact, ...]:
    """Admit bounded, unique semantic contact topology and aggregate content."""
    contacts = tuple(
        _admit_contact(item, horizon, step_dt) for item in _sequence(value, "contacts")
    )
    channel_ids = [contact.channel_id for contact in contacts]
    geom_pairs = [contact.geom_names for contact in contacts]
    if channel_ids != sorted(channel_ids) or len(channel_ids) != len(set(channel_ids)):
        raise RuntimeEvidenceAdmissionError("contacts must have unique lexical channel IDs")
    if len(geom_pairs) != len(set(geom_pairs)):
        raise RuntimeEvidenceAdmissionError("contact geometry pairs must be unique")
    return contacts


def _admit_contact(value: object, horizon: str, step_dt: str) -> EvidenceContact:
    """Admit one exact alternating contact event sequence and consistent segments."""
    contact = _exact_object(
        value,
        frozenset(
            {
                "channel_id",
                "geom_names",
                "events",
                "segments",
                "persistence",
                "aggregate_normal_impulse",
            }
        ),
        "contact",
    )
    geom_values = tuple(
        _string(item, "contact geom name")
        for item in _sequence(contact["geom_names"], "contact geom names")
    )
    if (
        len(geom_values) != 2
        or tuple(sorted(geom_values)) != geom_values
        or geom_values[0] == geom_values[1]
    ):
        raise RuntimeEvidenceAdmissionError("contact geom names must be two distinct lexical names")
    events = _admit_contact_events(contact["events"], horizon, step_dt)
    segments = _admit_contact_segments(contact["segments"], horizon, step_dt)
    persistence = _nonnegative_decimal_token(contact["persistence"], "contact persistence")
    impulse = _nonnegative_decimal_token(
        contact["aggregate_normal_impulse"], "contact aggregate impulse"
    )
    event_segments = tuple(
        (events[index][1], events[index + 1][1]) for index in range(0, len(events), 2)
    )
    if event_segments != tuple((onset, release) for onset, release, _duration in segments):
        raise RuntimeEvidenceAdmissionError(
            "contact events and segments describe different topology"
        )
    if sum(Decimal(duration) for _onset, _release, duration in segments) != Decimal(persistence):
        raise RuntimeEvidenceAdmissionError("contact aggregate persistence is inconsistent")
    if not _manifest_divides(Decimal(persistence), Decimal(step_dt)):
        raise RuntimeEvidenceAdmissionError(
            "contact aggregate persistence must land on the simulation grid"
        )
    return EvidenceContact(
        channel_id=_string(contact["channel_id"], "contact channel_id"),
        geom_names=geom_values,
        events=events,
        segments=segments,
        persistence=persistence,
        aggregate_normal_impulse=impulse,
    )


def _admit_contact_events(value: object, horizon: str, step_dt: str) -> tuple[tuple[str, str], ...]:
    """Admit an alternating ONSET/RELEASE sequence within the declared horizon."""
    events: list[tuple[str, str]] = []
    for index, raw_event in enumerate(_sequence(value, "contact events")):
        event = _exact_object(raw_event, frozenset({"event", "time"}), "contact event")
        name = _string(event["event"], "contact event name")
        time = _nonnegative_decimal_token(event["time"], "contact event time")
        decimal_time = Decimal(time)
        if (
            name != ("ONSET" if index % 2 == 0 else "RELEASE")
            or decimal_time > Decimal(horizon)
            or not _manifest_divides(decimal_time, Decimal(step_dt))
        ):
            raise RuntimeEvidenceAdmissionError("contact events must alternate within horizon")
        if events and decimal_time <= Decimal(events[-1][1]):
            raise RuntimeEvidenceAdmissionError("contact event times must strictly increase")
        events.append((name, time))
    if len(events) % 2:
        raise RuntimeEvidenceAdmissionError("every contact onset must have a release")
    return tuple(events)


def _admit_contact_segments(
    value: object, horizon: str, step_dt: str
) -> tuple[tuple[str, str, str], ...]:
    """Admit contact segments with exact decimal duration consistency."""
    segments: list[tuple[str, str, str]] = []
    for raw_segment in _sequence(value, "contact segments"):
        segment = _exact_object(
            raw_segment, frozenset({"onset", "release", "persistence"}), "contact segment"
        )
        onset = _nonnegative_decimal_token(segment["onset"], "contact onset")
        release = _nonnegative_decimal_token(segment["release"], "contact release")
        duration = _nonnegative_decimal_token(segment["persistence"], "contact duration")
        decimal_onset = Decimal(onset)
        decimal_release = Decimal(release)
        decimal_duration = Decimal(duration)
        if (
            decimal_onset >= decimal_release
            or decimal_release > Decimal(horizon)
            or decimal_release - decimal_onset != decimal_duration
            or not _manifest_divides(decimal_onset, Decimal(step_dt))
            or not _manifest_divides(decimal_release, Decimal(step_dt))
            or not _manifest_divides(decimal_duration, Decimal(step_dt))
            or (segments and decimal_onset <= Decimal(segments[-1][1]))
        ):
            raise RuntimeEvidenceAdmissionError("contact segment times are inconsistent")
        segments.append((onset, release, duration))
    return tuple(segments)


def _validate_contact_trace_semantics(
    channels: Sequence[EvidenceChannel],
    contacts: Sequence[EvidenceContact],
    times: FloatArray,
) -> None:
    """Bind contact event metadata to occupancy, force, and cumulative impulse arrays."""
    by_kind = {(channel.object_name, channel.kind): channel for channel in channels}
    for contact in contacts:
        required = {
            "CONTACT_OCCUPANCY",
            "CONTACT_NORMAL_FORCE",
            "CONTACT_NORMAL_IMPULSE",
        }
        if any((contact.channel_id, kind) not in by_kind for kind in required):
            raise RuntimeEvidenceAdmissionError("contact metadata lacks its three trace channels")
        occupancy = by_kind[(contact.channel_id, "CONTACT_OCCUPANCY")].values
        expected = np.asarray(
            [
                any(
                    float(Decimal(onset)) <= float(time) < float(Decimal(release))
                    for onset, release, _duration in contact.segments
                )
                for time in times
            ],
            dtype="|u1",
        )
        if not np.array_equal(occupancy, expected):
            raise RuntimeEvidenceAdmissionError("contact occupancy differs from semantic segments")
        force = by_kind[(contact.channel_id, "CONTACT_NORMAL_FORCE")].values
        impulse = by_kind[(contact.channel_id, "CONTACT_NORMAL_IMPULSE")].values
        if np.all(np.isfinite(force)) and np.any(force < 0.0):
            raise RuntimeEvidenceAdmissionError("finite contact force values must be nonnegative")
        if np.all(np.isfinite(impulse)):
            if np.any(impulse < 0.0) or np.any(np.diff(impulse) < 0.0):
                raise RuntimeEvidenceAdmissionError("finite cumulative contact impulse is invalid")
            if float(impulse[-1]) != float(Decimal(contact.aggregate_normal_impulse)):
                raise RuntimeEvidenceAdmissionError("contact aggregate impulse differs from trace")


def _canonical_trace_sha256(
    descriptors: list[object],
    contacts: list[object],
    observation_times: FloatArray,
    arrays: Mapping[str, npt.NDArray[np.generic]],
) -> str:
    """Independently reproduce the worker's framed scientific trace identity."""
    header = {
        "channels": descriptors,
        "observation_times": {
            "array_key": "observation_times",
            "dtype": "<f8",
            "shape": [len(observation_times)],
            "unit": "seconds",
        },
        "schema": "metrifid.native_upgrade_canonical_trace",
        "schema_version": 1,
        "semantic_contacts": contacts,
    }
    digest = hashlib.sha256()
    _add_hash_frame(digest, "metadata", _worker_json_bytes(header))
    _add_hash_frame(digest, "observation_times", observation_times.tobytes(order="C"))
    for index in range(len(descriptors)):
        key = f"channel_{index:04d}"
        _add_hash_frame(digest, key, arrays[key].tobytes(order="C"))
    return digest.hexdigest()


def _add_hash_frame(digest: object, label: str, payload: bytes) -> None:
    """Append one length-framed label and payload to a SHA-256-compatible object."""
    writer = cast("_HashWriter", digest)
    label_bytes = label.encode("utf-8")
    writer.update(struct.pack(">Q", len(label_bytes)))
    writer.update(label_bytes)
    writer.update(struct.pack(">Q", len(payload)))
    writer.update(payload)


class _HashWriter:
    """Structural protocol substitute for hashlib objects on supported Python versions."""

    def update(self, data: bytes) -> object:
        """Accept one next digest byte fragment."""
        raise NotImplementedError


def _admit_diagnostics(
    value: object,
    observation_tokens: tuple[str, ...],
) -> tuple[dict[str, CanonicalValue], tuple[DiagnosticSample, ...]]:
    """Admit aggregate and time-local diagnostics and reproduce their invariants."""
    diagnostics = _exact_object(
        value,
        frozenset(
            {
                "finite_values",
                "warnings_passed",
                "solver_converged",
                "warning_records",
                "max_solver_iterations",
                "solver_iteration_limit",
                "max_solver_residual",
                "raw_contact_order_sha256",
                "samples",
            }
        ),
        "diagnostics",
    )
    raw_samples = _sequence(diagnostics["samples"], "diagnostic samples")
    if len(raw_samples) != len(observation_tokens):
        raise RuntimeEvidenceAdmissionError("diagnostic samples must cover every observation time")
    samples = tuple(
        _admit_diagnostic_sample(item, observation_tokens[index])
        for index, item in enumerate(raw_samples)
    )
    finite = _boolean(diagnostics["finite_values"], "diagnostics.finite_values")
    warnings = _boolean(diagnostics["warnings_passed"], "diagnostics.warnings_passed")
    solver = _boolean(diagnostics["solver_converged"], "diagnostics.solver_converged")
    warning_records = _sequence(diagnostics["warning_records"], "diagnostic warning records")
    maximum_iterations = _integer(
        diagnostics["max_solver_iterations"], "diagnostics.max_solver_iterations"
    )
    iteration_limit = _integer(
        diagnostics["solver_iteration_limit"], "diagnostics.solver_iteration_limit", minimum=1
    )
    maximum_residual = _nullable_nonnegative_decimal_token(
        diagnostics["max_solver_residual"], "diagnostics.max_solver_residual"
    )
    _sha256(diagnostics["raw_contact_order_sha256"], "raw_contact_order_sha256")
    _validate_diagnostic_cumulative_contract(samples, iteration_limit)
    residuals = tuple(sample.max_solver_residual for sample in samples)
    finite_residuals = tuple(Decimal(item) for item in residuals if item is not None)
    if (
        maximum_iterations > iteration_limit
        or maximum_iterations != samples[-1].max_solver_iterations
        or maximum_iterations != max(sample.max_solver_iterations for sample in samples)
    ):
        raise RuntimeEvidenceAdmissionError("aggregate solver iteration maximum is inconsistent")
    if any(item is None for item in residuals):
        if maximum_residual is not None or samples[-1].max_solver_residual is not None or solver:
            raise RuntimeEvidenceAdmissionError(
                "nonfinite residual must be null and fail solver gate"
            )
    elif (
        maximum_residual is None
        or maximum_residual != samples[-1].max_solver_residual
        or Decimal(maximum_residual) != max(finite_residuals)
    ):
        raise RuntimeEvidenceAdmissionError("aggregate solver residual maximum is inconsistent")
    if finite != samples[-1].finite_values or finite != all(
        sample.finite_values for sample in samples
    ):
        raise RuntimeEvidenceAdmissionError("aggregate finite gate differs from samples")
    if (
        warnings != samples[-1].warnings_passed
        or warnings != all(sample.warnings_passed for sample in samples)
        or warnings == bool(warning_records)
    ):
        raise RuntimeEvidenceAdmissionError("aggregate warning gate/records are inconsistent")
    if solver != samples[-1].solver_converged or solver != all(
        sample.solver_converged for sample in samples
    ):
        raise RuntimeEvidenceAdmissionError("aggregate solver gate differs from samples")
    return _canonical_mapping(diagnostics, "diagnostics"), samples


def _admit_diagnostic_sample(value: object, expected_time: str) -> DiagnosticSample:
    """Admit one cumulative diagnostic sample aligned to the exact trace clock."""
    sample = _exact_object(
        value,
        frozenset(
            {
                "time",
                "finite_values",
                "warnings_passed",
                "solver_converged",
                "max_solver_iterations",
                "max_solver_residual",
            }
        ),
        "diagnostic sample",
    )
    time = _decimal_token(sample["time"], "diagnostic sample time")
    solver = _boolean(sample["solver_converged"], "diagnostic sample solver gate")
    residual = _nullable_nonnegative_decimal_token(
        sample["max_solver_residual"], "diagnostic sample residual"
    )
    if time != expected_time or (residual is None and solver):
        raise RuntimeEvidenceAdmissionError("diagnostic sample is not aligned or self-consistent")
    return DiagnosticSample(
        time_token=time,
        finite_values=_boolean(sample["finite_values"], "diagnostic finite gate"),
        warnings_passed=_boolean(sample["warnings_passed"], "diagnostic warning gate"),
        solver_converged=solver,
        max_solver_iterations=_integer(
            sample["max_solver_iterations"], "diagnostic solver iterations"
        ),
        max_solver_residual=residual,
    )


def _validate_diagnostic_cumulative_contract(
    samples: tuple[DiagnosticSample, ...], iteration_limit: int
) -> None:
    """Mirror the worker's cumulative gates, maxima, and strict convergence predicate."""
    for index, sample in enumerate(samples):
        expected_solver = (
            sample.finite_values
            and sample.warnings_passed
            and sample.max_solver_iterations < iteration_limit
            and sample.max_solver_residual is not None
        )
        if sample.max_solver_iterations > iteration_limit:
            raise RuntimeEvidenceAdmissionError(
                "diagnostic solver iterations exceed the configured limit"
            )
        if sample.solver_converged != expected_solver:
            raise RuntimeEvidenceAdmissionError(
                "diagnostic solver gate differs from the frozen convergence predicate"
            )
        if index == 0:
            continue
        previous = samples[index - 1]
        if sample.finite_values and not previous.finite_values:
            raise RuntimeEvidenceAdmissionError("diagnostic finite gate is not cumulative")
        if sample.warnings_passed and not previous.warnings_passed:
            raise RuntimeEvidenceAdmissionError("diagnostic warning gate is not cumulative")
        if sample.max_solver_iterations < previous.max_solver_iterations:
            raise RuntimeEvidenceAdmissionError(
                "diagnostic solver iteration maximum is not cumulative"
            )
        if previous.max_solver_residual is None:
            if sample.max_solver_residual is not None:
                raise RuntimeEvidenceAdmissionError(
                    "diagnostic nonfinite residual state is not cumulative"
                )
        elif sample.max_solver_residual is None:
            continue
        elif Decimal(sample.max_solver_residual) < Decimal(previous.max_solver_residual):
            raise RuntimeEvidenceAdmissionError(
                "diagnostic solver residual maximum is not cumulative"
            )


def _bind_cross_cell_evidence(
    configuration: AdmittedRuntimeReviewConfigurationAny,
    cells: tuple[AdmittedEvidenceCell, ...],
) -> None:
    """Bind exact subject/workload/layout clocks and profile/common execution identities."""
    first = cells[0]
    expected_fixture = configuration.config.expected_subject.fixture_id
    if any(cell.fixture_id != expected_fixture for cell in cells):
        raise RuntimeEvidenceAdmissionError("not every cell has the configured subject identity")
    if any(cell.workload != first.workload for cell in cells[1:]):
        raise RuntimeEvidenceAdmissionError("workload identity differs across evidence cells")
    subject_identity = _subject_without_compiled(first.subject)
    if any(_subject_without_compiled(cell.subject) != subject_identity for cell in cells[1:]):
        raise RuntimeEvidenceAdmissionError("source-closure identity differs across evidence cells")
    layout = tuple(channel.to_primitive() for channel in first.channels)
    if any(
        tuple(channel.to_primitive() for channel in cell.channels) != layout for cell in cells[1:]
    ):
        raise RuntimeEvidenceAdmissionError(
            "channel layout/tolerances differ across evidence cells"
        )
    if any(cell.observation_time_tokens != first.observation_time_tokens for cell in cells[1:]):
        raise RuntimeEvidenceAdmissionError("observation times differ across evidence cells")
    if any(
        cell.contact_event_time_tolerance != first.contact_event_time_tolerance
        for cell in cells[1:]
    ):
        raise RuntimeEvidenceAdmissionError("contact event tolerances differ across evidence cells")
    for role in ("baseline", "candidate"):
        profile_cells = tuple(cell for cell in cells if cell.profile_role == role)
        if any(cell.runtime != profile_cells[0].runtime for cell in profile_cells[1:]):
            raise RuntimeEvidenceAdmissionError(f"{role} cells do not share one runtime identity")
    baseline = next(cell for cell in cells if cell.profile_role == "baseline")
    candidate = next(cell for cell in cells if cell.profile_role == "candidate")
    if _common_runtime_projection(baseline.runtime) != _common_runtime_projection(
        candidate.runtime
    ):
        raise RuntimeEvidenceAdmissionError(
            "profiles do not share exact Python/NumPy/host/thread environment"
        )


def _subject_without_compiled(
    subject: Mapping[str, CanonicalValue],
) -> dict[str, CanonicalValue]:
    """Return the source identity while permitting runtime/grid-specific compiled MJB bytes."""
    result = dict(subject)
    result.pop("compiled_mjb_sha256", None)
    result.pop("compiled_mjb_size_bytes", None)
    return result


def _common_runtime_projection(runtime: Mapping[str, CanonicalValue]) -> dict[str, CanonicalValue]:
    """Project exact cross-profile interpreter, NumPy, host, and thread identities."""
    python = cast(dict[str, CanonicalValue], runtime["python"])
    numpy = cast(dict[str, CanonicalValue], runtime["numpy"])
    python_keys: tuple[str, ...]
    if runtime["schema_version"] == 2:
        python_keys = (
            "cache_tag",
            "compiler",
            "implementation",
            "implementation_name",
            "resolved_executable_sha256",
            "version",
            "version_full",
        )
    else:
        python_keys = (
            "cache_tag",
            "compiler",
            "implementation",
            "implementation_name",
            "resolved_executable",
            "resolved_executable_sha256",
            "version",
            "version_full",
        )
    common: dict[str, CanonicalValue] = {
        "python": {key: python[key] for key in python_keys},
        "host": runtime["host"],
        "thread_environment": runtime["thread_environment"],
    }
    if runtime["schema_version"] == 2:
        common["numpy"] = {
            "python_version": numpy["python_version"],
            "distribution": cast(
                CanonicalValue,
                _record_bound_distribution_identity(numpy["distribution"]),
            ),
        }
    else:
        distribution = cast(dict[str, CanonicalValue], numpy["distribution"])
        common["numpy"] = {
            "name": distribution["name"],
            "version": distribution["version"],
            "record_bound_member_count": distribution["record_bound_member_count"],
            "record_bound_payload_sha256": distribution["record_bound_payload_sha256"],
        }
    return common


def _build_case(
    configuration: AdmittedRuntimeReviewConfigurationAny,
    cells: tuple[AdmittedEvidenceCell, ...],
) -> CaseEvidence:
    """Reconstruct private CaseEvidence from raw traces, diagnostics, contacts, and repeats."""
    index = {(cell.profile_role, cell.step_dt, cell.repeat_id): cell for cell in cells}
    profile_runs = {
        role: tuple(
            (index[(role, step, 0)], index[(role, step, 1)])
            for step in configuration.config.step_dts
        )
        for role in ("baseline", "candidate")
    }
    all_cells = tuple(cell for pairs in profile_runs.values() for pair in pairs for cell in pair)
    events = [
        *_repeat_gate_events(profile_runs["baseline"]),
        *_repeat_gate_events(profile_runs["candidate"]),
        *_diagnostic_gate_events(all_cells),
        *_topology_gate_events(profile_runs["baseline"], profile_runs["candidate"]),
    ]
    selected_baseline = tuple(pair[0] for pair in profile_runs["baseline"])
    selected_candidate = tuple(pair[0] for pair in profile_runs["candidate"])
    scalar, orientations = _raw_observations(selected_baseline, selected_candidate)
    scalar.extend(_contact_event_observations(selected_baseline, selected_candidate))
    first = cells[0]
    return CaseEvidence(
        case_id=configuration.config.expected_subject.fixture_id,
        scalar_observations=tuple(scalar),
        orientation_observations=tuple(orientations),
        gate_events=tuple(events),
        minimum_prefix=0.5,
        horizon=float(Decimal(configuration.config.required_horizon)),
        campaign_role=_string(first.subject["campaign_role"], "subject campaign_role"),
        baseline_fixture_id=_string(
            first.subject["baseline_fixture_id"], "subject baseline_fixture_id"
        ),
    )


def _repeat_gate_events(
    pairs: Sequence[tuple[AdmittedEvidenceCell, AdmittedEvidenceCell]],
) -> list[GateEvent]:
    """Derive time-local repeatability gates from same-profile, same-grid raw evidence."""
    events: list[GateEvent] = []
    for first, second in pairs:
        difference = _first_repeat_difference(first, second)
        if difference is not None:
            time, channel = difference
            events.append(
                GateEvent(
                    time,
                    "REPEATABILITY_FAILED",
                    channel,
                    f"same-profile repeats differ at step_dt={first.step_dt}",
                )
            )
    return events


def _first_repeat_difference(
    first: AdmittedEvidenceCell, second: AdmittedEvidenceCell
) -> tuple[float, str] | None:
    """Locate the first content mismatch between two configured repeats."""
    if first.compiled_mjb_sha256 != second.compiled_mjb_sha256:
        return 0.0, "compiled_mjb_sha256"
    if first.observation_times.tobytes() != second.observation_times.tobytes():
        return 0.0, "observation_times"
    if len(first.channels) != len(second.channels):
        return 0.0, "channel_layout"
    differences: list[tuple[float, str]] = []
    for left, right in zip(first.channels, second.channels, strict=True):
        if left.to_primitive() != right.to_primitive() or left.values.shape != right.values.shape:
            return 0.0, "channel_layout"
        for index, time in enumerate(first.observation_times):
            if left.values[index].tobytes() != right.values[index].tobytes():
                differences.append((float(time), left.channel_id))
                break
    contact_difference = _first_contact_difference(first.contacts, second.contacts)
    if contact_difference is not None:
        differences.append(contact_difference)
    if first.diagnostic_primitive != second.diagnostic_primitive:
        for left_sample, right_sample in zip(first.diagnostics, second.diagnostics, strict=True):
            if left_sample != right_sample:
                differences.append((left_sample.time, "diagnostics"))
                break
        else:
            differences.append((0.0, "diagnostics"))
    return min(differences) if differences else None


def _first_contact_difference(
    left: Sequence[EvidenceContact], right: Sequence[EvidenceContact]
) -> tuple[float, str] | None:
    """Locate the first exact semantic-contact mismatch between two cells."""
    left_map = {item.channel_id: item for item in left}
    right_map = {item.channel_id: item for item in right}
    differences: list[tuple[float, str]] = []
    for channel_id in sorted(set(left_map) | set(right_map)):
        first = left_map.get(channel_id)
        second = right_map.get(channel_id)
        if first == second:
            continue
        events = () if first is None else first.events
        other_events = () if second is None else second.events
        times = [float(Decimal(item[1])) for item in (*events, *other_events)]
        differences.append((min(times) if times else 0.0, channel_id))
    return min(differences) if differences else None


def _diagnostic_gate_events(cells: Iterable[AdmittedEvidenceCell]) -> list[GateEvent]:
    """Convert aggregate and cumulative raw diagnostics into exact solver gates."""
    events: list[GateEvent] = []
    for cell in cells:
        aggregate = cell.finite_values and cell.warnings_passed and cell.solver_converged
        if not aggregate and all(
            sample.finite_values and sample.warnings_passed and sample.solver_converged
            for sample in cell.diagnostics
        ):
            events.append(
                GateEvent(
                    0.0, "SOLVER_NOT_CONVERGED", "diagnostics", "aggregate diagnostic gate failed"
                )
            )
        for index, sample in enumerate(cell.diagnostics):
            if sample.finite_values and sample.warnings_passed and sample.solver_converged:
                continue
            lower = 0.0 if index == 0 else cell.diagnostics[index - 1].time
            events.append(
                GateEvent(
                    lower,
                    "SOLVER_NOT_CONVERGED",
                    "diagnostics",
                    f"cumulative diagnostic gate failed at step_dt={cell.step_dt}",
                )
            )
    return events


def _topology_gate_events(
    baseline: Sequence[tuple[AdmittedEvidenceCell, AdmittedEvidenceCell]],
    candidate: Sequence[tuple[AdmittedEvidenceCell, AdmittedEvidenceCell]],
) -> list[GateEvent]:
    """Require stable contact-event labels across grids and exact profiles."""
    selected_baseline = [pair[0] for pair in baseline]
    selected_candidate = [pair[0] for pair in candidate]
    comparisons = [
        *((selected_baseline[0], item, "within baseline") for item in selected_baseline[1:]),
        *((selected_candidate[0], item, "within candidate") for item in selected_candidate[1:]),
        *(
            (left, right, f"cross-profile at step_dt={left.step_dt}")
            for left, right in zip(selected_baseline, selected_candidate, strict=True)
        ),
    ]
    events: list[GateEvent] = []
    for left, right, detail in comparisons:
        if _topology_signature(left) != _topology_signature(right):
            difference = _first_contact_difference(left.contacts, right.contacts)
            events.append(
                GateEvent(
                    0.0 if difference is None else difference[0],
                    "CONTACT_EVENT_TOPOLOGY_CHANGED",
                    "semantic_contacts",
                    f"contact event topology changed {detail}",
                )
            )
    return events


def _topology_signature(cell: AdmittedEvidenceCell) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return contact IDs and event labels without grid-sensitive physical times."""
    return tuple(
        (contact.channel_id, tuple(name for name, _time in contact.events))
        for contact in cell.contacts
    )


def _raw_observations(
    baseline: tuple[AdmittedEvidenceCell, ...],
    candidate: tuple[AdmittedEvidenceCell, ...],
) -> tuple[list[ScalarObservation], list[OrientationObservation]]:
    """Build three-grid scalar and quaternion observations directly from raw trace arrays."""
    baseline_maps = [
        {channel.channel_id: channel for channel in cell.channels} for cell in baseline
    ]
    candidate_maps = [
        {channel.channel_id: channel for channel in cell.channels} for cell in candidate
    ]
    scalar: list[ScalarObservation] = []
    orientations: list[OrientationObservation] = []
    for channel in baseline[0].channels:
        if channel.kind == "CONTACT_OCCUPANCY":
            continue
        if channel.tolerance is None:
            raise RuntimeEvidenceAdmissionError("decision-bearing channel lacks tolerance")
        for time_index, time in enumerate(baseline[0].observation_times):
            if channel.kind == "BODY_QUATERNION":
                orientations.append(
                    OrientationObservation(
                        channel_id=channel.channel_id,
                        time=float(time),
                        tolerance=channel.tolerance,
                        profile_a=_quaternion_grid_values(
                            baseline_maps, channel.channel_id, time_index
                        ),
                        profile_b=_quaternion_grid_values(
                            candidate_maps, channel.channel_id, time_index
                        ),
                        semantic_type=channel.semantic_type,
                    )
                )
            else:
                if channel.scale is None:
                    raise RuntimeEvidenceAdmissionError("decision-bearing scalar lacks scale")
                scalar.append(
                    ScalarObservation(
                        channel_id=channel.channel_id,
                        time=float(time),
                        scale=channel.scale,
                        tolerance=channel.tolerance,
                        profile_a=_scalar_grid_values(
                            baseline_maps, channel.channel_id, time_index
                        ),
                        profile_b=_scalar_grid_values(
                            candidate_maps, channel.channel_id, time_index
                        ),
                        semantic_type=channel.semantic_type,
                    )
                )
    return scalar, orientations


def _scalar_grid_values(
    channel_maps: Sequence[Mapping[str, EvidenceChannel]],
    channel_id: str,
    time_index: int,
) -> tuple[float, float, float]:
    """Return coarse/fine/finest scalar values in exact evaluator order."""
    values = tuple(float(channels[channel_id].values[time_index]) for channels in channel_maps)
    if len(values) != 3:
        raise RuntimeEvidenceAdmissionError("scalar witness requires exactly three grids")
    return values


def _quaternion_grid_values(
    channel_maps: Sequence[Mapping[str, EvidenceChannel]],
    channel_id: str,
    time_index: int,
) -> tuple[Quaternion, Quaternion, Quaternion]:
    """Return coarse/fine/finest ``wxyz`` quaternion values in evaluator order."""
    values = tuple(
        cast(Quaternion, tuple(float(item) for item in channels[channel_id].values[time_index]))
        for channels in channel_maps
    )
    if len(values) != 3:
        raise RuntimeEvidenceAdmissionError("orientation witness requires exactly three grids")
    return values


def _contact_event_observations(
    baseline: tuple[AdmittedEvidenceCell, ...],
    candidate: tuple[AdmittedEvidenceCell, ...],
) -> list[ScalarObservation]:
    """Reconstruct semantic contact-event-time witnesses after topology admission."""
    if any(
        _topology_signature(cell) != _topology_signature(baseline[0])
        for cell in (*baseline, *candidate)
    ):
        return []
    baseline_maps = [
        {contact.channel_id: contact for contact in cell.contacts} for cell in baseline
    ]
    candidate_maps = [
        {contact.channel_id: contact for contact in cell.contacts} for cell in candidate
    ]
    tolerance = float(Decimal(baseline[0].contact_event_time_tolerance))
    observations: list[ScalarObservation] = []
    for contact in baseline[0].contacts:
        for event_index, (event_name, _time) in enumerate(contact.events):
            profile_a = cast(
                tuple[float, float, float],
                tuple(
                    float(Decimal(mapping[contact.channel_id].events[event_index][1]))
                    for mapping in baseline_maps
                ),
            )
            profile_b = cast(
                tuple[float, float, float],
                tuple(
                    float(Decimal(mapping[contact.channel_id].events[event_index][1]))
                    for mapping in candidate_maps
                ),
            )
            observations.append(
                ScalarObservation(
                    channel_id=(
                        f"{contact.channel_id}.event.{event_index:02d}.{event_name.lower()}"
                    ),
                    time=min(*profile_a, *profile_b),
                    scale=1.0,
                    tolerance=tolerance,
                    profile_a=profile_a,
                    profile_b=profile_b,
                    semantic_type="CONTACT_EVENT_TIME_SECONDS",
                )
            )
    return observations


def _profiles_primitive(cells: tuple[AdmittedEvidenceCell, ...]) -> dict[str, CanonicalValue]:
    """Build exact baseline/candidate and verified common execution-identity projections."""
    baseline = next(cell for cell in cells if cell.profile_role == "baseline")
    candidate = next(cell for cell in cells if cell.profile_role == "candidate")
    return {
        "baseline": _one_profile_primitive(baseline),
        "candidate": _one_profile_primitive(candidate),
        "common_environment": _common_runtime_projection(baseline.runtime),
    }


def _one_profile_primitive(cell: AdmittedEvidenceCell) -> dict[str, CanonicalValue]:
    """Project one profile's declared and independently hashed MuJoCo/native identity."""
    if cell.schema_version == 2:
        if cell.profile_identity is None:
            raise RuntimeEvidenceAdmissionError("role-based cell lacks profile identity evidence")
        try:
            projected = profile_identity_receipt_projection_v2(
                cast(dict[str, object], cell.profile_identity)
            )
        except ProfileIdentityRefusal as exc:  # pragma: no cover - admitted construction invariant
            raise RuntimeEvidenceAdmissionError(
                "role-based cell profile identity projection failed"
            ) from exc
        projected["runtime_identity_sha256"] = cell.runtime_identity_sha256
        return cast(dict[str, CanonicalValue], projected)
    mujoco = cast(dict[str, CanonicalValue], cell.runtime["mujoco"])
    distribution = cast(dict[str, CanonicalValue], mujoco["distribution"])
    return {
        "profile_id": cell.profile_id,
        "mujoco_version": cell.profile_version,
        "runtime_identity_sha256": cell.runtime_identity_sha256,
        "distribution": {
            "name": distribution["name"],
            "version": distribution["version"],
            "record_bound_payload_sha256": distribution["record_bound_payload_sha256"],
            "record_bound_member_count": distribution["record_bound_member_count"],
        },
        "loaded_native_library": mujoco["loaded_native_library"],
        "external_profile_identity": cell.runtime["external_profile_identity"],
    }


def _subject_primitive(
    configuration: AdmittedRuntimeReviewConfigurationAny,
    cells: tuple[AdmittedEvidenceCell, ...],
) -> dict[str, CanonicalValue]:
    """Project expected source identities plus every profile/grid/repeat compiled MJB hash."""
    expected = configuration.config.expected_subject
    compiled: dict[str, CanonicalValue] = {}
    for role in ("baseline", "candidate"):
        by_step: dict[str, CanonicalValue] = {}
        for step in configuration.config.step_dts:
            by_step[step] = {
                f"repeat_{cell.repeat_id}": cell.compiled_mjb_sha256
                for cell in cells
                if cell.profile_role == role and cell.step_dt == step
            }
        compiled[role] = by_step
    first = cells[0]
    return {
        "fixture_id": expected.fixture_id,
        "source_closure_sha256": expected.source_closure_sha256,
        "fixture_manifest_sha256": expected.fixture_manifest_sha256,
        "fixture_raw_sha256": first.fixture_raw_sha256,
        "compiled_mjb_sha256": compiled,
    }


def _workload_primitive(
    configuration: AdmittedRuntimeReviewConfigurationAny,
    first: AdmittedEvidenceCell,
) -> dict[str, CanonicalValue]:
    """Project exact expected hashes and their independently admitted semantic objects."""
    expected = configuration.config.expected_workload
    return {
        "semantic_sha256": expected.semantic_sha256,
        "initial_state_semantic_sha256": expected.initial_state_semantic_sha256,
        "action_program_semantic_sha256": expected.action_program_semantic_sha256,
        "initial_state": first.workload["initial_state"],
        "action_program": first.workload["action_program"],
        "channels": [channel.to_primitive() for channel in first.channels],
        "contact_event_time_tolerance": first.contact_event_time_tolerance,
    }


def _exact_object(value: object, expected: frozenset[str], field: str) -> dict[str, object]:
    """Require a concrete JSON object with exactly the expected member names."""
    if type(value) is not dict:
        raise RuntimeEvidenceAdmissionError(f"{field} must be an object")
    result = cast(dict[str, object], value)
    actual = frozenset(result)
    if actual != expected:
        raise RuntimeEvidenceAdmissionError(
            f"{field} keys mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return result


def _sequence(value: object, field: str) -> list[object]:
    """Require a concrete JSON array at one admitted field."""
    if type(value) is not list:
        raise RuntimeEvidenceAdmissionError(f"{field} must be an array")
    return cast(list[object], value)


def _string(value: object, field: str) -> str:
    """Require one nonempty JSON string."""
    if not isinstance(value, str) or not value:
        raise RuntimeEvidenceAdmissionError(f"{field} must be a nonempty string")
    return value


def _boolean(value: object, field: str) -> bool:
    """Require one exact JSON boolean."""
    if type(value) is not bool:
        raise RuntimeEvidenceAdmissionError(f"{field} must be a boolean")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    """Require one exact JSON integer no smaller than the declared minimum."""
    if type(value) is not int or value < minimum:
        raise RuntimeEvidenceAdmissionError(f"{field} must be an integer >= {minimum}")
    return value


def _sha256(value: object, field: str) -> str:
    """Require one exact lowercase SHA-256 hexadecimal token."""
    token = _string(value, field)
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise RuntimeEvidenceAdmissionError(f"{field} must be lowercase SHA-256 hexadecimal")
    return token


def _decimal_token(value: object, field: str) -> str:
    """Admit one finite canonical ordinary-decimal string token."""
    token = _string(value, field)
    try:
        decimal = Decimal(token)
    except InvalidOperation as exc:
        raise RuntimeEvidenceAdmissionError(f"{field} must be a decimal token") from exc
    if not decimal.is_finite() or _render_decimal(decimal) != token:
        raise RuntimeEvidenceAdmissionError(f"{field} must be a canonical finite decimal token")
    return token


def _positive_decimal_token(value: object, field: str) -> str:
    """Admit one canonical finite decimal token strictly greater than zero."""
    token = _decimal_token(value, field)
    if Decimal(token) <= 0:
        raise RuntimeEvidenceAdmissionError(f"{field} must be positive")
    return token


def _nonnegative_decimal_token(value: object, field: str) -> str:
    """Admit one canonical finite decimal token no smaller than zero."""
    token = _decimal_token(value, field)
    if Decimal(token) < 0:
        raise RuntimeEvidenceAdmissionError(f"{field} must be nonnegative")
    return token


def _nullable_nonnegative_decimal_token(value: object, field: str) -> str | None:
    """Admit a nonnegative canonical decimal token or explicit null sentinel."""
    return None if value is None else _nonnegative_decimal_token(value, field)


def _manifest_number_token(value: object, field: str) -> str:
    """Convert one finite manifest JSON number to the worker's canonical decimal token."""
    return _render_decimal(_manifest_decimal(value, field))


def _manifest_decimal(value: object, field: str) -> Decimal:
    """Require one finite manifest JSON number without losing its lexical precision."""
    if type(value) not in {Decimal, int, float}:
        raise RuntimeEvidenceAdmissionError(
            f"{field} must be a JSON number and not a boolean or string"
        )
    if type(value) is float and not math.isfinite(value):
        raise RuntimeEvidenceAdmissionError(f"{field} must be finite")
    try:
        result = value if type(value) is Decimal else Decimal(str(value))
    except InvalidOperation as exc:
        raise RuntimeEvidenceAdmissionError(f"{field} must be a decimal number") from exc
    if not result.is_finite():
        raise RuntimeEvidenceAdmissionError(f"{field} must be finite")
    return result


def _manifest_bounded_string(value: object, field: str) -> str:
    """Require one nonempty manifest string of at most 256 Unicode code points."""
    if type(value) is not str or not value or len(value) > 256:
        raise RuntimeEvidenceAdmissionError(f"{field} must be a nonempty bounded string")
    return value


def _manifest_identifier(value: object, field: str) -> str:
    """Require one lowercase bounded manifest semantic identifier."""
    token = _manifest_bounded_string(value, field)
    if _IDENTIFIER_PATTERN.fullmatch(token) is None:
        raise RuntimeEvidenceAdmissionError(f"{field} is not a valid identifier")
    return token


def _manifest_relative_xml_path(value: object) -> str:
    """Require one normalized confined relative POSIX XML path token."""
    token = _manifest_bounded_string(value, "manifest xml_path")
    path = PurePosixPath(token)
    if (
        path.is_absolute()
        or path.as_posix() != token
        or path.suffix != ".xml"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeEvidenceAdmissionError(
            "manifest xml_path must be a normalized confined relative XML path"
        )
    return token


def _bounded_manifest_sequence(value: object, field: str, *, maximum: int) -> list[object]:
    """Require one concrete manifest array no larger than its worker-side bound."""
    result = _sequence(value, field)
    if len(result) > maximum:
        raise RuntimeEvidenceAdmissionError(f"{field} exceeds its member limit")
    return result


def _bounded_manifest_integer(value: object, field: str, minimum: int, maximum: int) -> int:
    """Require one exact bounded manifest integer, excluding booleans."""
    if type(value) is not int or not minimum <= value <= maximum:
        raise RuntimeEvidenceAdmissionError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def _manifest_divides(numerator: Decimal, denominator: Decimal) -> bool:
    """Return whether one positive exact-decimal interval divides another exactly."""
    return denominator > 0 and numerator % denominator == 0


def _render_decimal(value: Decimal) -> str:
    """Render a finite Decimal using the worker's shortest ordinary notation."""
    if value == 0:
        return "0"
    token = format(value, "f")
    return token.rstrip("0").rstrip(".") if "." in token else token


def _worker_json_bytes(value: object) -> bytes:
    """Encode worker-compatible compact canonical JSON, including genuine finite floats."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RuntimeEvidenceAdmissionError(
            "worker JSON projection is not canonicalizable"
        ) from exc


def _canonical_mapping(value: object, field: str) -> dict[str, CanonicalValue]:
    """Validate and defensively copy one canonical JSON object with no raw floats."""
    if type(value) is not dict:
        raise RuntimeEvidenceAdmissionError(f"{field} must be a canonical object")
    try:
        encoded = canonical_json_bytes(cast(CanonicalValue, value))
        copied = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeEvidenceAdmissionError(f"{field} contains noncanonical values") from exc
    return cast(dict[str, CanonicalValue], copied)


def _self_hash(value: Mapping[str, object], field: str) -> str:
    """Compute a canonical object self-hash with its digest member omitted."""
    if field not in value:
        raise RuntimeEvidenceAdmissionError(f"self-hashed object lacks {field}")
    projected = dict(value)
    projected.pop(field)
    return canonical_sha256(cast(CanonicalValue, projected))


def _exact_string_mapping(value: object, field: str) -> dict[str, str]:
    """Admit one mapping containing only exact string keys and values."""
    if type(value) is not dict:
        raise RuntimeEvidenceAdmissionError(f"{field} must be an object")
    result = cast(dict[str, object], value)
    if any(not isinstance(key, str) or not isinstance(item, str) for key, item in result.items()):
        raise RuntimeEvidenceAdmissionError(f"{field} must contain only strings")
    return cast(dict[str, str], result)


__all__ = [
    "AdmittedEvidenceCell",
    "AdmittedRuntimeEvidence",
    "DiagnosticSample",
    "EvidenceChannel",
    "EvidenceContact",
    "EvidenceMember",
    "RuntimeEvidenceAdmissionError",
    "admit_runtime_evidence",
]
