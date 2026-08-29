"""Execute the fixed twelve-cell native Runtime Review journey.

This module is deliberately product-specific.  It admits two explicit Python launchers, invokes
only the packaged profile collector and frozen evidence worker, validates every result before the
next process starts, and delegates the scientific decision to the existing Runtime Review referee.
It is not an executor framework and exposes no caller-controlled command, environment, or retry.
"""

from __future__ import annotations

import hashlib
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from .._json_admission import (
    CONFIG_JSON_LIMITS,
    JsonAdmissionError,
    read_bounded_regular_file,
)
from ..compare._failure import ComparisonOperationError, operational_error
from ..distribution import installed_distribution_sha256
from ..json_values import CanonicalValue, canonical_json_bytes, canonical_sha256, require_sha256
from ..operational import OperationalReasonCode, OperationalToolObservation
from ..version import __version__
from ._config import (
    CONFIG_SCHEMA,
    CONFIG_SCHEMA_VERSION_V2,
    PROFILE_ROLES,
    REPEAT_IDS,
    REQUIRED_HORIZON,
    STEP_DTS,
    AdmittedRuntimeReviewConfigurationV2,
    RuntimeReviewCellConfig,
    RuntimeReviewConfigV2,
)
from ._execution_config import (
    AdmittedRuntimeReviewRunConfiguration,
    InterpreterIdentity,
    load_runtime_review_run_configuration,
    recheck_runtime_review_run_configuration,
    recheck_runtime_review_run_inputs,
)
from ._execution_output import (
    RUN_RECORD_SCHEMA,
    RUN_RECORD_SCHEMA_VERSION_V2,
    RUNTIME_REVIEW_RUN_CLAIM_LIMITATIONS,
    OwnedRuntimeReviewRunOutput,
    RetainedProcessOutput,
    RuntimeReviewRunOutputError,
    VerifiedRuntimeReviewRunRecord,
    prepare_runtime_review_run_output,
)
from ._native_profile_identity import (
    _FROZEN_WORKER_SHA256,
    ProfileIdentityRefusal,
    load_native_profile_identity_v2,
    require_compatible_profile_identities_v2,
)
from ._paths import PathAdmissionError
from ._status import RuntimeReviewReasonCode, RuntimeReviewStatus, runtime_review_exit_code

if TYPE_CHECKING:
    from ._run import RuntimeReviewResult

_PROCESS_TIMEOUT_SECONDS: Final[int] = 300
_RESOURCE_BYTE_LIMIT: Final[int] = 4 * 1024 * 1024
_WORKER_RESOURCE_NAME: Final[str] = "native_evidence_worker.py.txt"
_COLLECTOR_RESOURCE_NAME: Final[str] = "_native_profile_identity.py"
_PROCESS_COMMAND_SCHEMA: Final[str] = "metrifid.runtime_review_process_command"
_PROCESS_COMMAND_SCHEMA_VERSION: Final[int] = 1
_OPERATION: Final[str] = "compare"
_STEP_DIRECTORY_TOKENS: Final[Mapping[str, str]] = {
    "0.004": "0p004",
    "0.002": "0p002",
    "0.001": "0p001",
}


@dataclass(frozen=True, slots=True)
class RuntimeReviewRunResult:
    """One completed execution journey carrying the existing scientific receipt."""

    status: RuntimeReviewStatus
    reason_code: RuntimeReviewReasonCode | None
    receipt: dict[str, CanonicalValue]
    receipt_sha256: str
    runtime_review_json: Path
    runtime_review_markdown: Path
    runtime_review_run_json: Path
    run_sha256: str
    captured_evidence_root: Path
    generated_runtime_review_config: Path

    @property
    def exit_code(self) -> int:
        """Return the existing frozen process exit code for the scientific status."""
        return int(runtime_review_exit_code(self.status))


@dataclass(frozen=True, slots=True)
class _ResourceIdentity:
    """Absolute installed resource locator and exact regular-file digest."""

    path: Path
    sha256: str

    def __post_init__(self) -> None:
        """Require an absolute locator and lowercase SHA-256 identity."""
        if not self.path.is_absolute():
            raise ValueError("resource locator must be absolute")
        require_sha256(self.sha256, "resource SHA-256")

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Return the exact external-resource record projection."""
        return {"locator": self.path.as_posix(), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class _CompletedProcessObservation:
    """Raw bounded subprocess outcome returned by the narrow execution seam."""

    stdout: bytes
    stderr: bytes
    exit_code: int | None
    no_exit_status: str | None


@dataclass(frozen=True, slots=True)
class _AdmittedProfilePreflight:
    """One retained successful preflight and its strictly admitted identity."""

    role: str
    process: RetainedProcessOutput
    identity_path: Path
    identity_file_sha256: str
    identity: dict[str, Any]
    interpreter: InterpreterIdentity

    def to_primitive(self, output_root: Path) -> dict[str, CanonicalValue]:
        """Return the completed-record projection for this successful preflight."""
        primitive = self.process.to_primitive()
        primitive.update(
            {
                "role": self.role,
                "package_version": cast(str, self.identity["package_version"]),
                "native_version": cast(str, self.identity["native_version"]),
                "native_version_integer": cast(int, self.identity["native_version_integer"]),
                "support_tier": cast(str, self.identity["support_tier"]),
                "lexical_interpreter": self.interpreter.lexical_path.as_posix(),
                "resolved_interpreter": self.interpreter.resolved_path.as_posix(),
                "resolved_executable_sha256": self.interpreter.resolved_sha256,
                "identity_locator": self.identity_path.relative_to(output_root).as_posix(),
                "identity_file_sha256": self.identity_file_sha256,
                "profile_identity_sha256": cast(str, self.identity["profile_identity_sha256"]),
                "sentinel_identity_sha256": cast(
                    str,
                    _mapping(self.identity["sentinel"], "profile sentinel")[
                        "sentinel_identity_sha256"
                    ],
                ),
            }
        )
        return primitive


@dataclass(frozen=True, slots=True)
class _AdmittedEvidenceAttempt:
    """One retained successful evidence attempt admitted before later execution."""

    role: str
    package_version: str
    native_version: str
    native_version_integer: int
    profile_identity_sha256: str
    runtime_identity_sha256: str
    sentinel_identity_sha256: str
    step_dt: str
    repeat_id: int
    process: RetainedProcessOutput
    cell_path: Path
    result_sha256: str
    checksum_manifest_sha256: str
    interpreter: InterpreterIdentity

    def to_primitive(self, output_root: Path) -> dict[str, CanonicalValue]:
        """Return the completed-record projection for this canonical evidence slot."""
        primitive = self.process.to_primitive()
        primitive.update(
            {
                "role": self.role,
                "package_version": self.package_version,
                "native_version": self.native_version,
                "native_version_integer": self.native_version_integer,
                "profile_identity_sha256": self.profile_identity_sha256,
                "runtime_identity_sha256": self.runtime_identity_sha256,
                "sentinel_identity_sha256": self.sentinel_identity_sha256,
                "step_dt": self.step_dt,
                "repeat_id": self.repeat_id,
                "lexical_interpreter": self.interpreter.lexical_path.as_posix(),
                "resolved_interpreter": self.interpreter.resolved_path.as_posix(),
                "resolved_executable_sha256": self.interpreter.resolved_sha256,
                "cell_locator": self.cell_path.relative_to(output_root).as_posix(),
                "result_sha256": self.result_sha256,
                "checksum_manifest_sha256": self.checksum_manifest_sha256,
            }
        )
        return primitive


class _ExecutionRefusal(ValueError):
    """Carry one bounded operational reason from an already-retained partial run."""

    def __init__(
        self,
        code: OperationalReasonCode,
        field: str,
        message: str,
        *,
        evidence: Mapping[str, CanonicalValue] | None = None,
    ) -> None:
        """Retain one registry reason and bounded canonical evidence projection."""
        self.code = code
        self.field = field
        self.evidence = dict(evidence or {})
        self.evidence.setdefault("message", message[:300])
        super().__init__(message)


def run_runtime_review_configuration_file(config_path: str | Path) -> RuntimeReviewRunResult:
    """Run two preflights, twelve one-shot cells, and the existing Runtime Review referee."""
    admitted = _load_run_configuration(config_path)
    worker, collector = _resolve_execution_resources()
    try:
        recheck_runtime_review_run_configuration(admitted)
        with prepare_runtime_review_run_output(admitted) as output:
            return _execute_owned_run(admitted, output, worker, collector)
    except ComparisonOperationError:
        raise
    except _ExecutionRefusal as exc:
        raise _refuse(exc.code, field=exc.field, **exc.evidence) from exc
    except (RuntimeReviewRunOutputError, PathAdmissionError, ProfileIdentityRefusal) as exc:
        raise _refuse(
            OperationalReasonCode.OUTPUT_WRITE_FAILED,
            field="runtime_review_run_output",
            exception_type=type(exc).__name__,
            message=str(exc)[:300],
        ) from exc
    except OSError as exc:
        raise _refuse(
            OperationalReasonCode.OUTPUT_WRITE_FAILED,
            field="runtime_review_run_output",
            exception_type=type(exc).__name__,
            message=str(exc)[:300],
        ) from exc


def _load_run_configuration(
    config_path: str | Path,
) -> AdmittedRuntimeReviewRunConfiguration:
    """Load the strict run declaration and map caller-controlled failures to exit 64."""
    try:
        return load_runtime_review_run_configuration(config_path)
    except (JsonAdmissionError, PathAdmissionError, TypeError, ValueError, UnicodeError) as exc:
        raise _refuse(
            OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            field="runtime_review_run_config",
            exception_type=type(exc).__name__,
            message=str(exc)[:300],
        ) from exc
    except OSError as exc:
        raise _refuse(
            OperationalReasonCode.CONFIGURATION_IO_FAILED,
            field="runtime_review_run_config",
            exception_type=type(exc).__name__,
            message=str(exc)[:300],
        ) from exc


def _resolve_execution_resources() -> tuple[_ResourceIdentity, _ResourceIdentity]:
    """Resolve and hash both installed source resources before any subprocess starts."""
    try:
        package_root = resources.files("metrifid.runtime_review")
        worker_path = Path(str(package_root.joinpath(_WORKER_RESOURCE_NAME))).absolute()
        collector_path = Path(str(package_root.joinpath(_COLLECTOR_RESOURCE_NAME))).absolute()
        worker = _measure_resource(worker_path, _WORKER_RESOURCE_NAME)
        collector = _measure_resource(collector_path, _COLLECTOR_RESOURCE_NAME)
    except (OSError, JsonAdmissionError, TypeError, ValueError) as exc:
        raise _refuse(
            OperationalReasonCode.EXECUTING_CODE_NOT_INSTALLED_DISTRIBUTION,
            field="runtime_review_execution_resources",
            exception_type=type(exc).__name__,
            message=str(exc)[:300],
        ) from exc
    if worker.sha256 != _FROZEN_WORKER_SHA256:
        raise _refuse(
            OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
            field="packaged_worker",
            observed_sha256=worker.sha256,
            expected_sha256=_FROZEN_WORKER_SHA256,
        )
    return worker, collector


def _execute_owned_run(
    admitted: AdmittedRuntimeReviewRunConfiguration,
    output: OwnedRuntimeReviewRunOutput,
    worker: _ResourceIdentity,
    collector: _ResourceIdentity,
) -> RuntimeReviewRunResult:
    """Complete one descriptor-owned run while preserving every partial failure artifact."""
    preflights = _run_profile_preflights(admitted, output, worker, collector)
    attempts, generated = _run_evidence_cells(admitted, output, worker, preflights)
    retained_generated = output.write_generated_runtime_review_configuration(generated)
    strict_generated = _strict_reload_generated_configuration(
        output.generated_runtime_review_config
    )
    result = _call_runtime_reviewer(strict_generated.path)
    _recheck_final_inputs(admitted, worker, collector)
    record = _build_completed_run_record(
        admitted,
        output,
        worker,
        collector,
        preflights,
        attempts,
        retained_generated.sha256,
        result,
    )
    verified = output.publish_completed_run_record(record)
    return _build_public_run_result(result, verified, output)


def _build_public_run_result(
    result: RuntimeReviewResult,
    verified: VerifiedRuntimeReviewRunRecord,
    output: OwnedRuntimeReviewRunOutput,
) -> RuntimeReviewRunResult:
    """Carry the existing referee result into the completed public execution value."""
    return RuntimeReviewRunResult(
        status=result.status,
        reason_code=result.reason_code,
        receipt=result.receipt,
        receipt_sha256=result.receipt_sha256,
        runtime_review_json=result.runtime_review_json,
        runtime_review_markdown=result.runtime_review_markdown,
        runtime_review_run_json=verified.path,
        run_sha256=verified.run_sha256,
        captured_evidence_root=output.root / "captured_evidence",
        generated_runtime_review_config=output.generated_runtime_review_config,
    )


def _run_profile_preflights(
    admitted: AdmittedRuntimeReviewRunConfiguration,
    output: OwnedRuntimeReviewRunOutput,
    worker: _ResourceIdentity,
    collector: _ResourceIdentity,
) -> tuple[_AdmittedProfilePreflight, _AdmittedProfilePreflight]:
    """Run baseline then candidate identity collection before any evidence cell."""
    completed: list[_AdmittedProfilePreflight] = []
    for role in PROFILE_ROLES:
        interpreter = _role_interpreter(admitted, role)
        _recheck_before_process(admitted, worker, collector)
        identity_path = output.new_profile_identity_path(role)
        argv = (
            interpreter.lexical_path.as_posix(),
            collector.path.as_posix(),
            "--worker",
            worker.path.as_posix(),
            "--manifest",
            admitted.manifest_path.as_posix(),
            "--fixture-id",
            admitted.config.fixture_id,
            "--profile-role",
            role,
            "--output",
            identity_path.as_posix(),
        )
        environment = _child_environment(interpreter, profile_identity=None)
        command = _command_document(
            argv,
            environment,
            interpreter,
            worker,
            output.root,
            collector=collector,
        )
        observation = _execute_process(argv, environment, output.root)
        retained = output.write_profile_preflight(
            role,
            command=command,
            stdout=observation.stdout,
            stderr=observation.stderr,
            exit_code=observation.exit_code,
            no_exit_status=observation.no_exit_status,
        )
        _require_success(observation, f"profile_preflight.{role}")
        try:
            raw = read_bounded_regular_file(identity_path, CONFIG_JSON_LIMITS.max_bytes)
            identity = load_native_profile_identity_v2(
                identity_path,
                expected_profile_role=role,
                expected_worker_sha256=worker.sha256,
            )
            if raw != canonical_json_bytes(cast(CanonicalValue, identity)) + b"\n":
                raise ProfileIdentityRefusal("profile identity is not canonical JSON")
            _bind_profile_identity_to_run(admitted, interpreter, identity)
            sentinel = _mapping(identity.get("sentinel"), "profile sentinel")
            if sentinel.get("status") != "PASS":
                raise ProfileIdentityRefusal(
                    "profile complete-integration-state sentinel did not pass"
                )
        except (JsonAdmissionError, ProfileIdentityRefusal, OSError, TypeError, ValueError) as exc:
            raise _ExecutionRefusal(
                OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
                f"profile_identities.{role}",
                str(exc),
                evidence={"exception_type": type(exc).__name__, "role": role},
            ) from exc
        completed.append(
            _AdmittedProfilePreflight(
                role=role,
                process=retained,
                identity_path=identity_path,
                identity_file_sha256=hashlib.sha256(raw).hexdigest(),
                identity=identity,
                interpreter=interpreter,
            )
        )
    try:
        require_compatible_profile_identities_v2(completed[0].identity, completed[1].identity)
    except ProfileIdentityRefusal as exc:
        raise _ExecutionRefusal(
            OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            "profile_identities",
            str(exc),
            evidence={"exception_type": type(exc).__name__},
        ) from exc
    return completed[0], completed[1]


def _run_evidence_cells(
    admitted: AdmittedRuntimeReviewRunConfiguration,
    output: OwnedRuntimeReviewRunOutput,
    worker: _ResourceIdentity,
    preflights: tuple[_AdmittedProfilePreflight, _AdmittedProfilePreflight],
) -> tuple[tuple[_AdmittedEvidenceAttempt, ...], bytes]:
    """Run and immediately admit each canonical evidence slot exactly once."""
    completed: list[_AdmittedEvidenceAttempt] = []
    provisional: AdmittedRuntimeReviewConfigurationV2 | None = None
    first_result: dict[str, object] | None = None
    profile_by_role = {preflight.role: preflight for preflight in preflights}
    for role in PROFILE_ROLES:
        interpreter = _role_interpreter(admitted, role)
        preflight = profile_by_role[role]
        for step_dt in STEP_DTS:
            for repeat_id in REPEAT_IDS:
                _recheck_before_process(admitted, worker)
                cell_path = output.new_evidence_cell_path(role, step_dt, repeat_id)
                argv = (
                    interpreter.lexical_path.as_posix(),
                    worker.path.as_posix(),
                    "--manifest",
                    admitted.manifest_path.as_posix(),
                    "--fixture-id",
                    admitted.config.fixture_id,
                    "--profile-role",
                    role,
                    "--step-dt",
                    step_dt,
                    "--repeat-id",
                    str(repeat_id),
                    "--output",
                    cell_path.as_posix(),
                )
                environment = _child_environment(
                    interpreter, profile_identity=preflight.identity_path
                )
                command = _command_document(
                    argv,
                    environment,
                    interpreter,
                    worker,
                    output.root,
                )
                observation = _execute_process(argv, environment, output.root)
                retained = output.write_evidence_attempt(
                    role,
                    step_dt,
                    repeat_id,
                    command=command,
                    stdout=observation.stdout,
                    stderr=observation.stderr,
                    exit_code=observation.exit_code,
                    no_exit_status=observation.no_exit_status,
                )
                _require_success(
                    observation,
                    f"evidence_attempt.{role}.{_STEP_DIRECTORY_TOKENS[step_dt]}.repeat_{repeat_id}",
                )
                try:
                    result_document, observed_result_sha256 = _load_worker_result(cell_path)
                    if provisional is None:
                        first_result = result_document
                        provisional = _provisional_review_configuration(
                            admitted, output.root, result_document, preflights
                        )
                    admitted_cell = _admit_one_cell(
                        provisional, role, step_dt, repeat_id, cell_path
                    )
                    _require_observed_result_unchanged(admitted_cell, observed_result_sha256)
                    _bind_cell_to_profile(
                        admitted,
                        admitted_cell,
                        preflight,
                        interpreter,
                    )
                except (JsonAdmissionError, TypeError, ValueError, OSError) as exc:
                    raise _ExecutionRefusal(
                        OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
                        "evidence_cells",
                        str(exc),
                        evidence={
                            "exception_type": type(exc).__name__,
                            "role": role,
                            "step_dt": step_dt,
                            "repeat_id": repeat_id,
                        },
                    ) from exc
                member_hashes = admitted_cell.member_sha256
                runtime = _mapping(admitted_cell.runtime, "admitted_cell.runtime")
                sentinel = _mapping(preflight.identity.get("sentinel"), "profile sentinel")
                completed.append(
                    _AdmittedEvidenceAttempt(
                        role=role,
                        package_version=_required_string(
                            preflight.identity.get("package_version"), "package_version"
                        ),
                        native_version=_required_string(
                            preflight.identity.get("native_version"), "native_version"
                        ),
                        native_version_integer=_required_integer(
                            preflight.identity.get("native_version_integer"),
                            "native_version_integer",
                        ),
                        profile_identity_sha256=_required_hash(
                            preflight.identity.get("profile_identity_sha256"),
                            "profile_identity_sha256",
                        ),
                        runtime_identity_sha256=_required_hash(
                            runtime.get("runtime_identity_sha256"),
                            "runtime_identity_sha256",
                        ),
                        sentinel_identity_sha256=_required_hash(
                            sentinel.get("sentinel_identity_sha256"),
                            "sentinel_identity_sha256",
                        ),
                        step_dt=step_dt,
                        repeat_id=repeat_id,
                        process=retained,
                        cell_path=cell_path,
                        result_sha256=member_hashes["result.json"],
                        checksum_manifest_sha256=member_hashes["CHECKSUMS.sha256"],
                        interpreter=interpreter,
                    )
                )
    if provisional is None or first_result is None:
        raise RuntimeError("canonical evidence loop completed without a first cell")
    generated = RuntimeReviewConfigV2.from_primitive(
        _generated_review_configuration_primitive(first_result, preflights)
    )
    if generated != provisional.config:
        raise RuntimeError("generated configuration differs from the cell-admission contract")
    return tuple(completed), canonical_json_bytes(generated.to_primitive()) + b"\n"


def _execute_process(
    argv: tuple[str, ...], environment: dict[str, str], working_directory: Path
) -> _CompletedProcessObservation:
    """Invoke the narrow seam once and preserve truthful no-exit process outcomes."""
    try:
        completed = _run_subprocess(
            argv,
            env=environment,
            timeout=_PROCESS_TIMEOUT_SECONDS,
            cwd=working_directory,
        )
    except subprocess.TimeoutExpired as exc:
        return _CompletedProcessObservation(
            _captured_bytes(exc.stdout),
            _captured_bytes(exc.stderr),
            None,
            "TIMEOUT",
        )
    except OSError:
        return _CompletedProcessObservation(b"", b"", None, "NOT_STARTED")
    return _CompletedProcessObservation(
        _captured_bytes(completed.stdout),
        _captured_bytes(completed.stderr),
        completed.returncode,
        None,
    )


def _run_subprocess(
    argv: tuple[str, ...],
    *,
    env: Mapping[str, str],
    timeout: int,
    cwd: Path,
) -> subprocess.CompletedProcess[bytes]:
    """Run one fixed child without a shell; this is the sole monkeypatchable process seam."""
    return subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        shell=False,
        env=dict(env),
        timeout=timeout,
        cwd=cwd,
    )


def _captured_bytes(value: bytes | str | None) -> bytes:
    """Normalize subprocess capture fields to exact retained bytes."""
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8", errors="replace")


def _require_success(observation: _CompletedProcessObservation, field: str) -> None:
    """Refuse a timeout, launch failure, or nonzero one-shot result without retrying."""
    if observation.no_exit_status == "TIMEOUT":
        raise _ExecutionRefusal(
            OperationalReasonCode.CHILD_PROCESS_SUPERVISION_FAILED,
            field,
            "child process exceeded the fixed timeout",
            evidence={"timeout_seconds": _PROCESS_TIMEOUT_SECONDS},
        )
    if observation.no_exit_status == "NOT_STARTED":
        raise _ExecutionRefusal(
            OperationalReasonCode.CHILD_PROCESS_START_FAILED,
            field,
            "child process could not be started",
        )
    if observation.exit_code != 0:
        raise _ExecutionRefusal(
            OperationalReasonCode.CHILD_PROCESS_SUPERVISION_FAILED,
            field,
            "child process returned a bounded refusal",
            evidence={"exit_code": observation.exit_code},
        )


def _child_environment(
    interpreter: InterpreterIdentity, *, profile_identity: Path | None
) -> dict[str, str]:
    """Build the closed deterministic environment without copying caller variables."""
    environment = {
        "PATH": f"{interpreter.lexical_path.parent.as_posix()}:/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    if profile_identity is not None:
        environment["METRIFID_NATIVE_UPGRADE_PROFILE_IDENTITY"] = profile_identity.as_posix()
    return environment


def _command_document(
    argv: tuple[str, ...],
    environment: Mapping[str, str],
    interpreter: InterpreterIdentity,
    worker: _ResourceIdentity,
    working_directory: Path,
    *,
    collector: _ResourceIdentity | None = None,
) -> dict[str, CanonicalValue]:
    """Build one canonical command record with exact argv, environment, and resource hashes."""
    document: dict[str, CanonicalValue] = {
        "schema": _PROCESS_COMMAND_SCHEMA,
        "schema_version": _PROCESS_COMMAND_SCHEMA_VERSION,
        "argv": list(argv),
        "environment": dict(environment),
        "timeout_seconds": _PROCESS_TIMEOUT_SECONDS,
        "shell": False,
        "working_directory": working_directory.as_posix(),
        "interpreter": interpreter.to_primitive(),
        "worker_sha256": worker.sha256,
        "collector_sha256": None if collector is None else collector.sha256,
    }
    return document


def _bind_profile_identity_to_run(
    admitted: AdmittedRuntimeReviewRunConfiguration,
    interpreter: InterpreterIdentity,
    identity: Mapping[str, Any],
) -> None:
    """Bind collector output to the admitted fixture, manifest, and launcher identity."""
    smoke = _mapping(identity.get("native_smoke"), "native_smoke")
    if smoke.get("fixture_id") != admitted.config.fixture_id:
        raise ProfileIdentityRefusal("profile smoke fixture differs from the admitted fixture")
    if smoke.get("manifest_raw_sha256") != admitted.manifest_identity.sha256:
        raise ProfileIdentityRefusal("profile smoke manifest differs from the admitted manifest")
    python = _mapping(identity.get("python"), "profile python")
    if (
        python.get("executable") != interpreter.lexical_path.as_posix()
        or python.get("resolved_executable") != interpreter.resolved_path.as_posix()
        or python.get("resolved_executable_sha256") != interpreter.resolved_sha256
    ):
        raise ProfileIdentityRefusal("profile identity does not bind the declared interpreter")


def _load_worker_result(cell_path: Path) -> tuple[dict[str, object], str]:
    """Load and hash one worker result before deriving accepted configuration semantics."""
    raw = read_bounded_regular_file(cell_path / "result.json", 64 * 1024 * 1024)
    from ._evidence import _strict_json_bytes

    return _strict_json_bytes(raw, "result.json"), hashlib.sha256(raw).hexdigest()


def _provisional_review_configuration(
    admitted: AdmittedRuntimeReviewRunConfiguration,
    output_root: Path,
    first_result: Mapping[str, object],
    preflights: tuple[_AdmittedProfilePreflight, _AdmittedProfilePreflight],
) -> AdmittedRuntimeReviewConfigurationV2:
    """Construct a strict in-memory referee config from the first admitted-result identities."""
    primitive = _generated_review_configuration_primitive(first_result, preflights)
    config = RuntimeReviewConfigV2.from_primitive(primitive)
    raw = canonical_json_bytes(config.to_primitive()) + b"\n"
    directories = tuple(output_root / cell.directory for cell in config.cells)
    identity_paths = cast(
        tuple[Path, Path],
        tuple(
            output_root / profile.identity_file
            for profile in (config.baseline_profile, config.candidate_profile)
        ),
    )
    return AdmittedRuntimeReviewConfigurationV2(
        config=config,
        path=output_root / "generated_runtime_review_config.json",
        base_dir=output_root,
        raw_bytes=raw,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        semantic_sha256=canonical_sha256(config.to_primitive()),
        cell_directories=directories,
        profile_identity_paths=identity_paths,
        profile_identity_file_sha256=cast(
            tuple[str, str],
            tuple(preflight.identity_file_sha256 for preflight in preflights),
        ),
        output_dir=output_root / "decision",
    )


def _generated_review_configuration_primitive(
    first_result: Mapping[str, object],
    preflights: tuple[_AdmittedProfilePreflight, _AdmittedProfilePreflight],
) -> dict[str, CanonicalValue]:
    """Derive the accepted Runtime Review identity fields from the first baseline result."""
    subject = _mapping(first_result.get("subject"), "result.subject")
    source_closure = _mapping(subject.get("source_closure"), "subject.source_closure")
    workload = _mapping(first_result.get("workload"), "result.workload")
    initial_state = _mapping(workload.get("initial_state"), "workload.initial_state")
    action_program = _mapping(workload.get("action_program"), "workload.action_program")
    fixture_id = _required_string(subject.get("fixture_id"), "subject.fixture_id")
    cells: list[CanonicalValue] = []
    for role in PROFILE_ROLES:
        for step_dt in STEP_DTS:
            for repeat_id in REPEAT_IDS:
                cells.append(
                    RuntimeReviewCellConfig(
                        profile_role=role,
                        step_dt=step_dt,
                        repeat_id=repeat_id,
                        directory=(
                            f"captured_evidence/{role}/{_STEP_DIRECTORY_TOKENS[step_dt]}"
                            f"/repeat_{repeat_id}"
                        ),
                    ).to_primitive()
                )
    profile_by_role = {preflight.role: preflight for preflight in preflights}

    def profile(role: str) -> dict[str, CanonicalValue]:
        """Project one fresh preflight into the role-based generated configuration."""
        identity = profile_by_role[role].identity
        return {
            "profile_role": role,
            "package_version": _required_string(
                identity.get("package_version"), f"{role}.package_version"
            ),
            "native_version": _required_string(
                identity.get("native_version"), f"{role}.native_version"
            ),
            "native_version_integer": _required_integer(
                identity.get("native_version_integer"), f"{role}.native_version_integer"
            ),
            "profile_identity_sha256": _required_hash(
                identity.get("profile_identity_sha256"),
                f"{role}.profile_identity_sha256",
            ),
            "identity_file": f"profile_identities/{role}.json",
        }

    return {
        "schema": CONFIG_SCHEMA,
        "schema_version": CONFIG_SCHEMA_VERSION_V2,
        "baseline_profile": profile("baseline"),
        "candidate_profile": profile("candidate"),
        "expected_subject": {
            "fixture_id": fixture_id,
            "source_closure_sha256": _required_hash(
                source_closure.get("closure_sha256"), "source_closure.closure_sha256"
            ),
            "fixture_manifest_sha256": _required_hash(
                subject.get("fixture_manifest_sha256"), "subject.fixture_manifest_sha256"
            ),
        },
        "expected_workload": {
            "semantic_sha256": _required_hash(
                workload.get("semantic_sha256"), "workload.semantic_sha256"
            ),
            "initial_state_semantic_sha256": _required_hash(
                initial_state.get("semantic_sha256"), "initial_state.semantic_sha256"
            ),
            "action_program_semantic_sha256": _required_hash(
                action_program.get("semantic_sha256"), "action_program.semantic_sha256"
            ),
        },
        "required_horizon": REQUIRED_HORIZON,
        "step_dts": list(STEP_DTS),
        "repeat_ids": list(REPEAT_IDS),
        "cells": cells,
        "output_dir": "decision",
    }


def _admit_one_cell(
    configuration: AdmittedRuntimeReviewConfigurationV2,
    role: str,
    step_dt: str,
    repeat_id: int,
    cell_path: Path,
) -> Any:
    """Invoke the existing deep single-cell evidence admission before any later slot."""
    slot = next(
        cell for cell in configuration.config.cells if cell.slot == (role, step_dt, repeat_id)
    )
    from ._evidence import _admit_cell

    return _admit_cell(configuration, slot, cell_path)


def _bind_cell_to_profile(
    admitted: AdmittedRuntimeReviewRunConfiguration,
    admitted_cell: Any,
    preflight: _AdmittedProfilePreflight,
    interpreter: InterpreterIdentity,
) -> None:
    """Bind one deeply admitted cell to its fresh profile identity and original manifest."""
    if admitted_cell.fixture_id != admitted.config.fixture_id:
        raise ValueError("evidence fixture differs from the admitted run fixture")
    if admitted_cell.manifest_raw_sha256 != admitted.manifest_identity.sha256:
        raise ValueError("evidence manifest bytes differ from the admitted run manifest")
    runtime = _mapping(admitted_cell.runtime, "admitted_cell.runtime")
    external = _mapping(runtime.get("external_profile_identity"), "external_profile_identity")
    sentinel = _mapping(preflight.identity.get("sentinel"), "profile sentinel")
    if (
        external.get("available") is not True
        or external.get("raw_sha256") != preflight.identity_file_sha256
        or external.get("profile_identity_sha256")
        != preflight.identity.get("profile_identity_sha256")
        or runtime.get("profile_identity_sha256")
        != preflight.identity.get("profile_identity_sha256")
        or runtime.get("sentinel_identity_sha256") != sentinel.get("sentinel_identity_sha256")
    ):
        raise ValueError("evidence runtime does not bind the fresh profile identity")
    runtime_python = _mapping(runtime.get("python"), "runtime.python")
    profile_python = _mapping(preflight.identity.get("python"), "profile.python")
    for field in (
        "executable",
        "resolved_executable",
        "resolved_executable_sha256",
        "version",
        "version_full",
        "implementation",
        "implementation_name",
        "compiler",
        "cache_tag",
    ):
        if runtime_python.get(field) != profile_python.get(field):
            raise ValueError(f"evidence Python identity differs from preflight: {field}")
    if (
        runtime_python.get("executable") != interpreter.lexical_path.as_posix()
        or runtime_python.get("resolved_executable") != interpreter.resolved_path.as_posix()
        or runtime_python.get("resolved_executable_sha256") != interpreter.resolved_sha256
        or runtime.get("host") != preflight.identity.get("host")
        or runtime.get("thread_environment") != preflight.identity.get("environment")
    ):
        raise ValueError("evidence runtime differs from its admitted launcher or host")
    _bind_runtime_measurements_to_profile(runtime, preflight.identity)


def _require_observed_result_unchanged(admitted_cell: Any, observed_sha256: str) -> None:
    """Bind first-result configuration derivation to the exact deeply admitted result bytes."""
    require_sha256(observed_sha256, "observed result SHA-256")
    if admitted_cell.member_sha256["result.json"] != observed_sha256:
        raise ValueError("worker result changed between derivation and admission")


def _bind_runtime_measurements_to_profile(
    runtime: Mapping[str, Any], profile: Mapping[str, Any]
) -> None:
    """Bind cell-side native, distribution, installation, and pip measurements to preflight."""
    runtime_mujoco = _mapping(runtime.get("mujoco"), "runtime.mujoco")
    profile_mujoco = _mapping(profile.get("mujoco"), "profile.mujoco")
    for field in ("package_version", "native_version", "native_version_integer"):
        if runtime_mujoco.get(field) != profile_mujoco.get(field):
            raise ValueError(f"evidence MuJoCo identity differs from preflight: {field}")
    if runtime_mujoco.get("loaded_native_library") != profile_mujoco.get("loaded_native_library"):
        raise ValueError("evidence native library identity differs from preflight")
    _bind_distribution_projection(
        runtime_mujoco.get("distribution"),
        profile_mujoco.get("distribution"),
        "MuJoCo",
    )
    runtime_numpy = _mapping(runtime.get("numpy"), "runtime.numpy")
    profile_numpy = _mapping(profile.get("numpy"), "profile.numpy")
    if runtime_numpy.get("python_version") != profile_numpy.get("python_version"):
        raise ValueError("evidence NumPy version differs from preflight")
    _bind_distribution_projection(
        runtime_numpy.get("distribution"),
        profile_numpy.get("distribution"),
        "NumPy",
    )
    if runtime.get("installation") != profile.get("installation"):
        raise ValueError("evidence installation observation differs from preflight")
    if runtime.get("pip_check") != profile.get("pip_check"):
        raise ValueError("evidence pip-check observation differs from preflight")


def _bind_distribution_projection(runtime: object, profile: object, field: str) -> None:
    """Require every compact preflight distribution field to match the cell measurement."""
    runtime_distribution = _mapping(runtime, f"runtime.{field}.distribution")
    profile_distribution = _mapping(profile, f"profile.{field}.distribution")
    if any(runtime_distribution.get(key) != value for key, value in profile_distribution.items()):
        raise ValueError(f"evidence {field} distribution identity differs from preflight")


def _strict_reload_generated_configuration(path: Path) -> AdmittedRuntimeReviewConfigurationV2:
    """Immediately reload the generated configuration through the existing strict loader."""
    from ._config import load_runtime_review_configuration

    try:
        loaded = load_runtime_review_configuration(path)
        if not isinstance(loaded, AdmittedRuntimeReviewConfigurationV2):
            raise ValueError("new Runtime Review execution must generate schema version 2")
        return loaded
    except (JsonAdmissionError, PathAdmissionError, OSError, TypeError, ValueError) as exc:
        raise _ExecutionRefusal(
            OperationalReasonCode.INTERNAL_INVARIANT_FAILED,
            "generated_runtime_review_config",
            str(exc),
            evidence={"exception_type": type(exc).__name__},
        ) from exc


def _call_runtime_reviewer(path: Path) -> RuntimeReviewResult:
    """Call the existing scientific Runtime Review referee exactly once."""
    from ._run import review_runtime_configuration_file

    return review_runtime_configuration_file(path)


def _build_completed_run_record(
    admitted: AdmittedRuntimeReviewRunConfiguration,
    output: OwnedRuntimeReviewRunOutput,
    worker: _ResourceIdentity,
    collector: _ResourceIdentity,
    preflights: tuple[_AdmittedProfilePreflight, _AdmittedProfilePreflight],
    attempts: tuple[_AdmittedEvidenceAttempt, ...],
    generated_sha256: str,
    result: RuntimeReviewResult,
) -> dict[str, CanonicalValue]:
    """Construct the operational record around the existing validated scientific result."""
    reason = None if result.reason_code is None else result.reason_code.value
    return {
        "schema": RUN_RECORD_SCHEMA,
        "schema_version": RUN_RECORD_SCHEMA_VERSION_V2,
        "status": result.status.value,
        "reason_code": reason,
        "exit_code": result.exit_code,
        "input_configuration": {
            "locator": output.admitted_configuration.locator,
            "raw_sha256": admitted.raw_sha256,
            "semantic_sha256": admitted.semantic_sha256,
        },
        "packaged_worker": worker.to_primitive(),
        "profile_identity_collector": collector.to_primitive(),
        "profile_preflights": [item.to_primitive(output.root) for item in preflights],
        "evidence_attempts": [item.to_primitive(output.root) for item in attempts],
        "generated_runtime_review_config": {
            "locator": output.generated_runtime_review_config.relative_to(output.root).as_posix(),
            "sha256": generated_sha256,
        },
        "runtime_review_receipt": {
            "locator": result.runtime_review_json.relative_to(output.root).as_posix(),
            "file_sha256": hashlib.sha256(
                read_bounded_regular_file(result.runtime_review_json, 64 * 1024 * 1024)
            ).hexdigest(),
            "receipt_sha256": result.receipt_sha256,
        },
        "claim_limitations": list(RUNTIME_REVIEW_RUN_CLAIM_LIMITATIONS),
        "run_sha256": None,
    }


def _recheck_before_process(
    admitted: AdmittedRuntimeReviewRunConfiguration,
    worker: _ResourceIdentity,
    collector: _ResourceIdentity | None = None,
) -> None:
    """Recheck all caller inputs and process resources immediately before one attempt."""
    try:
        recheck_runtime_review_run_inputs(admitted)
        _recheck_resource(worker)
        if collector is not None:
            _recheck_resource(collector)
    except (JsonAdmissionError, PathAdmissionError, OSError, TypeError, ValueError) as exc:
        raise _ExecutionRefusal(
            OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            "runtime_review_run_inputs",
            str(exc),
            evidence={"exception_type": type(exc).__name__},
        ) from exc


def _recheck_final_inputs(
    admitted: AdmittedRuntimeReviewRunConfiguration,
    worker: _ResourceIdentity,
    collector: _ResourceIdentity,
) -> None:
    """Recheck launchers, manifest, and installed resources before publication and return."""
    _recheck_before_process(admitted, worker, collector)


def _measure_resource(path: Path, field: str) -> _ResourceIdentity:
    """Read and hash one bounded regular nonsymlink installed resource."""
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{field} must be a regular nonsymlink installed resource")
    payload = read_bounded_regular_file(path, _RESOURCE_BYTE_LIMIT)
    return _ResourceIdentity(path, hashlib.sha256(payload).hexdigest())


def _recheck_resource(resource: _ResourceIdentity) -> None:
    """Remeasure one installed resource and refuse byte or locator substitution."""
    observed = _measure_resource(resource.path, resource.path.name)
    if observed != resource:
        raise ValueError(f"installed resource changed during execution: {resource.path.name}")
    if resource.path.name == _WORKER_RESOURCE_NAME and resource.sha256 != _FROZEN_WORKER_SHA256:
        raise ValueError("packaged native evidence worker differs from frozen bytes")


def _role_interpreter(
    admitted: AdmittedRuntimeReviewRunConfiguration, role: str
) -> InterpreterIdentity:
    """Return one semantic role's explicitly admitted interpreter launcher."""
    if role == "baseline":
        return admitted.baseline_interpreter
    if role == "candidate":
        return admitted.candidate_interpreter
    raise ValueError("role must be baseline or candidate")


def _mapping(value: object, field: str) -> dict[str, Any]:
    """Require one concrete JSON object for orchestration-level identity binding."""
    if type(value) is not dict:
        raise ValueError(f"{field} must be an object")
    return cast(dict[str, Any], value)


def _required_string(value: object, field: str) -> str:
    """Require one nonempty exact string from an admitted worker result."""
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _required_hash(value: object, field: str) -> str:
    """Require one lowercase SHA-256 token from an admitted worker result."""
    return require_sha256(value, field)


def _required_integer(value: object, field: str) -> int:
    """Require one exact nonnegative integer and reject boolean aliases."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _tool() -> OperationalToolObservation:
    """Observe the installed distribution that owns one bounded operational refusal."""
    return OperationalToolObservation(
        __version__, "VERIFIED_INSTALLED_DISTRIBUTION", installed_distribution_sha256()
    )


def _refuse(
    code: OperationalReasonCode,
    *,
    field: str,
    **evidence: CanonicalValue,
) -> ComparisonOperationError:
    """Construct one canonical exit-64 operational failure for the execution journey."""
    if code in {
        OperationalReasonCode.CHILD_PROCESS_START_FAILED,
        OperationalReasonCode.CHILD_PROCESS_SUPERVISION_FAILED,
    }:
        evidence = dict(evidence)
        evidence["child_process_reason_code"] = code.value
        code = OperationalReasonCode.OUTPUT_WRITE_FAILED
    return operational_error(
        tool=_tool(),
        code=code,
        role=None,
        evidence=dict(evidence),
        field=field,
        operation=_OPERATION,
    )


__all__ = ["RuntimeReviewRunResult", "run_runtime_review_configuration_file"]
