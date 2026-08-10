"""Immutable campaign execution and descriptor-confined audit publication."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from ._atomic_output import (
    PairedOutputDirectory,
    PairedOutputNames,
    _publish_paired_results,
    prepare_paired_output_directory,
)
from ._audit_artifacts import AuditArtifactRegistry, AuditOwnedDirectory
from ._audit_config import (
    OPERATION,
    AuditAbort,
    AuditConfig,
    _abort,
    _compiled_timestep,
    _parse_config,
    _rational_lt,
    _rational_primitive,
    _require_output_outside_model_root,
    _steps_per_control,
    _tool,
)
from ._audit_reporting import (
    _STATUS_CLASSIFICATION,
    INCONCLUSIVE,
    REFUSED,
    _aggregate,
    _candidate_row,
    _render_markdown,
    candidate_token,
)
from ._json_admission import (
    CONFIG_JSON_LIMITS,
    JsonAdmissionError,
    read_bounded_regular_file,
)
from ._model_closure import (
    ModelAdmissionRefusal,
    ModelClosureSnapshot,
    create_model_closure_snapshot,
    verify_model_closure_unchanged,
)
from ._npz import ArtifactAdmissionRefusal, _read_bounded_bytes, refuse
from ._owned_artifacts import RetainedArtifactPair
from ._workload import WorkloadArtifacts, _load_workload_artifacts_from_bytes
from .compare._failure import ComparisonOperationError, operational_error
from .compare._model_pair import LiveModelPair, open_snapshot_model_pair
from .compare._output import COMPARISON_OUTPUT_NAMES, OutputDirectory
from .errors import ReasonRole
from .json_values import CanonicalValue, ExactRational, canonical_json_bytes
from .operational import OperationalReasonCode, OperationalToolObservation
from .schemas import ComparisonConfig

if TYPE_CHECKING:
    from .compare._orchestrator import ComparisonRunResult

_WORKSPACE_DIR: Final = ".audit_workspace"
_CANDIDATES_DIR: Final = "candidates"
_AUDIT_OUTPUT_NAMES = PairedOutputNames("timestep_audit.json", "timestep_audit.md")


@dataclass(frozen=True, slots=True)
class AuditRunResult:
    """Return a completed aggregate with its published JSON and Markdown paths."""

    audit_json: Path
    audit_markdown: Path
    aggregate: dict[str, CanonicalValue]


@dataclass(frozen=True, slots=True)
class _AuditInputPaths:
    """The snapshot-bound source and stable live workload paths for one campaign."""

    snapshot: ModelClosureSnapshot
    source_root: Path
    state_path: Path
    actions_path: Path
    source_tree_sha256: str


@dataclass(frozen=True, slots=True)
class _FrozenAuditInputs:
    """One campaign-owned source snapshot and exact bounded workload byte pair."""

    paths: _AuditInputPaths
    state_raw: bytes
    actions_raw: bytes

    @property
    def snapshot(self) -> ModelClosureSnapshot:
        """Return the campaign's only retained model snapshot."""
        return self.paths.snapshot

    @property
    def source_root(self) -> Path:
        """Return the canonical admitted source-root path."""
        return self.paths.source_root

    @property
    def state_path(self) -> Path:
        """Return the stable live state path used only for final verification."""
        return self.paths.state_path

    @property
    def actions_path(self) -> Path:
        """Return the stable live actions path used only for final verification."""
        return self.paths.actions_path

    @property
    def source_tree_sha256(self) -> str:
        """Return the source-tree digest derived from the frozen closure identity."""
        return self.paths.source_tree_sha256


def audit_configuration_file(config_path: str | Path) -> AuditRunResult:
    """Run one strict timestep audit over a single immutable admitted input campaign."""
    tool = _tool()
    path = Path(config_path)
    raw = _read_audit_configuration(path, tool)
    config = _parse_config(raw, tool)
    base = path.absolute().parent
    snapshot = _create_campaign_snapshot(base, config, tool)
    with snapshot:
        inputs = _campaign_input_paths(base, config, snapshot, tool)
        output = _admitted_output_directory(base, config, tool)
        return _execute_admitted_audit(raw, config, inputs, output, tool)


def _read_audit_configuration(path: Path, tool: OperationalToolObservation) -> bytes:
    """Read the declared audit configuration or raise its typed I/O failure."""
    try:
        return read_bounded_regular_file(path, CONFIG_JSON_LIMITS.max_bytes)
    except (OSError, JsonAdmissionError) as exc:
        raise _abort(
            tool,
            OperationalReasonCode.CONFIGURATION_IO_FAILED,
            {"exception_type": type(exc).__name__},
            field="timestep_audit_config",
        ) from exc


def _audit_admission_abort(
    tool: OperationalToolObservation,
    exc: ModelAdmissionRefusal | ArtifactAdmissionRefusal,
    *,
    field: str | None = None,
) -> AuditAbort:
    """Preserve one admission refusal at the audit operation boundary."""
    return AuditAbort(
        operational_error(
            tool=tool,
            code=exc.reason,
            role=cast(ReasonRole, exc.role),
            evidence=cast("Mapping[str, CanonicalValue]", exc.evidence),
            field=field,
            operation=OPERATION,
        )
    )


def _create_campaign_snapshot(
    base: Path,
    config: AuditConfig,
    tool: OperationalToolObservation,
) -> ModelClosureSnapshot:
    """Create the audit's only complete source-closure snapshot before any output."""
    declared_root = (base / config.model_root).absolute()
    try:
        return create_model_closure_snapshot(declared_root, config.entrypoint, "comparison")
    except ModelAdmissionRefusal as exc:
        raise _audit_admission_abort(tool, exc) from exc


def _campaign_input_paths(
    base: Path,
    config: AuditConfig,
    snapshot: ModelClosureSnapshot,
    tool: OperationalToolObservation,
) -> _AuditInputPaths:
    """Bind stable source/workload path spellings without reading workload content."""
    source_root = snapshot.source_root.resolve(strict=True)
    _require_output_outside_model_root(base, config, source_root, tool)
    state_path = (base / config.initial_state).absolute()
    actions_path = (base / config.actions).absolute()
    return _AuditInputPaths(
        snapshot,
        source_root,
        state_path,
        actions_path,
        _snapshot_tree_digest(snapshot),
    )


def _freeze_workload_inputs(inputs: _AuditInputPaths) -> _FrozenAuditInputs:
    """Read each bounded live workload file exactly once before the first candidate."""
    state_raw = _read_bounded_bytes(inputs.state_path, OperationalReasonCode.STATE_ARTIFACT_INVALID)
    actions_raw = _read_bounded_bytes(
        inputs.actions_path, OperationalReasonCode.ACTIONS_ARTIFACT_INVALID
    )
    return _FrozenAuditInputs(inputs, state_raw, actions_raw)


def _snapshot_tree_digest(snapshot: ModelClosureSnapshot) -> str:
    """Preserve the audit tree-digest meaning using only frozen closure identity."""
    members: list[CanonicalValue] = [
        {"path": member.path, "sha256": member.sha256} for member in snapshot.identity.members
    ]
    return hashlib.sha256(canonical_json_bytes(cast(CanonicalValue, members))).hexdigest()


def _admitted_output_directory(
    base: Path, config: AuditConfig, tool: OperationalToolObservation
) -> PairedOutputDirectory:
    """Admit and retain the declared audit output directory descriptor."""
    try:
        return prepare_paired_output_directory(base / config.output_dir, _AUDIT_OUTPUT_NAMES)
    except ArtifactAdmissionRefusal as exc:
        raise _audit_admission_abort(tool, exc) from exc


def _admitted_reference(
    config: AuditConfig,
    live: LiveModelPair,
    tool: OperationalToolObservation,
) -> tuple[ExactRational, int]:
    """Read the reference timestep from the campaign's retained baseline model."""
    reference_dt = _compiled_timestep(float(live.baseline_model.opt.timestep), tool)
    reference_steps = _steps_per_control(config.control_dt, reference_dt)
    if reference_steps is None:
        raise _abort(
            tool,
            OperationalReasonCode.CONTROL_GRID_NONINTEGRAL,
            {
                "message": "reference timestep does not divide control_dt exactly",
                "control_dt": _rational_primitive(config.control_dt),
                "reference_step_dt": _rational_primitive(reference_dt),
            },
            field="control_dt",
        )
    for candidate in config.candidate_step_dts:
        if not _rational_lt(reference_dt, candidate):
            raise _abort(
                tool,
                OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
                {
                    "message": "every candidate timestep must exceed the reference timestep",
                    "reference_step_dt": _rational_primitive(reference_dt),
                    "candidate_step_dt": _rational_primitive(candidate),
                },
                field="candidate_step_dts",
            )
    return reference_dt, reference_steps


def _execute_admitted_audit(
    raw: bytes,
    config: AuditConfig,
    inputs: _AuditInputPaths,
    output: PairedOutputDirectory,
    tool: OperationalToolObservation,
) -> AuditRunResult:
    """Retain every audit artifact through final live-input and public-tree checks."""
    registry = AuditArtifactRegistry(output)
    try:
        workspace = registry.create_directory((), _WORKSPACE_DIR, group="private")
        candidates = registry.create_directory((), _CANDIDATES_DIR, group="candidate")
        with open_snapshot_model_pair(inputs.snapshot) as live:
            reference_dt, reference_steps = _admitted_reference(config, live, tool)
            campaign = _freeze_workload_inputs(inputs)
            workload = _load_workload_artifacts_from_bytes(
                campaign.state_raw, campaign.actions_raw, live.identity
            )
            rows = _run_candidates(
                config,
                campaign,
                registry,
                workspace,
                candidates,
                live,
                workload,
                reference_dt,
                reference_steps,
            )
            aggregate = _aggregate(
                config,
                raw,
                tool,
                reference_dt,
                reference_steps,
                rows,
                campaign.source_tree_sha256,
                workload,
            )
            _verify_before_aggregate(campaign, registry)
            registry.remove_private()
            registry.verify_public_tree()
            audit_json, audit_md, retained = _publish_audit(output, aggregate)
            registry.register_pair((), retained, group="aggregate")
        _verify_before_success(campaign, registry)
        result = AuditRunResult(audit_json, audit_md, aggregate)
    except (ModelAdmissionRefusal, ArtifactAdmissionRefusal) as exc:
        registry.cleanup()
        output.close()
        raise _audit_admission_abort(tool, exc) from exc
    except BaseException:
        registry.cleanup()
        output.close()
        raise
    registry.close()
    output.close()
    return result


def _verify_before_aggregate(campaign: _FrozenAuditInputs, registry: AuditArtifactRegistry) -> None:
    """Verify live inputs and candidate evidence before private workspace removal."""
    _verify_campaign_live(campaign)
    registry.verify_candidate_evidence()


def _verify_before_success(campaign: _FrozenAuditInputs, registry: AuditArtifactRegistry) -> None:
    """Verify inputs and retained bytes, then check the public path and tree last."""
    _verify_campaign_live(campaign)
    registry.verify_candidate_evidence()
    registry.verify_aggregate()
    registry.verify_public_tree()


def _verify_campaign_live(campaign: _FrozenAuditInputs) -> None:
    """Reverify all three live inputs against the frozen campaign immediately at commit."""
    verify_model_closure_unchanged(campaign.snapshot, "comparison")
    state_now = _read_bounded_bytes(
        campaign.state_path, OperationalReasonCode.STATE_ARTIFACT_INVALID
    )
    actions_now = _read_bounded_bytes(
        campaign.actions_path, OperationalReasonCode.ACTIONS_ARTIFACT_INVALID
    )
    if state_now != campaign.state_raw:
        raise refuse(
            OperationalReasonCode.STATE_ARTIFACT_INVALID,
            issue="artifact_changed_since_campaign_start",
        )
    if actions_now != campaign.actions_raw:
        raise refuse(
            OperationalReasonCode.ACTIONS_ARTIFACT_INVALID,
            issue="artifact_changed_since_campaign_start",
        )


def _candidate_comparison_config(
    config: AuditConfig,
    campaign: _FrozenAuditInputs,
    reference_dt: ExactRational,
    candidate: ExactRational,
    case_out: Path,
) -> dict[str, CanonicalValue]:
    """Build one internal candidate configuration while retaining frozen path spellings."""
    role = {"model_root": str(campaign.source_root), "entrypoint": config.entrypoint}
    return {
        "schema_version": 1,
        "baseline": {**role, "declared_step_dt": reference_dt.to_decimal_token()},
        "candidate": {**role, "declared_step_dt": candidate.to_decimal_token()},
        "initial_state": str(campaign.state_path),
        "actions": str(campaign.actions_path),
        "control_dt": config.control_dt.to_decimal_token(),
        "repeats": config.repeats,
        "joint_tolerances": cast("dict[str, CanonicalValue]", dict(config.joint_tolerances)),
        "aliases": None,
        "output_dir": str(case_out),
    }


def _run_candidates(
    config: AuditConfig,
    campaign: _FrozenAuditInputs,
    registry: AuditArtifactRegistry,
    workspace: AuditOwnedDirectory,
    candidates: AuditOwnedDirectory,
    live: LiveModelPair,
    workload: WorkloadArtifacts,
    reference_dt: ExactRational,
    reference_steps: int,
) -> list[dict[str, CanonicalValue]]:
    """Attempt every candidate while registering every created directory and file."""
    return [
        _run_candidate(
            config,
            campaign,
            registry,
            workspace,
            candidates,
            live,
            workload,
            reference_dt,
            reference_steps,
            candidate,
        )
        for candidate in config.candidate_step_dts
    ]


def _run_candidate(
    config: AuditConfig,
    campaign: _FrozenAuditInputs,
    registry: AuditArtifactRegistry,
    workspace: AuditOwnedDirectory,
    candidates: AuditOwnedDirectory,
    live: LiveModelPair,
    workload: WorkloadArtifacts,
    reference_dt: ExactRational,
    reference_steps: int,
    candidate: ExactRational,
) -> dict[str, CanonicalValue]:
    """Execute one candidate and transfer its retained evidence to the audit registry."""
    from .compare._orchestrator import _compare_frozen_campaign_candidate

    token = candidate_token(candidate)
    steps = _steps_per_control(config.control_dt, candidate)
    case_dir = registry.create_directory(candidates.key, token, group="candidate")
    work_dir = registry.create_directory(workspace.key, token, group="private")
    case_output = OutputDirectory(registry.paired_output(case_dir, COMPARISON_OUTPUT_NAMES))
    try:
        primitive = _candidate_comparison_config(
            config, campaign, reference_dt, candidate, case_dir.path
        )
        config_raw = canonical_json_bytes(cast(CanonicalValue, primitive)) + b"\n"
        config_path = work_dir.path / "comparison.json"
        registry.write_file(work_dir.key, "comparison.json", config_raw, group="private")
        comparison_config = ComparisonConfig.from_primitive(primitive)
        try:
            result = _compare_frozen_campaign_candidate(
                config_path=config_path,
                config_raw=config_raw,
                config=comparison_config,
                candidate_step_dt=candidate,
                live=live,
                workload=workload,
                output=case_output,
            )
        except ComparisonOperationError as exc:
            return _failed_candidate_row(
                exc, token, candidate, steps, reference_steps, registry, case_dir
            )
        retained = case_output._take_retained_pair()
        registry.register_pair(case_dir.key, retained, group="candidate")
        return _completed_candidate_row(
            result, token, candidate, steps, reference_steps, registry.root.path
        )
    finally:
        case_output._paired().close()


def _failed_candidate_row(
    exc: ComparisonOperationError,
    token: str,
    candidate: ExactRational,
    steps: int | None,
    reference_steps: int,
    registry: AuditArtifactRegistry,
    case_dir: AuditOwnedDirectory,
) -> dict[str, CanonicalValue]:
    """Register one comparison refusal through its exclusively created case directory."""
    failure_path = case_dir.path / "operational_failure.json"
    registry.write_file(
        case_dir.key,
        failure_path.name,
        canonical_json_bytes(cast(CanonicalValue, exc.failure.to_primitive())) + b"\n",
        group="candidate",
    )
    skippable = (
        exc.failure.reason.code is OperationalReasonCode.CONTROL_GRID_NONINTEGRAL
        and exc.failure.reason.role == "candidate"
        and steps is None
    )
    return _candidate_row(
        token,
        candidate,
        steps,
        reference_steps,
        classification=REFUSED if skippable else INCONCLUSIVE,
        operational_reason=exc.failure.reason.code.value,
        failure_sha256=exc.failure.failure_sha256,
        operational_failure_json=_relative_output_path(failure_path, registry.root.path),
    )


def _completed_candidate_row(
    result: ComparisonRunResult,
    token: str,
    candidate: ExactRational,
    steps: int | None,
    reference_steps: int,
    output_root: Path,
) -> dict[str, CanonicalValue]:
    """Build one completed candidate row from its frozen-input comparison result."""
    receipt = result.receipt.to_primitive()
    status = cast(str, receipt["status"])
    return _candidate_row(
        token,
        candidate,
        steps,
        reference_steps,
        classification=_STATUS_CLASSIFICATION[status],
        comparison_status=status,
        receipt=receipt,
        comparison_json=_relative_output_path(Path(result.comparison_json), output_root),
        comparison_markdown=_relative_output_path(Path(result.comparison_markdown), output_root),
    )


def _relative_output_path(path: Path, output_root: Path) -> str:
    """Render one stable POSIX result path relative to the public audit root."""
    return str(path.relative_to(output_root)).replace("\\", "/")


def _publish_audit(
    output: PairedOutputDirectory,
    aggregate: Mapping[str, CanonicalValue],
) -> tuple[Path, Path, RetainedArtifactPair]:
    """Publish the aggregate pair and retain it for final audit verification."""
    retained = _publish_paired_results(
        output,
        json_bytes=canonical_json_bytes(cast(CanonicalValue, aggregate)) + b"\n",
        markdown_text=_render_markdown(aggregate),
        require_empty=False,
    )
    return output.json_path, output.markdown_path, retained
