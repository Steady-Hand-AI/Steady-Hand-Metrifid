"""Installed comparison comparison orchestration over accepted model admission/artifact admission contracts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .._json_admission import (
    CONFIG_JSON_LIMITS,
    JsonAdmissionError,
    bounded_strict_json_loads,
    read_bounded_regular_file,
)
from .._model_closure import ModelAdmissionRefusal
from .._npz import ArtifactAdmissionRefusal, refuse
from .._timegrid import build_time_grid
from .._workload import load_workload_artifacts
from ..distribution import installed_distribution_sha256
from ..errors import (
    EngineThreadpoolState,
    ReasonCode,
    ReasonRecord,
    ReasonRole,
)
from ..json_values import (
    CanonicalValue,
    ExactRational,
    canonical_json_bytes,
)
from ..operational import (
    InputDigest,
    InputDigestCode,
    OperationalReasonCode,
    OperationalToolObservation,
)
from ..schemas import (
    ComparisonConfig,
    ComparisonReceipt,
    MonitoredJoint,
)
from ..version import __version__
from ._budget import evaluate_preexecution_budgets
from ._environment import build_environment_identity, combine_threadpool_states
from ._failure import ComparisonOperationError, operational_error
from ._markdown import render_markdown
from ._metrics import evaluate_role_pair
from ._model_pair import open_live_model_pair
from ._monitoring import monitored_joints_from_config
from ._mujoco_backend import MuJoCoBackend, observed_threadpool_state
from ._output import (
    OutputDirectory,
    cleanup_output_after_failure,
    prepare_output_directory,
    publish_results,
)
from ._runner import ActionsArtifact, StateArtifact, run_role_repeats
from ._trace import Role, RoleRepeatSet

if TYPE_CHECKING:
    # annotations only; avoids importing the identity module at runtime
    from collections.abc import Mapping, Sequence

    from .._model_identity import ModelPairIdentity
    from .._timegrid import TimeGrid
    from .._workload import WorkloadArtifacts
    from ..schemas import ComparisonContractIdentity, EnvironmentIdentity
    from ._model_pair import LiveModelPair
from ._orchestration_receipt import (
    _comparison_contract,
    _comparison_reason,
    _comparison_receipt,
    _preexecution_metrics,
    _preexecution_numerical_evidence,
    _preexecution_repeatability,
)


@dataclass(frozen=True, slots=True)
class ComparisonRunResult:
    """Return a completed receipt together with its published JSON and Markdown paths."""

    receipt: ComparisonReceipt
    comparison_json: Path
    comparison_markdown: Path


def compare_configuration_file(config_path: str | Path) -> ComparisonRunResult:
    """Run one strict JSON comparison from the installed distribution.

    The public comparator applies no timestep override. A configuration whose declared candidate
    timestep disagrees with the compiled model is still refused by the accepted time contract.
    """
    return _run_comparison(config_path, None)


def _compare_configuration_file_with_candidate_timestep(
    config_path: str | Path,
    candidate_step_dt: ExactRational,
) -> ComparisonRunResult:
    """Run one comparison whose candidate role is stepped at `candidate_step_dt`.

    This is the private seam the timestep audit uses so both roles compile the exact admitted
    user model. MuJoCo documents `option` as mapping to `mjModel.opt`: it is runtime simulation
    state that does not affect compilation, so only the candidate model's `opt.timestep` is
    assigned and nothing is serialized, rewritten, or copied.

    It is deliberately private: no `__all__` entry, no package export, no CLI argument, and no
    public schema. It fails closed when the requested timestep is not the configuration's own
    declared candidate timestep, or when the two roles are not the same source model.
    """
    return _run_comparison(config_path, candidate_step_dt)


def _compare_frozen_campaign_candidate(
    *,
    config_path: Path,
    config_raw: bytes,
    config: ComparisonConfig,
    candidate_step_dt: ExactRational,
    live: LiveModelPair,
    workload: WorkloadArtifacts,
    output: OutputDirectory,
) -> ComparisonRunResult:
    """Compare one audit candidate using only campaign-owned model and workload objects."""
    distribution_sha = installed_distribution_sha256()
    tool = _comparison_tool(distribution_sha)
    inputs = [
        InputDigest(InputDigestCode.CONFIGURATION_RAW, hashlib.sha256(config_raw).hexdigest())
    ]
    try:
        _validate_candidate_override(config, candidate_step_dt)
        if config.aliases is not None:
            raise ValueError("a frozen audit campaign cannot load a live aliases file")
        receipt = _compare_within_live_pair(
            live=live,
            config=config,
            config_raw=config_raw,
            base=config_path.absolute().parent,
            candidate_step_dt=candidate_step_dt,
            distribution_sha=distribution_sha,
            input_digests=inputs,
            frozen_workload=workload,
        )
        publish_results(
            output,
            json_bytes=canonical_json_bytes(receipt.to_primitive()),
            markdown_text=render_markdown(receipt, config_path.absolute()),
        )
        return ComparisonRunResult(receipt, output.json_path, output.markdown_path)
    except ComparisonOperationError:
        output._cleanup_retained_pair()
        raise
    except (ModelAdmissionRefusal, ArtifactAdmissionRefusal) as exc:
        output._cleanup_retained_pair()
        raise operational_error(
            tool=tool,
            code=exc.reason,
            role=cast(ReasonRole, exc.role),
            evidence=cast("Mapping[str, CanonicalValue]", exc.evidence),
            available_inputs=inputs,
        ) from exc
    except Exception as exc:
        output._cleanup_retained_pair()
        raise operational_error(
            tool=tool,
            code=OperationalReasonCode.INTERNAL_INVARIANT_FAILED,
            role=None,
            evidence={"exception_type": type(exc).__name__, "message": str(exc)},
            available_inputs=inputs,
        ) from exc


@dataclass(slots=True)
class _ReplayEvidence:
    """The five evidence members a comparison produces, however it produced them."""

    reasons: list[ReasonRecord]
    repeatability: object
    numerical_evidence: object
    metrics: object
    first_crossing: object


def _apply_candidate_timestep_override(
    live: LiveModelPair, candidate_step_dt: ExactRational | None
) -> None:
    """Apply the private same-model timestep override, or fail closed."""
    if candidate_step_dt is None:
        return
    pair = live.identity
    if pair.baseline_closure.sha256() != pair.candidate_closure.sha256():
        # The private path exists only for a same-source comparison. Overriding the candidate
        # timestep across two different models would silently compare a different model at a
        # different timestep. Fail closed before replay.
        raise ValueError(
            "candidate timestep override requires identical baseline and candidate model closures"
        )
    # `mjOption` is runtime state, not a compile input. Assign only this field, only on the
    # candidate, and only before any backend, MjData, threadpool observation, budget evaluation,
    # or replay object exists.
    live.candidate_model.opt.timestep = float(
        Fraction(candidate_step_dt.numerator, candidate_step_dt.denominator)
    )


def _replay_evidence(
    *,
    budget_reasons: Sequence[ReasonRecord],
    config: ComparisonConfig,
    pair: ModelPairIdentity,
    monitored: tuple[MonitoredJoint, ...],
    workload: WorkloadArtifacts,
    grid: TimeGrid,
    baseline_backend: MuJoCoBackend,
    candidate_backend: MuJoCoBackend,
) -> _ReplayEvidence:
    """Replay both roles, or report the preexecution evidence when a budget already refused."""
    if budget_reasons:
        return _ReplayEvidence(
            reasons=list(budget_reasons),
            repeatability=_preexecution_repeatability(config.repeats),
            numerical_evidence=_preexecution_numerical_evidence(grid.boundary_count),
            metrics=_preexecution_metrics(),
            first_crossing=None,
        )
    roles: dict[str, RoleRepeatSet] = {}
    pairs: tuple[tuple[Role, MuJoCoBackend], ...] = (
        ("baseline", baseline_backend),
        ("candidate", candidate_backend),
    )
    for role, backend in pairs:
        roles[role] = run_role_repeats(
            backend=backend,
            role=role,
            joints=pair.alignment.joints,
            actuators=pair.alignment.actuators,
            state=cast("StateArtifact", workload.state),
            actions=cast("ActionsArtifact", workload.actions),
            time_grid=grid,
            repeats=config.repeats,
        )
    evaluation = evaluate_role_pair(
        baseline=roles["baseline"],
        candidate=roles["candidate"],
        joints=pair.alignment.joints,
        monitored_joints=monitored,
        time_grid=grid,
    )
    return _ReplayEvidence(
        reasons=list(evaluation.reasons),
        repeatability=evaluation.repeatability,
        numerical_evidence=evaluation.numerical_evidence,
        metrics=evaluation.metrics,
        first_crossing=evaluation.first_crossing,
    )


def _compare_within_live_pair(
    *,
    live: LiveModelPair,
    config: ComparisonConfig,
    config_raw: bytes,
    base: Path,
    candidate_step_dt: ExactRational | None,
    distribution_sha: str,
    input_digests: list[InputDigest],
    frozen_workload: WorkloadArtifacts | None = None,
) -> ComparisonReceipt:
    """Everything that happens while both compiled models are alive."""
    _apply_candidate_timestep_override(live, candidate_step_dt)
    pair, monitored, workload, grid, contract = _admitted_comparison_inputs(
        live, config, base, input_digests, frozen_workload
    )
    environment, evidence = _runtime_evidence(live, config, pair, monitored, workload, grid)
    return _comparison_receipt(
        distribution_sha=distribution_sha,
        evidence=evidence,
        environment=environment,
        config_raw=config_raw,
        contract=contract,
        pair=pair,
        workload=workload,
        monitored=monitored,
        grid=grid,
    )


def _admitted_comparison_inputs(
    live: LiveModelPair,
    config: ComparisonConfig,
    base: Path,
    input_digests: list[InputDigest],
    frozen_workload: WorkloadArtifacts | None = None,
) -> tuple[
    ModelPairIdentity,
    tuple[MonitoredJoint, ...],
    WorkloadArtifacts,
    TimeGrid,
    ComparisonContractIdentity,
]:
    """Load admitted workload inputs and construct their exact time and comparison contracts."""
    pair = live.identity
    input_digests.extend(
        [
            InputDigest(InputDigestCode.BASELINE_MODEL_CLOSURE, pair.baseline_closure.sha256()),
            InputDigest(InputDigestCode.CANDIDATE_MODEL_CLOSURE, pair.candidate_closure.sha256()),
        ]
    )
    monitored = monitored_joints_from_config(config, pair.alignment.joints)
    workload = frozen_workload or load_workload_artifacts(
        _resolve(base, config.initial_state), _resolve(base, config.actions), pair
    )
    if (
        frozen_workload is not None
        and workload.model_pair_identity_sha256 != pair.model_pair_identity_sha256
    ):
        raise ValueError("frozen workload is bound to a different model-pair identity")
    input_digests.extend(
        [
            InputDigest(InputDigestCode.INITIAL_STATE_RAW, workload.state.raw_file_sha256),
            InputDigest(InputDigestCode.ACTIONS_RAW, workload.actions.raw_file_sha256),
        ]
    )
    grid = build_time_grid(
        baseline_step_dt=config.baseline.declared_step_dt,
        candidate_step_dt=config.candidate.declared_step_dt,
        control_dt=config.control_dt,
        baseline_compiled_timestep=float(live.baseline_model.opt.timestep),
        candidate_compiled_timestep=float(live.candidate_model.opt.timestep),
        control_intervals=workload.actions.metadata.control_intervals,
    )
    contract = _comparison_contract(config, pair, workload, monitored)
    input_digests.append(InputDigest(InputDigestCode.COMPARISON_CONTRACT, contract.sha256()))
    return pair, monitored, workload, grid, contract


def _runtime_evidence(
    live: LiveModelPair,
    config: ComparisonConfig,
    pair: ModelPairIdentity,
    monitored: tuple[MonitoredJoint, ...],
    workload: WorkloadArtifacts,
    grid: TimeGrid,
) -> tuple[EnvironmentIdentity, _ReplayEvidence]:
    """Evaluate budgets, threadpool state, and any replay evidence for admitted inputs."""
    budget_reasons = evaluate_preexecution_budgets(
        grid, config.repeats, pair.alignment.joints, pair.alignment.actuators
    )
    baseline_backend = MuJoCoBackend(live.baseline_model)
    candidate_backend = MuJoCoBackend(live.candidate_model)
    threadpool = combine_threadpool_states(
        observed_threadpool_state(baseline_backend.new_data()),
        observed_threadpool_state(candidate_backend.new_data()),
    )
    environment = build_environment_identity(threadpool)
    evidence = _replay_evidence(
        budget_reasons=budget_reasons,
        config=config,
        pair=pair,
        monitored=monitored,
        workload=workload,
        grid=grid,
        baseline_backend=baseline_backend,
        candidate_backend=candidate_backend,
    )
    if threadpool is EngineThreadpoolState.ACTIVE:
        evidence.reasons.append(_comparison_reason(ReasonCode.ENGINE_THREADPOOL_ACTIVE))
    elif threadpool is EngineThreadpoolState.UNKNOWN:
        evidence.reasons.append(_comparison_reason(ReasonCode.ENGINE_THREADPOOL_STATE_UNKNOWN))
    return environment, evidence


def _run_comparison(
    config_path: str | Path,
    candidate_step_dt: ExactRational | None,
) -> ComparisonRunResult:
    """Shared comparison implementation behind the public and private entry points."""
    distribution_sha = installed_distribution_sha256()
    tool_observation = _comparison_tool(distribution_sha)
    path = Path(config_path)
    config_raw, config = _load_config(path, tool_observation)
    input_digests: list[InputDigest] = [
        InputDigest(InputDigestCode.CONFIGURATION_RAW, hashlib.sha256(config_raw).hexdigest())
    ]
    base = path.absolute().parent
    output: OutputDirectory | None = None
    try:
        _validate_candidate_override(config, candidate_step_dt)
        baseline_root = _resolve(base, config.baseline.model_root)
        candidate_root = _resolve(base, config.candidate.model_root)
        output_path = _resolve(base, config.output_dir)
        _require_output_outside_model_roots(output_path, (baseline_root, candidate_root))
        output = prepare_output_directory(output_path)
        result = _execute_comparison(
            path,
            config_raw,
            config,
            base,
            output,
            candidate_step_dt,
            distribution_sha,
            tool_observation,
            input_digests,
        )
        return result
    except ComparisonOperationError:
        cleanup_output_after_failure(output)
        raise
    except (ModelAdmissionRefusal, ArtifactAdmissionRefusal) as exc:
        cleanup_output_after_failure(output)
        raise operational_error(
            tool=tool_observation,
            code=exc.reason,
            role=cast(ReasonRole, exc.role),
            evidence=cast("Mapping[str, CanonicalValue]", exc.evidence),
            available_inputs=input_digests,
        ) from exc
    except Exception as exc:
        cleanup_output_after_failure(output)
        raise operational_error(
            tool=tool_observation,
            code=OperationalReasonCode.INTERNAL_INVARIANT_FAILED,
            role=None,
            evidence={"exception_type": type(exc).__name__, "message": str(exc)},
            available_inputs=input_digests,
        ) from exc


def _comparison_tool(distribution_sha: str) -> OperationalToolObservation:
    """Build the verified installed-distribution observation for one comparison."""
    return OperationalToolObservation(
        __version__, "VERIFIED_INSTALLED_DISTRIBUTION", distribution_sha
    )


def _validate_candidate_override(
    config: ComparisonConfig, candidate_step_dt: ExactRational | None
) -> None:
    """Require the private timestep override to restate the declared candidate timestep."""
    if candidate_step_dt is not None and candidate_step_dt != config.candidate.declared_step_dt:
        raise ValueError(
            "candidate timestep override does not equal the declared candidate timestep"
        )


def _execute_comparison(
    path: Path,
    config_raw: bytes,
    config: ComparisonConfig,
    base: Path,
    output: OutputDirectory,
    candidate_step_dt: ExactRational | None,
    distribution_sha: str,
    tool: OperationalToolObservation,
    input_digests: list[InputDigest],
) -> ComparisonRunResult:
    """Execute and publish one comparison after outer failure accounting is installed."""
    aliases_bytes = _read_optional_aliases(base, config.aliases, tool, input_digests)
    with open_live_model_pair(
        baseline_root=_resolve(base, config.baseline.model_root),
        baseline_entrypoint=config.baseline.entrypoint,
        candidate_root=_resolve(base, config.candidate.model_root),
        candidate_entrypoint=config.candidate.entrypoint,
        aliases_json=aliases_bytes,
    ) as live:
        receipt = _compare_within_live_pair(
            live=live,
            config=config,
            config_raw=config_raw,
            base=base,
            candidate_step_dt=candidate_step_dt,
            distribution_sha=distribution_sha,
            input_digests=input_digests,
        )
        _reverify_live_sources(live)
        publish_results(
            output,
            json_bytes=canonical_json_bytes(receipt.to_primitive()),
            markdown_text=render_markdown(receipt, path.absolute()),
        )
        _reverify_live_sources(live)
    output._verify_retained_pair()
    result = ComparisonRunResult(receipt, output.json_path, output.markdown_path)
    output._close_after_success()
    return result


def _reverify_live_sources(live: LiveModelPair) -> None:
    """Reverify real retained snapshots while allowing lightweight isolated test doubles."""
    verifier = getattr(live, "verify_sources_unchanged", None)
    if verifier is not None:
        verifier()


def _canonical_output_path(path: Path) -> Path | None:
    """Resolve an existing destination or its existing parent without creating it."""
    absolute = path.absolute()
    try:
        absolute.lstat()
    except FileNotFoundError:
        try:
            return absolute.parent.resolve(strict=True) / absolute.name
        except OSError:
            return None
    except OSError:
        return None
    try:
        return absolute.resolve(strict=True)
    except OSError:
        return None


def _require_output_outside_model_roots(output: Path, roots: tuple[Path, Path]) -> None:
    """Refuse an output canonically equal to or below either declared model root."""
    canonical_output = _canonical_output_path(output)
    if canonical_output is None:
        return
    for root in roots:
        try:
            canonical_root = root.resolve(strict=True)
        except OSError:
            continue
        if canonical_output == canonical_root or canonical_root in canonical_output.parents:
            raise refuse(
                OperationalReasonCode.OUTPUT_PATH_INVALID,
                issue="output_inside_model_root",
            )


def _load_config(path: Path, tool: OperationalToolObservation) -> tuple[bytes, ComparisonConfig]:
    """Load config from path and tool for compare orchestrator, rejecting invalid input with operational_error."""
    try:
        raw = read_bounded_regular_file(path, CONFIG_JSON_LIMITS.max_bytes)
    except (OSError, JsonAdmissionError) as exc:
        raise operational_error(
            tool=tool,
            code=OperationalReasonCode.CONFIGURATION_IO_FAILED,
            role=None,
            field="comparison_config",
            evidence={"exception_type": type(exc).__name__},
        ) from exc
    try:
        primitive = bounded_strict_json_loads(raw, CONFIG_JSON_LIMITS)
        return raw, ComparisonConfig.from_primitive(primitive)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise operational_error(
            tool=tool,
            code=OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            role=None,
            field="comparison_config",
            evidence={"exception_type": type(exc).__name__, "message": str(exc)},
            available_inputs=(
                InputDigest(InputDigestCode.CONFIGURATION_RAW, hashlib.sha256(raw).hexdigest()),
            ),
        ) from exc


def _read_optional_aliases(
    base: Path,
    raw_path: str | None,
    tool: OperationalToolObservation,
    inputs: list[InputDigest],
) -> bytes | None:
    """Load optional aliases from base, raw path and tool for compare orchestrator, rejecting invalid input with operational_error."""
    if raw_path is None:
        return None
    try:
        payload = read_bounded_regular_file(_resolve(base, raw_path), CONFIG_JSON_LIMITS.max_bytes)
    except (OSError, JsonAdmissionError) as exc:
        raise operational_error(
            tool=tool,
            code=OperationalReasonCode.CONFIGURATION_IO_FAILED,
            role=None,
            field="aliases",
            evidence={"exception_type": type(exc).__name__},
            available_inputs=inputs,
        ) from exc
    inputs.append(InputDigest(InputDigestCode.ALIASES_RAW, hashlib.sha256(payload).hexdigest()))
    try:
        bounded_strict_json_loads(payload, CONFIG_JSON_LIMITS)
    except JsonAdmissionError as exc:
        raise operational_error(
            tool=tool,
            code=OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            role=None,
            field="aliases",
            evidence={"exception_type": type(exc).__name__, "message": str(exc)},
            available_inputs=inputs,
        ) from exc
    return payload


def _resolve(base: Path, value: str) -> Path:
    """Resolve one configuration path relative to its containing directory."""
    path = Path(value)
    return path if path.is_absolute() else base / path
