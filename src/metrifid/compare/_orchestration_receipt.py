"""Installed comparison comparison orchestration over accepted model admission/artifact admission contracts."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, cast

from ..errors import (
    LimitationCode,
    ReasonCode,
    ReasonRecord,
    derive_comparison_status,
)
from ..json_values import (
    CanonicalValue,
    FrozenCanonicalObject,
    freeze_canonical,
)
from ..schemas import (
    CanonicalSummary,
    ComparisonConfig,
    ComparisonContractIdentity,
    ComparisonInputsIdentity,
    ComparisonReceipt,
    MetricEvidenceSummary,
    ModelClosures,
    MonitoredJoint,
    NumericalEvidenceSummary,
    RepeatabilitySummary,
    TimeContract,
    ToolIdentity,
    finalize_receipt,
)
from ..version import __version__

if TYPE_CHECKING:
    # annotations only; avoids importing the identity module at runtime

    from .._model_identity import ModelPairIdentity
    from .._timegrid import TimeGrid
    from .._workload import WorkloadArtifacts
    from ..schemas import EnvironmentIdentity
    from ._orchestrator import _ReplayEvidence


def _comparison_contract(
    config: ComparisonConfig,
    pair: ModelPairIdentity,
    workload: WorkloadArtifacts,
    monitored: tuple[MonitoredJoint, ...],
) -> ComparisonContractIdentity:
    """Build the frozen contract identity that names exactly what is being compared."""
    return ComparisonContractIdentity(
        schema="metrifid.comparison_contract",
        schema_version=1,
        baseline_model_closure_sha256=pair.baseline_closure.sha256(),
        candidate_model_closure_sha256=pair.candidate_closure.sha256(),
        initial_state_semantic_sha256=workload.state.semantic_sha256,
        actions_semantic_sha256=workload.actions.semantic_sha256,
        aliases_semantic_sha256=pair.alignment.aliases_semantic_sha256,
        baseline_step_dt=config.baseline.declared_step_dt,
        candidate_step_dt=config.candidate.declared_step_dt,
        control_dt=config.control_dt,
        repeats=config.repeats,
        monitored_joints=monitored,
    )


def _time_contract(grid: TimeGrid) -> TimeContract:
    """Convert the admitted replay grid into the frozen receipt time contract."""
    return TimeContract(
        baseline_step_dt=grid.baseline_step_dt,
        candidate_step_dt=grid.candidate_step_dt,
        control_dt=grid.control_dt,
        control_intervals=grid.control_intervals,
        state_samples=grid.boundary_count,
        horizon=grid.horizon,
        sample_phase="BOUNDARY_BEFORE_CONTROL",
        action_semantics="LEFT_BOUNDARY_ZERO_ORDER_HOLD",
        terminal_sample="INCLUDED",
        interpolation="FORBIDDEN",
    )


def _comparison_receipt(
    *,
    distribution_sha: str,
    evidence: _ReplayEvidence,
    environment: EnvironmentIdentity,
    config_raw: bytes,
    contract: ComparisonContractIdentity,
    pair: ModelPairIdentity,
    workload: WorkloadArtifacts,
    monitored: tuple[MonitoredJoint, ...],
    grid: TimeGrid,
) -> ComparisonReceipt:
    """Assemble and finalize the complete comparison receipt."""
    return finalize_receipt(
        ComparisonReceipt(
            schema="metrifid.comparison_receipt",
            schema_version=1,
            status_rule_schema="metrifid.status_precedence",
            status_rule_schema_version=1,
            tool=ToolIdentity(__version__, distribution_sha),
            status=derive_comparison_status(evidence.reasons),
            reason_codes=(),
            reasons=tuple(evidence.reasons),
            environment=environment,
            inputs=_comparison_inputs(config_raw, contract, pair, workload),
            comparison_contract=contract,
            model_closures=ModelClosures(pair.baseline_closure, pair.candidate_closure),
            time=_time_contract(grid),
            alignment=pair.alignment_summary,
            monitored_joints=monitored,
            tolerances=CanonicalSummary.from_primitive(_tolerance_summary(monitored)),
            repeatability=RepeatabilitySummary.from_primitive(evidence.repeatability),
            numerical_evidence=NumericalEvidenceSummary.from_primitive(evidence.numerical_evidence),
            metrics=MetricEvidenceSummary.from_primitive(evidence.metrics),
            first_crossing=(
                None
                if evidence.first_crossing is None
                else CanonicalSummary.from_primitive(evidence.first_crossing)
            ),
            limitations=tuple(LimitationCode),
            receipt_sha256=None,
        )
    )


def _comparison_inputs(
    config_raw: bytes,
    contract: ComparisonContractIdentity,
    pair: ModelPairIdentity,
    workload: WorkloadArtifacts,
) -> ComparisonInputsIdentity:
    """Bind raw and semantic input identities into one comparison receipt member."""
    return ComparisonInputsIdentity(
        configuration_raw_sha256=hashlib.sha256(config_raw).hexdigest(),
        comparison_contract_sha256=contract.sha256(),
        baseline_model_closure_sha256=pair.baseline_closure.sha256(),
        candidate_model_closure_sha256=pair.candidate_closure.sha256(),
        initial_state_raw_sha256=workload.state.raw_file_sha256,
        initial_state_semantic_sha256=workload.state.semantic_sha256,
        actions_raw_sha256=workload.actions.raw_file_sha256,
        actions_semantic_sha256=workload.actions.semantic_sha256,
        aliases_raw_sha256=pair.alignment.aliases_raw_sha256,
        aliases_semantic_sha256=pair.alignment.aliases_semantic_sha256,
    )


def _preexecution_repeatability(repeats: int) -> dict[str, CanonicalValue]:
    """Build explicit not-executed repeatability evidence for preexecution refusal."""
    role_summary: dict[str, CanonicalValue] = {
        "captured_boundary_counts": [],
        "complete_repeats": 0,
        "evaluated": False,
        "reason": "preexecution_budget_exceeded",
        "repeat_count": repeats,
        "signatures": [],
        "stable": None,
    }
    return {
        "schema": "metrifid.repeatability_evidence",
        "schema_version": 1,
        "baseline": dict(role_summary),
        "candidate": dict(role_summary),
    }


def _preexecution_numerical_evidence(boundary_count: int) -> dict[str, CanonicalValue]:
    """Build empty numerical evidence for a comparison refused before stepping."""
    role_summary: dict[str, CanonicalValue] = {
        "captured_boundary_count": 0,
        "complete": False,
        "error_logs": [],
        "evaluated": False,
        "expected_boundary_count": boundary_count,
        "first_warning": None,
        "initial_state_preserved": None,
        "invalid_boundary_index": None,
        "invalid_kind": None,
        "reason": "preexecution_budget_exceeded",
    }
    return {
        "schema": "metrifid.numerical_evidence",
        "schema_version": 1,
        "baseline": dict(role_summary),
        "candidate": dict(role_summary),
    }


def _preexecution_metrics() -> dict[str, CanonicalValue]:
    """Build empty metric evidence for a comparison refused before stepping."""
    return {
        "schema": "metrifid.metric_evidence",
        "schema_version": 1,
        "evaluated": False,
        "joints": [],
        "reason": "preexecution_budget_exceeded",
    }


def _tolerance_summary(monitored: tuple[MonitoredJoint, ...]) -> dict[str, CanonicalValue]:
    """Emit exact joint-type-specific tolerances in canonical monitored order."""
    result: dict[str, CanonicalValue] = {}
    for joint in monitored:
        result[joint.canonical_name] = {
            "joint_type": joint.joint_type,
            **{metric: joint.tolerances[metric].to_primitive() for metric in joint.tolerances},
        }
    return result


def _comparison_reason(code: ReasonCode) -> ReasonRecord:
    """Build a comparison-scoped orchestration reason with empty evidence."""
    frozen = freeze_canonical({})
    return ReasonRecord(
        code=code,
        role="comparison",
        object_type=None,
        object_name=None,
        metric=None,
        boundary_index=None,
        evidence=cast(FrozenCanonicalObject, frozen),
    )
