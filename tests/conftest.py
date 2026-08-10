"""Shared strict test fixtures."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import cast

import pytest

from metrifid import (
    ComparisonStatus,
    EngineThreadpoolState,
    ExactRational,
    LimitationCode,
    ReasonCode,
    ReasonRecord,
)
from metrifid.errors import ReasonRole
from metrifid.json_values import CanonicalValue, FrozenCanonicalObject, freeze_canonical
from metrifid.schemas import (
    AlignmentSummary,
    CanonicalSummary,
    ComparisonContractIdentity,
    ComparisonInputsIdentity,
    ComparisonReceipt,
    EnvironmentIdentity,
    MetricEvidenceSummary,
    ModelClosureIdentity,
    ModelClosureMember,
    ModelClosures,
    MonitoredJoint,
    NumericalEvidenceSummary,
    RepeatabilitySummary,
    TimeContract,
    ToolIdentity,
)

_DEFAULT_ROLE = object()

_DEFAULT_REASON_ROLES: dict[ReasonCode, ReasonRole] = {
    ReasonCode.ENGINE_THREADPOOL_ACTIVE: "comparison",
    ReasonCode.ENGINE_THREADPOOL_STATE_UNKNOWN: "comparison",
    ReasonCode.INITIAL_STATE_NOT_PRESERVED: "baseline",
    ReasonCode.INTERNAL_STEP_BUDGET_EXCEEDED: "comparison",
    ReasonCode.TRACE_MEMORY_BUDGET_EXCEEDED: "comparison",
    ReasonCode.COMPARISON_TIMEOUT: "comparison",
    ReasonCode.BASELINE_NONDETERMINISTIC: "baseline",
    ReasonCode.CANDIDATE_NONDETERMINISTIC: "candidate",
    ReasonCode.BASELINE_NONFINITE_STATE: "baseline",
    ReasonCode.BASELINE_INVALID_QUATERNION: "baseline",
    ReasonCode.BASELINE_MUJOCO_WARNING: "baseline",
    ReasonCode.BASELINE_NUMERICAL_ERROR_LOG: "baseline",
    ReasonCode.BASELINE_EARLY_TERMINATION: "baseline",
    ReasonCode.CANDIDATE_NONFINITE_STATE: "candidate",
    ReasonCode.CANDIDATE_INVALID_QUATERNION: "candidate",
    ReasonCode.CANDIDATE_MUJOCO_WARNING: "candidate",
    ReasonCode.CANDIDATE_NUMERICAL_ERROR_LOG: "candidate",
    ReasonCode.CANDIDATE_EARLY_TERMINATION: "candidate",
    ReasonCode.TRACE_SAMPLE_COUNT_MISMATCH: "comparison",
    ReasonCode.TRACE_BOUNDARY_INDEX_MISMATCH: "comparison",
    ReasonCode.TRACE_TIME_RECURRENCE_MISMATCH: "comparison",
    ReasonCode.TRACE_CHANNEL_LAYOUT_MISMATCH: "comparison",
    ReasonCode.TRACE_MALFORMED: "comparison",
    ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED: "comparison",
}


def digest(label: str) -> str:
    """Return a deterministic lowercase SHA-256 test value."""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def frozen_object(value: dict[str, CanonicalValue]) -> FrozenCanonicalObject:
    """Freeze one test object with the production canonical adapter."""
    frozen = freeze_canonical(value)
    if not isinstance(frozen, dict) and not hasattr(frozen, "items"):
        raise AssertionError("expected frozen mapping")
    return frozen  # type: ignore[return-value]


def build_test_model_closure(label: str) -> ModelClosureIdentity:
    """Build a one-member closure identity with deterministic test hashes."""
    member = ModelClosureMember("robot.xml", len(label), digest(f"member:{label}"))
    return ModelClosureIdentity("robot.xml", 1, (member,))


def build_test_reason_record(
    code: ReasonCode, *, role: ReasonRole | object = _DEFAULT_ROLE
) -> ReasonRecord:
    """Build a reason record with the code's valid role and evidence shape."""
    selected_role: ReasonRole
    if role is _DEFAULT_ROLE:
        selected_role = _DEFAULT_REASON_ROLES[code]
    else:
        selected_role = cast(ReasonRole, role)
    object_type = "joint" if code is ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED else None
    object_name = "elbow" if code is ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED else None
    metric = "angle_rad" if code is ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED else None
    boundary_index = 0 if code is ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED else None
    return ReasonRecord(
        code=code,
        role=selected_role,
        object_type=object_type,
        object_name=object_name,
        metric=metric,
        boundary_index=boundary_index,
        evidence=frozen_object({}),
    )


def make_receipt_candidate(
    status: ComparisonStatus,
    reasons: Sequence[ReasonRecord] = (),
    *,
    threadpool_state: EngineThreadpoolState = EngineThreadpoolState.DISABLED,
    tool_version: str = "0.1.0a3",
) -> ComparisonReceipt:
    """Construct the make receipt candidate fixture used by conftest scenarios.

    Deterministic setup isolates conftest without bypassing the contract boundary under
    assertion.
    """
    baseline = build_test_model_closure("baseline")
    candidate = build_test_model_closure("candidate")
    monitored, tolerance_object = _receipt_monitoring()
    time = _receipt_time_contract()
    comparison_contract = _receipt_comparison_contract(baseline, candidate, monitored, time)
    return ComparisonReceipt(
        schema="metrifid.comparison_receipt",
        schema_version=1,
        status_rule_schema="metrifid.status_precedence",
        status_rule_schema_version=1,
        tool=ToolIdentity(tool_version, digest("tool")),
        status=status,
        reason_codes=(),
        reasons=tuple(reasons),
        environment=_receipt_environment(threadpool_state),
        inputs=_receipt_inputs(baseline, candidate, comparison_contract),
        comparison_contract=comparison_contract,
        model_closures=ModelClosures(baseline, candidate),
        time=time,
        alignment=AlignmentSummary(None, ("elbow",), ("elbow_motor",), ()),
        monitored_joints=(monitored,),
        tolerances=tolerance_object,
        repeatability=RepeatabilitySummary.from_primitive({}),
        numerical_evidence=NumericalEvidenceSummary.from_primitive({}),
        metrics=MetricEvidenceSummary.from_primitive({}),
        first_crossing=None,
        limitations=tuple(LimitationCode),
        receipt_sha256=None,
    )


def _receipt_monitoring() -> tuple[MonitoredJoint, CanonicalSummary]:
    """Build the monitored joint and exactly matching tolerance summary fixture."""
    monitored = MonitoredJoint(
        "elbow",
        "hinge",
        {
            "angle_rad": ExactRational(1, 1000),
            "angular_velocity_rad_s": ExactRational(1, 100),
        },
    )
    tolerance_object = CanonicalSummary.from_primitive(
        {
            "elbow": {
                "joint_type": "hinge",
                "angle_rad": {"numerator": 1, "denominator": 1000},
                "angular_velocity_rad_s": {"numerator": 1, "denominator": 100},
            }
        }
    )
    return monitored, tolerance_object


def _receipt_time_contract() -> TimeContract:
    """Build the shared exact three-boundary time contract fixture."""
    return TimeContract(
        baseline_step_dt=ExactRational(1, 500),
        candidate_step_dt=ExactRational(1, 500),
        control_dt=ExactRational(1, 100),
        control_intervals=2,
        state_samples=3,
        horizon=ExactRational(1, 50),
        sample_phase="BOUNDARY_BEFORE_CONTROL",
        action_semantics="LEFT_BOUNDARY_ZERO_ORDER_HOLD",
        terminal_sample="INCLUDED",
        interpolation="FORBIDDEN",
    )


def _receipt_comparison_contract(
    baseline: ModelClosureIdentity,
    candidate: ModelClosureIdentity,
    monitored: MonitoredJoint,
    time: TimeContract,
) -> ComparisonContractIdentity:
    """Bind fixture closures, workload digests, time, and monitoring into a contract."""
    return ComparisonContractIdentity(
        schema="metrifid.comparison_contract",
        schema_version=1,
        baseline_model_closure_sha256=baseline.sha256(),
        candidate_model_closure_sha256=candidate.sha256(),
        initial_state_semantic_sha256=digest("state-semantic"),
        actions_semantic_sha256=digest("actions-semantic"),
        aliases_semantic_sha256=None,
        baseline_step_dt=time.baseline_step_dt,
        candidate_step_dt=time.candidate_step_dt,
        control_dt=time.control_dt,
        repeats=3,
        monitored_joints=(monitored,),
    )


def _receipt_environment(state: EngineThreadpoolState) -> EnvironmentIdentity:
    """Build the finalized-environment candidate used by receipt fixtures."""
    return EnvironmentIdentity(
        mujoco_version="3.10.0",
        python_version="3.13.5",
        numpy_version="2.3.0",
        mujoco_python_distribution_sha256=digest("mujoco-python"),
        mujoco_native_library_sha256=digest("mujoco-native"),
        platform="linux-x86_64",
        platform_release="test-kernel",
        libc="glibc-test",
        cpu_identity_sha256=digest("cpu"),
        engine_threadpool_state=state,
        environment_sha256=None,
    )


def _receipt_inputs(
    baseline: ModelClosureIdentity,
    candidate: ModelClosureIdentity,
    contract: ComparisonContractIdentity,
) -> ComparisonInputsIdentity:
    """Build raw and semantic input bindings for a receipt candidate fixture."""
    return ComparisonInputsIdentity(
        configuration_raw_sha256=digest("config-raw"),
        comparison_contract_sha256=contract.sha256(),
        baseline_model_closure_sha256=baseline.sha256(),
        candidate_model_closure_sha256=candidate.sha256(),
        initial_state_raw_sha256=digest("state-raw"),
        initial_state_semantic_sha256=digest("state-semantic"),
        actions_raw_sha256=digest("actions-raw"),
        actions_semantic_sha256=digest("actions-semantic"),
        aliases_raw_sha256=None,
        aliases_semantic_sha256=None,
    )


@pytest.fixture
def green_candidate() -> ComparisonReceipt:
    """Publish a valid green candidate fixture through the real product path.

    Consumers begin from self-consistent on-disk evidence before exercising conftest mutations.
    """
    return make_receipt_candidate(ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD)
