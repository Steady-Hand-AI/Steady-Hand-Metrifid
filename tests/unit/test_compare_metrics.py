"""Constructed hinge/slide formula and strict-crossing tests for comparison."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from metrifid import ComparisonConfig, ExactRational
from metrifid._npz import ArtifactAdmissionRefusal
from metrifid.compare._metrics import evaluate_role_pair
from metrifid.compare._monitoring import monitored_joints_from_config
from metrifid.compare._trace import RoleRepeatSet
from metrifid.errors import ReasonCode, derive_comparison_status
from metrifid.operational import OperationalReasonCode
from metrifid.schemas import MonitoredJoint
from tests._support.constructed_models import aligned_joints, role_trace, time_grid


def _repeat(trace):
    """Construct the repeat fixture used by compare metrics scenarios.

    Deterministic setup isolates compare metrics without bypassing the contract boundary under
    assertion.
    """
    return RoleRepeatSet(trace.role, (trace, trace))


def test_hinge_wrapping_slide_difference_and_first_crossing() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises hinge wrapping slide difference and first crossing; status,
    numerical evidence, and artifact publication must remain stable for the declared workload.
    """
    baseline = role_trace(
        "baseline",
        qpos_rows=[[math.pi - 0.01, 0.0], [math.pi - 0.01, 0.0], [math.pi - 0.01, 0.0]],
        qvel_rows=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    )
    candidate = role_trace(
        "candidate",
        qpos_rows=[[-math.pi + 0.01, 0.0], [-math.pi + 0.03, 0.001], [-math.pi + 0.03, 0.002]],
        qvel_rows=[[0.0, 0.0], [0.02, 0.0], [0.02, 0.04]],
    )
    monitored = (
        MonitoredJoint(
            "hinge",
            "hinge",
            {
                "angle_rad": ExactRational(3, 100),
                "angular_velocity_rad_s": ExactRational(1, 100),
            },
        ),
        MonitoredJoint(
            "slide",
            "slide",
            {
                "translation_m": ExactRational(1, 1000),
                "linear_velocity_m_s": ExactRational(1, 100),
            },
        ),
    )
    result = evaluate_role_pair(
        baseline=_repeat(baseline),
        candidate=_repeat(candidate),
        joints=aligned_joints(),
        monitored_joints=monitored,
        time_grid=time_grid(),
    )
    assert result.first_crossing is not None
    assert result.first_crossing["boundary_index"] == 1
    assert result.first_crossing["joint_name"] == "hinge"
    assert result.first_crossing["metric"] == "angle_rad"
    assert ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED in {reason.code for reason in result.reasons}
    assert derive_comparison_status(result.reasons).value == "MATERIAL_BEHAVIOR_CHANGE"


def test_error_equal_to_exact_tolerance_does_not_cross() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises error equal to exact tolerance does not cross; status, numerical
    evidence, and artifact publication must remain stable for the declared workload.
    """
    baseline = role_trace(
        "baseline",
        qpos_rows=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        qvel_rows=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    )
    candidate = role_trace(
        "candidate",
        qpos_rows=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        qvel_rows=[[0.125, 0.0], [0.125, 0.0], [0.125, 0.0]],
    )
    monitored = (
        MonitoredJoint(
            "hinge",
            "hinge",
            {
                "angle_rad": ExactRational(1, 1000),
                "angular_velocity_rad_s": ExactRational(1, 8),
            },
        ),
    )
    result = evaluate_role_pair(
        baseline=_repeat(baseline),
        candidate=_repeat(candidate),
        joints=aligned_joints(),
        monitored_joints=monitored,
        time_grid=time_grid(),
    )
    assert result.first_crossing is None
    assert ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED not in {
        reason.code for reason in result.reasons
    }


def test_incomplete_candidate_trace_is_not_metric_evaluated() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises incomplete candidate trace is not metric evaluated; status,
    numerical evidence, and artifact publication must remain stable for the declared workload.
    """
    baseline = role_trace(
        "baseline",
        qpos_rows=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        qvel_rows=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    )
    candidate = role_trace(
        "candidate",
        qpos_rows=[[0.0, 0.0], [0.0, 0.0]],
        qvel_rows=[[0.0, 0.0], [0.0, 0.0]],
        expected_boundary_count=3,
        invalid_kind="NONFINITE_STATE",
        invalid_boundary_index=2,
    )
    monitored = (
        MonitoredJoint(
            "hinge",
            "hinge",
            {
                "angle_rad": ExactRational(1, 1000),
                "angular_velocity_rad_s": ExactRational(1, 8),
            },
        ),
    )
    result = evaluate_role_pair(
        baseline=_repeat(baseline),
        candidate=_repeat(candidate),
        joints=aligned_joints(),
        monitored_joints=monitored,
        time_grid=time_grid(),
    )
    assert result.metrics["evaluated"] is False
    codes = {reason.code for reason in result.reasons}
    assert ReasonCode.CANDIDATE_NONFINITE_STATE in codes
    assert ReasonCode.CANDIDATE_EARLY_TERMINATION in codes
    assert ReasonCode.TRACE_SAMPLE_COUNT_MISMATCH in codes


def test_ball_joint_is_refused_before_rollout_with_existing_reason() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises ball joint is refused before rollout with existing reason; status,
    numerical evidence, and artifact publication must remain stable for the declared workload.
    """
    config = ComparisonConfig.from_primitive(
        {
            "schema_version": 1,
            "baseline": {
                "model_root": "baseline",
                "entrypoint": "model.xml",
                "declared_step_dt": "0.005",
            },
            "candidate": {
                "model_root": "candidate",
                "entrypoint": "model.xml",
                "declared_step_dt": "0.005",
            },
            "initial_state": "state.npz",
            "actions": "actions.npz",
            "control_dt": "0.01",
            "repeats": 2,
            "joint_tolerances": {
                "ball": {
                    "joint_type": "ball",
                    "orientation_rad": "0.001",
                    "angular_velocity_rad_s": "0.01",
                }
            },
            "aliases": None,
            "output_dir": "result",
        }
    )
    with pytest.raises(ArtifactAdmissionRefusal) as error:
        monitored_joints_from_config(
            config,
            (SimpleNamespace(canonical_name="ball", joint_type="BALL"),),
        )
    assert error.value.reason is OperationalReasonCode.TOLERANCE_UNIT_MISMATCH
    assert error.value.evidence["object_name"] == "ball"
    assert error.value.evidence["issue"] == "monitoring_supports_hinge_and_slide_only"
