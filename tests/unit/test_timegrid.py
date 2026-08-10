"""Collect timegrid scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from metrifid import Binary64, ExactRational, canonical_sha256
from metrifid import _timegrid as timegrid
from metrifid._npz import ArtifactAdmissionRefusal
from metrifid.operational import OperationalReasonCode


def _decimal(token: str) -> ExactRational:
    """Construct the decimal fixture used by timegrid scenarios.

    Deterministic setup isolates timegrid without bypassing the contract boundary under
    assertion.
    """
    return ExactRational.from_decimal_token(token)


def _valid_grid() -> timegrid.TimeGrid:
    """Construct the valid grid fixture used by timegrid scenarios.

    Deterministic setup isolates timegrid without bypassing the contract boundary under
    assertion.
    """
    return timegrid.build_time_grid(
        baseline_step_dt=_decimal("0.002"),
        candidate_step_dt=_decimal("0.005"),
        control_dt=_decimal("0.01"),
        baseline_compiled_timestep=0.002,
        candidate_compiled_timestep=0.005,
        control_intervals=3,
    )


def _refusal_reason(exc: pytest.ExceptionInfo[ArtifactAdmissionRefusal]) -> OperationalReasonCode:
    """Extract the artifact refusal code from a time-grid validation failure."""
    return exc.value.reason


def test_different_role_steps_share_exact_control_boundaries() -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises different role steps share exact control boundaries; exact-grid
    evidence and completed-prefix rules must never promote an unqualified candidate.
    """
    grid = _valid_grid()
    assert grid.baseline_substeps_per_control == 5
    assert grid.candidate_substeps_per_control == 2
    assert grid.control_intervals == 3
    assert grid.boundary_count == 4
    assert grid.horizon == ExactRational(3, 100)
    assert list(grid.iter_canonical_boundaries()) == [
        ExactRational(0, 1),
        ExactRational(1, 100),
        ExactRational(1, 50),
        ExactRational(3, 100),
    ]
    assert grid.sha256() == canonical_sha256(grid.to_primitive())
    assert grid.sha256() == grid.sha256()
    assert len(grid.sha256()) == 64


def test_role_local_boundary_bits_use_repeated_addition_not_direct_multiplication() -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises role local boundary bits use repeated addition not direct
    multiplication; exact-grid evidence and completed-prefix rules must never promote an
    unqualified candidate.
    """
    grid = timegrid.build_time_grid(
        baseline_step_dt=_decimal("0.01"),
        candidate_step_dt=_decimal("0.1"),
        control_dt=_decimal("0.1"),
        baseline_compiled_timestep=0.01,
        candidate_compiled_timestep=0.1,
        control_intervals=2,
    )
    baseline = list(grid.iter_role_boundary_time_bits("baseline"))
    candidate = list(grid.iter_role_boundary_time_bits("candidate"))
    assert baseline[0].bits == Binary64.from_float(0.0).bits
    repeated = 0.0
    for _ in range(10):
        repeated += 0.01
    assert baseline[1].bits == Binary64.from_float(repeated).bits
    assert baseline[1].bits != Binary64.from_float(0.1).bits
    assert candidate[1].bits == Binary64.from_float(0.1).bits
    with pytest.raises(ValueError):
        list(grid.iter_role_boundary_time_bits(cast(timegrid.TimeRole, "other")))


@pytest.mark.parametrize(
    ("field", "value", "role"),
    [
        ("baseline_step_dt", ExactRational(0, 1), "baseline"),
        ("baseline_step_dt", ExactRational(-1, 100), "baseline"),
        ("candidate_step_dt", ExactRational(0, 1), "candidate"),
        ("candidate_step_dt", ExactRational(-1, 100), "candidate"),
    ],
)
def test_zero_and_negative_declared_periods_refuse(
    field: str, value: ExactRational, role: str
) -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises zero and negative declared periods refuse; exact-grid evidence and
    completed-prefix rules must never promote an unqualified candidate.
    """
    kwargs = {
        "baseline_step_dt": _decimal("0.002"),
        "candidate_step_dt": _decimal("0.005"),
        "control_dt": _decimal("0.01"),
        "baseline_compiled_timestep": 0.002,
        "candidate_compiled_timestep": 0.005,
        "control_intervals": 1,
    }
    kwargs[field] = value
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        timegrid.build_time_grid(**kwargs)  # type: ignore[arg-type]
    assert _refusal_reason(exc) is OperationalReasonCode.DECLARED_STEP_DT_INVALID
    assert exc.value.role == role


@pytest.mark.parametrize("value", [ExactRational(0, 1), ExactRational(-1, 10)])
def test_zero_and_negative_control_periods_refuse(value: ExactRational) -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises zero and negative control periods refuse; exact-grid evidence and
    completed-prefix rules must never promote an unqualified candidate.
    """
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        timegrid.build_time_grid(
            baseline_step_dt=_decimal("0.002"),
            candidate_step_dt=_decimal("0.005"),
            control_dt=value,
            baseline_compiled_timestep=0.002,
            candidate_compiled_timestep=0.005,
            control_intervals=1,
        )
    assert _refusal_reason(exc) is OperationalReasonCode.CONTROL_DT_INVALID


@pytest.mark.parametrize("compiled", [0.003, 0.0, -0.002, float("inf"), float("nan")])
def test_compiled_timestep_bit_or_validity_mismatch_refuses(compiled: float) -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises compiled timestep bit or validity mismatch refuses; exact-grid
    evidence and completed-prefix rules must never promote an unqualified candidate.
    """
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        timegrid.build_time_grid(
            baseline_step_dt=_decimal("0.002"),
            candidate_step_dt=_decimal("0.005"),
            control_dt=_decimal("0.01"),
            baseline_compiled_timestep=compiled,
            candidate_compiled_timestep=0.005,
            control_intervals=1,
        )
    assert _refusal_reason(exc) is OperationalReasonCode.DECLARED_STEP_DT_MISMATCH
    assert exc.value.role == "baseline"


def test_non_float_compiled_timestep_refuses() -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises non float compiled timestep refuses; exact-grid evidence and
    completed-prefix rules must never promote an unqualified candidate.
    """
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        timegrid.build_time_grid(
            baseline_step_dt=_decimal("0.002"),
            candidate_step_dt=_decimal("0.005"),
            control_dt=_decimal("0.01"),
            baseline_compiled_timestep=cast(float, 2),
            candidate_compiled_timestep=0.005,
            control_intervals=1,
        )
    assert _refusal_reason(exc) is OperationalReasonCode.DECLARED_STEP_DT_MISMATCH


def test_declared_period_that_underflows_or_overflows_binary64_refuses() -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises declared period that underflows or overflows binary64 refuses;
    exact-grid evidence and completed-prefix rules must never promote an unqualified candidate.
    """
    for period in (ExactRational(1, 10**400), ExactRational(10**400, 1)):
        with pytest.raises(ArtifactAdmissionRefusal) as exc:
            timegrid.build_time_grid(
                baseline_step_dt=period,
                candidate_step_dt=_decimal("0.005"),
                control_dt=ExactRational(10**400, 1),
                baseline_compiled_timestep=0.0,
                candidate_compiled_timestep=0.005,
                control_intervals=1,
            )
        assert _refusal_reason(exc) is OperationalReasonCode.DECLARED_STEP_DT_INVALID


def test_exact_nonintegral_and_near_integral_float_trap_refuse() -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises exact nonintegral and near integral float trap refuse; exact-grid
    evidence and completed-prefix rules must never promote an unqualified candidate.
    """
    for step_token in ("0.003", "0.0033333333333333335"):
        step = _decimal(step_token)
        with pytest.raises(ArtifactAdmissionRefusal) as exc:
            timegrid.build_time_grid(
                baseline_step_dt=step,
                candidate_step_dt=_decimal("0.005"),
                control_dt=_decimal("0.01"),
                baseline_compiled_timestep=float(step_token),
                candidate_compiled_timestep=0.005,
                control_intervals=1,
            )
        assert _refusal_reason(exc) is OperationalReasonCode.CONTROL_GRID_NONINTEGRAL
        assert exc.value.role == "baseline"
    assert 0.01 / float("0.0033333333333333335") == 3.0


@pytest.mark.parametrize("count", [0, 100_001, True])
def test_invalid_control_interval_count_refuses(count: object) -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises invalid control interval count refuses; exact-grid evidence and
    completed-prefix rules must never promote an unqualified candidate.
    """
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        timegrid.build_time_grid(
            baseline_step_dt=_decimal("0.002"),
            candidate_step_dt=_decimal("0.005"),
            control_dt=_decimal("0.01"),
            baseline_compiled_timestep=0.002,
            candidate_compiled_timestep=0.005,
            control_intervals=cast(int, count),
        )
    assert _refusal_reason(exc) is OperationalReasonCode.CONTROL_INTERVAL_COUNT_INVALID


def _token(value: int, scale: int) -> str:
    """Construct the token fixture used by timegrid scenarios.

    Deterministic setup isolates timegrid without bypassing the contract boundary under
    assertion.
    """
    if scale == 0:
        return str(value)
    digits = str(value).rjust(scale + 1, "0")
    return f"{digits[:-scale]}.{digits[-scale:]}"


@settings(max_examples=120, deadline=None)
@given(
    control_units=st.integers(min_value=1, max_value=10_000),
    step_units=st.integers(min_value=1, max_value=10_000),
    scale=st.integers(min_value=0, max_value=6),
)
def test_exact_divisibility_agrees_with_fraction(
    control_units: int, step_units: int, scale: int
) -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises exact divisibility agrees with fraction; exact-grid evidence and
    completed-prefix rules must never promote an unqualified candidate.
    """
    control_token = _token(control_units, scale)
    step_token = _token(step_units, scale)
    control = _decimal(control_token)
    step = _decimal(step_token)
    expected = Fraction(control_units, step_units)
    if expected.denominator == 1:
        grid = timegrid.build_time_grid(
            baseline_step_dt=step,
            candidate_step_dt=step,
            control_dt=control,
            baseline_compiled_timestep=float(step_token),
            candidate_compiled_timestep=float(step_token),
            control_intervals=1,
        )
        assert grid.baseline_substeps_per_control == expected.numerator
    else:
        with pytest.raises(ArtifactAdmissionRefusal) as exc:
            timegrid.build_time_grid(
                baseline_step_dt=step,
                candidate_step_dt=step,
                control_dt=control,
                baseline_compiled_timestep=float(step_token),
                candidate_compiled_timestep=float(step_token),
                control_intervals=1,
            )
        assert _refusal_reason(exc) is OperationalReasonCode.CONTROL_GRID_NONINTEGRAL


def test_time_grid_direct_constructor_invariants() -> None:
    """Keep timestep recommendations conservative and reproducible.

    This scenario exercises time grid direct constructor invariants; exact-grid evidence and
    completed-prefix rules must never promote an unqualified candidate.
    """
    grid = _valid_grid()
    invalid = [
        {"baseline_step_dt": ExactRational(0, 1)},
        {"candidate_step_dt": ExactRational(0, 1)},
        {"control_dt": ExactRational(0, 1)},
        {"baseline_compiled_timestep": cast(Binary64, object())},
        {"candidate_compiled_timestep": cast(Binary64, object())},
        {"baseline_compiled_timestep": Binary64.from_float(0.003)},
        {"candidate_compiled_timestep": Binary64.from_float(float("nan"))},
        {"baseline_step_dt": ExactRational(10**400, 1)},
        {"baseline_substeps_per_control": 0},
        {"candidate_substeps_per_control": True},
        {"control_intervals": 0},
        {"boundary_count": 99},
        {"horizon": ExactRational(1, 1)},
        {"baseline_substeps_per_control": 6},
        {"candidate_substeps_per_control": 3},
    ]
    for change in invalid:
        with pytest.raises((TypeError, ValueError)):
            replace(grid, **change)
