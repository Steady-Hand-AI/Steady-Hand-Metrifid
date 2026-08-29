"""Direct black-box tests for the private native-upgrade decision core.

Every expectation below is derived from the module's declared contract or from arithmetic simple
enough to state in the test itself: the contraction gate widens the finest value by
``epsilon + 3 * d1`` because ``rho / (1 - rho)`` is ``3`` at the frozen ``rho = 0.75``; two constant
profiles therefore differ by their nominal separation plus a few multiples of ``epsilon``. No
expectation is produced by calling the implementation under test, and nothing here reads a file, an
environment variable, or another process.

Frozen contract values referenced by name below, so a change to any of them fails a test rather than
silently rewriting what the suite means:

``epsilon = 1e-12`` numerical floor · ``rho = 0.75`` contraction ratio ·
``minimum_prefix = 0.50`` · ``horizon = 1.00``
"""

from __future__ import annotations

import math

import pytest

import metrifid._native_upgrade as product

_EPSILON = 1e-12
_MINIMUM_PREFIX = 0.50
_HORIZON = 1.00
_TOLERANCE = 1e-3


def _constant_scalar(
    channel_id: str,
    time: float,
    value_a: float,
    value_b: float,
    tolerance: float = _TOLERANCE,
) -> product.ScalarObservation:
    """Return one scalar witness whose two profiles are each constant across the three grids."""
    return product.ScalarObservation(
        channel_id=channel_id,
        time=time,
        scale=1.0,
        tolerance=tolerance,
        profile_a=(value_a, value_a, value_a),
        profile_b=(value_b, value_b, value_b),
    )


def _rotation_about_x(angle: float) -> product.Quaternion:
    """Return the unit quaternion of a rotation through ``angle`` about the x axis, in wxyz order."""
    return (math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0)


def _constant_orientation(
    channel_id: str,
    time: float,
    quaternion_a: product.Quaternion,
    quaternion_b: product.Quaternion,
    tolerance: float = _TOLERANCE,
) -> product.OrientationObservation:
    """Return one orientation witness whose two profiles are each constant across the three grids."""
    return product.OrientationObservation(
        channel_id=channel_id,
        time=time,
        tolerance=tolerance,
        profile_a=(quaternion_a, quaternion_a, quaternion_a),
        profile_b=(quaternion_b, quaternion_b, quaternion_b),
    )


def _case(
    case_id: str = "case",
    *,
    scalars: tuple[product.ScalarObservation, ...] = (),
    orientations: tuple[product.OrientationObservation, ...] = (),
    gates: tuple[product.GateEvent, ...] = (),
    minimum_prefix: float = _MINIMUM_PREFIX,
    horizon: float = _HORIZON,
) -> product.CaseEvidence:
    """Return one case built only from the module's own public value objects."""
    return product.CaseEvidence(
        case_id=case_id,
        scalar_observations=scalars,
        orientation_observations=orientations,
        gate_events=gates,
        minimum_prefix=minimum_prefix,
        horizon=horizon,
    )


# --------------------------------------------------------------------------------------------
# Completed scalar decisions and the tolerance boundary
# --------------------------------------------------------------------------------------------


def test_identical_stable_scalar_profiles_are_within_the_declared_envelope() -> None:
    """Two identical constant profiles differ only by the numerical floor, far inside tolerance."""
    result = product.evaluate_case(_case(scalars=(_constant_scalar("ch.a", 0.25, 1.0, 1.0),)))

    assert result["status"] == "WITHIN_DECLARED_MIGRATION_ENVELOPE"
    assert result["witness_count"] == 1
    witness = result["witnesses"][0]
    assert witness["classification"] == "WITHIN_DECLARED_MIGRATION_ENVELOPE"
    # The separation is zero, so the whole difference interval is a few epsilon wide.
    assert witness["upper_abs"] < 10.0 * _EPSILON


def test_a_separation_far_above_tolerance_is_outside_the_declared_envelope() -> None:
    """A half-unit separation cannot be confused with a 1e-3 tolerance at any epsilon."""
    result = product.evaluate_case(_case(scalars=(_constant_scalar("ch.a", 0.25, 1.0, 1.5),)))

    assert result["status"] == "OUTSIDE_DECLARED_MIGRATION_ENVELOPE"
    witness = result["witnesses"][0]
    assert witness["classification"] == "OUTSIDE_DECLARED_MIGRATION_ENVELOPE"
    # The lower magnitude bound stays just under the nominal 0.5 separation.
    assert 0.5 - 10.0 * _EPSILON <= witness["lower_abs"] <= 0.5


def test_a_separation_exactly_at_tolerance_is_unresolved_near_the_boundary() -> None:
    """At the boundary the interval straddles the tolerance, which is neither within nor outside.

    Two constant profiles separated by exactly the tolerance produce a difference interval of about
    ``[tolerance - 2 * epsilon, tolerance + 2 * epsilon]``. Its upper magnitude exceeds tolerance, so
    the case is not within; its lower magnitude does not, so the case is not outside.
    """
    result = product.evaluate_case(
        _case(scalars=(_constant_scalar("ch.a", 0.25, 0.0, _TOLERANCE),))
    )

    assert result["status"] == "UNRESOLVED_NEAR_BOUNDARY"
    witness = result["witnesses"][0]
    assert witness["lower_abs"] <= _TOLERANCE < witness["upper_abs"]


def test_the_worst_classification_decides_a_multi_witness_case() -> None:
    """Outside takes precedence over unresolved, which takes precedence over within."""
    result = product.evaluate_case(
        _case(
            scalars=(
                _constant_scalar("ch.a", 0.10, 1.0, 1.0),
                _constant_scalar("ch.b", 0.20, 0.0, _TOLERANCE),
                _constant_scalar("ch.c", 0.30, 1.0, 1.5),
            )
        )
    )

    assert result["status"] == "OUTSIDE_DECLARED_MIGRATION_ENVELOPE"
    assert result["witness_count"] == 3


def test_a_noncontracting_scalar_profile_is_a_non_asymptotic_regime() -> None:
    """A profile whose refinement grows cannot pass the frozen contraction gate.

    The successive differences are both 1.0, so the second is not below ``rho`` times the first and
    the profile is not stable. The gate fires at the observation's own time, which is zero here, so
    it decides the case outright.
    """
    diverging = product.ScalarObservation(
        channel_id="ch.a",
        time=0.0,
        scale=1.0,
        tolerance=_TOLERANCE,
        profile_a=(0.0, 1.0, 2.0),
        profile_b=(0.0, 0.0, 0.0),
    )

    result = product.evaluate_case(_case(scalars=(diverging,)))

    assert result["status"] == "NON_ASYMPTOTIC_REGIME"
    assert result["admitted_prefix"] == 0.0
    assert result["witness_count"] == 0
    assert result["first_failing_gate"]["status"] == "NON_ASYMPTOTIC_REGIME"
    assert result["first_failing_gate"]["channel_id"] == "ch.a"


# --------------------------------------------------------------------------------------------
# Gate precedence, prefix, and the declared horizon
# --------------------------------------------------------------------------------------------


def test_the_earliest_gate_decides_regardless_of_declaration_order() -> None:
    """Gate selection is by time first, so a later-declared earlier gate still wins."""
    result = product.evaluate_case(
        _case(
            scalars=(_constant_scalar("ch.a", 0.10, 1.0, 1.0),),
            gates=(
                product.GateEvent(0.80, "SOLVER_NOT_CONVERGED", "late.channel", "later"),
                product.GateEvent(0.60, "SOLVER_NOT_CONVERGED", "early.channel", "earlier"),
            ),
        )
    )

    assert result["first_failing_gate"]["time"] == 0.60
    assert result["first_failing_gate"]["channel_id"] == "early.channel"
    assert result["admitted_prefix"] == 0.60
    assert result["unqualified_suffix"] == [0.60, _HORIZON]


def test_gate_label_precedence_breaks_a_time_tie_before_the_channel_name() -> None:
    """At equal time the declared label order decides, ahead of the lexical channel name.

    The frozen order is refused, repeatability, solver, contact topology, non-asymptotic. The
    solver gate below therefore wins even though its channel name sorts after the other's.
    """
    result = product.evaluate_case(
        _case(
            gates=(
                product.GateEvent(0.60, "CONTACT_EVENT_TOPOLOGY_CHANGED", "aaa.channel", "later"),
                product.GateEvent(0.60, "SOLVER_NOT_CONVERGED", "zzz.channel", "earlier label"),
            ),
            scalars=(_constant_scalar("ch.a", 0.10, 1.0, 1.0),),
        )
    )

    assert result["first_failing_gate"]["status"] == "SOLVER_NOT_CONVERGED"
    assert result["first_failing_gate"]["channel_id"] == "zzz.channel"


def test_the_lexically_smaller_channel_breaks_a_full_gate_tie() -> None:
    """With time and label equal, the channel name is the last deterministic tie break."""
    result = product.evaluate_case(
        _case(
            gates=(
                product.GateEvent(0.60, "SOLVER_NOT_CONVERGED", "b.channel", "second"),
                product.GateEvent(0.60, "SOLVER_NOT_CONVERGED", "a.channel", "first"),
            ),
            scalars=(_constant_scalar("ch.a", 0.10, 1.0, 1.0),),
        )
    )

    assert result["first_failing_gate"]["channel_id"] == "a.channel"


def test_a_gate_at_time_zero_decides_the_case_with_its_own_label() -> None:
    """Nothing is admitted before time zero, so the gate's own status becomes the case status."""
    result = product.evaluate_case(
        _case(
            gates=(product.GateEvent(0.0, "REPEATABILITY_FAILED", "diagnostics", "unstable"),),
            scalars=(_constant_scalar("ch.a", 0.10, 1.0, 1.0),),
        )
    )

    assert result["status"] == "REPEATABILITY_FAILED"
    assert result["admitted_prefix"] == 0.0
    assert result["unqualified_suffix"] == [0.0, _HORIZON]
    assert result["witness_count"] == 0


def test_a_gate_inside_the_minimum_prefix_reports_prefix_too_short() -> None:
    """An admitted prefix shorter than the declared minimum cannot carry a completed decision."""
    result = product.evaluate_case(
        _case(
            gates=(product.GateEvent(0.25, "SOLVER_NOT_CONVERGED", "diagnostics", "early"),),
            scalars=(_constant_scalar("ch.a", 0.10, 1.0, 1.0),),
        )
    )

    assert result["status"] == "PREFIX_TOO_SHORT"
    assert result["admitted_prefix"] == 0.25
    assert result["unqualified_suffix"] == [0.25, _HORIZON]
    assert result["witness_count"] == 0


def test_only_observations_strictly_inside_the_admitted_prefix_become_witnesses() -> None:
    """The admitted prefix ends at the gate, so an observation at or after it is excluded."""
    result = product.evaluate_case(
        _case(
            gates=(product.GateEvent(0.60, "SOLVER_NOT_CONVERGED", "diagnostics", "gate"),),
            scalars=(
                _constant_scalar("inside", 0.10, 1.0, 1.0),
                _constant_scalar("at.the.gate", 0.60, 1.0, 1.5),
                _constant_scalar("after", 0.80, 1.0, 1.5),
            ),
        )
    )

    assert result["witness_count"] == 1
    assert result["witnesses"][0]["channel_id"] == "inside"
    assert result["status"] == "WITHIN_DECLARED_MIGRATION_ENVELOPE"


def test_an_empty_case_refuses_over_its_own_nondefault_horizon() -> None:
    """The unqualified suffix always spans the case's declared horizon, not the default one."""
    result = product.evaluate_case(_case("empty", minimum_prefix=0.5, horizon=2.0))

    assert result["status"] == "REFUSED"
    assert result["admitted_prefix"] == 0.0
    assert result["unqualified_suffix"] == [0.0, 2.0]


def test_a_gate_beyond_the_horizon_refuses_over_the_declared_horizon() -> None:
    """Gate evidence outside the declared horizon is malformed, so the case fails closed."""
    result = product.evaluate_case(
        _case(
            "beyond-horizon",
            gates=(product.GateEvent(2.0, "SOLVER_NOT_CONVERGED", "diagnostics", "late"),),
            horizon=1.0,
        )
    )

    assert result["status"] == "REFUSED"
    assert result["admitted_prefix"] == 0.0
    assert result["unqualified_suffix"] == [0.0, 1.0]
    assert result["first_failing_gate"]["status"] == "REFUSED"
    assert result["first_failing_gate"]["time"] == 0.0


# --------------------------------------------------------------------------------------------
# Non-finite evidence fails closed ahead of any decision work
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_non_finite_scalar_evidence_fails_closed_through_the_solver_gate(bad: float) -> None:
    """Non-finite required evidence is a solver failure, decided before any witness is built."""
    result = product.evaluate_case(
        _case(
            scalars=(
                product.ScalarObservation(
                    channel_id="ch.bad",
                    time=0.0,
                    scale=1.0,
                    tolerance=_TOLERANCE,
                    profile_a=(bad, 0.0, 0.0),
                    profile_b=(0.0, 0.0, 0.0),
                ),
                _constant_scalar("ch.good", 0.25, 1.0, 1.0),
            )
        )
    )

    assert result["status"] == "SOLVER_NOT_CONVERGED"
    assert result["witness_count"] == 0
    assert result["first_failing_gate"]["channel_id"] == "ch.bad"


def test_non_finite_quaternion_evidence_fails_closed_through_the_solver_gate() -> None:
    """A non-finite quaternion component is a solver failure, not an orientation measurement."""
    result = product.evaluate_case(
        _case(
            orientations=(
                _constant_orientation(
                    "q.bad", 0.0, (math.nan, 0.0, 0.0, 0.0), _rotation_about_x(0.0)
                ),
            )
        )
    )

    assert result["status"] == "SOLVER_NOT_CONVERGED"
    assert result["witness_count"] == 0
    assert result["first_failing_gate"]["channel_id"] == "q.bad"


def test_the_solver_gate_outranks_a_non_asymptotic_gate_at_the_same_time() -> None:
    """Solver failure precedes the contraction gate in the frozen label order."""
    diverging = product.ScalarObservation(
        channel_id="zzz.diverging",
        time=0.0,
        scale=1.0,
        tolerance=_TOLERANCE,
        profile_a=(0.0, 1.0, 2.0),
        profile_b=(0.0, 0.0, 0.0),
    )
    non_finite = product.ScalarObservation(
        channel_id="aaa.nonfinite",
        time=0.0,
        scale=1.0,
        tolerance=_TOLERANCE,
        profile_a=(math.nan, 0.0, 0.0),
        profile_b=(0.0, 0.0, 0.0),
    )

    result = product.evaluate_case(_case(scalars=(diverging, non_finite)))

    assert result["status"] == "SOLVER_NOT_CONVERGED"
    assert result["first_failing_gate"]["channel_id"] == "aaa.nonfinite"


# --------------------------------------------------------------------------------------------
# Orientation evidence
# --------------------------------------------------------------------------------------------


def test_a_quaternion_and_its_negation_describe_the_same_orientation() -> None:
    """``q`` and ``-q`` are the same rotation, so their separation must measure as zero.

    If the angle were not sign invariant this pair would measure a half turn instead, which is far
    outside the tolerance, so this assertion is what pins the invariance.
    """
    identity = _rotation_about_x(0.0)
    negated = tuple(-component for component in identity)

    result = product.evaluate_case(
        _case(orientations=(_constant_orientation("q.a", 0.25, identity, negated),))
    )

    assert result["status"] == "WITHIN_DECLARED_MIGRATION_ENVELOPE"
    assert result["witnesses"][0]["upper_abs"] < 10.0 * _EPSILON


def test_a_small_orientation_separation_is_within_the_declared_envelope() -> None:
    """A rotation of one microradian is three orders of magnitude inside a 1e-3 tolerance."""
    result = product.evaluate_case(
        _case(
            orientations=(
                _constant_orientation("q.a", 0.25, _rotation_about_x(0.0), _rotation_about_x(1e-6)),
            )
        )
    )

    assert result["status"] == "WITHIN_DECLARED_MIGRATION_ENVELOPE"


def test_a_near_half_turn_orientation_separation_is_outside_the_declared_envelope() -> None:
    """A separation just under a half turn is a meaningfully large orientation difference.

    The geodesic angle is bounded above by pi, and this measured interval stays inside ``[0, pi]``
    while its lower bound sits far above tolerance. It does not reach the upper clamp: the interval
    widening here is far smaller than the remaining distance to pi. The exact-half-turn case below
    is the one that sits on that clamp.
    """
    angle = math.pi - 1e-6

    result = product.evaluate_case(
        _case(
            orientations=(
                _constant_orientation(
                    "q.a", 0.25, _rotation_about_x(0.0), _rotation_about_x(angle)
                ),
            )
        )
    )

    assert result["status"] == "OUTSIDE_DECLARED_MIGRATION_ENVELOPE"
    witness = result["witnesses"][0]
    lower, upper = witness["difference_interval"]
    assert 0.0 <= lower <= angle <= upper <= math.pi


def test_a_tolerance_equal_to_the_measured_bound_is_still_within_the_envelope() -> None:
    """The envelope test is inclusive: a bound exactly equal to tolerance is inside it.

    The geodesic angle between two orientations is bounded above by pi, and the module clamps the
    measured interval to that bound. An exact half turn therefore has an upper magnitude of exactly
    pi, so declaring a tolerance of pi is the one construction where the comparison sits precisely
    on the boundary rather than near it.
    """
    result = product.evaluate_case(
        _case(
            orientations=(
                _constant_orientation(
                    "q.a",
                    0.25,
                    _rotation_about_x(0.0),
                    _rotation_about_x(math.pi),
                    tolerance=math.pi,
                ),
            )
        )
    )

    witness = result["witnesses"][0]
    assert witness["difference_interval"][1] == math.pi
    assert witness["classification"] == "WITHIN_DECLARED_MIGRATION_ENVELOPE"
    assert result["status"] == "WITHIN_DECLARED_MIGRATION_ENVELOPE"


def test_a_noncontracting_orientation_profile_is_a_non_asymptotic_regime() -> None:
    """An orientation profile whose refinement grows fails the same frozen contraction gate."""
    diverging = product.OrientationObservation(
        channel_id="q.a",
        time=0.0,
        tolerance=_TOLERANCE,
        profile_a=(
            _rotation_about_x(0.0),
            _rotation_about_x(0.1),
            _rotation_about_x(0.3),
        ),
        profile_b=(_rotation_about_x(0.0),) * 3,
    )

    result = product.evaluate_case(_case(orientations=(diverging,)))

    assert result["status"] == "NON_ASYMPTOTIC_REGIME"
    assert result["first_failing_gate"]["channel_id"] == "q.a"


# --------------------------------------------------------------------------------------------
# Witness ordering and the two independent witness selections
# --------------------------------------------------------------------------------------------


def test_witnesses_are_ordered_by_time_then_channel_name() -> None:
    """Witness order and the earliest-witness choice are both deterministic.

    Two of these three share the earliest time, so the lexically smaller channel is the only thing
    that can separate them. That makes this the case that binds the channel component of the
    earliest-witness selection, not just the order of the reported list.
    """
    result = product.evaluate_case(
        _case(
            scalars=(
                _constant_scalar("z.channel", 0.30, 1.0, 1.0),
                _constant_scalar("b.channel", 0.10, 1.0, 1.0),
                _constant_scalar("a.channel", 0.10, 1.0, 1.0),
            )
        )
    )

    assert [witness["channel_id"] for witness in result["witnesses"]] == [
        "a.channel",
        "b.channel",
        "z.channel",
    ]
    assert result["first_witness"]["time"] == 0.10
    assert result["first_witness"]["channel_id"] == "a.channel"


def test_the_first_and_worst_witnesses_are_selected_independently() -> None:
    """The earliest witness and the largest ratio are different selections over the same rows.

    The early witness is identical between profiles, so its ratio is about ``2e-12 / 1e-3``. The
    later witness is separated by half a unit, so its ratio is about ``500``.
    """
    result = product.evaluate_case(
        _case(
            scalars=(
                _constant_scalar("a.early.small", 0.10, 1.0, 1.0),
                _constant_scalar("z.later.large", 0.40, 1.0, 1.5),
            )
        )
    )

    assert result["first_witness"]["channel_id"] == "a.early.small"
    assert result["worst_witness"]["channel_id"] == "z.later.large"
    assert result["worst_witness"]["ratio"] > result["first_witness"]["ratio"]


def test_the_worst_witness_tie_breaks_on_time_then_channel() -> None:
    """Three witnesses tie on ratio, so time decides first and the channel name decides after it.

    All three carry the same separation and the same tolerance, so their ratios are equal and the
    ratio cannot separate them. Two share the earliest time and are separated only by channel name;
    the third has a lexically earlier name but a later time, so it can only win if time stops being
    consulted before the name. Both components are therefore load-bearing here: the winner must be
    the lexically smaller member of the earliest-time pair, and neither the earliest time alone nor
    the smallest name alone identifies it.
    """
    result = product.evaluate_case(
        _case(
            scalars=(
                _constant_scalar("z.channel", 0.10, 1.0, 1.5),
                _constant_scalar("b.channel", 0.10, 1.0, 1.5),
                _constant_scalar("a.channel", 0.30, 1.0, 1.5),
            )
        )
    )

    assert result["witness_count"] == 3
    ratios = {witness["ratio"] for witness in result["witnesses"]}
    assert len(ratios) == 1
    assert result["worst_witness"]["time"] == 0.10
    assert result["worst_witness"]["channel_id"] == "b.channel"


# --------------------------------------------------------------------------------------------
# Malformed metadata and the exported surface
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case_id", "minimum_prefix", "horizon"),
    [
        pytest.param("", 0.5, 1.0, id="empty_case_id"),
        pytest.param("c", 0.0, 1.0, id="nonpositive_minimum_prefix"),
        pytest.param("c", 0.5, 0.25, id="horizon_shorter_than_minimum_prefix"),
        pytest.param("c", math.nan, 1.0, id="non_finite_minimum_prefix"),
    ],
)
def test_malformed_case_metadata_refuses(
    case_id: str, minimum_prefix: float, horizon: float
) -> None:
    """Metadata is validated before any scientific work, and failure is a closed refusal."""
    result = product.evaluate_case(_case(case_id, minimum_prefix=minimum_prefix, horizon=horizon))

    assert result["status"] == "REFUSED"
    assert result["witness_count"] == 0
    assert result["first_failing_gate"]["status"] == "REFUSED"


def test_a_duplicate_time_and_channel_identity_refuses() -> None:
    """Two observations cannot claim the same witness identity."""
    result = product.evaluate_case(
        _case(
            scalars=(
                _constant_scalar("ch.a", 0.25, 1.0, 1.0),
                _constant_scalar("ch.a", 0.25, 1.0, 1.5),
            )
        )
    )

    assert result["status"] == "REFUSED"


def test_every_completed_result_names_the_one_selected_method() -> None:
    """The core implements a single preregistered method and always says which."""
    result = product.evaluate_case(_case(scalars=(_constant_scalar("ch.a", 0.25, 1.0, 1.0),)))

    assert result["method"] == "CONDITIONAL_TAIL_ENVELOPE"


def test_the_exported_surface_is_exactly_the_declared_value_objects_and_entry_point() -> None:
    """The private core exports its value objects and one entry point, and nothing else.

    A generic method selector would need either a second entry point or a method argument. Neither
    exists: the export list is fixed, and the single entry point takes one case and nothing more.
    """
    assert product.__all__ == [
        "Interval",
        "GateEvent",
        "ScalarObservation",
        "OrientationObservation",
        "CaseEvidence",
        "evaluate_case",
    ]
    for name in product.__all__:
        assert hasattr(product, name)

    parameters = product.evaluate_case.__code__.co_varnames[
        : product.evaluate_case.__code__.co_argcount
    ]
    assert parameters == ("case",)
