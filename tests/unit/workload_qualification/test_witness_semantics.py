"""The status-bearing witness must explain the status it is attached to.

The defect this pins: a green group published an unresolved rung from below its required magnitude
as the single first witness. That witness explained nothing about the green decision, and the
ordering text claimed probe-only ordering while control-first ordering was applied.
"""

from __future__ import annotations

import pytest
import pytest_check as check

from metrifid.errors import ComparisonStatus
from metrifid.workload_qualification import QualificationStatus
from metrifid.workload_qualification._decision import decide
from metrifid.workload_qualification._evidence import exclusion_reason
from metrifid.workload_qualification._status import ProbeGroupStatus
from metrifid.workload_qualification._witness import collect_witnesses
from tests._support.workload_qualification import config, matrix, probe_group

LADDER = ("0.1", "0.2", "0.3", "0.4")
IDS = ("w1", "w2", "w3")


def _run(signature: str, required: str, *, excluded: tuple[str, ...] = ()):
    """Adjudicate one signature shared by every workload and collect its witnesses."""
    group = probe_group("p", LADDER, required)
    qualification = config([group], (*IDS, *excluded))
    cells = matrix({name: {"p": signature} for name in (*IDS, *excluded)})
    decision = decide(qualification, cells, IDS)
    reasons = {
        name: exclusion_reason(ComparisonStatus.MATERIAL_BEHAVIOR_CHANGE) for name in excluded
    }
    witnesses = collect_witnesses(
        qualification, decision.selected, decision.status, cells, excluded, reasons
    )
    return decision, witnesses


def test_a_green_group_with_an_unresolved_rung_below_the_requirement_publishes_no_witness() -> None:
    """The mandatory regression from the cross-audit, exactly as specified.

    Signature U D D D with a required magnitude of 0.3 has a suffix floor of 0.2, so the group is
    qualified and the run is green. The unresolved rung at 0.1 is below the requirement: it stays
    visible in the matrix and must not become the first witness for a completed green decision.
    """
    decision, witnesses = _run("UDDD", "0.3")
    group = decision.selected.groups[0]

    check.equal(
        group.status,
        ProbeGroupStatus.QUALIFIED,
        "the probe group is not reported as qualified even though its detected suffix reaches the "
        "required magnitude",
    )
    assert group.floor_magnitude is not None, (
        "the qualified group reports no qualifying floor at all, so the magnitude its receipt "
        "would publish cannot be read"
    )
    check.equal(
        group.floor_magnitude.to_decimal_token(),
        "0.2",
        "the published qualifying floor is not the smallest magnitude whose detected suffix runs "
        "unbroken to the top of the ladder",
    )
    check.equal(
        decision.status,
        QualificationStatus.QUALIFIED_FOR_DECLARED_PROBES,
        "a run whose only probe group qualified is not reported as qualified for the declared "
        "probes",
    )
    check.is_none(
        witnesses["first_witness"],
        "the green run publishes a status-bearing witness, which tells the reader the "
        "qualification was blocked when it was not",
    )
    check.is_none(
        witnesses["first_unresolved_witness"],
        "an unresolved rung below the required magnitude is published as the unresolved witness, "
        "even though such a rung can never hold back the qualifying floor",
    )


def test_a_green_run_with_an_excluded_workload_still_publishes_no_status_witness() -> None:
    """A false alarm is an eligibility warning, never the status-bearing witness."""
    _decision, witnesses = _run("DDDD", "0.3", excluded=("noisy",))

    check.is_none(
        witnesses["first_witness"],
        "excluding a workload for a zero-change false alarm turned into the status-bearing "
        "witness, which explains nothing about the decision the retained workloads produced",
    )
    alarm = witnesses["first_false_alarm_witness"]
    assert alarm is not None, (
        "a workload was excluded for a zero-change false alarm, yet the receipt carries no false "
        "alarm witness to warn the reader about it"
    )
    check.equal(
        alarm["kind"],
        "ZERO_CHANGE_FALSE_ALARM",
        "the eligibility warning is not reported as a zero-change false alarm",
    )
    check.equal(
        alarm["workload_id"],
        "noisy",
        "the false alarm warning names a workload other than the one that was excluded",
    )


def test_an_unresolved_run_publishes_the_first_unresolved_rung_at_or_above_the_requirement() -> (
    None
):
    """UNRESOLVED is explained by an unresolved rung that blocks the floor."""
    decision, witnesses = _run("DDUD", "0.3")

    check.equal(
        decision.status,
        QualificationStatus.UNRESOLVED,
        "a run whose rung at the required magnitude never completed as a decision is not reported "
        "as unresolved",
    )
    first = witnesses["first_witness"]
    assert first is not None, (
        "the unresolved run publishes no status-bearing witness, so the receipt gives the reader "
        "nothing that explains why qualification was blocked"
    )
    check.equal(
        first["kind"],
        "UNRESOLVED_RUNG",
        "the witness explaining an unresolved run is not an unresolved rung",
    )
    check.equal(
        first["magnitude"],
        "0.3",
        "the published witness is not the rung at the required magnitude that actually blocked "
        "the qualifying floor",
    )
    check.equal(
        witnesses["first_witness"],
        witnesses["first_unresolved_witness"],
        "the status-bearing witness for an unresolved run is not the same rung the receipt reports "
        "as its first unresolved witness",
    )


def test_an_insufficient_run_publishes_the_first_blind_rung() -> None:
    """INSUFFICIENT_EXCITATION is explained by the rung nothing detected."""
    decision, witnesses = _run("DDND", "0.3")

    check.equal(
        decision.status,
        QualificationStatus.INSUFFICIENT_EXCITATION,
        "a run whose rung at the required magnitude went undetected is not reported as "
        "insufficient excitation",
    )
    first = witnesses["first_witness"]
    assert first is not None, (
        "the insufficient run publishes no status-bearing witness, so the receipt gives the reader "
        "nothing that explains why qualification was blocked"
    )
    check.equal(
        first["kind"],
        "BLIND_RUNG",
        "the witness explaining an insufficient run is not a blind rung",
    )
    check.equal(
        first["magnitude"],
        "0.3",
        "the published witness is not the rung at the required magnitude that no selected workload "
        "detected",
    )
    check.equal(
        witnesses["first_witness"],
        witnesses["first_blind_witness"],
        "the status-bearing witness for an insufficient run is not the same rung the receipt "
        "reports as its first blind witness",
    )


def test_a_partially_qualified_run_publishes_the_first_blind_rung_of_a_failing_group() -> None:
    """A qualified group never supplies the witness for a partially qualified run."""
    good = probe_group("good", LADDER, "0.3")
    bad = probe_group("bad", LADDER, "0.3")
    ids = IDS
    qualification = config([good, bad], ids)
    cells = matrix({name: {"good": "DDDD", "bad": "DDND"} for name in ids})
    decision = decide(qualification, cells, ids)
    witnesses = collect_witnesses(qualification, decision.selected, decision.status, cells, (), {})

    check.equal(
        decision.status,
        QualificationStatus.PARTIALLY_QUALIFIED,
        "a run with one qualified group and one failing group is not reported as partially "
        "qualified",
    )
    first = witnesses["first_witness"]
    assert first is not None, (
        "the partially qualified run publishes no status-bearing witness, so the receipt gives the "
        "reader nothing that explains which probe fell short"
    )
    check.equal(
        first["probe_id"],
        "bad",
        "the published witness comes from the probe group that qualified, so it explains nothing "
        "about the group that fell short",
    )


def test_the_published_ordering_text_matches_the_applied_order() -> None:
    """The receipt states probe-declaration order and applies exactly that."""
    _decision, witnesses = _run("DDND", "0.3")

    order = str(witnesses["witness_order"])
    check.is_in(
        "probe group declaration order",
        order,
        "the published ordering text does not say witnesses are ordered by probe group "
        "declaration order, which is the order actually applied",
    )
    check.is_in(
        "increasing exact magnitude",
        order,
        "the published ordering text does not say rungs are ordered by increasing exact magnitude, "
        "which is the order actually applied",
    )
    check.is_in(
        "workload_id Unicode-code-point lexical order",
        order,
        "the published ordering text does not say contributing workloads are broken by "
        "Unicode-code-point lexical order, which is the order actually applied",
    )
    rule = str(witnesses["status_witness_rule"])
    check.is_in(
        "QUALIFIED_FOR_DECLARED_PROBES publishes no first witness",
        rule,
        "the published rule does not state that a green run publishes no status-bearing witness",
    )


@pytest.mark.parametrize("signature", ["NDDD", "UDDD"])
def test_a_rung_below_the_requirement_never_becomes_the_first_witness(signature: str) -> None:
    """Only rungs at or above the requirement can explain a completed status."""
    _decision, witnesses = _run(signature, "0.3")

    check.is_none(
        witnesses["first_witness"],
        f"signature {signature} publishes a status-bearing witness drawn from a rung below the "
        "required magnitude, which cannot explain the completed status it is attached to",
    )
