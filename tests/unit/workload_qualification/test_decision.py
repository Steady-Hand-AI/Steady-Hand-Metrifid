"""The qualification decision, checked against the exact rules it is specified by."""

from __future__ import annotations

import pytest
import pytest_check as check

from metrifid.workload_qualification import QualificationStatus
from metrifid.workload_qualification._decision import (
    adjudicate_group,
    adjudicate_subset,
    decide,
    detection_floor,
)
from metrifid.workload_qualification._status import (
    CellOutcome,
    ProbeGroupStatus,
)
from tests._support.workload_qualification import config, matrix, probe_group

LADDER = ("0.002", "0.005", "0.010", "0.020")
REQUIRED = "0.005"


def _group(required: str = REQUIRED):
    return probe_group("hinge_damping_increase", LADDER, required)


def _decide(signatures, workload_ids=("w1", "w2", "w3")):
    group = _group()
    return decide(config([group], workload_ids), matrix(signatures), workload_ids)


def test_full_qualification_when_every_declared_group_is_detected() -> None:
    """Every group detected at or above its required magnitude qualifies the whole run."""
    decision = _decide(
        {
            "w1": {"hinge_damping_increase": "NNDD"},
            "w2": {"hinge_damping_increase": "NDDD"},
            "w3": {"hinge_damping_increase": "NNND"},
        }
    )
    check.equal(
        decision.status,
        QualificationStatus.QUALIFIED_FOR_DECLARED_PROBES,
        "every declared probe group cleared its required magnitude, yet the run was not reported "
        "as qualified for the declared probes",
    )
    check.equal(
        decision.selected.groups[0].status,
        ProbeGroupStatus.QUALIFIED,
        "the damping group's detected suffix reaches below its required rung, yet the selected "
        "subset does not report the group as qualified",
    )


def test_partial_qualification_mixes_qualified_and_insufficient_groups() -> None:
    """One qualified group and one insufficient group, with none unresolved, is partial."""
    good = probe_group("good", LADDER, REQUIRED)
    bad = probe_group("bad", LADDER, REQUIRED)
    workload_ids = ("w1", "w2", "w3")
    cells = matrix(
        {
            "w1": {"good": "NDDD", "bad": "NNND"},
            "w2": {"good": "NDDD", "bad": "NNNN"},
            "w3": {"good": "NDDD", "bad": "NNNN"},
        }
    )
    decision = decide(config([good, bad], workload_ids), cells, workload_ids)
    check.equal(
        decision.status,
        QualificationStatus.PARTIALLY_QUALIFIED,
        "one group qualified and one fell short with nothing left unresolved, yet the run was "
        "not reported as partially qualified",
    )
    statuses = {g.probe_id: g.status for g in decision.selected.groups}
    check.equal(
        statuses["good"],
        ProbeGroupStatus.QUALIFIED,
        "the group detected from its required rung upwards is not reported as qualified",
    )
    check.equal(
        statuses["bad"],
        ProbeGroupStatus.INSUFFICIENT,
        "the group detected only at the largest rung, above the magnitude it was required to "
        "reach, is not reported as insufficient",
    )


def test_insufficient_excitation_when_no_group_qualifies_and_none_is_unresolved() -> None:
    """No qualified group and no unresolved group is insufficient excitation, not unresolved."""
    decision = _decide(
        {
            "w1": {"hinge_damping_increase": "NNNN"},
            "w2": {"hinge_damping_increase": "NNNN"},
            "w3": {"hinge_damping_increase": "NNND"},
        }
    )
    check.equal(
        decision.status,
        QualificationStatus.INSUFFICIENT_EXCITATION,
        "no group qualified and every rung was resolved, yet the run was not reported as "
        "insufficient excitation",
    )


def test_unresolved_required_rung_is_never_read_as_a_non_detection() -> None:
    """An unresolved rung at or above the requirement blocks a floor rather than denying one."""
    decision = _decide(
        {
            "w1": {"hinge_damping_increase": "NUDD"},
            "w2": {"hinge_damping_increase": "NUDD"},
            "w3": {"hinge_damping_increase": "NUDD"},
        }
    )
    check.equal(
        decision.status,
        QualificationStatus.UNRESOLVED,
        "a rung at the required magnitude is unresolved, yet the run was reported as something "
        "other than unresolved",
    )
    check.equal(
        decision.selected.groups[0].status,
        ProbeGroupStatus.UNRESOLVED,
        "the group's unresolved required rung was read as a non-detection instead of blocking "
        "the floor",
    )


@pytest.mark.parametrize(
    ("signature", "expected_floor"),
    [
        ("DDDD", 0),
        ("NDDD", 1),
        ("DNDD", 2),
        ("DDDN", None),
        ("NNNN", None),
        ("DUDD", 2),
    ],
)
def test_the_floor_is_the_smallest_rung_whose_whole_suffix_is_detected(
    signature: str, expected_floor: int | None
) -> None:
    """A lone detection with a gap above it establishes no floor."""
    letters = {
        "D": CellOutcome.DETECTED,
        "N": CellOutcome.NOT_DETECTED,
        "U": CellOutcome.UNRESOLVED,
    }
    check.equal(
        detection_floor([letters[c] for c in signature]),
        expected_floor,
        f"ladder signature {signature} does not yield the smallest rung whose whole suffix is "
        f"detected as its floor",
    )


def test_a_nonmonotonic_signature_qualifies_only_on_its_suffix() -> None:
    """Detection at the smallest rung does not qualify a group when a larger rung is missed."""
    letters = {"D": CellOutcome.DETECTED, "N": CellOutcome.NOT_DETECTED}
    group = _group(required="0.010")
    adjudication = adjudicate_group(group, [letters[c] for c in "DNDD"])
    check.equal(
        adjudication.status,
        ProbeGroupStatus.QUALIFIED,
        "the unbroken detected suffix starts at the required rung, yet the group is not "
        "reported as qualified",
    )
    check.equal(
        adjudication.floor_index,
        2,
        "the reported detection floor is not the required rung, where the unbroken detected "
        "suffix begins",
    )

    missed = adjudicate_group(group, [letters[c] for c in "DDND"])
    check.equal(
        missed.status,
        ProbeGroupStatus.INSUFFICIENT,
        "a group that missed its required rung is still reported as qualified on the strength "
        "of its smaller detections",
    )
    check.equal(
        missed.floor_index,
        3,
        "the reported detection floor is not the largest rung, the only one whose whole suffix "
        "is detected",
    )
    check.is_not_none(
        missed.no_floor_reason,
        "the group failed to qualify without recording why no qualifying floor was established",
    )


def test_exact_enumeration_beats_a_greedy_trap() -> None:
    """A selector that seeds on the most-detecting workloads qualifies fewer groups than exact.

    Detection is a union over the subset, so adding a workload never makes a subset worse. The trap
    is therefore not "a bad workload poisons the set"; it is that the two workloads with the most
    detected cells pile those detections onto one group that is already covered, spending budget
    that the other group needed. A greedy on detected-cell count takes ``bulk`` and ``twin`` and
    qualifies one group. Exact enumeration keeps one of them and adds the two partial workloads
    that together complete the second group.
    """
    first = probe_group("first", LADDER, "0.002")
    second = probe_group("second", LADDER, "0.002")
    workload_ids = ("bulk", "twin", "lower", "upper")
    cells = matrix(
        {
            "bulk": {"first": "DDDD", "second": "NNNN"},
            "twin": {"first": "DDDD", "second": "NNNN"},
            "lower": {"first": "NNNN", "second": "DDNN"},
            "upper": {"first": "NNNN", "second": "NNDD"},
        }
    )
    qualification = config([first, second], workload_ids)
    decision = decide(qualification, cells, workload_ids)

    # What a greedy on detected-cell count would have taken, scored by the same adjudication.
    greedy = adjudicate_subset(qualification, cells, ("bulk", "twin", "lower"))
    check.equal(
        greedy.qualified,
        1,
        "the greedy pick of the most-detecting workloads does not qualify exactly the one probe "
        "group it can actually complete",
    )

    check.equal(
        decision.selected.qualified,
        2,
        "exact enumeration did not find the subset that qualifies both declared probe groups",
    )
    check.equal(
        decision.status,
        QualificationStatus.QUALIFIED_FOR_DECLARED_PROBES,
        "both declared probe groups are reachable within the budget, yet the run was not "
        "reported as qualified for the declared probes",
    )
    check.equal(
        decision.selected.workload_ids,
        ("bulk", "lower", "upper"),
        "the selected subset is not the covering one: a single bulk workload plus the two "
        "partial workloads that together complete the second group",
    )
    check.less(
        decision.selected.rank_key(),
        greedy.rank_key(),
        "the selected subset does not outrank the greedy pick under the declared total order",
    )


def test_a_tie_is_broken_by_the_lexicographically_smaller_workload_tuple() -> None:
    """Equally good subsets resolve to the smaller sorted identity tuple, not to iteration order."""
    group = _group(required="0.002")
    workload_ids = ("delta", "alpha", "charlie", "bravo")
    cells = matrix({name: {"hinge_damping_increase": "DDDD"} for name in workload_ids})
    decision = decide(config([group], workload_ids), cells, workload_ids)
    check.equal(
        decision.selected.workload_ids,
        ("alpha", "bravo", "charlie"),
        "equally good subsets were not resolved to the lexicographically smaller workload tuple, "
        "so the selection still depends on the order the workloads were declared in",
    )


def test_fewer_eligible_workloads_than_the_budget_refuses() -> None:
    """A subset cannot be selected from fewer workloads than the declared budget."""
    group = _group()
    workload_ids = ("w1", "w2", "w3")
    cells = matrix({name: {"hinge_damping_increase": "DDDD"} for name in workload_ids})
    with pytest.raises(ValueError, match="fewer eligible workloads"):
        decide(config([group], workload_ids), cells, workload_ids[:2])


def test_every_subset_of_the_maximum_candidate_set_is_enumerated() -> None:
    """Sixteen candidates and a budget of three is exactly 560 subsets, the schema's bound."""
    group = _group(required="0.002")
    workload_ids = tuple(f"w{index:02d}" for index in range(16))
    cells = matrix({name: {"hinge_damping_increase": "DDDD"} for name in workload_ids})
    decision = decide(config([group], workload_ids), cells, workload_ids)
    check.equal(
        decision.subsets_evaluated,
        560,
        "the selector did not evaluate every three-workload subset of the sixteen-candidate "
        "maximum, so the enumeration is not exhaustive",
    )
