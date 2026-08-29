"""The exact in-house qualification decision, over compare-backed detection evidence only.

Nothing here simulates, estimates, or infers. It consumes one detection cell per
(workload, probe variant) pair, already decided by a completed Metrifid comparison, and applies two
rules:

D3-BL-065 suffix/floor adjudication
    Rungs are ordered by increasing magnitude. A detection floor exists only when some rung and
    every larger rung are DETECTED. The floor is the smallest such rung. A single detection at a
    small magnitude with a gap above it establishes nothing, which is why the suffix condition, not
    the first detection, defines the floor.

Exact enumeration over every three-workload subset
    Schema version 1 admits at most sixteen workloads and freezes the budget at three, so at most
    560 subsets exist. The search is exhaustive by construction and needs no host-timed bound and no
    third-party optimizer.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from ..json_values import CanonicalValue, ExactRational
from ._config import ProbeGroup, QualificationConfig
from ._status import CellOutcome, ProbeGroupStatus, QualificationStatus

MAX_SUBSETS: Final[int] = 560


@dataclass(frozen=True, slots=True)
class GroupAdjudication:
    """One probe group's ordered signature, floor, and status under one workload subset."""

    probe_id: str
    status: ProbeGroupStatus
    signature: tuple[CellOutcome, ...]
    floor_magnitude: ExactRational | None
    floor_index: int | None
    no_floor_reason: str | None
    detected_variants: int

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the adjudication with the ladder order preserved."""
        return {
            "probe_id": self.probe_id,
            "status": self.status.value,
            "detection_signature": [outcome.value for outcome in self.signature],
            "floor_magnitude": (
                None if self.floor_magnitude is None else self.floor_magnitude.to_decimal_token()
            ),
            "floor_variant_index": self.floor_index,
            "no_floor_reason": self.no_floor_reason,
            "detected_variants": self.detected_variants,
        }


@dataclass(frozen=True, slots=True)
class SubsetAdjudication:
    """One three-workload subset, fully adjudicated against every declared probe group."""

    workload_ids: tuple[str, ...]
    groups: tuple[GroupAdjudication, ...]

    @property
    def qualified(self) -> int:
        """Number of probe groups this subset qualifies."""
        return sum(1 for group in self.groups if group.status is ProbeGroupStatus.QUALIFIED)

    @property
    def unresolved(self) -> int:
        """Number of probe groups this subset leaves unresolved."""
        return sum(1 for group in self.groups if group.status is ProbeGroupStatus.UNRESOLVED)

    @property
    def detected_variants(self) -> int:
        """Total detected variants across every probe group."""
        return sum(group.detected_variants for group in self.groups)

    def rank_key(self) -> tuple[int, int, int, tuple[str, ...]]:
        """Return the exact total order declared for subset ranking.

        More qualified groups first, then fewer unresolved groups, then more detected variants,
        then the lexicographically smaller tuple of workload identities. The final component makes
        the order total, so the selected subset never depends on iteration order.
        """
        return (-self.qualified, self.unresolved, -self.detected_variants, self.workload_ids)

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the subset adjudication."""
        return {
            "workload_ids": list(self.workload_ids),
            "qualified_groups": self.qualified,
            "unresolved_groups": self.unresolved,
            "detected_variants": self.detected_variants,
            "groups": [group.to_primitive() for group in self.groups],
        }


@dataclass(frozen=True, slots=True)
class QualificationDecision:
    """The selected subset, its status, the ranking behind it, and the counts."""

    status: QualificationStatus
    selected: SubsetAdjudication
    subsets_evaluated: int
    eligible_workload_ids: tuple[str, ...]
    ranking: tuple[dict[str, CanonicalValue], ...]


def cell_outcomes_for_subset(
    matrix: Mapping[tuple[str, str, int], CellOutcome],
    workload_ids: Sequence[str],
    group: ProbeGroup,
) -> tuple[CellOutcome, ...]:
    """Collapse one probe ladder across the workloads in one subset.

    A variant is detected when any workload in the subset detected it. Otherwise it is unresolved
    when any workload left it unresolved, and only then not detected. An unresolved comparison is
    never silently read as a non-detection.
    """
    signature: list[CellOutcome] = []
    for index in range(len(group.variants)):
        cells = [matrix[(workload_id, group.probe_id, index)] for workload_id in workload_ids]
        if any(cell is CellOutcome.DETECTED for cell in cells):
            signature.append(CellOutcome.DETECTED)
        elif any(cell is CellOutcome.UNRESOLVED for cell in cells):
            signature.append(CellOutcome.UNRESOLVED)
        else:
            signature.append(CellOutcome.NOT_DETECTED)
    return tuple(signature)


def detection_floor(signature: Sequence[CellOutcome]) -> int | None:
    """Return the smallest rung whose whole suffix is DETECTED, or None when none exists."""
    floor: int | None = None
    for index in range(len(signature) - 1, -1, -1):
        if signature[index] is CellOutcome.DETECTED:
            floor = index
        else:
            break
    return floor


def adjudicate_group(group: ProbeGroup, signature: Sequence[CellOutcome]) -> GroupAdjudication:
    """Decide one probe group from its ordered detection signature."""
    required = group.required_index()
    floor = detection_floor(signature)
    detected = sum(1 for outcome in signature if outcome is CellOutcome.DETECTED)
    suffix = signature[required:]

    if floor is not None and floor <= required:
        return GroupAdjudication(
            probe_id=group.probe_id,
            status=ProbeGroupStatus.QUALIFIED,
            signature=tuple(signature),
            floor_magnitude=group.variants[floor].magnitude,
            floor_index=floor,
            no_floor_reason=None,
            detected_variants=detected,
        )
    if any(outcome is CellOutcome.UNRESOLVED for outcome in suffix):
        reason = (
            "a rung at or above the required detection magnitude is unresolved, so no qualifying "
            "floor can be established"
        )
        status = ProbeGroupStatus.UNRESOLVED
    else:
        reason = (
            "every rung at or above the required detection magnitude is completed, and no rung at "
            "or below the required magnitude begins an unbroken detected suffix"
        )
        status = ProbeGroupStatus.INSUFFICIENT
    return GroupAdjudication(
        probe_id=group.probe_id,
        status=status,
        signature=tuple(signature),
        floor_magnitude=None if floor is None else group.variants[floor].magnitude,
        floor_index=floor,
        no_floor_reason=reason,
        detected_variants=detected,
    )


def adjudicate_subset(
    config: QualificationConfig,
    matrix: Mapping[tuple[str, str, int], CellOutcome],
    workload_ids: Sequence[str],
) -> SubsetAdjudication:
    """Adjudicate every declared probe group under one workload subset."""
    groups = tuple(
        adjudicate_group(group, cell_outcomes_for_subset(matrix, workload_ids, group))
        for group in config.probe_groups
    )
    # Sorted, so the lexicographic tie-break compares the subset itself rather than the order the
    # user happened to declare its members in.
    return SubsetAdjudication(workload_ids=tuple(sorted(workload_ids)), groups=groups)


def overall_status(subset: SubsetAdjudication) -> QualificationStatus:
    """Map one adjudicated subset to exactly one completed qualification status."""
    statuses = [group.status for group in subset.groups]
    if any(status is ProbeGroupStatus.UNRESOLVED for status in statuses):
        return QualificationStatus.UNRESOLVED
    if all(status is ProbeGroupStatus.QUALIFIED for status in statuses):
        return QualificationStatus.QUALIFIED_FOR_DECLARED_PROBES
    if any(status is ProbeGroupStatus.QUALIFIED for status in statuses):
        return QualificationStatus.PARTIALLY_QUALIFIED
    return QualificationStatus.INSUFFICIENT_EXCITATION


def decide(
    config: QualificationConfig,
    matrix: Mapping[tuple[str, str, int], CellOutcome],
    eligible_workload_ids: Sequence[str],
) -> QualificationDecision:
    """Enumerate every admitted subset exactly and return the first under the declared order."""
    eligible = tuple(eligible_workload_ids)
    if len(eligible) < config.budget:
        raise ValueError("fewer eligible workloads than the declared budget")
    best: SubsetAdjudication | None = None
    evaluated = 0
    ranking: list[dict[str, CanonicalValue]] = []
    for combination in itertools.combinations(eligible, config.budget):
        subset = adjudicate_subset(config, matrix, combination)
        evaluated += 1
        # Recorded for every evaluated subset, so a reader can reproduce the winner from the
        # receipt alone instead of trusting that the selection was made correctly.
        ranking.append(
            {
                "workload_ids": list(subset.workload_ids),
                "qualified_groups": subset.qualified,
                "unresolved_groups": subset.unresolved,
                "detected_variants": subset.detected_variants,
            }
        )
        if best is None or subset.rank_key() < best.rank_key():
            best = subset
    if best is None:  # pragma: no cover - guarded by the budget check above
        raise ValueError("no subset was evaluated")
    if evaluated > MAX_SUBSETS:  # pragma: no cover - bounded by the schema
        raise ValueError("subset enumeration exceeded the schema bound")

    def _rank(item: dict[str, CanonicalValue]) -> tuple[int, int, int, tuple[str, ...]]:
        """Order the recorded ranking exactly as the selector's own total order does."""
        qualified = item["qualified_groups"]
        unresolved = item["unresolved_groups"]
        detected = item["detected_variants"]
        identities = item["workload_ids"]
        assert isinstance(qualified, int)
        assert isinstance(unresolved, int)
        assert isinstance(detected, int)
        assert isinstance(identities, list)
        return (-qualified, unresolved, -detected, tuple(str(name) for name in identities))

    ranking.sort(key=_rank)
    return QualificationDecision(
        status=overall_status(best),
        selected=best,
        subsets_evaluated=evaluated,
        eligible_workload_ids=eligible,
        ranking=tuple(ranking),
    )
