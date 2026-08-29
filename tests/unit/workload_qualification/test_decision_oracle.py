"""An independent reference implementation of the qualification decision, over 5,000 campaigns.

The production selector is checked against a second implementation written here from the declared
rules rather than from the production code. This module deliberately does **not** import
``cell_outcomes_for_subset``, ``detection_floor``, ``adjudicate_group``, ``adjudicate_subset`` or
any ranking helper: it imports only the data types needed to build an input and the single
production entry point under test, ``decide``.

The rules the oracle implements independently:

```text
cell collapse     a rung is DETECTED when any selected workload detected it; otherwise UNRESOLVED
                  when any left it unresolved; otherwise NOT_DETECTED
detection floor   the smallest rung whose entire suffix is DETECTED, or none
group status      QUALIFIED when a floor exists at or below the required rung; else UNRESOLVED when
                  any rung at or above the required rung is unresolved; else INSUFFICIENT
overall status    any UNRESOLVED group wins; then all QUALIFIED; then any QUALIFIED; else
                  INSUFFICIENT_EXCITATION
ranking           more qualified groups, then fewer unresolved groups, then more detected variants,
                  then the lexicographically smaller sorted workload-identity tuple
selection         the first subset under that total order, over every three-workload combination
```

The campaign generator is seeded once, so a reported failure is reproducible from the case index
alone; the failure message also carries the complete campaign that produced it.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

import pytest_check as check

from metrifid._configuration_schemas import JointToleranceConfig, ModelRoleConfig
from metrifid.json_values import ExactRational
from metrifid.workload_qualification._config import (
    ProbeGroup,
    ProbeVariant,
    QualificationConfig,
    WorkloadCandidate,
)
from metrifid.workload_qualification._decision import decide
from metrifid.workload_qualification._status import (
    CellOutcome,
    ProbeGroupStatus,
    QualificationStatus,
)

SEED = 20260819
CAMPAIGNS = 5000
WORKLOAD_RANGE = (3, 8)
GROUP_RANGE = (1, 4)
RUNG_RANGE = (2, 6)
BUDGET = 3

_OUTCOMES = (CellOutcome.DETECTED, CellOutcome.NOT_DETECTED, CellOutcome.UNRESOLVED)


# --- the independent oracle --------------------------------------------------------------------


@dataclass(frozen=True)
class OracleGroup:
    """One probe group as the oracle adjudicated it."""

    probe_id: str
    status: ProbeGroupStatus
    signature: tuple[CellOutcome, ...]
    floor_index: int | None
    detected: int


@dataclass(frozen=True)
class OracleSubset:
    """One three-workload subset as the oracle adjudicated it."""

    workload_ids: tuple[str, ...]
    groups: tuple[OracleGroup, ...]

    @property
    def qualified(self) -> int:
        """Number of probe groups this subset qualifies."""
        return sum(1 for group in self.groups if group.status is ProbeGroupStatus.QUALIFIED)

    @property
    def unresolved(self) -> int:
        """Number of probe groups this subset leaves unresolved."""
        return sum(1 for group in self.groups if group.status is ProbeGroupStatus.UNRESOLVED)

    @property
    def detected(self) -> int:
        """Total detected rungs across every probe group."""
        return sum(group.detected for group in self.groups)

    def key(self) -> tuple[int, int, int, tuple[str, ...]]:
        """Return the declared total order over subsets."""
        return (-self.qualified, self.unresolved, -self.detected, self.workload_ids)


def oracle_collapse(
    cells: dict[tuple[str, str, int], CellOutcome],
    workload_ids: tuple[str, ...],
    probe_id: str,
    rungs: int,
) -> tuple[CellOutcome, ...]:
    """Collapse one ladder across a subset, independently of the production helper."""
    collapsed: list[CellOutcome] = []
    for index in range(rungs):
        observed = [cells[(workload, probe_id, index)] for workload in workload_ids]
        if CellOutcome.DETECTED in observed:
            collapsed.append(CellOutcome.DETECTED)
        elif CellOutcome.UNRESOLVED in observed:
            collapsed.append(CellOutcome.UNRESOLVED)
        else:
            collapsed.append(CellOutcome.NOT_DETECTED)
    return tuple(collapsed)


def oracle_floor(signature: tuple[CellOutcome, ...]) -> int | None:
    """Return the smallest index whose entire suffix is DETECTED, or None."""
    candidate: int | None = None
    for index in reversed(range(len(signature))):
        if signature[index] is not CellOutcome.DETECTED:
            break
        candidate = index
    return candidate


def oracle_group(probe_id: str, signature: tuple[CellOutcome, ...], required: int) -> OracleGroup:
    """Adjudicate one probe group from its collapsed signature."""
    floor = oracle_floor(signature)
    detected = sum(1 for outcome in signature if outcome is CellOutcome.DETECTED)
    if floor is not None and floor <= required:
        status = ProbeGroupStatus.QUALIFIED
    elif any(outcome is CellOutcome.UNRESOLVED for outcome in signature[required:]):
        status = ProbeGroupStatus.UNRESOLVED
    else:
        status = ProbeGroupStatus.INSUFFICIENT
    return OracleGroup(probe_id, status, signature, floor, detected)


def oracle_overall(groups: tuple[OracleGroup, ...]) -> QualificationStatus:
    """Map adjudicated groups to exactly one completed qualification status."""
    statuses = [group.status for group in groups]
    if any(status is ProbeGroupStatus.UNRESOLVED for status in statuses):
        return QualificationStatus.UNRESOLVED
    if all(status is ProbeGroupStatus.QUALIFIED for status in statuses):
        return QualificationStatus.QUALIFIED_FOR_DECLARED_PROBES
    if any(status is ProbeGroupStatus.QUALIFIED for status in statuses):
        return QualificationStatus.PARTIALLY_QUALIFIED
    return QualificationStatus.INSUFFICIENT_EXCITATION


def oracle_decide(
    workload_ids: tuple[str, ...],
    ladders: tuple[tuple[str, int, int], ...],
    cells: dict[tuple[str, str, int], CellOutcome],
) -> tuple[QualificationStatus, OracleSubset, int]:
    """Enumerate every three-workload subset and return the winner under the declared order."""
    best: OracleSubset | None = None
    evaluated = 0
    for combination in itertools.combinations(workload_ids, BUDGET):
        members = tuple(sorted(combination))
        groups = tuple(
            oracle_group(probe_id, oracle_collapse(cells, members, probe_id, rungs), required)
            for probe_id, rungs, required in ladders
        )
        subset = OracleSubset(members, groups)
        evaluated += 1
        if best is None or subset.key() < best.key():
            best = subset
    assert best is not None, "the generator must produce at least one admissible subset"
    return oracle_overall(best.groups), best, evaluated


# --- campaign generation -----------------------------------------------------------------------


def _token(index: int) -> ExactRational:
    """Return one strictly increasing exact magnitude token."""
    return ExactRational.from_decimal_token(f"0.{index + 1:03d}")


@dataclass(frozen=True)
class Campaign:
    """One generated pure campaign and everything needed to reproduce it."""

    index: int
    config: QualificationConfig
    workload_ids: tuple[str, ...]
    ladders: tuple[tuple[str, int, int], ...]
    cells: dict[tuple[str, str, int], CellOutcome]

    def describe(self) -> str:
        """Return a compact, complete reproduction of this campaign."""
        rows = [
            f"case={self.index} seed={SEED} workloads={list(self.workload_ids)}",
            f"ladders(probe_id, rungs, required_index)={list(self.ladders)}",
        ]
        for probe_id, rungs, _required in self.ladders:
            for workload in self.workload_ids:
                observed = "".join(
                    self.cells[(workload, probe_id, index)].value[0] for index in range(rungs)
                )
                rows.append(f"  {probe_id} {workload}: {observed}")
        return "\n".join(rows)


def _campaign(index: int, rng: random.Random) -> Campaign:
    """Generate one bounded pure campaign with no native or filesystem dependency."""
    workload_count = rng.randint(*WORKLOAD_RANGE)
    workload_ids = tuple(f"w{number}" for number in range(workload_count))
    group_count = rng.randint(*GROUP_RANGE)

    probe_groups: list[ProbeGroup] = []
    ladders: list[tuple[str, int, int]] = []
    cells: dict[tuple[str, str, int], CellOutcome] = {}
    for group_index in range(group_count):
        rungs = rng.randint(*RUNG_RANGE)
        probe_id = f"p{group_index}"
        required_index = rng.randrange(rungs)
        magnitudes = tuple(_token(rung) for rung in range(rungs))
        probe_groups.append(
            ProbeGroup(
                probe_id=probe_id,
                parameter=f"joint{group_index}.damping",
                direction="increase",
                magnitude_semantics="absolute source-model native-unit increase",
                required_detection_magnitude=magnitudes[required_index],
                variants=tuple(
                    ProbeVariant(
                        magnitude,
                        ModelRoleConfig(
                            f"probes/{probe_id}/r{rung}",
                            "model.xml",
                            ExactRational.from_decimal_token("0.001"),
                        ),
                    )
                    for rung, magnitude in enumerate(magnitudes)
                ),
            )
        )
        ladders.append((probe_id, rungs, required_index))
        for workload in workload_ids:
            for rung in range(rungs):
                cells[(workload, probe_id, rung)] = rng.choice(_OUTCOMES)

    config = QualificationConfig(
        schema_version=1,
        baseline=ModelRoleConfig(
            "baseline", "model.xml", ExactRational.from_decimal_token("0.001")
        ),
        probe_groups=tuple(probe_groups),
        workloads=tuple(
            WorkloadCandidate(
                workload,
                f"workloads/{workload}/state.npz",
                f"workloads/{workload}/actions.npz",
                ExactRational.from_decimal_token("0.01"),
            )
            for workload in workload_ids
        ),
        repeats=2,
        joint_tolerances={
            "joint": JointToleranceConfig(
                "hinge",
                {
                    "angle_rad": ExactRational.from_decimal_token("0.01"),
                    "angular_velocity_rad_s": ExactRational.from_decimal_token("0.05"),
                },
            )
        },
        aliases=None,
        budget=BUDGET,
        output_dir="qualification_out",
    )
    return Campaign(index, config, workload_ids, tuple(ladders), cells)


# --- the cross-check ---------------------------------------------------------------------------


def test_the_production_decision_matches_an_independent_oracle_over_5000_campaigns() -> None:
    """Run the seeded campaign sweep and require zero disagreements.

    Only ``decide`` is called on the production side. Every expected value below is computed by the
    oracle above, so a shared bug in a production adjudication helper cannot make this pass.
    """
    rng = random.Random(SEED)
    mismatches: list[str] = []
    statuses: dict[QualificationStatus, int] = {}

    for index in range(CAMPAIGNS):
        campaign = _campaign(index, rng)
        expected_status, expected_subset, expected_evaluated = oracle_decide(
            campaign.workload_ids, campaign.ladders, campaign.cells
        )
        actual = decide(campaign.config, campaign.cells, campaign.workload_ids)
        statuses[expected_status] = statuses.get(expected_status, 0) + 1

        observed = (
            actual.status,
            actual.selected.workload_ids,
            actual.subsets_evaluated,
            tuple(
                (group.probe_id, group.status, group.floor_index)
                for group in actual.selected.groups
            ),
            tuple(group.signature for group in actual.selected.groups),
        )
        wanted = (
            expected_status,
            expected_subset.workload_ids,
            expected_evaluated,
            tuple(
                (group.probe_id, group.status, group.floor_index)
                for group in expected_subset.groups
            ),
            tuple(group.signature for group in expected_subset.groups),
        )
        if observed != wanted:
            mismatches.append(f"{campaign.describe()}\n  expected={wanted}\n  actual={observed}")
            if len(mismatches) >= 3:
                break

    assert not mismatches, (
        f"the production decision disagreed with the independent oracle in "
        f"{len(mismatches)} campaign(s):\n\n" + "\n\n".join(mismatches)
    )
    check.equal(
        sum(statuses.values()),
        CAMPAIGNS,
        f"the sweep did not run all {CAMPAIGNS} campaigns",
    )
    for status in QualificationStatus:
        check.is_in(
            status,
            statuses,
            f"the seeded sweep never produced {status.value}, so that branch is unexercised",
        )
