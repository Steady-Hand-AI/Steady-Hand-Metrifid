"""Witnesses that explain the completed status, in one declared canonical order.

The previous implementation ranked a zero-change false alarm ahead of everything and admitted an
unresolved rung from anywhere in the ladder. A green run could therefore publish, as its single
first witness, an unresolved rung below the required magnitude, which explains nothing about the
green decision it was attached to.

The rule now follows the status:

    QUALIFIED_FOR_DECLARED_PROBES                first_witness is null
    UNRESOLVED                                   first_witness is the first unresolved witness
    PARTIALLY_QUALIFIED / INSUFFICIENT_EXCITATION first_witness is the first blind witness

A false alarm is an eligibility warning reported in its own field. It never replaces the
status-bearing witness, because excluding a workload does not explain the decision the remaining
workloads produced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from ..json_values import CanonicalValue
from ._config import ProbeGroup, QualificationConfig
from ._decision import SubsetAdjudication
from ._status import CellOutcome, ProbeGroupStatus, QualificationStatus

WITNESS_ORDER: Final[str] = (
    "probe group declaration order, then increasing exact magnitude, then workload_id "
    "Unicode-code-point lexical order"
)
BLIND_KIND: Final[str] = "BLIND_RUNG"
UNRESOLVED_KIND: Final[str] = "UNRESOLVED_RUNG"
FALSE_ALARM_KIND: Final[str] = "ZERO_CHANGE_FALSE_ALARM"


@dataclass(frozen=True, slots=True)
class Witness:
    """One concrete example of a blind rung, an unresolved rung, or a zero-change false alarm."""

    kind: str
    probe_id: str | None
    parameter: str | None
    magnitude: str | None
    variant_index: int | None
    workload_id: str | None
    detail: str

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the witness."""
        return {
            "kind": self.kind,
            "probe_id": self.probe_id,
            "parameter": self.parameter,
            "magnitude": self.magnitude,
            "variant_index": self.variant_index,
            "workload_id": self.workload_id,
            "detail": self.detail,
        }


def _contributors(
    matrix: Mapping[tuple[str, str, int], CellOutcome],
    workload_ids: Sequence[str],
    probe_id: str,
    index: int,
    outcome: CellOutcome,
) -> str | None:
    """Return the first selected workload showing this outcome, in code-point lexical order."""
    names = sorted(
        workload_id
        for workload_id in workload_ids
        if matrix.get((workload_id, probe_id, index)) is outcome
    )
    return names[0] if names else None


def _rung_witness(
    config: QualificationConfig,
    subset: SubsetAdjudication,
    matrix: Mapping[tuple[str, str, int], CellOutcome],
    *,
    outcome: CellOutcome,
    kind: str,
    detail: str,
    only_failing_groups: bool,
) -> Witness | None:
    """Return the first rung at or above a requirement that shows one outcome.

    Only rungs at or above the group's required magnitude can explain a completed status, so a rung
    below the requirement is visible in the matrix but never becomes the status-bearing witness.
    """
    by_probe: dict[str, ProbeGroup] = {g.probe_id: g for g in config.probe_groups}
    for adjudication in subset.groups:
        if only_failing_groups and adjudication.status is ProbeGroupStatus.QUALIFIED:
            continue
        group = by_probe[adjudication.probe_id]
        required = group.required_index()
        for index in range(required, len(adjudication.signature)):
            if adjudication.signature[index] is not outcome:
                continue
            return Witness(
                kind=kind,
                probe_id=group.probe_id,
                parameter=group.parameter,
                magnitude=group.variants[index].magnitude.to_decimal_token(),
                variant_index=index,
                workload_id=_contributors(
                    matrix, subset.workload_ids, group.probe_id, index, outcome
                ),
                detail=detail,
            )
    return None


def collect_witnesses(
    config: QualificationConfig,
    subset: SubsetAdjudication,
    status: QualificationStatus,
    matrix: Mapping[tuple[str, str, int], CellOutcome],
    excluded_workload_ids: Sequence[str],
    exclusion_reasons: Mapping[str, str],
) -> dict[str, CanonicalValue]:
    """Return every witness field, with the status-bearing one chosen by the completed status."""
    unresolved = _rung_witness(
        config,
        subset,
        matrix,
        outcome=CellOutcome.UNRESOLVED,
        kind=UNRESOLVED_KIND,
        detail=(
            "the comparison for this rung did not complete as a decision, so no qualifying floor "
            "can be established at or above the required magnitude"
        ),
        only_failing_groups=True,
    )
    blind = _rung_witness(
        config,
        subset,
        matrix,
        outcome=CellOutcome.NOT_DETECTED,
        kind=BLIND_KIND,
        detail=(
            "no selected workload produced a material behavior change at this rung, so no "
            "unbroken detected suffix reaches the required magnitude"
        ),
        only_failing_groups=True,
    )

    false_alarm: Witness | None = None
    for workload_id in sorted(excluded_workload_ids):
        false_alarm = Witness(
            kind=FALSE_ALARM_KIND,
            probe_id=None,
            parameter=None,
            magnitude=None,
            variant_index=None,
            workload_id=workload_id,
            detail=exclusion_reasons.get(
                workload_id, "this workload's zero-change control did not complete as no change"
            ),
        )
        break

    if status is QualificationStatus.QUALIFIED_FOR_DECLARED_PROBES:
        first = None
    elif status is QualificationStatus.UNRESOLVED:
        first = unresolved
    else:
        first = blind

    return {
        "first_witness": None if first is None else first.to_primitive(),
        "first_blind_witness": None if blind is None else blind.to_primitive(),
        "first_unresolved_witness": None if unresolved is None else unresolved.to_primitive(),
        "first_false_alarm_witness": None if false_alarm is None else false_alarm.to_primitive(),
        "witness_order": WITNESS_ORDER,
        "status_witness_rule": (
            "QUALIFIED_FOR_DECLARED_PROBES publishes no first witness; UNRESOLVED publishes the "
            "first unresolved witness; PARTIALLY_QUALIFIED and INSUFFICIENT_EXCITATION publish the "
            "first blind witness. A zero-change false alarm is an eligibility warning reported "
            "separately and never becomes the status-bearing witness."
        ),
    }
