"""Recompute every derived claim in a published receipt from its own source data.

This is what makes a resealed receipt useless. The aggregate carries the admitted configuration and
one record per completed comparison; those are the facts. Everything else in the document — the
outcome of each cell, which workloads were eligible, which subsets existed, which one won, each
group's signature and floor, the completed status and exit code, the counts, the witnesses — is
derived, and is recomputed here and compared.

A mismatch rejects the receipt even when its self-hash was recomputed correctly, because a hash over
contradictory content is still contradictory content.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping

from ..errors import ComparisonStatus
from ..json_values import CanonicalValue, compute_self_hash
from ._aggregate import AggregateReceipt
from ._config import QualificationConfig
from ._decision import adjudicate_subset, decide
from ._evidence import detection_outcome, exclusion_reason, planned_comparisons
from ._status import (
    QUALIFICATION_LIMITATIONS,
    CellOutcome,
    QualificationStatus,
    qualification_exit_code,
)
from ._witness import collect_witnesses


class ReconstructionError(ValueError):
    """Raised when a published receipt disagrees with what its own evidence implies."""


def _require(condition: bool, message: str) -> None:
    """Reject the receipt when one recomputed claim does not match the published one."""
    if not condition:
        raise ReconstructionError(message)


def _expected_control_keys(config: QualificationConfig) -> list[tuple[int, str]]:
    """Return the exact zero-change control key set, in canonical order."""
    return [(index, w.workload_id) for index, w in enumerate(config.workloads)]


def _expected_cell_keys(config: QualificationConfig) -> list[tuple[int, int, int]]:
    """Return the complete workload x group x rung key set, in canonical order."""
    return [
        (workload_index, group_index, variant_index)
        for workload_index in range(len(config.workloads))
        for group_index, group in enumerate(config.probe_groups)
        for variant_index in range(len(group.variants))
    ]


def reconstruct(receipt: AggregateReceipt) -> QualificationConfig:
    """Recompute the whole decision from the receipt's own configuration and cell records."""
    try:
        config = QualificationConfig.from_primitive(receipt.configuration)
    except (TypeError, ValueError) as exc:
        raise ReconstructionError(
            f"the embedded qualification configuration is not admissible: {exc}"
        ) from exc

    _require(
        receipt.planned_comparisons == planned_comparisons(config),
        "planned_comparisons does not match the admitted configuration",
    )

    published_controls = [(c.workload_index, c.workload_id) for c in receipt.zero_change_controls]
    _require(
        published_controls == _expected_control_keys(config),
        "the zero-change control key set or its order does not match the configuration",
    )
    published_cells = [
        (c.workload_index, c.group_index, c.variant_index) for c in receipt.probe_cells
    ]
    _require(
        published_cells == _expected_cell_keys(config),
        "the probe cell key set or its order is not the complete Cartesian product in canonical "
        "order",
    )

    matrix: dict[tuple[str, str, int], CellOutcome] = {}
    for cell in receipt.probe_cells:
        group = config.probe_groups[cell.group_index]
        _require(cell.probe_id == group.probe_id, "a probe cell names the wrong probe group")
        _require(
            cell.workload_id == config.workloads[cell.workload_index].workload_id,
            "a probe cell names the wrong workload",
        )
        _require(
            cell.magnitude == group.variants[cell.variant_index].magnitude.to_decimal_token(),
            "a probe cell records a magnitude the configuration does not declare for that rung",
        )
        recomputed = detection_outcome(ComparisonStatus(cell.comparison_status))
        _require(
            cell.outcome == recomputed.value,
            "a probe cell's outcome is not the mapping of its own comparison status",
        )
        matrix[(cell.workload_id, cell.probe_id, cell.variant_index)] = recomputed

    eligible: list[str] = []
    excluded: list[str] = []
    reasons: dict[str, str] = {}
    for control in receipt.zero_change_controls:
        status = ComparisonStatus(control.comparison_status)
        _require(
            control.outcome == detection_outcome(status).value,
            "a zero-change control's outcome is not the mapping of its own comparison status",
        )
        is_eligible = status is ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD
        _require(
            control.eligible == is_eligible,
            "a zero-change control's eligibility does not follow from its comparison status",
        )
        if is_eligible:
            _require(
                control.exclusion_reason is None,
                "an eligible workload must not carry an exclusion reason",
            )
            eligible.append(control.workload_id)
        else:
            _require(
                control.exclusion_reason == exclusion_reason(status),
                "an excluded workload's reason is not the exact reason its status implies",
            )
            excluded.append(control.workload_id)
            reasons[control.workload_id] = control.exclusion_reason or ""

    _require(
        list(receipt.eligible_workload_ids) == eligible,
        "the eligible workload list does not follow from the zero-change controls",
    )
    _require(
        list(receipt.excluded_workload_ids) == excluded,
        "the excluded workload list does not follow from the zero-change controls",
    )
    _require(
        len(eligible) >= config.budget,
        "a completed receipt cannot exist with fewer eligible workloads than the budget",
    )

    _reconstruct_counts(receipt)
    decision = decide(config, matrix, eligible)
    _reconstruct_selection(receipt, config, matrix, decision)
    _reconstruct_witnesses(receipt, config, matrix, decision, excluded, reasons)
    _reconstruct_registries(receipt)
    _reconstruct_self_hash(receipt)
    return config


def _reconstruct_counts(receipt: AggregateReceipt) -> None:
    """Recompute every execution count from the published records."""
    completed = sum(1 for c in receipt.probe_cells if c.outcome != CellOutcome.UNRESOLVED.value)
    unresolved = sum(1 for c in receipt.probe_cells if c.outcome == CellOutcome.UNRESOLVED.value)
    expected: Mapping[str, CanonicalValue] = {
        "zero_change_comparisons": len(receipt.zero_change_controls),
        "probe_comparisons": len(receipt.probe_cells),
        "total_comparisons": len(receipt.zero_change_controls) + len(receipt.probe_cells),
        "completed_cells": completed,
        "unresolved_cells": unresolved,
        "failed_cells": 0,
    }
    _require(
        dict(receipt.execution_counts) == dict(expected),
        "the execution counts do not follow from the published control and cell records",
    )


def _reconstruct_selection(
    receipt: AggregateReceipt,
    config: QualificationConfig,
    matrix: Mapping[tuple[str, str, int], CellOutcome],
    decision: object,
) -> None:
    """Recompute the subset search, the winner, and every per-group adjudication."""
    expected_subsets = len(
        list(itertools.combinations(receipt.eligible_workload_ids, config.budget))
    )
    _require(
        receipt.subsets_evaluated == expected_subsets,
        "subsets_evaluated is not the number of three-workload subsets of the eligible set",
    )
    selected = tuple(receipt.selected_workload_ids)
    _require(
        selected == decision.selected.workload_ids,  # type: ignore[attr-defined]
        "the selected subset is not the winner under the frozen total order",
    )
    _require(
        tuple(receipt.selection.workload_ids) == selected,
        "the selection block names a different subset from selected_workload_ids",
    )

    recomputed = adjudicate_subset(config, matrix, selected)
    _require(
        receipt.selection.qualified_groups == recomputed.qualified
        and receipt.selection.unresolved_groups == recomputed.unresolved
        and receipt.selection.detected_variants == recomputed.detected_variants,
        "the selection summary does not match the adjudication of the selected subset",
    )
    _require(
        len(receipt.selection.groups) == len(recomputed.groups),
        "the selection reports a different number of probe groups from the configuration",
    )
    for published, expected in zip(receipt.selection.groups, recomputed.groups, strict=True):
        _require(published.probe_id == expected.probe_id, "a group adjudication is out of order")
        _require(
            published.status == expected.status.value,
            f"the published status for probe group {expected.probe_id!r} is not the status its "
            "own detection signature implies",
        )
        _require(
            tuple(published.detection_signature) == tuple(o.value for o in expected.signature),
            f"the detection signature for {expected.probe_id!r} does not follow from the cells",
        )
        expected_floor = (
            None
            if expected.floor_magnitude is None
            else expected.floor_magnitude.to_decimal_token()
        )
        _require(
            published.floor_magnitude == expected_floor
            and published.floor_variant_index == expected.floor_index,
            f"the detection floor for {expected.probe_id!r} is not the suffix floor of its "
            "signature",
        )
        _require(
            published.no_floor_reason == expected.no_floor_reason,
            f"the no-floor reason for {expected.probe_id!r} does not match its adjudication",
        )
        _require(
            published.detected_variants == expected.detected_variants,
            f"the detected-variant count for {expected.probe_id!r} does not match its signature",
        )

    status = QualificationStatus(receipt.status)
    _require(
        status is decision.status,  # type: ignore[attr-defined]
        "the completed status does not follow from the per-group adjudications",
    )
    _require(
        receipt.completed_exit_code == int(qualification_exit_code(status)),
        "the completed exit code is not the frozen code for the published status",
    )
    published_ranking = [dict(item) for item in receipt.subset_ranking]
    expected_ranking = [dict(item) for item in decision.ranking]  # type: ignore[attr-defined]
    _require(
        published_ranking == expected_ranking,
        "the recorded subset ranking is not the ranking the evidence produces",
    )


def _reconstruct_witnesses(
    receipt: AggregateReceipt,
    config: QualificationConfig,
    matrix: Mapping[tuple[str, str, int], CellOutcome],
    decision: object,
    excluded: list[str],
    reasons: Mapping[str, str],
) -> None:
    """Recompute every witness field and the declared ordering text."""
    expected = collect_witnesses(
        config,
        decision.selected,  # type: ignore[attr-defined]
        decision.status,  # type: ignore[attr-defined]
        matrix,
        excluded,
        reasons,
    )
    _require(
        dict(receipt.witnesses) == dict(expected),
        "the published witnesses do not follow from the evidence and the completed status",
    )


def _reconstruct_registries(receipt: AggregateReceipt) -> None:
    """Check the closed limitation registry and the not-claimed block."""
    _require(
        list(receipt.limitations) == [code.value for code in QUALIFICATION_LIMITATIONS],
        "limitations must be the complete registry in canonical order",
    )
    _require(bool(receipt.not_claimed), "the not-claimed block must not be empty")


def _reconstruct_self_hash(receipt: AggregateReceipt) -> None:
    """Confirm the self-hash matches the canonical content it is computed over.

    This is a corruption check, not authentication. It is verified last and on its own so nothing
    above can be read as depending on it.
    """
    # The typed model does not round-trip to the published mapping, so the caller checks the hash
    # against the raw document; this function exists to state the boundary in one place.
    _require(len(receipt.receipt_sha256) == 64, "receipt_sha256 must be a SHA-256")


def verify_self_hash(document: Mapping[str, CanonicalValue]) -> None:
    """Recompute the aggregate self-hash over the canonical primitive it is defined over."""
    published = document.get("receipt_sha256")
    working = dict(document)
    working["receipt_sha256"] = None
    if compute_self_hash(working, "receipt_sha256") != published:
        raise ReconstructionError(
            "receipt_sha256 does not match the canonical content it is computed over"
        )
