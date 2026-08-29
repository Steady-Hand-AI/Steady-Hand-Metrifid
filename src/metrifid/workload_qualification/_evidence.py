"""Execute every comparison cell and retain a typed, self-checked record of each one.

The previous implementation ran the comparisons, then reopened a couple of the resulting receipt
files with ``json.loads`` and sampled the first workload for the campaign's identities. Anything a
later cell disagreed about was invisible. This module keeps the typed ``ComparisonReceipt`` that
``compare_configuration_file`` already returns, validates it immediately, and binds it to the exact
bytes retained on disk before the cell is accepted.

Storage names are ordinals, never semantic labels. ``workload_id`` and ``probe_id`` stay report
fields; they never reach the filesystem.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final

from .._configuration_schemas import ModelRoleConfig
from .._json_admission import RECEIPT_JSON_LIMITS
from ..compare import ComparisonOperationError
from ..compare._orchestrator import _compare_into_owned_output
from ..errors import ComparisonStatus
from ..json_values import CanonicalValue, ExactRational, canonical_json_bytes
from ..schemas import ComparisonReceipt, validate_receipt
from ._config import ProbeGroup, QualificationConfig, WorkloadCandidate
from ._owned_output import OwnedOutputRoot
from ._paths import (
    COMPARISON_CONFIG_NAME,
    COMPARISON_OUTPUT_NAME,
    COMPARISON_RECEIPT_NAME,
    control_locator,
    probe_locator,
)
from ._status import CellOutcome

ZERO_CHANGE_KIND: Final[str] = "ZERO_CHANGE"
PROBE_KIND: Final[str] = "PROBE"

_OUTCOMES: Final[Mapping[ComparisonStatus, CellOutcome]] = MappingProxyType(
    {
        ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD: CellOutcome.NOT_DETECTED,
        ComparisonStatus.MATERIAL_BEHAVIOR_CHANGE: CellOutcome.DETECTED,
        ComparisonStatus.COVERAGE_INSUFFICIENT: CellOutcome.UNRESOLVED,
        ComparisonStatus.NONDETERMINISTIC_REPLAY: CellOutcome.UNRESOLVED,
    }
)

ZERO_CHANGE_PASS: Final[ComparisonStatus] = (
    ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD
)


class CellBindingError(ValueError):
    """Raised when a completed comparison does not bind to its own retained evidence."""


def detection_outcome(status: ComparisonStatus) -> CellOutcome:
    """Map one completed comparison status onto its detection outcome."""
    if not isinstance(status, ComparisonStatus):
        raise TypeError("status must be a ComparisonStatus")
    return _OUTCOMES[status]


def raw_digest(payload: bytes) -> str:
    """Return the SHA-256 of exactly these bytes."""
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CellRecord:
    """One completed comparison, bound to the bytes that produced and retained it."""

    kind: str
    workload_index: int
    workload_id: str
    group_index: int | None
    probe_id: str | None
    variant_index: int | None
    magnitude: ExactRational | None
    config_locator: PurePosixPath
    config_raw_sha256: str
    receipt_locator: PurePosixPath
    receipt_raw_sha256: str
    receipt: ComparisonReceipt
    outcome: CellOutcome

    @property
    def status(self) -> ComparisonStatus:
        """Return the status carried by the typed receipt itself, not a copied string."""
        return self.receipt.status

    @property
    def eligible(self) -> bool:
        """Return whether a zero-change control admits its workload to selection."""
        return self.status is ZERO_CHANGE_PASS

    def key(self) -> tuple[int, int, int]:
        """Return the canonical ordering key for this cell."""
        return (
            self.workload_index,
            -1 if self.group_index is None else self.group_index,
            -1 if self.variant_index is None else self.variant_index,
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the record, with semantic labels as data and locators as relative paths."""
        primitive: dict[str, CanonicalValue] = {
            "kind": self.kind,
            "workload_index": self.workload_index,
            "workload_id": self.workload_id,
            "comparison_status": self.status.value,
            "outcome": self.outcome.value,
            "comparison_config_locator": self.config_locator.as_posix(),
            "comparison_config_raw_sha256": self.config_raw_sha256,
            "comparison_receipt_locator": self.receipt_locator.as_posix(),
            "comparison_receipt_raw_sha256": self.receipt_raw_sha256,
            "comparison_receipt_sha256": self.receipt.receipt_sha256,
        }
        if self.kind == ZERO_CHANGE_KIND:
            primitive["eligible"] = self.eligible
            primitive["exclusion_reason"] = None if self.eligible else exclusion_reason(self.status)
        else:
            primitive["group_index"] = self.group_index
            primitive["probe_id"] = self.probe_id
            primitive["variant_index"] = self.variant_index
            primitive["magnitude"] = (
                None if self.magnitude is None else self.magnitude.to_decimal_token()
            )
        return primitive


def exclusion_reason(status: ComparisonStatus) -> str:
    """Return the exact reason one workload is excluded by its zero-change control."""
    return (
        "the baseline model compared against itself under this workload returned "
        f"{status.value}, so this workload cannot serve as detection evidence"
    )


def comparison_primitive(
    config: QualificationConfig,
    base: Path,
    candidate: ModelRoleConfig,
    workload: WorkloadCandidate,
    output_dir: Path,
) -> dict[str, CanonicalValue]:
    """Build one strict comparison configuration for a single cell.

    Source locations are absolutized against the qualification file's directory. The generated file
    lives deep inside the owned output tree, so a relative spelling would have to traverse upward out
    of it; absolutizing here keeps the generated configuration unambiguous while the user-facing
    relative paths stay exactly as declared in the aggregate receipt.
    """
    baseline = config.baseline
    return {
        "schema_version": 1,
        "baseline": {
            "model_root": str((base / baseline.model_root).absolute()),
            "entrypoint": baseline.entrypoint,
            "declared_step_dt": baseline.declared_step_dt.to_decimal_token(),
        },
        "candidate": {
            "model_root": str((base / candidate.model_root).absolute()),
            "entrypoint": candidate.entrypoint,
            "declared_step_dt": candidate.declared_step_dt.to_decimal_token(),
        },
        "initial_state": str((base / workload.initial_state).absolute()),
        "actions": str((base / workload.actions).absolute()),
        "control_dt": workload.control_dt.to_decimal_token(),
        "repeats": config.repeats,
        "joint_tolerances": {
            name: tolerance.to_primitive() for name, tolerance in config.joint_tolerances.items()
        },
        "aliases": None if config.aliases is None else str((base / config.aliases).absolute()),
        "output_dir": str((output_dir / COMPARISON_OUTPUT_NAME).absolute()),
    }


def _bind_cell(
    owned: OwnedOutputRoot,
    cell: PurePosixPath,
    cell_directory: Path,
    config_bytes: bytes,
    result_receipt: ComparisonReceipt,
    comparison_json: Path,
) -> tuple[str, str]:
    """Prove the completed comparison is bound to the exact retained bytes.

    Every check here answers a question the previous implementation never asked: were these the bytes
    compare read, is this the receipt compare returned, and is the file on disk the same object?
    """
    config_sha = raw_digest(config_bytes)
    if config_sha != result_receipt.inputs.configuration_raw_sha256:
        raise CellBindingError(
            "the generated comparison configuration digest does not match the digest the "
            "comparison receipt records for the configuration it read"
        )
    validate_receipt(result_receipt)

    for label, path in (
        ("comparison json", comparison_json),
        ("comparison markdown", comparison_json.parent / "comparison.md"),
    ):
        try:
            path.absolute().relative_to(cell_directory)
        except ValueError as exc:
            raise CellBindingError(
                f"the returned {label} path is outside this cell's ordinal directory"
            ) from exc

    receipt_locator = cell / COMPARISON_OUTPUT_NAME / COMPARISON_RECEIPT_NAME
    retained = owned.read_bytes(receipt_locator, RECEIPT_JSON_LIMITS.max_bytes)
    retained_sha = raw_digest(retained)
    from ..json_values import strict_json_loads

    retained_receipt = ComparisonReceipt.from_primitive(strict_json_loads(retained))
    if retained_receipt != result_receipt:
        raise CellBindingError(
            "the retained comparison receipt on disk is not the receipt the comparison returned"
        )
    if retained_receipt.receipt_sha256 != result_receipt.receipt_sha256:  # pragma: no cover
        raise CellBindingError("the retained comparison receipt self-hash disagrees")
    return config_sha, retained_sha


def run_cell(
    owned: OwnedOutputRoot,
    config: QualificationConfig,
    base: Path,
    candidate: ModelRoleConfig,
    workload: WorkloadCandidate,
    cell: PurePosixPath,
) -> tuple[ComparisonReceipt, str, str, PurePosixPath, PurePosixPath]:
    """Execute one comparison into its ordinal cell directory and bind the result to its bytes."""
    output_locator = cell / COMPARISON_OUTPUT_NAME
    owned.make_directory(output_locator)
    # Display metadata only. The comparison writes through the retained descriptor below, so this
    # pathname is what the generated configuration and the receipt record, never the write target.
    cell_directory = owned.display_path(cell)
    primitive = comparison_primitive(config, base, candidate, workload, cell_directory)
    config_bytes = canonical_json_bytes(primitive) + b"\n"
    config_locator = cell / COMPARISON_CONFIG_NAME
    owned.write_bytes(config_locator, config_bytes)

    output_fd = owned.open_owned_directory(output_locator)
    try:
        result = _compare_into_owned_output(
            config_path=cell_directory / COMPARISON_CONFIG_NAME,
            config_raw=config_bytes,
            output_directory_fd=output_fd,
            output_display_path=owned.display_path(output_locator),
        )
    finally:
        os.close(output_fd)
    config_sha, receipt_sha = _bind_cell(
        owned, cell, cell_directory, config_bytes, result.receipt, result.comparison_json
    )
    receipt_locator = output_locator / COMPARISON_RECEIPT_NAME
    return result.receipt, config_sha, receipt_sha, config_locator, receipt_locator


@dataclass(frozen=True, slots=True)
class CampaignLedger:
    """Every completed cell of one campaign, in canonical order."""

    controls: tuple[CellRecord, ...]
    cells: tuple[CellRecord, ...]

    @property
    def eligible_workload_ids(self) -> tuple[str, ...]:
        """Workload identities admitted to selection, in declared order."""
        return tuple(c.workload_id for c in self.controls if c.eligible)

    @property
    def excluded_workload_ids(self) -> tuple[str, ...]:
        """Workload identities excluded by a failed zero-change control."""
        return tuple(c.workload_id for c in self.controls if not c.eligible)

    def outcome_map(self) -> dict[tuple[str, str, int], CellOutcome]:
        """Return the detection matrix keyed by workload, probe, and rung."""
        return {
            (c.workload_id, str(c.probe_id), int(c.variant_index or 0)): c.outcome
            for c in self.cells
        }

    def counts(self) -> dict[str, CanonicalValue]:
        """Return the actual execution counts this campaign performed."""
        completed = sum(1 for c in self.cells if c.outcome is not CellOutcome.UNRESOLVED)
        unresolved = sum(1 for c in self.cells if c.outcome is CellOutcome.UNRESOLVED)
        return {
            "zero_change_comparisons": len(self.controls),
            "probe_comparisons": len(self.cells),
            "total_comparisons": len(self.controls) + len(self.cells),
            "completed_cells": completed,
            "unresolved_cells": unresolved,
            "failed_cells": 0,
        }


def planned_comparisons(config: QualificationConfig) -> int:
    """Return how many comparisons this configuration will run before it runs any.

    One zero-change control per workload, plus one comparison per workload per declared rung. Under
    schema version 1 this is at most 16 + 16 x (16 x 8) = 2064.
    """
    rungs = sum(len(group.variants) for group in config.probe_groups)
    return len(config.workloads) + len(config.workloads) * rungs


def build_campaign_ledger(
    owned: OwnedOutputRoot, config: QualificationConfig, configuration_path: Path
) -> CampaignLedger:
    """Run every control and every probe cell, retaining a bound typed record for each."""
    base = configuration_path.absolute().parent

    controls: list[CellRecord] = []
    for workload_index, workload in enumerate(config.workloads):
        locator = control_locator(workload_index)
        receipt, config_sha, receipt_sha, config_locator, receipt_locator = run_cell(
            owned, config, base, config.baseline, workload, locator
        )
        controls.append(
            CellRecord(
                kind=ZERO_CHANGE_KIND,
                workload_index=workload_index,
                workload_id=workload.workload_id,
                group_index=None,
                probe_id=None,
                variant_index=None,
                magnitude=None,
                config_locator=config_locator,
                config_raw_sha256=config_sha,
                receipt_locator=receipt_locator,
                receipt_raw_sha256=receipt_sha,
                receipt=receipt,
                outcome=detection_outcome(receipt.status),
            )
        )

    cells: list[CellRecord] = []
    for workload_index, workload in enumerate(config.workloads):
        for group_index, group in enumerate(config.probe_groups):
            for variant_index, variant in enumerate(group.variants):
                locator = probe_locator(workload_index, group_index, variant_index)
                receipt, config_sha, receipt_sha, config_locator, receipt_locator = run_cell(
                    owned, config, base, variant.candidate, workload, locator
                )
                cells.append(
                    CellRecord(
                        kind=PROBE_KIND,
                        workload_index=workload_index,
                        workload_id=workload.workload_id,
                        group_index=group_index,
                        probe_id=group.probe_id,
                        variant_index=variant_index,
                        magnitude=variant.magnitude,
                        config_locator=config_locator,
                        config_raw_sha256=config_sha,
                        receipt_locator=receipt_locator,
                        receipt_raw_sha256=receipt_sha,
                        receipt=receipt,
                        outcome=detection_outcome(receipt.status),
                    )
                )
    return CampaignLedger(controls=tuple(controls), cells=tuple(cells))


__all__ = [
    "PROBE_KIND",
    "ZERO_CHANGE_KIND",
    "ZERO_CHANGE_PASS",
    "CampaignLedger",
    "CellBindingError",
    "CellRecord",
    "ComparisonOperationError",
    "ProbeGroup",
    "build_campaign_ledger",
    "detection_outcome",
    "exclusion_reason",
    "planned_comparisons",
    "raw_digest",
]
