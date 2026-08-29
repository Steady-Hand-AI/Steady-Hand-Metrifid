"""Completed cells must describe one campaign, checked one mutated invariant at a time.

Each case starts from a real published campaign, rebuilds its typed ledger from the retained
receipts, changes exactly one thing a coherent campaign cannot disagree about, and requires the
cross-cell validation to refuse before any subset could be selected.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path, PurePosixPath

import numpy as np
import pytest
import pytest_check as check

from metrifid import write_state_artifact
from metrifid.json_values import strict_json_loads
from metrifid.schemas import ComparisonReceipt
from metrifid.workload_qualification._campaign import (
    CampaignInvariantError,
    validate_campaign,
    verify_retained_evidence,
)
from metrifid.workload_qualification._config import QualificationConfig
from metrifid.workload_qualification._evidence import (
    PROBE_KIND,
    ZERO_CHANGE_KIND,
    CampaignLedger,
    CellRecord,
    detection_outcome,
    raw_digest,
    run_cell,
)
from metrifid.workload_qualification._owned_output import OwnedOutputRoot
from metrifid.workload_qualification._paths import admit_locator

from .conftest import RECEIPT_RELATIVE


def _load(case: Path) -> tuple[QualificationConfig, CampaignLedger, Path]:
    """Rebuild the typed ledger of a published campaign from its retained evidence."""
    root = (case / "qualification_out").resolve()
    aggregate = json.loads((case / RECEIPT_RELATIVE).read_text(encoding="utf-8"))
    config = QualificationConfig.from_primitive(aggregate["configuration"])

    def record(entry: dict, kind: str) -> CellRecord:
        receipt_path = root / entry["comparison_receipt_locator"]
        config_path = root / entry["comparison_config_locator"]
        receipt = ComparisonReceipt.from_primitive(strict_json_loads(receipt_path.read_bytes()))
        return CellRecord(
            kind=kind,
            workload_index=entry["workload_index"],
            workload_id=entry["workload_id"],
            group_index=entry.get("group_index"),
            probe_id=entry.get("probe_id"),
            variant_index=entry.get("variant_index"),
            magnitude=None,
            config_locator=admit_locator(entry["comparison_config_locator"], "config"),
            config_raw_sha256=raw_digest(config_path.read_bytes()),
            receipt_locator=admit_locator(entry["comparison_receipt_locator"], "receipt"),
            receipt_raw_sha256=raw_digest(receipt_path.read_bytes()),
            receipt=receipt,
            outcome=detection_outcome(receipt.status),
        )

    ledger = CampaignLedger(
        controls=tuple(record(e, ZERO_CHANGE_KIND) for e in aggregate["zero_change_controls"]),
        cells=tuple(record(e, PROBE_KIND) for e in aggregate["probe_cells"]),
    )
    return config, ledger, root


def _spliced(ledger: CampaignLedger, which: str, index: int, donor: CellRecord) -> CampaignLedger:
    """Return the ledger with one cell's typed receipt replaced by another valid receipt.

    Editing a field inside one comparison receipt is not a constructible attack: the comparison
    receipt schema already cross-validates its own inputs against its embedded closures and refuses.
    The real cross-cell threat is a receipt that is perfectly valid on its own and describes a
    different campaign, so that is what is spliced in here.
    """
    records = list(getattr(ledger, which))
    original = records[index]
    records[index] = CellRecord(
        kind=original.kind,
        workload_index=original.workload_index,
        workload_id=original.workload_id,
        group_index=original.group_index,
        probe_id=original.probe_id,
        variant_index=original.variant_index,
        magnitude=original.magnitude,
        config_locator=original.config_locator,
        config_raw_sha256=original.config_raw_sha256,
        receipt_locator=original.receipt_locator,
        receipt_raw_sha256=original.receipt_raw_sha256,
        receipt=donor.receipt,
        outcome=detection_outcome(donor.receipt.status),
    )
    if which == "controls":
        return CampaignLedger(controls=tuple(records), cells=ledger.cells)
    return CampaignLedger(controls=ledger.controls, cells=tuple(records))


def _run_probe_receipt(
    case: Path,
    root: Path,
    config: QualificationConfig,
    label: str,
) -> ComparisonReceipt:
    """Run one genuine probe receipt under a deliberately different campaign contract."""
    owned = OwnedOutputRoot.bind_existing(root)
    try:
        receipt, _config_sha, _receipt_sha, _config_locator, _receipt_locator = run_cell(
            owned,
            config,
            case.resolve(),
            config.probe_groups[0].variants[0].candidate,
            config.workloads[0],
            PurePosixPath("campaign_invariant_cells") / label,
        )
        return receipt
    finally:
        owned.close()


def test_a_coherent_campaign_is_accepted(case_copy: Path) -> None:
    """A positive control: validation that refused everything would prove nothing."""
    config, ledger, _root = _load(case_copy)
    identity = validate_campaign(config, ledger)
    check.equal(
        len(identity.baseline_model_closure_sha256),
        64,
        "the accepted campaign's identity names a baseline model closure that is not a SHA-256 "
        "digest, so the thing every cell was said to agree on is not a usable model identity",
    )


def test_a_cell_from_a_different_baseline_campaign_refuses(
    case_copy: Path, other_case: Path
) -> None:
    """A later cell that used a different baseline model is caught before selection."""
    config, ledger, _root = _load(case_copy)
    _other_config, other_ledger, _other_root = _load(other_case)
    with pytest.raises(CampaignInvariantError):
        validate_campaign(config, _spliced(ledger, "cells", -1, other_ledger.cells[-1]))


def test_a_control_from_a_different_baseline_campaign_refuses(
    case_copy: Path, other_case: Path
) -> None:
    """The same holds for a zero-change control."""
    config, ledger, _root = _load(case_copy)
    _other_config, other_ledger, _other_root = _load(other_case)
    with pytest.raises(CampaignInvariantError):
        validate_campaign(config, _spliced(ledger, "controls", 0, other_ledger.controls[0]))


def test_a_cell_carrying_another_workloads_artifacts_refuses(case_copy: Path) -> None:
    """One workload's cells must all describe that workload's own state and action artifacts."""
    config, ledger, _root = _load(case_copy)
    donor = next(
        c
        for c in ledger.cells
        if c.workload_index == 1 and c.group_index == 0 and c.variant_index == 0
    )
    target = next(
        index
        for index, c in enumerate(ledger.cells)
        if c.workload_index == 0 and c.group_index == 0 and c.variant_index == 0
    )
    with pytest.raises(CampaignInvariantError, match="actions_raw_sha256"):
        validate_campaign(config, _spliced(ledger, "cells", target, donor))


def test_a_probe_receipt_with_another_initial_state_identity_refuses(case_copy: Path) -> None:
    """A genuine same-rung receipt cannot carry a different valid state for the workload."""
    config, ledger, root = _load(case_copy)
    state_path = case_copy / config.workloads[0].initial_state
    state_path.unlink()
    write_state_artifact(
        state_path,
        joint_names=("shoulder",),
        qpos_offsets=(0, 1),
        qpos=np.array([0.1], dtype=np.float64),
        qvel_offsets=(0, 1),
        qvel=np.zeros(1),
        actuator_names=("shoulder_motor",),
        act_offsets=(0, 0),
        act=np.empty(0),
    )
    receipt = _run_probe_receipt(case_copy, root, config, "initial_state")
    donor = replace(ledger.cells[0], receipt=receipt, outcome=detection_outcome(receipt.status))
    with pytest.raises(CampaignInvariantError, match="initial_state_raw_sha256"):
        validate_campaign(config, _spliced(ledger, "cells", 0, donor))


def test_a_probe_rung_compared_against_a_different_closure_across_workloads_refuses(
    case_copy: Path,
) -> None:
    """One rung must name one candidate closure for every workload that ran it."""
    config, ledger, _root = _load(case_copy)
    donor = next(c for c in ledger.cells if c.workload_index == 0 and c.variant_index == 1)
    target = next(
        index
        for index, c in enumerate(ledger.cells)
        if c.workload_index == 1 and c.variant_index == 0
    )
    with pytest.raises(CampaignInvariantError):
        validate_campaign(config, _spliced(ledger, "cells", target, donor))


def test_a_zero_change_control_that_is_not_baseline_against_itself_refuses(
    case_copy: Path,
) -> None:
    """A control comparing something other than the baseline against itself is not a control."""
    config, ledger, _root = _load(case_copy)
    probe = ledger.cells[0]
    with pytest.raises(CampaignInvariantError):
        validate_campaign(config, _spliced(ledger, "controls", 0, probe))


def test_a_probe_receipt_with_the_wrong_repeat_count_refuses(case_copy: Path) -> None:
    """Every cell's genuine typed contract must carry the admitted campaign repeat count."""
    config, ledger, root = _load(case_copy)
    altered = replace(config, repeats=config.repeats + 1)
    receipt = _run_probe_receipt(case_copy, root, altered, "repeats")
    donor = replace(ledger.cells[0], receipt=receipt, outcome=detection_outcome(receipt.status))
    with pytest.raises(CampaignInvariantError, match="repeat count"):
        validate_campaign(config, _spliced(ledger, "cells", 0, donor))


def test_a_probe_receipt_with_the_wrong_monitored_contract_refuses(case_copy: Path) -> None:
    """Joint names, types, metrics, and exact tolerance values are campaign-wide."""
    config, ledger, root = _load(case_copy)
    primitive = config.to_primitive()
    tolerances = primitive["joint_tolerances"]
    assert isinstance(tolerances, dict)
    shoulder = tolerances["shoulder"]
    assert isinstance(shoulder, dict)
    shoulder["angle_rad"] = "0.02"
    altered = QualificationConfig.from_primitive(primitive)
    receipt = _run_probe_receipt(case_copy, root, altered, "monitoring")
    donor = replace(ledger.cells[0], receipt=receipt, outcome=detection_outcome(receipt.status))
    with pytest.raises(CampaignInvariantError, match="monitored joint"):
        validate_campaign(config, _spliced(ledger, "cells", 0, donor))


def test_a_probe_receipt_with_the_wrong_control_period_refuses(case_copy: Path) -> None:
    """Every cell uses the exact control period admitted for its workload index."""
    config, ledger, root = _load(case_copy)
    primitive = config.to_primitive()
    workloads = primitive["workloads"]
    assert isinstance(workloads, list)
    first = workloads[0]
    assert isinstance(first, dict)
    first["control_dt"] = "0.02"
    altered = QualificationConfig.from_primitive(primitive)
    receipt = _run_probe_receipt(case_copy, root, altered, "control_period")
    donor = replace(ledger.cells[0], receipt=receipt, outcome=detection_outcome(receipt.status))
    with pytest.raises(CampaignInvariantError, match="control period"):
        validate_campaign(config, _spliced(ledger, "cells", 0, donor))


def test_a_disagreeing_campaign_property_is_reported_not_averaged() -> None:
    """The identity helper refuses two observations rather than picking one.

    Tool build and runtime environment cannot be made to differ inside one test process, so the
    agreement rule itself is exercised directly.
    """
    from metrifid.workload_qualification._campaign import _require_single

    check.equal(
        _require_single([{"a": 1}, {"a": 1}], "example"),
        {"a": 1},
        "the agreement rule did not return the single property value every cell reported "
        "identically, so a campaign whose cells actually agree would be described wrongly",
    )
    with pytest.raises(CampaignInvariantError):
        _require_single([{"a": 1}, {"a": 2}], "example")


def test_a_replaced_retained_receipt_refuses_before_publication(case_copy: Path) -> None:
    """Pre-publication verification re-reads the owned tree, not the in-memory decision."""
    _config, ledger, root = _load(case_copy)
    target = root / ledger.controls[0].receipt_locator
    target.write_text("{}", encoding="utf-8")
    owned = OwnedOutputRoot.bind_existing(root)
    try:
        with pytest.raises(CampaignInvariantError):
            verify_retained_evidence(owned, ledger)
    finally:
        owned.close()


def test_a_replaced_generated_configuration_refuses_before_publication(case_copy: Path) -> None:
    """The generated configuration is evidence too, and is rebound the same way."""
    _config, ledger, root = _load(case_copy)
    (root / ledger.cells[0].config_locator).write_text("{}", encoding="utf-8")
    owned = OwnedOutputRoot.bind_existing(root)
    try:
        with pytest.raises(CampaignInvariantError):
            verify_retained_evidence(owned, ledger)
    finally:
        owned.close()


def test_a_symlinked_retained_receipt_refuses_before_publication(
    case_copy: Path, tmp_path: Path
) -> None:
    """A decision-bearing member must be a regular file, not a link to one."""
    _config, ledger, root = _load(case_copy)
    target = root / ledger.controls[0].receipt_locator
    decoy = tmp_path / "decoy.json"
    decoy.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(decoy)
    owned = OwnedOutputRoot.bind_existing(root)
    try:
        with pytest.raises(CampaignInvariantError):
            verify_retained_evidence(owned, ledger)
    finally:
        owned.close()
