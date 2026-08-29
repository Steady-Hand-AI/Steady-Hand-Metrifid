"""Prove every completed comparison belongs to one coherent campaign.

An individual comparison already freezes and reverifies its own sources. What it cannot know is
whether the twenty other comparisons the qualification aggregated saw the same baseline model, the
same workload artifacts, the same tool build, and the same runtime. Sampling one receipt for those
identities, as the previous implementation did, let a later cell disagree silently.

So the invariants are checked across the whole ledger before any subset is selected, and the
retained files are re-read from the owned output tree immediately before the aggregate is published.
Nothing here copies model trees or re-implements compare's source freezing: the guarantee is that
every typed receipt refers to one campaign identity, not that external source paths never change
afterwards.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, cast

from .._json_admission import CONFIG_JSON_LIMITS, RECEIPT_JSON_LIMITS
from ..json_values import CanonicalValue, strict_json_loads
from ..schemas import ComparisonReceipt
from ._config import QualificationConfig
from ._evidence import (
    PROBE_KIND,
    CampaignLedger,
    CellRecord,
    raw_digest,
)
from ._owned_output import OwnedOutputError, OwnedOutputRoot

# A probe cell legitimately differs from its control in exactly these contract members: it compares a
# different candidate closure, and that candidate may declare its own timestep.
_ROLE_VARYING_CONTRACT_FIELDS: Final[frozenset[str]] = frozenset(
    {"candidate_model_closure_sha256", "candidate_step_dt"}
)


class CampaignInvariantError(ValueError):
    """Raised when completed cells do not describe one coherent campaign."""


@dataclass(frozen=True, slots=True)
class CampaignIdentity:
    """The single identity every cell of one campaign must agree on."""

    tool: CanonicalValue
    environment: CanonicalValue
    baseline_model_closure_sha256: str
    aliases_raw_sha256: str | None
    aliases_semantic_sha256: str | None

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the campaign identity."""
        return {
            "tool": self.tool,
            "environment": self.environment,
            "baseline_model_closure_sha256": self.baseline_model_closure_sha256,
            "aliases_raw_sha256": self.aliases_raw_sha256,
            "aliases_semantic_sha256": self.aliases_semantic_sha256,
        }


def _require_single(values: Iterable[object], subject: str) -> object:
    """Require every observation of one campaign property to be identical."""
    distinct = []
    for value in values:
        if value not in distinct:
            distinct.append(value)
    if not distinct:
        raise CampaignInvariantError(f"no cell reported {subject}")
    if len(distinct) > 1:
        raise CampaignInvariantError(
            f"{subject} differs across cells of one campaign; the aggregate would describe a "
            "identity that not every comparison actually used"
        )
    return distinct[0]


def _contract_without_role_variation(receipt: ComparisonReceipt) -> dict[str, object]:
    """Return the contract members that must not vary between a control and its probe cells."""
    contract = dict(receipt.comparison_contract.to_primitive())
    return {k: v for k, v in contract.items() if k not in _ROLE_VARYING_CONTRACT_FIELDS}


def validate_campaign(config: QualificationConfig, ledger: CampaignLedger) -> CampaignIdentity:
    """Check every cross-cell invariant and return the campaign's single identity.

    Deliberately absent: any rule that a candidate timestep must equal the baseline timestep. A probe
    may legitimately declare a different timestep. What is required is that every cell's declared
    timestep equals the value the admitted configuration declared for that role.
    """
    records = (*ledger.controls, *ledger.cells)
    if not records:
        raise CampaignInvariantError("a campaign must contain at least one completed comparison")

    tool = cast(
        "CanonicalValue",
        _require_single((dict(r.receipt.tool.to_primitive()) for r in records), "tool identity"),
    )
    environment = cast(
        "CanonicalValue",
        _require_single(
            (dict(r.receipt.environment.to_primitive()) for r in records),
            "runtime environment identity",
        ),
    )
    baseline_closure = str(
        _require_single(
            (r.receipt.inputs.baseline_model_closure_sha256 for r in records),
            "baseline model closure identity",
        )
    )
    aliases_raw = _require_single(
        (r.receipt.inputs.aliases_raw_sha256 for r in records), "aliases raw identity"
    )
    aliases_semantic = _require_single(
        (r.receipt.inputs.aliases_semantic_sha256 for r in records), "aliases semantic identity"
    )

    for control in ledger.controls:
        if control.receipt.inputs.candidate_model_closure_sha256 != baseline_closure:
            raise CampaignInvariantError(
                f"the zero-change control for {control.workload_id!r} did not compare the baseline "
                "model against itself"
            )

    _validate_probe_closures(config, ledger)
    _validate_workload_artifacts(config, ledger)
    _validate_declared_timesteps(config, ledger)
    _validate_campaign_contract(config, ledger)
    _validate_contracts(ledger)
    _validate_ladder_closures(config, ledger, baseline_closure)

    return CampaignIdentity(
        tool=tool,
        environment=environment,
        baseline_model_closure_sha256=baseline_closure,
        aliases_raw_sha256=None if aliases_raw is None else str(aliases_raw),
        aliases_semantic_sha256=None if aliases_semantic is None else str(aliases_semantic),
    )


def _validate_probe_closures(config: QualificationConfig, ledger: CampaignLedger) -> None:
    """Require one candidate closure per admitted probe-group rung across every workload."""
    by_rung: dict[tuple[int, int], list[CellRecord]] = {}
    for cell in ledger.cells:
        key = (int(cell.group_index or 0), int(cell.variant_index or 0))
        by_rung.setdefault(key, []).append(cell)
    for (group_index, variant_index), records in sorted(by_rung.items()):
        closures = {r.receipt.inputs.candidate_model_closure_sha256 for r in records}
        if len(closures) > 1:
            raise CampaignInvariantError(
                f"probe group {group_index} rung {variant_index} compared more than one candidate "
                "model closure across workloads"
            )
        try:
            variant = config.probe_groups[group_index].variants[variant_index]
        except IndexError as exc:
            raise CampaignInvariantError("a probe cell names an undeclared group or rung") from exc
        for record in records:
            if record.receipt.model_closures.candidate.entrypoint != variant.candidate.entrypoint:
                raise CampaignInvariantError(
                    f"probe group {group_index} rung {variant_index} used a candidate entrypoint "
                    "different from the admitted rung"
                )


def _validate_workload_artifacts(config: QualificationConfig, ledger: CampaignLedger) -> None:
    """Bind every control and probe receipt to its exact admitted workload identity."""
    fields = (
        "initial_state_raw_sha256",
        "initial_state_semantic_sha256",
        "actions_raw_sha256",
        "actions_semantic_sha256",
    )
    controls = {control.workload_index: control for control in ledger.controls}
    for record in (*ledger.controls, *ledger.cells):
        try:
            workload = config.workloads[record.workload_index]
        except IndexError as exc:
            raise CampaignInvariantError("a cell names an undeclared workload index") from exc
        if record.workload_id != workload.workload_id:
            raise CampaignInvariantError(
                "a cell's workload identifier does not match its admitted workload index"
            )
        control = controls.get(record.workload_index)
        if control is None:
            raise CampaignInvariantError(
                f"workload {workload.workload_id!r} has no zero-change control identity"
            )
        for field in fields:
            if getattr(record.receipt.inputs, field) != getattr(control.receipt.inputs, field):
                raise CampaignInvariantError(
                    f"a cell for workload {workload.workload_id!r} used a different {field} from "
                    "that workload's admitted control identity"
                )


def _validate_declared_timesteps(config: QualificationConfig, ledger: CampaignLedger) -> None:
    """Require each cell's declared timesteps to equal the admitted configuration's own values."""
    baseline_token = config.baseline.declared_step_dt.to_decimal_token()
    for record in (*ledger.controls, *ledger.cells):
        try:
            workload = config.workloads[record.workload_index]
        except IndexError as exc:
            raise CampaignInvariantError("a cell names an undeclared workload index") from exc
        time = record.receipt.comparison_contract
        if _token(time.baseline_step_dt) != baseline_token:
            raise CampaignInvariantError(
                f"a cell for workload {record.workload_id!r} declared a baseline timestep that is "
                "not the admitted baseline timestep"
            )
        if record.kind == PROBE_KIND:
            variant = config.probe_groups[int(record.group_index or 0)].variants[
                int(record.variant_index or 0)
            ]
            expected = variant.candidate.declared_step_dt.to_decimal_token()
        else:
            expected = baseline_token
        if _token(time.candidate_step_dt) != expected:
            raise CampaignInvariantError(
                "a cell declared a candidate timestep that is not the admitted value for its role"
            )
        if _token(time.control_dt) != workload.control_dt.to_decimal_token():
            raise CampaignInvariantError(
                f"workload {record.workload_id!r} used a control period different from its "
                "admitted configuration"
            )


def _token(value: object) -> str:
    """Return one exact decimal token for a contract timing member."""
    if hasattr(value, "to_decimal_token"):
        return str(value.to_decimal_token())
    return str(value)


def _expected_monitored_joints(
    config: QualificationConfig,
) -> tuple[dict[str, CanonicalValue], ...]:
    """Return the exact monitored-joint contract admitted by the qualification configuration."""
    return tuple(
        {"canonical_name": name, **tolerance.semantic_primitive()}
        for name, tolerance in config.joint_tolerances.items()
    )


def _validate_campaign_contract(config: QualificationConfig, ledger: CampaignLedger) -> None:
    """Bind every campaign-wide receipt contract member to the admitted configuration."""
    expected_monitored = _expected_monitored_joints(config)
    aliases_expected = config.aliases is not None
    for record in (*ledger.controls, *ledger.cells):
        receipt = record.receipt
        contract = receipt.comparison_contract
        if receipt.model_closures.baseline.entrypoint != config.baseline.entrypoint:
            raise CampaignInvariantError(
                "a cell used a baseline entrypoint different from the admitted configuration"
            )
        if contract.repeats != config.repeats:
            raise CampaignInvariantError(
                "a cell used a repeat count different from the admitted configuration"
            )
        monitored = tuple(joint.to_primitive() for joint in contract.monitored_joints)
        if monitored != expected_monitored:
            raise CampaignInvariantError(
                "a cell used monitored joint names, types, metrics, or exact tolerances different "
                "from the admitted configuration"
            )
        aliases_present = receipt.inputs.aliases_semantic_sha256 is not None
        if aliases_present != aliases_expected:
            raise CampaignInvariantError(
                "a cell's aliases semantic identity does not match the admitted configuration"
            )


def _validate_contracts(ledger: CampaignLedger) -> None:
    """Require one comparison contract per workload, up to the expected role variation."""
    by_workload: dict[str, list[CellRecord]] = {}
    for record in (*ledger.controls, *ledger.cells):
        by_workload.setdefault(record.workload_id, []).append(record)
    for workload_id, records in sorted(by_workload.items()):
        _require_single(
            (_contract_without_role_variation(r.receipt) for r in records),
            f"comparison contract for workload {workload_id!r}",
        )


def _validate_ladder_closures(
    config: QualificationConfig, ledger: CampaignLedger, baseline_closure: str
) -> None:
    """Refuse a ladder whose rungs impersonate the baseline or each other.

    Distinct magnitude labels that identify one closure make a reported detection floor misleading
    even when every measurement is correct. This does not verify the user's source-edit claim; it
    only stops one closure from standing in for several rungs.
    """
    for group_index, group in enumerate(config.probe_groups):
        seen: dict[str, int] = {}
        for variant_index in range(len(group.variants)):
            cells = [
                c
                for c in ledger.cells
                if c.group_index == group_index and c.variant_index == variant_index
            ]
            if not cells:
                continue
            closure = cells[0].receipt.inputs.candidate_model_closure_sha256
            if closure == baseline_closure:
                raise CampaignInvariantError(
                    f"probe group {group.probe_id!r} rung {variant_index} admits the same model "
                    "closure as the baseline, so it declares a perturbation that is not one"
                )
            if closure in seen:
                raise CampaignInvariantError(
                    f"probe group {group.probe_id!r} rungs {seen[closure]} and {variant_index} "
                    "admit the same model closure, so two declared magnitudes name one model"
                )
            seen[closure] = variant_index


def verify_retained_evidence(owned: OwnedOutputRoot, ledger: CampaignLedger) -> None:
    """Re-read every registered artifact from the owned tree and rebind it to the typed ledger.

    Run immediately before the aggregate is published. Building a correct decision in memory is not
    enough if the evidence it points at has since gone missing, been replaced, or been swapped for a
    symlink.
    """
    for record in (*ledger.controls, *ledger.cells):
        try:
            config_bytes = owned.read_bytes(record.config_locator, CONFIG_JSON_LIMITS.max_bytes)
            receipt_bytes = owned.read_bytes(record.receipt_locator, RECEIPT_JSON_LIMITS.max_bytes)
        except OwnedOutputError as exc:
            raise CampaignInvariantError(
                f"retained evidence for workload index {record.workload_index} is missing or is "
                f"not a regular file: {exc}"
            ) from exc
        if raw_digest(config_bytes) != record.config_raw_sha256:
            raise CampaignInvariantError(
                "a retained generated comparison configuration no longer matches the bytes the "
                "comparison read"
            )
        if raw_digest(receipt_bytes) != record.receipt_raw_sha256:
            raise CampaignInvariantError(
                "a retained comparison receipt no longer matches the bytes recorded for it"
            )
        retained = ComparisonReceipt.from_primitive(strict_json_loads(receipt_bytes))
        if retained != record.receipt:
            raise CampaignInvariantError(
                "a retained comparison receipt is not the receipt this campaign decided from"
            )
