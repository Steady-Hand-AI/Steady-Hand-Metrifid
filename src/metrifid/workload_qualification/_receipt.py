"""Build the aggregate receipt, and validate a published one against its own linked evidence.

What `receipt_sha256` is, exactly:

    it detects accidental corruption of the canonical receipt content
    it is not a digital signature
    it does not authenticate an author, a machine, or a campaign
    it alone does not validate decision semantics
    recomputing it cannot make a contradictory receipt valid

It is computed over the canonical self-hash primitive of the receipt object, not over the raw file
bytes, so it is stable across equivalent JSON spellings and says nothing about who produced them.
Trust comes from `_reconstruct`, which recomputes every derived claim, and from the linked-evidence
pass below, which rebinds the document to the comparison receipts it was decided from.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Final

from .._json_admission import (
    CONFIG_JSON_LIMITS,
    RECEIPT_JSON_LIMITS,
    JsonAdmissionError,
    bounded_strict_json_loads,
)
from ..json_values import CanonicalValue, compute_self_hash, strict_json_loads
from ..schemas import ComparisonConfig, ComparisonReceipt
from ._aggregate import (
    AggregateReceipt,
    AggregateSchemaError,
    ControlRecord,
    ProbeCellRecord,
)
from ._campaign import CampaignIdentity
from ._config import QualificationConfig
from ._decision import QualificationDecision
from ._evidence import CampaignLedger, planned_comparisons, raw_digest
from ._owned_output import OwnedOutputError, OwnedOutputRoot
from ._paths import (
    RECEIPT_DIRECTORY,
    PathAdmissionError,
    admit_locator,
)
from ._reconstruct import ReconstructionError, reconstruct, verify_self_hash
from ._status import (
    QUALIFICATION_LIMITATIONS,
    QualificationStatus,
    qualification_exit_code,
)

QUALIFICATION_RECEIPT_SCHEMA: Final[str] = "metrifid.workload_qualification_receipt"
QUALIFICATION_RECEIPT_SCHEMA_VERSION: Final[int] = 1

_NOT_CLAIMED: Final[tuple[str, ...]] = (
    "This qualification does not claim that the selected workloads detect any perturbation that "
    "was not declared as a probe group.",
    "It does not claim that an undeclared workload would behave the same way.",
    "It does not claim global model equivalence; it reports detection under the declared joint "
    "tolerances on the monitored coordinates only.",
    "It does not claim task safety, controller performance, or real-world transfer.",
    "It does not claim that local parameter estimability implies detection. Every cell in this "
    "receipt is a completed Metrifid comparison, not an information-matrix surrogate.",
    "It does not claim that the declared parameter, direction, magnitude, or magnitude semantics "
    "faithfully describe the source edits, or that no other source change exists. Those labels are "
    "preserved user declarations. The Model Change Gate is separate supporting evidence about "
    "source change; it is not applied here and is not automatic proof of these labels.",
    "receipt_sha256 detects accidental corruption. It is not a signature, it authenticates nobody, "
    "and recomputing it cannot make a contradictory receipt valid.",
)


class LinkedEvidenceError(ValueError):
    """Raised when a published receipt does not bind to the evidence retained beside it."""


def build_qualification_receipt(
    *,
    config: QualificationConfig,
    configuration_sha256: str,
    configuration_locator: PurePosixPath,
    decision: QualificationDecision,
    ledger: CampaignLedger,
    identity: CampaignIdentity,
    witnesses: Mapping[str, CanonicalValue],
    baseline_closure: Mapping[str, CanonicalValue],
    probe_closures: Mapping[str, CanonicalValue],
    workload_identities: Mapping[str, CanonicalValue],
) -> dict[str, CanonicalValue]:
    """Build and self-hash one completed qualification receipt."""
    status: QualificationStatus = decision.status
    receipt: dict[str, CanonicalValue] = {
        "schema": QUALIFICATION_RECEIPT_SCHEMA,
        "schema_version": QUALIFICATION_RECEIPT_SCHEMA_VERSION,
        "status": status.value,
        "completed_exit_code": int(qualification_exit_code(status)),
        "configuration": config.to_primitive(),
        "configuration_raw_sha256": configuration_sha256,
        "configuration_locator": configuration_locator.as_posix(),
        "campaign_identity": identity.to_primitive(),
        "baseline_model_closure": dict(baseline_closure),
        "probe_model_closures": dict(probe_closures),
        "workload_artifact_identities": dict(workload_identities),
        "zero_change_controls": [record.to_primitive() for record in ledger.controls],
        "probe_cells": [record.to_primitive() for record in ledger.cells],
        "eligible_workload_ids": list(ledger.eligible_workload_ids),
        "excluded_workload_ids": list(ledger.excluded_workload_ids),
        "selected_workload_ids": list(decision.selected.workload_ids),
        "selection": decision.selected.to_primitive(),
        "subset_ranking": [dict(item) for item in decision.ranking],
        "subsets_evaluated": decision.subsets_evaluated,
        "planned_comparisons": planned_comparisons(config),
        "execution_counts": ledger.counts(),
        "witnesses": dict(witnesses),
        "limitations": [code.value for code in QUALIFICATION_LIMITATIONS],
        "not_claimed": list(_NOT_CLAIMED),
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = compute_self_hash(receipt, "receipt_sha256")
    return receipt


def validate_qualification_receipt(document: Mapping[str, CanonicalValue]) -> AggregateReceipt:
    """Parse one receipt strictly and recompute every derived claim it makes.

    This is the pure half: it needs no filesystem and proves the document is internally coherent. It
    does not, and cannot, prove the document describes evidence that exists.
    """
    try:
        parsed = AggregateReceipt.from_primitive(document)
        parsed.check_registry_invariants()
    except AggregateSchemaError as exc:
        raise ReconstructionError(f"receipt schema: {exc}") from exc
    if parsed.schema != QUALIFICATION_RECEIPT_SCHEMA:
        raise ReconstructionError("unexpected receipt schema")
    if parsed.schema_version != QUALIFICATION_RECEIPT_SCHEMA_VERSION:
        raise ReconstructionError("unexpected receipt schema_version")
    verify_self_hash(document)
    reconstruct(parsed)
    return parsed


def _receipt_root_and_locator(receipt_path: Path) -> tuple[Path, PurePosixPath]:
    """Derive the owned qualification root and the receipt's own locator inside it.

    Only enough lexical shape is read here to say which root to bind and which member to ask that
    root for. Nothing is opened by this pathname: the receipt is the public entry artifact and is
    read through the same binding as the members it registers.
    """
    absolute = receipt_path.absolute()
    parent = absolute.parent
    if parent.name != RECEIPT_DIRECTORY:
        raise LinkedEvidenceError(
            "the published receipt is not in its owned receipt directory, so its linked evidence "
            "root cannot be derived"
        )
    return parent.parent, PurePosixPath(RECEIPT_DIRECTORY) / absolute.name


def _admit_receipt_document(raw: bytes) -> dict[str, CanonicalValue]:
    """Admit the aggregate byte string this owned root returned as strict bounded JSON."""
    try:
        document = bounded_strict_json_loads(raw, RECEIPT_JSON_LIMITS)
    except JsonAdmissionError as exc:
        raise LinkedEvidenceError(f"the published receipt was not admissible: {exc}") from exc
    if not isinstance(document, dict):
        raise LinkedEvidenceError("the published receipt must be a JSON object")
    return document


def _validate_linked_members(parsed: AggregateReceipt, owned: OwnedOutputRoot) -> None:
    """Rebind every recorded locator through the retained owned-root descriptor."""
    config_locator = admit_locator(parsed.configuration_locator, "configuration_locator")
    raw = owned.read_bytes(config_locator, CONFIG_JSON_LIMITS.max_bytes)
    if raw_digest(raw) != parsed.configuration_raw_sha256:
        raise LinkedEvidenceError(
            "the retained raw qualification configuration does not match its recorded digest"
        )
    # The digest binds the retained bytes, and this binds the receipt's own canonical echo of
    # them. Without it a semantic label could be rewritten in the echo while the retained file,
    # and therefore the digest, stayed honest.
    try:
        retained_config = QualificationConfig.from_primitive(
            bounded_strict_json_loads(raw, CONFIG_JSON_LIMITS)
        )
    except (JsonAdmissionError, PathAdmissionError, TypeError, ValueError) as exc:
        raise LinkedEvidenceError(
            f"the retained raw qualification configuration is not admissible: {exc}"
        ) from exc
    if retained_config.to_primitive() != dict(parsed.configuration):
        raise LinkedEvidenceError(
            "the configuration echoed in the receipt is not the configuration retained beside it"
        )
    retained: dict[tuple[int, int | None, int | None], ComparisonReceipt] = {}
    generated: dict[tuple[int, int | None, int | None], ComparisonConfig] = {}
    for control in parsed.zero_change_controls:
        control_key: tuple[int, int | None, int | None] = (
            control.workload_index,
            None,
            None,
        )
        generated[control_key], retained[control_key] = _validate_linked_cell(control, owned)
    for cell in parsed.probe_cells:
        probe_key: tuple[int, int | None, int | None] = (
            cell.workload_index,
            cell.group_index,
            cell.variant_index,
        )
        generated[probe_key], retained[probe_key] = _validate_linked_cell(cell, owned)
    _bind_generated_configurations(parsed, retained_config, generated)
    _validate_linked_identities(parsed, retained_config, retained)


def _validate_linked_cell(
    record: ControlRecord | ProbeCellRecord, owned: OwnedOutputRoot
) -> tuple[ComparisonConfig, ComparisonReceipt]:
    """Rebind one control or cell record to the two files it registers."""
    config_locator = admit_locator(
        record.comparison_config_locator,
        "comparison_config_locator",
    )
    receipt_locator = admit_locator(
        record.comparison_receipt_locator,
        "comparison_receipt_locator",
    )
    config_bytes = owned.read_bytes(config_locator, CONFIG_JSON_LIMITS.max_bytes)
    if raw_digest(config_bytes) != record.comparison_config_raw_sha256:
        raise LinkedEvidenceError(
            "a retained generated comparison configuration does not match its recorded digest"
        )
    try:
        generated = ComparisonConfig.from_primitive(
            bounded_strict_json_loads(config_bytes, CONFIG_JSON_LIMITS)
        )
    except (JsonAdmissionError, TypeError, ValueError) as exc:
        raise LinkedEvidenceError(
            f"a retained generated comparison configuration is not admissible: {exc}"
        ) from exc
    receipt_bytes = owned.read_bytes(receipt_locator, RECEIPT_JSON_LIMITS.max_bytes)
    if raw_digest(receipt_bytes) != record.comparison_receipt_raw_sha256:
        raise LinkedEvidenceError(
            "a retained comparison receipt does not match its recorded raw digest"
        )
    try:
        comparison = ComparisonReceipt.from_primitive(strict_json_loads(receipt_bytes))
    except (TypeError, ValueError) as exc:
        raise LinkedEvidenceError(
            f"a retained comparison receipt is not admissible: {exc}"
        ) from exc
    if comparison.receipt_sha256 != record.comparison_receipt_sha256:
        raise LinkedEvidenceError(
            "a retained comparison receipt's self-hash is not the one the aggregate records"
        )
    if comparison.status.value != record.comparison_status:
        raise LinkedEvidenceError(
            "a retained comparison receipt's status is not the status the aggregate records"
        )
    if comparison.inputs.configuration_raw_sha256 != record.comparison_config_raw_sha256:
        raise LinkedEvidenceError(
            "a retained comparison receipt was not produced from the retained configuration"
        )
    return generated, comparison


def _record_by_key(
    parsed: AggregateReceipt,
) -> dict[tuple[int, int | None, int | None], ControlRecord | ProbeCellRecord]:
    """Index the already reconstructed aggregate records by their canonical campaign coordinates."""
    records: dict[tuple[int, int | None, int | None], ControlRecord | ProbeCellRecord] = {}
    for control in parsed.zero_change_controls:
        records[(control.workload_index, None, None)] = control
    for cell in parsed.probe_cells:
        records[(cell.workload_index, cell.group_index, cell.variant_index)] = cell
    return records


def _declared_campaign_base(observed_model_root: str, declared_model_root: str) -> Path:
    """Recover the original configuration base from one generated absolute model root."""
    observed = Path(observed_model_root)
    declared_parts = PurePosixPath(declared_model_root).parts
    if not observed.is_absolute() or ".." in observed.parts:
        raise LinkedEvidenceError(
            "a retained generated comparison configuration has a non-absolute or non-normalized "
            "baseline model root"
        )
    if not declared_parts or tuple(observed.parts[-len(declared_parts) :]) != declared_parts:
        raise LinkedEvidenceError(
            "a retained generated comparison configuration does not name the admitted baseline "
            "model root"
        )
    return Path(*observed.parts[: -len(declared_parts)])


def _recorded_output_root(observed_output_dir: str, locator: PurePosixPath) -> PurePosixPath:
    """Recover the campaign output root one generated configuration records, by its exact locator."""
    observed = PurePosixPath(observed_output_dir)
    if not observed.is_absolute() or ".." in observed.parts:
        raise LinkedEvidenceError(
            "a retained generated comparison configuration has a non-absolute or non-normalized "
            "comparison output directory"
        )
    tail = locator.parts
    if len(observed.parts) <= len(tail) or observed.parts[-len(tail) :] != tail:
        raise LinkedEvidenceError(
            "a retained generated comparison configuration does not name the exact owned evidence "
            "locator its campaign cell registers"
        )
    return PurePosixPath(*observed.parts[: -len(tail)])


def _campaign_output_root(
    generated: Mapping[tuple[int, int | None, int | None], ComparisonConfig],
    records: Mapping[tuple[int, int | None, int | None], ControlRecord | ProbeCellRecord],
) -> PurePosixPath:
    """Recover the one historical campaign output root every generated configuration records.

    These recorded absolute paths are historical coherence metadata, never an access path: the
    current evidence bytes are authoritative only through the descriptor-bound current root. So the
    requirement here is internal rather than positional. Comparing the recorded root with the
    current one — by full path or by basename — would neither prove object identity, which the
    descriptor already proves, nor survive an honest output root copied elsewhere and renamed.

    What must hold is that every generated configuration names the same historical root and its own
    exact admitted cell locator, so no cell can claim another cell's evidence location.
    """
    roots = {
        _recorded_output_root(
            item.output_dir, PurePosixPath(records[key].comparison_receipt_locator).parent
        )
        for key, item in generated.items()
    }
    if len(roots) != 1:
        raise LinkedEvidenceError(
            "retained generated comparison configurations do not share one campaign output root"
        )
    return next(iter(roots))


def _expected_source_path(base: Path, declared: str) -> str:
    """Reconstruct one generated absolute path from its admitted relative declaration."""
    return str((base / PurePosixPath(declared)).absolute())


def _bind_generated_configurations(
    parsed: AggregateReceipt,
    config: QualificationConfig,
    generated: Mapping[tuple[int, int | None, int | None], ComparisonConfig],
) -> None:
    """Bind every generated comparison configuration to its exact workload and role.

    Model, workload and tolerance declarations are reconstructed from the campaign source base, but
    the comparison output is not. Ownership admits a canonical output root, which need not be the
    spelling the configuration declared, so reconstructing the output from that declaration would
    reject a correct run whose declared path reached its root through a link. The expected output is
    therefore the one historical root every generated configuration agrees on, joined with each
    cell's own exact admitted locator.
    """
    bases = {
        _declared_campaign_base(item.baseline.model_root, config.baseline.model_root)
        for item in generated.values()
    }
    if len(bases) != 1:
        raise LinkedEvidenceError(
            "retained generated comparison configurations do not share one campaign source base"
        )
    base = next(iter(bases))
    records = _record_by_key(parsed)
    output_root = _campaign_output_root(generated, records)
    for key, item in generated.items():
        record = records[key]
        workload = config.workloads[key[0]]
        group_index, variant_index = key[1], key[2]
        if group_index is None:
            candidate = config.baseline
        else:
            if variant_index is None:  # pragma: no cover - reconstructed keys forbid this shape
                raise LinkedEvidenceError("a retained probe configuration has no rung index")
            candidate = config.probe_groups[group_index].variants[variant_index].candidate
        checks = (
            (
                item.baseline.model_root,
                _expected_source_path(base, config.baseline.model_root),
                "baseline model root",
            ),
            (item.baseline.entrypoint, config.baseline.entrypoint, "baseline entrypoint"),
            (
                item.baseline.declared_step_dt,
                config.baseline.declared_step_dt,
                "baseline timestep",
            ),
            (
                item.candidate.model_root,
                _expected_source_path(base, candidate.model_root),
                "candidate model root",
            ),
            (item.candidate.entrypoint, candidate.entrypoint, "candidate entrypoint"),
            (
                item.candidate.declared_step_dt,
                candidate.declared_step_dt,
                "candidate timestep",
            ),
            (
                item.initial_state,
                _expected_source_path(base, workload.initial_state),
                "initial state",
            ),
            (item.actions, _expected_source_path(base, workload.actions), "actions"),
            (item.control_dt, workload.control_dt, "control period"),
            (item.repeats, config.repeats, "repeat count"),
            (item.joint_tolerances, config.joint_tolerances, "joint tolerances"),
            (
                item.aliases,
                None if config.aliases is None else _expected_source_path(base, config.aliases),
                "aliases",
            ),
            (
                item.output_dir,
                str(output_root / PurePosixPath(record.comparison_receipt_locator).parent),
                "comparison output directory",
            ),
        )
        for observed, expected, subject in checks:
            if observed != expected:
                raise LinkedEvidenceError(
                    f"a retained generated comparison configuration names the wrong {subject} "
                    "for its exact campaign cell"
                )


def _validate_linked_identities(
    parsed: AggregateReceipt,
    config: QualificationConfig,
    retained: Mapping[tuple[int, int | None, int | None], ComparisonReceipt],
) -> None:
    """Bind every identity the aggregate reports to the receipts it was actually decided from.

    Reconstruction recomputes derived claims; it cannot check an identity, because an identity is a
    fact the aggregate reports rather than a conclusion it draws. Without this pass a forged
    baseline closure or a tool identity that no cell used still passes, which is exactly what the
    cross-audit reproduced.
    """
    _bind_campaign_identity(parsed, retained)
    _bind_baseline_closure(parsed, config, retained)
    _bind_probe_closures(parsed, config, retained)
    _bind_workload_identities(parsed, config, retained)
    _bind_campaign_contract(config, retained)


def _reported_object(identity: Mapping[str, CanonicalValue], key: str) -> dict[str, CanonicalValue]:
    """Read one reported identity object, treating anything else as absent."""
    value = identity.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _bind_campaign_identity(
    parsed: AggregateReceipt,
    retained: Mapping[tuple[int, int | None, int | None], ComparisonReceipt],
) -> None:
    """Require every retained receipt to carry the campaign identity the aggregate reports."""
    identity = parsed.campaign_identity
    for key, receipt in sorted(retained.items(), key=lambda item: str(item[0])):
        checks = (
            (
                receipt.inputs.baseline_model_closure_sha256,
                identity.get("baseline_model_closure_sha256"),
                "a different baseline model closure",
            ),
            (
                dict(receipt.tool.to_primitive()),
                _reported_object(identity, "tool"),
                "a different tool build",
            ),
            (
                dict(receipt.environment.to_primitive()),
                _reported_object(identity, "environment"),
                "a different runtime environment",
            ),
            (
                receipt.inputs.aliases_raw_sha256,
                identity.get("aliases_raw_sha256"),
                "different raw aliases",
            ),
            (
                receipt.inputs.aliases_semantic_sha256,
                identity.get("aliases_semantic_sha256"),
                "different semantic aliases",
            ),
        )
        for observed, reported, subject in checks:
            if observed != reported:
                raise LinkedEvidenceError(
                    f"a retained comparison receipt used {subject} from the one the aggregate "
                    "reports for the campaign"
                )
        if key[1] is None and receipt.inputs.candidate_model_closure_sha256 != identity.get(
            "baseline_model_closure_sha256"
        ):
            raise LinkedEvidenceError(
                "a retained zero-change control did not compare the baseline against itself"
            )


def _bind_baseline_closure(
    parsed: AggregateReceipt,
    config: QualificationConfig,
    retained: Mapping[tuple[int, int | None, int | None], ComparisonReceipt],
) -> None:
    """Require every retained receipt to carry the admitted campaign baseline closure."""
    reported = dict(parsed.baseline_model_closure)
    if reported.get("entrypoint") != config.baseline.entrypoint:
        raise LinkedEvidenceError(
            "the aggregate's baseline closure entrypoint is not the admitted baseline entrypoint"
        )
    for receipt in retained.values():
        if dict(receipt.model_closures.baseline.to_primitive()) != reported:
            raise LinkedEvidenceError(
                "a retained comparison receipt carries a different baseline model closure from "
                "the one the aggregate reports"
            )
        if receipt.inputs.baseline_model_closure_sha256 != receipt.model_closures.baseline.sha256():
            raise LinkedEvidenceError(
                "a retained comparison receipt's baseline closure hash does not bind its closure"
            )


def _refuse_reused_reported_closures(parsed: AggregateReceipt) -> None:
    """Refuse a reported ladder that reuses the baseline or one closure for two rungs."""
    baseline_sha256 = parsed.campaign_identity.get("baseline_model_closure_sha256")
    for probe_id, rungs in parsed.probe_model_closures.items():
        if not isinstance(rungs, list):
            raise LinkedEvidenceError("probe_model_closures entries must be arrays")
        _group_index_for(parsed, probe_id)
        seen: set[str] = set()
        for rung in rungs:
            if not isinstance(rung, dict) or not isinstance(rung.get("closure_sha256"), str):
                raise LinkedEvidenceError("a probe closure entry must carry a closure SHA-256")
            closure_sha256 = str(rung["closure_sha256"])
            if closure_sha256 == baseline_sha256:
                raise LinkedEvidenceError(
                    f"probe group {probe_id!r} contains a rung with the baseline model closure"
                )
            if closure_sha256 in seen:
                raise LinkedEvidenceError(
                    f"probe group {probe_id!r} contains two rungs with one compiled model closure"
                )
            seen.add(closure_sha256)


def _group_index_for(parsed: AggregateReceipt, probe_id: str) -> int:
    """Return the declared index of one probe group, refusing an unknown identifier."""
    for cell in parsed.probe_cells:
        if cell.probe_id == probe_id:
            return cell.group_index
    raise LinkedEvidenceError(f"probe_model_closures names an unknown probe group {probe_id!r}")


def _bind_probe_closures(
    parsed: AggregateReceipt,
    config: QualificationConfig,
    retained: Mapping[tuple[int, int | None, int | None], ComparisonReceipt],
) -> None:
    """Require every workload's probe receipt to carry its exact admitted rung closure."""
    _refuse_reused_reported_closures(parsed)
    for cell in parsed.probe_cells:
        rungs = parsed.probe_model_closures.get(cell.probe_id)
        if not isinstance(rungs, list):
            raise LinkedEvidenceError("probe_model_closures entries must be arrays")
        rung = next(
            (
                item
                for item in rungs
                if isinstance(item, dict) and item.get("variant_index") == cell.variant_index
            ),
            None,
        )
        if not isinstance(rung, dict):
            raise LinkedEvidenceError(
                "a retained probe cell has no admitted probe closure for its rung"
            )
        receipt = retained[(cell.workload_index, cell.group_index, cell.variant_index)]
        if receipt.inputs.candidate_model_closure_sha256 != rung.get("closure_sha256"):
            raise LinkedEvidenceError(
                "a retained probe receipt carries a candidate closure from the wrong probe rung"
            )
        reported_closure = rung.get("closure")
        if not isinstance(reported_closure, dict) or dict(
            receipt.model_closures.candidate.to_primitive()
        ) != dict(reported_closure):
            raise LinkedEvidenceError(
                "a retained probe receipt's candidate closure object is not its admitted rung"
            )
        variant = config.probe_groups[cell.group_index].variants[cell.variant_index]
        if receipt.model_closures.candidate.entrypoint != variant.candidate.entrypoint:
            raise LinkedEvidenceError(
                "a retained probe receipt used a candidate entrypoint different from its admitted "
                "rung"
            )


def _bind_workload_identities(
    parsed: AggregateReceipt,
    config: QualificationConfig,
    retained: Mapping[tuple[int, int | None, int | None], ComparisonReceipt],
) -> None:
    """Require every control and probe receipt to match its exact admitted workload identity."""
    fields = (
        "initial_state_raw_sha256",
        "initial_state_semantic_sha256",
        "actions_raw_sha256",
        "actions_semantic_sha256",
        "aliases_raw_sha256",
        "aliases_semantic_sha256",
    )
    records: tuple[ControlRecord | ProbeCellRecord, ...] = (
        *parsed.zero_change_controls,
        *parsed.probe_cells,
    )
    for record in records:
        key = (
            (record.workload_index, None, None)
            if isinstance(record, ControlRecord)
            else (record.workload_index, record.group_index, record.variant_index)
        )
        receipt = retained[key]
        workload = config.workloads[record.workload_index]
        declared = parsed.workload_artifact_identities.get(record.workload_id)
        if not isinstance(declared, dict):
            raise LinkedEvidenceError(
                f"the aggregate reports no artifact identity for {record.workload_id!r}"
            )
        for field in fields:
            if declared.get(field) != getattr(receipt.inputs, field):
                raise LinkedEvidenceError(
                    f"a retained cell for {record.workload_id!r} carries a {field} from the wrong "
                    "workload"
                )
        expected_control_dt = workload.control_dt.to_decimal_token()
        if declared.get("control_dt") != expected_control_dt:
            raise LinkedEvidenceError(
                f"the aggregate's control period for {record.workload_id!r} is not the admitted "
                "workload control period"
            )
        if receipt.comparison_contract.control_dt.to_decimal_token() != expected_control_dt:
            raise LinkedEvidenceError(
                f"a retained cell for {record.workload_id!r} used the wrong control period"
            )


def _expected_monitored_joints(
    config: QualificationConfig,
) -> tuple[dict[str, CanonicalValue], ...]:
    """Return the exact monitored-joint contract admitted by the qualification configuration."""
    return tuple(
        {"canonical_name": name, **tolerance.semantic_primitive()}
        for name, tolerance in config.joint_tolerances.items()
    )


def _bind_campaign_contract(
    config: QualificationConfig,
    retained: Mapping[tuple[int, int | None, int | None], ComparisonReceipt],
) -> None:
    """Bind every campaign-wide and role-varying contract member to the configuration."""
    baseline_step = config.baseline.declared_step_dt
    expected_monitored = _expected_monitored_joints(config)
    aliases_expected = config.aliases is not None
    for key, receipt in retained.items():
        contract = receipt.comparison_contract
        if contract.baseline_step_dt != baseline_step:
            raise LinkedEvidenceError(
                "a retained comparison receipt used the wrong admitted baseline timestep"
            )
        if key[1] is None:
            expected_candidate_step = baseline_step
            if dict(receipt.model_closures.candidate.to_primitive()) != dict(
                receipt.model_closures.baseline.to_primitive()
            ):
                raise LinkedEvidenceError(
                    "a retained zero-change control did not compare the baseline against itself"
                )
        else:
            if key[2] is None:  # pragma: no cover - reconstructed keys forbid this shape
                raise LinkedEvidenceError("a retained probe receipt has no rung index")
            expected_candidate_step = (
                config.probe_groups[int(key[1])].variants[int(key[2])].candidate.declared_step_dt
            )
        if contract.candidate_step_dt != expected_candidate_step:
            raise LinkedEvidenceError(
                "a retained comparison receipt used the wrong admitted candidate timestep"
            )
        if contract.repeats != config.repeats:
            raise LinkedEvidenceError(
                "a retained comparison receipt used a repeat count different from the admitted "
                "configuration"
            )
        monitored = tuple(joint.to_primitive() for joint in contract.monitored_joints)
        if monitored != expected_monitored:
            raise LinkedEvidenceError(
                "a retained comparison receipt used different monitored joint names, types, "
                "metrics, or exact tolerances from the admitted configuration"
            )
        if (receipt.inputs.aliases_semantic_sha256 is not None) != aliases_expected:
            raise LinkedEvidenceError(
                "a retained comparison receipt's aliases semantic identity does not match the "
                "admitted configuration"
            )


def load_and_validate_workload_qualification_receipt(
    path: str | Path,
) -> dict[str, CanonicalValue]:
    """Fully validate one published receipt: schema, semantics, and its linked evidence.

    The order matters, and the binding comes first. The owned root the receipt's own location names
    is bound component by component before anything is read, the aggregate is read through that
    binding as one of its members, and the document is then admitted as bounded strict JSON, parsed
    into the strict typed model, recomputed end to end from its own configuration and cell records,
    and rebound to the retained comparison configurations and receipts through the same still-open
    object. The public entry artifact is therefore no weaker than the members it registers: an
    intermediate linked receipt directory refuses here rather than being followed.

    A receipt that passes every stage still carries no authentication: it carries evidence a reader
    can check.
    """
    root, locator = _receipt_root_and_locator(Path(path))
    try:
        owned = OwnedOutputRoot.bind_existing(root)
    except (OwnedOutputError, OSError) as exc:
        raise LinkedEvidenceError(
            f"the owned qualification root could not be bound without following a link: {exc}"
        ) from exc
    with owned:
        try:
            raw = owned.read_bytes(locator, RECEIPT_JSON_LIMITS.max_bytes)
        except (OwnedOutputError, OSError) as exc:
            raise LinkedEvidenceError(f"the published receipt was not admissible: {exc}") from exc
        document = _admit_receipt_document(raw)
        parsed = validate_qualification_receipt(document)
        try:
            _validate_linked_members(parsed, owned)
        except (PathAdmissionError, JsonAdmissionError, OwnedOutputError, OSError) as exc:
            raise LinkedEvidenceError(str(exc)) from exc
    return document


__all__ = [
    "QUALIFICATION_RECEIPT_SCHEMA",
    "QUALIFICATION_RECEIPT_SCHEMA_VERSION",
    "AggregateReceipt",
    "LinkedEvidenceError",
    "build_qualification_receipt",
    "load_and_validate_workload_qualification_receipt",
    "validate_qualification_receipt",
]
