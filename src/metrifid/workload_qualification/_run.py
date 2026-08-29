"""Orchestrate one qualification: admit, own, execute, bind, decide, verify, publish.

The order is fail-closed and each stage is a separate function, because each one answers a different
question and mixing them is what let the previous version create directories inside a model root
before discovering it should have refused.

    admit        every path and the configuration document, bounded and no-follow
    own          decide output ownership completely before creating anything
    execute      run every cell into an ordinal directory and bind it to its bytes
    bind         prove the completed cells describe one campaign
    decide       select the subset by exact enumeration
    verify       re-read the retained evidence from the owned tree
    publish      write the aggregate pair atomically
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from .._atomic_output import (
    PairedOutputNames,
    _adopt_paired_output_descriptor,
    cleanup_paired_output_after_failure,
    publish_paired_results,
    verify_paired_results,
)
from .._json_admission import (
    CONFIG_JSON_LIMITS,
    JsonAdmissionError,
    bounded_strict_json_loads,
    read_bounded_regular_file,
)
from ..compare._failure import ComparisonOperationError, operational_error
from ..distribution import installed_distribution_sha256
from ..json_values import CanonicalValue, canonical_json_bytes
from ..operational import OperationalReasonCode, OperationalToolObservation
from ..version import __version__
from ._campaign import CampaignInvariantError, validate_campaign, verify_retained_evidence
from ._config import QualificationConfig
from ._decision import decide
from ._evidence import (
    CampaignLedger,
    CellBindingError,
    build_campaign_ledger,
    exclusion_reason,
    raw_digest,
)
from ._markdown import render_markdown
from ._owned_output import OwnedOutputError, OwnedOutputRoot, preflight
from ._paths import (
    ADMITTED_CONFIG_NAME,
    RECEIPT_DIRECTORY,
    PathAdmissionError,
    resolve_under,
)
from ._receipt import build_qualification_receipt
from ._status import QualificationStatus, qualification_exit_code
from ._witness import collect_witnesses

QUALIFICATION_OUTPUT_NAMES: Final[PairedOutputNames] = PairedOutputNames(
    "workload_qualification.json", "workload_qualification.md"
)
OPERATION: Final[str] = "qualify-workload"


@dataclass(frozen=True, slots=True)
class QualificationResult:
    """One published qualification and the two files it wrote."""

    status: QualificationStatus
    receipt: dict[str, CanonicalValue]
    receipt_sha256: str
    qualification_json: Path
    qualification_markdown: Path

    @property
    def exit_code(self) -> int:
        """Return the frozen process exit code for this completed status."""
        return int(qualification_exit_code(self.status))


def _tool() -> OperationalToolObservation:
    """Observe the installed distribution this qualification is executing from."""
    return OperationalToolObservation(
        __version__, "VERIFIED_INSTALLED_DISTRIBUTION", installed_distribution_sha256()
    )


def _refuse(
    code: OperationalReasonCode, *, field: str | None = None, **evidence: CanonicalValue
) -> ComparisonOperationError:
    """Build one bounded operational failure on the frozen operational registry."""
    return operational_error(
        tool=_tool(),
        code=code,
        role=None,
        evidence=dict(evidence),
        field=field,
        operation=OPERATION,
    )


def admit_configuration(path: Path) -> tuple[QualificationConfig, bytes, str]:
    """Admit the configuration as a bounded strict JSON document from a regular no-follow file."""
    try:
        raw = read_bounded_regular_file(path, CONFIG_JSON_LIMITS.max_bytes)
    except (JsonAdmissionError, OSError) as exc:
        raise _refuse(
            OperationalReasonCode.CONFIGURATION_IO_FAILED,
            field="qualification_config",
            exception_type=type(exc).__name__,
            message=str(exc)[:200],
        ) from exc
    try:
        primitive = bounded_strict_json_loads(raw, CONFIG_JSON_LIMITS)
    except (JsonAdmissionError, UnicodeDecodeError, TypeError, ValueError) as exc:
        raise _refuse(
            OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            field="qualification_config",
            exception_type=type(exc).__name__,
            message=str(exc)[:200],
        ) from exc
    try:
        config = QualificationConfig.from_primitive(primitive)
    except (PathAdmissionError, TypeError, ValueError) as exc:
        raise _refuse(
            OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            field="qualification_config",
            exception_type=type(exc).__name__,
            message=str(exc)[:200],
        ) from exc
    return config, raw, raw_digest(raw)


def own_output(config: QualificationConfig, base: Path) -> OwnedOutputRoot:
    """Decide ownership of the output root completely, then create it.

    Every model root is resolved and compared against the proposed output in canonical form before
    anything exists on disk, so a configuration that would write inside an observed model leaves
    that model untouched. The canonical output that check accepted is the path that is created.
    """
    roots: dict[str, Path] = {
        "baseline": resolve_under(base, config.baseline.model_root, "baseline.model_root")
    }
    for group in config.probe_groups:
        for index, variant in enumerate(group.variants):
            roots[f"probe {group.probe_id!r} rung {index}"] = resolve_under(
                base, variant.candidate.model_root, "probe candidate.model_root"
            )
    proposed = resolve_under(base, config.output_dir, "output_dir")
    try:
        # The canonical result is the identity ownership was decided against, so it is the only
        # spelling creation may bind. Recreating the original spelling would let an intermediate
        # component be retargeted between the decision and the write.
        accepted = preflight(proposed, roots)
    except OwnedOutputError as exc:
        raise _refuse(
            OperationalReasonCode.OUTPUT_PATH_INVALID,
            field="output_dir",
            message=str(exc)[:300],
        ) from exc
    try:
        return OwnedOutputRoot(accepted)
    except (OwnedOutputError, OSError) as exc:
        raise _refuse(
            OperationalReasonCode.OUTPUT_PATH_INVALID,
            field="output_dir",
            exception_type=type(exc).__name__,
            message=str(exc)[:200],
        ) from exc


def _harvest_identities(
    config: QualificationConfig, ledger: CampaignLedger
) -> tuple[dict[str, CanonicalValue], dict[str, CanonicalValue], dict[str, CanonicalValue]]:
    """Read the campaign's identities from the validated typed receipts, not from sampled JSON."""
    first_control = ledger.controls[0]
    baseline_closure = first_control.receipt.model_closures.baseline.to_primitive()

    probe_closures: dict[str, CanonicalValue] = {}
    for group_index, group in enumerate(config.probe_groups):
        rungs: list[CanonicalValue] = []
        for variant_index, variant in enumerate(group.variants):
            cell = next(
                c
                for c in ledger.cells
                if c.group_index == group_index and c.variant_index == variant_index
            )
            rungs.append(
                {
                    "variant_index": variant_index,
                    "magnitude": variant.magnitude.to_decimal_token(),
                    "closure": cell.receipt.model_closures.candidate.to_primitive(),
                    "closure_sha256": cell.receipt.inputs.candidate_model_closure_sha256,
                }
            )
        probe_closures[group.probe_id] = rungs

    workload_identities: dict[str, CanonicalValue] = {}
    for control in ledger.controls:
        inputs = control.receipt.inputs
        workload_identities[control.workload_id] = {
            "initial_state_raw_sha256": inputs.initial_state_raw_sha256,
            "initial_state_semantic_sha256": inputs.initial_state_semantic_sha256,
            "actions_raw_sha256": inputs.actions_raw_sha256,
            "actions_semantic_sha256": inputs.actions_semantic_sha256,
            "aliases_raw_sha256": inputs.aliases_raw_sha256,
            "aliases_semantic_sha256": inputs.aliases_semantic_sha256,
            "control_dt": config.workloads[control.workload_index].control_dt.to_decimal_token(),
        }
    return baseline_closure, probe_closures, workload_identities


def qualify_configuration_file(config_path: str | Path) -> QualificationResult:
    """Run one complete workload qualification and publish its receipt pair atomically."""
    path = Path(config_path).absolute()
    config, raw, configuration_sha256 = admit_configuration(path)
    owned = own_output(config, path.parent)
    try:
        config_locator = PurePosixPath(ADMITTED_CONFIG_NAME)
        owned.write_bytes(config_locator, raw)

        try:
            ledger = build_campaign_ledger(owned, config, path)
        except CellBindingError as exc:
            raise _refuse(
                OperationalReasonCode.INTERNAL_INVARIANT_FAILED,
                field="comparison_cell",
                message=str(exc)[:300],
            ) from exc

        try:
            identity = validate_campaign(config, ledger)
        except CampaignInvariantError as exc:
            raise _refuse(
                OperationalReasonCode.INTERNAL_INVARIANT_FAILED,
                field="campaign",
                message=str(exc)[:300],
            ) from exc

        eligible = ledger.eligible_workload_ids
        if len(eligible) < config.budget:
            raise _refuse(
                OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
                field="workloads",
                message=(
                    f"{len(eligible)} workloads passed their zero-change control, but "
                    f"{config.budget} are required to select a subset"
                ),
                excluded_workload_ids=list(ledger.excluded_workload_ids),
            )

        decision = decide(config, ledger.outcome_map(), eligible)
        reasons = {
            control.workload_id: exclusion_reason(control.status)
            for control in ledger.controls
            if not control.eligible
        }
        witnesses = collect_witnesses(
            config,
            decision.selected,
            decision.status,
            ledger.outcome_map(),
            ledger.excluded_workload_ids,
            reasons,
        )
        baseline_closure, probe_closures, workload_identities = _harvest_identities(config, ledger)
        receipt = build_qualification_receipt(
            config=config,
            configuration_sha256=configuration_sha256,
            configuration_locator=config_locator,
            decision=decision,
            ledger=ledger,
            identity=identity,
            witnesses=witnesses,
            baseline_closure=baseline_closure,
            probe_closures=probe_closures,
            workload_identities=workload_identities,
        )
        markdown = render_markdown(receipt)

        try:
            verify_retained_evidence(owned, ledger)
        except CampaignInvariantError as exc:
            raise _refuse(
                OperationalReasonCode.INTERNAL_INVARIANT_FAILED,
                field="retained_evidence",
                message=str(exc)[:300],
            ) from exc

        json_path, markdown_path = _publish(owned, receipt, markdown)
    finally:
        owned.close()

    return QualificationResult(
        status=QualificationStatus(str(receipt["status"])),
        receipt=receipt,
        receipt_sha256=str(receipt["receipt_sha256"]),
        qualification_json=json_path,
        qualification_markdown=markdown_path,
    )


def _publish(
    owned: OwnedOutputRoot, receipt: dict[str, CanonicalValue], markdown: str
) -> tuple[Path, Path]:
    """Publish the aggregate pair atomically into the owned receipt directory."""
    locator = PurePosixPath(RECEIPT_DIRECTORY)
    owned.make_directory(locator)
    receipt_fd = owned.open_owned_directory(locator)
    try:
        # The publisher is built from the retained descriptor itself, so a replacement of the public
        # receipt path can only cause a typed refusal; it can never redirect the write.
        output = _adopt_paired_output_descriptor(
            owned.display_path(locator), QUALIFICATION_OUTPUT_NAMES, os.dup(receipt_fd)
        )
    finally:
        os.close(receipt_fd)
    retained = None
    try:
        retained = publish_paired_results(
            output,
            json_bytes=canonical_json_bytes(receipt) + b"\n",
            markdown_text=markdown,
        )
        verify_paired_results(output, retained)
        return (
            output.path / QUALIFICATION_OUTPUT_NAMES.json_name,
            output.path / QUALIFICATION_OUTPUT_NAMES.markdown_name,
        )
    except BaseException:
        cleanup_paired_output_after_failure(output, retained)
        raise
    finally:
        output.close()
