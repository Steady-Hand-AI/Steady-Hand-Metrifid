"""Fail-closed orchestration for one installed Native Runtime Review."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .._json_admission import JsonAdmissionError
from .._npz import ArtifactAdmissionRefusal
from ..compare._failure import ComparisonOperationError, operational_error
from ..distribution import installed_distribution_sha256
from ..json_values import CanonicalValue, canonical_json_bytes
from ..operational import OperationalReasonCode, OperationalToolObservation
from ..version import __version__
from ._config import (
    AdmittedRuntimeReviewConfigurationAny,
    AdmittedRuntimeReviewConfigurationV2,
    load_runtime_review_configuration,
)
from ._decision import RuntimeReviewDecision, evaluate_runtime_evidence
from ._evidence import (
    AdmittedRuntimeEvidence,
    RuntimeEvidenceAdmissionError,
    admit_runtime_evidence,
)
from ._markdown import render_runtime_review_markdown
from ._owned_output import (
    OwnedRuntimeReviewOutputError,
    PublishedRuntimeReviewOutput,
    prepare_owned_runtime_review_output,
    verify_published_runtime_review_output,
)
from ._paths import PathAdmissionError
from ._receipt import build_runtime_review_receipt, build_runtime_review_receipt_v2
from ._receipt_validation import (
    load_and_validate_runtime_review_receipt,
    validate_runtime_review_tree,
)
from ._status import RuntimeReviewReasonCode, RuntimeReviewStatus, runtime_review_exit_code

_OPERATION = "compare"


@dataclass(frozen=True, slots=True)
class RuntimeReviewResult:
    """One completed decision and its independently validated owned output."""

    status: RuntimeReviewStatus
    reason_code: RuntimeReviewReasonCode | None
    receipt: dict[str, CanonicalValue]
    receipt_sha256: str
    runtime_review_json: Path
    runtime_review_markdown: Path

    @property
    def exit_code(self) -> int:
        """Return the frozen process exit code for this completed status."""
        return int(runtime_review_exit_code(self.status))


def review_runtime_configuration_file(config_path: str | Path) -> RuntimeReviewResult:
    """Admit, decide, own, publish, and independently replay one runtime review."""
    configuration = _load_configuration(Path(config_path).absolute())
    evidence = _admit_evidence(configuration)
    decision = evaluate_runtime_evidence(evidence)
    published, receipt = _publish(configuration, evidence, decision)

    receipt_sha256 = receipt.get("receipt_sha256")
    if not isinstance(receipt_sha256, str):
        raise RuntimeError("completed runtime-review receipt lacks its self-hash")
    return RuntimeReviewResult(
        status=decision.status,
        reason_code=decision.reason_code,
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        runtime_review_json=published.runtime_review_json,
        runtime_review_markdown=published.runtime_review_markdown,
    )


def _load_configuration(path: Path) -> AdmittedRuntimeReviewConfigurationAny:
    """Load one configuration and convert all caller-controlled failures to exit 64."""
    try:
        return load_runtime_review_configuration(path)
    except (JsonAdmissionError, PathAdmissionError, TypeError, ValueError, UnicodeError) as exc:
        raise _refuse(
            OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            field="runtime_review_config",
            exception_type=type(exc).__name__,
            message=str(exc)[:300],
        ) from exc
    except OSError as exc:
        raise _refuse(
            OperationalReasonCode.CONFIGURATION_IO_FAILED,
            field="runtime_review_config",
            exception_type=type(exc).__name__,
            message=str(exc)[:300],
        ) from exc


def _admit_evidence(
    configuration: AdmittedRuntimeReviewConfigurationAny,
) -> AdmittedRuntimeEvidence:
    """Admit every configured cell and convert invalid evidence to exit 64."""
    try:
        return admit_runtime_evidence(configuration)
    except (ArtifactAdmissionRefusal, JsonAdmissionError, RuntimeEvidenceAdmissionError) as exc:
        raise _refuse(
            OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            field="evidence_cells",
            exception_type=type(exc).__name__,
            message=str(exc)[:300],
        ) from exc
    except OSError as exc:
        raise _refuse(
            OperationalReasonCode.CONFIGURATION_IO_FAILED,
            field="evidence_cells",
            exception_type=type(exc).__name__,
            message=str(exc)[:300],
        ) from exc


def _publish(
    configuration: AdmittedRuntimeReviewConfigurationAny,
    evidence: AdmittedRuntimeEvidence,
    decision: RuntimeReviewDecision,
) -> tuple[PublishedRuntimeReviewOutput, dict[str, CanonicalValue]]:
    """Publish, independently replay, and final-boundary verify one completed review."""
    published = None
    try:
        with prepare_owned_runtime_review_output(
            configuration.output_dir, configuration.raw_bytes
        ) as staging:
            owned_cells = staging.copy_evidence_cells(evidence.cells)
            if isinstance(configuration, AdmittedRuntimeReviewConfigurationV2):
                declarations = (
                    configuration.config.baseline_profile,
                    configuration.config.candidate_profile,
                )
                owned_profiles = staging.copy_profile_identities(
                    {
                        role: (
                            configuration.profile_identity_path(role),
                            configuration.profile_identity_file_hash(role),
                        )
                        for role, declaration in zip(
                            ("baseline", "candidate"), declarations, strict=True
                        )
                    }
                )
                receipt = build_runtime_review_receipt_v2(
                    configuration=configuration,
                    evidence=evidence,
                    decision=decision,
                    evidence_cells=owned_cells,
                    profile_identities=owned_profiles,
                )
            else:
                receipt = build_runtime_review_receipt(
                    configuration=configuration,
                    evidence=evidence,
                    decision=decision,
                    evidence_cells=owned_cells,
                )
            published = staging.publish(
                receipt,
                render_runtime_review_markdown(receipt),
                prepublication_validator=validate_runtime_review_tree,
            )
        loaded = load_and_validate_runtime_review_receipt(published.runtime_review_json)
        if canonical_json_bytes(loaded) != canonical_json_bytes(receipt):
            raise OwnedRuntimeReviewOutputError(
                "published receipt differs from its independently replayed document"
            )
        verify_published_runtime_review_output(published)
    except ComparisonOperationError:
        raise
    except (OwnedRuntimeReviewOutputError, OSError) as exc:
        raise _refuse(
            OperationalReasonCode.OUTPUT_WRITE_FAILED,
            field="output_dir",
            exception_type=type(exc).__name__,
            message=str(exc)[:300],
        ) from exc
    except (JsonAdmissionError, RuntimeEvidenceAdmissionError, TypeError, ValueError) as exc:
        raise _refuse(
            OperationalReasonCode.INTERNAL_INVARIANT_FAILED,
            field="runtime_review_output",
            exception_type=type(exc).__name__,
            message=str(exc)[:300],
        ) from exc
    except BaseException:
        raise
    return published, receipt


def _tool() -> OperationalToolObservation:
    """Observe the installed distribution that owns one operational refusal."""
    return OperationalToolObservation(
        __version__, "VERIFIED_INSTALLED_DISTRIBUTION", installed_distribution_sha256()
    )


def _refuse(
    code: OperationalReasonCode,
    *,
    field: str,
    **evidence: CanonicalValue,
) -> ComparisonOperationError:
    """Construct one bounded exit-64 failure without expanding the operational ABI."""
    return operational_error(
        tool=_tool(),
        code=code,
        role=None,
        evidence=dict(evidence),
        field=field,
        operation=_OPERATION,
    )


__all__ = ["RuntimeReviewResult", "review_runtime_configuration_file"]
