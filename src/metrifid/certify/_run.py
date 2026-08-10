"""Orchestration for one compiled-artifact certification.

The two roles are processed strictly one at a time. A role compiles, is admitted, serializes
into a private file and re-verifies its source closure before the next role compiles, so no two
compiled models and no two MJB buffers are ever resident together.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .._atomic_output import (
    PairedOutputDirectory,
    PairedOutputNames,
    cleanup_paired_output_after_failure,
    prepare_paired_output_directory,
    publish_paired_results,
    verify_paired_results,
)
from .._model_admission import (
    admit_external_implementation_free_model,
    compile_snapshot_model,
    require_supported_runtime,
)
from .._model_closure import (
    ModelAdmissionRefusal,
    ModelClosureSnapshot,
    ModelRole,
    create_model_closure_snapshot,
    refuse,
    verify_model_closure_unchanged,
)
from .._npz import ArtifactAdmissionRefusal
from .._npz import refuse as refuse_artifact
from .._owned_artifacts import RetainedArtifactPair
from .._runtime_identity import CertifyRuntimeIdentity, build_certify_runtime_identity
from ..compare._failure import ComparisonOperationError, operational_error
from ..distribution import installed_distribution_sha256
from ..json_values import CanonicalValue, canonical_json_bytes
from ..operational import OperationalReasonCode, OperationalToolObservation
from ..schemas import ModelClosureIdentity
from ..version import __version__
from ._artifact import SerializedArtifact, serialize_complete_artifact
from ._bytes import compare_artifact_bytes
from ._entrypoint import ResolvedEntrypoint, resolve_entrypoint
from ._fields import build_field_report
from ._markdown import render_markdown
from ._receipt import CertifyResult, RoleCertification, build_receipt
from ._status import CertifyStatus

CERTIFY_OUTPUT_NAMES = PairedOutputNames("certification.json", "certification.md")


@dataclass(frozen=True, slots=True)
class _RoleArtifact:
    """Carry one role's measured closure, live model, and serialized MJB artifact."""

    closure: ModelClosureIdentity
    serialized: SerializedArtifact


def certify_models(
    baseline_mjcf: str,
    candidate_mjcf: str,
    output_directory: str,
    *,
    baseline_root: str | None = None,
    candidate_root: str | None = None,
) -> CertifyResult:
    """Certify whether two source closures compile to byte-identical complete MJB artifacts."""
    tool = OperationalToolObservation(
        __version__, "VERIFIED_INSTALLED_DISTRIBUTION", installed_distribution_sha256()
    )
    published: PairedOutputDirectory | None = None
    try:
        require_supported_runtime()
        baseline_target = resolve_entrypoint(baseline_mjcf, baseline_root, "baseline")
        candidate_target = resolve_entrypoint(candidate_mjcf, candidate_root, "candidate")
        output_path = Path(output_directory)
        _require_output_outside_model_roots(
            output_path, (baseline_target.model_root, candidate_target.model_root)
        )
        published = prepare_paired_output_directory(output_path, CERTIFY_OUTPUT_NAMES)
        return _certify(tool, baseline_target, candidate_target, published)
    except (ModelAdmissionRefusal, ArtifactAdmissionRefusal) as exc:
        cleanup_paired_output_after_failure(published)
        raise _failure(tool, exc) from exc
    except BaseException:
        cleanup_paired_output_after_failure(published)
        raise


def _certify(
    tool: OperationalToolObservation,
    baseline_target: ResolvedEntrypoint,
    candidate_target: ResolvedEntrypoint,
    published: PairedOutputDirectory,
) -> CertifyResult:
    """Run both roles and retain output descriptors through the final decision checks."""
    scratch = Path(tempfile.mkdtemp(prefix="metrifid-certify-"))
    retained: RetainedArtifactPair | None = None
    succeeded = False
    try:
        with ExitStack() as stack:
            baseline_snapshot = stack.enter_context(
                create_model_closure_snapshot(
                    baseline_target.model_root, baseline_target.entrypoint, "baseline"
                )
            )
            baseline = _certify_role(baseline_snapshot, "baseline", scratch)
            runtime = build_certify_runtime_identity(baseline.serialized.header_words)
            candidate_snapshot = stack.enter_context(
                create_model_closure_snapshot(
                    candidate_target.model_root, candidate_target.entrypoint, "candidate"
                )
            )
            candidate = _certify_role(candidate_snapshot, "candidate", scratch)
            status, receipt = _certify_decision(tool, baseline, candidate, runtime)
            verify_model_closure_unchanged(baseline_snapshot, "baseline")
            verify_model_closure_unchanged(candidate_snapshot, "candidate")
            retained = publish_paired_results(
                published,
                json_bytes=canonical_json_bytes(receipt) + b"\n",
                markdown_text=render_markdown(receipt),
            )
            verify_model_closure_unchanged(baseline_snapshot, "baseline")
            verify_model_closure_unchanged(candidate_snapshot, "candidate")
        assert retained is not None
        verify_paired_results(published, retained)
        succeeded = True
        return CertifyResult(status, receipt, published.json_path, published.markdown_path)
    finally:
        if retained is not None:
            if not succeeded:
                retained.cleanup()
            retained.close()
        if succeeded:
            published.close()
        shutil.rmtree(scratch, ignore_errors=True)


def _certify_role(snapshot: ModelClosureSnapshot, role: ModelRole, scratch: Path) -> _RoleArtifact:
    """Compile and serialize one retained role snapshot, then release only its live model."""
    model = compile_snapshot_model(snapshot, role)
    try:
        admit_external_implementation_free_model(model, role)
        serialized = serialize_complete_artifact(model, role, scratch)
    finally:
        del model
    return _RoleArtifact(snapshot.identity, serialized)


def _certify_decision(
    tool: OperationalToolObservation,
    baseline: _RoleArtifact,
    candidate: _RoleArtifact,
    runtime: CertifyRuntimeIdentity,
) -> tuple[CertifyStatus, dict[str, CanonicalValue]]:
    """Complete comparison evidence and receipt construction for two serialized roles."""
    _require_same_header(baseline, candidate)
    comparison = compare_artifact_bytes(baseline.serialized.path, candidate.serialized.path)
    status = (
        CertifyStatus.CERTIFIED_COMPILED_EQUIVALENCE
        if comparison.equal
        else CertifyStatus.NOT_CERTIFIED_COMPILED_DIFFERS
    )
    field_report = (
        None
        if comparison.equal
        else build_field_report(baseline.serialized.path, candidate.serialized.path, comparison)
    )
    receipt = build_receipt(
        status=status,
        tool=_tool_primitive(tool),
        runtime=runtime,
        baseline=_role_certification(baseline, "baseline", runtime.runtime_identity_sha256),
        candidate=_role_certification(candidate, "candidate", runtime.runtime_identity_sha256),
        comparison=comparison,
        field_report=field_report,
    )
    return status, receipt


def _canonical_output_path(path: Path) -> Path | None:
    """Resolve an existing output or its existing parent without creating anything."""
    absolute = path.absolute()
    try:
        absolute.lstat()
    except FileNotFoundError:
        try:
            return absolute.parent.resolve(strict=True) / absolute.name
        except OSError:
            return None
    except OSError:
        return None
    try:
        return absolute.resolve(strict=True)
    except OSError:
        return None


def _require_output_outside_model_roots(output: Path, roots: tuple[Path, Path]) -> None:
    """Refuse an output canonically equal to or below either admitted model root."""
    canonical_output = _canonical_output_path(output)
    if canonical_output is None:
        return
    for root in roots:
        canonical_root = root.resolve(strict=True)
        if canonical_output == canonical_root or canonical_root in canonical_output.parents:
            raise refuse_artifact(
                OperationalReasonCode.OUTPUT_PATH_INVALID,
                issue="output_inside_model_root",
            )


def _role_certification(
    artifact: _RoleArtifact, role: str, runtime_identity_sha256: str | None
) -> RoleCertification:
    """Bind one role's measured closure and MJB to the common Certify runtime."""
    if runtime_identity_sha256 is None:
        raise ValueError("the runtime identity must be finalized before a receipt is built")
    return RoleCertification(
        role, artifact.closure, artifact.serialized.identity(runtime_identity_sha256)
    )


def _require_same_header(baseline: _RoleArtifact, candidate: _RoleArtifact) -> None:
    """Both roles compile in one process, so one differing header word means a bad artifact."""
    if baseline.serialized.header_words == candidate.serialized.header_words:
        return
    raise refuse(
        OperationalReasonCode.COMPILED_ARTIFACT_INVALID,
        "candidate",
        issue="header_words_differ_within_one_runtime",
        baseline_header_words=list(baseline.serialized.header_words),
        candidate_header_words=list(candidate.serialized.header_words),
    )


def _tool_primitive(tool: OperationalToolObservation) -> Mapping[str, CanonicalValue]:
    """Emit the verified tool observation for Certify receipt construction."""
    return {"name": "metrifid", **tool.to_primitive()}


def _failure(
    tool: OperationalToolObservation, exc: ModelAdmissionRefusal | ArtifactAdmissionRefusal
) -> ComparisonOperationError:
    """Convert an expected model or artifact refusal into Certify operational evidence."""
    return operational_error(
        tool=tool,
        code=exc.reason,
        role=exc.role,
        evidence=cast("Mapping[str, CanonicalValue]", exc.evidence),
        operation="certify",
    )


__all__ = ["CERTIFY_OUTPUT_NAMES", "certify_models"]
