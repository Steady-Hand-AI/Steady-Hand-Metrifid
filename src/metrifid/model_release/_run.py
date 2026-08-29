"""Installed Model Change Gate orchestration over exact private Certify artifacts."""

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
from .._json_admission import JsonAdmissionError
from .._model_admission import (
    MujocoClaimSurface,
    MujocoRuntimeAdmission,
    admit_external_implementation_free_model,
    compile_snapshot_model,
    require_supported_runtime,
)
from .._model_closure import (
    ModelAdmissionRefusal,
    ModelClosureSnapshot,
    ModelRole,
    create_model_closure_snapshot,
    verify_model_closure_unchanged,
)
from .._npz import ArtifactAdmissionRefusal
from .._npz import refuse as refuse_artifact
from .._owned_artifacts import RetainedArtifactPair
from .._runtime_identity import CertifyRuntimeIdentity, build_certify_runtime_identity
from ..certify._artifact import SerializedArtifact, serialize_complete_artifact
from ..certify._bytes import compare_retained_artifacts
from ..certify._entrypoint import ResolvedEntrypoint, resolve_entrypoint
from ..certify._fields import build_field_report
from ..certify._receipt import RoleCertification
from ..certify._receipt import build_receipt as build_certification_receipt
from ..certify._status import CertifyStatus
from ..compare._failure import ComparisonOperationError, operational_error
from ..distribution import installed_distribution_sha256
from ..json_values import CanonicalValue, canonical_json_bytes
from ..operational import (
    InputDigest,
    InputDigestCode,
    OperationalReasonCode,
    OperationalToolObservation,
)
from ..schemas import ModelClosureIdentity
from ..version import __version__
from ._decision import ModelReleaseDecisionRefusal, decide_model_release
from ._markdown import render_markdown
from ._policy import ModelReleasePolicy, load_model_release_policy
from ._receipt import ModelReleaseResult, build_model_release_receipt
from ._snapshot import SnapshotRefusal, build_compiled_model_snapshot

MODEL_RELEASE_OUTPUT_NAMES = PairedOutputNames("model_release.json", "model_release.md")


@dataclass(frozen=True, slots=True)
class _RoleArtifact:
    """One measured source closure and its private complete MJB."""

    closure: ModelClosureIdentity
    serialized: SerializedArtifact


class _RetainedSubjects:
    """Own every retained compiled subject admitted by one review.

    The descriptors held here are the only handles on the compiled artifacts this review decides
    from. They are reverified together at each boundary that could publish a result, and released
    exactly once when the review ends, on success or on failure.
    """

    __slots__ = ("_artifacts",)

    def __init__(self) -> None:
        """Start with no admitted subjects."""
        self._artifacts: list[_RoleArtifact] = []

    def adopt(self, artifact: _RoleArtifact) -> None:
        """Take ownership of one newly serialized role's retained subject."""
        self._artifacts.append(artifact)

    def verify(self) -> None:
        """Require every adopted subject to still hold its exact measured bytes."""
        for artifact in self._artifacts:
            artifact.serialized.verify()

    def close(self) -> None:
        """Release every retained descriptor, discarding the nameless compiled artifacts."""
        for artifact in self._artifacts:
            artifact.serialized.retained.close()
        self._artifacts.clear()


def review_model_release(
    baseline_mjcf: str,
    candidate_mjcf: str,
    policy_path: str,
    output_directory: str,
    *,
    baseline_root: str | None = None,
    candidate_root: str | None = None,
) -> ModelReleaseResult:
    """Classify every compiled-model change under one admitted maintainer policy."""
    tool = OperationalToolObservation(
        __version__, "VERIFIED_INSTALLED_DISTRIBUTION", installed_distribution_sha256()
    )
    published: PairedOutputDirectory | None = None
    policy: ModelReleasePolicy | None = None
    try:
        try:
            policy = load_model_release_policy(policy_path)
        except (JsonAdmissionError, TypeError, ValueError) as exc:
            raise _policy_failure(tool, exc) from exc
        runtime_admission = require_supported_runtime(MujocoClaimSurface.STATIC_MODEL_REVIEW)
        baseline_target = resolve_entrypoint(baseline_mjcf, baseline_root, "baseline")
        candidate_target = resolve_entrypoint(candidate_mjcf, candidate_root, "candidate")
        output_path = Path(output_directory)
        _require_output_outside_model_roots(
            output_path, (baseline_target.model_root, candidate_target.model_root)
        )
        published = prepare_paired_output_directory(output_path, MODEL_RELEASE_OUTPUT_NAMES)
        return _review(
            tool,
            policy,
            baseline_target,
            candidate_target,
            published,
            runtime_admission,
        )
    except ComparisonOperationError:
        cleanup_paired_output_after_failure(published)
        raise
    except (ModelAdmissionRefusal, ArtifactAdmissionRefusal) as exc:
        cleanup_paired_output_after_failure(published)
        raise _failure(tool, exc, policy) from exc
    except SnapshotRefusal as exc:
        cleanup_paired_output_after_failure(published)
        raise _snapshot_failure(tool, exc, policy) from exc
    except ModelReleaseDecisionRefusal as exc:
        cleanup_paired_output_after_failure(published)
        raise _decision_refusal(tool, exc, policy) from exc
    except BaseException:
        cleanup_paired_output_after_failure(published)
        raise


def _review(
    tool: OperationalToolObservation,
    policy: ModelReleasePolicy,
    baseline_target: ResolvedEntrypoint,
    candidate_target: ResolvedEntrypoint,
    published: PairedOutputDirectory,
    runtime_admission: MujocoRuntimeAdmission,
) -> ModelReleaseResult:
    """Compile, certify, classify, publish, and reverify one review."""
    scratch = Path(tempfile.mkdtemp(prefix="metrifid-model-release-"))
    retained: RetainedArtifactPair | None = None
    subjects = _RetainedSubjects()
    scratch_removed = False
    succeeded = False
    try:
        with ExitStack() as stack:
            stack.callback(subjects.close)
            baseline_snapshot = stack.enter_context(
                create_model_closure_snapshot(
                    baseline_target.model_root, baseline_target.entrypoint, "baseline"
                )
            )
            baseline = _compile_role(
                baseline_snapshot,
                "baseline",
                scratch,
                runtime_admission,
            )
            subjects.adopt(baseline)
            runtime = build_certify_runtime_identity(baseline.serialized.header_words)
            candidate_snapshot = stack.enter_context(
                create_model_closure_snapshot(
                    candidate_target.model_root, candidate_target.entrypoint, "candidate"
                )
            )
            candidate = _compile_role(
                candidate_snapshot,
                "candidate",
                scratch,
                runtime_admission,
            )
            subjects.adopt(candidate)
            _require_same_header(baseline, candidate)
            certification = _certification_receipt(tool, baseline, candidate, runtime)
            baseline_facts = build_compiled_model_snapshot(
                baseline.serialized.retained,
                "baseline",
                runtime_admission.package_base_version,
                runtime_admission.to_evidence(),
            )
            candidate_facts = build_compiled_model_snapshot(
                candidate.serialized.retained,
                "candidate",
                runtime_admission.package_base_version,
                runtime_admission.to_evidence(),
            )
            _require_policy_subject(policy, baseline.serialized.mjb_sha256)
            decision = decide_model_release(
                policy=policy,
                baseline=baseline_facts,
                candidate=candidate_facts,
                baseline_mjb_sha256=baseline.serialized.mjb_sha256,
                candidate_mjb_sha256=candidate.serialized.mjb_sha256,
            )
            receipt = build_model_release_receipt(
                policy=policy,
                decision=decision,
                certification_receipt=certification,
                registry_sha256=baseline_facts.registry_sha256,
                registry_count=len(baseline_facts.public_fields),
            )
            verify_model_closure_unchanged(baseline_snapshot, "baseline")
            verify_model_closure_unchanged(candidate_snapshot, "candidate")
            # Nothing may be published until both retained subjects still hold exactly the bytes
            # every fact above was derived from.
            subjects.verify()
            retained = publish_paired_results(
                published,
                json_bytes=canonical_json_bytes(receipt) + b"\n",
                markdown_text=render_markdown(receipt),
            )
            verify_model_closure_unchanged(baseline_snapshot, "baseline")
            verify_model_closure_unchanged(candidate_snapshot, "candidate")
            subjects.verify()
        assert retained is not None
        verify_paired_results(published, retained)
        _remove_private_scratch(scratch)
        scratch_removed = True
        verify_paired_results(published, retained)
        succeeded = True
        return ModelReleaseResult(
            decision.status,
            receipt,
            published.json_path,
            published.markdown_path,
        )
    finally:
        if retained is not None:
            if not succeeded:
                retained.cleanup()
            retained.close()
        if succeeded:
            published.close()
        if not scratch_removed:
            shutil.rmtree(scratch, ignore_errors=True)


def _remove_private_scratch(scratch: Path) -> None:
    """Require every private compiled artifact to be gone before reporting success."""
    shutil.rmtree(scratch)
    if scratch.exists():
        raise OSError("private model-release scratch directory survived cleanup")


def _compile_role(
    snapshot: ModelClosureSnapshot,
    role: ModelRole,
    scratch: Path,
    runtime_admission: MujocoRuntimeAdmission,
) -> _RoleArtifact:
    """Compile/admit/serialize one role, then release its live model."""
    model = compile_snapshot_model(snapshot, role)
    try:
        admit_external_implementation_free_model(model, role, runtime_admission)
        serialized = serialize_complete_artifact(model, role, scratch)
    finally:
        del model
    verify_model_closure_unchanged(snapshot, role)
    return _RoleArtifact(snapshot.identity, serialized)


def _certification_receipt(
    tool: OperationalToolObservation,
    baseline: _RoleArtifact,
    candidate: _RoleArtifact,
    runtime: CertifyRuntimeIdentity,
) -> dict[str, CanonicalValue]:
    """Build the linked Certify receipt from the exact same private MJB pair."""
    baseline.serialized.verify()
    candidate.serialized.verify()
    comparison = compare_retained_artifacts(
        baseline.serialized.retained, candidate.serialized.retained
    )
    baseline.serialized.verify()
    candidate.serialized.verify()
    status = (
        CertifyStatus.CERTIFIED_COMPILED_EQUIVALENCE
        if comparison.equal
        else CertifyStatus.NOT_CERTIFIED_COMPILED_DIFFERS
    )
    field_report = (
        None
        if comparison.equal
        else build_field_report(
            baseline.serialized.retained, candidate.serialized.retained, comparison
        )
    )
    return build_certification_receipt(
        status=status,
        tool={"name": "metrifid", **tool.to_primitive()},
        runtime=runtime,
        baseline=_role_certification(baseline, "baseline", runtime),
        candidate=_role_certification(candidate, "candidate", runtime),
        comparison=comparison,
        field_report=field_report,
    )


def _role_certification(
    artifact: _RoleArtifact, role: str, runtime: CertifyRuntimeIdentity
) -> RoleCertification:
    """Bind one role's source closure and MJB to the shared static runtime."""
    digest = runtime.runtime_identity_sha256
    if digest is None:
        raise ValueError("runtime identity must be finalized")
    return RoleCertification(role, artifact.closure, artifact.serialized.identity(digest))


def _require_same_header(baseline: _RoleArtifact, candidate: _RoleArtifact) -> None:
    """Refuse an impossible within-runtime complete-MJB header disagreement."""
    if baseline.serialized.header_words == candidate.serialized.header_words:
        return
    raise refuse_artifact(
        OperationalReasonCode.COMPILED_ARTIFACT_INVALID,
        "candidate",
        issue="header_words_differ_within_one_runtime",
        baseline_header_words=list(baseline.serialized.header_words),
        candidate_header_words=list(candidate.serialized.header_words),
    )


def _require_policy_subject(policy: ModelReleasePolicy, baseline_mjb_sha256: str) -> None:
    """Refuse a policy bound to a different baseline compiled artifact."""
    if policy.baseline_compiled_sha256 == baseline_mjb_sha256:
        return
    raise refuse_artifact(
        OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
        "comparison",
        issue="policy_baseline_compiled_sha256_mismatch",
        expected_baseline_compiled_sha256=policy.baseline_compiled_sha256,
        observed_baseline_compiled_sha256=baseline_mjb_sha256,
    )


def _canonical_output_path(path: Path) -> Path | None:
    """Resolve an existing output or its existing parent without creating it."""
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
    """Refuse output canonically equal to or below either admitted model root."""
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


def _available_policy_input(policy: ModelReleasePolicy | None) -> tuple[InputDigest, ...]:
    """Bind later operational refusals to the exact admitted policy bytes."""
    if policy is None:
        return ()
    return (InputDigest(InputDigestCode.CONFIGURATION_RAW, policy.raw_sha256),)


def _policy_failure(tool: OperationalToolObservation, exc: Exception) -> ComparisonOperationError:
    """Convert bounded policy admission failure into the existing exit-64 ABI."""
    return operational_error(
        tool=tool,
        code=OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
        role="comparison",
        field="policy",
        evidence={"exception_type": type(exc).__name__, "message": str(exc)},
        operation="review-model",
    )


def _failure(
    tool: OperationalToolObservation,
    exc: ModelAdmissionRefusal | ArtifactAdmissionRefusal,
    policy: ModelReleasePolicy | None,
) -> ComparisonOperationError:
    """Convert an expected model/artifact refusal into review-model evidence."""
    return operational_error(
        tool=tool,
        code=exc.reason,
        role=exc.role,
        evidence=cast("Mapping[str, CanonicalValue]", exc.evidence),
        available_inputs=_available_policy_input(policy),
        operation="review-model",
    )


def _snapshot_failure(
    tool: OperationalToolObservation,
    exc: SnapshotRefusal,
    policy: ModelReleasePolicy | None,
) -> ComparisonOperationError:
    """Convert a closed compiled-snapshot refusal into review-model evidence."""
    feature_coverage_issues = {
        "opaque_public_model_members",
        "uncharacterized_public_model_surface",
        "public_model_surface_mismatch",
        "public_field_registry_mismatch",
    }
    reason = (
        OperationalReasonCode.MUJOCO_FEATURE_COVERAGE_INCOMPLETE
        if exc.issue in feature_coverage_issues
        else OperationalReasonCode.COMPILED_ARTIFACT_INVALID
    )
    return operational_error(
        tool=tool,
        code=reason,
        role=cast("ModelRole", exc.role),
        evidence=exc.evidence,
        available_inputs=_available_policy_input(policy),
        operation="review-model",
    )


def _decision_refusal(
    tool: OperationalToolObservation,
    exc: ModelReleaseDecisionRefusal,
    policy: ModelReleasePolicy | None,
) -> ComparisonOperationError:
    """Convert a bounded decision-capacity refusal into the existing exit-64 ABI."""
    return operational_error(
        tool=tool,
        code=OperationalReasonCode.COMPILED_ARTIFACT_INVALID,
        role="comparison",
        evidence=exc.evidence,
        available_inputs=_available_policy_input(policy),
        operation="review-model",
    )


__all__ = ["MODEL_RELEASE_OUTPUT_NAMES", "review_model_release"]
