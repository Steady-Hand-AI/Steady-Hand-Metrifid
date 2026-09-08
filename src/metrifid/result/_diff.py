"""Compare two supplied MJCF versions and retain the evidence.

``diff`` runs the existing review lifecycle with no policy to satisfy, so the question it answers
is what changed rather than whether the change was allowed.  Nothing here decides anything: the
compiled comparison, the canonical change rows and the published receipt are all the lifecycle's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from .._model_closure import ModelAdmissionRefusal, ModelRole
from ..certify._entrypoint import ResolvedEntrypoint, resolve_entrypoint
from ..compare._failure import ComparisonOperationError, operational_error
from ..distribution import installed_distribution_sha256
from ..json_values import CanonicalValue
from ..operational import OperationalReasonCode, OperationalToolObservation
from ..version import __version__
from ._contract import (
    OBSERVATION_PRODUCED,
    POLICY_GENERATED_EMPTY,
    POLICY_NOT_CREATED,
    build_document,
    build_failure_document,
    closure_identity,
    compiled_comparison,
)
from ._findings import PRESENTATION
from ._runs import allocate_run_directory, require_admissible_model_root

if TYPE_CHECKING:
    from .._atomic_output import PairedOutputDirectory
    from .._npz import ArtifactAdmissionRefusal
    from ..model_release._receipt import ModelReleaseResult

__all__ = ["DIFF_OPERATION", "DiffOutcome", "run_diff"]

DIFF_OPERATION: Final = "review-model"
_IDENTICAL_EXIT: Final = 0
_DIFFERENT_EXIT: Final = 40
_REPORT_NAME: Final = "report.html"
_REPORT_PREFIX: Final = "report"


@dataclass(frozen=True, slots=True)
class DiffOutcome:
    """One completed or failed comparison, ready for the caller to emit."""

    document: dict[str, CanonicalValue]
    exit_code: int
    failure: ComparisonOperationError | None


def run_diff(
    baseline_mjcf: str,
    candidate_mjcf: str,
    *,
    baseline_root: str | None = None,
    candidate_root: str | None = None,
    output_directory: str | None = None,
) -> DiffOutcome:
    """Compare two supplied models and return the result document and exit code.

    A completed comparison exits 0 when the compiled artifacts are byte-identical and 40 when they
    differ.  A failure keeps the core's own reason and exit code, reports the comparison as not
    established, and leaves every committed file untouched.

    Args:
        baseline_mjcf: Path to the baseline MJCF entrypoint.
        candidate_mjcf: Path to the candidate MJCF entrypoint.
        baseline_root: Explicit baseline model root, for a model with shared assets beside it.
        candidate_root: Explicit candidate model root.
        output_directory: Explicit output directory, or None to retain a new run.

    Returns:
        The result document, the exit code, and the failure when one occurred.
    """
    run_directory: Path | None = None
    output: PairedOutputDirectory | None = None
    try:
        baseline_target = _resolved(baseline_mjcf, baseline_root, "baseline")
        candidate_target = _resolved(candidate_mjcf, candidate_root, "candidate")
        roots = (baseline_target.model_root, candidate_target.model_root)
        output = _output(output_directory, roots)
        run_directory = output.path
        result = _review(
            baseline_mjcf,
            candidate_mjcf,
            output,
            baseline_root=baseline_root,
            candidate_root=candidate_root,
        )
    except ComparisonOperationError as failure:
        # The run directory is left exactly as the core left it, and is still reported because
        # it is already known. Nothing here inspects or removes anything on the way out.
        if output is not None:
            output.close()
        return _failed(failure, run_directory)
    try:
        certification = result.receipt.get("certification_receipt")
        comparison = compiled_comparison(certification if isinstance(certification, dict) else None)
        exit_code = _IDENTICAL_EXIT if comparison["equal"] is True else _DIFFERENT_EXIT
        document = build_document(
            command="diff",
            exit_code=exit_code,
            observation=OBSERVATION_PRODUCED,
            receipt=result.receipt,
            policy_origin=POLICY_GENERATED_EMPTY,
            reader_version=__version__,
            inputs=_live_inputs(certification, baseline_target, candidate_target),
            artifacts={
                "output_dir": str(run_directory),
                "receipt": str(result.model_release_json),
                "receipt_sha256": result.receipt_sha256,
                "markdown": str(result.model_release_markdown),
                "html": None,
            },
        )
        _attach_report(document, output)
    finally:
        output.close()
    return DiffOutcome(document, exit_code, None)


def _attach_report(document: dict[str, CanonicalValue], output: PairedOutputDirectory) -> None:
    """Publish the offline report beside the canonical pair, or record why there is none.

    The canonical files have already been published and verified when this runs. A report that
    cannot be rendered or written is a presentation problem and nothing more: the comparison keeps
    its completed exit code, the committed evidence is untouched, and the artifact stays null so
    nothing claims a report that does not exist.

    Args:
        document: The completed result document, updated in place.
        output: The bound directory the canonical pair was published into.
    """
    artifacts = document["artifacts"]
    limitations = document["limitations"]
    if not isinstance(artifacts, dict) or not isinstance(limitations, list):
        return
    try:
        artifacts["html"] = str(_publish_report(document, output))
    except Exception as exc:  # presentation only; a completed comparison keeps its result
        limitations.append(
            {
                "category": PRESENTATION,
                # Two shapes reach here and the message must be true of both: the report could
                # not be rendered or written at all, or it was written and the output pathname
                # stopped naming the directory it went into. Claiming it "was not written" would
                # be false in the second case, so say only what is certain either way.
                "message": (
                    "No offline report is available at a known path for this run, so the "
                    "canonical receipt and Markdown are its complete record."
                ),
                "source": "",
                "detail": {"exception_type": type(exc).__name__},
            }
        )


def _review(
    baseline_mjcf: str,
    candidate_mjcf: str,
    output: str | PairedOutputDirectory,
    *,
    baseline_root: str | None,
    candidate_root: str | None,
) -> ModelReleaseResult:
    """Run the existing review lifecycle with a generated empty policy."""
    from ..model_release._run import discover_model_release

    return discover_model_release(
        baseline_mjcf,
        candidate_mjcf,
        output,
        baseline_root=baseline_root,
        candidate_root=candidate_root,
    )


def _resolved(mjcf: str, root: str | None, role: ModelRole) -> ResolvedEntrypoint:
    """Resolve one role's root and entrypoint, refusing a root too broad to measure."""
    try:
        target = resolve_entrypoint(mjcf, root, role)
        require_admissible_model_root(target.model_root, role)
    except ModelAdmissionRefusal as exc:
        raise _refusal(exc) from exc
    return target


def _output(output_directory: str | None, roots: tuple[Path, Path]) -> PairedOutputDirectory:
    """Bind the directory this comparison publishes into, whichever route chose it.

    Both routes hand back one already-bound output, so the report is written into the same object
    the canonical pair was published into rather than through a second pathname lookup. An explicit
    directory keeps its existing checks: it must lie outside both model roots and be absent or
    empty under a real parent.

    Args:
        output_directory: The caller's explicit directory, or None to retain a new run.
        roots: The resolved baseline and candidate model roots.

    Returns:
        The bound output directory, which the caller closes.

    Raises:
        ComparisonOperationError: The directory was refused, or no run could be allocated.
    """
    # The output primitives raise the artifact refusal type; importing them here keeps reading a
    # saved receipt free of the native-backed modules that define them.
    from .._atomic_output import prepare_paired_output_directory
    from .._npz import ArtifactAdmissionRefusal
    from ..model_release._run import MODEL_RELEASE_OUTPUT_NAMES, require_output_outside_model_roots

    try:
        if output_directory is None:
            return allocate_run_directory(roots)
        explicit = Path(output_directory).absolute()
        require_output_outside_model_roots(explicit, roots)
        return prepare_paired_output_directory(explicit, MODEL_RELEASE_OUTPUT_NAMES)
    except (ArtifactAdmissionRefusal, ModelAdmissionRefusal) as exc:
        raise _refusal(exc) from exc


def _refusal(exc: ArtifactAdmissionRefusal | ModelAdmissionRefusal) -> ComparisonOperationError:
    """Convert one path or output refusal into the existing operational-failure ABI."""
    return operational_error(
        tool=OperationalToolObservation(
            __version__, "VERIFIED_INSTALLED_DISTRIBUTION", installed_distribution_sha256()
        ),
        code=exc.reason,
        role=exc.role,
        evidence=cast("dict[str, CanonicalValue]", exc.evidence),
        operation=DIFF_OPERATION,
    )


_CERTIFY_SUGGESTION: Final = (
    " This runtime has no characterized named-field surface, so diff refuses rather than "
    "reporting a partial one. `metrifid certify BASELINE CANDIDATE --output DIR` still "
    "establishes whether the two models compile to byte-identical artifacts on this runtime."
)


def _failed(failure: ComparisonOperationError, run_directory: Path | None) -> DiffOutcome:
    """Build the result document for a comparison that returned no completed receipt.

    An uncatalogued named-field runtime keeps its existing refusal and exit code, and is told
    about certify in the same breath. Nothing falls back automatically.
    """
    reason = failure.failure.reason
    evidence = reason.evidence
    message = str(evidence.get("message") or evidence.get("issue") or reason.code.value)
    if reason.code is OperationalReasonCode.MUJOCO_FEATURE_COVERAGE_INCOMPLETE:
        message = f"{message}.{_CERTIFY_SUGGESTION}"
    document = build_failure_document(
        command="diff",
        exit_code=int(failure.failure.exit_code),
        reason_code=reason.code.value,
        message=message,
        reader_version=__version__,
        inputs={"baseline": None, "candidate": None, "roots_correspond": None},
        artifacts={
            "output_dir": None if run_directory is None else str(run_directory),
            "receipt": None,
            "receipt_sha256": None,
            "markdown": None,
            "html": None,
        },
        policy_origin=POLICY_NOT_CREATED,
    )
    return DiffOutcome(document, int(failure.failure.exit_code), failure)


def _publish_report(document: dict[str, CanonicalValue], output: PairedOutputDirectory) -> Path:
    """Write the offline report into the bound output through the owned-artifact publisher.

    The report is written to a private temporary inside the same bound directory, sealed, then
    linked to its final name. Linking refuses an existing name, so a report already there is never
    overwritten and no pathname is resolved a second time.

    Args:
        document: The completed result document to render.
        output: The bound directory the canonical pair was published into.

    Returns:
        The published report path.
    """
    from .._atomic_output import verify_paired_output_path_unchanged
    from .._owned_artifacts import commit_owned_artifact, create_owned_artifact, write_owned_bytes
    from ._report import render_report

    payload = render_report(document).encode("utf-8", errors="strict")
    # Rendering is the one unbounded step between publishing the canonical pair and writing the
    # report. Re-establish that the public pathname still names the bound directory before adding
    # anything to it, so a directory replaced meanwhile cannot receive a file this run announces.
    verify_paired_output_path_unchanged(output)
    artifact = create_owned_artifact(output.directory_fd, _REPORT_PREFIX)
    try:
        try:
            write_owned_bytes(artifact, payload)
        except BaseException:
            artifact.cleanup()
            raise
        # Committing links the final name, which refuses one that already exists, and removes
        # only the private temporary on failure. A published report is never overwritten.
        commit_owned_artifact(artifact, _REPORT_NAME)
    finally:
        artifact.close()
    # The returned path is a public claim about where this report can be read. Check the binding
    # once more so a directory replaced during publication is never advertised as its location.
    # A report already committed through the descriptor stays where it was written; only the claim
    # is withheld, and the caller records that as a presentation limitation.
    verify_paired_output_path_unchanged(output)
    return output.path / _REPORT_NAME


def _live_inputs(
    certification: CanonicalValue,
    baseline_target: ResolvedEntrypoint,
    candidate_target: ResolvedEntrypoint,
) -> dict[str, CanonicalValue]:
    """Report each role's live path and root beside the identity the receipt retained."""
    roles = {"baseline": baseline_target, "candidate": candidate_target}
    block = certification if isinstance(certification, dict) else {}
    inputs: dict[str, CanonicalValue] = {
        "roots_correspond": baseline_target.model_root == candidate_target.model_root
    }
    for role, target in roles.items():
        recorded = block.get(role)
        identity = closure_identity(recorded) if isinstance(recorded, dict) else {}
        identity["path"] = str(target.model_root.joinpath(*Path(target.entrypoint).parts))
        identity["root"] = str(target.model_root)
        inputs[role] = identity
    return inputs
