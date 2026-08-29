"""Installed CLI for certification, comparison, qualification, and runtime review."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import NoReturn

from .distribution import DistributionIdentityError
from .errors import OperationalExitCode, status_exit_code
from .json_values import canonical_json_bytes
from .operational import (
    OperationalFailure,
    OperationalReason,
    OperationalReasonCode,
    OperationalToolObservation,
)
from .version import __version__


class _InvocationError(ValueError):
    """Carry one controlled CLI parse failure without argparse process exit."""

    pass


class _Parser(argparse.ArgumentParser):
    """Route argparse validation failures through Metrifid's operational-failure ABI."""

    def error(self, message: str) -> NoReturn:
        """Convert argparse failures into the CLI's controlled invocation exception."""
        raise _InvocationError(message)


def _parser() -> argparse.ArgumentParser:
    """Build the installed command tree without importing native-backed product modules."""
    parser = _Parser(
        prog="metrifid",
        # The description is emitted verbatim. Argparse's default formatter rewraps it to the
        # terminal width, which splits the product sentence and makes the installed root help
        # unsearchable for the exact phrase users and release checks look for.
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Verify compiled MuJoCo models, review static model changes, compare declared "
            "workloads, audit timesteps, and qualify whether declared workloads detect declared "
            "model perturbations. Create or review a complete native-runtime migration evidence "
            "set through explicit prepared profiles."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    compare = subcommands.add_parser(
        "compare",
        help="run one baseline/candidate comparison from comparison.json",
        description="Run one strict metrifid ComparisonConfig JSON file.",
    )
    compare.add_argument("configuration", help="path to comparison.json")
    audit = subcommands.add_parser(
        "audit-timestep",
        help="audit candidate timesteps for one model and declared workload",
        description=(
            "Evaluate every declared candidate timestep against one compiled reference "
            "model and recommend the largest candidate supported by an unbroken "
            "within-tolerance completed prefix."
        ),
    )
    audit.add_argument("configuration", help="path to timestep_audit.json")
    certify = subcommands.add_parser(
        "certify",
        help="certify whether two MJCF closures compile to byte-identical MJB artifacts",
        description=(
            "Compile two source closures with the exact admitted MuJoCo runtime and state, over "
            "every serialized byte, whether they produce identical complete MJB artifacts."
        ),
    )
    certify.add_argument("baseline_mjcf", help="path to the baseline MJCF entrypoint")
    certify.add_argument("candidate_mjcf", help="path to the candidate MJCF entrypoint")
    certify.add_argument("--output", required=True, help="output directory to publish into")
    certify.add_argument("--baseline-root", default=None, help="explicit baseline model root")
    certify.add_argument("--candidate-root", default=None, help="explicit candidate model root")
    review = subcommands.add_parser(
        "review-model",
        help="review compiled model changes against a declared release policy",
        description=(
            "Compile two MuJoCo model sources and classify their typed static changes "
            "against one strict model release policy."
        ),
    )
    review.add_argument("baseline", metavar="BASELINE", help="baseline MJCF entrypoint")
    review.add_argument("candidate", metavar="CANDIDATE", help="candidate MJCF entrypoint")
    review.add_argument("--policy", required=True, help="strict model release policy JSON")
    review.add_argument("--output", required=True, help="output directory to publish into")
    review.add_argument("--baseline-root", default=None, help="explicit baseline model root")
    review.add_argument("--candidate-root", default=None, help="explicit candidate model root")
    qualify = subcommands.add_parser(
        "qualify-workload",
        help="qualify whether declared workloads detect declared model perturbations",
        description=(
            "Run every declared probe comparison, select the best three-workload subset by "
            "exact enumeration, and state which declared perturbations remain blind."
        ),
    )
    qualify.add_argument("configuration", help="path to qualification.json")
    runtime_review = subcommands.add_parser(
        "review-runtime",
        help="review one exact native-runtime migration from retained three-grid evidence",
        description=(
            "Evaluate one strict runtime_review.json over twelve retained evidence cells and "
            "publish a full-horizon native-runtime replacement decision."
        ),
    )
    runtime_review.add_argument("configuration", help="path to runtime_review.json")
    runtime_review_run = subcommands.add_parser(
        "run-runtime-review",
        help="create twelve native evidence cells and immediately run Runtime Review",
        description=(
            "Use two already-prepared explicit Python profiles to run two native identity "
            "preflights and twelve sequential evidence cells, then call the existing Runtime "
            "Review evaluator. Metrifid never discovers, creates, or installs an environment."
        ),
    )
    runtime_review_run.add_argument("configuration", help="path to runtime_review_run.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the installed CLI without exposing tracebacks as product evidence."""
    try:
        arguments = _parser().parse_args(None if argv is None else list(argv))
    except _InvocationError as exc:
        return _emit_failure(_invocation_failure(str(exc), _OPERATIONS.get(_peek(argv), "compare")))
    if arguments.command not in _OPERATIONS:
        return _emit_failure(_invocation_failure("unknown command", "compare"))
    operation = _OPERATIONS[arguments.command]
    if operation == "audit-timestep":
        return _run_audit(arguments.configuration)
    if operation == "certify":
        return _run_certify(arguments)
    if operation == "review-model":
        return _run_review_model(arguments)
    if operation == "qualify-workload":
        return _run_qualify_workload(arguments.configuration)
    if arguments.command == "review-runtime":
        return _run_review_runtime(arguments.configuration)
    if arguments.command == "run-runtime-review":
        return _run_runtime_review_execution(arguments.configuration)
    try:
        from .compare import ComparisonOperationError, compare_configuration_file

        result = compare_configuration_file(arguments.configuration)
    except DistributionIdentityError as exc:
        return _emit_failure(exc.to_operational_failure("compare"))
    except ComparisonOperationError as exc:
        return _emit_failure(exc.failure)
    except Exception as exc:  # defensive boundary; detailed failures should be produced internally
        return _emit_failure(_internal_failure(exc, operation))
    sys.stdout.buffer.write(
        canonical_json_bytes(
            {
                "status": result.receipt.status.value,
                "receipt_sha256": result.receipt.receipt_sha256,
                "comparison_json": str(result.comparison_json),
                "comparison_markdown": str(result.comparison_markdown),
            }
        )
        + b"\n"
    )
    return int(status_exit_code(result.receipt.status))


_OPERATIONS: dict[str, str] = {
    "compare": "compare",
    "audit-timestep": "audit-timestep",
    "certify": "certify",
    "review-model": "review-model",
    "qualify-workload": "qualify-workload",
    # The operational-failure registry is deliberately frozen. Runtime Review is data-only and
    # reuses the existing bounded comparison failure ABI while keeping its own completed statuses.
    "review-runtime": "compare",
    "run-runtime-review": "compare",
}


def _peek(argv: Sequence[str] | None) -> str:
    """Best-effort subcommand for an invocation that never reached the parser."""
    import sys as _sys

    tokens = list(_sys.argv[1:]) if argv is None else list(argv)
    for token in tokens:
        if token in _OPERATIONS:
            return token
    return "compare"


def _run_audit(configuration: str) -> int:
    """Publish the audit surface, or emit one strict audit-timestep failure."""
    try:
        from .compare import ComparisonOperationError
        from .timestep_audit import AuditAbort, audit_configuration_file

        result = audit_configuration_file(configuration)
    except DistributionIdentityError as exc:
        return _emit_failure(exc.to_operational_failure("audit-timestep"))
    except AuditAbort as exc:
        return _emit_failure(exc.error.failure)
    except ComparisonOperationError as exc:
        return _emit_failure(exc.failure)
    except Exception as exc:  # defensive boundary
        return _emit_failure(_internal_failure(exc, "audit-timestep"))
    recommendation = result.aggregate["recommendation"]
    token = recommendation["candidate_token"] if isinstance(recommendation, dict) else None
    sys.stdout.buffer.write(
        canonical_json_bytes(
            {
                "audit_sha256": result.aggregate["audit_sha256"],
                "recommended_candidate_token": token,
                "timestep_audit_json": str(result.audit_json),
                "timestep_audit_markdown": str(result.audit_markdown),
            }
        )
        + b"\n"
    )
    return 0


def _run_certify(arguments: argparse.Namespace) -> int:
    """Publish one certification, or emit one strict certify failure."""
    try:
        # Imported before the run so the handlers below can name the certify error type.
        from .certify import CertifyOperationError, certify_exit_code, certify_models
    except Exception as exc:  # defensive boundary
        return _emit_failure(_internal_failure(exc, "certify"))
    try:
        result = certify_models(
            arguments.baseline_mjcf,
            arguments.candidate_mjcf,
            arguments.output,
            baseline_root=arguments.baseline_root,
            candidate_root=arguments.candidate_root,
        )
    except DistributionIdentityError as exc:
        return _emit_failure(exc.to_operational_failure("certify"))
    except CertifyOperationError as exc:
        return _emit_failure(exc.failure)
    except Exception as exc:  # defensive boundary
        return _emit_failure(_internal_failure(exc, "certify"))
    sys.stdout.buffer.write(
        canonical_json_bytes(
            {
                "status": result.status.value,
                "receipt_sha256": result.receipt_sha256,
                "certification_json": str(result.certification_json),
                "certification_markdown": str(result.certification_markdown),
            }
        )
        + b"\n"
    )
    return certify_exit_code(result.status)


def _run_review_model(arguments: argparse.Namespace) -> int:
    """Publish one static model release review, or emit its strict failure."""
    try:
        from .model_release import (
            ModelReleaseOperationError,
            model_release_exit_code,
            review_model_release,
        )
    except Exception as exc:  # defensive boundary
        return _emit_failure(_internal_failure(exc, "review-model"))
    try:
        result = review_model_release(
            arguments.baseline,
            arguments.candidate,
            arguments.policy,
            arguments.output,
            baseline_root=arguments.baseline_root,
            candidate_root=arguments.candidate_root,
        )
    except DistributionIdentityError as exc:
        return _emit_failure(exc.to_operational_failure("review-model"))
    except ModelReleaseOperationError as exc:
        return _emit_failure(exc.failure)
    except Exception as exc:  # defensive boundary
        return _emit_failure(_internal_failure(exc, "review-model"))
    sys.stdout.buffer.write(
        canonical_json_bytes(
            {
                "status": result.status.value,
                "receipt_sha256": result.receipt_sha256,
                "model_release_json": result.model_release_json.name,
                "model_release_markdown": result.model_release_markdown.name,
            }
        )
        + b"\n"
    )
    return model_release_exit_code(result.status)


def _run_qualify_workload(configuration: str) -> int:
    """Publish one workload qualification, or emit its strict operational failure."""
    try:
        from .compare import ComparisonOperationError
        from .workload_qualification import qualify_configuration_file

        result = qualify_configuration_file(configuration)
    except DistributionIdentityError as exc:
        return _emit_failure(exc.to_operational_failure("qualify-workload"))
    except ComparisonOperationError as exc:
        return _emit_failure(exc.failure)
    except Exception as exc:  # defensive boundary
        return _emit_failure(_internal_failure(exc, "qualify-workload"))
    sys.stdout.buffer.write(
        canonical_json_bytes(
            {
                "status": result.status.value,
                "receipt_sha256": result.receipt_sha256,
                "workload_qualification_json": str(result.qualification_json),
                "workload_qualification_markdown": str(result.qualification_markdown),
            }
        )
        + b"\n"
    )
    return result.exit_code


def _run_review_runtime(configuration: str) -> int:
    """Publish one full-horizon runtime review or its bounded comparison failure."""
    try:
        from .runtime_review import (
            RuntimeReviewOperationError,
            review_runtime_configuration_file,
        )

        result = review_runtime_configuration_file(configuration)
    except DistributionIdentityError as exc:
        return _emit_failure(exc.to_operational_failure("compare"))
    except RuntimeReviewOperationError as exc:
        return _emit_failure(exc.failure)
    except Exception as exc:  # defensive boundary
        return _emit_failure(_internal_failure(exc, "compare"))
    sys.stdout.buffer.write(
        canonical_json_bytes(
            {
                "receipt_sha256": result.receipt_sha256,
                "runtime_review_json": result.runtime_review_json.name,
                "runtime_review_markdown": result.runtime_review_markdown.name,
                "status": result.status.value,
            }
        )
        + b"\n"
    )
    return result.exit_code


def _run_runtime_review_execution(configuration: str) -> int:
    """Create native evidence and publish its existing Runtime Review decision."""
    try:
        from .runtime_review import (
            RuntimeReviewOperationError,
            run_runtime_review_configuration_file,
        )

        result = run_runtime_review_configuration_file(configuration)
    except DistributionIdentityError as exc:
        return _emit_failure(exc.to_operational_failure("compare"))
    except RuntimeReviewOperationError as exc:
        return _emit_failure(exc.failure)
    except Exception as exc:  # defensive boundary
        return _emit_failure(_internal_failure(exc, "compare"))
    sys.stdout.buffer.write(
        canonical_json_bytes(
            {
                "reason_code": (None if result.reason_code is None else result.reason_code.value),
                "receipt_sha256": result.receipt_sha256,
                "run_sha256": result.run_sha256,
                "runtime_review_json": str(result.runtime_review_json),
                "runtime_review_markdown": str(result.runtime_review_markdown),
                "runtime_review_run_json": str(result.runtime_review_run_json),
                "status": result.status.value,
            }
        )
        + b"\n"
    )
    return result.exit_code


def _invocation_failure(message: str, operation: str = "compare") -> OperationalFailure:
    """Build an unbound pre-contract failure for invalid command-line input."""
    return OperationalFailure(
        schema="metrifid.operational_failure",
        schema_version=1,
        failure_rule_schema="metrifid.operational_failure_rules",
        failure_rule_schema_version=1,
        tool=OperationalToolObservation(__version__, "UNBOUND", None),
        operation=operation,
        stage=OperationalReasonCode.INVALID_CLI_INVOCATION.stage,
        reason=OperationalReason(
            code=OperationalReasonCode.INVALID_CLI_INVOCATION,
            role=None,
            field="argv",
            object_name=None,
            evidence={"message": message},
        ),
        available_inputs=(),
        environment=None,
        exit_code=OperationalExitCode.INVALID_INVOCATION_INPUT_OUTPUT,
        failure_sha256=None,
    ).finalized()


def _internal_failure(exc: Exception, operation: str = "compare") -> OperationalFailure:
    """Build a fail-closed internal-error artifact without exposing a traceback."""
    return OperationalFailure(
        schema="metrifid.operational_failure",
        schema_version=1,
        failure_rule_schema="metrifid.operational_failure_rules",
        failure_rule_schema_version=1,
        tool=OperationalToolObservation(__version__, "UNBOUND", None),
        operation=operation,
        stage=OperationalReasonCode.INTERNAL_INVARIANT_FAILED.stage,
        reason=OperationalReason(
            code=OperationalReasonCode.INTERNAL_INVARIANT_FAILED,
            role=None,
            field=None,
            object_name=None,
            evidence={"exception_type": type(exc).__name__, "message": str(exc)},
        ),
        available_inputs=(),
        environment=None,
        exit_code=OperationalExitCode.INTERNAL_PROJECT_FAILURE,
        failure_sha256=None,
    ).finalized()


def _emit_failure(failure: OperationalFailure) -> int:
    """Write one canonical operational failure to stderr and return its exit code."""
    sys.stderr.buffer.write(canonical_json_bytes(failure.to_primitive()) + b"\n")
    return int(failure.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
