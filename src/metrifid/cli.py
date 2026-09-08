"""Installed CLI for certification, comparison, qualification, and runtime review."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn, TypeAlias

from .distribution import DistributionIdentityError
from .errors import OperationalExitCode, status_exit_code
from .json_values import CanonicalValue, canonical_json_bytes
from .operational import (
    OperationalFailure,
    OperationalReason,
    OperationalReasonCode,
    OperationalToolObservation,
)
from .version import __version__

_Encoder: TypeAlias = Callable[..., bytes]
_Renderer: TypeAlias = Callable[[Mapping[str, CanonicalValue]], str]
_RESULT_COMMANDS: Final = frozenset({"diff", "show"})


@dataclass(frozen=True, slots=True)
class _ResultRequest:
    """One recognized diff or show invocation and the output options it actually carried."""

    command: str
    as_json: bool
    full: bool


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
    diff = subcommands.add_parser(
        "diff",
        help="compare two supplied MJCF versions and retain the evidence",
        description=(
            "Compile two supplied MuJoCo model versions with no declared policy to satisfy, "
            "report every retained compiled-model change and every limit on that reading, and "
            "retain the evidence. Each model root is measured whole: every file beneath it is "
            "admitted, not only the named entrypoint. A root defaults to the directory holding "
            "the entrypoint; name a wider one explicitly when assets are shared beside it. "
            "Three files are retained: model_release.json, model_release.md and an offline "
            "report.html. Without --output they are kept in a new run under ~/.metrifid/runs. "
            "Exit 0 means the compiled models are byte-identical and 40 means they differ; both "
            "are completed comparisons, while 64 and 70 are refusals and failures."
        ),
    )
    diff.add_argument("baseline", metavar="BASELINE", help="baseline MJCF entrypoint")
    diff.add_argument("candidate", metavar="CANDIDATE", help="candidate MJCF entrypoint")
    diff.add_argument("--baseline-root", default=None, help="explicit baseline model root")
    diff.add_argument("--candidate-root", default=None, help="explicit candidate model root")
    diff.add_argument(
        "--output", default=None, help="output directory; a new retained run is used when absent"
    )
    _add_result_options(diff)
    show = subcommands.add_parser(
        "show",
        help="read one saved certification or model release receipt",
        description=(
            "Validate one saved Metrifid receipt and report what it recorded. Nothing is "
            "compiled, no output is written, and the recorded outcome is preserved. Reading a "
            "receipt needs no MuJoCo, so a retained result stays readable elsewhere. Exit 0 "
            "means the receipt was read, not that the two models were identical: the recorded "
            "status is reported in the result."
        ),
    )
    show.add_argument("receipt", metavar="RECEIPT", help="path to a saved Metrifid receipt")
    _add_result_options(show)
    return parser


def _add_result_options(parser: argparse.ArgumentParser) -> None:
    """Add the two output options shared by every command that emits a result document."""
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit one metrifid.result JSON document instead of the readable report",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="emit every retained finding, without the default output limit",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the installed CLI without exposing tracebacks as product evidence."""
    try:
        arguments = _parser().parse_args(None if argv is None else list(argv))
    except _InvocationError as exc:
        failure = _invocation_failure(str(exc), _OPERATIONS.get(_peek(argv), "compare"))
        # A recognized diff or show request that asked for JSON gets one result document even
        # when it never parsed, so a caller reading stdout can parse an ordinary usage error.
        _write_requested_failure_result(argv, failure)
        return _emit_failure(failure)
    if arguments.command not in _OPERATIONS:
        return _emit_failure(_invocation_failure("unknown command", "compare"))
    return _dispatch(arguments)


def _result_request(argv: Sequence[str] | None) -> _ResultRequest | None:
    """Return the diff or show output options this invocation actually asked for.

    Only tokens after the recognized subcommand are considered, and scanning stops at the
    end-of-options marker, so a file literally named ``--json`` never turns JSON output on.

    Args:
        argv: The argument vector, or None to read the process arguments.

    Returns:
        The command and its two output options, or None when this was not a diff or show request.
    """
    tokens = list(sys.argv[1:]) if argv is None else list(argv)
    if tokens and tokens[0] == "--":
        # Whether a leading end-of-options marker ever reaches the subcommand is argparse's
        # business, and it differs by interpreter: 3.12 and later consume one before the
        # subcommand, while 3.11 rejects it as an invalid command choice. Ask this interpreter's
        # own parser instead of inferring from a version number, so the scanner can never
        # recognize a command the parser did not. The probe parses one fixed complete vector and
        # nothing else happens: no command is dispatched and the placeholder is never opened.
        try:
            _parser().parse_args(["--", "show", "RECEIPT"])
        except _InvocationError:
            return None
        tokens = tokens[1:]
    if not tokens or tokens[0] not in _RESULT_COMMANDS:
        # Anything else in command position is not a diff or show request, whatever appears
        # later in the line, so nothing may be promised for it.
        return None
    command = tokens[0]
    supplied = _options_before_end_of_options(tokens[1:])
    declared = _long_options(command)
    return _ResultRequest(
        command,
        _requests(supplied, "--json", declared),
        _requests(supplied, "--full", declared),
    )


def _options_before_end_of_options(tokens: Sequence[str]) -> list[str]:
    """Collect the tokens that precede the end-of-options marker."""
    supplied: list[str] = []
    for token in tokens:
        if token == "--":
            break
        supplied.append(token)
    return supplied


def _long_options(command: str) -> frozenset[str]:
    """Return the long options one subcommand actually declares.

    Read from the parser itself so this stand-in cannot drift from the options it stands in for.
    """
    for action in _parser()._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        subcommand = action.choices.get(command)
        if subcommand is None:
            return frozenset()
        return frozenset(
            option
            for entry in subcommand._actions
            for option in entry.option_strings
            if option.startswith("--")
        )
    return frozenset()


def _requests(supplied: Sequence[str], option: str, declared: frozenset[str]) -> bool:
    """Whether one supplied token names that option, exactly or by unambiguous abbreviation.

    The parser accepts an unambiguous prefix, so a request that reached it as an abbreviation is
    still a request for that option here.
    """
    for token in supplied:
        if token == option:
            return True
        if not token.startswith("--") or not option.startswith(token):
            continue
        if sum(1 for candidate in declared if candidate.startswith(token)) == 1:
            return True
    return False


def _write_requested_failure_result(
    argv: Sequence[str] | None, failure: OperationalFailure
) -> None:
    """Write one bounded result document when a diff or show request asked for JSON.

    Nothing here can replace the operational failure the caller is about to receive, so a reader
    that cannot itself be built is dropped rather than reported.
    """
    request = _result_request(argv)
    if request is None or not request.as_json:
        return
    try:
        from .result import build_failure_document, encode_result

        reason = failure.reason
        evidence = reason.evidence
        document = build_failure_document(
            command=request.command,
            exit_code=int(failure.exit_code),
            reason_code=reason.code.value,
            message=str(evidence.get("message") or evidence.get("issue") or reason.code.value),
            reader_version=__version__,
            inputs={"baseline": None, "candidate": None, "roots_correspond": None},
            artifacts={
                "output_dir": None,
                "receipt": None,
                "receipt_sha256": None,
                "markdown": None,
                "html": None,
            },
            policy_origin="not_created" if request.command == "diff" else "not_applicable",
        )
        _write_stdout(encode_result(document, full=request.full))
    except Exception:  # defensive boundary; the operational failure below is still emitted
        return


def _dispatch(arguments: argparse.Namespace) -> int:
    """Route one parsed invocation to the command that owns it.

    Two commands are routed by name before the label-keyed routes below, because diff and show
    deliberately share the frozen review-model failure label with review-model itself.
    """
    operation = _OPERATIONS[arguments.command]
    if arguments.command == "diff":
        return _run_diff(arguments)
    if arguments.command == "show":
        return _run_show(arguments)
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
    return _run_compare(arguments.configuration, operation)


def _run_compare(configuration: str, operation: str) -> int:
    """Publish one declared-workload comparison, or emit its strict failure."""
    try:
        from .compare import ComparisonOperationError, compare_configuration_file

        result = compare_configuration_file(configuration)
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
    # diff runs the static model-review lifecycle and show reads what that lifecycle published,
    # so both bind their failures to the existing review-model label rather than adding one.
    "diff": "review-model",
    "show": "review-model",
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


def _run_diff(arguments: argparse.Namespace) -> int:
    """Publish one retained comparison of two supplied models, or emit its strict failure."""
    try:
        from .result import encode_result, render_text, run_diff
    except Exception as exc:  # defensive boundary
        return _emit_failure(_internal_failure(exc, "review-model"))
    try:
        outcome = run_diff(
            arguments.baseline,
            arguments.candidate,
            baseline_root=arguments.baseline_root,
            candidate_root=arguments.candidate_root,
            output_directory=arguments.output,
        )
    except DistributionIdentityError as exc:
        failure = exc.to_operational_failure("review-model")
        _write_requested_failure_result(_argv_of(arguments, "diff"), failure)
        return _emit_failure(failure)
    except Exception as exc:  # defensive boundary
        return _emit_failure(_internal_failure(exc, "review-model"))
    if outcome.failure is not None:
        # The refusal is the message. A report that cannot be rendered must not replace the
        # core reason and its exit code with an internal failure.
        _write_result_safely(arguments, outcome.document, encode_result, render_text)
        return _emit_failure(outcome.failure.failure)
    try:
        _write_result(arguments, outcome.document, encode_result, render_text)
    except Exception as exc:  # defensive boundary; a report must never escape as a traceback
        return _emit_failure(_internal_failure(exc, "review-model"))
    return outcome.exit_code


def _run_show(arguments: argparse.Namespace) -> int:
    """Report one saved receipt, or emit the strict failure that refused it."""
    try:
        from .compare._failure import ComparisonOperationError
        from .result import build_failure_document, encode_result, render_text, show_receipt
    except Exception as exc:  # defensive boundary
        return _emit_failure(_internal_failure(exc, "review-model"))
    try:
        document = show_receipt(arguments.receipt)
    except DistributionIdentityError as exc:
        failure = exc.to_operational_failure("review-model")
        _write_requested_failure_result(_argv_of(arguments, "show"), failure)
        return _emit_failure(failure)
    except ComparisonOperationError as exc:
        reason = exc.failure.reason
        evidence = reason.evidence
        _write_result_safely(
            arguments,
            build_failure_document(
                command="show",
                exit_code=int(exc.failure.exit_code),
                reason_code=reason.code.value,
                message=str(evidence.get("message") or evidence.get("issue") or reason.code.value),
                reader_version=__version__,
                inputs={"baseline": None, "candidate": None, "roots_correspond": None},
                artifacts={
                    "output_dir": None,
                    "receipt": None,
                    "receipt_sha256": None,
                    "markdown": None,
                    "html": None,
                },
                policy_origin="not_applicable",
            ),
            encode_result,
            render_text,
        )
        return _emit_failure(exc.failure)
    except Exception as exc:  # defensive boundary
        return _emit_failure(_internal_failure(exc, "review-model"))
    try:
        _write_result(arguments, document, encode_result, render_text)
    except Exception as exc:  # defensive boundary; a report must never escape as a traceback
        return _emit_failure(_internal_failure(exc, "review-model"))
    return 0


def _argv_of(arguments: argparse.Namespace, command: str) -> list[str]:
    """Rebuild the output options an already-parsed request carried, for the shared writer."""
    tokens = [command]
    if arguments.as_json:
        tokens.append("--json")
    if arguments.full:
        tokens.append("--full")
    return tokens


def _write_stdout(payload: bytes) -> None:
    """Write one complete result to stdout, tolerating a reader that stopped listening.

    The write is flushed here so a closed pipe is refused while it can still be handled. Left to
    interpreter shutdown it would surface as an ignored exception, replace the command's real exit
    code, and append non-JSON text after the canonical failure on stderr. Nothing is promised once
    stdout is gone, so the stream is detached and the command's own outcome stands.
    """
    try:
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()
    except OSError:
        try:
            sys.stdout = Path(os.devnull).open("w")
        except OSError:
            return


def _write_result_safely(
    arguments: argparse.Namespace,
    document: dict[str, CanonicalValue],
    encode_result: _Encoder,
    render_text: _Renderer,
) -> None:
    """Write one result document, discarding a report that cannot itself be rendered.

    The refusal already on its way to the caller is the important message; failing to render the
    accompanying report must not replace it with a traceback.
    """
    try:
        _write_result(arguments, document, encode_result, render_text)
    except Exception:  # defensive boundary; the operational failure below is still emitted
        return


def _write_result(
    arguments: argparse.Namespace,
    document: dict[str, CanonicalValue],
    encode_result: _Encoder,
    render_text: _Renderer,
) -> None:
    """Write one result document to stdout, as bounded JSON or as the readable report."""
    if arguments.as_json:
        _write_stdout(encode_result(document, full=arguments.full))
        return
    _write_stdout(render_text(document).encode("utf-8", errors="strict"))


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
