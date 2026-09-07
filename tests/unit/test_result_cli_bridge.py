"""The installed CLI surface for the two result-shaped commands, ``diff`` and ``show``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import metrifid
from metrifid import cli
from metrifid._operational_registry import _OPERATIONS as REGISTERED_OPERATIONS
from metrifid.compare._failure import ComparisonOperationError, operational_error
from metrifid.json_values import CanonicalValue
from metrifid.operational import (
    OperationalFailure,
    OperationalReasonCode,
    OperationalToolObservation,
)
from metrifid.result import build_document, encode_result, render_text
from metrifid.result import build_failure_document as real_build_failure_document
from metrifid.version import __version__

_VERIFIED_TOOL = OperationalToolObservation(
    __version__, "VERIFIED_INSTALLED_DISTRIBUTION", "c" * 64
)


class _FakeOutcome:
    """Carry the three members of DiffOutcome that the CLI actually reads."""

    def __init__(
        self,
        document: dict[str, CanonicalValue],
        exit_code: int,
        failure: ComparisonOperationError | None = None,
    ) -> None:
        """Retain one document, its exit code, and the failure that produced it."""
        self.document = document
        self.exit_code = exit_code
        self.failure = failure


def _install_fake_result(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_diff: Any = None,
    show_receipt: Any = None,
) -> None:
    """Install one fake metrifid.result exposing exactly the surface the CLI imports."""
    module = ModuleType("metrifid.result")
    module.run_diff = run_diff  # type: ignore[attr-defined]
    module.show_receipt = show_receipt  # type: ignore[attr-defined]
    module.encode_result = encode_result  # type: ignore[attr-defined]
    module.render_text = render_text  # type: ignore[attr-defined]
    module.build_failure_document = real_build_failure_document  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metrifid.result", module)
    monkeypatch.setattr(metrifid, "result", module, raising=False)


def _install_recording_model_release(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Install a metrifid.model_release whose runner records a call it must never receive."""
    calls: list[object] = []

    def review(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        raise AssertionError("the review-model runner must not own diff or show")

    module = ModuleType("metrifid.model_release")
    module.ModelReleaseOperationError = ComparisonOperationError  # type: ignore[attr-defined]
    module.model_release_exit_code = lambda _status: 0  # type: ignore[attr-defined]
    module.review_model_release = review  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metrifid.model_release", module)
    monkeypatch.setattr(metrifid, "model_release", module, raising=False)
    return calls


def _document(*, equal: bool, exit_code: int) -> dict[str, CanonicalValue]:
    """Build one real result document over a minimal model-release-shaped receipt."""
    receipt: dict[str, CanonicalValue] = {
        "schema": "metrifid.model_release_receipt",
        "status": "NO_COMPILED_CHANGE" if equal else "REVIEW_REQUIRED",
        "completed_exit_code": exit_code,
        "receipt_sha256": "b" * 64,
        "changes": [],
        "certification_receipt": {
            "byte_comparison": {
                "equal": equal,
                "first_differing_byte_offset": None if equal else 128,
                "differing_byte_count": 0 if equal else 4,
            }
        },
    }
    return build_document(
        command="diff",
        exit_code=exit_code,
        observation="produced_now",
        receipt=receipt,
        policy_origin="generated_empty",
        reader_version=__version__,
        inputs={"baseline": None, "candidate": None, "roots_correspond": None},
        artifacts={
            "output_dir": None,
            "receipt": None,
            "receipt_sha256": None,
            "markdown": None,
            "html": None,
        },
    )


def _parse_failure(stderr: bytes) -> dict[str, object]:
    """Parse one canonical operational failure emitted on standard error."""
    value = json.loads(stderr)
    assert isinstance(value, dict)
    return value


def _controlled_show_failure() -> ComparisonOperationError:
    """Build one real controlled reader failure of the kind show_receipt raises."""
    return operational_error(
        tool=_VERIFIED_TOOL,
        code=OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
        role="comparison",
        field="receipt",
        evidence={"message": "receipt is not valid JSON"},
        operation="review-model",
    )


def test_the_diff_parser_freezes_its_defaults_when_only_the_two_models_are_supplied() -> None:
    """Freeze every diff argument, including the three options that default to null."""
    arguments = cli._parser().parse_args(["diff", "a.xml", "b.xml"])
    assert vars(arguments) == {
        "command": "diff",
        "baseline": "a.xml",
        "candidate": "b.xml",
        "baseline_root": None,
        "candidate_root": None,
        "output": None,
        "as_json": False,
        "full": False,
    }


def test_the_diff_parser_freezes_the_complete_contract_when_every_option_is_supplied() -> None:
    """Freeze the fully specified diff invocation, options and flags together."""
    arguments = cli._parser().parse_args(
        [
            "diff",
            "before/model.xml",
            "after/model.xml",
            "--baseline-root",
            "before",
            "--candidate-root",
            "after",
            "--output",
            "diff-out",
            "--json",
            "--full",
        ]
    )
    assert vars(arguments) == {
        "command": "diff",
        "baseline": "before/model.xml",
        "candidate": "after/model.xml",
        "baseline_root": "before",
        "candidate_root": "after",
        "output": "diff-out",
        "as_json": True,
        "full": True,
    }


def test_the_show_parser_freezes_one_receipt_and_two_output_flags() -> None:
    """Freeze the minimal show invocation to one receipt with both flags off."""
    arguments = cli._parser().parse_args(["show", "r.json"])
    assert vars(arguments) == {
        "command": "show",
        "receipt": "r.json",
        "as_json": False,
        "full": False,
    }


def test_the_show_parser_records_both_output_flags_when_they_are_supplied() -> None:
    """Freeze the show invocation that asks for the complete JSON export."""
    arguments = cli._parser().parse_args(
        ["show", "receipts/model_release.json", "--json", "--full"]
    )
    assert vars(arguments) == {
        "command": "show",
        "receipt": "receipts/model_release.json",
        "as_json": True,
        "full": True,
    }


def test_the_output_option_is_optional_for_diff_and_still_required_for_review_model() -> None:
    """Keep diff runnable with no output directory, unlike every publishing command."""
    arguments = cli._parser().parse_args(["diff", "a.xml", "b.xml"])
    assert arguments.output is None
    with pytest.raises(cli._InvocationError):
        cli._parser().parse_args(["review-model", "a.xml", "b.xml", "--policy", "policy.json"])
    with pytest.raises(cli._InvocationError):
        cli._parser().parse_args(["certify", "a.xml", "b.xml"])


def test_end_of_options_makes_a_dashed_show_argument_the_receipt_and_not_a_flag() -> None:
    """Read a receipt whose own name looks like an option without turning the flag on."""
    arguments = cli._parser().parse_args(["show", "--", "--json"])
    assert arguments.receipt == "--json"
    assert arguments.as_json is False
    assert arguments.full is False


def test_end_of_options_makes_dashed_diff_positionals_the_two_model_paths() -> None:
    """Compare two models whose paths begin with a dash without consuming them as options."""
    arguments = cli._parser().parse_args(["diff", "--", "-a.xml", "-b.xml"])
    assert arguments.baseline == "-a.xml"
    assert arguments.candidate == "-b.xml"
    assert arguments.as_json is False
    assert arguments.output is None


def test_both_new_commands_bind_to_the_frozen_review_model_operation_label() -> None:
    """Keep diff and show inside the frozen operational registry rather than adding a label."""
    assert cli._OPERATIONS["diff"] == "review-model"
    assert cli._OPERATIONS["show"] == "review-model"
    assert "review-model" in REGISTERED_OPERATIONS
    assert set(cli._OPERATIONS.values()) <= REGISTERED_OPERATIONS


def test_diff_is_dispatched_by_command_name_and_never_through_the_review_model_runner(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Route diff by its own command name even though it shares the review-model label."""
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def run_diff(*args: object, **kwargs: object) -> _FakeOutcome:
        calls.append((args, kwargs))
        return _FakeOutcome(_document(equal=True, exit_code=0), 0)

    _install_fake_result(monkeypatch, run_diff=run_diff)
    review_calls = _install_recording_model_release(monkeypatch)
    assert cli.main(["diff", "a.xml", "b.xml", "--output", "out"]) == 0
    assert calls == [
        (
            ("a.xml", "b.xml"),
            {"baseline_root": None, "candidate_root": None, "output_directory": "out"},
        )
    ]
    assert review_calls == []
    assert capsysbinary.readouterr().err == b""


def test_show_is_dispatched_by_command_name_and_never_through_the_review_model_runner(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Route show by its own command name and forward exactly the one receipt path."""
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def show_receipt(*args: object, **kwargs: object) -> dict[str, CanonicalValue]:
        calls.append((args, kwargs))
        return _document(equal=False, exit_code=40)

    _install_fake_result(monkeypatch, show_receipt=show_receipt)
    review_calls = _install_recording_model_release(monkeypatch)
    assert cli.main(["show", "r.json"]) == 0
    assert calls == [(("r.json",), {})]
    assert review_calls == []
    assert capsysbinary.readouterr().err == b""


def test_diff_forwards_every_supplied_option_to_the_runner_without_reshaping_it(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Freeze the diff runner call signature the CLI is allowed to make."""
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def run_diff(*args: object, **kwargs: object) -> _FakeOutcome:
        calls.append((args, kwargs))
        return _FakeOutcome(_document(equal=False, exit_code=40), 40)

    _install_fake_result(monkeypatch, run_diff=run_diff)
    assert (
        cli.main(
            [
                "diff",
                "before/model.xml",
                "after/model.xml",
                "--baseline-root",
                "before",
                "--candidate-root",
                "after",
            ]
        )
        == 40
    )
    assert calls == [
        (
            ("before/model.xml", "after/model.xml"),
            {
                "baseline_root": "before",
                "candidate_root": "after",
                "output_directory": None,
            },
        )
    ]
    capsysbinary.readouterr()


@pytest.mark.parametrize(("equal", "expected"), [(True, 0), (False, 40)])
def test_the_diff_exit_code_reaches_the_process_unchanged(
    equal: bool,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Return the runner's own identical or different exit code without reinterpreting it."""

    def run_diff(*_args: object, **_kwargs: object) -> _FakeOutcome:
        return _FakeOutcome(_document(equal=equal, exit_code=expected), expected)

    _install_fake_result(monkeypatch, run_diff=run_diff)
    assert cli.main(["diff", "a.xml", "b.xml"]) == expected
    captured = capsysbinary.readouterr()
    assert captured.err == b""
    assert captured.out.endswith(b"\n")


def test_diff_with_json_writes_exactly_one_bounded_result_document_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Emit the encoder's exact bytes and nothing else when JSON output is requested."""
    document = _document(equal=False, exit_code=40)

    def run_diff(*_args: object, **_kwargs: object) -> _FakeOutcome:
        return _FakeOutcome(document, 40)

    _install_fake_result(monkeypatch, run_diff=run_diff)
    assert cli.main(["diff", "a.xml", "b.xml", "--json"]) == 40
    captured = capsysbinary.readouterr()
    assert captured.err == b""
    assert captured.out == encode_result(document, full=False)
    assert captured.out.count(b"\n") == 1
    assert captured.out.endswith(b"\n")
    assert not captured.out.endswith(b"\n\n")
    emitted = json.loads(captured.out)
    assert emitted == document
    assert set(emitted) == {
        "schema",
        "schema_version",
        "command",
        "observation",
        "compiled_comparison",
        "evaluation",
        "runtime",
        "inputs",
        "findings",
        "finding_count",
        "limitations",
        "artifacts",
        "truncated",
        "problems",
    }


def test_show_with_json_and_full_writes_the_complete_unbounded_export(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Pass the --full flag through to the encoder rather than bounding the export."""
    document = _document(equal=True, exit_code=0)
    _install_fake_result(monkeypatch, show_receipt=lambda _path: document)
    assert cli.main(["show", "r.json", "--json", "--full"]) == 0
    captured = capsysbinary.readouterr()
    assert captured.err == b""
    assert captured.out == encode_result(document, full=True)
    assert captured.out.count(b"\n") == 1


def test_show_without_json_writes_the_readable_text_view_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Write the readable report, ending in exactly one newline, when JSON is not requested."""
    document = _document(equal=True, exit_code=0)
    _install_fake_result(monkeypatch, show_receipt=lambda _path: document)
    assert cli.main(["show", "r.json"]) == 0
    captured = capsysbinary.readouterr()
    assert captured.err == b""
    assert captured.out == render_text(document).encode("utf-8")
    assert captured.out.startswith(b"Compiled model identical.\n")
    assert captured.out.endswith(b"\n")
    assert not captured.out.endswith(b"\n\n")
    assert not captured.out.lstrip().startswith(b"{")


def test_a_controlled_show_failure_writes_the_canonical_failure_json_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Deliver the reader's own reason and exit code over the existing operational ABI."""
    failure = _controlled_show_failure()

    def show_receipt(_path: str) -> dict[str, CanonicalValue]:
        raise failure

    _install_fake_result(monkeypatch, show_receipt=show_receipt)
    assert cli.main(["show", "r.json", "--json"]) == 64
    captured = capsysbinary.readouterr()
    emitted = _parse_failure(captured.err)
    assert emitted["operation"] == "review-model"
    assert emitted["reason"]["code"] == "CONFIGURATION_PARSE_FAILED"  # type: ignore[index]
    assert emitted["exit_code"] == 64
    assert OperationalFailure.from_primitive(emitted).operation == "review-model"
    assert captured.err.count(b"\n") == 1


def test_a_controlled_show_failure_also_writes_a_result_document_carrying_the_reason(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Keep the JSON result contract satisfied on a failure by reporting the reason as a problem."""
    failure = _controlled_show_failure()

    def show_receipt(_path: str) -> dict[str, CanonicalValue]:
        raise failure

    _install_fake_result(monkeypatch, show_receipt=show_receipt)
    assert cli.main(["show", "r.json", "--json"]) == 64
    captured = capsysbinary.readouterr()
    document = json.loads(captured.out)
    assert document["schema"] == "metrifid.result"
    assert document["command"] == {"name": "show", "exit_code": 64}
    assert document["observation"] == "none"
    assert document["compiled_comparison"]["state"] == "not_established"
    assert document["problems"] == [
        {"code": "CONFIGURATION_PARSE_FAILED", "message": "receipt is not valid JSON"}
    ]
    assert document["findings"] == []
    assert document["finding_count"] == 0
    assert captured.out.count(b"\n") == 1


def test_a_controlled_show_failure_without_json_still_reports_readably_on_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Report the refusal in the readable view as well, not only in the failure JSON."""
    failure = _controlled_show_failure()

    def show_receipt(_path: str) -> dict[str, CanonicalValue]:
        raise failure

    _install_fake_result(monkeypatch, show_receipt=show_receipt)
    assert cli.main(["show", "r.json"]) == 64
    captured = capsysbinary.readouterr()
    assert captured.out.startswith(b"Compiled comparison not established.\n")
    assert b"CONFIGURATION_PARSE_FAILED: receipt is not valid JSON" in captured.out
    assert captured.out.endswith(b"\n")
    assert _parse_failure(captured.err)["reason"]["code"] == (  # type: ignore[index]
        "CONFIGURATION_PARSE_FAILED"
    )


@pytest.mark.parametrize(
    "argv",
    [["diff", "a.xml", "b.xml"], ["diff", "a.xml", "b.xml", "--json"], ["show", "r.json"]],
)
def test_an_unexpected_runner_exception_becomes_an_internal_failure_without_a_traceback(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Convert an unforeseen implementation error into exit 70 with no traceback anywhere."""

    def explode(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("forced")

    _install_fake_result(monkeypatch, run_diff=explode, show_receipt=explode)
    assert cli.main(argv) == 70
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    failure = _parse_failure(captured.err)
    assert failure["operation"] == "review-model"
    assert failure["reason"]["code"] == "INTERNAL_INVARIANT_FAILED"  # type: ignore[index]
    assert failure["exit_code"] == 70
    assert b"Traceback" not in captured.out
    assert b"Traceback" not in captured.err


def test_an_invalid_diff_invocation_is_refused_before_any_product_module_is_imported(
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Refuse a diff missing its candidate model as an invocation failure owned by diff."""
    assert cli.main(["diff", "only.xml"]) == 64
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    failure = _parse_failure(captured.err)
    assert failure["reason"]["code"] == "INVALID_CLI_INVOCATION"  # type: ignore[index]
    assert failure["operation"] == "review-model"
    assert cli._peek(["diff", "only.xml"]) == "diff"
    assert cli._peek(["show"]) == "show"


def test_importing_the_installed_cli_still_loads_no_numpy_and_no_mujoco(tmp_path: Path) -> None:
    """Keep the command tree importable, with both new commands, where no runtime exists."""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    source = (
        "import sys\n"
        "import metrifid.cli as loaded_cli\n"
        "native = sorted(name for name in sys.modules if name in {'numpy', 'mujoco'})\n"
        "commands = sorted({'diff', 'show'} & set(loaded_cli._OPERATIONS))\n"
        "print(native, commands)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.strip() == "[] ['diff', 'show']"


def test_a_report_that_cannot_be_rendered_becomes_an_internal_failure_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Writing the report is inside the boundary: a renderer fault never reaches the user raw."""

    def _explode(document: object, **_: object) -> bytes:
        raise MemoryError("renderer exhausted")

    def _render(document: object) -> str:
        raise MemoryError("renderer exhausted")

    module = ModuleType("metrifid.result")
    module.run_diff = None  # type: ignore[attr-defined]
    module.show_receipt = lambda _path: {"schema": "metrifid.result"}  # type: ignore[attr-defined]
    module.encode_result = _explode  # type: ignore[attr-defined]
    module.render_text = _render  # type: ignore[attr-defined]
    module.build_failure_document = real_build_failure_document  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metrifid.result", module)
    monkeypatch.setattr(metrifid, "result", module, raising=False)
    code = cli.main(["show", "receipt.json"])
    captured = capsysbinary.readouterr()
    assert code == 70
    failure = json.loads(captured.err)
    assert failure["reason"]["code"] == "INTERNAL_INVARIANT_FAILED"
    assert b"Traceback" not in captured.out + captured.err
    assert captured.out == b""


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["diff", "--json"], ("diff", True, False)),
        (["show", "--json", "--full"], ("show", True, True)),
        (["diff", "a.xml", "b.xml"], ("diff", False, False)),
        (["show", "--full"], ("show", False, True)),
    ],
)
def test_the_output_options_are_read_from_the_tokens_that_follow_the_subcommand(
    argv: list[str], expected: tuple[str, bool, bool]
) -> None:
    """A recognized diff or show request carries exactly the options it actually typed."""
    request = cli._result_request(argv)
    assert request is not None
    assert (request.command, request.as_json, request.full) == expected


@pytest.mark.parametrize(
    "argv",
    [
        ["show", "--", "--json"],
        ["diff", "--", "--json", "--full"],
        ["show", "receipt.json", "--", "--json"],
    ],
)
def test_a_json_token_after_the_end_of_options_never_turns_json_output_on(
    argv: list[str],
) -> None:
    """A file named --json is a file name, not a request for a different output format."""
    request = cli._result_request(argv)
    assert request is not None
    assert request.as_json is False


@pytest.mark.parametrize(
    "argv",
    [["certify", "--json"], ["compare", "config.json"], ["--json"], []],
)
def test_other_commands_are_not_treated_as_result_requests(argv: list[str]) -> None:
    """Only diff and show promise a result document, so only they may emit one."""
    assert cli._result_request(argv) is None


@pytest.mark.parametrize(
    "argv",
    [["diff", "--json"], ["diff", "one.xml", "--json"], ["show", "--json"]],
)
def test_a_usage_error_in_json_mode_writes_one_document_and_the_usual_failure(
    argv: list[str],
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """A caller that asked for JSON can parse an ordinary missing-argument error."""
    code = cli.main(argv)
    captured = capsysbinary.readouterr()
    assert code == 64
    document = json.loads(captured.out)
    assert document["schema"] == "metrifid.result"
    assert document["command"] == {"name": argv[0], "exit_code": 64}
    assert document["observation"] == "none"
    assert document["compiled_comparison"]["state"] == "not_established"
    assert [problem["code"] for problem in document["problems"]] == ["INVALID_CLI_INVOCATION"]
    assert captured.out.endswith(b"\n")
    assert captured.out.count(b"\n") == 1
    assert json.loads(captured.err)["reason"]["code"] == "INVALID_CLI_INVOCATION"
    assert b"Traceback" not in captured.out + captured.err


@pytest.mark.parametrize("argv", [["diff"], ["show"]])
def test_a_usage_error_without_json_leaves_stdout_empty(
    argv: list[str],
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Nothing is promised on stdout unless the request actually asked for a document."""
    assert cli.main(argv) == 64
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    assert json.loads(captured.err)["reason"]["code"] == "INVALID_CLI_INVOCATION"


def test_a_receipt_named_like_a_flag_is_read_as_a_file_and_reports_readably(
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """After the end-of-options marker the token is a path, so this parses and then fails to read.

    The refusal is reported in the readable view because no JSON output was ever requested.
    """
    assert cli.main(["show", "--", "--json"]) == 64
    captured = capsysbinary.readouterr()
    assert not captured.out.lstrip().startswith(b"{")
    assert b"CONFIGURATION_IO_FAILED" in captured.out
    assert json.loads(captured.err)["reason"]["code"] == "CONFIGURATION_IO_FAILED"


@pytest.mark.parametrize(
    "argv",
    [
        ["show", "receipt.json", "--js"],
        ["show", "receipt.json", "--fu"],
        ["diff", "a.xml", "b.xml", "--js", "--f"],
        ["diff", "a.xml", "b.xml", "--json"],
        ["show", "receipt.json"],
        ["diff", "a.xml", "b.xml", "--out", "run"],
        ["show", "--", "--json"],
    ],
)
def test_the_output_option_scanner_agrees_with_the_parser_it_stands_in_for(
    argv: list[str],
) -> None:
    """The scanner exists only for requests that never parsed, so it must read them the same way.

    The parser accepts an unambiguous abbreviation, so a request that reached it that way is still
    a request for the same option here.
    """
    parsed = cli._parser().parse_args(argv)
    request = cli._result_request(argv)
    assert request is not None
    assert (request.as_json, request.full) == (parsed.as_json, parsed.full)


def test_an_abbreviated_json_option_on_a_usage_error_still_returns_one_document(
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """An abbreviation the parser would have accepted is still an actual request for JSON."""
    assert cli.main(["diff", "--js"]) == 64
    captured = capsysbinary.readouterr()
    document = json.loads(captured.out)
    assert document["command"] == {"name": "diff", "exit_code": 64}
    assert document["problems"][0]["code"] == "INVALID_CLI_INVOCATION"


@pytest.mark.parametrize(
    "argv",
    [
        ["--output", "/tmp/out", "diff", "a.xml", "b.xml", "--json"],
        ["typo", "diff", "--json"],
        ["config.json", "show", "--json"],
    ],
)
def test_a_subcommand_appearing_after_junk_is_not_a_recognized_request(
    argv: list[str],
) -> None:
    """The subcommand is the token in command position, not the first one found anywhere.

    The parser rejects these as an invalid command choice, so no diff or show request was ever
    recognized and none may be reported as attempted.
    """
    assert cli._result_request(argv) is None


@pytest.mark.parametrize(
    "argv",
    [["--", "diff", "--json", "a.xml", "b.xml"], ["--", "show", "--json", "receipt.json"]],
)
def test_one_leading_end_of_options_marker_is_stripped_like_the_parser_strips_it(
    argv: list[str],
) -> None:
    """The parser consumes a single leading marker, so the request behind it is still a request."""
    parsed = cli._parser().parse_args(argv)
    request = cli._result_request(argv)
    assert request is not None
    assert (request.command, request.as_json) == (parsed.command, parsed.as_json)


def test_a_reader_that_stops_listening_leaves_the_exit_code_and_stderr_intact(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Nothing is promised once stdout is gone, and the command's own outcome must still stand.

    Left to interpreter shutdown a closed pipe replaces the real exit code and appends non-JSON
    text after the canonical failure, so the write is flushed while it can still be handled.
    """

    class _ClosedPipe:
        """Stand in for a pipe whose reader has already exited."""

        def write(self, payload: bytes) -> int:
            """Refuse the write exactly as a closed pipe does."""
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self) -> None:
            """Refuse the flush exactly as a closed pipe does."""
            raise BrokenPipeError(32, "Broken pipe")

    class _ClosedStdout:
        """Stand in for the stream wrapping that pipe."""

        buffer = _ClosedPipe()

        def write(self, text: str) -> int:
            """Refuse the text write the same way."""
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self) -> None:
            """Refuse the flush the same way."""
            raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(sys, "stdout", _ClosedStdout())
    code = cli.main(["diff", "--json"])
    assert code == 64
    captured = capsysbinary.readouterr()
    assert json.loads(captured.err)["reason"]["code"] == "INVALID_CLI_INVOCATION"
    assert b"Traceback" not in captured.err
