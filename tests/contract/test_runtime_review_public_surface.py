"""Pin the Runtime Review public boundary and keep its selected core private."""

from __future__ import annotations

import argparse
import errno
import json
import os
import subprocess
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import pytest_check as check

import metrifid
from metrifid import cli
from metrifid.version import __version__

EXPECTED_PUBLIC_API = frozenset(
    {
        "__version__",
        "Binary64",
        "ExactRational",
        "canonical_json_bytes",
        "canonical_sha256",
        "strict_json_loads",
        "ComparisonStatus",
        "EngineThreadpoolState",
        "LimitationCode",
        "OperationalExitCode",
        "ReasonCode",
        "ReasonRecord",
        "STATUS_PRECEDENCE",
        "REASON_REGISTRY",
        "ComparisonConfig",
        "ComparisonContractIdentity",
        "ComparisonReceipt",
        "finalize_receipt",
        "validate_receipt",
        "OperationalStage",
        "OperationalReasonCode",
        "OperationalFailure",
        "OperationalToolObservation",
        "InputDigestCode",
        "InputDigest",
        "write_state_artifact",
        "write_actions_artifact",
    }
)
EXPECTED_CLI_COMMANDS = frozenset(
    {
        "audit-timestep",
        "certify",
        "compare",
        "qualify-workload",
        "review-model",
        "review-runtime",
        "run-runtime-review",
    }
)
EXPECTED_RUNTIME_REVIEW_API = frozenset(
    {
        "RuntimeReviewStatus",
        "RuntimeReviewReasonCode",
        "RuntimeReviewExitCode",
        "RuntimeReviewResult",
        "RuntimeReviewRunResult",
        "RuntimeReviewOperationError",
        "runtime_review_exit_code",
        "review_runtime_configuration_file",
        "run_runtime_review_configuration_file",
        "load_and_validate_runtime_review_receipt",
    }
)
PRIVATE_PRODUCT_NAMES = frozenset(
    {
        "CaseEvidence",
        "GateEvent",
        "Interval",
        "OrientationObservation",
        "ScalarObservation",
        "evaluate_case",
    }
)


def _parser_commands() -> frozenset[str]:
    """Return the exact installed subcommand names registered with argparse."""
    parser = cli._parser()
    subparsers = [
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(subparsers) == 1
    return frozenset(subparsers[0].choices)


def test_runtime_review_preserves_the_top_level_public_api() -> None:
    """Keep Runtime Review on its supported subpackage rather than the top-level namespace."""
    check.equal(metrifid.__version__, __version__, "the package exposes a different version")
    check.equal(
        frozenset(metrifid.__all__),
        EXPECTED_PUBLIC_API,
        "the top-level supported API changed with the private method",
    )
    check.equal(
        frozenset(metrifid.__dir__()),
        EXPECTED_PUBLIC_API,
        "package introspection advertises a name outside the supported API",
    )
    for name in PRIVATE_PRODUCT_NAMES | {"_native_upgrade"}:
        check.is_not_in(name, metrifid.__all__, f"private name {name!r} is publicly exported")
        check.is_not_in(name, metrifid.__dir__(), f"private name {name!r} is publicly advertised")


def test_cli_exposes_both_bounded_runtime_review_paths() -> None:
    """Retain existing commands and expose only the two bounded Runtime Review paths."""
    check.equal(
        frozenset(cli._OPERATIONS),
        EXPECTED_CLI_COMMANDS,
        "the CLI dispatch table changed",
    )
    check.equal(
        _parser_commands(),
        EXPECTED_CLI_COMMANDS,
        "the installed argparse command set changed",
    )
    for forbidden in ("upgrade", "native-upgrade", "upgrade-gate"):
        check.is_not_in(
            forbidden, cli._OPERATIONS, f"forbidden command {forbidden!r} is dispatched"
        )


def test_runtime_review_subpackage_has_the_exact_lazy_surface() -> None:
    """Expose exactly ten supported names without changing the root package API."""
    import metrifid.runtime_review as runtime_review

    check.equal(frozenset(runtime_review.__all__), EXPECTED_RUNTIME_REVIEW_API)
    check.equal(frozenset(runtime_review.__dir__()), EXPECTED_RUNTIME_REVIEW_API)
    check.is_not_in("runtime_review", metrifid.__all__)


def test_public_runner_remains_lazy_until_execution(tmp_path: Path) -> None:
    """Import the public subpackage without native imports, execution, or external writes."""
    package_file = metrifid.__file__
    assert package_file is not None
    import_root = Path(package_file).resolve().parent.parent
    probe = r"""
import importlib
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
import metrifid

spec = importlib.util.find_spec("metrifid.runtime_review")
if spec is None or spec.origin is None:
    raise RuntimeError("runtime_review has no importable origin")
allowed_reads = {
    os.path.abspath(spec.origin),
    os.path.abspath(importlib.util.cache_from_source(spec.origin)),
}
effects = []
write_mask = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
mutating = {
    "os.chdir", "os.chmod", "os.chown", "os.link", "os.mkdir", "os.remove",
    "os.rename", "os.rmdir", "os.symlink", "os.truncate", "os.utime",
    "shutil.copyfile", "shutil.move", "tempfile.mkdtemp", "tempfile.mkstemp",
}
processes = {"os.posix_spawn", "os.spawn", "os.system", "subprocess.Popen"}

def audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.path.abspath(os.fsdecode(args[0]))
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else None
        writes = (
            isinstance(mode, str) and any(token in mode for token in "wax+")
        ) or (isinstance(flags, int) and bool(flags & write_mask))
        if writes or path not in allowed_reads:
            effects.append([event, path])
    elif event in mutating or event in processes or event.startswith("socket."):
        effects.append([event, str(args[0]) if args else ""])

sys.addaudithook(audit)
module = importlib.import_module("metrifid.runtime_review")
print(json.dumps({
    "effects": effects,
    "execution_loaded": "metrifid.runtime_review._execution" in sys.modules,
    "collector_loaded": "metrifid.runtime_review._native_profile_identity" in sys.modules,
    "numpy_loaded": "numpy" in sys.modules,
    "mujoco_loaded": "mujoco" in sys.modules,
    "surface": sorted(module.__all__),
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(import_root)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    observed = json.loads(completed.stdout)
    assert observed["effects"] == []
    assert observed["execution_loaded"] is False
    assert observed["collector_loaded"] is False
    assert observed["numpy_loaded"] is False
    assert observed["mujoco_loaded"] is False
    assert frozenset(observed["surface"]) == EXPECTED_RUNTIME_REVIEW_API


def test_runtime_review_result_exposes_the_completed_decision_fields() -> None:
    """Give SDK callers direct access to the status reason and published evidence paths."""
    from metrifid.runtime_review import RuntimeReviewResult

    check.is_true(is_dataclass(RuntimeReviewResult), "RuntimeReviewResult must remain a dataclass")
    check.equal(
        tuple(field.name for field in fields(RuntimeReviewResult)),
        (
            "status",
            "reason_code",
            "receipt",
            "receipt_sha256",
            "runtime_review_json",
            "runtime_review_markdown",
        ),
        "RuntimeReviewResult does not expose the frozen completed-decision fields",
    )
    check.is_true(
        isinstance(RuntimeReviewResult.exit_code, property),
        "RuntimeReviewResult.exit_code must remain a read-only property",
    )


def test_runtime_review_run_result_exposes_execution_and_decision_fields() -> None:
    """Expose the existing decision alongside the completed operational run record."""
    from metrifid.runtime_review import RuntimeReviewRunResult

    check.is_true(
        is_dataclass(RuntimeReviewRunResult),
        "RuntimeReviewRunResult must remain a dataclass",
    )
    check.is_true(
        RuntimeReviewRunResult.__dataclass_params__.frozen,
        "RuntimeReviewRunResult must remain frozen",
    )
    check.is_true(
        hasattr(RuntimeReviewRunResult, "__slots__"),
        "RuntimeReviewRunResult must remain slotted",
    )
    check.equal(
        tuple(field.name for field in fields(RuntimeReviewRunResult)),
        (
            "status",
            "reason_code",
            "receipt",
            "receipt_sha256",
            "runtime_review_json",
            "runtime_review_markdown",
            "runtime_review_run_json",
            "run_sha256",
            "captured_evidence_root",
            "generated_runtime_review_config",
        ),
        "RuntimeReviewRunResult does not expose the frozen execution fields",
    )
    check.is_true(
        isinstance(RuntimeReviewRunResult.exit_code, property),
        "RuntimeReviewRunResult.exit_code must remain a read-only property",
    )


def test_completed_insufficient_evidence_result_exposes_its_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Populate the public SDK reason from the completed insufficient decision."""
    from metrifid.runtime_review import (
        RuntimeReviewReasonCode,
        RuntimeReviewStatus,
        review_runtime_configuration_file,
    )
    from metrifid.runtime_review import _run as runtime_run

    configuration = object()
    evidence = object()
    decision = SimpleNamespace(
        status=RuntimeReviewStatus.INSUFFICIENT_EVIDENCE,
        reason_code=RuntimeReviewReasonCode.PREFIX_TOO_SHORT,
    )
    published = SimpleNamespace(
        runtime_review_json=tmp_path / "owned" / "runtime_review.json",
        runtime_review_markdown=tmp_path / "owned" / "runtime_review.md",
    )
    receipt = {"receipt_sha256": "a" * 64}
    monkeypatch.setattr(runtime_run, "_load_configuration", Mock(return_value=configuration))
    monkeypatch.setattr(runtime_run, "_admit_evidence", Mock(return_value=evidence))
    monkeypatch.setattr(runtime_run, "evaluate_runtime_evidence", Mock(return_value=decision))
    monkeypatch.setattr(runtime_run, "_publish", Mock(return_value=(published, receipt)))

    result = review_runtime_configuration_file(tmp_path / "runtime_review.json")

    assert result.status is RuntimeReviewStatus.INSUFFICIENT_EVIDENCE
    assert result.reason_code is RuntimeReviewReasonCode.PREFIX_TOO_SHORT
    assert result.exit_code == 20
    assert result.receipt is receipt


def test_runtime_review_cli_converts_an_unexpected_exception_to_exit_seventy(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Emit one canonical internal failure without leaking a traceback."""
    import metrifid.runtime_review as runtime_review

    monkeypatch.setattr(
        runtime_review,
        "review_runtime_configuration_file",
        Mock(side_effect=RuntimeError("forced runtime-review failure")),
    )

    assert cli.main(["review-runtime", "runtime_review.json"]) == 70
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    assert b"Traceback" not in captured.err
    failure = json.loads(captured.err)
    assert failure["operation"] == "compare"
    assert failure["reason"]["code"] == "INTERNAL_INVARIANT_FAILED"
    assert failure["exit_code"] == 70


def test_runtime_review_execution_cli_propagates_the_existing_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Emit exactly the completed execution and existing referee fields with its exit."""
    import metrifid.runtime_review as runtime_review
    from metrifid.runtime_review import RuntimeReviewStatus

    completed = SimpleNamespace(
        status=RuntimeReviewStatus.UNRESOLVED_NEAR_BOUNDARY,
        reason_code=None,
        receipt_sha256="a" * 64,
        run_sha256="b" * 64,
        runtime_review_json=tmp_path / "decision" / "runtime_review.json",
        runtime_review_markdown=tmp_path / "decision" / "runtime_review.md",
        runtime_review_run_json=tmp_path / "runtime_review_run.json",
        exit_code=30,
    )
    runner = Mock(return_value=completed)
    monkeypatch.setattr(runtime_review, "run_runtime_review_configuration_file", runner)

    assert cli.main(["run-runtime-review", "runtime_review_run.json"]) == 30
    runner.assert_called_once_with("runtime_review_run.json")
    captured = capsysbinary.readouterr()
    assert captured.err == b""
    payload = json.loads(captured.out)
    assert set(payload) == {
        "status",
        "reason_code",
        "receipt_sha256",
        "run_sha256",
        "runtime_review_json",
        "runtime_review_markdown",
        "runtime_review_run_json",
    }
    assert payload["status"] == "UNRESOLVED_NEAR_BOUNDARY"
    assert payload["reason_code"] is None
    assert payload["receipt_sha256"] == "a" * 64
    assert payload["run_sha256"] == "b" * 64


def test_runtime_review_execution_cli_contains_unexpected_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Map an unexpected execution failure to exit seventy without a traceback."""
    import metrifid.runtime_review as runtime_review

    monkeypatch.setattr(
        runtime_review,
        "run_runtime_review_configuration_file",
        Mock(side_effect=RuntimeError("forced execution failure")),
    )

    assert cli.main(["run-runtime-review", "runtime_review_run.json"]) == 70
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    assert b"Traceback" not in captured.err
    failure = json.loads(captured.err)
    assert failure["operation"] == "compare"
    assert failure["reason"]["code"] == "INTERNAL_INVARIANT_FAILED"
    assert failure["exit_code"] == 70


@pytest.mark.parametrize(
    ("entrypoint", "exception_type", "error_number"),
    [
        ("sdk", OSError, errno.EIO),
        ("cli", PermissionError, errno.EACCES),
    ],
)
def test_owned_output_oserror_is_exit_sixty_four_and_preserves_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    entrypoint: str,
    exception_type: type[OSError],
    error_number: int,
) -> None:
    """Translate native output I/O failures without deleting incomplete evidence."""
    from metrifid.operational import (
        OperationalFailure,
        OperationalReasonCode,
        OperationalToolObservation,
    )
    from metrifid.runtime_review import (
        RuntimeReviewOperationError,
        review_runtime_configuration_file,
    )
    from metrifid.runtime_review import _run as runtime_run

    output_dir = tmp_path / "output"
    configuration = SimpleNamespace(output_dir=output_dir, raw_bytes=b"{}\n")
    evidence = SimpleNamespace(cells=())
    decision = SimpleNamespace(status=object(), reason_code=None)
    native_error = exception_type(error_number, "deterministic owned-output failure")
    tool = OperationalToolObservation(__version__, "VERIFIED_INSTALLED_DISTRIBUTION", "a" * 64)

    def fail_owned_output_preparation(path: Path, configuration_bytes: bytes) -> object:
        """Retain deterministic incomplete bytes, then model a native output I/O failure."""
        staging = path / ".runtime_review.staging"
        staging.mkdir(parents=True)
        (staging / "admitted_runtime_review_config.json").write_bytes(configuration_bytes)
        raise native_error

    monkeypatch.setattr(runtime_run, "_load_configuration", Mock(return_value=configuration))
    monkeypatch.setattr(runtime_run, "_admit_evidence", Mock(return_value=evidence))
    monkeypatch.setattr(runtime_run, "evaluate_runtime_evidence", Mock(return_value=decision))
    monkeypatch.setattr(runtime_run, "_tool", Mock(return_value=tool))
    monkeypatch.setattr(
        runtime_run,
        "prepare_owned_runtime_review_output",
        fail_owned_output_preparation,
    )

    if entrypoint == "sdk":
        with pytest.raises(RuntimeReviewOperationError) as caught:
            review_runtime_configuration_file(tmp_path / "runtime_review.json")
        failure = caught.value.failure
    else:
        assert cli.main(["review-runtime", "runtime_review.json"]) == 64
        captured = capsysbinary.readouterr()
        assert captured.out == b""
        assert b"Traceback" not in captured.err
        failure = OperationalFailure.from_primitive(json.loads(captured.err))

    assert failure.operation == "compare"
    assert failure.reason.code is OperationalReasonCode.OUTPUT_WRITE_FAILED
    assert failure.reason.field == "output_dir"
    assert failure.reason.evidence["exception_type"] == exception_type.__name__
    assert int(failure.exit_code) == 64
    staging = output_dir / ".runtime_review.staging"
    assert (staging / "admitted_runtime_review_config.json").read_bytes() == b"{}\n"
    assert not (output_dir / "runtime_review").exists()


def test_postpublication_failure_is_exit_seventy_and_preserves_the_public_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Never erase an already published tree when independent replay fails unexpectedly."""
    from metrifid.runtime_review import _run as runtime_run

    output_dir = tmp_path / "output"
    configuration = SimpleNamespace(output_dir=output_dir, raw_bytes=b"{}\n")
    evidence = SimpleNamespace(cells=())
    decision = SimpleNamespace(status=object(), reason_code=None)

    class PublishedThenReplayFails:
        """Model the publication boundary while retaining one foreign sentinel."""

        def __enter__(self) -> PublishedThenReplayFails:
            """Enter the deterministic fake staging context."""
            return self

        def __exit__(self, *exc_info: object) -> None:
            """Leave the fake context without deleting the published sentinel."""
            return None

        def copy_evidence_cells(self, cells: object) -> tuple[object, ...]:
            """Accept the bounded empty mocked evidence sequence."""
            assert cells == ()
            return ()

        def publish(self, *args: object, **kwargs: object) -> SimpleNamespace:
            """Create a visible final tree before the forced replay failure."""
            root = output_dir / "runtime_review"
            root.mkdir(parents=True)
            (root / "foreign").write_bytes(b"preserve")
            return SimpleNamespace(
                root=root,
                runtime_review_json=root / "runtime_review.json",
                runtime_review_markdown=root / "runtime_review.md",
            )

    monkeypatch.setattr(runtime_run, "_load_configuration", Mock(return_value=configuration))
    monkeypatch.setattr(runtime_run, "_admit_evidence", Mock(return_value=evidence))
    monkeypatch.setattr(runtime_run, "evaluate_runtime_evidence", Mock(return_value=decision))
    monkeypatch.setattr(
        runtime_run,
        "prepare_owned_runtime_review_output",
        Mock(return_value=PublishedThenReplayFails()),
    )
    monkeypatch.setattr(
        runtime_run,
        "build_runtime_review_receipt",
        Mock(return_value={"receipt_sha256": "a" * 64}),
    )
    monkeypatch.setattr(runtime_run, "render_runtime_review_markdown", Mock(return_value="# ok\n"))
    monkeypatch.setattr(
        runtime_run,
        "load_and_validate_runtime_review_receipt",
        Mock(side_effect=RuntimeError("postpublication replay failed")),
    )

    assert cli.main(["review-runtime", "runtime_review.json"]) == 70
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    failure = json.loads(captured.err)
    assert failure["reason"]["code"] == "INTERNAL_INVARIANT_FAILED"
    assert failure["exit_code"] == 70
    assert (output_dir / "runtime_review" / "foreign").read_bytes() == b"preserve"


def test_private_module_import_is_data_only_and_has_no_external_side_effects(
    tmp_path: Path,
) -> None:
    """Import the selected module with MuJoCo blocked and audit external side effects."""
    package_file = metrifid.__file__
    assert package_file is not None
    import_root = Path(package_file).resolve().parent.parent
    probe = r"""
import collections.abc
import dataclasses
import importlib
import importlib.abc
import importlib.util
import json
import math
import os
import sys
import typing

sys.dont_write_bytecode = True
side_effects = []
mujoco_attempts = []
write_mask = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
mutating_events = {
    "os.chdir", "os.chmod", "os.chown", "os.link", "os.mkdir", "os.remove",
    "os.rename", "os.rmdir", "os.symlink", "os.truncate", "os.utime",
    "shutil.copyfile", "shutil.move", "tempfile.mkdtemp", "tempfile.mkstemp",
}
process_events = {"os.posix_spawn", "os.spawn", "os.system", "subprocess.Popen"}

import metrifid
private_loaded_by_package = "metrifid._native_upgrade" in sys.modules
mujoco_loaded_by_package = any(
    name == "mujoco" or name.startswith("mujoco.") for name in sys.modules
)
target_spec = importlib.util.find_spec("metrifid._native_upgrade")
if target_spec is None or target_spec.origin is None:
    raise RuntimeError("private native-upgrade module has no importable origin")
allowed_import_reads = {
    os.path.abspath(target_spec.origin),
    os.path.abspath(importlib.util.cache_from_source(target_spec.origin)),
}

def audit(event, args):
    if event == "open":
        opened = os.path.abspath(os.fsdecode(args[0]))
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else None
        writes = (
            isinstance(mode, str) and any(token in mode for token in "wax+")
        ) or (isinstance(flags, int) and bool(flags & write_mask))
        if writes or opened not in allowed_import_reads:
            side_effects.append([event, opened])
    elif event in mutating_events or event in process_events or event.startswith("socket."):
        side_effects.append([event, str(args[0]) if args else ""])

class BlockMujoco(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mujoco" or fullname.startswith("mujoco."):
            mujoco_attempts.append(fullname)
            raise ImportError("MuJoCo import blocked by private-surface contract")
        return None

sys.addaudithook(audit)
sys.meta_path.insert(0, BlockMujoco())
module = importlib.import_module("metrifid._native_upgrade")
required = (
    "CaseEvidence", "GateEvent", "Interval", "OrientationObservation",
    "ScalarObservation", "evaluate_case",
)
losing = (
    "FINEST_POINT", "OBSERVED_GRID_HULL", "evaluate_finest_point",
    "evaluate_observed_grid_hull",
)
print(json.dumps({
    "private_loaded_by_package": private_loaded_by_package,
    "mujoco_loaded_by_package": mujoco_loaded_by_package,
    "mujoco_attempts": mujoco_attempts,
    "mujoco_loaded": any(
        name == "mujoco" or name.startswith("mujoco.") for name in sys.modules
    ),
    "side_effects": side_effects,
    "required_names": {name: hasattr(module, name) for name in required},
    "frozen_dataclasses": {
        name: bool(
            dataclasses.is_dataclass(getattr(module, name, None))
            and getattr(getattr(module, name), "__dataclass_params__").frozen
        )
        for name in required[:-1]
    },
    "evaluate_case_callable": callable(getattr(module, "evaluate_case", None)),
    "losing_names": [name for name in losing if hasattr(module, name)],
    "top_level_exports": [name for name in required if name in metrifid.__all__],
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        token for token in (str(import_root), environment.get("PYTHONPATH", "")) if token
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    observed = json.loads(completed.stdout)
    assert isinstance(observed, dict)
    check.is_false(
        observed["private_loaded_by_package"],
        "importing metrifid eagerly imported its private native-upgrade module",
    )
    check.is_false(
        observed["mujoco_loaded_by_package"],
        "importing metrifid loaded MuJoCo before the private-module probe",
    )
    check.equal(observed["mujoco_attempts"], [], "the private module attempted to import MuJoCo")
    check.is_false(observed["mujoco_loaded"], "the private module loaded MuJoCo")
    check.equal(
        observed["side_effects"],
        [],
        f"the private module performed an external side effect: {observed['side_effects']}",
    )
    check.equal(
        observed["required_names"],
        dict.fromkeys(PRIVATE_PRODUCT_NAMES, True),
        "the selected private method surface is incomplete",
    )
    check.equal(
        observed["frozen_dataclasses"],
        {name: True for name in PRIVATE_PRODUCT_NAMES if name != "evaluate_case"},
        "private evidence records must be frozen dataclasses",
    )
    check.is_true(observed["evaluate_case_callable"], "the selected evaluator is not callable")
    check.equal(observed["losing_names"], [], "a losing reference method entered product code")
    check.equal(observed["top_level_exports"], [], "a private method name reached the public API")
