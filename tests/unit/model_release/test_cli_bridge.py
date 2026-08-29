"""The non-native CLI and operational bridge for the Model Change Gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import metrifid
from metrifid import cli
from metrifid._distribution_types import DistributionIdentityError
from metrifid.operational import OperationalFailure, OperationalReasonCode


class _FakeModelReleaseOperationError(Exception):
    """Carry the same failure attribute as the future public operation error."""

    def __init__(self, failure: OperationalFailure) -> None:
        """Retain one strict operational failure for CLI delivery."""
        super().__init__(failure.reason.code.value)
        self.failure = failure


def _install_fake_module(
    monkeypatch: pytest.MonkeyPatch,
    review: Any,
    *,
    exit_code: int = 0,
) -> None:
    """Install a temporary API-shaped module without creating product implementation files."""
    module = ModuleType("metrifid.model_release")
    module.ModelReleaseOperationError = _FakeModelReleaseOperationError  # type: ignore[attr-defined]
    module.model_release_exit_code = lambda _status: exit_code  # type: ignore[attr-defined]
    module.review_model_release = review  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metrifid.model_release", module)
    monkeypatch.setattr(metrifid, "model_release", module, raising=False)


def _failure(stderr: bytes) -> dict[str, object]:
    """Parse one canonical operational failure emitted by the CLI."""
    value = json.loads(stderr)
    assert isinstance(value, dict)
    return value


def test_review_model_parser_freezes_the_required_argument_contract() -> None:
    """Require both models, policy, output, and the two optional model roots."""
    arguments = cli._parser().parse_args(
        [
            "review-model",
            "before/model.xml",
            "after/model.xml",
            "--policy",
            "release-policy.json",
            "--output",
            "review-out",
            "--baseline-root",
            "before",
            "--candidate-root",
            "after",
        ]
    )
    assert vars(arguments) == {
        "command": "review-model",
        "baseline": "before/model.xml",
        "candidate": "after/model.xml",
        "policy": "release-policy.json",
        "output": "review-out",
        "baseline_root": "before",
        "candidate_root": "after",
    }


@pytest.mark.parametrize("missing", ["policy", "output"])
def test_review_model_missing_required_option_is_its_own_invocation_failure(
    missing: str,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Label parser refusals as review-model before importing the implementation."""
    arguments = ["review-model", "before.xml", "after.xml"]
    if missing != "policy":
        arguments += ["--policy", "policy.json"]
    if missing != "output":
        arguments += ["--output", "out"]
    assert cli.main(arguments) == 64
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    failure = _failure(captured.err)
    assert failure["operation"] == "review-model"
    assert failure["reason"]["code"] == "INVALID_CLI_INVOCATION"  # type: ignore[index]


def test_review_model_dispatch_and_completed_summary_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Forward the frozen signature and emit exactly the four completed summary fields."""
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    status = SimpleNamespace(value="REVIEW_REQUIRED")
    result = SimpleNamespace(
        status=status,
        receipt_sha256="a" * 64,
        model_release_json=tmp_path / "out" / "model_release.json",
        model_release_markdown=tmp_path / "out" / "model_release.md",
    )

    def review(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return result

    _install_fake_module(monkeypatch, review, exit_code=40)
    assert (
        cli.main(
            [
                "review-model",
                "before.xml",
                "after.xml",
                "--policy",
                "policy.json",
                "--output",
                str(tmp_path / "out"),
                "--baseline-root",
                "before-root",
                "--candidate-root",
                "after-root",
            ]
        )
        == 40
    )
    assert calls == [
        (
            ("before.xml", "after.xml", "policy.json", str(tmp_path / "out")),
            {"baseline_root": "before-root", "candidate_root": "after-root"},
        )
    ]
    captured = capsysbinary.readouterr()
    assert captured.err == b""
    assert json.loads(captured.out) == {
        "status": "REVIEW_REQUIRED",
        "receipt_sha256": "a" * 64,
        "model_release_json": "model_release.json",
        "model_release_markdown": "model_release.md",
    }
    assert captured.out.endswith(b"\n")
    assert captured.out.count(b"\n") == 1


def test_review_model_distribution_refusal_uses_the_truthful_operation(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Keep an execution-identity refusal owned by the command that encountered it."""
    refusal = DistributionIdentityError(
        OperationalReasonCode.EDITABLE_INSTALL_UNSUPPORTED,
        "editable install",
    )

    def review(*_args: object, **_kwargs: object) -> object:
        raise refusal

    _install_fake_module(monkeypatch, review)
    assert (
        cli.main(
            [
                "review-model",
                "before.xml",
                "after.xml",
                "--policy",
                "policy.json",
                "--output",
                "out",
            ]
        )
        == 64
    )
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    failure = _failure(captured.err)
    assert failure["operation"] == "review-model"
    assert failure["reason"]["code"] == "EDITABLE_INSTALL_UNSUPPORTED"  # type: ignore[index]
    assert OperationalFailure.from_primitive(failure).operation == "review-model"


def test_review_model_defensive_failure_uses_exit_seventy_and_its_operation(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Convert an unexpected implementation exception without mislabelling the command."""

    def review(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("forced")

    _install_fake_module(monkeypatch, review)
    assert (
        cli.main(
            [
                "review-model",
                "before.xml",
                "after.xml",
                "--policy",
                "policy.json",
                "--output",
                "out",
            ]
        )
        == 70
    )
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    failure = _failure(captured.err)
    assert failure["operation"] == "review-model"
    assert failure["reason"]["code"] == "INTERNAL_INVARIANT_FAILED"  # type: ignore[index]
