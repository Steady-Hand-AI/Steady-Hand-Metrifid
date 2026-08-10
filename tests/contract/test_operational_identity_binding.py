"""Exact operational reason-to-execution-identity binding matrix."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import cast

import pytest

from metrifid import (
    OperationalFailure,
    OperationalReasonCode,
    OperationalStage,
    OperationalToolObservation,
)
from metrifid.operational import ExecutionIdentityState, OperationalReason


def _digest(label: str) -> str:
    """Compute the canonical digest value used by operational identity binding fixtures.

    Content addressing keeps the mutation boundary explicit for operational identity binding.
    """
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _tool(state: ExecutionIdentityState) -> OperationalToolObservation:
    """Construct the tool fixture used by operational identity binding scenarios.

    Deterministic setup isolates operational identity binding without bypassing the contract
    boundary under assertion.
    """
    return OperationalToolObservation(
        "0.1.0a3",
        state,
        _digest("distribution") if state == "VERIFIED_INSTALLED_DISTRIBUTION" else None,
    )


def _failure(
    code: OperationalReasonCode,
    state: ExecutionIdentityState,
) -> OperationalFailure:
    """Inject the deterministic failure branch required by this scenario.

    The operational identity binding test can assert failure delivery for operational identity
    binding without depending on incidental runtime errors.
    """
    return OperationalFailure(
        schema="metrifid.operational_failure",
        schema_version=1,
        failure_rule_schema="metrifid.operational_failure_rules",
        failure_rule_schema_version=1,
        tool=_tool(state),
        operation="compare",
        stage=code.stage,
        reason=OperationalReason(
            code,
            "comparison",
            None,
            None,
            {"quoted_comparison_token": "TRACE_MALFORMED"},
        ),
        available_inputs=(),
        environment=None,
        exit_code=code.exit_code,
        failure_sha256=None,
    )


@pytest.mark.parametrize(
    "code",
    [
        OperationalReasonCode.EXECUTING_CODE_NOT_INSTALLED_DISTRIBUTION,
        OperationalReasonCode.MIXED_METRIFID_MODULE_ROOTS,
    ],
)
@pytest.mark.parametrize(
    ("state", "accepted"),
    [
        ("MISMATCH", True),
        ("UNBOUND", False),
        ("VERIFIED_INSTALLED_DISTRIBUTION", False),
    ],
)
def test_mismatch_reasons_require_mismatch(
    code: OperationalReasonCode,
    state: ExecutionIdentityState,
    accepted: bool,
) -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises mismatch reasons require mismatch; each refusal must retain its
    exact stage, exit code, and structured evidence contract.
    """
    if accepted:
        assert _failure(code, state).tool.execution_identity_state == "MISMATCH"
    else:
        with pytest.raises(ValueError, match="mismatch reason"):
            _failure(code, state)


@pytest.mark.parametrize(
    "code",
    [
        OperationalReasonCode.EDITABLE_INSTALL_UNSUPPORTED,
        OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
    ],
)
@pytest.mark.parametrize(
    ("state", "accepted"),
    [
        ("UNBOUND", True),
        ("MISMATCH", False),
        ("VERIFIED_INSTALLED_DISTRIBUTION", False),
    ],
)
def test_unbound_reasons_require_unbound(
    code: OperationalReasonCode,
    state: ExecutionIdentityState,
    accepted: bool,
) -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises unbound reasons require unbound; each refusal must retain its exact
    stage, exit code, and structured evidence contract.
    """
    if accepted:
        assert _failure(code, state).tool.execution_identity_state == "UNBOUND"
    else:
        with pytest.raises(ValueError, match="unbound reason"):
            _failure(code, state)


@pytest.mark.parametrize(
    "code",
    [
        OperationalReasonCode.UNSUPPORTED_PLATFORM,
        OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
        OperationalReasonCode.MODEL_CLOSURE_MUTATED,
        OperationalReasonCode.BASELINE_MODEL_COMPILE_ERROR,
        OperationalReasonCode.JOINT_NAME_MISSING,
        OperationalReasonCode.STATE_ARTIFACT_INVALID,
        OperationalReasonCode.CONTROL_GRID_NONINTEGRAL,
        OperationalReasonCode.TOLERANCE_MISSING,
        OperationalReasonCode.OUTPUT_WRITE_FAILED,
        OperationalReasonCode.CHILD_PROCESS_START_FAILED,
    ],
)
@pytest.mark.parametrize(
    ("state", "accepted"),
    [
        ("VERIFIED_INSTALLED_DISTRIBUTION", True),
        ("UNBOUND", False),
        ("MISMATCH", False),
    ],
)
def test_post_identity_stages_require_verified(
    code: OperationalReasonCode,
    state: ExecutionIdentityState,
    accepted: bool,
) -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises post identity stages require verified; each refusal must retain its
    exact stage, exit code, and structured evidence contract.
    """
    assert code.stage in {
        OperationalStage.ENVIRONMENT,
        OperationalStage.CONFIGURATION,
        OperationalStage.MODEL_CLOSURE,
        OperationalStage.MODEL_COMPILE,
        OperationalStage.SEMANTIC_IDENTITY,
        OperationalStage.INPUT_ARTIFACT,
        OperationalStage.TIME_CONTRACT,
        OperationalStage.TOLERANCE_CONTRACT,
        OperationalStage.OUTPUT,
        OperationalStage.CHILD_PROCESS,
    }
    if accepted:
        assert _failure(code, state).tool.distribution_sha256 == _digest("distribution")
    else:
        with pytest.raises(ValueError, match="post-identity"):
            _failure(code, state)


@pytest.mark.parametrize(
    "code",
    [
        OperationalReasonCode.INVALID_CLI_INVOCATION,
        OperationalReasonCode.INTERNAL_INVARIANT_FAILED,
    ],
)
@pytest.mark.parametrize(
    ("state", "accepted"),
    [
        ("UNBOUND", True),
        ("VERIFIED_INSTALLED_DISTRIBUTION", True),
        ("MISMATCH", False),
    ],
)
def test_invocation_and_internal_exact_state_matrix(
    code: OperationalReasonCode,
    state: ExecutionIdentityState,
    accepted: bool,
) -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises invocation and internal exact state matrix; each refusal must retain
    its exact stage, exit code, and structured evidence contract.
    """
    if accepted:
        completed = _failure(code, state).finalized()
        assert OperationalFailure.from_primitive(completed.to_primitive()) == completed
    else:
        with pytest.raises(ValueError, match="forbid MISMATCH"):
            _failure(code, state)


def test_verified_identity_refuses_placeholder_distribution_hash() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises verified identity refuses placeholder distribution hash; each
    refusal must retain its exact stage, exit code, and structured evidence contract.
    """
    with pytest.raises(ValueError, match="placeholder hash"):
        OperationalToolObservation(
            "0.1.0a3",
            "VERIFIED_INSTALLED_DISTRIBUTION",
            "0" * 64,
        )


def test_defensive_incomplete_identity_binding_refuses() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises defensive incomplete identity binding refuses; each refusal must
    retain its exact stage, exit code, and structured evidence contract.
    """
    import metrifid.operational as operational_module

    malformed = SimpleNamespace(
        reason=SimpleNamespace(code=OperationalReasonCode.INVALID_CLI_INVOCATION),
        stage=cast(OperationalStage, "UNKNOWN_STAGE"),
        tool=_tool("UNBOUND"),
    )
    with pytest.raises(ValueError, match="binding is incomplete"):
        operational_module._validate_execution_identity_binding(malformed)  # type: ignore[arg-type]
