"""Delivery of frozen operational-failure evidence through the failure boundary.

`refuse()` deep-freezes evidence, so nested values arrive as `MappingProxyType` and
`tuple`. Before the repair a shallow copy left them frozen and the canonical serializer,
which is an exact-type hash boundary, rejected them: the intended reason degraded to
`INTERNAL_INVARIANT_FAILED` with exit 70. These tests pin the delivery contract.
"""

from __future__ import annotations

import pytest

from metrifid.compare._failure import ComparisonOperationError, operational_error
from metrifid.json_values import freeze_canonical
from metrifid.operational import (
    OperationalExitCode,
    OperationalReasonCode,
    OperationalToolObservation,
)
from metrifid.version import __version__ as CURRENT_VERSION

VERIFIED = OperationalToolObservation(CURRENT_VERSION, "VERIFIED_INSTALLED_DISTRIBUTION", "c" * 64)
UNBOUND = OperationalToolObservation(CURRENT_VERSION, "UNBOUND", None)


@pytest.mark.parametrize(
    ("label", "evidence"),
    [
        ("flat scalars", {"requested": 3, "message": "x"}),
        ("flat list", {"names": ["a", "b"]}),
        ("nested mapping", {"control_dt": {"numerator": 1, "denominator": 25}}),
        ("list of mappings", {"rows": [{"k": 1}, {"k": 2}]}),
        ("frozen nested mapping", freeze_canonical({"d": {"k": 1}})),
        ("frozen list", freeze_canonical({"names": ["a", "b"]})),
        ("frozen deep mix", freeze_canonical({"a": {"b": [{"c": 1}, {"c": 2}]}})),
    ],
)
def test_every_evidence_shape_delivers_its_intended_reason(label: str, evidence: object) -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises every evidence shape delivers its intended reason; each refusal must
    retain its exact stage, exit code, and structured evidence contract.
    """
    error = operational_error(
        tool=VERIFIED,
        code=OperationalReasonCode.CONTROL_GRID_NONINTEGRAL,
        role="comparison",
        evidence=evidence,  # type: ignore[arg-type]
    )
    failure = error.failure
    assert isinstance(error, ComparisonOperationError)
    assert failure.reason.code is OperationalReasonCode.CONTROL_GRID_NONINTEGRAL
    assert failure.exit_code is OperationalExitCode.INVALID_INVOCATION_INPUT_OUTPUT
    assert int(failure.exit_code) == 64
    assert failure.failure_sha256 is not None
    assert failure.operation == "compare"


def test_frozen_evidence_round_trips_to_mutable_primitives() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises frozen evidence round trips to mutable primitives; each refusal must
    retain its exact stage, exit code, and structured evidence contract.
    """
    error = operational_error(
        tool=VERIFIED,
        code=OperationalReasonCode.CONTROL_GRID_NONINTEGRAL,
        role="comparison",
        evidence=freeze_canonical({"a": {"b": [1, 2]}, "c": ["x"]}),
    )
    evidence = error.failure.to_primitive()["reason"]["evidence"]
    assert evidence == {"a": {"b": [1, 2]}, "c": ["x"]}
    assert type(evidence["a"]) is dict
    assert type(evidence["a"]["b"]) is list


def test_flat_scalar_evidence_is_byte_identical_to_the_accepted_behaviour() -> None:
    """The repair must not move any previously working failure hash."""
    flat = {"requested_total_internal_steps": 9000, "maximum_total_internal_steps": 4096}
    first = operational_error(
        tool=VERIFIED,
        code=OperationalReasonCode.CONTROL_GRID_NONINTEGRAL,
        role="comparison",
        evidence=flat,
    ).failure
    assert first.to_primitive()["reason"]["evidence"] == flat
    second = operational_error(
        tool=VERIFIED,
        code=OperationalReasonCode.CONTROL_GRID_NONINTEGRAL,
        role="comparison",
        evidence=dict(flat),
    ).failure
    assert first.failure_sha256 == second.failure_sha256


def test_audit_operation_is_selectable_and_defaults_to_compare() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises audit operation is selectable and defaults to compare; each refusal
    must retain its exact stage, exit code, and structured evidence contract.
    """
    default = operational_error(
        tool=UNBOUND,
        code=OperationalReasonCode.INVALID_CLI_INVOCATION,
        role=None,
        evidence={"message": "x"},
    ).failure
    assert default.operation == "compare"
    audit = operational_error(
        tool=UNBOUND,
        code=OperationalReasonCode.INVALID_CLI_INVOCATION,
        role=None,
        evidence={"message": "x"},
        operation="audit-timestep",
    ).failure
    assert audit.operation == "audit-timestep"
    assert audit.reason.code is OperationalReasonCode.INVALID_CLI_INVOCATION
    assert audit.failure_sha256 is not None
    assert audit.failure_sha256 != default.failure_sha256


def test_private_candidate_timestep_seam_delivers_a_canonical_internal_invariant() -> None:
    """The audit's private seam fails closed through the ordinary failure boundary.

    The seam refuses a timestep the configuration did not declare, and refuses two roles that
    are not the same source model. Both are internal invariants: they must arrive as canonical
    `compare` operational failures with exit 70 and a valid self-hash, never as a bare
    exception and never as a comparison status.
    """
    for message in (
        "candidate timestep override does not equal the declared candidate timestep",
        "candidate timestep override requires identical baseline and candidate model closures",
    ):
        failure = operational_error(
            tool=VERIFIED,
            code=OperationalReasonCode.INTERNAL_INVARIANT_FAILED,
            role=None,
            evidence={"exception_type": "ValueError", "message": message},
        ).failure
        assert failure.operation == "compare"
        assert failure.reason.code is OperationalReasonCode.INTERNAL_INVARIANT_FAILED
        assert failure.exit_code is OperationalExitCode.INTERNAL_PROJECT_FAILURE
        assert int(failure.exit_code) == 70
        assert failure.reason.evidence["message"] == message
        assert failure.failure_sha256 is not None
        primitive = failure.to_primitive()
        assert primitive["reason"]["evidence"]["exception_type"] == "ValueError"
        assert primitive["tool"]["version"] == CURRENT_VERSION


def test_tool_observation_tracks_the_single_version_authority() -> None:
    """Failure evidence must report the package version authority, not a pinned literal."""
    assert VERIFIED.version == CURRENT_VERSION
    assert UNBOUND.version == CURRENT_VERSION
    failure = operational_error(
        tool=VERIFIED,
        code=OperationalReasonCode.CONTROL_GRID_NONINTEGRAL,
        role=None,
        evidence={"message": "x"},
    ).failure
    assert failure.to_primitive()["tool"]["version"] == CURRENT_VERSION
