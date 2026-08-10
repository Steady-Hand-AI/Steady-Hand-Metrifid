"""Strict pre-contract OperationalFailure lifecycle tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, cast

import pytest

from metrifid import (
    InputDigest,
    InputDigestCode,
    OperationalExitCode,
    OperationalFailure,
    OperationalReasonCode,
    OperationalStage,
    OperationalToolObservation,
)
from metrifid.operational import OperationalReason


def digest(label: str) -> str:
    """Compute the canonical digest value used by operational failure fixtures.

    Content addressing keeps the mutation boundary explicit for operational failure.
    """
    return hashlib.sha256(label.encode()).hexdigest()


def default_tool(code: OperationalReasonCode) -> OperationalToolObservation:
    """Construct the default tool fixture used by operational failure scenarios.

    Deterministic setup isolates operational failure without bypassing the contract boundary
    under assertion.
    """
    mismatch = {
        OperationalReasonCode.EXECUTING_CODE_NOT_INSTALLED_DISTRIBUTION,
        OperationalReasonCode.MIXED_METRIFID_MODULE_ROOTS,
    }
    unbound = {
        OperationalReasonCode.EDITABLE_INSTALL_UNSUPPORTED,
        OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
    }
    if code in mismatch:
        return OperationalToolObservation("0.1.0a3", "MISMATCH", None)
    if code in unbound or code.stage in {
        OperationalStage.INVOCATION,
        OperationalStage.INTERNAL,
    }:
        return OperationalToolObservation("0.1.0a3", "UNBOUND", None)
    return OperationalToolObservation(
        "0.1.0a3", "VERIFIED_INSTALLED_DISTRIBUTION", digest("distribution")
    )


def make_failure(
    code: OperationalReasonCode = OperationalReasonCode.MONITORED_JOINT_SET_EMPTY,
    *,
    inputs: tuple[InputDigest, ...] = (),
    tool: OperationalToolObservation | None = None,
) -> OperationalFailure:
    """Inject the deterministic make failure branch required by this scenario.

    The operational failure test can assert failure delivery for operational failure without
    depending on incidental runtime errors.
    """
    return OperationalFailure(
        schema="metrifid.operational_failure",
        schema_version=1,
        failure_rule_schema="metrifid.operational_failure_rules",
        failure_rule_schema_version=1,
        tool=tool or default_tool(code),
        operation="compare",
        stage=code.stage,
        reason=OperationalReason(code, "comparison", "joint_tolerances", None, {}),
        available_inputs=inputs,
        environment=None,
        exit_code=code.exit_code,
        failure_sha256=None,
    )


def test_strict_round_trip_and_self_hash() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises strict round trip and self hash; each refusal must retain its exact
    stage, exit code, and structured evidence contract.
    """
    inputs = (
        InputDigest(InputDigestCode.ACTIONS_RAW, digest("actions")),
        InputDigest(InputDigestCode.CONFIGURATION_RAW, digest("config")),
    )
    finalized = make_failure(inputs=inputs).finalized()
    assert finalized.failure_sha256 is not None
    assert [item.code for item in finalized.available_inputs] == [
        InputDigestCode.CONFIGURATION_RAW,
        InputDigestCode.ACTIONS_RAW,
    ]
    assert finalized.finalized() is finalized
    finalized.validate_hash()
    assert OperationalFailure.from_primitive(finalized.to_primitive()) == finalized


def test_bad_self_hash_refuses_during_construction_and_parse() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises bad self hash refuses during construction and parse; each refusal
    must retain its exact stage, exit code, and structured evidence contract.
    """
    finalized = make_failure().finalized()
    with pytest.raises(ValueError):
        replace(finalized, failure_sha256=digest("wrong"))
    primitive = finalized.to_primitive()
    with pytest.raises(ValueError):
        OperationalFailure.from_primitive({**primitive, "failure_sha256": digest("wrong")})
    with pytest.raises(ValueError):
        OperationalFailure.from_primitive({**primitive, "failure_sha256": None})


def test_wrong_stage_and_exit_code_refuse() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises wrong stage and exit code refuse; each refusal must retain its exact
    stage, exit code, and structured evidence contract.
    """
    failure = make_failure()
    with pytest.raises(ValueError, match="stage"):
        replace(failure, stage=OperationalStage.INPUT_ARTIFACT)
    with pytest.raises(ValueError, match="exit_code"):
        replace(failure, exit_code=OperationalExitCode.INTERNAL_PROJECT_FAILURE)

    primitive = failure.finalized().to_primitive()
    with pytest.raises(ValueError, match="stage"):
        OperationalFailure.from_primitive({**primitive, "stage": "INPUT_ARTIFACT"})
    with pytest.raises(ValueError, match="exit_code"):
        OperationalFailure.from_primitive({**primitive, "exit_code": 70})
    with pytest.raises(ValueError, match="unknown operational stage"):
        OperationalFailure.from_primitive({**primitive, "stage": "UNKNOWN"})
    with pytest.raises(ValueError, match="unknown operational exit"):
        OperationalFailure.from_primitive({**primitive, "exit_code": 999})
    with pytest.raises(TypeError, match="exit_code"):
        OperationalFailure.from_primitive({**primitive, "exit_code": True})


def test_duplicate_unknown_and_placeholder_input_digest_refuse() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises duplicate unknown and placeholder input digest refuse; each refusal
    must retain its exact stage, exit code, and structured evidence contract.
    """
    item = InputDigest(InputDigestCode.CONFIGURATION_RAW, digest("config"))
    with pytest.raises(ValueError, match="duplicate"):
        make_failure(inputs=(item, item))
    with pytest.raises(TypeError, match="available_inputs"):
        replace(make_failure(), available_inputs=cast(Any, "bad"))
    with pytest.raises(TypeError, match="available_inputs"):
        replace(make_failure(), available_inputs=(cast(Any, object()),))

    primitive = make_failure().finalized().to_primitive()
    with pytest.raises(ValueError, match="unknown input digest"):
        OperationalFailure.from_primitive(
            {
                **primitive,
                "available_inputs": [{"code": "UNKNOWN", "sha256": digest("x")}],
            }
        )
    with pytest.raises(ValueError, match="placeholder"):
        OperationalFailure.from_primitive(
            {
                **primitive,
                "available_inputs": [{"code": "CONFIGURATION_RAW", "sha256": "0" * 64}],
            }
        )


def test_tool_observation_identity_states() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises tool observation identity states; each refusal must retain its exact
    stage, exit code, and structured evidence contract.
    """
    unbound = OperationalToolObservation("0.1.0a3", "UNBOUND", None)
    assert unbound.distribution_sha256 is None
    mismatch = OperationalToolObservation("0.1.0a3", "MISMATCH", None)
    assert mismatch.distribution_sha256 is None
    verified = OperationalToolObservation(
        "0.1.0a3", "VERIFIED_INSTALLED_DISTRIBUTION", digest("distribution")
    )
    assert verified.distribution_sha256 == digest("distribution")


def test_comparison_status_or_comparison_reason_fields_refuse() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises comparison status or comparison reason fields refuse; each refusal
    must retain its exact stage, exit code, and structured evidence contract.
    """
    primitive = make_failure().finalized().to_primitive()
    with pytest.raises(ValueError, match="unknown fields"):
        OperationalFailure.from_primitive({**primitive, "status": "COVERAGE_INSUFFICIENT"})
    reason = dict(cast(dict[str, object], primitive["reason"]))
    reason["code"] = "TRACE_MALFORMED"
    with pytest.raises(ValueError, match="unknown operational reason"):
        OperationalFailure.from_primitive({**primitive, "reason": reason})


def test_precontract_failure_has_no_fabricated_comparison_identities() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises precontract failure has no fabricated comparison identities; each
    refusal must retain its exact stage, exit code, and structured evidence contract.
    """
    primitive = make_failure().finalized().to_primitive()
    forbidden = {
        "status",
        "comparison_contract_sha256",
        "baseline_model_closure_sha256",
        "candidate_model_closure_sha256",
        "initial_state_semantic_sha256",
        "actions_semantic_sha256",
        "alignment",
        "monitored_joints",
    }
    assert forbidden.isdisjoint(primitive)


@pytest.mark.parametrize("field", ["field", "object_name"])
def test_operational_optional_strings_null_nonempty_empty(field: str) -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises operational optional strings null nonempty empty; each refusal must
    retain its exact stage, exit code, and structured evidence contract.
    """
    base = {
        "code": "MONITORED_JOINT_SET_EMPTY",
        "role": "comparison",
        "field": None,
        "object_name": None,
        "evidence": {},
    }
    assert OperationalReason.from_primitive(base).to_primitive()[field] is None
    assert OperationalReason.from_primitive({**base, field: "x"}).to_primitive()[field] == "x"
    with pytest.raises(ValueError, match=field):
        OperationalReason.from_primitive({**base, field: ""})


def test_operational_reason_strictness() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises operational reason strictness; each refusal must retain its exact
    stage, exit code, and structured evidence contract.
    """
    with pytest.raises(TypeError):
        OperationalReason(cast(Any, "MONITORED_JOINT_SET_EMPTY"), None, None, None, {})
    with pytest.raises(ValueError, match="role"):
        OperationalReason(
            OperationalReasonCode.MONITORED_JOINT_SET_EMPTY,
            cast(Any, "other"),
            None,
            None,
            {},
        )
    with pytest.raises(TypeError, match="evidence"):
        OperationalReason(
            OperationalReasonCode.MONITORED_JOINT_SET_EMPTY,
            None,
            None,
            None,
            cast(Any, []),
        )
    with pytest.raises(TypeError, match="evidence"):
        OperationalReason.from_primitive(
            {
                "code": "MONITORED_JOINT_SET_EMPTY",
                "role": None,
                "field": None,
                "object_name": None,
                "evidence": [],
            }
        )
    with pytest.raises(ValueError, match="role"):
        OperationalReason.from_primitive(
            {
                "code": "MONITORED_JOINT_SET_EMPTY",
                "role": "bad",
                "field": None,
                "object_name": None,
                "evidence": {},
            }
        )


def test_operational_failure_direct_nested_type_strictness() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises operational failure direct nested type strictness; each refusal must
    retain its exact stage, exit code, and structured evidence contract.
    """
    failure = make_failure()
    mutations: list[tuple[str, object]] = [
        ("tool", object()),
        ("stage", "TOLERANCE_CONTRACT"),
        ("reason", object()),
        ("environment", object()),
        ("exit_code", 64),
    ]
    for field, value in mutations:
        with pytest.raises((TypeError, ValueError)) as exc_info:
            replace(failure, **{field: value})
        assert not isinstance(exc_info.value, AttributeError)
    assert replace(failure, available_inputs=[]).available_inputs == ()


def test_operational_failure_schema_operation_and_unknown_fields() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises operational failure schema operation and unknown fields; each
    refusal must retain its exact stage, exit code, and structured evidence contract.
    """
    failure = make_failure()
    with pytest.raises(ValueError, match="schema_version"):
        replace(failure, schema_version="wrong")
    with pytest.raises(ValueError, match="failure_rule_schema"):
        replace(failure, failure_rule_schema="wrong")
    with pytest.raises(ValueError, match="failure_rule_schema_version"):
        replace(failure, failure_rule_schema_version=2)
    with pytest.raises(ValueError, match="operation"):
        replace(failure, operation="other")
    # The operation registry is exactly these two installed commands.
    assert replace(failure, operation="compare").operation == "compare"
    assert replace(failure, operation="audit-timestep").operation == "audit-timestep"
    for rejected in ("Compare", "audit_timestep", "audit-timestep ", "", "audit"):
        with pytest.raises(ValueError, match="operation"):
            replace(failure, operation=rejected)
    primitive = failure.finalized().to_primitive()
    with pytest.raises(ValueError, match="unknown fields"):
        OperationalFailure.from_primitive({**primitive, "extra": 1})
    missing = dict(primitive)
    del missing["tool"]
    with pytest.raises(ValueError, match="missing fields"):
        OperationalFailure.from_primitive(missing)


def test_operational_environment_and_defensive_helpers(
    green_candidate: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises operational environment and defensive helpers; each refusal must
    retain its exact stage, exit code, and structured evidence contract.
    """
    from types import MappingProxyType

    import metrifid.operational as operational_module
    from metrifid import finalize_receipt

    environment = finalize_receipt(green_candidate).environment
    reason = OperationalReason(
        OperationalReasonCode.UNSUPPORTED_PLATFORM,
        None,
        None,
        None,
        {},
    )
    failure = OperationalFailure(
        "metrifid.operational_failure",
        1,
        "metrifid.operational_failure_rules",
        1,
        OperationalToolObservation(
            "0.1.0a3", "VERIFIED_INSTALLED_DISTRIBUTION", digest("distribution")
        ),
        "compare",
        reason.code.stage,
        reason,
        (),
        environment,
        reason.code.exit_code,
        None,
    ).finalized()
    assert OperationalFailure.from_primitive(failure.to_primitive()) == failure

    frozen = operational_module._freeze_object(MappingProxyType({"x": 1}), "evidence")
    assert operational_module.thaw_canonical(frozen) == {"x": 1}

    monkeypatch.setattr(operational_module, "thaw_canonical", lambda value: [])
    with pytest.raises(TypeError, match="canonical object"):
        operational_module._freeze_object(MappingProxyType({"x": 1}), "evidence")

    monkeypatch.setattr(operational_module, "thaw_canonical", lambda value: {})
    monkeypatch.setattr(operational_module, "freeze_canonical", lambda value: ())
    with pytest.raises(TypeError, match="canonical object"):
        operational_module._freeze_object(MappingProxyType({"x": 1}), "evidence")

    monkeypatch.undo()

    with pytest.raises(TypeError, match="must be an object"):
        OperationalFailure.from_primitive([])

    primitive = failure.to_primitive()
    primitive["available_inputs"] = "not-an-array"
    with pytest.raises(TypeError, match="available_inputs must be an array"):
        OperationalFailure.from_primitive(primitive)

    with pytest.raises(TypeError, match="version must be a string"):
        OperationalToolObservation.from_primitive(
            {
                "version": 1,
                "execution_identity_state": "UNBOUND",
                "distribution_sha256": None,
            }
        )
