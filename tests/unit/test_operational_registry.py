"""Exact operational stage, reason, input-digest, and exit registries."""

from __future__ import annotations

import hashlib
from typing import Any, cast

import pytest

from metrifid import (
    InputDigest,
    InputDigestCode,
    OperationalExitCode,
    OperationalReasonCode,
    OperationalStage,
    OperationalToolObservation,
)
from metrifid.operational import OPERATIONAL_REASON_REGISTRY


def digest(label: str) -> str:
    """Compute the canonical digest value used by operational registry fixtures.

    Content addressing keeps the mutation boundary explicit for operational registry.
    """
    return hashlib.sha256(label.encode()).hexdigest()


EXPECTED_STAGES = [
    "INVOCATION",
    "EXECUTION_IDENTITY",
    "ENVIRONMENT",
    "CONFIGURATION",
    "MODEL_CLOSURE",
    "MODEL_COMPILE",
    "SEMANTIC_IDENTITY",
    "INPUT_ARTIFACT",
    "TIME_CONTRACT",
    "TOLERANCE_CONTRACT",
    "OUTPUT",
    "CHILD_PROCESS",
    "INTERNAL",
]

EXPECTED_INPUTS = [
    "CONFIGURATION_RAW",
    "BASELINE_MODEL_ENTRYPOINT_RAW",
    "CANDIDATE_MODEL_ENTRYPOINT_RAW",
    "BASELINE_MODEL_CLOSURE",
    "CANDIDATE_MODEL_CLOSURE",
    "INITIAL_STATE_RAW",
    "ACTIONS_RAW",
    "ALIASES_RAW",
    "COMPARISON_CONTRACT",
]


def test_operational_registries_are_exact_total_and_unique() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises operational registries are exact total and unique; each refusal must
    retain its exact stage, exit code, and structured evidence contract.
    """
    assert [stage.value for stage in OperationalStage] == EXPECTED_STAGES
    assert [code.value for code in InputDigestCode] == EXPECTED_INPUTS
    assert len(OperationalReasonCode) == 78
    assert set(OPERATIONAL_REASON_REGISTRY) == set(OperationalReasonCode)
    assert len({code.value for code in OperationalReasonCode}) == 78
    assert all(rule.code is code for code, rule in OPERATIONAL_REASON_REGISTRY.items())


def test_each_reason_has_exactly_one_stage_and_exit_code() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises each reason has exactly one stage and exit code; each refusal must
    retain its exact stage, exit code, and structured evidence contract.
    """
    internal = {
        OperationalReasonCode.CHILD_PROCESS_START_FAILED,
        OperationalReasonCode.CHILD_PROCESS_SUPERVISION_FAILED,
        OperationalReasonCode.INTERNAL_INVARIANT_FAILED,
    }
    for code in OperationalReasonCode:
        rule = OPERATIONAL_REASON_REGISTRY[code]
        assert code.stage is rule.stage
        assert code.exit_code is rule.exit_code
        expected_exit = (
            OperationalExitCode.INTERNAL_PROJECT_FAILURE
            if code in internal
            else OperationalExitCode.INVALID_INVOCATION_INPUT_OUTPUT
        )
        assert rule.exit_code is expected_exit
    assert {
        code for code, rule in OPERATIONAL_REASON_REGISTRY.items() if rule.exit_code == 70
    } == internal


def test_stage_group_boundaries() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises stage group boundaries; each refusal must retain its exact stage,
    exit code, and structured evidence contract.
    """
    expected = {
        OperationalReasonCode.INVALID_CLI_INVOCATION: OperationalStage.INVOCATION,
        OperationalReasonCode.EXECUTING_CODE_NOT_INSTALLED_DISTRIBUTION: OperationalStage.EXECUTION_IDENTITY,
        OperationalReasonCode.UNSUPPORTED_PLATFORM: OperationalStage.ENVIRONMENT,
        OperationalReasonCode.CONFIGURATION_IO_FAILED: OperationalStage.CONFIGURATION,
        OperationalReasonCode.MODEL_ROOT_INVALID: OperationalStage.MODEL_CLOSURE,
        OperationalReasonCode.BASELINE_MODEL_COMPILE_ERROR: OperationalStage.MODEL_COMPILE,
        OperationalReasonCode.JOINT_NAME_MISSING: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.STATE_ARTIFACT_INVALID: OperationalStage.INPUT_ARTIFACT,
        OperationalReasonCode.DECLARED_STEP_DT_INVALID: OperationalStage.TIME_CONTRACT,
        OperationalReasonCode.MONITORED_JOINT_SET_EMPTY: OperationalStage.TOLERANCE_CONTRACT,
        OperationalReasonCode.OUTPUT_PATH_INVALID: OperationalStage.OUTPUT,
        OperationalReasonCode.CHILD_PROCESS_START_FAILED: OperationalStage.CHILD_PROCESS,
        OperationalReasonCode.INTERNAL_INVARIANT_FAILED: OperationalStage.INTERNAL,
    }
    for code, stage in expected.items():
        assert code.stage is stage


def test_input_digest_strictness_and_placeholder_refusal() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises input digest strictness and placeholder refusal; each refusal must
    retain its exact stage, exit code, and structured evidence contract.
    """
    item = InputDigest(InputDigestCode.CONFIGURATION_RAW, digest("config"))
    assert InputDigest.from_primitive(item.to_primitive()) == item
    with pytest.raises(TypeError):
        InputDigest(cast(Any, "CONFIGURATION_RAW"), digest("x"))
    with pytest.raises(ValueError, match="placeholder"):
        InputDigest(InputDigestCode.CONFIGURATION_RAW, "0" * 64)
    with pytest.raises(ValueError, match="unknown input digest"):
        InputDigest.from_primitive({"code": "UNKNOWN", "sha256": digest("x")})
    with pytest.raises((TypeError, ValueError)):
        InputDigest.from_primitive({"code": "CONFIGURATION_RAW"})


@pytest.mark.parametrize("state", ["UNBOUND", "MISMATCH"])
def test_unverified_tool_identity_requires_null_hash(state: str) -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises unverified tool identity requires null hash; each refusal must
    retain its exact stage, exit code, and structured evidence contract.
    """
    tool = OperationalToolObservation("0.1.0a3", cast(Any, state), None)
    assert OperationalToolObservation.from_primitive(tool.to_primitive()) == tool
    with pytest.raises(ValueError, match="null"):
        OperationalToolObservation("0.1.0a3", cast(Any, state), digest("distribution"))


def test_verified_tool_identity_requires_hash() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises verified tool identity requires hash; each refusal must retain its
    exact stage, exit code, and structured evidence contract.
    """
    tool = OperationalToolObservation(
        "0.1.0a3", "VERIFIED_INSTALLED_DISTRIBUTION", digest("distribution")
    )
    assert OperationalToolObservation.from_primitive(tool.to_primitive()) == tool
    with pytest.raises(ValueError, match="requires"):
        OperationalToolObservation("0.1.0a3", "VERIFIED_INSTALLED_DISTRIBUTION", None)
    with pytest.raises(ValueError, match="registry"):
        OperationalToolObservation("0.1.0a3", cast(Any, "OTHER"), None)
    with pytest.raises(ValueError, match="nonempty"):
        OperationalToolObservation("", "UNBOUND", None)
    with pytest.raises(ValueError, match="registry"):
        OperationalToolObservation.from_primitive(
            {"version": "0.1", "execution_identity_state": "OTHER", "distribution_sha256": None}
        )


# The exact reason -> stage -> exit triples of every reason that existed before the Certify
# pass replaced the positional _STAGE_GROUPS enum slices with an explicit per-reason mapping.
# Positional slices silently rebound existing codes whenever a member was inserted; this table
# is the pin that makes any such rebinding a test failure rather than a silent contract change.
PREEXISTING_REASON_TRIPLES: tuple[tuple[str, str, int], ...] = (
    ("INVALID_CLI_INVOCATION", "INVOCATION", 64),
    ("CONFIGURATION_IO_FAILED", "CONFIGURATION", 64),
    ("CONFIGURATION_PARSE_FAILED", "CONFIGURATION", 64),
    ("OUTPUT_PATH_INVALID", "OUTPUT", 64),
    ("OUTPUT_DIRECTORY_NOT_EMPTY", "OUTPUT", 64),
    ("OUTPUT_WRITE_FAILED", "OUTPUT", 64),
    ("CHILD_PROCESS_START_FAILED", "CHILD_PROCESS", 70),
    ("CHILD_PROCESS_SUPERVISION_FAILED", "CHILD_PROCESS", 70),
    ("INTERNAL_INVARIANT_FAILED", "INTERNAL", 70),
    ("EXECUTING_CODE_NOT_INSTALLED_DISTRIBUTION", "EXECUTION_IDENTITY", 64),
    ("MIXED_METRIFID_MODULE_ROOTS", "EXECUTION_IDENTITY", 64),
    ("EDITABLE_INSTALL_UNSUPPORTED", "EXECUTION_IDENTITY", 64),
    ("DISTRIBUTION_MANIFEST_INVALID", "EXECUTION_IDENTITY", 64),
    ("UNSUPPORTED_PLATFORM", "ENVIRONMENT", 64),
    ("UNSUPPORTED_PYTHON_VERSION", "ENVIRONMENT", 64),
    ("UNSUPPORTED_MUJOCO_VERSION", "ENVIRONMENT", 64),
    ("MUJOCO_PYTHON_NATIVE_VERSION_MISMATCH", "ENVIRONMENT", 64),
    ("UNSUPPORTED_PLUGIN_STATE", "ENVIRONMENT", 64),
    ("UNSUPPORTED_USER_CALLBACK", "ENVIRONMENT", 64),
    ("UNSUPPORTED_HISTORY_STATE", "ENVIRONMENT", 64),
    ("UNSUPPORTED_MOCAP_STATE", "ENVIRONMENT", 64),
    ("BASELINE_MODEL_COMPILE_ERROR", "MODEL_COMPILE", 64),
    ("CANDIDATE_MODEL_COMPILE_ERROR", "MODEL_COMPILE", 64),
    ("BASELINE_MODEL_COMPILE_WARNING", "MODEL_COMPILE", 64),
    ("CANDIDATE_MODEL_COMPILE_WARNING", "MODEL_COMPILE", 64),
    ("MODEL_ROOT_INVALID", "MODEL_CLOSURE", 64),
    ("MODEL_ENTRYPOINT_INVALID", "MODEL_CLOSURE", 64),
    ("MODEL_CLOSURE_PATH_ESCAPE", "MODEL_CLOSURE", 64),
    ("MODEL_CLOSURE_SYMLINK_REFUSED", "MODEL_CLOSURE", 64),
    ("MODEL_CLOSURE_MEMBER_INVALID", "MODEL_CLOSURE", 64),
    ("MODEL_CLOSURE_BUDGET_EXCEEDED", "MODEL_CLOSURE", 64),
    ("MODEL_CLOSURE_MUTATED", "MODEL_CLOSURE", 64),
    ("JOINT_NAME_MISSING", "SEMANTIC_IDENTITY", 64),
    ("JOINT_NAME_DUPLICATE", "SEMANTIC_IDENTITY", 64),
    ("JOINT_IDENTITY_MISSING", "SEMANTIC_IDENTITY", 64),
    ("JOINT_TYPE_MISMATCH", "SEMANTIC_IDENTITY", 64),
    ("JOINT_QPOS_WIDTH_MISMATCH", "SEMANTIC_IDENTITY", 64),
    ("JOINT_QVEL_WIDTH_MISMATCH", "SEMANTIC_IDENTITY", 64),
    ("ACTUATOR_IDENTITY_MISSING", "SEMANTIC_IDENTITY", 64),
    ("ACTUATOR_IDENTITY_AMBIGUOUS", "SEMANTIC_IDENTITY", 64),
    ("ACTUATOR_TRANSMISSION_MISMATCH", "SEMANTIC_IDENTITY", 64),
    ("ACTUATOR_TARGET_MISMATCH", "SEMANTIC_IDENTITY", 64),
    ("ACTUATOR_ACTIVATION_FAMILY_MISMATCH", "SEMANTIC_IDENTITY", 64),
    ("ACTUATOR_ACTIVATION_WIDTH_MISMATCH", "SEMANTIC_IDENTITY", 64),
    ("ALIAS_SCHEMA_INVALID", "SEMANTIC_IDENTITY", 64),
    ("ALIAS_CLOSURE_HASH_MISMATCH", "SEMANTIC_IDENTITY", 64),
    ("ALIAS_BINDING_DUPLICATE", "SEMANTIC_IDENTITY", 64),
    ("ALIAS_SELECTOR_NO_MATCH", "SEMANTIC_IDENTITY", 64),
    ("ALIAS_SELECTOR_AMBIGUOUS", "SEMANTIC_IDENTITY", 64),
    ("MONITORED_JOINT_NOT_ALIGNED", "SEMANTIC_IDENTITY", 64),
    ("MODEL_LAYOUT_INCOMPATIBLE", "SEMANTIC_IDENTITY", 64),
    ("STATE_ARTIFACT_INVALID", "INPUT_ARTIFACT", 64),
    ("ACTIONS_ARTIFACT_INVALID", "INPUT_ARTIFACT", 64),
    ("NPZ_MEMBER_SET_INVALID", "INPUT_ARTIFACT", 64),
    ("NPZ_DUPLICATE_MEMBER", "INPUT_ARTIFACT", 64),
    ("NPZ_ENCRYPTED_MEMBER", "INPUT_ARTIFACT", 64),
    ("NPZ_PATH_MEMBER_INVALID", "INPUT_ARTIFACT", 64),
    ("NPZ_OBJECT_ARRAY_REFUSED", "INPUT_ARTIFACT", 64),
    ("NPZ_DTYPE_MISMATCH", "INPUT_ARTIFACT", 64),
    ("NPZ_SIZE_BUDGET_EXCEEDED", "INPUT_ARTIFACT", 64),
    ("STATE_NAME_SET_MISMATCH", "INPUT_ARTIFACT", 64),
    ("STATE_OFFSET_INVALID", "INPUT_ARTIFACT", 64),
    ("STATE_WIDTH_MISMATCH", "INPUT_ARTIFACT", 64),
    ("ACTION_NAME_SET_MISMATCH", "INPUT_ARTIFACT", 64),
    ("ACTION_SHAPE_MISMATCH", "INPUT_ARTIFACT", 64),
    ("INPUT_NONFINITE_VALUE", "INPUT_ARTIFACT", 64),
    ("MONITORED_JOINT_SET_EMPTY", "TOLERANCE_CONTRACT", 64),
    ("TOLERANCE_MISSING", "TOLERANCE_CONTRACT", 64),
    ("TOLERANCE_FIELD_INVALID", "TOLERANCE_CONTRACT", 64),
    ("TOLERANCE_UNIT_MISMATCH", "TOLERANCE_CONTRACT", 64),
    ("TOLERANCE_NONPOSITIVE", "TOLERANCE_CONTRACT", 64),
    ("DECLARED_STEP_DT_INVALID", "TIME_CONTRACT", 64),
    ("DECLARED_STEP_DT_MISMATCH", "TIME_CONTRACT", 64),
    ("CONTROL_DT_INVALID", "TIME_CONTRACT", 64),
    ("CONTROL_GRID_NONINTEGRAL", "TIME_CONTRACT", 64),
    ("CONTROL_INTERVAL_COUNT_INVALID", "TIME_CONTRACT", 64),
)


def test_every_preexisting_reason_keeps_its_stage_and_exit_binding() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises every preexisting reason keeps its stage and exit binding; each
    refusal must retain its exact stage, exit code, and structured evidence contract.
    """
    for name, stage, exit_code in PREEXISTING_REASON_TRIPLES:
        rule = OPERATIONAL_REASON_REGISTRY[OperationalReasonCode(name)]
        assert rule.stage.value == stage, name
        assert int(rule.exit_code) == exit_code, name


def test_the_pinned_triples_cover_every_reason_declared_before_the_certify_pass() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises the pinned triples cover every reason declared before the certify
    pass; each refusal must retain its exact stage, exit code, and structured evidence contract.
    """
    pinned = {name for name, _, _ in PREEXISTING_REASON_TRIPLES}
    assert len(pinned) == len(PREEXISTING_REASON_TRIPLES) == 76
    added = {code.value for code in OperationalReasonCode} - pinned
    assert added == {"COMPILED_ARTIFACT_INVALID", "COMPILED_ARTIFACT_SIZE_EXCEEDED"}


def test_the_certify_reasons_are_appended_after_every_preexisting_reason() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises the certify reasons are appended after every preexisting reason;
    each refusal must retain its exact stage, exit code, and structured evidence contract.
    """
    declared = [code.value for code in OperationalReasonCode]
    assert declared[:76] == [name for name, _, _ in PREEXISTING_REASON_TRIPLES]
    for name in ("COMPILED_ARTIFACT_INVALID", "COMPILED_ARTIFACT_SIZE_EXCEEDED"):
        rule = OPERATIONAL_REASON_REGISTRY[OperationalReasonCode(name)]
        assert rule.stage is OperationalStage.MODEL_COMPILE
        assert rule.exit_code is OperationalExitCode.INVALID_INVOCATION_INPUT_OUTPUT


def test_the_registry_binds_each_reason_exactly_once() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises the registry binds each reason exactly once; each refusal must
    retain its exact stage, exit code, and structured evidence contract.
    """
    assert set(OPERATIONAL_REASON_REGISTRY) == set(OperationalReasonCode)
    assert len(OPERATIONAL_REASON_REGISTRY) == len(list(OperationalReasonCode))
