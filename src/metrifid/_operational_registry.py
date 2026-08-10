"""Strict pre-contract operational-failure ABI and frozen registries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Literal, TypeAlias

from .errors import OperationalExitCode

ExecutionIdentityState: TypeAlias = Literal[
    "VERIFIED_INSTALLED_DISTRIBUTION", "UNBOUND", "MISMATCH"
]

_OPERATIONAL_FAILURE_SCHEMA = "metrifid.operational_failure"
_OPERATIONAL_FAILURE_SCHEMA_VERSION = 1
_OPERATIONAL_RULE_SCHEMA = "metrifid.operational_failure_rules"
_OPERATIONAL_RULE_SCHEMA_VERSION = 1

# The exact installed operations that may own an operational failure. A candidate
# failure raised by the accepted comparison engine stays "compare"; only audit-level
# invocation, configuration, generation, output, or internal failures are
# "audit-timestep".
_OPERATIONS: frozenset[str] = frozenset({"compare", "audit-timestep", "certify"})


class OperationalStage(StrEnum):
    """The exact ordered operational-stage registry."""

    INVOCATION = "INVOCATION"
    EXECUTION_IDENTITY = "EXECUTION_IDENTITY"
    ENVIRONMENT = "ENVIRONMENT"
    CONFIGURATION = "CONFIGURATION"
    MODEL_CLOSURE = "MODEL_CLOSURE"
    MODEL_COMPILE = "MODEL_COMPILE"
    SEMANTIC_IDENTITY = "SEMANTIC_IDENTITY"
    INPUT_ARTIFACT = "INPUT_ARTIFACT"
    TIME_CONTRACT = "TIME_CONTRACT"
    TOLERANCE_CONTRACT = "TOLERANCE_CONTRACT"
    OUTPUT = "OUTPUT"
    CHILD_PROCESS = "CHILD_PROCESS"
    INTERNAL = "INTERNAL"


class OperationalReasonCode(StrEnum):
    """The exact ordered pre-contract operational-reason registry."""

    INVALID_CLI_INVOCATION = "INVALID_CLI_INVOCATION"
    CONFIGURATION_IO_FAILED = "CONFIGURATION_IO_FAILED"
    CONFIGURATION_PARSE_FAILED = "CONFIGURATION_PARSE_FAILED"
    OUTPUT_PATH_INVALID = "OUTPUT_PATH_INVALID"
    OUTPUT_DIRECTORY_NOT_EMPTY = "OUTPUT_DIRECTORY_NOT_EMPTY"
    OUTPUT_WRITE_FAILED = "OUTPUT_WRITE_FAILED"
    CHILD_PROCESS_START_FAILED = "CHILD_PROCESS_START_FAILED"
    CHILD_PROCESS_SUPERVISION_FAILED = "CHILD_PROCESS_SUPERVISION_FAILED"
    INTERNAL_INVARIANT_FAILED = "INTERNAL_INVARIANT_FAILED"
    EXECUTING_CODE_NOT_INSTALLED_DISTRIBUTION = "EXECUTING_CODE_NOT_INSTALLED_DISTRIBUTION"
    MIXED_METRIFID_MODULE_ROOTS = "MIXED_METRIFID_MODULE_ROOTS"
    EDITABLE_INSTALL_UNSUPPORTED = "EDITABLE_INSTALL_UNSUPPORTED"
    DISTRIBUTION_MANIFEST_INVALID = "DISTRIBUTION_MANIFEST_INVALID"
    UNSUPPORTED_PLATFORM = "UNSUPPORTED_PLATFORM"
    UNSUPPORTED_PYTHON_VERSION = "UNSUPPORTED_PYTHON_VERSION"
    UNSUPPORTED_MUJOCO_VERSION = "UNSUPPORTED_MUJOCO_VERSION"
    MUJOCO_PYTHON_NATIVE_VERSION_MISMATCH = "MUJOCO_PYTHON_NATIVE_VERSION_MISMATCH"
    UNSUPPORTED_PLUGIN_STATE = "UNSUPPORTED_PLUGIN_STATE"
    UNSUPPORTED_USER_CALLBACK = "UNSUPPORTED_USER_CALLBACK"
    UNSUPPORTED_HISTORY_STATE = "UNSUPPORTED_HISTORY_STATE"
    UNSUPPORTED_MOCAP_STATE = "UNSUPPORTED_MOCAP_STATE"
    BASELINE_MODEL_COMPILE_ERROR = "BASELINE_MODEL_COMPILE_ERROR"
    CANDIDATE_MODEL_COMPILE_ERROR = "CANDIDATE_MODEL_COMPILE_ERROR"
    BASELINE_MODEL_COMPILE_WARNING = "BASELINE_MODEL_COMPILE_WARNING"
    CANDIDATE_MODEL_COMPILE_WARNING = "CANDIDATE_MODEL_COMPILE_WARNING"
    MODEL_ROOT_INVALID = "MODEL_ROOT_INVALID"
    MODEL_ENTRYPOINT_INVALID = "MODEL_ENTRYPOINT_INVALID"
    MODEL_CLOSURE_PATH_ESCAPE = "MODEL_CLOSURE_PATH_ESCAPE"
    MODEL_CLOSURE_SYMLINK_REFUSED = "MODEL_CLOSURE_SYMLINK_REFUSED"
    MODEL_CLOSURE_MEMBER_INVALID = "MODEL_CLOSURE_MEMBER_INVALID"
    MODEL_CLOSURE_BUDGET_EXCEEDED = "MODEL_CLOSURE_BUDGET_EXCEEDED"
    MODEL_CLOSURE_MUTATED = "MODEL_CLOSURE_MUTATED"
    JOINT_NAME_MISSING = "JOINT_NAME_MISSING"
    JOINT_NAME_DUPLICATE = "JOINT_NAME_DUPLICATE"
    JOINT_IDENTITY_MISSING = "JOINT_IDENTITY_MISSING"
    JOINT_TYPE_MISMATCH = "JOINT_TYPE_MISMATCH"
    JOINT_QPOS_WIDTH_MISMATCH = "JOINT_QPOS_WIDTH_MISMATCH"
    JOINT_QVEL_WIDTH_MISMATCH = "JOINT_QVEL_WIDTH_MISMATCH"
    ACTUATOR_IDENTITY_MISSING = "ACTUATOR_IDENTITY_MISSING"
    ACTUATOR_IDENTITY_AMBIGUOUS = "ACTUATOR_IDENTITY_AMBIGUOUS"
    ACTUATOR_TRANSMISSION_MISMATCH = "ACTUATOR_TRANSMISSION_MISMATCH"
    ACTUATOR_TARGET_MISMATCH = "ACTUATOR_TARGET_MISMATCH"
    ACTUATOR_ACTIVATION_FAMILY_MISMATCH = "ACTUATOR_ACTIVATION_FAMILY_MISMATCH"
    ACTUATOR_ACTIVATION_WIDTH_MISMATCH = "ACTUATOR_ACTIVATION_WIDTH_MISMATCH"
    ALIAS_SCHEMA_INVALID = "ALIAS_SCHEMA_INVALID"
    ALIAS_CLOSURE_HASH_MISMATCH = "ALIAS_CLOSURE_HASH_MISMATCH"
    ALIAS_BINDING_DUPLICATE = "ALIAS_BINDING_DUPLICATE"
    ALIAS_SELECTOR_NO_MATCH = "ALIAS_SELECTOR_NO_MATCH"
    ALIAS_SELECTOR_AMBIGUOUS = "ALIAS_SELECTOR_AMBIGUOUS"
    MONITORED_JOINT_NOT_ALIGNED = "MONITORED_JOINT_NOT_ALIGNED"
    MODEL_LAYOUT_INCOMPATIBLE = "MODEL_LAYOUT_INCOMPATIBLE"
    STATE_ARTIFACT_INVALID = "STATE_ARTIFACT_INVALID"
    ACTIONS_ARTIFACT_INVALID = "ACTIONS_ARTIFACT_INVALID"
    NPZ_MEMBER_SET_INVALID = "NPZ_MEMBER_SET_INVALID"
    NPZ_DUPLICATE_MEMBER = "NPZ_DUPLICATE_MEMBER"
    NPZ_ENCRYPTED_MEMBER = "NPZ_ENCRYPTED_MEMBER"
    NPZ_PATH_MEMBER_INVALID = "NPZ_PATH_MEMBER_INVALID"
    NPZ_OBJECT_ARRAY_REFUSED = "NPZ_OBJECT_ARRAY_REFUSED"
    NPZ_DTYPE_MISMATCH = "NPZ_DTYPE_MISMATCH"
    NPZ_SIZE_BUDGET_EXCEEDED = "NPZ_SIZE_BUDGET_EXCEEDED"
    STATE_NAME_SET_MISMATCH = "STATE_NAME_SET_MISMATCH"
    STATE_OFFSET_INVALID = "STATE_OFFSET_INVALID"
    STATE_WIDTH_MISMATCH = "STATE_WIDTH_MISMATCH"
    ACTION_NAME_SET_MISMATCH = "ACTION_NAME_SET_MISMATCH"
    ACTION_SHAPE_MISMATCH = "ACTION_SHAPE_MISMATCH"
    INPUT_NONFINITE_VALUE = "INPUT_NONFINITE_VALUE"
    MONITORED_JOINT_SET_EMPTY = "MONITORED_JOINT_SET_EMPTY"
    TOLERANCE_MISSING = "TOLERANCE_MISSING"
    TOLERANCE_FIELD_INVALID = "TOLERANCE_FIELD_INVALID"
    TOLERANCE_UNIT_MISMATCH = "TOLERANCE_UNIT_MISMATCH"
    TOLERANCE_NONPOSITIVE = "TOLERANCE_NONPOSITIVE"
    DECLARED_STEP_DT_INVALID = "DECLARED_STEP_DT_INVALID"
    DECLARED_STEP_DT_MISMATCH = "DECLARED_STEP_DT_MISMATCH"
    CONTROL_DT_INVALID = "CONTROL_DT_INVALID"
    CONTROL_GRID_NONINTEGRAL = "CONTROL_GRID_NONINTEGRAL"
    CONTROL_INTERVAL_COUNT_INVALID = "CONTROL_INTERVAL_COUNT_INVALID"
    # Certify-only reasons. Appended at the end so no preexisting member is renumbered.
    COMPILED_ARTIFACT_INVALID = "COMPILED_ARTIFACT_INVALID"
    COMPILED_ARTIFACT_SIZE_EXCEEDED = "COMPILED_ARTIFACT_SIZE_EXCEEDED"

    @property
    def stage(self) -> OperationalStage:
        """Return the frozen operational stage bound to this reason code."""
        return OPERATIONAL_REASON_REGISTRY[self].stage

    @property
    def exit_code(self) -> OperationalExitCode:
        """Return the frozen process exit code bound to this reason code."""
        return OPERATIONAL_REASON_REGISTRY[self].exit_code


class InputDigestCode(StrEnum):
    """The exact ordered available-input digest registry."""

    CONFIGURATION_RAW = "CONFIGURATION_RAW"
    BASELINE_MODEL_ENTRYPOINT_RAW = "BASELINE_MODEL_ENTRYPOINT_RAW"
    CANDIDATE_MODEL_ENTRYPOINT_RAW = "CANDIDATE_MODEL_ENTRYPOINT_RAW"
    BASELINE_MODEL_CLOSURE = "BASELINE_MODEL_CLOSURE"
    CANDIDATE_MODEL_CLOSURE = "CANDIDATE_MODEL_CLOSURE"
    INITIAL_STATE_RAW = "INITIAL_STATE_RAW"
    ACTIONS_RAW = "ACTIONS_RAW"
    ALIASES_RAW = "ALIASES_RAW"
    COMPARISON_CONTRACT = "COMPARISON_CONTRACT"


@dataclass(frozen=True, slots=True)
class OperationalReasonRule:
    """One frozen operational-reason stage and exit binding."""

    code: OperationalReasonCode
    stage: OperationalStage
    exit_code: OperationalExitCode


# Every reason is bound to its stage explicitly. The previous positional enum slices meant that
# inserting a reason code silently rebound existing codes to the wrong stage; the regression test
# in tests/unit/test_operational_registry.py pins all preexisting triples against this mapping.
_REASON_STAGE: Mapping[OperationalReasonCode, OperationalStage] = MappingProxyType(
    {
        OperationalReasonCode.INVALID_CLI_INVOCATION: OperationalStage.INVOCATION,
        OperationalReasonCode.EXECUTING_CODE_NOT_INSTALLED_DISTRIBUTION: OperationalStage.EXECUTION_IDENTITY,
        OperationalReasonCode.MIXED_METRIFID_MODULE_ROOTS: OperationalStage.EXECUTION_IDENTITY,
        OperationalReasonCode.EDITABLE_INSTALL_UNSUPPORTED: OperationalStage.EXECUTION_IDENTITY,
        OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID: OperationalStage.EXECUTION_IDENTITY,
        OperationalReasonCode.UNSUPPORTED_PLATFORM: OperationalStage.ENVIRONMENT,
        OperationalReasonCode.UNSUPPORTED_PYTHON_VERSION: OperationalStage.ENVIRONMENT,
        OperationalReasonCode.UNSUPPORTED_MUJOCO_VERSION: OperationalStage.ENVIRONMENT,
        OperationalReasonCode.MUJOCO_PYTHON_NATIVE_VERSION_MISMATCH: OperationalStage.ENVIRONMENT,
        OperationalReasonCode.UNSUPPORTED_PLUGIN_STATE: OperationalStage.ENVIRONMENT,
        OperationalReasonCode.UNSUPPORTED_USER_CALLBACK: OperationalStage.ENVIRONMENT,
        OperationalReasonCode.UNSUPPORTED_HISTORY_STATE: OperationalStage.ENVIRONMENT,
        OperationalReasonCode.UNSUPPORTED_MOCAP_STATE: OperationalStage.ENVIRONMENT,
        OperationalReasonCode.CONFIGURATION_IO_FAILED: OperationalStage.CONFIGURATION,
        OperationalReasonCode.CONFIGURATION_PARSE_FAILED: OperationalStage.CONFIGURATION,
        OperationalReasonCode.MODEL_ROOT_INVALID: OperationalStage.MODEL_CLOSURE,
        OperationalReasonCode.MODEL_ENTRYPOINT_INVALID: OperationalStage.MODEL_CLOSURE,
        OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE: OperationalStage.MODEL_CLOSURE,
        OperationalReasonCode.MODEL_CLOSURE_SYMLINK_REFUSED: OperationalStage.MODEL_CLOSURE,
        OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID: OperationalStage.MODEL_CLOSURE,
        OperationalReasonCode.MODEL_CLOSURE_BUDGET_EXCEEDED: OperationalStage.MODEL_CLOSURE,
        OperationalReasonCode.MODEL_CLOSURE_MUTATED: OperationalStage.MODEL_CLOSURE,
        OperationalReasonCode.BASELINE_MODEL_COMPILE_ERROR: OperationalStage.MODEL_COMPILE,
        OperationalReasonCode.CANDIDATE_MODEL_COMPILE_ERROR: OperationalStage.MODEL_COMPILE,
        OperationalReasonCode.BASELINE_MODEL_COMPILE_WARNING: OperationalStage.MODEL_COMPILE,
        OperationalReasonCode.CANDIDATE_MODEL_COMPILE_WARNING: OperationalStage.MODEL_COMPILE,
        OperationalReasonCode.JOINT_NAME_MISSING: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.JOINT_NAME_DUPLICATE: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.JOINT_IDENTITY_MISSING: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.JOINT_TYPE_MISMATCH: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.JOINT_QPOS_WIDTH_MISMATCH: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.JOINT_QVEL_WIDTH_MISMATCH: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.ACTUATOR_IDENTITY_MISSING: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.ACTUATOR_IDENTITY_AMBIGUOUS: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.ACTUATOR_TRANSMISSION_MISMATCH: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.ACTUATOR_TARGET_MISMATCH: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.ACTUATOR_ACTIVATION_FAMILY_MISMATCH: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.ACTUATOR_ACTIVATION_WIDTH_MISMATCH: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.ALIAS_SCHEMA_INVALID: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.ALIAS_CLOSURE_HASH_MISMATCH: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.ALIAS_BINDING_DUPLICATE: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.ALIAS_SELECTOR_NO_MATCH: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.ALIAS_SELECTOR_AMBIGUOUS: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.MONITORED_JOINT_NOT_ALIGNED: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.MODEL_LAYOUT_INCOMPATIBLE: OperationalStage.SEMANTIC_IDENTITY,
        OperationalReasonCode.STATE_ARTIFACT_INVALID: OperationalStage.INPUT_ARTIFACT,
        OperationalReasonCode.ACTIONS_ARTIFACT_INVALID: OperationalStage.INPUT_ARTIFACT,
        OperationalReasonCode.NPZ_MEMBER_SET_INVALID: OperationalStage.INPUT_ARTIFACT,
        OperationalReasonCode.NPZ_DUPLICATE_MEMBER: OperationalStage.INPUT_ARTIFACT,
        OperationalReasonCode.NPZ_ENCRYPTED_MEMBER: OperationalStage.INPUT_ARTIFACT,
        OperationalReasonCode.NPZ_PATH_MEMBER_INVALID: OperationalStage.INPUT_ARTIFACT,
        OperationalReasonCode.NPZ_OBJECT_ARRAY_REFUSED: OperationalStage.INPUT_ARTIFACT,
        OperationalReasonCode.NPZ_DTYPE_MISMATCH: OperationalStage.INPUT_ARTIFACT,
        OperationalReasonCode.NPZ_SIZE_BUDGET_EXCEEDED: OperationalStage.INPUT_ARTIFACT,
        OperationalReasonCode.STATE_NAME_SET_MISMATCH: OperationalStage.INPUT_ARTIFACT,
        OperationalReasonCode.STATE_OFFSET_INVALID: OperationalStage.INPUT_ARTIFACT,
        OperationalReasonCode.STATE_WIDTH_MISMATCH: OperationalStage.INPUT_ARTIFACT,
        OperationalReasonCode.ACTION_NAME_SET_MISMATCH: OperationalStage.INPUT_ARTIFACT,
        OperationalReasonCode.ACTION_SHAPE_MISMATCH: OperationalStage.INPUT_ARTIFACT,
        OperationalReasonCode.INPUT_NONFINITE_VALUE: OperationalStage.INPUT_ARTIFACT,
        OperationalReasonCode.DECLARED_STEP_DT_INVALID: OperationalStage.TIME_CONTRACT,
        OperationalReasonCode.DECLARED_STEP_DT_MISMATCH: OperationalStage.TIME_CONTRACT,
        OperationalReasonCode.CONTROL_DT_INVALID: OperationalStage.TIME_CONTRACT,
        OperationalReasonCode.CONTROL_GRID_NONINTEGRAL: OperationalStage.TIME_CONTRACT,
        OperationalReasonCode.CONTROL_INTERVAL_COUNT_INVALID: OperationalStage.TIME_CONTRACT,
        OperationalReasonCode.COMPILED_ARTIFACT_INVALID: OperationalStage.MODEL_COMPILE,
        OperationalReasonCode.COMPILED_ARTIFACT_SIZE_EXCEEDED: OperationalStage.MODEL_COMPILE,
        OperationalReasonCode.MONITORED_JOINT_SET_EMPTY: OperationalStage.TOLERANCE_CONTRACT,
        OperationalReasonCode.TOLERANCE_MISSING: OperationalStage.TOLERANCE_CONTRACT,
        OperationalReasonCode.TOLERANCE_FIELD_INVALID: OperationalStage.TOLERANCE_CONTRACT,
        OperationalReasonCode.TOLERANCE_UNIT_MISMATCH: OperationalStage.TOLERANCE_CONTRACT,
        OperationalReasonCode.TOLERANCE_NONPOSITIVE: OperationalStage.TOLERANCE_CONTRACT,
        OperationalReasonCode.OUTPUT_PATH_INVALID: OperationalStage.OUTPUT,
        OperationalReasonCode.OUTPUT_DIRECTORY_NOT_EMPTY: OperationalStage.OUTPUT,
        OperationalReasonCode.OUTPUT_WRITE_FAILED: OperationalStage.OUTPUT,
        OperationalReasonCode.CHILD_PROCESS_START_FAILED: OperationalStage.CHILD_PROCESS,
        OperationalReasonCode.CHILD_PROCESS_SUPERVISION_FAILED: OperationalStage.CHILD_PROCESS,
        OperationalReasonCode.INTERNAL_INVARIANT_FAILED: OperationalStage.INTERNAL,
    }
)


_INTERNAL_REASONS = {
    OperationalReasonCode.CHILD_PROCESS_START_FAILED,
    OperationalReasonCode.CHILD_PROCESS_SUPERVISION_FAILED,
    OperationalReasonCode.INTERNAL_INVARIANT_FAILED,
}
OPERATIONAL_REASON_REGISTRY: Mapping[OperationalReasonCode, OperationalReasonRule] = (
    MappingProxyType(
        {
            code: OperationalReasonRule(
                code,
                stage,
                (
                    OperationalExitCode.INTERNAL_PROJECT_FAILURE
                    if code in _INTERNAL_REASONS
                    else OperationalExitCode.INVALID_INVOCATION_INPUT_OUTPUT
                ),
            )
            for code, stage in _REASON_STAGE.items()
        }
    )
)
_INPUT_DIGEST_RANK = MappingProxyType({code: rank for rank, code in enumerate(InputDigestCode)})
_EXECUTION_IDENTITY_MISMATCH_REASONS = frozenset(
    {
        OperationalReasonCode.EXECUTING_CODE_NOT_INSTALLED_DISTRIBUTION,
        OperationalReasonCode.MIXED_METRIFID_MODULE_ROOTS,
    }
)
_EXECUTION_IDENTITY_UNBOUND_REASONS = frozenset(
    {
        OperationalReasonCode.EDITABLE_INSTALL_UNSUPPORTED,
        OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID,
    }
)
_POST_IDENTITY_STAGES = frozenset(
    {
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
)
