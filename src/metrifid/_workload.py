"""Strict state/action artifact loading bound to one completed model admission model alignment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from ._model_identity import ModelPairIdentity, SemanticAlignment
from ._npz import LoadedNpz, _load_npz_arrays_from_bytes, load_npz_arrays, refuse
from ._workload_validation import (
    _actions_semantic_sha256,
    _float_array,
    _integer_scalar,
    _make_read_only,
    _offset_array,
    _require_finite,
    _require_runtime_float_array,
    _state_semantic_sha256,
    _unicode_names,
    _unicode_scalar,
    _validate_offsets,
    _validate_widths,
)
from .json_values import require_sha256
from .operational import OperationalReasonCode
from .schemas import ActionsArtifactMetadata, StateArtifactMetadata

_STATE_MEMBERS: Final[frozenset[str]] = frozenset(
    {
        "schema.npy",
        "schema_version.npy",
        "joint_names.npy",
        "qpos_offsets.npy",
        "qpos.npy",
        "qvel_offsets.npy",
        "qvel.npy",
        "actuator_names.npy",
        "act_offsets.npy",
        "act.npy",
    }
)
_ACTIONS_MEMBERS: Final[frozenset[str]] = frozenset(
    {"schema.npy", "schema_version.npy", "actuator_names.npy", "values.npy"}
)
_STATE_SCHEMA: Final[str] = "metrifid.state"
_ACTIONS_SCHEMA: Final[str] = "metrifid.actions"
_SCHEMA_VERSION: Final[int] = 1


@dataclass(frozen=True, slots=True, eq=False)
class LoadedStateArtifact:
    """One immutable state artifact after exact semantic binding to model admission alignment."""

    raw_file_sha256: str
    semantic_sha256: str
    metadata: StateArtifactMetadata
    qpos: npt.NDArray[np.float64]
    qvel: npt.NDArray[np.float64]
    act: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate immutable finite state vectors against their metadata counts."""
        require_sha256(self.raw_file_sha256, "raw_file_sha256")
        require_sha256(self.semantic_sha256, "semantic_sha256")
        if not isinstance(self.metadata, StateArtifactMetadata):
            raise TypeError("metadata must be StateArtifactMetadata")
        for name, array in (("qpos", self.qpos), ("qvel", self.qvel), ("act", self.act)):
            _require_runtime_float_array(array, name, 1)
        if (
            self.qpos.size != self.metadata.qpos_count
            or self.qvel.size != self.metadata.qvel_count
            or self.act.size != self.metadata.act_count
        ):
            raise ValueError("state arrays must match metadata counts")
        if self.semantic_sha256 != _state_semantic_sha256(
            self.metadata, self.qpos, self.qvel, self.act
        ):
            raise ValueError("semantic_sha256 does not match state content")


@dataclass(frozen=True, slots=True, eq=False)
class LoadedActionsArtifact:
    """One immutable action artifact after exact semantic binding to model admission alignment."""

    raw_file_sha256: str
    semantic_sha256: str
    metadata: ActionsArtifactMetadata
    values: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate the immutable finite control matrix against actions metadata."""
        require_sha256(self.raw_file_sha256, "raw_file_sha256")
        require_sha256(self.semantic_sha256, "semantic_sha256")
        if not isinstance(self.metadata, ActionsArtifactMetadata):
            raise TypeError("metadata must be ActionsArtifactMetadata")
        _require_runtime_float_array(self.values, "values", 2)
        if self.values.shape != (
            self.metadata.control_intervals,
            self.metadata.actuator_count,
        ):
            raise ValueError("action values must match metadata shape")
        if self.semantic_sha256 != _actions_semantic_sha256(self.metadata, self.values):
            raise ValueError("semantic_sha256 does not match action content")


@dataclass(frozen=True, slots=True, eq=False)
class WorkloadArtifacts:
    """State and actions bound to one exact completed model admission model-pair identity."""

    model_pair_identity_sha256: str
    state: LoadedStateArtifact
    actions: LoadedActionsArtifact

    def __post_init__(self) -> None:
        """Bind typed state and actions artifacts to one complete model-pair identity."""
        require_sha256(self.model_pair_identity_sha256, "model_pair_identity_sha256")
        if not isinstance(self.state, LoadedStateArtifact):
            raise TypeError("state must be LoadedStateArtifact")
        if not isinstance(self.actions, LoadedActionsArtifact):
            raise TypeError("actions must be LoadedActionsArtifact")
        if self.state.metadata.actuator_names != self.actions.metadata.actuator_names:
            raise ValueError("state and actions must use the same canonical actuator order")


def _validate_state_layout(
    alignment: SemanticAlignment,
    *,
    joint_names: tuple[str, ...],
    actuator_names: tuple[str, ...],
    offsets: tuple[Any, Any, Any],
    values: tuple[Any, Any, Any],
) -> None:
    """Require every declared offset block to cover its aligned width and hold finite values."""
    qpos_offsets, qvel_offsets, act_offsets = offsets
    qpos, qvel, act = values
    _validate_offsets(qpos_offsets, len(joint_names), len(qpos), "qpos_offsets")
    _validate_offsets(qvel_offsets, len(joint_names), len(qvel), "qvel_offsets")
    _validate_offsets(act_offsets, len(actuator_names), len(act), "act_offsets")
    _validate_widths(
        qpos_offsets, tuple(item.baseline_qpos[1] for item in alignment.joints), "qpos_offsets"
    )
    _validate_widths(
        qvel_offsets, tuple(item.baseline_qvel[1] for item in alignment.joints), "qvel_offsets"
    )
    _validate_widths(
        act_offsets, tuple(item.activation_width for item in alignment.actuators), "act_offsets"
    )
    for array, label in ((qpos, "qpos"), (qvel, "qvel"), (act, "act")):
        _require_finite(array, label)


def load_state_artifact(
    path: str | Path,
    model_pair: ModelPairIdentity,
) -> LoadedStateArtifact:
    """Load one strict state NPZ and bind every slice to model admission alignment."""
    alignment = _completed_alignment(model_pair)
    loaded = load_npz_arrays(
        path,
        expected_members=_STATE_MEMBERS,
        invalid_reason=OperationalReasonCode.STATE_ARTIFACT_INVALID,
    )
    return _loaded_state_artifact(loaded, alignment)


def _load_state_artifact_from_bytes(
    raw: bytes,
    model_pair: ModelPairIdentity,
) -> LoadedStateArtifact:
    """Bind one already-frozen state NPZ byte string to the admitted model pair."""
    alignment = _completed_alignment(model_pair)
    loaded = _load_npz_arrays_from_bytes(
        raw,
        expected_members=_STATE_MEMBERS,
        invalid_reason=OperationalReasonCode.STATE_ARTIFACT_INVALID,
    )
    return _loaded_state_artifact(loaded, alignment)


def _loaded_state_artifact(
    loaded: LoadedNpz,
    alignment: SemanticAlignment,
) -> LoadedStateArtifact:
    """Validate and construct one state artifact from preflighted NPZ arrays."""
    arrays = loaded.arrays
    schema, schema_version = _artifact_schema(
        arrays, OperationalReasonCode.STATE_ARTIFACT_INVALID, _STATE_SCHEMA
    )
    joint_names, actuator_names = _state_names(arrays, alignment)
    offsets, values = _state_arrays(arrays)
    _validate_state_layout(
        alignment,
        joint_names=joint_names,
        actuator_names=actuator_names,
        offsets=offsets,
        values=values,
    )
    qpos_offsets, qvel_offsets, act_offsets = offsets
    qpos, qvel, act = values
    metadata = StateArtifactMetadata(
        schema,
        schema_version,
        joint_names,
        qpos_offsets,
        len(qpos),
        qvel_offsets,
        len(qvel),
        actuator_names,
        act_offsets,
        len(act),
    )
    _make_read_only(qpos, qvel, act)
    semantic = _state_semantic_sha256(metadata, qpos, qvel, act)
    return LoadedStateArtifact(loaded.raw_file_sha256, semantic, metadata, qpos, qvel, act)


def load_actions_artifact(
    path: str | Path,
    model_pair: ModelPairIdentity,
) -> LoadedActionsArtifact:
    """Load one strict actions NPZ and bind channels to model admission alignment."""
    alignment = _completed_alignment(model_pair)
    loaded = load_npz_arrays(
        path,
        expected_members=_ACTIONS_MEMBERS,
        invalid_reason=OperationalReasonCode.ACTIONS_ARTIFACT_INVALID,
    )
    return _loaded_actions_artifact(loaded, alignment)


def _load_actions_artifact_from_bytes(
    raw: bytes,
    model_pair: ModelPairIdentity,
) -> LoadedActionsArtifact:
    """Bind one already-frozen actions NPZ byte string to the admitted model pair."""
    alignment = _completed_alignment(model_pair)
    loaded = _load_npz_arrays_from_bytes(
        raw,
        expected_members=_ACTIONS_MEMBERS,
        invalid_reason=OperationalReasonCode.ACTIONS_ARTIFACT_INVALID,
    )
    return _loaded_actions_artifact(loaded, alignment)


def _loaded_actions_artifact(
    loaded: LoadedNpz,
    alignment: SemanticAlignment,
) -> LoadedActionsArtifact:
    """Validate and construct one actions artifact from preflighted NPZ arrays."""
    arrays = loaded.arrays
    schema, schema_version = _artifact_schema(
        arrays, OperationalReasonCode.ACTIONS_ARTIFACT_INVALID, _ACTIONS_SCHEMA
    )
    actuator_names = _action_names(arrays, alignment)
    values = _float_array(arrays["values"], "values", 2)
    rows, columns = _validate_action_shape(values, actuator_names)
    _require_finite(values, "values")
    metadata = ActionsArtifactMetadata(schema, schema_version, actuator_names, rows, columns)
    _make_read_only(values)
    semantic = _actions_semantic_sha256(metadata, values)
    return LoadedActionsArtifact(loaded.raw_file_sha256, semantic, metadata, values)


def _artifact_schema(
    arrays: Mapping[str, Any], reason: OperationalReasonCode, expected: str
) -> tuple[str, int]:
    """Validate the schema token and strict integer version shared by workload artifacts."""
    schema = _unicode_scalar(arrays["schema"], reason, "schema")
    if schema != expected:
        raise refuse(reason, field="schema", issue="unexpected_schema_token")
    version = _integer_scalar(arrays["schema_version"], reason, "schema_version")
    if version != _SCHEMA_VERSION:
        raise refuse(reason, field="schema_version", issue="unsupported_schema_version")
    return schema, version


def _state_names(
    arrays: Mapping[str, Any], alignment: SemanticAlignment
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate the canonical joint and actuator order in a state artifact."""
    reason = OperationalReasonCode.STATE_NAME_SET_MISMATCH
    joints = _unicode_names(arrays["joint_names"], reason, "joint_names")
    actuators = _unicode_names(arrays["actuator_names"], reason, "actuator_names")
    expected_joints = tuple(item.canonical_name for item in alignment.joints)
    expected_actuators = tuple(item.canonical_name for item in alignment.actuators)
    if joints != expected_joints or actuators != expected_actuators:
        raise refuse(
            reason,
            issue="canonical_name_order_mismatch",
            joint_count=len(joints),
            actuator_count=len(actuators),
        )
    return joints, actuators


def _state_arrays(
    arrays: Mapping[str, Any],
) -> tuple[tuple[Any, Any, Any], tuple[Any, Any, Any]]:
    """Parse state offsets and exact float64 value vectors."""
    offsets = (
        _offset_array(arrays["qpos_offsets"], "qpos_offsets"),
        _offset_array(arrays["qvel_offsets"], "qvel_offsets"),
        _offset_array(arrays["act_offsets"], "act_offsets"),
    )
    values = (
        _float_array(arrays["qpos"], "qpos", 1),
        _float_array(arrays["qvel"], "qvel", 1),
        _float_array(arrays["act"], "act", 1),
    )
    return offsets, values


def _action_names(arrays: Mapping[str, Any], alignment: SemanticAlignment) -> tuple[str, ...]:
    """Validate the canonical actuator order in an actions artifact."""
    reason = OperationalReasonCode.ACTION_NAME_SET_MISMATCH
    names = _unicode_names(arrays["actuator_names"], reason, "actuator_names")
    expected = tuple(item.canonical_name for item in alignment.actuators)
    if names != expected:
        raise refuse(reason, issue="canonical_name_order_mismatch", actuator_count=len(names))
    return names


def _validate_action_shape(
    values: npt.NDArray[np.float64], actuator_names: tuple[str, ...]
) -> tuple[int, int]:
    """Validate the bounded action matrix shape and return its dimensions."""
    rows, columns = values.shape
    if not 1 <= rows <= 100_000 or columns != len(actuator_names):
        raise refuse(
            OperationalReasonCode.ACTION_SHAPE_MISMATCH,
            rows=rows,
            columns=columns,
            expected_columns=len(actuator_names),
        )
    return rows, columns


def load_workload_artifacts(
    state_path: str | Path,
    actions_path: str | Path,
    model_pair: ModelPairIdentity,
) -> WorkloadArtifacts:
    """Load both artifacts and retain their exact completed model admission pair binding."""
    pair_sha = _completed_pair_sha256(model_pair)
    state = load_state_artifact(state_path, model_pair)
    actions = load_actions_artifact(actions_path, model_pair)
    return WorkloadArtifacts(pair_sha, state, actions)


def _load_workload_artifacts_from_bytes(
    state_raw: bytes,
    actions_raw: bytes,
    model_pair: ModelPairIdentity,
) -> WorkloadArtifacts:
    """Load one workload from campaign-frozen bytes without reopening either live path."""
    pair_sha = _completed_pair_sha256(model_pair)
    state = _load_state_artifact_from_bytes(state_raw, model_pair)
    actions = _load_actions_artifact_from_bytes(actions_raw, model_pair)
    return WorkloadArtifacts(pair_sha, state, actions)


def _completed_alignment(model_pair: ModelPairIdentity) -> SemanticAlignment:
    """Require a typed completed semantic alignment before workload admission."""
    if not isinstance(model_pair, ModelPairIdentity):
        raise TypeError("model_pair must be ModelPairIdentity")
    _completed_pair_sha256(model_pair)
    if model_pair.alignment.semantic_alignment_sha256 is None:
        raise refuse(
            OperationalReasonCode.MODEL_LAYOUT_INCOMPATIBLE,
            issue="semantic_alignment_not_completed",
        )
    return model_pair.alignment


def _completed_pair_sha256(model_pair: ModelPairIdentity) -> str:
    """Require the model pair's finalized SHA-256 before binding workload evidence."""
    if not isinstance(model_pair, ModelPairIdentity):
        raise TypeError("model_pair must be ModelPairIdentity")
    if model_pair.model_pair_identity_sha256 is None:
        raise refuse(
            OperationalReasonCode.MODEL_LAYOUT_INCOMPATIBLE,
            issue="model_pair_identity_not_completed",
        )
    return require_sha256(model_pair.model_pair_identity_sha256, "model_pair_identity_sha256")
