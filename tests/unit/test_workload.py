"""Collect workload scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from metrifid import _model_admission as admission
from metrifid import _model_identity as identity
from metrifid import _workload as workload
from metrifid._npz import ArtifactAdmissionRefusal
from metrifid.operational import OperationalReasonCode
from metrifid.schemas import (
    ActionsArtifactMetadata,
    ModelClosureIdentity,
    ModelClosureMember,
    TargetReference,
)


def _closure(label: str) -> ModelClosureIdentity:
    """Construct the closure fixture used by workload scenarios.

    Deterministic setup isolates workload without bypassing the contract boundary under
    assertion.
    """
    digest = hashlib.sha256(label.encode()).hexdigest()
    return ModelClosureIdentity(
        "model.xml", 1, (ModelClosureMember("model.xml", len(label), digest),)
    )


def _pair() -> identity.ModelPairIdentity:
    """Construct the pair fixture used by workload scenarios.

    Deterministic setup isolates workload without bypassing the contract boundary under
    assertion.
    """
    baseline_closure = _closure("baseline")
    candidate_closure = _closure("candidate")
    joints, actuators = _pair_descriptors()
    baseline = _compiled_pair_member(baseline_closure.sha256(), joints, actuators)
    candidate = _compiled_pair_member(candidate_closure.sha256(), joints, actuators)
    alignment = identity.align_compiled_models(baseline, candidate)
    return identity.ModelPairIdentity(
        identity.ModelPairIdentity._SCHEMA,
        identity.ModelPairIdentity._SCHEMA_VERSION,
        None,
        baseline_closure,
        candidate_closure,
        baseline,
        candidate,
        alignment,
        alignment.summary(),
    ).finalized()


def _pair_descriptors() -> tuple[
    tuple[admission.JointDescriptor, ...], tuple[admission.ActuatorDescriptor, ...]
]:
    """Build the canonical joint and actuator descriptors for workload fixtures."""
    joints = (
        admission.JointDescriptor("ball", "BALL", 4, 3, 2, 2),
        admission.JointDescriptor("free", "FREE", 7, 6, 6, 5),
        admission.JointDescriptor("hinge", "HINGE", 1, 1, 0, 0),
        admission.JointDescriptor("slide", "SLIDE", 1, 1, 1, 1),
    )
    actuators = (
        admission.ActuatorDescriptor(
            "integrator",
            "JOINT",
            (TargetReference("JOINT", "slide"),),
            "INTEGRATOR",
            1,
            1,
            0,
        ),
        admission.ActuatorDescriptor(
            "motor",
            "JOINT",
            (TargetReference("JOINT", "hinge"),),
            "NONE",
            0,
            0,
            None,
        ),
    )
    return joints, actuators


def _compiled_pair_member(
    closure_sha: str,
    joints: tuple[admission.JointDescriptor, ...],
    actuators: tuple[admission.ActuatorDescriptor, ...],
) -> admission.CompiledModelIdentity:
    """Finalize one role-local compiled identity for the workload model pair."""
    return admission.CompiledModelIdentity(
        admission.CompiledModelIdentity._SCHEMA,
        admission.CompiledModelIdentity._SCHEMA_VERSION,
        None,
        closure_sha,
        13,
        11,
        2,
        1,
        joints,
        actuators,
    ).finalized()


def _state_arrays(
    pair: identity.ModelPairIdentity,
) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
    """Construct the state arrays fixture used by workload scenarios.

    Deterministic setup isolates workload without bypassing the contract boundary under
    assertion.
    """
    joints = pair.alignment.joints
    actuators = pair.alignment.actuators
    qpos_widths = [item.baseline_qpos[1] for item in joints]
    qvel_widths = [item.baseline_qvel[1] for item in joints]
    act_widths = [item.activation_width for item in actuators]

    def offsets(widths: list[int]) -> np.ndarray[tuple[int, ...], np.dtype[np.int64]]:
        """Construct the offsets fixture used by workload scenarios.

        Deterministic setup isolates state arrays without bypassing the contract boundary under
        assertion.
        """
        values = [0]
        for width in widths:
            values.append(values[-1] + width)
        return np.array(values, dtype="<i8")

    return {
        "schema": np.array("metrifid.state", dtype="<U32"),
        "schema_version": np.array(1, dtype="<i8"),
        "joint_names": np.array([item.canonical_name for item in joints], dtype="<U16"),
        "qpos_offsets": offsets(qpos_widths),
        "qpos": np.arange(sum(qpos_widths), dtype="<f8") / 10.0,
        "qvel_offsets": offsets(qvel_widths),
        "qvel": np.arange(sum(qvel_widths), dtype="<f8") / 20.0,
        "actuator_names": np.array([item.canonical_name for item in actuators], dtype="<U16"),
        "act_offsets": offsets(act_widths),
        "act": np.array([0.25], dtype="<f8"),
    }


def _actions_arrays(
    pair: identity.ModelPairIdentity, rows: int = 3
) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
    """Construct the actions arrays fixture used by workload scenarios.

    Deterministic setup isolates workload without bypassing the contract boundary under
    assertion.
    """
    names = [item.canonical_name for item in pair.alignment.actuators]
    return {
        "schema": np.array("metrifid.actions", dtype="<U32"),
        "schema_version": np.array(1, dtype="<i8"),
        "actuator_names": np.array(names, dtype="<U16"),
        "values": np.arange(rows * len(names), dtype="<f8").reshape(rows, len(names)),
    }


def _write_npz(
    path: Path,
    arrays: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
    *,
    compressed: bool = False,
) -> None:
    """Write write NPZ data into the isolated test workspace.

    The workload scenario observes real bytes and filesystem effects for workload.
    """
    writer = np.savez_compressed if compressed else np.savez
    writer(path, **arrays)


def _refusal_reason(exc: pytest.ExceptionInfo[ArtifactAdmissionRefusal]) -> OperationalReasonCode:
    """Extract the admission code from a malformed workload refusal."""
    return exc.value.reason


def test_valid_state_and_actions_cover_all_supported_joint_and_activation_widths(
    tmp_path: Path,
) -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises valid state and actions cover all supported joint and activation
    widths; malformed arrays, names, or dimensions must fail before comparison evidence is
    produced.
    """
    pair = _pair()
    state_path = tmp_path / "state.npz"
    actions_path = tmp_path / "actions.npz"
    _write_npz(state_path, _state_arrays(pair))
    _write_npz(actions_path, _actions_arrays(pair))

    state = workload.load_state_artifact(state_path, pair)
    actions = workload.load_actions_artifact(actions_path, pair)
    combined = workload.load_workload_artifacts(state_path, actions_path, pair)

    assert state.metadata.joint_names == ("ball", "free", "hinge", "slide")
    assert state.metadata.qpos_offsets == (0, 4, 11, 12, 13)
    assert state.metadata.qvel_offsets == (0, 3, 9, 10, 11)
    assert state.metadata.actuator_names == ("integrator", "motor")
    assert state.metadata.act_offsets == (0, 1, 1)
    assert actions.metadata.control_intervals == 3
    assert actions.values.shape == (3, 2)
    assert combined.model_pair_identity_sha256 == pair.model_pair_identity_sha256
    assert combined.state.semantic_sha256 == state.semantic_sha256
    assert combined.actions.semantic_sha256 == actions.semantic_sha256
    for array in (state.qpos, state.qvel, state.act, actions.values):
        assert array.flags.c_contiguous
        assert not array.flags.writeable


def test_frozen_workload_bytes_share_path_admission_without_reopening_paths(
    tmp_path: Path,
) -> None:
    """Frozen campaign bytes retain the path loader's exact raw and semantic identities."""
    pair = _pair()
    state_path = tmp_path / "state.npz"
    actions_path = tmp_path / "actions.npz"
    _write_npz(state_path, _state_arrays(pair))
    _write_npz(actions_path, _actions_arrays(pair))
    expected = workload.load_workload_artifacts(state_path, actions_path, pair)
    state_raw = state_path.read_bytes()
    actions_raw = actions_path.read_bytes()

    state_path.write_bytes(b"replaced state")
    actions_path.write_bytes(b"replaced actions")
    frozen = workload._load_workload_artifacts_from_bytes(state_raw, actions_raw, pair)

    assert frozen.model_pair_identity_sha256 == expected.model_pair_identity_sha256
    assert frozen.state.raw_file_sha256 == expected.state.raw_file_sha256
    assert frozen.state.semantic_sha256 == expected.state.semantic_sha256
    assert frozen.actions.raw_file_sha256 == expected.actions.raw_file_sha256
    assert frozen.actions.semantic_sha256 == expected.actions.semantic_sha256
    np.testing.assert_array_equal(frozen.state.qpos, expected.state.qpos)
    np.testing.assert_array_equal(frozen.actions.values, expected.actions.values)


def test_raw_hash_changes_and_semantic_hash_survives_npz_repack(tmp_path: Path) -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises raw hash changes and semantic hash survives NPZ repack; malformed
    arrays, names, or dimensions must fail before comparison evidence is produced.
    """
    pair = _pair()
    arrays = _state_arrays(pair)
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_npz(first, arrays, compressed=False)
    _write_npz(second, arrays, compressed=True)
    left = workload.load_state_artifact(first, pair)
    right = workload.load_state_artifact(second, pair)
    assert left.raw_file_sha256 != right.raw_file_sha256
    assert left.semantic_sha256 == right.semantic_sha256
    assert left.raw_file_sha256 == hashlib.sha256(first.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("field", "replacement", "reason"),
    [
        (
            "schema_version",
            np.array(2, dtype="<i8"),
            OperationalReasonCode.STATE_ARTIFACT_INVALID,
        ),
        (
            "schema_version",
            np.array(True, dtype="bool"),
            OperationalReasonCode.NPZ_DTYPE_MISMATCH,
        ),
        (
            "schema_version",
            np.array(1, dtype=">i8"),
            OperationalReasonCode.NPZ_DTYPE_MISMATCH,
        ),
        (
            "schema_version",
            np.array([1], dtype="<i8"),
            OperationalReasonCode.STATE_ARTIFACT_INVALID,
        ),
        (
            "schema",
            np.array("metrifid.unknown", dtype="<U32"),
            OperationalReasonCode.STATE_ARTIFACT_INVALID,
        ),
        ("qpos", np.arange(13, dtype=">f8"), OperationalReasonCode.NPZ_DTYPE_MISMATCH),
        (
            "qpos",
            np.arange(13, dtype="<f8").reshape(1, 13),
            OperationalReasonCode.STATE_ARTIFACT_INVALID,
        ),
        (
            "qpos_offsets",
            np.array([0, 4, 11, 12, 13], dtype="<i4"),
            OperationalReasonCode.NPZ_DTYPE_MISMATCH,
        ),
        (
            "qpos_offsets",
            np.array([[0, 4, 11, 12, 13]], dtype="<i8"),
            OperationalReasonCode.STATE_OFFSET_INVALID,
        ),
        (
            "qpos_offsets",
            np.array([1, 4, 11, 12, 13], dtype="<i8"),
            OperationalReasonCode.STATE_OFFSET_INVALID,
        ),
        (
            "qpos_offsets",
            np.array([0, 5, 11, 12, 13], dtype="<i8"),
            OperationalReasonCode.STATE_WIDTH_MISMATCH,
        ),
        (
            "act_offsets",
            np.array([0, 0, 1], dtype="<i8"),
            OperationalReasonCode.STATE_WIDTH_MISMATCH,
        ),
    ],
)
def test_state_structural_refusals(
    tmp_path: Path,
    field: str,
    replacement: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    reason: OperationalReasonCode,
) -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises state structural refusals; malformed arrays, names, or dimensions
    must fail before comparison evidence is produced.
    """
    pair = _pair()
    arrays = _state_arrays(pair)
    arrays[field] = replacement
    path = tmp_path / f"{field}.npz"
    _write_npz(path, arrays)
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        workload.load_state_artifact(path, pair)
    assert _refusal_reason(exc) is reason


@pytest.mark.parametrize(
    "names",
    [
        ["free", "ball", "hinge", "slide"],
        ["ball", "free", "hinge", "hinge"],
        ["ball", "free", "", "slide"],
        ["ball", "free", "bad\x00name", "slide"],
        ["ball", "free", "\ud800", "slide"],
    ],
)
def test_state_joint_name_refusals(tmp_path: Path, names: list[str]) -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises state joint name refusals; malformed arrays, names, or dimensions
    must fail before comparison evidence is produced.
    """
    pair = _pair()
    arrays = _state_arrays(pair)
    arrays["joint_names"] = np.array(names, dtype="<U16")
    path = tmp_path / "bad-names.npz"
    _write_npz(path, arrays)
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        workload.load_state_artifact(path, pair)
    assert _refusal_reason(exc) is OperationalReasonCode.STATE_NAME_SET_MISMATCH


@pytest.mark.parametrize("field", ["qpos", "qvel", "act"])
def test_state_nonfinite_refuses(tmp_path: Path, field: str) -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises state nonfinite refuses; malformed arrays, names, or dimensions must
    fail before comparison evidence is produced.
    """
    pair = _pair()
    arrays = _state_arrays(pair)
    value = cast(np.ndarray[tuple[int, ...], np.dtype[np.float64]], arrays[field]).copy()
    value[0] = np.nan
    arrays[field] = value
    path = tmp_path / f"nonfinite-{field}.npz"
    _write_npz(path, arrays)
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        workload.load_state_artifact(path, pair)
    assert _refusal_reason(exc) is OperationalReasonCode.INPUT_NONFINITE_VALUE


@pytest.mark.parametrize(
    ("mutation", "field", "replacement", "reason"),
    [
        (
            "schema",
            "schema",
            np.array("metrifid.unknown", dtype="<U32"),
            OperationalReasonCode.ACTIONS_ARTIFACT_INVALID,
        ),
        (
            "name_order",
            "actuator_names",
            np.array(["motor", "integrator"], dtype="<U16"),
            OperationalReasonCode.ACTION_NAME_SET_MISMATCH,
        ),
        (
            "name_dtype",
            "actuator_names",
            np.array([b"integrator", b"motor"], dtype="S16"),
            OperationalReasonCode.NPZ_DTYPE_MISMATCH,
        ),
        (
            "name_endian",
            "actuator_names",
            np.array(["integrator", "motor"], dtype=">U16"),
            OperationalReasonCode.NPZ_DTYPE_MISMATCH,
        ),
        (
            "name_set",
            "actuator_names",
            np.array(["integrator", "extra"], dtype="<U16"),
            OperationalReasonCode.ACTION_NAME_SET_MISMATCH,
        ),
        (
            "name_rank",
            "actuator_names",
            np.array([["integrator", "motor"]], dtype="<U16"),
            OperationalReasonCode.ACTION_NAME_SET_MISMATCH,
        ),
        (
            "value_dtype",
            "values",
            np.ones((3, 2), dtype=">f8"),
            OperationalReasonCode.NPZ_DTYPE_MISMATCH,
        ),
        ("rank", "values", np.ones(6, dtype="<f8"), OperationalReasonCode.ACTION_SHAPE_MISMATCH),
        (
            "zero_rows",
            "values",
            np.empty((0, 2), dtype="<f8"),
            OperationalReasonCode.ACTION_SHAPE_MISMATCH,
        ),
        (
            "too_many_rows",
            "values",
            np.zeros((100_001, 2), dtype="<f8"),
            OperationalReasonCode.ACTION_SHAPE_MISMATCH,
        ),
        (
            "missing_channel",
            "values",
            np.zeros((3, 1), dtype="<f8"),
            OperationalReasonCode.ACTION_SHAPE_MISMATCH,
        ),
        (
            "fortran",
            "values",
            np.asfortranarray(np.ones((3, 2), dtype="<f8")),
            OperationalReasonCode.ACTION_SHAPE_MISMATCH,
        ),
        (
            "nonfinite",
            "values",
            np.array([[0.0, np.inf]], dtype="<f8"),
            OperationalReasonCode.INPUT_NONFINITE_VALUE,
        ),
    ],
)
def test_actions_refusals(
    tmp_path: Path,
    mutation: str,
    field: str,
    replacement: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    reason: OperationalReasonCode,
) -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises actions refusals; malformed arrays, names, or dimensions must fail
    before comparison evidence is produced.
    """
    pair = _pair()
    arrays = _actions_arrays(pair)
    arrays[field] = replacement
    path = tmp_path / f"actions-{mutation}.npz"
    _write_npz(path, arrays)
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        workload.load_actions_artifact(path, pair)
    assert _refusal_reason(exc) is reason


def test_incomplete_or_wrong_model_pair_refuses_deliberately(tmp_path: Path) -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises incomplete or wrong model pair refuses deliberately; malformed
    arrays, names, or dimensions must fail before comparison evidence is produced.
    """
    pair = _pair()
    path = tmp_path / "state.npz"
    _write_npz(path, _state_arrays(pair))
    unfinished = identity.ModelPairIdentity(
        identity.ModelPairIdentity._SCHEMA,
        identity.ModelPairIdentity._SCHEMA_VERSION,
        None,
        pair.baseline_closure,
        pair.candidate_closure,
        pair.baseline_compiled,
        pair.candidate_compiled,
        pair.alignment,
        pair.alignment_summary,
    )
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        workload.load_state_artifact(path, unfinished)
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_LAYOUT_INCOMPATIBLE
    with pytest.raises(TypeError):
        workload.load_state_artifact(path, cast(identity.ModelPairIdentity, object()))


def test_loaded_runtime_object_invariants_cannot_be_forged(tmp_path: Path) -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises loaded runtime object invariants cannot be forged; malformed arrays,
    names, or dimensions must fail before comparison evidence is produced.
    """
    pair = _pair()
    state_path = tmp_path / "state.npz"
    actions_path = tmp_path / "actions.npz"
    _write_npz(state_path, _state_arrays(pair))
    _write_npz(actions_path, _actions_arrays(pair))
    state = workload.load_state_artifact(state_path, pair)
    actions = workload.load_actions_artifact(actions_path, pair)
    combined = workload.WorkloadArtifacts(pair.model_pair_identity_sha256 or "", state, actions)

    with pytest.raises(TypeError):
        replace(state, metadata=cast(object, object()))
    larger = np.zeros(state.qpos.size + 1, dtype="<f8")
    larger.setflags(write=False)
    with pytest.raises(ValueError, match="metadata counts"):
        replace(state, qpos=larger)
    with pytest.raises(ValueError, match="semantic_sha256"):
        replace(state, semantic_sha256="0" * 64)

    with pytest.raises(TypeError):
        replace(actions, metadata=cast(object, object()))
    wrong_shape = np.zeros((actions.values.shape[0] + 1, actions.values.shape[1]), dtype="<f8")
    wrong_shape.setflags(write=False)
    with pytest.raises(ValueError, match="metadata shape"):
        replace(actions, values=wrong_shape)
    with pytest.raises(ValueError, match="semantic_sha256"):
        replace(actions, semantic_sha256="0" * 64)

    with pytest.raises(TypeError):
        replace(combined, state=cast(object, object()))
    with pytest.raises(TypeError):
        replace(combined, actions=cast(object, object()))

    mismatched_metadata = ActionsArtifactMetadata(
        "metrifid.actions",
        1,
        ("different", "motor"),
        actions.metadata.control_intervals,
        actions.metadata.actuator_count,
    )
    mismatched_values = actions.values.copy()
    mismatched_values.setflags(write=False)
    mismatched_actions = workload.LoadedActionsArtifact(
        actions.raw_file_sha256,
        workload._actions_semantic_sha256(mismatched_metadata, mismatched_values),
        mismatched_metadata,
        mismatched_values,
    )
    with pytest.raises(ValueError, match="canonical actuator order"):
        workload.WorkloadArtifacts(pair.model_pair_identity_sha256 or "", state, mismatched_actions)


def test_runtime_array_guard_rejects_every_invalid_surface() -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises runtime array guard rejects every invalid surface; malformed arrays,
    names, or dimensions must fail before comparison evidence is produced.
    """
    with pytest.raises(TypeError):
        workload._require_runtime_float_array(object(), "value", 1)

    wrong_dtype = np.array([1.0], dtype="<f4")
    wrong_dtype.setflags(write=False)
    with pytest.raises(ValueError, match="dtype or rank"):
        workload._require_runtime_float_array(wrong_dtype, "value", 1)

    writable = np.array([1.0], dtype="<f8")
    with pytest.raises(ValueError, match="read-only"):
        workload._require_runtime_float_array(writable, "value", 1)

    nonfinite = np.array([np.inf], dtype="<f8")
    nonfinite.setflags(write=False)
    with pytest.raises(ValueError, match="finite"):
        workload._require_runtime_float_array(nonfinite, "value", 1)


def test_unfinished_alignment_and_private_pair_guard_refuse(tmp_path: Path) -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises unfinished alignment and private pair guard refuse; malformed
    arrays, names, or dimensions must fail before comparison evidence is produced.
    """
    pair = _pair()
    state_path = tmp_path / "state.npz"
    _write_npz(state_path, _state_arrays(pair))
    object.__setattr__(pair.alignment, "semantic_alignment_sha256", None)
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        workload.load_state_artifact(state_path, pair)
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_LAYOUT_INCOMPATIBLE
    with pytest.raises(TypeError):
        workload._completed_pair_sha256(cast(identity.ModelPairIdentity, object()))


def test_state_name_rank_refuses_with_name_reason(tmp_path: Path) -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises state name rank refuses with name reason; malformed arrays, names,
    or dimensions must fail before comparison evidence is produced.
    """
    pair = _pair()
    arrays = _state_arrays(pair)
    arrays["joint_names"] = np.array([["ball", "free", "hinge", "slide"]], dtype="<U16")
    path = tmp_path / "name-rank.npz"
    _write_npz(path, arrays)
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        workload.load_state_artifact(path, pair)
    assert _refusal_reason(exc) is OperationalReasonCode.STATE_NAME_SET_MISMATCH
