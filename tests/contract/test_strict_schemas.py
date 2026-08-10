"""Strict immutable schema conversion and refusal tests."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from metrifid import ComparisonConfig, ExactRational
from metrifid.schemas import (
    ActionsArtifactMetadata,
    ActuatorAliasEndpoint,
    AliasArtifact,
    JointAliasPair,
    JointToleranceConfig,
    ModelClosureIdentity,
    StateArtifactMetadata,
    TargetReference,
)


def digest(label: str) -> str:
    """Compute the canonical digest value used by strict schemas fixtures.

    Content addressing keeps the mutation boundary explicit for strict schemas.
    """
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def valid_config() -> dict[str, object]:
    """Construct the valid config fixture used by strict schemas scenarios.

    Deterministic setup isolates strict schemas without bypassing the contract boundary under
    assertion.
    """
    return {
        "schema_version": 1,
        "baseline": {
            "model_root": "models/before",
            "entrypoint": "robot.xml",
            "declared_step_dt": "0.002",
        },
        "candidate": {
            "model_root": "models/after",
            "entrypoint": "robot.xml",
            "declared_step_dt": "0.002",
        },
        "initial_state": "workloads/state.npz",
        "actions": "workloads/actions.npz",
        "control_dt": "0.01",
        "repeats": 3,
        "joint_tolerances": {
            "elbow": {
                "joint_type": "hinge",
                "angle_rad": "0.001",
                "angular_velocity_rad_s": "0.01",
            }
        },
        "aliases": None,
        "output_dir": "results/run",
    }


_DELETE = object()
_ELBOW_TOLERANCE = {
    "joint_type": "hinge",
    "angle_rad": "0.001",
    "angular_velocity_rad_s": "0.01",
}


def test_comparison_config_round_trip_and_immutability() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises comparison config round trip and immutability; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    parsed = ComparisonConfig.from_primitive(valid_config())
    assert parsed.to_primitive() == valid_config()
    assert parsed.control_dt == ExactRational(1, 100)
    with pytest.raises(FrozenInstanceError):
        parsed.repeats = 4  # type: ignore[misc]
    with pytest.raises(TypeError):
        parsed.joint_tolerances["new"] = parsed.joint_tolerances["elbow"]  # type: ignore[index]


@pytest.mark.parametrize(
    ("joint_type", "fields"),
    [
        ("hinge", {"angle_rad": "0.1", "angular_velocity_rad_s": "0.2"}),
        ("slide", {"translation_m": "0.1", "linear_velocity_m_s": "0.2"}),
        ("ball", {"orientation_rad": "0.1", "angular_velocity_rad_s": "0.2"}),
        (
            "free",
            {
                "translation_m": "0.1",
                "orientation_rad": "0.2",
                "linear_velocity_m_s": "0.3",
                "angular_velocity_rad_s": "0.4",
            },
        ),
    ],
)
def test_all_joint_tolerance_shapes(joint_type: str, fields: dict[str, str]) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises all joint tolerance shapes; accepting a contradictory or
    noncanonical value would make the signed decision evidence ambiguous.
    """
    primitive: dict[str, object] = {"joint_type": joint_type, **fields}
    parsed = JointToleranceConfig.from_primitive(primitive)
    assert parsed.to_primitive() == primitive
    semantic = parsed.semantic_primitive()
    assert semantic["joint_type"] == joint_type
    assert all(isinstance(semantic[key], dict) for key in fields)


@pytest.mark.parametrize(
    ("mutation", "path", "replacement"),
    [
        ("missing-field", ("actions",), _DELETE),
        ("unknown-field", ("extra",), 1),
        ("wrong-schema", ("schema_version",), 2),
        ("bool-schema", ("schema_version",), True),
        ("bad-repeat", ("repeats",), 1),
        ("bool-repeat", ("repeats",), True),
        ("empty-tolerances", ("joint_tolerances",), {}),
        ("bad-name", ("joint_tolerances",), {"bad\x00name": _ELBOW_TOLERANCE}),
        ("bad-decimal", ("control_dt",), "1e-2"),
        ("zero-tolerance", ("joint_tolerances", "elbow", "angle_rad"), "0"),
        ("wrong-joint-key", ("joint_tolerances", "elbow", "angle_rad"), _DELETE),
        ("aliases-type", ("aliases",), 1),
        ("output-empty", ("output_dir",), ""),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_config_refusal_matrix(mutation: str, path: tuple[str, ...], replacement: object) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises config refusal matrix; accepting a contradictory or noncanonical
    value would make the signed decision evidence ambiguous.
    """
    value = valid_config()
    _apply_config_mutation(value, path, replacement)
    with pytest.raises((TypeError, ValueError, UnicodeError)):
        ComparisonConfig.from_primitive(value)


def _apply_config_mutation(
    value: dict[str, object], path: tuple[str, ...], replacement: object
) -> None:
    """Apply one declarative nested config mutation used by the refusal matrix."""
    target = value
    for component in path[:-1]:
        nested = target[component]
        if not isinstance(nested, dict):
            raise AssertionError("mutation path must traverse objects")
        target = cast("dict[str, object]", nested)
    if replacement is _DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = replacement


def test_config_nested_shape_refusals() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises config nested shape refusals; accepting a contradictory or
    noncanonical value would make the signed decision evidence ambiguous.
    """
    with pytest.raises(TypeError):
        ComparisonConfig.from_primitive([])
    value = valid_config()
    value["baseline"] = []
    with pytest.raises(TypeError):
        ComparisonConfig.from_primitive(value)
    value = valid_config()
    value["joint_tolerances"] = []
    with pytest.raises(TypeError):
        ComparisonConfig.from_primitive(value)


def test_state_and_actions_metadata_round_trip() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises state and actions metadata round trip; accepting a contradictory or
    noncanonical value would make the signed decision evidence ambiguous.
    """
    state_primitive = {
        "schema": "metrifid.state",
        "schema_version": 1,
        "joint_names": ["elbow", "shoulder"],
        "qpos_offsets": [0, 1, 2],
        "qpos_count": 2,
        "qvel_offsets": [0, 1, 2],
        "qvel_count": 2,
        "actuator_names": ["elbow_motor"],
        "act_offsets": [0, 1],
        "act_count": 1,
    }
    state = StateArtifactMetadata.from_primitive(state_primitive)
    assert state.to_primitive() == state_primitive
    actions_primitive = {
        "schema": "metrifid.actions",
        "schema_version": 1,
        "actuator_names": ["elbow_motor"],
        "control_intervals": 10,
        "actuator_count": 1,
    }
    actions = ActionsArtifactMetadata.from_primitive(actions_primitive)
    assert actions.to_primitive() == actions_primitive


@pytest.mark.parametrize(
    "state",
    [
        {
            "schema_version": "wrong",
            "joint_names": [],
            "qpos_offsets": [0],
            "qpos_count": 0,
            "qvel_offsets": [0],
            "qvel_count": 0,
            "actuator_names": [],
            "act_offsets": [0],
            "act_count": 0,
        },
        {
            "schema": "metrifid.state",
            "schema_version": 1,
            "joint_names": ["x", "x"],
            "qpos_offsets": [0, 1, 2],
            "qpos_count": 2,
            "qvel_offsets": [0, 1, 2],
            "qvel_count": 2,
            "actuator_names": [],
            "act_offsets": [0],
            "act_count": 0,
        },
        {
            "schema": "metrifid.state",
            "schema_version": 1,
            "joint_names": ["x"],
            "qpos_offsets": [1, 2],
            "qpos_count": 2,
            "qvel_offsets": [0, 1],
            "qvel_count": 1,
            "actuator_names": [],
            "act_offsets": [0],
            "act_count": 0,
        },
        {
            "schema": "metrifid.state",
            "schema_version": 1,
            "joint_names": ["x"],
            "qpos_offsets": [0, 2],
            "qpos_count": 1,
            "qvel_offsets": [0, 1],
            "qvel_count": 1,
            "actuator_names": [],
            "act_offsets": [0],
            "act_count": 0,
        },
    ],
)
def test_invalid_state_metadata_refuses(state: dict[str, object]) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises invalid state metadata refuses; accepting a contradictory or
    noncanonical value would make the signed decision evidence ambiguous.
    """
    with pytest.raises((TypeError, ValueError)):
        StateArtifactMetadata.from_primitive(state)


def test_invalid_actions_metadata_refuses() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises invalid actions metadata refuses; accepting a contradictory or
    noncanonical value would make the signed decision evidence ambiguous.
    """
    with pytest.raises(ValueError):
        ActionsArtifactMetadata("metrifid.actions", 1, ("a",), 1, 2)
    with pytest.raises(ValueError):
        ActionsArtifactMetadata("wrong", 1, (), 1, 0)
    with pytest.raises(ValueError):
        ActionsArtifactMetadata("metrifid.actions", 1, (), 0, 0)


def test_alias_artifact_round_trip() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises alias artifact round trip; accepting a contradictory or noncanonical
    value would make the signed decision evidence ambiguous.
    """
    primitive = {
        "schema": "metrifid.aliases",
        "schema_version": 1,
        "baseline_model_closure_sha256": digest("baseline"),
        "candidate_model_closure_sha256": digest("candidate"),
        "joint_pairs": [
            {
                "canonical_name": "elbow",
                "baseline_name": "elbow_old",
                "candidate_name": "elbow",
            }
        ],
        "actuator_pairs": [
            {
                "canonical_name": "elbow_motor",
                "baseline": {"kind": "NAMED", "name": "old_motor"},
                "candidate": {
                    "kind": "UNNAMED_SELECTOR",
                    "transmission_type": "JOINT",
                    "targets": [{"object_type": "JOINT", "name": "elbow"}],
                    "activation_family": "NONE",
                    "activation_width": 0,
                },
            }
        ],
    }
    parsed = AliasArtifact.from_primitive(primitive)
    assert parsed.to_primitive() == primitive
    assert parsed.joint_pairs[0] == JointAliasPair("elbow", "elbow_old", "elbow")


@pytest.mark.parametrize(
    "endpoint",
    [
        {"kind": "NAMED"},
        {"kind": "NAMED", "name": "x", "extra": 1},
        {
            "kind": "UNNAMED_SELECTOR",
            "transmission_type": "JOINT",
            "targets": [],
            "activation_family": "NONE",
            "activation_width": 0,
        },
        {"kind": "UNKNOWN"},
    ],
)
def test_invalid_alias_endpoint_refuses(endpoint: dict[str, object]) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises invalid alias endpoint refuses; accepting a contradictory or
    noncanonical value would make the signed decision evidence ambiguous.
    """
    with pytest.raises((TypeError, ValueError)):
        ActuatorAliasEndpoint.from_primitive(endpoint)


def test_alias_direct_constructor_invariants() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises alias direct constructor invariants; accepting a contradictory or
    noncanonical value would make the signed decision evidence ambiguous.
    """
    target = TargetReference("JOINT", "elbow")
    with pytest.raises(ValueError):
        ActuatorAliasEndpoint("NAMED", None, None, (), None, None)
    with pytest.raises(ValueError):
        ActuatorAliasEndpoint("NAMED", "x", "JOINT", (), None, None)
    with pytest.raises(ValueError):
        ActuatorAliasEndpoint("UNNAMED_SELECTOR", "x", "JOINT", (target,), "NONE", 0)
    with pytest.raises(ValueError):
        ActuatorAliasEndpoint("UNNAMED_SELECTOR", None, "JOINT", (), "NONE", 0)
    with pytest.raises(ValueError):
        ActuatorAliasEndpoint("BAD", None, None, (), None, None)  # type: ignore[arg-type]


def test_alias_pairs_must_be_sorted_and_unique() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises alias pairs must be sorted and unique; accepting a contradictory or
    noncanonical value would make the signed decision evidence ambiguous.
    """
    pair_a = JointAliasPair("a", "a", "a")
    pair_b = JointAliasPair("b", "b", "b")
    with pytest.raises(ValueError):
        AliasArtifact("metrifid.aliases", 1, digest("b"), digest("c"), (pair_b, pair_a), ())
    with pytest.raises(ValueError):
        AliasArtifact("metrifid.aliases", 1, digest("b"), digest("c"), (pair_a, pair_a), ())


def test_model_closure_round_trip_and_hash() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises model closure round trip and hash; accepting a contradictory or
    noncanonical value would make the signed decision evidence ambiguous.
    """
    primitive = {
        "entrypoint": "robot.xml",
        "member_count": 2,
        "members": [
            {"path": "assets/mesh.bin", "size_bytes": 2, "sha256": digest("mesh")},
            {"path": "robot.xml", "size_bytes": 3, "sha256": digest("xml")},
        ],
    }
    closure = ModelClosureIdentity.from_primitive(primitive)
    assert closure.to_primitive() == primitive
    assert len(closure.sha256()) == 64
