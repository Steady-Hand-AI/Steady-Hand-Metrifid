"""Strict immutable schema conversion and refusal tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, cast

import pytest

from metrifid import ComparisonConfig, ComparisonContractIdentity, ExactRational
from metrifid.schemas import (
    AliasArtifact,
    AlignmentSummary,
    CanonicalSummary,
    ComparisonInputsIdentity,
    EnvironmentIdentity,
    JointToleranceConfig,
    ModelClosureIdentity,
    ModelClosureMember,
    ModelRoleConfig,
    MonitoredJoint,
    StateArtifactMetadata,
    TimeContract,
    ToolIdentity,
)


def digest(label: str) -> str:
    """Compute the canonical digest value used by strict receipt schemas fixtures.

    Content addressing keeps the mutation boundary explicit for strict receipt schemas.
    """
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def valid_config() -> dict[str, object]:
    """Construct the valid config fixture used by strict receipt schemas scenarios.

    Deterministic setup isolates strict receipt schemas without bypassing the contract boundary
    under assertion.
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


@pytest.mark.parametrize(
    "primitive",
    [
        {"entrypoint": "../robot.xml", "member_count": 0, "members": []},
        {"entrypoint": "robot.xml", "member_count": 2, "members": []},
        {
            "entrypoint": "robot.xml",
            "member_count": 1,
            "members": [{"path": "other.xml", "size_bytes": 1, "sha256": digest("x")}],
        },
        {
            "entrypoint": "robot.xml",
            "member_count": 2,
            "members": [
                {"path": "robot.xml", "size_bytes": 1, "sha256": digest("x")},
                {"path": "a.xml", "size_bytes": 1, "sha256": digest("y")},
            ],
        },
    ],
)
def test_invalid_model_closure_refuses(primitive: dict[str, object]) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises invalid model closure refuses; accepting a contradictory or
    noncanonical value would make the signed decision evidence ambiguous.
    """
    with pytest.raises((TypeError, ValueError)):
        ModelClosureIdentity.from_primitive(primitive)


def test_environment_hash_and_refusals() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises environment hash and refusals; accepting a contradictory or
    noncanonical value would make the signed decision evidence ambiguous.
    """
    primitive = {
        "mujoco_version": "3.10.0",
        "python_version": "3.13",
        "numpy_version": "2.3",
        "mujoco_python_distribution_sha256": digest("mp"),
        "mujoco_native_library_sha256": digest("mn"),
        "platform": "linux-x86_64",
        "platform_release": "kernel",
        "libc": "glibc",
        "cpu_identity_sha256": digest("cpu"),
        "engine_threadpool_state": "DISABLED",
        "environment_sha256": None,
    }
    env = EnvironmentIdentity.from_primitive(primitive)
    finalized = env.finalized()
    finalized.validate_hash()
    assert finalized.finalized() is finalized
    with pytest.raises(ValueError):
        EnvironmentIdentity.from_primitive({**primitive, "engine_threadpool_state": "BOGUS"})
    with pytest.raises(ValueError):
        EnvironmentIdentity.from_primitive(
            {**primitive, "environment_sha256": digest("bad")}
        ).finalized()


def test_time_contract_round_trip_and_refusals() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises time contract round trip and refusals; accepting a contradictory or
    noncanonical value would make the signed decision evidence ambiguous.
    """
    primitive = {
        "baseline_step_dt": {"numerator": 1, "denominator": 500},
        "candidate_step_dt": {"numerator": 1, "denominator": 1000},
        "control_dt": {"numerator": 1, "denominator": 100},
        "control_intervals": 2,
        "state_samples": 3,
        "horizon": {"numerator": 1, "denominator": 50},
        "sample_phase": "BOUNDARY_BEFORE_CONTROL",
        "action_semantics": "LEFT_BOUNDARY_ZERO_ORDER_HOLD",
        "terminal_sample": "INCLUDED",
        "interpolation": "FORBIDDEN",
    }
    assert TimeContract.from_primitive(primitive).to_primitive() == primitive
    for field, replacement in [
        ("state_samples", 2),
        ("horizon", {"numerator": 1, "denominator": 20}),
        ("baseline_step_dt", {"numerator": 3, "denominator": 1000}),
        ("sample_phase", "AFTER"),
        ("interpolation", "ALLOWED"),
    ]:
        with pytest.raises(ValueError):
            TimeContract.from_primitive({**primitive, field: replacement})


def test_alignment_hash_and_ordering() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises alignment hash and ordering; accepting a contradictory or
    noncanonical value would make the signed decision evidence ambiguous.
    """
    alignment = AlignmentSummary.from_primitive(
        {
            "alignment_sha256": None,
            "joint_order": ["a", "b"],
            "actuator_order": ["m"],
            "alias_bindings": [{"canonical_name": "a"}, {"canonical_name": "b"}],
        }
    )
    finalized = alignment.finalized()
    finalized.validate_hash()
    assert finalized.finalized() is finalized
    with pytest.raises(ValueError):
        AlignmentSummary(None, ("b", "a"), (), ())
    with pytest.raises(ValueError):
        AlignmentSummary.from_primitive(
            {
                "alignment_sha256": None,
                "joint_order": [],
                "actuator_order": [],
                "alias_bindings": [{"z": 1}, {"a": 1}],
            }
        )


def test_comparison_contract_round_trip_and_ordering() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises comparison contract round trip and ordering; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    primitive = {
        "schema": "metrifid.comparison_contract",
        "schema_version": 1,
        "baseline_model_closure_sha256": digest("b"),
        "candidate_model_closure_sha256": digest("c"),
        "initial_state_semantic_sha256": digest("s"),
        "actions_semantic_sha256": digest("a"),
        "aliases_semantic_sha256": None,
        "baseline_step_dt": {"numerator": 1, "denominator": 500},
        "candidate_step_dt": {"numerator": 1, "denominator": 500},
        "control_dt": {"numerator": 1, "denominator": 100},
        "repeats": 3,
        "monitored_joints": [
            {
                "canonical_name": "elbow",
                "joint_type": "hinge",
                "angle_rad": {"numerator": 1, "denominator": 1000},
                "angular_velocity_rad_s": {"numerator": 1, "denominator": 100},
            }
        ],
    }
    contract = ComparisonContractIdentity.from_primitive(primitive)
    assert contract.to_primitive() == primitive
    assert len(contract.sha256()) == 64
    with pytest.raises(TypeError):
        ComparisonContractIdentity.from_primitive({**primitive, "schema_version": "wrong"})
    with pytest.raises(TypeError):
        ComparisonContractIdentity.from_primitive({**primitive, "schema_version": True})
    with pytest.raises(ValueError):
        ComparisonContractIdentity.from_primitive({**primitive, "schema_version": 2})
    with pytest.raises(ValueError):
        ComparisonContractIdentity.from_primitive({**primitive, "monitored_joints": []})


def test_canonical_summary_is_strict_and_immutable() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises canonical summary is strict and immutable; accepting a contradictory
    or noncanonical value would make the signed decision evidence ambiguous.
    """
    summary = CanonicalSummary.from_primitive({"b": [1, 2], "a": {"x": True}})
    assert summary.to_primitive() == {"b": [1, 2], "a": {"x": True}}
    with pytest.raises(TypeError):
        CanonicalSummary.from_primitive([])
    with pytest.raises(TypeError):
        summary.value["x"] = 1  # type: ignore[index]


def test_direct_constructor_and_optional_branch_invariants() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises direct constructor and optional branch invariants; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    with pytest.raises(ValueError, match="joint_type"):
        JointToleranceConfig(cast(Any, "unknown"), {})
    with pytest.raises(ValueError, match="tolerance fields"):
        JointToleranceConfig("hinge", {"angle_rad": ExactRational(1, 10)})
    with pytest.raises(ValueError, match="joint_type"):
        JointToleranceConfig.from_primitive({"joint_type": "unknown"})

    parsed = ComparisonConfig.from_primitive(valid_config())
    with pytest.raises(TypeError, match="joint_tolerances"):
        replace(parsed, joint_tolerances={"elbow": cast(Any, object())})
    with_alias = valid_config()
    with_alias["aliases"] = "workloads/aliases.json"
    assert ComparisonConfig.from_primitive(with_alias).aliases == "workloads/aliases.json"
    with pytest.raises(TypeError, match="ExactRational"):
        ModelRoleConfig("models", "robot.xml", cast(Any, object()))

    with pytest.raises(ValueError, match="schema"):
        AliasArtifact("wrong", 1, digest("b"), digest("c"), (), ())
    with pytest.raises(ValueError, match="version"):
        ToolIdentity("", digest("distribution"))
    with pytest.raises(TypeError, match="alias_bindings"):
        AlignmentSummary(None, (), (), (cast(Any, 1),))
    with pytest.raises(ValueError, match="joint_type"):
        MonitoredJoint.from_primitive({"canonical_name": "joint", "joint_type": "unknown"})
    with pytest.raises(TypeError, match="summary"):
        CanonicalSummary(cast(Any, 1))


def test_optional_hashes_and_contract_ordering_branches() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises optional hashes and contract ordering branches; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    hashes = [digest(str(index)) for index in range(10)]
    identity = ComparisonInputsIdentity(
        configuration_raw_sha256=hashes[0],
        comparison_contract_sha256=hashes[1],
        baseline_model_closure_sha256=hashes[2],
        candidate_model_closure_sha256=hashes[3],
        initial_state_raw_sha256=hashes[4],
        initial_state_semantic_sha256=hashes[5],
        actions_raw_sha256=hashes[6],
        actions_semantic_sha256=hashes[7],
        aliases_raw_sha256=hashes[8],
        aliases_semantic_sha256=hashes[9],
    )
    assert identity.aliases_raw_sha256 == hashes[8]

    primitive = {
        "schema": "metrifid.comparison_contract",
        "schema_version": 1,
        "baseline_model_closure_sha256": digest("b"),
        "candidate_model_closure_sha256": digest("c"),
        "initial_state_semantic_sha256": digest("s"),
        "actions_semantic_sha256": digest("a"),
        "aliases_semantic_sha256": digest("aliases"),
        "baseline_step_dt": {"numerator": 1, "denominator": 500},
        "candidate_step_dt": {"numerator": 1, "denominator": 500},
        "control_dt": {"numerator": 1, "denominator": 100},
        "repeats": 3,
        "monitored_joints": [
            {
                "canonical_name": "elbow",
                "joint_type": "hinge",
                "angle_rad": {"numerator": 1, "denominator": 1000},
                "angular_velocity_rad_s": {"numerator": 1, "denominator": 100},
            }
        ],
    }
    contract = ComparisonContractIdentity.from_primitive(primitive)
    assert contract.aliases_semantic_sha256 == digest("aliases")
    joint = contract.monitored_joints[0]
    with pytest.raises(ValueError, match="monitored_joints"):
        replace(contract, monitored_joints=(joint, joint))


def test_sequence_count_path_and_offset_refusal_branches() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises sequence count path and offset refusal branches; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    base = {
        "schema": "metrifid.state",
        "schema_version": 1,
        "joint_names": ["x"],
        "qpos_offsets": [0, 1],
        "qpos_count": 1,
        "qvel_offsets": [0, 1],
        "qvel_count": 1,
        "actuator_names": [],
        "act_offsets": [0],
        "act_count": 0,
    }
    for mutation in (
        {"qpos_count": -1},
        {"joint_names": "x"},
        {"qpos_offsets": [0]},
        {
            "joint_names": ["a", "b"],
            "qpos_offsets": [0, 2, 1],
            "qpos_count": 1,
            "qvel_offsets": [0, 1, 2],
            "qvel_count": 2,
        },
    ):
        with pytest.raises((TypeError, ValueError)):
            StateArtifactMetadata.from_primitive({**base, **mutation})

    for path in ("assets\\mesh.bin", "assets//mesh.bin"):
        with pytest.raises(ValueError):
            ModelClosureIdentity.from_primitive(
                {
                    "entrypoint": "robot.xml",
                    "member_count": 2,
                    "members": [
                        {"path": path, "size_bytes": 1, "sha256": digest("asset")},
                        {"path": "robot.xml", "size_bytes": 1, "sha256": digest("xml")},
                    ],
                }
            )


def test_remaining_direct_nested_type_guards(
    green_candidate: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every defensive nested-type guard fails deliberately, never incidentally."""
    import metrifid._runtime_schemas as runtime_schemas
    from metrifid import finalize_receipt, validate_receipt

    with pytest.raises(TypeError, match="tolerances must be a mapping"):
        JointToleranceConfig("hinge", cast(Any, []))

    parsed = ComparisonConfig.from_primitive(valid_config())
    with pytest.raises(TypeError, match="joint_tolerances must be a mapping"):
        replace(parsed, joint_tolerances=cast(Any, []))

    finalized = finalize_receipt(green_candidate)
    with pytest.raises(TypeError, match="engine_threadpool_state"):
        replace(
            finalized.environment,
            engine_threadpool_state=cast(Any, "DISABLED"),
            environment_sha256=None,
        )

    monkeypatch.setattr(runtime_schemas, "_require_mapping_tuple", lambda value, field: None)
    with pytest.raises(TypeError, match="alias_bindings entries"):
        AlignmentSummary(None, ("elbow",), (), cast(Any, (1,)))

    with pytest.raises(TypeError, match="tolerances must be a mapping"):
        MonitoredJoint("elbow", "hinge", cast(Any, []))

    object.__setattr__(finalized, "limitations", tuple(list(finalized.limitations)[:-1]))
    with pytest.raises(ValueError, match="limitations"):
        validate_receipt(finalized)


def test_receipt_hash_requirement_and_private_sequence_guards(green_candidate: Any) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises receipt hash requirement and private sequence guards; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    import metrifid._schema_primitives as schema_primitives
    from metrifid import finalize_receipt, validate_receipt

    finalized = finalize_receipt(green_candidate)
    unhashed = replace(finalized, receipt_sha256=None)
    with pytest.raises(ValueError, match="receipt_sha256 is required"):
        validate_receipt(unhashed)

    assert schema_primitives._string_sequence(["a", "b"], "values") == ("a", "b")
    schema_primitives._unique_strings(("a", "b"), "values")
    with pytest.raises(ValueError, match="duplicates"):
        schema_primitives._unique_strings(("a", "a"), "values")
    with pytest.raises(TypeError, match="only ModelClosureMember"):
        schema_primitives._require_typed_tuple((object(),), ModelClosureMember, "members")
    with pytest.raises(TypeError, match="must be a tuple"):
        schema_primitives._require_string_tuple([], "values")
    with pytest.raises(TypeError, match="must be a tuple"):
        schema_primitives._require_int_tuple([], "values")
    with pytest.raises(TypeError, match="must be a tuple"):
        schema_primitives._require_mapping_tuple([], "values")


def test_top_level_schema_object_refuses_non_string_key_deliberately() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises top level schema object refuses non string key deliberately;
    accepting a contradictory or noncanonical value would make the signed decision evidence
    ambiguous.
    """
    with pytest.raises(TypeError, match="ComparisonConfig keys must be strings"):
        ComparisonConfig.from_primitive({1: "not-a-string-key"})
