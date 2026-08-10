"""Adversarial receipt contract, identity, alignment, and tolerance bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest

from metrifid import ComparisonReceipt, canonical_sha256, finalize_receipt
from metrifid.json_values import CanonicalValue, compute_self_hash

Primitive = dict[str, Any]
Mutation = Callable[[Primitive], None]


def _completed_primitive(green_candidate: ComparisonReceipt) -> Primitive:
    """Construct the completed primitive fixture used by receipt cross invariants scenarios.

    Deterministic setup isolates receipt cross invariants without bypassing the contract
    boundary under assertion.
    """
    return cast(Primitive, finalize_receipt(green_candidate).to_primitive())


def _rehash_receipt(value: Primitive) -> Primitive:
    """Compute the canonical rehash receipt value used by receipt cross invariants fixtures.

    Content addressing keeps the mutation boundary explicit for receipt cross invariants.
    """
    value["receipt_sha256"] = compute_self_hash(
        cast(dict[str, CanonicalValue], value), "receipt_sha256"
    )
    return value


def _sync_contract_digest(value: Primitive) -> None:
    """Compute the canonical sync contract digest value used by receipt cross invariants fixtures.

    Content addressing keeps the mutation boundary explicit for receipt cross invariants.
    """
    contract = cast(dict[str, CanonicalValue], value["comparison_contract"])
    inputs = cast(Primitive, value["inputs"])
    inputs["comparison_contract_sha256"] = canonical_sha256(contract)


def _mutate_contract_hash(value: Primitive) -> None:
    """Apply the targeted mutate contract hash mutation to otherwise valid evidence.

    Resealing the result isolates the semantic contradiction exercised by receipt cross
    invariants.
    """
    cast(Primitive, value["inputs"])["comparison_contract_sha256"] = "1" * 64


def _mutate_baseline_closure_hash(value: Primitive) -> None:
    """Apply the targeted mutate baseline closure hash mutation to otherwise valid evidence.

    Resealing the result isolates the semantic contradiction exercised by receipt cross
    invariants.
    """
    cast(Primitive, value["inputs"])["baseline_model_closure_sha256"] = "2" * 64


def _mutate_contract_baseline_closure_hash(value: Primitive) -> None:
    """Apply the targeted mutate contract baseline closure hash mutation to otherwise valid evidence.

    Resealing the result isolates the semantic contradiction exercised by receipt cross
    invariants.
    """
    cast(Primitive, value["comparison_contract"])["baseline_model_closure_sha256"] = "7" * 64
    _sync_contract_digest(value)


def _mutate_candidate_closure_hash(value: Primitive) -> None:
    """Apply the targeted mutate candidate closure hash mutation to otherwise valid evidence.

    Resealing the result isolates the semantic contradiction exercised by receipt cross
    invariants.
    """
    cast(Primitive, value["comparison_contract"])["candidate_model_closure_sha256"] = "3" * 64
    _sync_contract_digest(value)


def _mutate_initial_state_hash(value: Primitive) -> None:
    """Apply the targeted mutate initial state hash mutation to otherwise valid evidence.

    Resealing the result isolates the semantic contradiction exercised by receipt cross
    invariants.
    """
    cast(Primitive, value["comparison_contract"])["initial_state_semantic_sha256"] = "4" * 64
    _sync_contract_digest(value)


def _mutate_actions_hash(value: Primitive) -> None:
    """Apply the targeted mutate actions hash mutation to otherwise valid evidence.

    Resealing the result isolates the semantic contradiction exercised by receipt cross
    invariants.
    """
    cast(Primitive, value["comparison_contract"])["actions_semantic_sha256"] = "5" * 64
    _sync_contract_digest(value)


def _mutate_alias_hash(value: Primitive) -> None:
    """Apply the targeted mutate alias hash mutation to otherwise valid evidence.

    Resealing the result isolates the semantic contradiction exercised by receipt cross
    invariants.
    """
    cast(Primitive, value["comparison_contract"])["aliases_semantic_sha256"] = "6" * 64
    _sync_contract_digest(value)


def _mutate_baseline_step_dt(value: Primitive) -> None:
    """Apply the targeted mutate baseline step dt mutation to otherwise valid evidence.

    Resealing the result isolates the semantic contradiction exercised by receipt cross
    invariants.
    """
    cast(Primitive, value["comparison_contract"])["baseline_step_dt"] = {
        "numerator": 1,
        "denominator": 1000,
    }
    _sync_contract_digest(value)


def _mutate_candidate_step_dt(value: Primitive) -> None:
    """Apply the targeted mutate candidate step dt mutation to otherwise valid evidence.

    Resealing the result isolates the semantic contradiction exercised by receipt cross
    invariants.
    """
    cast(Primitive, value["comparison_contract"])["candidate_step_dt"] = {
        "numerator": 1,
        "denominator": 1000,
    }
    _sync_contract_digest(value)


def _mutate_control_dt(value: Primitive) -> None:
    """Apply the targeted mutate control dt mutation to otherwise valid evidence.

    Resealing the result isolates the semantic contradiction exercised by receipt cross
    invariants.
    """
    cast(Primitive, value["comparison_contract"])["control_dt"] = {
        "numerator": 1,
        "denominator": 50,
    }
    _sync_contract_digest(value)


def _mutate_monitored_tuple(value: Primitive) -> None:
    """Apply the targeted mutate monitored tuple mutation to otherwise valid evidence.

    Resealing the result isolates the semantic contradiction exercised by receipt cross
    invariants.
    """
    contract = cast(Primitive, value["comparison_contract"])
    joint = cast(Primitive, cast(list[object], contract["monitored_joints"])[0])
    joint["canonical_name"] = "wrist"
    _sync_contract_digest(value)


def _mutate_alignment_missing_monitored(value: Primitive) -> None:
    """Apply the targeted mutate alignment missing monitored mutation to otherwise valid evidence.

    Resealing the result isolates the semantic contradiction exercised by receipt cross
    invariants.
    """
    alignment = cast(Primitive, value["alignment"])
    alignment["joint_order"] = ["wrist"]
    alignment["alignment_sha256"] = compute_self_hash(
        cast(dict[str, CanonicalValue], alignment), "alignment_sha256"
    )


def _mutate_tolerance_projection(value: Primitive) -> None:
    """Apply the targeted mutate tolerance projection mutation to otherwise valid evidence.

    Resealing the result isolates the semantic contradiction exercised by receipt cross
    invariants.
    """
    tolerances = cast(Primitive, value["tolerances"])
    elbow = cast(Primitive, tolerances["elbow"])
    elbow["angle_rad"] = {"numerator": 1, "denominator": 500}


@pytest.mark.parametrize(
    "mutation",
    [
        _mutate_contract_hash,
        _mutate_baseline_closure_hash,
        _mutate_contract_baseline_closure_hash,
        _mutate_candidate_closure_hash,
        _mutate_initial_state_hash,
        _mutate_actions_hash,
        _mutate_alias_hash,
        _mutate_baseline_step_dt,
        _mutate_candidate_step_dt,
        _mutate_control_dt,
        _mutate_monitored_tuple,
        _mutate_alignment_missing_monitored,
        _mutate_tolerance_projection,
    ],
    ids=lambda function: function.__name__.removeprefix("_mutate_"),
)
def test_each_cross_invariant_mutation_refuses_after_outer_rehash(
    green_candidate: ComparisonReceipt,
    mutation: Mutation,
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises each cross invariant mutation refuses after outer rehash; accepting
    a contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    value = _completed_primitive(green_candidate)
    mutation(value)
    _rehash_receipt(value)
    with pytest.raises((TypeError, ValueError)):
        ComparisonReceipt.from_primitive(value)


def test_extra_unmonitored_aligned_joint_remains_valid(
    green_candidate: ComparisonReceipt,
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises extra unmonitored aligned joint remains valid; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    value = _completed_primitive(green_candidate)
    alignment = cast(Primitive, value["alignment"])
    alignment["joint_order"] = ["elbow", "wrist"]
    alignment["alignment_sha256"] = compute_self_hash(
        cast(dict[str, CanonicalValue], alignment), "alignment_sha256"
    )
    _rehash_receipt(value)
    parsed = ComparisonReceipt.from_primitive(value)
    assert parsed.alignment.joint_order == ("elbow", "wrist")
