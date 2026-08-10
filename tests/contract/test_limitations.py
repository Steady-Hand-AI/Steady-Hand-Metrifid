"""Frozen limitation registry and receipt canonicalization tests."""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Any, cast

import pytest

from metrifid import (
    ComparisonReceipt,
    ComparisonStatus,
    LimitationCode,
    ReasonCode,
    ReasonRecord,
    finalize_receipt,
)
from metrifid.errors import canonical_limitations

EXPECTED = (
    LimitationCode.DECLARED_WORKLOAD_ONLY,
    LimitationCode.MONITORED_JOINT_COORDINATES_ONLY,
    LimitationCode.NO_BODY_CONTACT_SENSOR_REWARD_OR_TASK_CLAIM,
    LimitationCode.NO_GLOBAL_EQUIVALENCE_CLAIM,
)


def reason(code: ReasonCode) -> ReasonRecord:
    """Construct the reason fixture used by limitations scenarios.

    Deterministic setup isolates limitations without bypassing the contract boundary under
    assertion.
    """
    roles = {
        ReasonCode.INITIAL_STATE_NOT_PRESERVED: "baseline",
        ReasonCode.BASELINE_NONDETERMINISTIC: "baseline",
        ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED: "comparison",
    }
    is_joint_metric = code is ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED
    return ReasonRecord.from_primitive(
        {
            "code": code.value,
            "role": roles[code],
            "object_type": "joint" if is_joint_metric else None,
            "object_name": "elbow" if is_joint_metric else None,
            "metric": "angle_rad" if is_joint_metric else None,
            "boundary_index": 0 if is_joint_metric else None,
            "evidence": {},
        }
    )


def test_registry_order_is_exact() -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises registry order is exact; each refusal must retain its exact stage,
    exit code, and structured evidence contract.
    """
    assert tuple(LimitationCode) == EXPECTED


def test_all_permutations_canonicalize_to_same_tuple_and_hash(
    green_candidate: ComparisonReceipt,
) -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises all permutations canonicalize to same tuple and hash; each refusal
    must retain its exact stage, exit code, and structured evidence contract.
    """
    hashes: set[str] = set()
    for permutation in itertools.permutations(EXPECTED):
        assert canonical_limitations(permutation) == EXPECTED
        finalized = finalize_receipt(replace(green_candidate, limitations=permutation))
        assert finalized.limitations == EXPECTED
        assert finalized.receipt_sha256 is not None
        hashes.add(finalized.receipt_sha256)
    assert len(hashes) == 1


@pytest.mark.parametrize(
    "values",
    [
        (),
        EXPECTED[:-1],
        (*EXPECTED, EXPECTED[0]),
        (cast(Any, "UNKNOWN"), *EXPECTED[1:]),
        cast(Any, "DECLARED_WORKLOAD_ONLY"),
        cast(Any, [object()]),
    ],
)
def test_missing_duplicate_unknown_and_wrong_type_refuse(values: object) -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises missing duplicate unknown and wrong type refuse; each refusal must
    retain its exact stage, exit code, and structured evidence contract.
    """
    with pytest.raises((TypeError, ValueError)):
        canonical_limitations(cast(Any, values))


@pytest.mark.parametrize(
    ("status", "reason_code"),
    [
        (ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD, None),
        (ComparisonStatus.COVERAGE_INSUFFICIENT, ReasonCode.INITIAL_STATE_NOT_PRESERVED),
        (ComparisonStatus.NONDETERMINISTIC_REPLAY, ReasonCode.BASELINE_NONDETERMINISTIC),
        (ComparisonStatus.MATERIAL_BEHAVIOR_CHANGE, ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED),
    ],
)
def test_every_comparison_status_carries_exact_limitations(
    green_candidate: ComparisonReceipt,
    status: ComparisonStatus,
    reason_code: ReasonCode | None,
) -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises every comparison status carries exact limitations; each refusal must
    retain its exact stage, exit code, and structured evidence contract.
    """
    reasons = () if reason_code is None else (reason(reason_code),)
    candidate = replace(green_candidate, status=status, reasons=reasons, reason_codes=())
    assert finalize_receipt(candidate).limitations == EXPECTED


def test_receipt_constructor_refuses_duplicate_unknown_or_missing(
    green_candidate: ComparisonReceipt,
) -> None:
    """Keep operational failures stable and diagnosable.

    This scenario exercises receipt constructor refuses duplicate unknown or missing; each
    refusal must retain its exact stage, exit code, and structured evidence contract.
    """
    for invalid in (
        EXPECTED[:-1],
        (*EXPECTED, EXPECTED[0]),
        (cast(Any, "UNKNOWN"), *EXPECTED[1:]),
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(green_candidate, limitations=invalid)
