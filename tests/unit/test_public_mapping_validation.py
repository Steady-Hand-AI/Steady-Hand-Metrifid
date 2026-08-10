"""Deliberate validation for malformed caller mappings."""

from __future__ import annotations

from typing import Any, cast

import pytest

from metrifid import ComparisonConfig, ExactRational
from metrifid.schemas import JointToleranceConfig, ModelRoleConfig, MonitoredJoint


def _tolerance() -> JointToleranceConfig:
    """Construct the tolerance fixture used by public mapping validation scenarios.

    Deterministic setup isolates public mapping validation without bypassing the contract
    boundary under assertion.
    """
    return JointToleranceConfig(
        "hinge",
        {
            "angle_rad": ExactRational(1, 1000),
            "angular_velocity_rad_s": ExactRational(1, 100),
        },
    )


def _config(mapping: object) -> ComparisonConfig:
    """Construct the config fixture used by public mapping validation scenarios.

    Deterministic setup isolates public mapping validation without bypassing the contract
    boundary under assertion.
    """
    role = ModelRoleConfig("models", "robot.xml", ExactRational(1, 500))
    return ComparisonConfig(
        1,
        role,
        role,
        "state.npz",
        "actions.npz",
        ExactRational(1, 100),
        3,
        cast(Any, mapping),
        None,
        "results",
    )


@pytest.mark.parametrize(
    "mapping",
    [
        {1: _tolerance()},
        {"elbow": _tolerance(), 1: _tolerance()},
    ],
)
def test_joint_tolerances_non_string_or_mixed_keys_are_deliberate(mapping: object) -> None:
    """Protect the public mapping validation assurance boundary from behavioral drift.

    This scenario exercises joint tolerances non string or mixed keys are deliberate; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    with pytest.raises(TypeError, match="joint_tolerances keys must be strings") as exc_info:
        _config(mapping)
    assert not isinstance(exc_info.value, (AttributeError, KeyError))


def test_joint_tolerances_wrong_nested_value_is_field_named() -> None:
    """Protect the public mapping validation assurance boundary from behavioral drift.

    This scenario exercises joint tolerances wrong nested value is field named; the assertions
    pin the user-visible result and the evidence needed to explain that result.
    """
    with pytest.raises(
        TypeError, match="joint_tolerances values must be JointToleranceConfig"
    ) as exc_info:
        _config({"elbow": object()})
    assert not isinstance(exc_info.value, (AttributeError, KeyError))


@pytest.mark.parametrize(
    ("factory", "mapping", "message"),
    [
        (
            lambda mapping: JointToleranceConfig("hinge", cast(Any, mapping)),
            {1: ExactRational(1, 100)},
            "tolerances keys must be strings",
        ),
        (
            lambda mapping: MonitoredJoint("elbow", "hinge", cast(Any, mapping)),
            {"angle_rad": ExactRational(1, 1000), 1: ExactRational(1, 100)},
            "tolerances keys must be strings",
        ),
    ],
)
def test_other_public_sorted_mappings_validate_keys_first(
    factory: Any,
    mapping: object,
    message: str,
) -> None:
    """Protect the public mapping validation assurance boundary from behavioral drift.

    This scenario exercises other public sorted mappings validate keys first; the assertions pin
    the user-visible result and the evidence needed to explain that result.
    """
    with pytest.raises(TypeError, match=message) as exc_info:
        factory(mapping)
    assert not isinstance(exc_info.value, (AttributeError, KeyError))


def test_tolerance_mapping_validates_values_before_semantic_set() -> None:
    """Protect the public mapping validation assurance boundary from behavioral drift.

    This scenario exercises tolerance mapping validates values before semantic set; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    mapping = {"angle_rad": object(), "angular_velocity_rad_s": ExactRational(1, 100)}
    with pytest.raises(TypeError, match="tolerances values must be ExactRational"):
        JointToleranceConfig("hinge", cast(Any, mapping))
