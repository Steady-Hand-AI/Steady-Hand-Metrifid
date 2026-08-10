"""Shared constants for strict Metrifid schemas."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, TypeAlias

JointType: TypeAlias = Literal["hinge", "slide", "ball", "free"]

_RECEIPT_SCHEMA = "metrifid.comparison_receipt"
_RECEIPT_SCHEMA_VERSION = 1
_STATUS_RULE_SCHEMA = "metrifid.status_precedence"
_STATUS_RULE_SCHEMA_VERSION = 1
_CONTRACT_SCHEMA = "metrifid.comparison_contract"
_CONTRACT_SCHEMA_VERSION = 1
_STATE_SCHEMA = "metrifid.state"
_STATE_SCHEMA_VERSION = 1
_ACTIONS_SCHEMA = "metrifid.actions"
_ACTIONS_SCHEMA_VERSION = 1
_ALIASES_SCHEMA = "metrifid.aliases"
_ALIASES_SCHEMA_VERSION = 1
_METRICS_BY_JOINT_TYPE: Mapping[JointType, tuple[str, ...]] = MappingProxyType(
    {
        "hinge": ("angle_rad", "angular_velocity_rad_s"),
        "slide": ("translation_m", "linear_velocity_m_s"),
        "ball": ("orientation_rad", "angular_velocity_rad_s"),
        "free": (
            "translation_m",
            "orientation_rad",
            "linear_velocity_m_s",
            "angular_velocity_rad_s",
        ),
    }
)
