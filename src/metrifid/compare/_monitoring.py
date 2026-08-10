"""Pure monitored-joint admission for the comparison hinge/slide metric surface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from .._model_closure import AlignedJoint
from .._npz import refuse
from ..operational import OperationalReasonCode
from ..schemas import ComparisonConfig, MonitoredJoint


def monitored_joints_from_config(
    config: ComparisonConfig,
    aligned_joints: Sequence[AlignedJoint],
) -> tuple[MonitoredJoint, ...]:
    """Bind declared tolerances to aligned joints and refuse unsupported families."""
    by_name = {item.canonical_name: item for item in aligned_joints}
    result: list[MonitoredJoint] = []
    for name, tolerance in config.joint_tolerances.items():
        aligned = by_name.get(name)
        if aligned is None:
            raise refuse(
                OperationalReasonCode.MONITORED_JOINT_NOT_ALIGNED,
                object_name=name,
            )
        compiled_type = cast(str, aligned.joint_type)
        schema_type = compiled_type.lower()
        if compiled_type not in {"HINGE", "SLIDE"}:
            raise refuse(
                OperationalReasonCode.TOLERANCE_UNIT_MISMATCH,
                object_name=name,
                issue="monitoring_supports_hinge_and_slide_only",
                compiled_joint_type=compiled_type,
            )
        if tolerance.joint_type != schema_type:
            raise refuse(
                OperationalReasonCode.TOLERANCE_UNIT_MISMATCH,
                object_name=name,
                declared_joint_type=tolerance.joint_type,
                compiled_joint_type=compiled_type,
            )
        result.append(MonitoredJoint(name, schema_type, tolerance.tolerances))
    return tuple(result)


__all__ = ["monitored_joints_from_config"]
