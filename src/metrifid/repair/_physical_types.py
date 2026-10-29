"""Native-free physical measurements and admitted reset state."""

from dataclasses import dataclass, field
from typing import Literal

Classification = Literal["COMPLETE", "HARD_REJECTION", "INCOMPLETE"]


@dataclass(frozen=True)
class Measurement:
    """One measured trajectory; absent hold data is never a zero error."""

    classification: Classification
    qualified: bool
    reason: str
    hold_maxima: tuple[float, float, float, float] | None = None
    metrics: dict[str, object] = field(default_factory=dict)
    hard_failures: tuple[str, ...] = ()
    completed_steps: int = 0
    expected_steps: int = 0
    dt_s: float = 0.0
    hold_samples: int = 0
    expected_hold_samples: int = 0


@dataclass(frozen=True)
class Snapshot:
    """Physical arrays with explicit ranks, pose and generated-model identity."""

    arrays: tuple[tuple[str, tuple[int, ...], tuple[float, ...]], ...]
    settled_block_com: tuple[float, float, float]
    settled_block_yaw: float
    nominal_scene_sha256: str
    state_sha256: str
    settling_failures: tuple[str, ...] = ()


def _snapshot_layout_valid(saved: object, nominal_scene_sha256: str) -> bool:
    """Check the fixed state layout and finite content before native setup."""
    import hashlib
    import math
    import struct

    if not isinstance(saved, Snapshot) or saved.nominal_scene_sha256 != nominal_scene_sha256:
        return False
    expected = {"qpos": (9,), "qvel": (8,), "act": (0,), "mocap_pos": (0, 3), "mocap_quat": (0, 4)}
    if len(saved.arrays) != len(expected):
        return False
    arrays = {name: (shape, values) for name, shape, values in saved.arrays}
    if set(arrays) != set(expected):
        return False
    payload = bytearray()
    for name in sorted(expected):
        shape, values = arrays[name]
        if any(type(dimension) is not int for dimension in shape):
            return False
        if shape != expected[name] or len(values) != math.prod(shape):
            return False
        if not all(type(value) is float and math.isfinite(value) for value in values):
            return False
        payload.extend(struct.pack("<" + "d" * len(values), *values))
    return (
        hashlib.sha256(payload).hexdigest() == saved.state_sha256
        and len(saved.settled_block_com) == 3
        and all(type(v) is float and math.isfinite(v) for v in saved.settled_block_com)
        and type(saved.settled_block_yaw) is float
        and math.isfinite(saved.settled_block_yaw)
        and all(type(reason) is str for reason in saved.settling_failures)
    )


def snapshot_well_formed(saved: object, nominal_scene_sha256: str) -> bool:
    """Refuse malformed typed records before any native setup."""
    try:
        return _snapshot_layout_valid(saved, nominal_scene_sha256)
    except (TypeError, ValueError, AttributeError, OverflowError):
        return False


def validate_snapshot(saved: object, fresh: Snapshot) -> bool:
    """Require equality with an independently settled admitted physical state."""
    return snapshot_well_formed(saved, fresh.nominal_scene_sha256) and saved == fresh
