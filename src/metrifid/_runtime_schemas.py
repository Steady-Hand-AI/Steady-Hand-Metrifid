"""Strict tool, runtime, time, and alignment identity schemas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import ClassVar, Self, cast

from ._schema_primitives import (
    _bounded_int,
    _engine_threadpool_state,
    _fields,
    _name_sequence,
    _nonempty_string,
    _nonnegative_int,
    _object,
    _optional_hash,
    _positive_rational,
    _positive_rational_primitive,
    _require_integral_ratio,
    _require_mapping_tuple,
    _require_string_tuple,
    _sequence,
    _sorted_unique_names,
    _string,
)
from .errors import EngineThreadpoolState
from .json_values import (
    CanonicalValue,
    ExactRational,
    FrozenCanonicalObject,
    canonical_json_bytes,
    compute_self_hash,
    freeze_canonical,
    require_sha256,
    thaw_canonical,
    validate_self_hash,
)


@dataclass(frozen=True, slots=True)
class ToolIdentity:
    """Bind a Metrifid version to the hash of its executing installed distribution."""

    version: str
    distribution_sha256: str

    def __post_init__(self) -> None:
        """Validate the nonempty version and installed-distribution digest."""
        _nonempty_string(self.version, "version")
        require_sha256(self.distribution_sha256, "distribution_sha256")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode an exact version/distribution-digest tool identity."""
        obj = _object(value, "ToolIdentity")
        _fields(obj, {"version", "distribution_sha256"}, "ToolIdentity")
        return cls(
            _string(obj["version"], "version"),
            require_sha256(obj["distribution_sha256"], "distribution_sha256"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit tool version and installed-distribution SHA-256."""
        return {"version": self.version, "distribution_sha256": self.distribution_sha256}


@dataclass(frozen=True, slots=True)
class EnvironmentIdentity:
    """Identify the Python, MuJoCo, native library, host, CPU, and threadpool runtime."""

    mujoco_version: str
    python_version: str
    numpy_version: str
    mujoco_python_distribution_sha256: str
    mujoco_native_library_sha256: str
    platform: str
    platform_release: str
    libc: str
    cpu_identity_sha256: str
    engine_threadpool_state: EngineThreadpoolState
    environment_sha256: str | None

    _HASH_FIELD: ClassVar[str] = "environment_sha256"

    def __post_init__(self) -> None:
        """Validate runtime descriptors, component digests, threadpool state, and self-hash."""
        for field_name in (
            "mujoco_version",
            "python_version",
            "numpy_version",
            "platform",
            "platform_release",
            "libc",
        ):
            _nonempty_string(getattr(self, field_name), field_name)
        for field_name in (
            "mujoco_python_distribution_sha256",
            "mujoco_native_library_sha256",
            "cpu_identity_sha256",
        ):
            require_sha256(getattr(self, field_name), field_name)
        if not isinstance(self.engine_threadpool_state, EngineThreadpoolState):
            raise TypeError("engine_threadpool_state must be an EngineThreadpoolState")
        if self.environment_sha256 is not None:
            require_sha256(self.environment_sha256, self._HASH_FIELD)
            validate_self_hash(self.to_primitive(), self._HASH_FIELD)

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode the exact bounded runtime-environment identity object."""
        obj = _object(value, "EnvironmentIdentity")
        expected = {
            "mujoco_version",
            "python_version",
            "numpy_version",
            "mujoco_python_distribution_sha256",
            "mujoco_native_library_sha256",
            "platform",
            "platform_release",
            "libc",
            "cpu_identity_sha256",
            "engine_threadpool_state",
            "environment_sha256",
        }
        _fields(obj, expected, "EnvironmentIdentity")
        return cls(
            _nonempty_string(obj["mujoco_version"], "mujoco_version"),
            _nonempty_string(obj["python_version"], "python_version"),
            _nonempty_string(obj["numpy_version"], "numpy_version"),
            require_sha256(
                obj["mujoco_python_distribution_sha256"],
                "mujoco_python_distribution_sha256",
            ),
            require_sha256(obj["mujoco_native_library_sha256"], "mujoco_native_library_sha256"),
            _nonempty_string(obj["platform"], "platform"),
            _nonempty_string(obj["platform_release"], "platform_release"),
            _nonempty_string(obj["libc"], "libc"),
            require_sha256(obj["cpu_identity_sha256"], "cpu_identity_sha256"),
            _engine_threadpool_state(obj["engine_threadpool_state"]),
            _optional_hash(obj["environment_sha256"], "environment_sha256"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit all runtime component identities and the optional aggregate hash."""
        return {
            "mujoco_version": self.mujoco_version,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "mujoco_python_distribution_sha256": self.mujoco_python_distribution_sha256,
            "mujoco_native_library_sha256": self.mujoco_native_library_sha256,
            "platform": self.platform,
            "platform_release": self.platform_release,
            "libc": self.libc,
            "cpu_identity_sha256": self.cpu_identity_sha256,
            "engine_threadpool_state": self.engine_threadpool_state.value,
            "environment_sha256": self.environment_sha256,
        }

    def finalized(self) -> Self:
        """Return this runtime identity with a validated canonical self-hash."""
        if self.environment_sha256 is not None:
            validate_self_hash(self.to_primitive(), self._HASH_FIELD)
            return self
        primitive = self.to_primitive()
        return replace(self, environment_sha256=compute_self_hash(primitive, self._HASH_FIELD))

    def validate_hash(self) -> None:
        """Recompute and require the environment identity's aggregate self-hash."""
        validate_self_hash(self.to_primitive(), self._HASH_FIELD)


@dataclass(frozen=True, slots=True)
class TimeContract:
    """Freeze exact role timesteps, control schedule, horizon, and sampling semantics."""

    baseline_step_dt: ExactRational
    candidate_step_dt: ExactRational
    control_dt: ExactRational
    control_intervals: int
    state_samples: int
    horizon: ExactRational
    sample_phase: str
    action_semantics: str
    terminal_sample: str
    interpolation: str

    def __post_init__(self) -> None:
        """Validate exact divisibility, sample counts, horizon, and frozen replay semantics."""
        for field_name in ("baseline_step_dt", "candidate_step_dt", "control_dt", "horizon"):
            _positive_rational(getattr(self, field_name), field_name)
        _bounded_int(self.control_intervals, "control_intervals", 1, 100_000)
        _nonnegative_int(self.state_samples, "state_samples")
        if self.state_samples != self.control_intervals + 1:
            raise ValueError("state_samples must equal control_intervals + 1")
        if self.horizon != self.control_dt.multiplied_by_int(self.control_intervals):
            raise ValueError("horizon must equal control_dt multiplied by control_intervals")
        _require_integral_ratio(self.control_dt, self.baseline_step_dt, "baseline_step_dt")
        _require_integral_ratio(self.control_dt, self.candidate_step_dt, "candidate_step_dt")
        constants = {
            "sample_phase": "BOUNDARY_BEFORE_CONTROL",
            "action_semantics": "LEFT_BOUNDARY_ZERO_ORDER_HOLD",
            "terminal_sample": "INCLUDED",
            "interpolation": "FORBIDDEN",
        }
        for field_name, expected in constants.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} must be {expected}")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode exact rational timing and the frozen boundary-sampling policy."""
        obj = _object(value, "TimeContract")
        expected = {
            "baseline_step_dt",
            "candidate_step_dt",
            "control_dt",
            "control_intervals",
            "state_samples",
            "horizon",
            "sample_phase",
            "action_semantics",
            "terminal_sample",
            "interpolation",
        }
        _fields(obj, expected, "TimeContract")
        return cls(
            _positive_rational_primitive(obj["baseline_step_dt"], "baseline_step_dt"),
            _positive_rational_primitive(obj["candidate_step_dt"], "candidate_step_dt"),
            _positive_rational_primitive(obj["control_dt"], "control_dt"),
            _bounded_int(obj["control_intervals"], "control_intervals", 1, 100_000),
            _nonnegative_int(obj["state_samples"], "state_samples"),
            _positive_rational_primitive(obj["horizon"], "horizon"),
            _string(obj["sample_phase"], "sample_phase"),
            _string(obj["action_semantics"], "action_semantics"),
            _string(obj["terminal_sample"], "terminal_sample"),
            _string(obj["interpolation"], "interpolation"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit exact timing primitives, schedule counts, and sampling-policy tokens."""
        return {
            "baseline_step_dt": self.baseline_step_dt.to_primitive(),
            "candidate_step_dt": self.candidate_step_dt.to_primitive(),
            "control_dt": self.control_dt.to_primitive(),
            "control_intervals": self.control_intervals,
            "state_samples": self.state_samples,
            "horizon": self.horizon.to_primitive(),
            "sample_phase": self.sample_phase,
            "action_semantics": self.action_semantics,
            "terminal_sample": self.terminal_sample,
            "interpolation": self.interpolation,
        }


@dataclass(frozen=True, slots=True)
class AlignmentSummary:
    """Summarize canonical joint, actuator, and alias order as self-hashed evidence."""

    alignment_sha256: str | None
    joint_order: tuple[str, ...]
    actuator_order: tuple[str, ...]
    alias_bindings: tuple[FrozenCanonicalObject, ...]

    _HASH_FIELD: ClassVar[str] = "alignment_sha256"

    def __post_init__(self) -> None:
        """Validate ordered unique names, canonical alias bindings, and optional self-hash."""
        if self.alignment_sha256 is not None:
            require_sha256(self.alignment_sha256, self._HASH_FIELD)
            validate_self_hash(self.to_primitive(), self._HASH_FIELD)
        _require_string_tuple(self.joint_order, "joint_order", names=True)
        _require_string_tuple(self.actuator_order, "actuator_order", names=True)
        _require_mapping_tuple(self.alias_bindings, "alias_bindings")
        _sorted_unique_names(self.joint_order, "joint_order")
        _sorted_unique_names(self.actuator_order, "actuator_order")
        normalized: list[FrozenCanonicalObject] = []
        for binding in self.alias_bindings:
            if not isinstance(binding, Mapping):
                raise TypeError("alias_bindings entries must be objects")
            thawed = {key: thaw_canonical(item) for key, item in binding.items()}
            frozen = freeze_canonical(cast(CanonicalValue, thawed))
            normalized.append(cast(FrozenCanonicalObject, frozen))
        if [canonical_json_bytes(thaw_canonical(item)) for item in normalized] != sorted(
            canonical_json_bytes(thaw_canonical(item)) for item in normalized
        ):
            raise ValueError("alias_bindings must be in canonical byte order")
        object.__setattr__(self, "alias_bindings", tuple(normalized))

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode canonical model-element order and frozen alias-binding evidence."""
        obj = _object(value, "AlignmentSummary")
        _fields(
            obj,
            {"alignment_sha256", "joint_order", "actuator_order", "alias_bindings"},
            "AlignmentSummary",
        )
        bindings: list[FrozenCanonicalObject] = []
        for item in _sequence(obj["alias_bindings"], "alias_bindings"):
            item_obj = _object(item, "alias_binding")
            frozen = freeze_canonical(cast(CanonicalValue, item_obj))
            bindings.append(cast(FrozenCanonicalObject, frozen))
        return cls(
            _optional_hash(obj["alignment_sha256"], "alignment_sha256"),
            _name_sequence(obj["joint_order"], "joint_order"),
            _name_sequence(obj["actuator_order"], "actuator_order"),
            tuple(bindings),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit alignment hash, ordered names, and thawed canonical alias bindings."""
        return {
            "alignment_sha256": self.alignment_sha256,
            "joint_order": list(self.joint_order),
            "actuator_order": list(self.actuator_order),
            "alias_bindings": [thaw_canonical(binding) for binding in self.alias_bindings],
        }

    def finalized(self) -> Self:
        """Return this alignment summary with a validated canonical self-hash."""
        if self.alignment_sha256 is not None:
            validate_self_hash(self.to_primitive(), self._HASH_FIELD)
            return self
        return replace(
            self,
            alignment_sha256=compute_self_hash(self.to_primitive(), self._HASH_FIELD),
        )

    def validate_hash(self) -> None:
        """Recompute and require the alignment summary's self-hash."""
        validate_self_hash(self.to_primitive(), self._HASH_FIELD)
