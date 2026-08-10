"""Strict timestep-audit configuration parsing and reference admission.

Everything that turns one `timestep_audit.json` file into an admitted, fully typed configuration
lives here, together with the exact-rational helpers the rest of the audit shares.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final, cast

from ._json_admission import (
    CONFIG_JSON_LIMITS,
    JsonAdmissionError,
    bounded_strict_json_loads,
)
from ._model_closure import ModelAdmissionRefusal, measure_model_closure
from .compare._failure import ComparisonOperationError, operational_error
from .distribution import installed_distribution_sha256
from .errors import ReasonRole
from .json_values import CanonicalValue, ExactRational, canonical_json_bytes
from .operational import OperationalReasonCode, OperationalToolObservation
from .version import __version__

OPERATION: Final = "audit-timestep"
_WORKLOAD_KINDS: Final = frozenset({"REAL_PROJECT", "SCREENING"})
_MAX_CANDIDATES: Final = 12
_CONFIG_KEYS: Final = frozenset(
    {
        "schema_version",
        "model_root",
        "entrypoint",
        "initial_state",
        "actions",
        "control_dt",
        "repeats",
        "joint_tolerances",
        "candidate_step_dts",
        "workload_kind",
        "workload_label",
        "output_dir",
    }
)


class AuditAbort(RuntimeError):
    """One completed audit-level operational failure that stops the whole audit."""

    def __init__(self, error: ComparisonOperationError) -> None:
        """Wrap the completed operational failure that aborts the whole audit."""
        self.error = error
        super().__init__(error.failure.reason.code.value)


@dataclass(frozen=True, slots=True)
class AuditConfig:
    """The exact frozen audit configuration shape, parsed privately."""

    model_root: str
    entrypoint: str
    initial_state: str
    actions: str
    control_dt: ExactRational
    repeats: int
    joint_tolerances: Mapping[str, CanonicalValue]
    candidate_step_dts: tuple[ExactRational, ...]
    workload_kind: str
    workload_label: str
    output_dir: str


def _tool() -> OperationalToolObservation:
    """Measure the executing installed Metrifid distribution for audit evidence."""
    return OperationalToolObservation(
        __version__, "VERIFIED_INSTALLED_DISTRIBUTION", installed_distribution_sha256()
    )


def _abort(
    tool: OperationalToolObservation,
    code: OperationalReasonCode,
    evidence: Mapping[str, CanonicalValue],
    *,
    field: str | None = None,
) -> AuditAbort:
    """Build an audit-level operational failure and wrap it as ``AuditAbort``."""
    return AuditAbort(
        operational_error(
            tool=tool,
            code=code,
            role=None,
            evidence=evidence,
            field=field,
            operation=OPERATION,
        )
    )


def _validate_scalar_fields(
    obj: Mapping[str, CanonicalValue], fail: Callable[[str, str], AuditAbort]
) -> None:
    """Require the exact scalar types the frozen configuration shape declares."""
    if obj["schema_version"] != 1 or type(obj["schema_version"]) is not int:
        raise fail("schema_version must be exactly 1", "schema_version")
    for name in (
        "model_root",
        "entrypoint",
        "initial_state",
        "actions",
        "workload_label",
        "output_dir",
    ):
        if type(obj[name]) is not str or not obj[name]:
            raise fail(f"{name} must be a nonempty string", name)
    if obj["workload_kind"] not in _WORKLOAD_KINDS:
        raise fail("workload_kind must be REAL_PROJECT or SCREENING", "workload_kind")
    if type(obj["repeats"]) is not int:
        raise fail("repeats must be an integer", "repeats")
    if type(obj["joint_tolerances"]) is not dict or not obj["joint_tolerances"]:
        raise fail("joint_tolerances must be a nonempty object", "joint_tolerances")


def _positive_decimal(
    value: CanonicalValue, field: str, fail: Callable[[str, str], AuditAbort]
) -> ExactRational:
    """Parse one exact positive decimal token."""
    try:
        parsed = ExactRational.from_decimal_token(cast(str, value))
    except (TypeError, ValueError) as exc:
        raise fail(f"{field} is not a valid decimal token: {exc}", field) from exc
    if parsed.numerator <= 0:
        raise fail(f"{field} must be positive", field)
    return parsed


def _parse_candidate_timesteps(
    tokens: CanonicalValue, fail: Callable[[str, str], AuditAbort]
) -> tuple[ExactRational, ...]:
    """Parse, uniquely admit and order the declared candidate timesteps."""
    field = "candidate_step_dts"
    if type(tokens) is not list or not (1 <= len(tokens) <= _MAX_CANDIDATES):
        raise fail(
            f"candidate_step_dts must be a list of 1 to {_MAX_CANDIDATES} decimal tokens", field
        )
    parsed: list[ExactRational] = []
    for token in tokens:
        try:
            value = ExactRational.from_decimal_token(cast(str, token))
        except (TypeError, ValueError) as exc:
            raise fail(f"candidate timestep is not a valid decimal token: {exc}", field) from exc
        if value.numerator <= 0:
            raise fail("candidate timestep must be positive", field)
        parsed.append(value)
    # Uniqueness is by normalized exact rational value, so 0.002 and 0.0020 collide.
    seen: set[tuple[int, int]] = set()
    for value in parsed:
        key = (value.numerator, value.denominator)
        if key in seen:
            raise fail(
                "candidate timesteps must be unique by normalized exact rational value", field
            )
        seen.add(key)
    return tuple(sorted(parsed, key=_rational_sort_key))


def _parse_config(raw: bytes, tool: OperationalToolObservation) -> AuditConfig:
    """Parse the exact frozen audit shape. Unknown or missing keys fail."""
    obj = _parse_config_object(raw, tool)

    def fail(message: str, field: str) -> AuditAbort:
        """Build the configuration failure associated with one invalid field."""
        return _abort(
            tool,
            OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            {"message": message},
            field=field,
        )

    _validate_scalar_fields(obj, fail)
    control_dt = _positive_decimal(obj["control_dt"], "control_dt", fail)
    ordered = _parse_candidate_timesteps(obj["candidate_step_dts"], fail)
    return AuditConfig(
        model_root=cast(str, obj["model_root"]),
        entrypoint=cast(str, obj["entrypoint"]),
        initial_state=cast(str, obj["initial_state"]),
        actions=cast(str, obj["actions"]),
        control_dt=control_dt,
        repeats=cast(int, obj["repeats"]),
        joint_tolerances=cast("Mapping[str, CanonicalValue]", obj["joint_tolerances"]),
        candidate_step_dts=ordered,
        workload_kind=cast(str, obj["workload_kind"]),
        workload_label=cast(str, obj["workload_label"]),
        output_dir=cast(str, obj["output_dir"]),
    )


def _parse_config_object(raw: bytes, tool: OperationalToolObservation) -> dict[str, CanonicalValue]:
    """Decode the configuration and require its exact frozen key set."""
    try:
        obj = bounded_strict_json_loads(raw, CONFIG_JSON_LIMITS)
    except (UnicodeDecodeError, JsonAdmissionError, ValueError, TypeError) as exc:
        raise _abort(
            tool,
            OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            {"exception_type": type(exc).__name__, "message": str(exc)},
            field="timestep_audit_config",
        ) from exc
    if type(obj) is not dict:
        raise _abort(
            tool,
            OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            {"message": "audit configuration must be a JSON object"},
            field="timestep_audit_config",
        )
    present = set(obj)
    if present != _CONFIG_KEYS:
        # sorted() yields list[str]; list is invariant, so the canonical slot needs a cast.
        missing = cast("list[CanonicalValue]", sorted(_CONFIG_KEYS - present))
        unknown = cast("list[CanonicalValue]", sorted(present - _CONFIG_KEYS))
        raise _abort(
            tool,
            OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            {
                "message": "audit configuration key set does not match the frozen shape",
                "missing": missing,
                "unknown": unknown,
            },
            field="timestep_audit_config",
        )
    return obj


def _compiled_timestep(value: float, tool: OperationalToolObservation) -> ExactRational:
    """Resolve the compiled reference timestep to the accepted decimal-token grammar.

    The shortest round-trip decimal is used, so a model compiled at 0.001 records exactly
    `0.001` rather than the exact dyadic expansion of the stored binary64. This is the same
    token the accepted comparison contract expects as `declared_step_dt`.
    """
    from decimal import Decimal

    if not (value > 0.0) or value != value or value in (float("inf"),):
        raise _abort(
            tool,
            OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            {"message": "compiled reference timestep is not a positive finite value"},
            field="reference_step_dt",
        )
    token = format(Decimal(repr(value)), "f")
    try:
        return ExactRational.from_decimal_token(token)
    except (TypeError, ValueError) as exc:
        raise _abort(
            tool,
            OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            {
                "message": "compiled reference timestep has no accepted decimal token",
                "token": token,
            },
            field="reference_step_dt",
        ) from exc


def _rational_sort_key(value: ExactRational) -> Fraction:
    """Ascending exact-rational order without floating point.

    A (numerator, denominator) tuple is NOT an ordering: 1/25 would sort before 3/100.
    """
    return Fraction(value.numerator, value.denominator)


def _rational_lt(left: ExactRational, right: ExactRational) -> bool:
    """Compare two exact rationals by cross multiplication."""
    return left.numerator * right.denominator < right.numerator * left.denominator


def _steps_per_control(control_dt: ExactRational, step_dt: ExactRational) -> int | None:
    """Exact integral steps per control interval, or None when not an exact divisor."""
    numerator = control_dt.numerator * step_dt.denominator
    denominator = control_dt.denominator * step_dt.numerator
    if denominator == 0 or numerator % denominator != 0:
        return None
    steps = numerator // denominator
    return steps if steps >= 1 else None


def _rational_primitive(value: ExactRational) -> dict[str, CanonicalValue]:
    """Emit numerator and denominator for audit operational evidence."""
    return {"numerator": value.numerator, "denominator": value.denominator}


def _reference_timestep(
    source_root: Path, entrypoint: str, tool: OperationalToolObservation
) -> ExactRational:
    """Compile the exact admitted entrypoint and read its reference timestep.

    The user's own bytes are compiled directly. Nothing is serialized, rewritten, or copied:
    an MJCF round trip re-formats float literals and moves compiled inertial and geometric
    values, which would make the audit report timestep fidelity plus XML canonicalization.
    """
    # Local import keeps module import cheap and mirrors the CLI boundary.
    import mujoco  # type: ignore[import-untyped]

    model = mujoco.MjModel.from_xml_path(str(source_root / entrypoint))
    return _compiled_timestep(float(model.opt.timestep), tool)


def _tree_digest(root: Path) -> str:
    """Deterministic digest over a model tree, used for immutability evidence."""
    members: list[CanonicalValue] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        members.append(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return hashlib.sha256(canonical_json_bytes(cast(CanonicalValue, members))).hexdigest()


def _admit_model_closure(
    source_root: Path,
    config: AuditConfig,
    tool: OperationalToolObservation,
) -> None:
    """Admit the declared model root and entrypoint through the accepted model admission measurement.

    An model admission refusal is an audit-level operational failure here: the audit, not the comparison
    engine, was the caller. Its reason, role, and evidence are preserved verbatim.
    """
    try:
        measure_model_closure(source_root, config.entrypoint, "comparison")
    except ModelAdmissionRefusal as exc:
        raise AuditAbort(
            operational_error(
                tool=tool,
                code=exc.reason,
                role=cast(ReasonRole, exc.role),
                evidence=cast("Mapping[str, CanonicalValue]", exc.evidence),
                operation=OPERATION,
            )
        ) from exc


def _require_output_outside_model_root(
    base: Path,
    config: AuditConfig,
    source_root: Path,
    tool: OperationalToolObservation,
) -> None:
    """Refuse an output directory equal to or below the model root before it is created."""
    raw_output = base / config.output_dir
    try:
        resolved_parent = raw_output.parent.resolve(strict=True)
    except OSError:
        # The accepted output contract already requires a real existing parent and emits its own
        # refusal for a missing one. Do not invent a second reason for the same condition.
        return
    resolved_output = resolved_parent / raw_output.name
    if resolved_output == source_root or source_root in resolved_output.parents:
        raise _abort(
            tool,
            OperationalReasonCode.OUTPUT_PATH_INVALID,
            {"issue": "output_inside_model_root"},
            field="output_dir",
        )
