"""Strict operational-failure construction for the installed commands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from ..errors import ReasonRole
from ..json_values import CanonicalValue, FrozenCanonicalValue, thaw_canonical
from ..operational import (
    InputDigest,
    OperationalFailure,
    OperationalReason,
    OperationalReasonCode,
    OperationalToolObservation,
)
from ..schemas import EnvironmentIdentity


class ComparisonOperationError(RuntimeError):
    """One completed strict operational failure ready for CLI serialization."""

    def __init__(self, failure: OperationalFailure) -> None:
        """Wrap one completed operational failure for controlled comparison unwinding."""
        self.failure = failure
        super().__init__(failure.reason.code.value)


def _thawed_primitive(value: object) -> CanonicalValue:
    """Recursively return a fresh mutable canonical primitive.

    `refuse()` deep-freezes evidence, so a nested mapping arrives as `MappingProxyType`
    and a nested sequence as `tuple`. A shallow copy left those frozen values in place
    and the canonical serializer, which is an exact-type hash boundary, rejected them.
    Flat scalar evidence is unaffected: it thaws to an equal primitive, so existing
    failure bytes and self-hashes are preserved.
    """
    if value is None or type(value) in {bool, int, str}:
        return cast(CanonicalValue, value)
    if isinstance(value, (tuple, list)):
        return [_thawed_primitive(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _thawed_primitive(item) for key, item in value.items()}
    return thaw_canonical(cast("FrozenCanonicalValue", value))


def operational_error(
    *,
    tool: OperationalToolObservation,
    code: OperationalReasonCode,
    role: ReasonRole,
    evidence: Mapping[str, CanonicalValue],
    available_inputs: Sequence[InputDigest] = (),
    environment: EnvironmentIdentity | None = None,
    field: str | None = None,
    object_name: str | None = None,
    operation: str = "compare",
) -> ComparisonOperationError:
    """Construct and self-hash one registry-bound operational failure."""
    primitive = cast(
        "dict[str, CanonicalValue]",
        {str(key): _thawed_primitive(item) for key, item in evidence.items()},
    )
    failure = OperationalFailure(
        schema="metrifid.operational_failure",
        schema_version=1,
        failure_rule_schema="metrifid.operational_failure_rules",
        failure_rule_schema_version=1,
        tool=tool,
        operation=operation,
        stage=code.stage,
        reason=OperationalReason(
            code=code,
            role=role,
            field=field,
            object_name=object_name,
            evidence=primitive,
        ),
        available_inputs=tuple(available_inputs),
        environment=environment,
        exit_code=code.exit_code,
        failure_sha256=None,
    ).finalized()
    return ComparisonOperationError(failure)


__all__ = ["ComparisonOperationError", "operational_error"]
