"""Build the canonical, self-hashed Native Runtime Review receipt.

Only stable decision fields enter this document.  In particular, platform-sensitive interval
endpoints, magnitudes, and ratios remain outside the canonical receipt; the independently
recomputed witness inputs and their classifications are the portable decision boundary.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Mapping, Sequence
from typing import Final, Protocol, cast

from ..json_values import (
    CanonicalValue,
    canonical_sha256,
    compute_self_hash,
    installed_distribution_identity,
    require_sha256,
)
from ..version import __version__
from ._config import (
    AdmittedRuntimeReviewConfiguration,
    AdmittedRuntimeReviewConfigurationV2,
    RuntimeProfileConfigV2,
)
from ._owned_output import OwnedEvidenceCell, OwnedProfileIdentity
from ._status import RuntimeReviewReasonCode, RuntimeReviewStatus

RUNTIME_REVIEW_RECEIPT_SCHEMA: Final[str] = "metrifid.runtime_review_receipt"
RUNTIME_REVIEW_RECEIPT_SCHEMA_VERSION: Final[int] = 1
RUNTIME_REVIEW_RECEIPT_SCHEMA_VERSION_V2: Final[int] = 2
RUNTIME_REVIEW_METHOD: Final[str] = "CONDITIONAL_TAIL_ENVELOPE"
RUNTIME_REVIEW_CLAIM_SCOPE: Final[str] = (
    "EXACT_NATIVE_PROFILES_EXACT_SOURCE_CLOSURE_EXACT_WORKLOAD_FULL_HORIZON"
)
RUNTIME_REVIEW_RECEIPT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        "reason_code",
        "method",
        "claim_scope",
        "configuration",
        "profiles",
        "subject",
        "workload",
        "campaign_shape",
        "required_horizon",
        "admitted_prefix",
        "first_failing_gate",
        "first_decisive_witness",
        "worst_witness",
        "witness_counts",
        "witnesses",
        "evidence_cells",
        "limitations",
        "tool",
        "receipt_sha256",
    }
)
RUNTIME_REVIEW_LIMITATIONS: Final[tuple[str, ...]] = (
    "The decision applies only to the exact baseline and candidate native profiles recorded in "
    "this receipt.",
    "The decision applies only to the exact source closure, compiled evidence, workload, channels, "
    "and user-declared tolerances recorded in this receipt.",
    "The decision is not a claim of universal MuJoCo equivalence.",
    "The decision is not a hardware-safety, policy-safety, task-success, or real-world-transfer "
    "claim.",
    "The decision makes no claim about another simulation backend.",
    "The receipt self-hash detects corruption; it is not an authenticity proof or digital "
    "signature.",
)
_PROFILE_BINDING_KEYS_V2: Final[frozenset[str]] = frozenset(
    {
        "profile_role",
        "package_version",
        "native_version",
        "native_version_integer",
        "profile_identity_sha256",
        "runtime_identity_sha256",
        "mujoco_distribution",
        "loaded_native_library",
        "numpy",
        "sentinel",
        "support_tier",
    }
)
_PROFILE_BINDING_KEYS_WITH_FILE_V2: Final[frozenset[str]] = _PROFILE_BINDING_KEYS_V2 | {
    "identity_file"
}
_SUPPORT_TIER_V2: Final[str] = "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE"


class RuntimeReviewEvidenceForReceipt(Protocol):
    """Stable campaign facts consumed by the canonical receipt builder."""

    @property
    def profiles(self) -> object:
        """Return exact admitted profile identities."""
        ...

    @property
    def subject(self) -> object:
        """Return the exact admitted source-closure identities."""
        ...

    @property
    def workload(self) -> object:
        """Return the exact admitted workload identities."""
        ...


class RuntimeReviewDecisionForReceipt(Protocol):
    """Stable completed-decision facts consumed by the receipt builder."""

    @property
    def status(self) -> RuntimeReviewStatus:
        """Return the completed public status."""
        ...

    @property
    def reason_code(self) -> RuntimeReviewReasonCode | None:
        """Return the optional insufficient-evidence reason."""
        ...

    @property
    def admitted_prefix(self) -> str:
        """Return the qualified prefix as an exact decimal token."""
        ...

    @property
    def first_failing_gate(self) -> object:
        """Return the optional stable gate primitive."""
        ...

    @property
    def first_decisive_witness(self) -> object:
        """Return the optional first decisive stable witness."""
        ...

    @property
    def worst_witness(self) -> object:
        """Return the optional stable worst-witness identity."""
        ...

    @property
    def witness_counts(self) -> object:
        """Return complete classification counts."""
        ...

    @property
    def witnesses(self) -> Sequence[object]:
        """Return every stable witness in canonical order."""
        ...


def runtime_review_tool_identity() -> dict[str, CanonicalValue]:
    """Observe the installed product and evaluator identities recorded by a new receipt."""
    distribution = installed_distribution_identity()
    evaluator: dict[str, CanonicalValue] = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag or "",
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
    }
    return {
        "metrifid_version": __version__,
        "distribution_identity": distribution,
        "distribution_identity_sha256": canonical_sha256(distribution),
        "evaluator": evaluator,
    }


def build_runtime_review_receipt(
    *,
    configuration: AdmittedRuntimeReviewConfiguration,
    evidence: RuntimeReviewEvidenceForReceipt,
    decision: RuntimeReviewDecisionForReceipt,
    evidence_cells: Sequence[OwnedEvidenceCell],
    tool: Mapping[str, CanonicalValue] | None = None,
) -> dict[str, CanonicalValue]:
    """Build and canonical-self-hash one completed full-horizon runtime-review receipt."""
    if not isinstance(configuration, AdmittedRuntimeReviewConfiguration):
        raise TypeError("configuration must be an AdmittedRuntimeReviewConfiguration")
    status = _status_value(decision.status)
    reason = _reason_value(decision.reason_code)
    _validate_status_reason(status, reason)
    cells = tuple(evidence_cells)
    if len(cells) != 12:
        raise ValueError("a runtime-review receipt requires exactly twelve owned evidence cells")
    witnesses = [_object_primitive(value, "witness") for value in decision.witnesses]
    receipt: dict[str, CanonicalValue] = {
        "schema": RUNTIME_REVIEW_RECEIPT_SCHEMA,
        "schema_version": RUNTIME_REVIEW_RECEIPT_SCHEMA_VERSION,
        "status": status,
        "reason_code": reason,
        "method": RUNTIME_REVIEW_METHOD,
        "claim_scope": RUNTIME_REVIEW_CLAIM_SCOPE,
        "configuration": {
            "locator": "admitted_runtime_review_config.json",
            "raw_sha256": configuration.raw_sha256,
            "semantic_sha256": configuration.semantic_sha256,
        },
        "profiles": _object_primitive(evidence.profiles, "profiles"),
        "subject": _object_primitive(evidence.subject, "subject"),
        "workload": _object_primitive(evidence.workload, "workload"),
        "campaign_shape": {
            "step_dts": list(configuration.config.step_dts),
            "repeat_ids": list(configuration.config.repeat_ids),
            "cell_count": len(cells),
        },
        "required_horizon": configuration.config.required_horizon,
        "admitted_prefix": decision.admitted_prefix,
        "first_failing_gate": _optional_object_primitive(
            decision.first_failing_gate, "first_failing_gate"
        ),
        "first_decisive_witness": _optional_object_primitive(
            decision.first_decisive_witness, "first_decisive_witness"
        ),
        "worst_witness": _optional_object_primitive(decision.worst_witness, "worst_witness"),
        "witness_counts": _object_primitive(decision.witness_counts, "witness_counts"),
        "witnesses": cast(CanonicalValue, witnesses),
        "evidence_cells": [cell.to_primitive() for cell in cells],
        "limitations": list(RUNTIME_REVIEW_LIMITATIONS),
        "tool": dict(tool) if tool is not None else runtime_review_tool_identity(),
        "receipt_sha256": None,
    }
    if set(receipt) != RUNTIME_REVIEW_RECEIPT_KEYS:  # pragma: no cover - construction invariant
        raise AssertionError("runtime-review receipt root fields drifted")
    receipt["receipt_sha256"] = compute_self_hash(receipt, "receipt_sha256")
    return receipt


def build_runtime_review_receipt_v2(
    *,
    configuration: AdmittedRuntimeReviewConfigurationV2,
    evidence: RuntimeReviewEvidenceForReceipt,
    decision: RuntimeReviewDecisionForReceipt,
    evidence_cells: Sequence[OwnedEvidenceCell],
    profile_identities: Sequence[OwnedProfileIdentity],
    tool: Mapping[str, CanonicalValue] | None = None,
) -> dict[str, CanonicalValue]:
    """Build one role-based receipt with portable profile and sentinel bindings."""
    if not isinstance(configuration, AdmittedRuntimeReviewConfigurationV2):
        raise TypeError("configuration must be an AdmittedRuntimeReviewConfigurationV2")
    status = _status_value(decision.status)
    reason = _reason_value(decision.reason_code)
    _validate_status_reason(status, reason)
    cells = tuple(evidence_cells)
    if len(cells) != 12:
        raise ValueError("a runtime-review receipt requires exactly twelve owned evidence cells")
    profiles = _profiles_primitive_v2(configuration, evidence.profiles, profile_identities)
    witnesses = [_object_primitive(value, "witness") for value in decision.witnesses]
    receipt: dict[str, CanonicalValue] = {
        "schema": RUNTIME_REVIEW_RECEIPT_SCHEMA,
        "schema_version": RUNTIME_REVIEW_RECEIPT_SCHEMA_VERSION_V2,
        "status": status,
        "reason_code": reason,
        "method": RUNTIME_REVIEW_METHOD,
        "claim_scope": RUNTIME_REVIEW_CLAIM_SCOPE,
        "configuration": {
            "locator": "admitted_runtime_review_config.json",
            "raw_sha256": configuration.raw_sha256,
            "semantic_sha256": configuration.semantic_sha256,
        },
        "profiles": profiles,
        "subject": _object_primitive(evidence.subject, "subject"),
        "workload": _object_primitive(evidence.workload, "workload"),
        "campaign_shape": {
            "step_dts": list(configuration.config.step_dts),
            "repeat_ids": list(configuration.config.repeat_ids),
            "cell_count": len(cells),
        },
        "required_horizon": configuration.config.required_horizon,
        "admitted_prefix": decision.admitted_prefix,
        "first_failing_gate": _optional_object_primitive(
            decision.first_failing_gate, "first_failing_gate"
        ),
        "first_decisive_witness": _optional_object_primitive(
            decision.first_decisive_witness, "first_decisive_witness"
        ),
        "worst_witness": _optional_object_primitive(decision.worst_witness, "worst_witness"),
        "witness_counts": _object_primitive(decision.witness_counts, "witness_counts"),
        "witnesses": cast(CanonicalValue, witnesses),
        "evidence_cells": [cell.to_primitive() for cell in cells],
        "limitations": list(RUNTIME_REVIEW_LIMITATIONS),
        "tool": dict(tool) if tool is not None else runtime_review_tool_identity(),
        "receipt_sha256": None,
    }
    if set(receipt) != RUNTIME_REVIEW_RECEIPT_KEYS:  # pragma: no cover - construction invariant
        raise AssertionError("runtime-review receipt root fields drifted")
    receipt["receipt_sha256"] = compute_self_hash(receipt, "receipt_sha256")
    return receipt


def _profiles_primitive_v2(
    configuration: AdmittedRuntimeReviewConfigurationV2,
    value: object,
    owned_identities: Sequence[OwnedProfileIdentity],
) -> dict[str, CanonicalValue]:
    """Bind role projections to config declarations and exact owned identity bytes."""
    profiles = _object_primitive(value, "profiles")
    if set(profiles) != {"baseline", "candidate", "common_environment"}:
        raise ValueError(
            "v2 profiles must contain baseline, candidate, and common_environment exactly"
        )
    owned = tuple(owned_identities)
    if len(owned) != 2 or tuple(item.profile_role for item in owned) != (
        "baseline",
        "candidate",
    ):
        raise ValueError("profile_identities must contain baseline then candidate exactly")
    result: dict[str, CanonicalValue] = {}
    declarations = (configuration.config.baseline_profile, configuration.config.candidate_profile)
    for role, declaration, identity in zip(
        ("baseline", "candidate"), declarations, owned, strict=True
    ):
        profile = _object_primitive(profiles[role], f"profiles.{role}")
        if set(profile) != _PROFILE_BINDING_KEYS_V2:
            raise ValueError(f"profiles.{role} has an invalid role-binding field set")
        _validate_profile_binding_v2(profile, declaration, role)
        if identity.locator.as_posix() != declaration.identity_file:
            raise ValueError(f"profiles.{role} owned identity locator differs from configuration")
        if identity.sha256 != configuration.profile_identity_file_hash(role):
            raise ValueError(
                f"profiles.{role} owned identity bytes differ from configuration admission"
            )
        profile["identity_file"] = {
            "locator": identity.locator.as_posix(),
            "raw_sha256": identity.sha256,
            "size_bytes": identity.size_bytes,
        }
        if set(profile) != _PROFILE_BINDING_KEYS_WITH_FILE_V2:  # pragma: no cover
            raise AssertionError("v2 profile binding construction drifted")
        result[role] = profile
    result["common_environment"] = _object_primitive(
        profiles["common_environment"], "profiles.common_environment"
    )
    return result


def _validate_profile_binding_v2(
    profile: Mapping[str, CanonicalValue], declaration: RuntimeProfileConfigV2, role: str
) -> None:
    """Require one evidence profile to equal its exact role-based declaration."""
    expected = {
        "profile_role": declaration.profile_role,
        "package_version": declaration.package_version,
        "native_version": declaration.native_version,
        "native_version_integer": declaration.native_version_integer,
        "profile_identity_sha256": declaration.profile_identity_sha256,
    }
    if any(profile[field] != member for field, member in expected.items()):
        raise ValueError(f"profiles.{role} differs from its configuration declaration")
    require_sha256(profile["runtime_identity_sha256"], f"profiles.{role}.runtime_identity_sha256")
    for field in ("mujoco_distribution", "loaded_native_library", "numpy"):
        _object_primitive(profile[field], f"profiles.{role}.{field}")
    sentinel = _object_primitive(profile["sentinel"], f"profiles.{role}.sentinel")
    if set(sentinel) != {"status", "sentinel_identity_sha256"} or sentinel["status"] != "PASS":
        raise ValueError(f"profiles.{role}.sentinel must bind one PASS result")
    require_sha256(
        sentinel["sentinel_identity_sha256"],
        f"profiles.{role}.sentinel.sentinel_identity_sha256",
    )
    if profile["support_tier"] != _SUPPORT_TIER_V2:
        raise ValueError(f"profiles.{role}.support_tier must record current capability admission")


def _status_value(value: object) -> str:
    """Return one completed status token from its public enum."""
    if not isinstance(value, RuntimeReviewStatus):
        raise TypeError("decision.status must be a RuntimeReviewStatus")
    return value.value


def _reason_value(value: object) -> str | None:
    """Return one optional insufficient-evidence reason token."""
    if value is None:
        return None
    if not isinstance(value, RuntimeReviewReasonCode):
        raise TypeError("decision.reason_code must be a RuntimeReviewReasonCode or None")
    return value.value


def _validate_status_reason(status: str, reason: str | None) -> None:
    """Require a reason exactly when the completed status is insufficient evidence."""
    insufficient = RuntimeReviewStatus.INSUFFICIENT_EVIDENCE.value
    if (status == insufficient) != (reason is not None):
        raise ValueError("reason_code must be present exactly for INSUFFICIENT_EVIDENCE")


def _object_primitive(value: object, field: str) -> dict[str, CanonicalValue]:
    """Convert a mapping or focused schema object into one canonical object."""
    primitive: object
    if isinstance(value, Mapping):
        primitive = dict(value)
    else:
        conversion = getattr(value, "to_primitive", None)
        if conversion is None:
            conversion = getattr(value, "primitive", None)
        if not callable(conversion):
            raise TypeError(f"{field} must be a mapping or expose a primitive conversion")
        primitive = conversion()
    if type(primitive) is not dict:
        raise TypeError(f"{field} primitive must be an object")
    # Canonical hashing is also a complete recursive type check.  It rejects raw binary64 values,
    # which prevents platform-sensitive diagnostics from entering this decision-bearing receipt.
    canonical_sha256(cast(CanonicalValue, primitive))
    return cast(dict[str, CanonicalValue], primitive)


def _optional_object_primitive(value: object, field: str) -> dict[str, CanonicalValue] | None:
    """Convert one optional stable receipt object."""
    return None if value is None else _object_primitive(value, field)


__all__ = [
    "RUNTIME_REVIEW_CLAIM_SCOPE",
    "RUNTIME_REVIEW_LIMITATIONS",
    "RUNTIME_REVIEW_METHOD",
    "RUNTIME_REVIEW_RECEIPT_KEYS",
    "RUNTIME_REVIEW_RECEIPT_SCHEMA",
    "RUNTIME_REVIEW_RECEIPT_SCHEMA_VERSION",
    "RUNTIME_REVIEW_RECEIPT_SCHEMA_VERSION_V2",
    "RuntimeReviewDecisionForReceipt",
    "RuntimeReviewEvidenceForReceipt",
    "build_runtime_review_receipt",
    "build_runtime_review_receipt_v2",
    "runtime_review_tool_identity",
]
