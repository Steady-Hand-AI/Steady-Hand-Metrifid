"""The frozen self-hashed compiled-equivalence receipt.

There is exactly one receipt schema. The artifact claim it carries is unconditional and
workload-free: it is a statement about bytes. The behavior implication is carried separately,
is explicitly conditional, prints every premise, and is excluded from the decision hash so it
can never be mistaken for part of the decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .._json_admission import (
    RECEIPT_JSON_LIMITS,
    JsonAdmissionError,
    bounded_strict_json_loads,
)
from ..json_values import (
    CanonicalValue,
    canonical_sha256,
    compute_self_hash,
    require_sha256,
    validate_self_hash,
)
from ..operational import OperationalToolObservation, _require_exact_object_fields
from ..schemas import ModelClosureIdentity
from ._artifact import COMPLETE_MJB_METHOD, CompiledArtifactIdentity
from ._bytes import ByteComparison
from ._field_schema import (
    _CHANGED_FIELD_MEMBERS as _CHANGED_FIELD_MEMBERS,
)
from ._field_schema import (
    _FIELD_REPORT_MEMBERS as _FIELD_REPORT_MEMBERS,
)
from ._field_schema import (
    _OMISSION_REASONS as _OMISSION_REASONS,
)
from ._field_schema import (
    _WITNESS_MEMBERS as _WITNESS_MEMBERS,
)
from ._receipt_contract import (
    _DECISION_MEMBERS,
    _HASH_FIELD,
    _LIMITATION_STATEMENTS,
    _ROLE_MEMBERS,
    _ROOT_MEMBERS,
)
from ._receipt_contract import (
    ARTIFACT_CLAIM_EXCLUSIONS as ARTIFACT_CLAIM_EXCLUSIONS,
)
from ._receipt_contract import (
    BEHAVIOR_IMPLICATION as BEHAVIOR_IMPLICATION,
)
from ._receipt_contract import (
    BEHAVIOR_IMPLICATION_PREMISES as BEHAVIOR_IMPLICATION_PREMISES,
)
from ._receipt_contract import (
    CERTIFIED_ARTIFACT_CLAIM as CERTIFIED_ARTIFACT_CLAIM,
)
from ._receipt_contract import (
    NOT_CERTIFIED_ARTIFACT_CLAIM as NOT_CERTIFIED_ARTIFACT_CLAIM,
)
from ._receipt_contract import (
    NOT_CERTIFIED_GUIDANCE as NOT_CERTIFIED_GUIDANCE,
)
from ._receipt_contract import (
    RECEIPT_SCHEMA as RECEIPT_SCHEMA,
)
from ._receipt_contract import (
    RECEIPT_SCHEMA_VERSION as RECEIPT_SCHEMA_VERSION,
)
from ._receipt_contract import (
    REQUIRED_LIMITATIONS as REQUIRED_LIMITATIONS,
)
from ._receipt_validation import (
    _artifact_claim,
    _behavior_implication,
    _validate_claims,
    _validate_field_report,
)
from ._receipt_validation import (
    _require_producer_value as _require_producer_value,
)
from ._runtime_schema import CertifyRuntimeIdentity
from ._status import CertifyStatus, certify_exit_code


@dataclass(frozen=True, slots=True)
class RoleCertification:
    """One role's measured source closure and complete compiled artifact."""

    role: str
    source_closure: ModelClosureIdentity
    artifact: CompiledArtifactIdentity

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit one role's closure and compiled-artifact identity evidence."""
        closure = self.source_closure.to_primitive()
        return {
            "role": self.role,
            "source_closure": closure,
            "source_closure_sha256": self.source_closure.sha256(),
            "source_closure_total_bytes": sum(
                member.size_bytes for member in self.source_closure.members
            ),
            "compiled_artifact": self.artifact.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class CertifyResult:
    """One completed certification and the two files it published."""

    status: CertifyStatus
    receipt: dict[str, CanonicalValue]
    certification_json: Path
    certification_markdown: Path

    @property
    def receipt_sha256(self) -> str:
        """Return the required SHA-256 identity from this completed receipt."""
        digest = self.receipt[_HASH_FIELD]
        if not isinstance(digest, str):
            raise TypeError("receipt_sha256 must be a string")
        return digest


def build_receipt(
    *,
    status: CertifyStatus,
    tool: Mapping[str, CanonicalValue],
    runtime: CertifyRuntimeIdentity,
    baseline: RoleCertification,
    candidate: RoleCertification,
    comparison: ByteComparison,
    field_report: dict[str, CanonicalValue] | None,
) -> dict[str, CanonicalValue]:
    """Assemble, decision-hash and self-hash one complete receipt."""
    receipt: dict[str, CanonicalValue] = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": status.value,
        "completed_exit_code": certify_exit_code(status),
        "tool": dict(tool),
        "runtime_identity": runtime.to_primitive(),
        "baseline": baseline.to_primitive(),
        "candidate": candidate.to_primitive(),
        "byte_comparison": comparison.to_primitive(),
        "field_report": field_report,
        "artifact_claim": _artifact_claim(status),
        "behavior_implication": _behavior_implication(status),
        "limitations": [
            {"code": code, "statement": _LIMITATION_STATEMENTS[code]}
            for code in REQUIRED_LIMITATIONS
        ],
    }
    receipt["decision_sha256"] = canonical_sha256(
        {name: receipt[name] for name in _DECISION_MEMBERS}
    )
    receipt[_HASH_FIELD] = None
    receipt[_HASH_FIELD] = compute_self_hash(receipt, _HASH_FIELD)
    validate_receipt(receipt)
    return receipt


def load_and_validate_certification_receipt(
    data: bytes | str,
) -> dict[str, CanonicalValue]:
    """Admit and revalidate one serialized certification receipt from untrusted bytes.

    This is the entry point for a reader holding a receipt *file* or bytes off the wire. It applies
    the bounded strict JSON admission limits for receipts before any structure exists, so duplicate
    member names, raw float tokens, ``NaN``/``Infinity``, malformed UTF-8, and oversized or
    excessively nested documents are refused before the semantic validator ever runs. Callers that
    already hold a strictly parsed in-memory mapping should keep using :func:`validate_receipt`.

    Args:
        data: The serialized receipt as UTF-8 bytes or text.

    Returns:
        The validated receipt as a mutable canonical mapping.

    Raises:
        JsonAdmissionError: The bytes failed bounded strict admission or the root is not an object.
        ValueError: The document parsed but failed semantic receipt validation.
        TypeError: The document parsed but a member had the wrong type.
    """
    parsed = bounded_strict_json_loads(data, RECEIPT_JSON_LIMITS)
    if type(parsed) is not dict:
        raise JsonAdmissionError("certification receipt must be a JSON object")
    validate_receipt(parsed)
    return parsed


def validate_receipt(receipt: Mapping[str, CanonicalValue]) -> None:
    """Re-verify a receipt exactly as an independent reader would.

    This establishes internal semantic consistency of an unsigned receipt. It is not signing:
    a party who rewrites a fact and recomputes both hashes still has to keep every nested
    invariant below true, and these checks are what make that hard rather than trivial.
    """
    obj = _require_exact_object_fields(receipt, set(_ROOT_MEMBERS), "CompiledEquivalenceReceipt")
    if obj["schema"] != RECEIPT_SCHEMA:
        raise ValueError("receipt schema is outside the frozen registry")
    if type(obj["schema_version"]) is not int or obj["schema_version"] != RECEIPT_SCHEMA_VERSION:
        raise ValueError("receipt schema_version is outside the frozen registry")
    status = CertifyStatus(str(obj["status"]))
    # bool is a subclass of int and False == 0, so an exact type check is required here.
    if type(obj["completed_exit_code"]) is not int:
        raise TypeError("completed_exit_code must be a JSON integer, not a boolean")
    if obj["completed_exit_code"] != certify_exit_code(status):
        raise ValueError("completed_exit_code does not match the Certify exit mapping")
    certified = status is CertifyStatus.CERTIFIED_COMPILED_EQUIVALENCE

    runtime = CertifyRuntimeIdentity.from_primitive(obj["runtime_identity"])
    _validate_tool(obj["tool"], runtime)
    baseline = _validated_role(obj["baseline"], "baseline", runtime)
    candidate = _validated_role(obj["candidate"], "candidate", runtime)
    comparison = _validate_comparison(obj["byte_comparison"], baseline, candidate, certified)
    _validate_claims(obj, certified)
    _validate_field_report(obj["field_report"], certified, comparison)

    decision = {name: receipt[name] for name in _DECISION_MEMBERS}
    if obj["decision_sha256"] != canonical_sha256(cast("CanonicalValue", decision)):
        raise ValueError("decision_sha256 does not match the decision-bearing members")
    validate_self_hash(dict(receipt), _HASH_FIELD)


def _validate_tool(value: object, runtime: CertifyRuntimeIdentity) -> None:
    """Require the frozen tool member set and bind the observation to the runtime identity.

    A completed receipt is only meaningful if the tool that produced it was a verified installed
    distribution, and if that distribution is the same one the runtime identity records.
    """
    obj = _require_exact_object_fields(
        value, {"name", "version", "execution_identity_state", "distribution_sha256"}, "tool"
    )
    if obj["name"] != "metrifid":
        raise ValueError("the receipt tool must name metrifid")
    observation = OperationalToolObservation.from_primitive(
        {name: obj[name] for name in ("version", "execution_identity_state", "distribution_sha256")}
    )
    if observation.execution_identity_state != "VERIFIED_INSTALLED_DISTRIBUTION":
        raise ValueError("a completed receipt requires a verified installed distribution")
    if observation.distribution_sha256 is None:
        raise ValueError("a completed receipt requires a bound distribution digest")
    require_sha256(observation.distribution_sha256, "tool.distribution_sha256")
    if observation.version != runtime.metrifid_version:
        raise ValueError("tool version does not match the runtime identity metrifid version")
    if observation.distribution_sha256 != runtime.metrifid_distribution_sha256:
        raise ValueError("tool distribution digest does not match the runtime identity digest")


def _validated_role(
    value: object, role: str, runtime: CertifyRuntimeIdentity
) -> CompiledArtifactIdentity:
    """Validate one role's closure and artifact, and bind the artifact to the runtime."""
    obj = _require_exact_object_fields(value, set(_ROLE_MEMBERS), f"RoleCertification[{role}]")
    if obj["role"] != role:
        raise ValueError(f"the {role} member must name the {role} role")
    closure = ModelClosureIdentity.from_primitive(obj["source_closure"])
    if obj["source_closure_sha256"] != closure.sha256():
        raise ValueError(f"{role} source_closure_sha256 does not match the closure")
    if obj["source_closure_total_bytes"] != sum(m.size_bytes for m in closure.members):
        raise ValueError(f"{role} source_closure_total_bytes does not match the closure members")
    artifact = CompiledArtifactIdentity.from_primitive(obj["compiled_artifact"])
    if artifact.method != COMPLETE_MJB_METHOD:
        raise ValueError(f"{role} artifact does not use the frozen complete-MJB method")
    if artifact.runtime_identity_sha256 != runtime.runtime_identity_sha256:
        raise ValueError(f"{role} artifact is not bound to the receipt runtime identity")
    if artifact.header_words != runtime.mjb_header_words:
        raise ValueError(f"{role} artifact header words differ from the runtime header words")
    if artifact.mujoco_version_integer != runtime.mujoco_version_integer:
        raise ValueError(f"{role} artifact version word differs from the runtime version")
    return artifact


def _validate_comparison(
    value: object,
    baseline: CompiledArtifactIdentity,
    candidate: CompiledArtifactIdentity,
    certified: bool,
) -> ByteComparison:
    """Require the comparison to describe exactly these two artifacts."""
    comparison = ByteComparison.from_primitive(value)
    if comparison.baseline_mjb_size_bytes != baseline.mjb_size_bytes:
        raise ValueError("comparison baseline size does not match the baseline artifact")
    if comparison.candidate_mjb_size_bytes != candidate.mjb_size_bytes:
        raise ValueError("comparison candidate size does not match the candidate artifact")
    identical = (
        baseline.mjb_size_bytes == candidate.mjb_size_bytes
        and baseline.mjb_sha256 == candidate.mjb_sha256
    )
    if comparison.equal is not identical:
        raise ValueError("byte equality must follow from the two complete artifact identities")
    if comparison.equal is not certified:
        raise ValueError("status does not follow from the complete artifact byte comparison")
    return comparison
