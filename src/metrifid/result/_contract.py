"""The small ``metrifid.result`` document shared by ``show`` and ``diff``.

This is a presentation of one already-authoritative receipt, never a second receipt system.  The
canonical receipt bytes, their schemas and their validators remain the authority; every member
below either quotes what the producer retained or states honestly that it is not known.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from ..json_values import CanonicalValue
from ._findings import (
    certification_findings,
    certification_limitations,
    model_release_findings,
    model_release_limitations,
)

__all__ = [
    "CERTIFICATION_SCHEMA",
    "MODEL_RELEASE_SCHEMA",
    "NOT_ESTABLISHED",
    "OBSERVATION_NONE",
    "OBSERVATION_PRODUCED",
    "OBSERVATION_RECORDED",
    "POLICY_GENERATED_EMPTY",
    "POLICY_NOT_APPLICABLE",
    "POLICY_NOT_CREATED",
    "POLICY_NOT_RECORDED",
    "RESULT_SCHEMA",
    "RESULT_SCHEMA_VERSION",
    "build_document",
    "build_failure_document",
    "closure_identity",
    "compiled_comparison",
    "receipt_view",
]

RESULT_SCHEMA: Final = "metrifid.result"
RESULT_SCHEMA_VERSION: Final = 1
CERTIFICATION_SCHEMA: Final = "metrifid.compiled_equivalence_receipt"
MODEL_RELEASE_SCHEMA: Final = "metrifid.model_release_receipt"

OBSERVATION_PRODUCED: Final = "produced_now"
OBSERVATION_RECORDED: Final = "recorded"
OBSERVATION_NONE: Final = "none"

POLICY_GENERATED_EMPTY: Final = "generated_empty"
POLICY_NOT_RECORDED: Final = "not_recorded"
POLICY_NOT_APPLICABLE: Final = "not_applicable"
POLICY_NOT_CREATED: Final = "not_created"

IDENTICAL: Final = "identical"
DIFFERENT: Final = "different"
NOT_ESTABLISHED: Final = "not_established"


def _mapping(value: CanonicalValue) -> Mapping[str, CanonicalValue]:
    """Return one canonical object, or an empty mapping when absent."""
    return value if isinstance(value, dict) else {}


def compiled_comparison(
    certification: Mapping[str, CanonicalValue] | None,
) -> dict[str, CanonicalValue]:
    """State compiled identity from the byte comparison alone.

    The state never comes from a policy outcome and never from a successful read.  When no
    certification is available the state is not_established and every member stays null.

    Args:
        certification: The standalone certification, or the embedded certification_receipt.

    Returns:
        The compiled_comparison member of a result document.
    """
    comparison = _mapping(certification.get("byte_comparison")) if certification else {}
    equal = comparison.get("equal")
    if not isinstance(equal, bool):
        return {
            "state": NOT_ESTABLISHED,
            "equal": None,
            "first_differing_byte_offset": None,
            "differing_byte_count": None,
        }
    return {
        "state": IDENTICAL if equal else DIFFERENT,
        "equal": equal,
        "first_differing_byte_offset": comparison.get("first_differing_byte_offset"),
        "differing_byte_count": comparison.get("differing_byte_count"),
    }


def closure_identity(role: Mapping[str, CanonicalValue]) -> dict[str, CanonicalValue]:
    """Return one role's retained closure and compiled identity, without live paths.

    Args:
        role: The baseline or candidate block of a certification.

    Returns:
        The retained identity members, each null when the producer retained none.
    """
    closure = _mapping(role.get("source_closure"))
    artifact = _mapping(role.get("compiled_artifact"))
    return {
        "entrypoint": closure.get("entrypoint"),
        "member_count": closure.get("member_count"),
        # The measured inventory is retained whole and in the producer's own shape: a reader
        # asking what was measured must get the files, not only how many there were.
        "members": closure.get("members"),
        "source_closure_sha256": role.get("source_closure_sha256"),
        "compiled_sha256": artifact.get("mjb_sha256"),
        "compiled_size_bytes": artifact.get("mjb_size_bytes"),
    }


def _runtime(
    certification: Mapping[str, CanonicalValue] | None, reader_version: str
) -> dict[str, CanonicalValue]:
    """Separate the reader's own version from the producer runtime the receipt recorded."""
    identity = _mapping(certification.get("runtime_identity")) if certification else {}
    producer: CanonicalValue = None
    if identity:
        producer = {
            "metrifid": identity.get("metrifid_version"),
            "mujoco": identity.get("mujoco_version"),
            "python": identity.get("python_version"),
            "python_implementation": identity.get("python_implementation"),
            "platform_system": identity.get("platform_system"),
            "platform_machine": identity.get("platform_machine"),
            "runtime_identity_sha256": identity.get("runtime_identity_sha256"),
        }
    return {"reader": reader_version, "producer": producer}


def receipt_view(
    receipt: Mapping[str, CanonicalValue],
) -> tuple[
    Mapping[str, CanonicalValue] | None,
    list[dict[str, CanonicalValue]],
    list[dict[str, CanonicalValue]],
]:
    """Return the certification, findings and limitations for one admitted receipt.

    Args:
        receipt: One validated certification or model-release receipt.

    Returns:
        The certification to read compiled identity from, the source-shaped findings, and the
        categorized limitations.
    """
    if receipt.get("schema") == MODEL_RELEASE_SCHEMA:
        certification = _mapping(receipt.get("certification_receipt"))
        return certification, model_release_findings(receipt), model_release_limitations(receipt)
    return receipt, certification_findings(receipt), certification_limitations(receipt)


def _evaluation(
    receipt: Mapping[str, CanonicalValue], policy_origin: str
) -> dict[str, CanonicalValue]:
    """Report the recorded evaluation outcome, never replacing it with this view's own result."""
    policy = _mapping(receipt.get("policy"))
    return {
        "status": receipt.get("status"),
        "exit_code": receipt.get("completed_exit_code"),
        "policy_origin": policy_origin,
        "policy_raw_sha256": policy.get("raw_sha256"),
        "policy_rule_count": policy.get("rule_count"),
        "receipt_sha256": receipt.get("receipt_sha256"),
    }


def build_document(
    *,
    command: str,
    exit_code: int,
    observation: str,
    receipt: Mapping[str, CanonicalValue],
    policy_origin: str,
    reader_version: str,
    inputs: dict[str, CanonicalValue],
    artifacts: dict[str, CanonicalValue],
) -> dict[str, CanonicalValue]:
    """Assemble one complete result document over a validated receipt.

    Args:
        command: The invoked command name.
        exit_code: The exit code this command is returning.
        observation: Whether the receipt was produced now or read from disk.
        receipt: One validated certification or model-release receipt.
        policy_origin: How the evaluated policy came to exist, as far as is known.
        reader_version: The version of the code rendering this document.
        inputs: Each role's known live paths plus its retained identity.
        artifacts: Known output paths; null means unknown, never proven absent.

    Returns:
        The complete result document, ready for canonical encoding.
    """
    certification, findings, limitations = receipt_view(receipt)
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "command": {"name": command, "exit_code": exit_code},
        "observation": observation,
        "compiled_comparison": compiled_comparison(certification),
        "evaluation": _evaluation(receipt, policy_origin),
        "runtime": _runtime(certification, reader_version),
        "inputs": inputs,
        "findings": list(findings),
        "finding_count": len(findings),
        "limitations": list(limitations),
        "artifacts": artifacts,
        "truncated": False,
        "problems": [],
    }


def build_failure_document(
    *,
    command: str,
    exit_code: int,
    reason_code: str,
    message: str,
    reader_version: str,
    inputs: dict[str, CanonicalValue],
    artifacts: dict[str, CanonicalValue],
    policy_origin: str,
) -> dict[str, CanonicalValue]:
    """Assemble one result document for a command that returned no completed receipt.

    Every member of the contract stays present and truthful: nothing was observed, compiled
    identity was not established, and the original evaluation is unknown.

    Args:
        command: The invoked command name.
        exit_code: The actual failing exit code, taken from the core failure.
        reason_code: The bounded reason code the core reported.
        message: A readable diagnostic for that reason.
        reader_version: The version of the code rendering this document.
        inputs: Whatever role paths were already known, each member null otherwise.
        artifacts: Known output paths; unknown paths stay null rather than claiming absence.
        policy_origin: How far policy creation got, as far as is known.

    Returns:
        The complete result document for a failed command.
    """
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "command": {"name": command, "exit_code": exit_code},
        "observation": OBSERVATION_NONE,
        "compiled_comparison": compiled_comparison(None),
        "evaluation": {
            "status": None,
            "exit_code": None,
            "policy_origin": policy_origin,
            "policy_raw_sha256": None,
            "policy_rule_count": None,
            "receipt_sha256": None,
        },
        "runtime": {"reader": reader_version, "producer": None},
        "inputs": inputs,
        "findings": [],
        "finding_count": 0,
        "limitations": [],
        "artifacts": artifacts,
        "truncated": False,
        "problems": [{"code": reason_code, "message": message}],
    }
