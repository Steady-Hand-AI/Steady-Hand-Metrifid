"""Read one saved receipt and present it, without a runtime and without an output directory.

``show`` is inspection of a recorded subject.  It compiles nothing, allocates nothing, imports no
native library, and never replaces the outcome the producer recorded with its own successful read.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

from .._json_admission import (
    RECEIPT_JSON_LIMITS,
    JsonAdmissionError,
    bounded_strict_json_loads,
    read_bounded_regular_file,
)
from ..compare._failure import ComparisonOperationError, operational_error
from ..distribution import installed_distribution_sha256
from ..json_values import CanonicalValue
from ..operational import OperationalReasonCode, OperationalToolObservation
from ..version import __version__
from ._contract import (
    CERTIFICATION_SCHEMA,
    MODEL_RELEASE_SCHEMA,
    OBSERVATION_RECORDED,
    POLICY_NOT_APPLICABLE,
    POLICY_NOT_RECORDED,
    build_document,
    closure_identity,
)

__all__ = ["SHOW_OPERATION", "show_receipt"]

SHOW_OPERATION: Final = "review-model"
_ADMITTED_SCHEMAS: Final = (CERTIFICATION_SCHEMA, MODEL_RELEASE_SCHEMA)


def _tool() -> OperationalToolObservation:
    """Bind the running distribution to any failure this command reports."""
    return OperationalToolObservation(
        __version__, "VERIFIED_INSTALLED_DISTRIBUTION", installed_distribution_sha256()
    )


def _failure(
    code: OperationalReasonCode, field: str, **evidence: CanonicalValue
) -> ComparisonOperationError:
    """Build one bounded reader failure over the existing operational ABI."""
    return operational_error(
        tool=_tool(),
        code=code,
        role="comparison",
        field=field,
        evidence=evidence,
        operation=SHOW_OPERATION,
    )


def show_receipt(receipt_path: str) -> dict[str, CanonicalValue]:
    """Admit one saved receipt and return its complete result document.

    The document is returned whatever outcome the receipt recorded: a receipt that records a
    negative evaluation is still a valid read.

    Args:
        receipt_path: Path to one saved certification or model-release receipt.

    Returns:
        The complete result document for that receipt.

    Raises:
        ComparisonOperationError: The file could not be read, admitted, or validated.
    """
    document = _admit(receipt_path)
    schema = document.get("schema")
    if schema == MODEL_RELEASE_SCHEMA:
        _validate_model_release(document)
        certification = document.get("certification_receipt")
        policy_origin = POLICY_NOT_RECORDED
    elif schema == CERTIFICATION_SCHEMA:
        _validate_certification(document)
        certification = document
        policy_origin = POLICY_NOT_APPLICABLE
    else:
        raise _failure(
            OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            "schema",
            issue="receipt_schema_not_admitted",
            observed_schema=schema if isinstance(schema, str) else None,
            admitted_schemas=list(_ADMITTED_SCHEMAS),
        )
    return build_document(
        command="show",
        exit_code=0,
        observation=OBSERVATION_RECORDED,
        receipt=document,
        policy_origin=policy_origin,
        reader_version=__version__,
        inputs=_recorded_inputs(certification),
        artifacts={
            "output_dir": None,
            "receipt": str(Path(receipt_path).absolute()),
            "receipt_sha256": document.get("receipt_sha256"),
            "markdown": None,
            "html": None,
        },
    )


def _admit(receipt_path: str) -> dict[str, CanonicalValue]:
    """Read and strictly parse one bounded regular file as a JSON object."""
    try:
        data = read_bounded_regular_file(receipt_path, RECEIPT_JSON_LIMITS.max_bytes)
    except (JsonAdmissionError, OSError, TypeError, ValueError) as exc:
        raise _failure(
            OperationalReasonCode.CONFIGURATION_IO_FAILED,
            "receipt",
            exception_type=type(exc).__name__,
            message=str(exc),
        ) from exc
    try:
        value = bounded_strict_json_loads(data, RECEIPT_JSON_LIMITS)
    except (JsonAdmissionError, TypeError, ValueError) as exc:
        raise _failure(
            OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            "receipt",
            exception_type=type(exc).__name__,
            message=str(exc),
        ) from exc
    if not isinstance(value, dict):
        raise _failure(
            OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            "receipt",
            issue="receipt_root_not_object",
        )
    return value


def _validate_model_release(document: Mapping[str, CanonicalValue]) -> None:
    """Run the complete model-release validator, including its hash linkage."""
    from ..model_release import validate_model_release_receipt

    try:
        validate_model_release_receipt(document)
    except (JsonAdmissionError, KeyError, TypeError, ValueError) as exc:
        raise _failure(
            OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            "receipt",
            issue="model_release_receipt_invalid",
            exception_type=type(exc).__name__,
            message=str(exc),
        ) from exc


def _validate_certification(document: Mapping[str, CanonicalValue]) -> None:
    """Run the complete certification validator, including its hash linkage."""
    from ..certify import validate_receipt

    try:
        validate_receipt(document)
    except (JsonAdmissionError, KeyError, TypeError, ValueError) as exc:
        raise _failure(
            OperationalReasonCode.CONFIGURATION_PARSE_FAILED,
            "receipt",
            issue="certification_receipt_invalid",
            exception_type=type(exc).__name__,
            message=str(exc),
        ) from exc


def _recorded_inputs(certification: CanonicalValue) -> dict[str, CanonicalValue]:
    """Report each role's retained identity; a saved read knows no live path or root."""
    if not isinstance(certification, dict):
        return {"baseline": None, "candidate": None, "roots_correspond": None}
    inputs: dict[str, CanonicalValue] = {"roots_correspond": None}
    for role in ("baseline", "candidate"):
        block = certification.get(role)
        if isinstance(block, dict):
            identity = closure_identity(block)
            identity["path"] = None
            identity["root"] = None
            inputs[role] = identity
        else:
            inputs[role] = None
    return inputs
