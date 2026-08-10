"""Validate workload NPZ primitives and compute semantic hashes."""

from __future__ import annotations

import hashlib
from itertools import pairwise
from typing import cast

import numpy as np
import numpy.typing as npt

from ._npz import refuse
from .json_values import CanonicalValue, canonical_sha256
from .operational import OperationalReasonCode
from .schemas import ActionsArtifactMetadata, StateArtifactMetadata


def _unicode_scalar(
    array: npt.NDArray[np.generic],
    invalid_reason: OperationalReasonCode,
    field: str,
) -> str:
    """Read one scalar NumPy Unicode metadata value or raise the field-specific refusal."""
    if array.dtype.kind != "U" or not array.dtype.str.startswith("<U"):
        raise refuse(OperationalReasonCode.NPZ_DTYPE_MISMATCH, field=field)
    if array.ndim != 0:
        raise refuse(invalid_reason, field=field, issue="rank_mismatch")
    value = str(array.item())
    _validate_name_text(value, invalid_reason, field)
    return value


def _unicode_names(
    array: npt.NDArray[np.generic],
    invalid_reason: OperationalReasonCode,
    field: str,
) -> tuple[str, ...]:
    """Decode a one-dimensional canonical NumPy Unicode name array."""
    if array.dtype.kind != "U" or not array.dtype.str.startswith("<U"):
        raise refuse(OperationalReasonCode.NPZ_DTYPE_MISMATCH, field=field)
    if array.ndim != 1:
        raise refuse(invalid_reason, field=field, issue="rank_mismatch")
    values = tuple(str(item) for item in array.tolist())
    for value in values:
        _validate_name_text(value, invalid_reason, field)
    if len(values) != len(set(values)):
        raise refuse(invalid_reason, field=field, issue="duplicate_name")
    return values


def _integer_scalar(
    array: npt.NDArray[np.generic],
    invalid_reason: OperationalReasonCode,
    field: str,
) -> int:
    """Read one strict little-endian signed integer scalar from an NPZ member."""
    if array.dtype.str != "<i8":
        raise refuse(OperationalReasonCode.NPZ_DTYPE_MISMATCH, field=field)
    if array.ndim != 0:
        raise refuse(invalid_reason, field=field, issue="rank_mismatch")
    return int(array.item())


def _validate_name_text(
    value: str,
    invalid_reason: OperationalReasonCode,
    field: str,
) -> None:
    """Reject empty, NUL-containing, or non-round-tripping workload names."""
    if not value or "\x00" in value:
        raise refuse(invalid_reason, field=field, issue="invalid_unicode_text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise refuse(invalid_reason, field=field, issue="invalid_unicode_text") from exc


def _offset_array(array: npt.NDArray[np.generic], field: str) -> tuple[int, ...]:
    """Decode a one-dimensional little-endian int64 segmentation-offset array."""
    if array.dtype.str != "<i8":
        raise refuse(OperationalReasonCode.NPZ_DTYPE_MISMATCH, field=field)
    if array.ndim != 1:
        raise refuse(OperationalReasonCode.STATE_OFFSET_INVALID, field=field, issue="rank")
    return tuple(int(item) for item in array.tolist())


def _float_array(
    array: npt.NDArray[np.generic],
    field: str,
    dimensions: int,
) -> npt.NDArray[np.float64]:
    """Admit a C-contiguous little-endian float64 workload array of fixed rank."""
    if array.dtype.str != "<f8":
        raise refuse(OperationalReasonCode.NPZ_DTYPE_MISMATCH, field=field)
    if array.ndim != dimensions:
        reason = (
            OperationalReasonCode.ACTION_SHAPE_MISMATCH
            if field == "values"
            else OperationalReasonCode.STATE_ARTIFACT_INVALID
        )
        raise refuse(reason, field=field, issue="rank_mismatch")
    if not array.flags.c_contiguous:
        reason = (
            OperationalReasonCode.ACTION_SHAPE_MISMATCH
            if field == "values"
            else OperationalReasonCode.STATE_ARTIFACT_INVALID
        )
        raise refuse(reason, field=field, issue="not_c_contiguous")
    return cast(npt.NDArray[np.float64], array)


def _validate_offsets(
    offsets: tuple[int, ...],
    item_count: int,
    value_count: int,
    field: str,
) -> None:
    """Require offsets to partition a flattened workload vector exactly."""
    valid = (
        len(offsets) == item_count + 1
        and bool(offsets)
        and offsets[0] == 0
        and all(left <= right for left, right in pairwise(offsets))
        and offsets[-1] == value_count
    )
    if not valid:
        raise refuse(OperationalReasonCode.STATE_OFFSET_INVALID, field=field)


def _validate_widths(
    offsets: tuple[int, ...],
    expected_widths: tuple[int, ...],
    field: str,
) -> None:
    """Match each segmented workload width to its compiled model element width."""
    actual = tuple(right - left for left, right in pairwise(offsets))
    if actual != expected_widths:
        raise refuse(
            OperationalReasonCode.STATE_WIDTH_MISMATCH,
            field=field,
            expected_count=len(expected_widths),
            actual_count=len(actual),
        )


def _require_finite(array: npt.NDArray[np.float64], field: str) -> None:
    """Refuse workload arrays containing NaN or infinite values."""
    if not bool(np.isfinite(array).all()):
        raise refuse(OperationalReasonCode.INPUT_NONFINITE_VALUE, field=field)


def _make_read_only(*arrays: npt.NDArray[np.float64]) -> None:
    """Freeze admitted workload arrays against mutation after validation."""
    for array in arrays:
        array.setflags(write=False)


def _array_semantic_primitive(
    array: npt.NDArray[np.float64],
) -> dict[str, CanonicalValue]:
    """Describe an admitted array by dtype, shape, and exact C-order byte hash."""
    return {
        "dtype": "<f8",
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _state_semantic_sha256(
    metadata: StateArtifactMetadata,
    qpos: npt.NDArray[np.float64],
    qvel: npt.NDArray[np.float64],
    act: npt.NDArray[np.float64],
) -> str:
    """Hash state metadata with exact qpos, qvel, and activation array identities."""
    return canonical_sha256(
        {
            "schema": "metrifid.state_semantic",
            "schema_version": 1,
            "metadata": metadata.to_primitive(),
            "qpos": _array_semantic_primitive(qpos),
            "qvel": _array_semantic_primitive(qvel),
            "act": _array_semantic_primitive(act),
        }
    )


def _actions_semantic_sha256(
    metadata: ActionsArtifactMetadata,
    values: npt.NDArray[np.float64],
) -> str:
    """Hash actions metadata and exact control-matrix identity."""
    return canonical_sha256(
        {
            "schema": "metrifid.actions_semantic",
            "schema_version": 1,
            "metadata": metadata.to_primitive(),
            "values": _array_semantic_primitive(values),
        }
    )


def _require_runtime_float_array(
    array: object,
    field: str,
    dimensions: int,
) -> None:
    """Require finite, read-only, C-order little-endian float64 runtime data of fixed rank."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{field} must be an ndarray")
    if array.dtype.str != "<f8" or array.ndim != dimensions:
        raise ValueError(f"{field} has an invalid runtime dtype or rank")
    if not array.flags.c_contiguous or array.flags.writeable:
        raise ValueError(f"{field} must be read-only and C-contiguous")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{field} must contain only finite values")
