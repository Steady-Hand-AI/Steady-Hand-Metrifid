"""Descriptive compiled-field report over the public MjModel surface.

This report is descriptive evidence, never a decision. The status is already fixed by the
complete-MJB byte comparison before any field is read, and nothing here can change it. The report
exists so that a difference can be located and understood, not so that any field can be excluded
from the identity claim.

It is deliberately bounded and therefore incomplete: the number of changed fields, the witnesses
per field, the omitted fields and the bytes copied per field are all capped, and reaching any cap
sets the report-level `truncated` flag. An unset flag means no cap was reached, not that the
report enumerates every difference in the artifact.

It reloads the two private artifacts rather than reusing the compiled models, so the fields it
describes come from the same bytes the certificate compared. Each model is loaded, read and
released on its own, so at most one model and one bounded set of captured baseline fields are
resident at a time.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mujoco  # type: ignore[import-untyped]
import numpy as np

from ..json_values import Binary64, CanonicalValue
from ._bytes import ByteComparison
from ._field_schema import (
    _CHANGED_FIELD_MEMBERS as _CHANGED_FIELD_MEMBERS,
)
from ._field_schema import (
    _EXPANDED_CONTAINERS,
    _OMITTED_CALLABLE,
    _OMITTED_CONTAINER,
    _OMITTED_INACCESSIBLE,
    _OMITTED_UNSUPPORTED,
    FIELD_DIFFERENCES_IDENTIFIED,
    FIELD_REPORT_SCHEMA,
    FIELD_REPORT_SCHEMA_VERSION,
    FIELD_REPORT_STATUSES,
    MAX_CHANGED_FIELDS_RETURNED,
    MAX_FIELD_WITNESS_COPY_BYTES,
    MAX_OMITTED_FIELDS_RETURNED,
    MAX_WITNESSES_PER_FIELD,
    NO_PUBLIC_FIELD_DIFFERENCE_IDENTIFIED,
)
from ._field_schema import (
    _FIELD_REPORT_MEMBERS as _FIELD_REPORT_MEMBERS,
)
from ._field_schema import (
    _OMISSION_REASONS as _OMISSION_REASONS,
)
from ._field_schema import (
    _OMITTED_FIELD_MEMBERS as _OMITTED_FIELD_MEMBERS,
)
from ._field_schema import (
    _WITNESS_MEMBERS as _WITNESS_MEMBERS,
)

_OVER_BUDGET = object()


@dataclass(frozen=True, slots=True)
class _FieldFacts:
    """What one comparable public member looks like in one role."""

    type_name: str
    dtype: str | None
    shape: tuple[int, ...] | None
    sha256: str


def build_field_report(
    baseline_mjb: Path, candidate_mjb: Path, comparison: ByteComparison
) -> dict[str, CanonicalValue]:
    """Describe, deterministically, where two differing artifacts differ publicly."""
    baseline_facts, omitted = _facts_from_artifact(baseline_mjb)
    candidate_facts, _ = _facts_from_artifact(candidate_mjb)
    paths = sorted(set(baseline_facts) | set(candidate_facts))
    changed = [path for path in paths if baseline_facts.get(path) != candidate_facts.get(path)]
    returned = changed[:MAX_CHANGED_FIELDS_RETURNED]
    entries, declined = _changed_entries(
        baseline_mjb, candidate_mjb, returned, baseline_facts, candidate_facts
    )
    status = FIELD_DIFFERENCES_IDENTIFIED if changed else NO_PUBLIC_FIELD_DIFFERENCE_IDENTIFIED
    return {
        "schema": FIELD_REPORT_SCHEMA,
        "schema_version": FIELD_REPORT_SCHEMA_VERSION,
        "baseline_mjb_size_bytes": comparison.baseline_mjb_size_bytes,
        "candidate_mjb_size_bytes": comparison.candidate_mjb_size_bytes,
        "first_differing_byte_offset": comparison.first_differing_byte_offset,
        "differing_byte_count": comparison.differing_byte_count,
        "field_report_status": status,
        "fields_compared_count": len(paths),
        "fields_omitted_count": len(omitted),
        "changed_fields_total": len(changed),
        "changed_fields_returned": len(returned),
        # True whenever any configured bound was reached: the changed-field bound, the omitted
        # field bound, a witness list shorter than the changed elements, or a field whose arrays
        # exceeded the copy budget and were never copied. A reader must never take a bounded
        # witness list for the whole difference.
        "truncated": len(changed) > len(returned)
        or len(omitted) > MAX_OMITTED_FIELDS_RETURNED
        or declined > 0
        or any(_witnesses_truncated(entry) for entry in entries),
        "changed_fields": entries,
        "omitted_fields": [
            {"path": path, "reason": reason}
            for path, reason in omitted[:MAX_OMITTED_FIELDS_RETURNED]
        ],
    }


def _witnesses_truncated(entry: CanonicalValue) -> bool:
    """True when a returned changed field has more changed elements than reported witnesses."""
    if type(entry) is not dict:
        return False
    changed = entry.get("changed_element_count")
    witnesses = entry.get("witnesses")
    if type(changed) is not int or type(witnesses) is not list:
        return False
    return changed > len(witnesses)


def _facts_from_artifact(path: Path) -> tuple[dict[str, _FieldFacts], list[tuple[str, str]]]:
    """Load one artifact, measure every comparable public member, then release the model."""
    model = mujoco.MjModel.from_binary_path(str(path))
    try:
        facts: dict[str, _FieldFacts] = {}
        omitted: list[tuple[str, str]] = []
        for member_path, value, reason in _public_members(model):
            if reason is not None:
                omitted.append((member_path, reason))
                continue
            facts[member_path] = _field_facts(value)
        return facts, sorted(omitted)
    finally:
        del model


def _public_members(model: mujoco.MjModel) -> Iterator[tuple[str, object, str | None]]:
    """Yield the proven public surface: top-level members plus one level under three containers."""
    for name in sorted(_member_names(model)):
        if name in _EXPANDED_CONTAINERS:
            yield name, None, _OMITTED_CONTAINER
            continue
        yield from _classified(model, name, name)
    for container in _EXPANDED_CONTAINERS:
        holder = getattr(model, container, None)
        if holder is None:
            continue
        for name in sorted(_member_names(holder)):
            yield from _classified(holder, name, f"{container}.{name}")


def _classified(holder: object, name: str, path: str) -> Iterator[tuple[str, object, str | None]]:
    """Classify one public MuJoCo member as scalar, array, omitted, or absent."""
    try:
        value = getattr(holder, name)
    except Exception:  # a member that cannot be read is reported, never silently dropped
        yield path, None, _OMITTED_INACCESSIBLE
        return
    if callable(value):
        yield path, None, _OMITTED_CALLABLE
    elif _comparable(value):
        yield path, value, None
    else:
        yield path, None, _OMITTED_UNSUPPORTED


def _member_names(holder: object) -> list[str]:
    """Return deterministic public data-member names for a MuJoCo holder."""
    return [name for name in dir(holder) if not name.startswith("_")]


def _comparable(value: object) -> bool:
    """Return whether a member is a supported scalar or NumPy array value."""
    return isinstance(value, (np.ndarray, np.generic, bool, int, float, bytes, str))


def _field_facts(value: object) -> _FieldFacts:
    """Describe one comparable member's kind, dtype, shape, and element count."""
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        payload = (
            f"array:{contiguous.dtype.str}:{contiguous.shape}:".encode() + contiguous.tobytes()
        )
        return _FieldFacts(
            type_name="ndarray",
            dtype=str(contiguous.dtype),
            shape=tuple(int(size) for size in contiguous.shape),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    scalar = value.item() if isinstance(value, np.generic) else value
    return _FieldFacts(
        type_name=type(scalar).__name__,
        dtype=None,
        shape=None,
        sha256=hashlib.sha256(_scalar_payload(scalar)).hexdigest(),
    )


def _scalar_payload(value: object) -> bytes:
    """Encode one supported scalar into its canonical comparison payload."""
    if isinstance(value, bool):
        return b"bool:" + (b"1" if value else b"0")
    if isinstance(value, int):
        return b"int:" + str(value).encode()
    if isinstance(value, float):
        return b"float:" + struct.pack("<d", value)
    if isinstance(value, bytes):
        return b"bytes:" + value
    if isinstance(value, str):
        return b"str:" + value.encode("utf-8", errors="strict")
    raise TypeError("scalar payload requires a proven public scalar type")


def _changed_entries(
    baseline_mjb: Path,
    candidate_mjb: Path,
    paths: list[str],
    baseline_facts: Mapping[str, _FieldFacts],
    candidate_facts: Mapping[str, _FieldFacts],
) -> tuple[list[CanonicalValue], int]:
    """Describe each changed field in sorted path order, one field's copies at a time.

    Both artifacts are open while this runs, but nothing is copied out of them except the single
    field currently being described, and only when that field fits the copy budget. The previous
    implementation copied every changed field from both roles before describing any of them.
    """
    if not paths:
        return [], 0
    declined = 0
    baseline_model = mujoco.MjModel.from_binary_path(str(baseline_mjb))
    try:
        candidate_model = mujoco.MjModel.from_binary_path(str(candidate_mjb))
        try:
            entries: list[CanonicalValue] = []
            for path in sorted(paths):
                # The copies live only inside _describe_one's frame, so they are unreachable
                # before the next field is copied. Binding them here would keep the previous
                # field alive across the next _bounded_pair call.
                entry, withheld = _describe_one(
                    baseline_model,
                    candidate_model,
                    path,
                    baseline_facts.get(path),
                    candidate_facts.get(path),
                )
                declined += withheld
                entries.append(entry)
            return entries, declined
        finally:
            del candidate_model
    finally:
        del baseline_model


def _describe_one(
    baseline_model: mujoco.MjModel,
    candidate_model: mujoco.MjModel,
    path: str,
    baseline_facts: _FieldFacts | None,
    candidate_facts: _FieldFacts | None,
) -> tuple[dict[str, CanonicalValue], int]:
    """Describe exactly one changed field. Its copies do not outlive this call."""
    baseline_value, candidate_value = _bounded_pair(baseline_model, candidate_model, path)
    entry = _changed_entry(path, baseline_value, candidate_value, baseline_facts, candidate_facts)
    return entry, 1 if baseline_value is _OVER_BUDGET else 0


def _member_view(model: mujoco.MjModel, path: str) -> object | None:
    """Read one member without copying it out of the model."""
    holder: object = model
    attribute = path
    if "." in path:
        container, attribute = path.split(".", 1)
        holder = getattr(model, container, None)
    if holder is None:
        return None
    try:
        return cast(object, getattr(holder, attribute))
    except Exception:
        return None


def _over_copy_budget(value: object) -> bool:
    """Return whether copying an array would exceed the field-evidence byte budget."""
    return isinstance(value, np.ndarray) and int(value.nbytes) > MAX_FIELD_WITNESS_COPY_BYTES


def _copied(value: object) -> object:
    """Copy a supported scalar or array into private comparison storage."""
    return np.array(value, copy=True) if isinstance(value, np.ndarray) else value


def _bounded_pair(
    baseline_model: mujoco.MjModel, candidate_model: mujoco.MjModel, path: str
) -> tuple[object, object]:
    """Copy one changed field from both roles, or decline the copy when it exceeds the budget."""
    baseline_value = _member_view(baseline_model, path)
    candidate_value = _member_view(candidate_model, path)
    if _over_copy_budget(baseline_value) or _over_copy_budget(candidate_value):
        return _OVER_BUDGET, _OVER_BUDGET
    return _copied(baseline_value), _copied(candidate_value)


def _changed_entry(
    path: str,
    baseline_value: object,
    candidate_value: object,
    baseline: _FieldFacts | None,
    candidate: _FieldFacts | None,
) -> dict[str, CanonicalValue]:
    """Build bounded difference evidence for one comparable public member."""
    entry: dict[str, CanonicalValue] = {
        "path": path,
        "baseline_type": None if baseline is None else baseline.type_name,
        "candidate_type": None if candidate is None else candidate.type_name,
        "baseline_dtype": None if baseline is None else baseline.dtype,
        "candidate_dtype": None if candidate is None else candidate.dtype,
        "baseline_shape": None
        if baseline is None or baseline.shape is None
        else list(baseline.shape),
        "candidate_shape": (
            None if candidate is None or candidate.shape is None else list(candidate.shape)
        ),
        "baseline_sha256": None if baseline is None else baseline.sha256,
        "candidate_sha256": None if candidate is None else candidate.sha256,
    }
    entry.update(_element_evidence(baseline_value, candidate_value))
    return entry


def _element_evidence(baseline: object, candidate: object) -> dict[str, CanonicalValue]:
    """Count changed elements and take up to eight sorted index/value witnesses.

    A field declined by the copy budget reports no count and no witnesses. Its identity - path,
    types, dtypes, shapes and digests - is still published, and the report-level `truncated`
    flag records that something was withheld.
    """
    if baseline is _OVER_BUDGET or candidate is _OVER_BUDGET:
        return {"changed_element_count": None, "witnesses": []}
    if not isinstance(baseline, np.ndarray) or not isinstance(candidate, np.ndarray):
        return {
            "changed_element_count": None,
            "witnesses": [_witness([], baseline, candidate)],
        }
    left = np.ascontiguousarray(baseline)
    right = np.ascontiguousarray(candidate)
    if left.shape != right.shape or left.dtype != right.dtype or left.size == 0:
        return {"changed_element_count": None, "witnesses": []}
    width = left.dtype.itemsize
    changed_mask = np.any(
        left.reshape(-1).view(np.uint8).reshape(left.size, width)
        != right.reshape(-1).view(np.uint8).reshape(right.size, width),
        axis=1,
    )
    flat = np.flatnonzero(changed_mask)
    witnesses: list[CanonicalValue] = []
    for index in flat[:MAX_WITNESSES_PER_FIELD]:
        position = np.unravel_index(int(index), left.shape)
        witnesses.append(
            _witness([int(axis) for axis in position], left[position], right[position])
        )
    return {"changed_element_count": int(flat.size), "witnesses": witnesses}


def _witness(index: list[int], baseline: object, candidate: object) -> dict[str, CanonicalValue]:
    """One witness: the exact value, plus a readable rendering for the Markdown report."""
    return {
        "index": cast("list[CanonicalValue]", index),
        "baseline_value": _canonical_element(baseline),
        "candidate_value": _canonical_element(candidate),
        "baseline_text": _element_text(baseline),
        "candidate_text": _element_text(candidate),
    }


def _canonical_element(value: object) -> CanonicalValue:
    """Render one element canonically. Floats keep every bit, including NaN and infinities."""
    element = value.item() if isinstance(value, np.generic) else value
    if isinstance(element, bool) or isinstance(element, int) or isinstance(element, str):
        return element
    if isinstance(element, float):
        return Binary64.from_float(element).to_primitive()
    if isinstance(element, bytes):
        return element.decode("utf-8", errors="replace")
    return None


def _element_text(value: object) -> str:
    """Render one differing scalar element in a stable human-readable form."""
    element = value.item() if isinstance(value, np.generic) else value
    if isinstance(element, bytes):
        return element.decode("utf-8", errors="replace")
    return repr(element)


__all__ = [
    "FIELD_DIFFERENCES_IDENTIFIED",
    "FIELD_REPORT_STATUSES",
    "FIELD_REPORT_SCHEMA",
    "MAX_CHANGED_FIELDS_RETURNED",
    "MAX_OMITTED_FIELDS_RETURNED",
    "MAX_WITNESSES_PER_FIELD",
    "NO_PUBLIC_FIELD_DIFFERENCE_IDENTIFIED",
    "build_field_report",
]
