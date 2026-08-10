"""Pure schema constants and member registries for the compiled-field report.

The field report is produced by :mod:`metrifid.certify._fields`, which necessarily imports MuJoCo
and NumPy to read a compiled model. A receipt *reader* needs only the frozen vocabulary: schema
names, statuses, witness bounds, member registries, and the omission registry. Those live here so
receipt parsing and validation stay importable without any native dependency.

The producer imports these same constants, so producer and validator cannot drift apart.
"""

from __future__ import annotations

FIELD_REPORT_SCHEMA = "metrifid.compiled_field_report"
FIELD_REPORT_SCHEMA_VERSION = 1

FIELD_DIFFERENCES_IDENTIFIED = "PUBLIC_FIELD_DIFFERENCES_IDENTIFIED"
NO_PUBLIC_FIELD_DIFFERENCE_IDENTIFIED = "NO_PUBLIC_FIELD_DIFFERENCE_IDENTIFIED"

# The complete frozen status registry a receipt reader may accept.
FIELD_REPORT_STATUSES = (FIELD_DIFFERENCES_IDENTIFIED, NO_PUBLIC_FIELD_DIFFERENCE_IDENTIFIED)

MAX_CHANGED_FIELDS_RETURNED = 100
MAX_WITNESSES_PER_FIELD = 8
MAX_OMITTED_FIELDS_RETURNED = 200

# Descriptive witnesses are evidence, not the decision. One changed field's copied arrays may not
# exceed this budget, and no two changed fields are ever copied at the same time. A field over the
# budget keeps its identity - path, types, dtypes, shapes and digests - and reports no element
# count and no witnesses, with the report-level `truncated` flag set.
MAX_FIELD_WITNESS_COPY_BYTES = 64 * 1024 * 1024

# The three documented one-level containers. Everything else is read at the top level only.
_EXPANDED_CONTAINERS = ("opt", "stat", "vis")

_OMITTED_CALLABLE = "CALLABLE_MEMBER"
_OMITTED_CONTAINER = "EXPANDED_ONE_LEVEL_BELOW"
_OMITTED_UNSUPPORTED = "UNSUPPORTED_MEMBER_TYPE"
_OMITTED_INACCESSIBLE = "MEMBER_ACCESS_FAILED"

# Marks a changed field whose arrays were never copied because they exceed the copy budget.
# The producer's complete frozen omission registry. The receipt validator reads this rather than
# repeating the four literals, so the two can never drift apart. It stays private to the package.
_OMISSION_REASONS = (
    _OMITTED_CALLABLE,
    _OMITTED_CONTAINER,
    _OMITTED_UNSUPPORTED,
    _OMITTED_INACCESSIBLE,
)

# The complete frozen field-report member sets a reader may accept.
_FIELD_REPORT_MEMBERS = (
    "schema",
    "schema_version",
    "baseline_mjb_size_bytes",
    "candidate_mjb_size_bytes",
    "first_differing_byte_offset",
    "differing_byte_count",
    "field_report_status",
    "fields_compared_count",
    "fields_omitted_count",
    "changed_fields_total",
    "changed_fields_returned",
    "truncated",
    "changed_fields",
    "omitted_fields",
)
_CHANGED_FIELD_MEMBERS = (
    "path",
    "baseline_type",
    "candidate_type",
    "baseline_dtype",
    "candidate_dtype",
    "baseline_shape",
    "candidate_shape",
    "baseline_sha256",
    "candidate_sha256",
    "changed_element_count",
    "witnesses",
)
_WITNESS_MEMBERS = ("index", "baseline_value", "candidate_value", "baseline_text", "candidate_text")
_OMITTED_FIELD_MEMBERS = ("path", "reason")


__all__ = [
    "FIELD_DIFFERENCES_IDENTIFIED",
    "FIELD_REPORT_SCHEMA",
    "FIELD_REPORT_SCHEMA_VERSION",
    "FIELD_REPORT_STATUSES",
    "MAX_CHANGED_FIELDS_RETURNED",
    "MAX_FIELD_WITNESS_COPY_BYTES",
    "MAX_OMITTED_FIELDS_RETURNED",
    "MAX_WITNESSES_PER_FIELD",
    "NO_PUBLIC_FIELD_DIFFERENCE_IDENTIFIED",
]
