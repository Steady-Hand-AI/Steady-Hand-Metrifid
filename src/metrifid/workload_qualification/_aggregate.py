"""The strict typed model of a published workload-qualification receipt.

The previous checker read a few envelope fields and recomputed a self-hash. Everything else in the
document was accepted as written, so a user could edit a decision-bearing field, reseal the hash,
and obtain an accepted receipt that contradicted its own evidence.

This model parses the whole document instead. Every level declares its exact field set and refuses
unknown or missing members, every enum is checked against its registry, every hash is checked for
shape, every ordered collection is checked for canonical order and uniqueness, and schema-v1
cardinality is enforced. Parsing alone still proves nothing about the decision; it makes the
document typed enough for `_reconstruct` to recompute the decision from it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Self

from ..errors import ComparisonStatus
from ..json_values import CanonicalValue
from ._config import MAX_PROBE_GROUPS, MAX_VARIANTS, MAX_WORKLOADS, REQUIRED_BUDGET
from ._paths import PathAdmissionError, admit_relative_path
from ._status import (
    QUALIFICATION_LIMITATIONS,
    CellOutcome,
    ProbeGroupStatus,
    QualificationStatus,
)

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_DECIMAL = re.compile(r"\A(0|[1-9][0-9]*)(\.[0-9]+)?\Z")
ZERO_CHANGE_KIND: Final[str] = "ZERO_CHANGE"
PROBE_KIND: Final[str] = "PROBE"
MAX_CELLS: Final[int] = MAX_WORKLOADS * MAX_PROBE_GROUPS * MAX_VARIANTS


class AggregateSchemaError(ValueError):
    """Raised when a published receipt does not match the strict aggregate schema."""


def _object(value: object, context: str) -> Mapping[str, CanonicalValue]:
    """Require one JSON object."""
    if not isinstance(value, dict):
        raise AggregateSchemaError(f"{context} must be an object")
    for key in value:
        if not isinstance(key, str):
            raise AggregateSchemaError(f"{context} keys must be strings")
    return value


def _exact_fields(obj: Mapping[str, CanonicalValue], expected: set[str], context: str) -> None:
    """Require exactly this field set, naming what is missing and what is unknown."""
    present = set(obj)
    missing = sorted(expected - present)
    unknown = sorted(present - expected)
    if missing or unknown:
        raise AggregateSchemaError(
            f"{context} field set is wrong; missing={missing} unknown={unknown}"
        )


def _list(value: object, context: str, *, maximum: int | None = None) -> list[CanonicalValue]:
    """Require one JSON array within its schema-v1 bound."""
    if not isinstance(value, list):
        raise AggregateSchemaError(f"{context} must be an array")
    if maximum is not None and len(value) > maximum:
        raise AggregateSchemaError(f"{context} exceeds the schema bound of {maximum}")
    return value


def _text(value: object, context: str) -> str:
    """Require one string."""
    if not isinstance(value, str):
        raise AggregateSchemaError(f"{context} must be a string")
    return value


def _integer(value: object, context: str) -> int:
    """Require one JSON integer that is not a boolean."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise AggregateSchemaError(f"{context} must be an integer")
    return value


def _boolean(value: object, context: str) -> bool:
    """Require one JSON boolean."""
    if not isinstance(value, bool):
        raise AggregateSchemaError(f"{context} must be a boolean")
    return value


def _sha256(value: object, context: str) -> str:
    """Require one lowercase hexadecimal SHA-256."""
    text = _text(value, context)
    if _SHA256.fullmatch(text) is None:
        raise AggregateSchemaError(f"{context} must be a lowercase hexadecimal SHA-256")
    return text


def _decimal_token(value: object, context: str) -> str:
    """Require one exact ordinary decimal token."""
    text = _text(value, context)
    if _DECIMAL.fullmatch(text) is None:
        raise AggregateSchemaError(f"{context} must be an exact decimal token")
    return text


def _locator(value: object, context: str) -> str:
    """Require one normalized, relative, traversal-free locator."""
    try:
        return admit_relative_path(value, context)
    except PathAdmissionError as exc:
        raise AggregateSchemaError(str(exc)) from exc


def _enum(value: object, allowed: Sequence[str], context: str) -> str:
    """Require one member of a closed registry."""
    text = _text(value, context)
    if text not in allowed:
        raise AggregateSchemaError(f"{context} must be one of {sorted(allowed)}")
    return text


@dataclass(frozen=True, slots=True)
class ControlRecord:
    """One zero-change control as published."""

    workload_index: int
    workload_id: str
    comparison_status: str
    outcome: str
    eligible: bool
    exclusion_reason: str | None
    comparison_config_locator: str
    comparison_config_raw_sha256: str
    comparison_receipt_locator: str
    comparison_receipt_raw_sha256: str
    comparison_receipt_sha256: str

    @classmethod
    def from_primitive(cls, value: object, context: str) -> Self:
        """Parse one control record strictly."""
        obj = _object(value, context)
        _exact_fields(
            obj,
            {
                "kind",
                "workload_index",
                "workload_id",
                "comparison_status",
                "outcome",
                "eligible",
                "exclusion_reason",
                "comparison_config_locator",
                "comparison_config_raw_sha256",
                "comparison_receipt_locator",
                "comparison_receipt_raw_sha256",
                "comparison_receipt_sha256",
            },
            context,
        )
        _enum(obj["kind"], (ZERO_CHANGE_KIND,), f"{context}.kind")
        reason = obj["exclusion_reason"]
        return cls(
            workload_index=_integer(obj["workload_index"], f"{context}.workload_index"),
            workload_id=_text(obj["workload_id"], f"{context}.workload_id"),
            comparison_status=_enum(
                obj["comparison_status"],
                [s.value for s in ComparisonStatus],
                f"{context}.comparison_status",
            ),
            outcome=_enum(obj["outcome"], [o.value for o in CellOutcome], f"{context}.outcome"),
            eligible=_boolean(obj["eligible"], f"{context}.eligible"),
            exclusion_reason=None if reason is None else _text(reason, f"{context}.reason"),
            comparison_config_locator=_locator(
                obj["comparison_config_locator"], f"{context}.comparison_config_locator"
            ),
            comparison_config_raw_sha256=_sha256(
                obj["comparison_config_raw_sha256"], f"{context}.comparison_config_raw_sha256"
            ),
            comparison_receipt_locator=_locator(
                obj["comparison_receipt_locator"], f"{context}.comparison_receipt_locator"
            ),
            comparison_receipt_raw_sha256=_sha256(
                obj["comparison_receipt_raw_sha256"], f"{context}.comparison_receipt_raw_sha256"
            ),
            comparison_receipt_sha256=_sha256(
                obj["comparison_receipt_sha256"], f"{context}.comparison_receipt_sha256"
            ),
        )


@dataclass(frozen=True, slots=True)
class ProbeCellRecord:
    """One probe cell as published."""

    workload_index: int
    workload_id: str
    group_index: int
    probe_id: str
    variant_index: int
    magnitude: str
    comparison_status: str
    outcome: str
    comparison_config_locator: str
    comparison_config_raw_sha256: str
    comparison_receipt_locator: str
    comparison_receipt_raw_sha256: str
    comparison_receipt_sha256: str

    @classmethod
    def from_primitive(cls, value: object, context: str) -> Self:
        """Parse one probe cell record strictly."""
        obj = _object(value, context)
        _exact_fields(
            obj,
            {
                "kind",
                "workload_index",
                "workload_id",
                "group_index",
                "probe_id",
                "variant_index",
                "magnitude",
                "comparison_status",
                "outcome",
                "comparison_config_locator",
                "comparison_config_raw_sha256",
                "comparison_receipt_locator",
                "comparison_receipt_raw_sha256",
                "comparison_receipt_sha256",
            },
            context,
        )
        _enum(obj["kind"], (PROBE_KIND,), f"{context}.kind")
        return cls(
            workload_index=_integer(obj["workload_index"], f"{context}.workload_index"),
            workload_id=_text(obj["workload_id"], f"{context}.workload_id"),
            group_index=_integer(obj["group_index"], f"{context}.group_index"),
            probe_id=_text(obj["probe_id"], f"{context}.probe_id"),
            variant_index=_integer(obj["variant_index"], f"{context}.variant_index"),
            magnitude=_decimal_token(obj["magnitude"], f"{context}.magnitude"),
            comparison_status=_enum(
                obj["comparison_status"],
                [s.value for s in ComparisonStatus],
                f"{context}.comparison_status",
            ),
            outcome=_enum(obj["outcome"], [o.value for o in CellOutcome], f"{context}.outcome"),
            comparison_config_locator=_locator(
                obj["comparison_config_locator"], f"{context}.comparison_config_locator"
            ),
            comparison_config_raw_sha256=_sha256(
                obj["comparison_config_raw_sha256"], f"{context}.comparison_config_raw_sha256"
            ),
            comparison_receipt_locator=_locator(
                obj["comparison_receipt_locator"], f"{context}.comparison_receipt_locator"
            ),
            comparison_receipt_raw_sha256=_sha256(
                obj["comparison_receipt_raw_sha256"], f"{context}.comparison_receipt_raw_sha256"
            ),
            comparison_receipt_sha256=_sha256(
                obj["comparison_receipt_sha256"], f"{context}.comparison_receipt_sha256"
            ),
        )


@dataclass(frozen=True, slots=True)
class GroupAdjudication:
    """One probe group's published adjudication."""

    probe_id: str
    status: str
    detection_signature: tuple[str, ...]
    floor_magnitude: str | None
    floor_variant_index: int | None
    no_floor_reason: str | None
    detected_variants: int

    @classmethod
    def from_primitive(cls, value: object, context: str) -> Self:
        """Parse one group adjudication strictly."""
        obj = _object(value, context)
        _exact_fields(
            obj,
            {
                "probe_id",
                "status",
                "detection_signature",
                "floor_magnitude",
                "floor_variant_index",
                "no_floor_reason",
                "detected_variants",
            },
            context,
        )
        floor = obj["floor_magnitude"]
        index = obj["floor_variant_index"]
        reason = obj["no_floor_reason"]
        signature = _list(
            obj["detection_signature"], f"{context}.detection_signature", maximum=MAX_VARIANTS
        )
        return cls(
            probe_id=_text(obj["probe_id"], f"{context}.probe_id"),
            status=_enum(obj["status"], [s.value for s in ProbeGroupStatus], f"{context}.status"),
            detection_signature=tuple(
                _enum(item, [o.value for o in CellOutcome], f"{context}.detection_signature")
                for item in signature
            ),
            floor_magnitude=None if floor is None else _decimal_token(floor, f"{context}.floor"),
            floor_variant_index=None
            if index is None
            else _integer(index, f"{context}.floor_index"),
            no_floor_reason=None if reason is None else _text(reason, f"{context}.no_floor_reason"),
            detected_variants=_integer(obj["detected_variants"], f"{context}.detected_variants"),
        )


@dataclass(frozen=True, slots=True)
class Selection:
    """The published selected subset and its adjudication."""

    workload_ids: tuple[str, ...]
    qualified_groups: int
    unresolved_groups: int
    detected_variants: int
    groups: tuple[GroupAdjudication, ...]

    @classmethod
    def from_primitive(cls, value: object, context: str) -> Self:
        """Parse the selection strictly."""
        obj = _object(value, context)
        _exact_fields(
            obj,
            {
                "workload_ids",
                "qualified_groups",
                "unresolved_groups",
                "detected_variants",
                "groups",
            },
            context,
        )
        ids = _list(obj["workload_ids"], f"{context}.workload_ids", maximum=REQUIRED_BUDGET)
        groups = _list(obj["groups"], f"{context}.groups", maximum=MAX_PROBE_GROUPS)
        return cls(
            workload_ids=tuple(_text(i, f"{context}.workload_ids") for i in ids),
            qualified_groups=_integer(obj["qualified_groups"], f"{context}.qualified_groups"),
            unresolved_groups=_integer(obj["unresolved_groups"], f"{context}.unresolved_groups"),
            detected_variants=_integer(obj["detected_variants"], f"{context}.detected_variants"),
            groups=tuple(
                GroupAdjudication.from_primitive(item, f"{context}.groups[{index}]")
                for index, item in enumerate(groups)
            ),
        )


@dataclass(frozen=True, slots=True)
class AggregateReceipt:
    """One published workload-qualification receipt, parsed strictly at every level."""

    schema: str
    schema_version: int
    status: QualificationStatus
    completed_exit_code: int
    configuration: Mapping[str, CanonicalValue]
    configuration_raw_sha256: str
    configuration_locator: str
    campaign_identity: Mapping[str, CanonicalValue]
    baseline_model_closure: Mapping[str, CanonicalValue]
    probe_model_closures: Mapping[str, CanonicalValue]
    workload_artifact_identities: Mapping[str, CanonicalValue]
    zero_change_controls: tuple[ControlRecord, ...]
    probe_cells: tuple[ProbeCellRecord, ...]
    eligible_workload_ids: tuple[str, ...]
    excluded_workload_ids: tuple[str, ...]
    selected_workload_ids: tuple[str, ...]
    selection: Selection
    subset_ranking: tuple[Mapping[str, CanonicalValue], ...]
    subsets_evaluated: int
    planned_comparisons: int
    execution_counts: Mapping[str, CanonicalValue]
    witnesses: Mapping[str, CanonicalValue]
    limitations: tuple[str, ...]
    not_claimed: tuple[str, ...]
    receipt_sha256: str

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Parse one complete published receipt strictly."""
        obj = _object(value, "workload qualification receipt")
        _exact_fields(
            obj,
            {
                "schema",
                "schema_version",
                "status",
                "completed_exit_code",
                "configuration",
                "configuration_raw_sha256",
                "configuration_locator",
                "campaign_identity",
                "baseline_model_closure",
                "probe_model_closures",
                "workload_artifact_identities",
                "zero_change_controls",
                "probe_cells",
                "eligible_workload_ids",
                "excluded_workload_ids",
                "selected_workload_ids",
                "selection",
                "subset_ranking",
                "subsets_evaluated",
                "planned_comparisons",
                "execution_counts",
                "witnesses",
                "limitations",
                "not_claimed",
                "receipt_sha256",
            },
            "workload qualification receipt",
        )
        controls = _list(obj["zero_change_controls"], "zero_change_controls", maximum=MAX_WORKLOADS)
        cells = _list(obj["probe_cells"], "probe_cells", maximum=MAX_CELLS)
        ranking = _list(obj["subset_ranking"], "subset_ranking", maximum=560)
        status = QualificationStatus(
            _enum(obj["status"], [s.value for s in QualificationStatus], "status")
        )
        limitations = _list(obj["limitations"], "limitations")
        return cls(
            schema=_text(obj["schema"], "schema"),
            schema_version=_integer(obj["schema_version"], "schema_version"),
            status=status,
            completed_exit_code=_integer(obj["completed_exit_code"], "completed_exit_code"),
            configuration=_object(obj["configuration"], "configuration"),
            configuration_raw_sha256=_sha256(
                obj["configuration_raw_sha256"], "configuration_raw_sha256"
            ),
            configuration_locator=_locator(obj["configuration_locator"], "configuration_locator"),
            campaign_identity=_object(obj["campaign_identity"], "campaign_identity"),
            baseline_model_closure=_object(obj["baseline_model_closure"], "baseline_model_closure"),
            probe_model_closures=_object(obj["probe_model_closures"], "probe_model_closures"),
            workload_artifact_identities=_object(
                obj["workload_artifact_identities"], "workload_artifact_identities"
            ),
            zero_change_controls=tuple(
                ControlRecord.from_primitive(item, f"zero_change_controls[{index}]")
                for index, item in enumerate(controls)
            ),
            probe_cells=tuple(
                ProbeCellRecord.from_primitive(item, f"probe_cells[{index}]")
                for index, item in enumerate(cells)
            ),
            eligible_workload_ids=tuple(
                _text(i, "eligible_workload_ids")
                for i in _list(obj["eligible_workload_ids"], "eligible", maximum=MAX_WORKLOADS)
            ),
            excluded_workload_ids=tuple(
                _text(i, "excluded_workload_ids")
                for i in _list(obj["excluded_workload_ids"], "excluded", maximum=MAX_WORKLOADS)
            ),
            selected_workload_ids=tuple(
                _text(i, "selected_workload_ids")
                for i in _list(obj["selected_workload_ids"], "selected", maximum=REQUIRED_BUDGET)
            ),
            selection=Selection.from_primitive(obj["selection"], "selection"),
            subset_ranking=tuple(
                _object(item, f"subset_ranking[{index}]") for index, item in enumerate(ranking)
            ),
            subsets_evaluated=_integer(obj["subsets_evaluated"], "subsets_evaluated"),
            planned_comparisons=_integer(obj["planned_comparisons"], "planned_comparisons"),
            execution_counts=_object(obj["execution_counts"], "execution_counts"),
            witnesses=_object(obj["witnesses"], "witnesses"),
            limitations=tuple(_text(item, "limitations") for item in limitations),
            not_claimed=tuple(
                _text(item, "not_claimed") for item in _list(obj["not_claimed"], "not_claimed")
            ),
            receipt_sha256=_sha256(obj["receipt_sha256"], "receipt_sha256"),
        )

    def check_registry_invariants(self) -> None:
        """Check the closed registries and simple structural uniqueness this model owns."""
        expected = [code.value for code in QUALIFICATION_LIMITATIONS]
        if list(self.limitations) != expected:
            raise AggregateSchemaError(
                "limitations must be the complete registry in canonical order"
            )
        if len(set(self.selected_workload_ids)) != len(self.selected_workload_ids):
            raise AggregateSchemaError("selected_workload_ids must be unique")
        if len(self.selected_workload_ids) != REQUIRED_BUDGET:
            raise AggregateSchemaError(
                f"schema version 1 selects exactly {REQUIRED_BUDGET} workloads"
            )
        locators = [c.comparison_config_locator for c in self.zero_change_controls]
        locators += [c.comparison_receipt_locator for c in self.zero_change_controls]
        locators += [c.comparison_config_locator for c in self.probe_cells]
        locators += [c.comparison_receipt_locator for c in self.probe_cells]
        if len(set(locators)) != len(locators):
            raise AggregateSchemaError("every retained locator must be unique")
