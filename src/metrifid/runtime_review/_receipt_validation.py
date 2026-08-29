"""Independently validate a published Native Runtime Review and its owned raw evidence.

The receipt self-hash is only a corruption check.  Authority for the decision comes from this
module reopening the retained configuration and all twelve owned evidence cells, reconstructing the
private evaluator inputs from the raw traces, rerunning the selected method, and comparing only the
stable public decision fields.  Original input paths are deliberately ignored.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Final, cast

from .._json_admission import (
    CONFIG_JSON_LIMITS,
    RECEIPT_JSON_LIMITS,
    bounded_strict_json_loads,
)
from .._schema_primitives import (
    _bounded_int,
    _fields,
    _nonempty_string,
    _nonnegative_int,
    _object,
    _sequence,
    _string,
)
from ..json_values import (
    CanonicalValue,
    ExactRational,
    canonical_json_bytes,
    canonical_sha256,
    require_sha256,
    validate_self_hash,
)
from ..version import __version__
from ._config import (
    AdmittedRuntimeReviewConfiguration,
    AdmittedRuntimeReviewConfigurationV2,
    RuntimeReviewConfig,
    RuntimeReviewConfigV2,
)
from ._markdown import render_runtime_review_markdown
from ._owned_output import (
    OwnedEvidenceCell,
    OwnedEvidenceMember,
    OwnedProfileIdentity,
    read_complete_owned_runtime_review_tree,
    validate_owned_evidence_cell,
)
from ._receipt import (
    RUNTIME_REVIEW_CLAIM_SCOPE,
    RUNTIME_REVIEW_LIMITATIONS,
    RUNTIME_REVIEW_METHOD,
    RUNTIME_REVIEW_RECEIPT_KEYS,
    RUNTIME_REVIEW_RECEIPT_SCHEMA,
    RUNTIME_REVIEW_RECEIPT_SCHEMA_VERSION,
    RUNTIME_REVIEW_RECEIPT_SCHEMA_VERSION_V2,
    _profiles_primitive_v2,
)
from ._status import RuntimeReviewReasonCode, RuntimeReviewStatus
from ._witness import StableWitness

_WITNESS_FIELDS: Final[set[str]] = {
    "channel_id",
    "classification",
    "kind",
    "semantic_type",
    "time",
    "tolerance",
    "decision_input_sha256",
}
_WITNESS_CLASSIFICATIONS: Final[tuple[str, ...]] = (
    RuntimeReviewStatus.WITHIN_DECLARED_MIGRATION_ENVELOPE.value,
    RuntimeReviewStatus.UNRESOLVED_NEAR_BOUNDARY.value,
    RuntimeReviewStatus.OUTSIDE_DECLARED_MIGRATION_ENVELOPE.value,
)
_CELL_MEMBERS: Final[tuple[str, ...]] = (
    "CHECKSUMS.sha256",
    "fixture.xml",
    "input_manifest.json",
    "model.mjb",
    "result.json",
    "trace.npz",
)
_CANONICAL_SLOTS: Final[tuple[tuple[str, str, int], ...]] = tuple(
    (role, step_dt, repeat_id)
    for role in ("baseline", "candidate")
    for step_dt in ("0.004", "0.002", "0.001")
    for repeat_id in (0, 1)
)
_STEP_TOKENS: Final[Mapping[str, str]] = {
    "0.004": "0p004",
    "0.002": "0p002",
    "0.001": "0p001",
}


def load_and_validate_runtime_review_receipt(
    path: str | Path,
) -> dict[str, CanonicalValue]:
    """Load one published receipt and independently recompute its evidence-backed decision."""
    target = Path(path).absolute()
    if target.name != "runtime_review.json" or target.parent.name != "runtime_review":
        raise ValueError(
            "runtime-review receipt must use the published runtime_review/runtime_review.json path"
        )
    return validate_runtime_review_tree(target.parent)


def validate_runtime_review_tree(review_root: Path) -> dict[str, CanonicalValue]:
    """Independently validate one complete owned tree, including a private staging tree."""
    root_members = read_complete_owned_runtime_review_tree(review_root)
    raw = root_members["runtime_review.json"]
    primitive = bounded_strict_json_loads(raw, RECEIPT_JSON_LIMITS)
    if type(primitive) is not dict:
        raise TypeError("runtime-review receipt must be a JSON object")
    receipt = primitive
    if raw != canonical_json_bytes(receipt) + b"\n":
        raise ValueError("runtime-review receipt file must use the canonical JSON serialization")
    schema_version = _validate_document_schema(receipt)
    validate_self_hash(receipt, "receipt_sha256")
    expected_markdown = render_runtime_review_markdown(receipt).encode("utf-8")
    if root_members["runtime_review.md"] != expected_markdown:
        raise ValueError("runtime-review Markdown does not exactly render its canonical receipt")

    owned_cells = _parse_owned_evidence_cells(receipt["evidence_cells"])
    owned_directories = tuple(
        validate_owned_evidence_cell(review_root, cell) for cell in owned_cells
    )
    if schema_version == RUNTIME_REVIEW_RECEIPT_SCHEMA_VERSION:
        if any(name.startswith("profile_identities/") for name in root_members):
            raise ValueError("legacy runtime-review trees must not add v2 profile identities")
        config, config_path = _load_retained_configuration(
            review_root,
            receipt,
            root_members["admitted_runtime_review_config.json"],
        )
        admitted = AdmittedRuntimeReviewConfiguration(
            config=config,
            path=config_path,
            base_dir=review_root,
            raw_bytes=root_members["admitted_runtime_review_config.json"],
            raw_sha256=cast(str, _object(receipt["configuration"], "configuration")["raw_sha256"]),
            semantic_sha256=cast(
                str, _object(receipt["configuration"], "configuration")["semantic_sha256"]
            ),
            cell_directories=owned_directories,
            output_dir=review_root / ".validator-unused-output",
        )
        _recompute_and_compare(receipt, admitted, owned_cells)
    else:
        config_v2, config_path = _load_retained_configuration_v2(
            review_root,
            receipt,
            root_members["admitted_runtime_review_config.json"],
        )
        owned_profiles = _parse_owned_profile_identities_v2(
            review_root,
            root_members,
            receipt["profiles"],
            config_v2,
        )
        admitted_v2 = AdmittedRuntimeReviewConfigurationV2(
            config=config_v2,
            path=config_path,
            base_dir=review_root,
            raw_bytes=root_members["admitted_runtime_review_config.json"],
            raw_sha256=cast(str, _object(receipt["configuration"], "configuration")["raw_sha256"]),
            semantic_sha256=cast(
                str, _object(receipt["configuration"], "configuration")["semantic_sha256"]
            ),
            cell_directories=owned_directories,
            profile_identity_paths=cast(
                tuple[Path, Path],
                tuple(review_root / item.locator for item in owned_profiles),
            ),
            profile_identity_file_sha256=cast(
                tuple[str, str], tuple(item.sha256 for item in owned_profiles)
            ),
            output_dir=review_root / ".validator-unused-output",
        )
        _recompute_and_compare_v2(receipt, admitted_v2, owned_cells, owned_profiles)
    for cell in owned_cells:
        validate_owned_evidence_cell(review_root, cell)
    if read_complete_owned_runtime_review_tree(review_root) != root_members:
        raise ValueError("runtime-review root documents changed during independent validation")
    return receipt


def _validate_document_schema(receipt: Mapping[str, CanonicalValue]) -> int:
    """Validate the closed receipt shape and every platform-neutral root invariant."""
    obj = _object(receipt, "runtime-review receipt")
    _fields(obj, set(RUNTIME_REVIEW_RECEIPT_KEYS), "runtime-review receipt")
    if obj["schema"] != RUNTIME_REVIEW_RECEIPT_SCHEMA:
        raise ValueError("unexpected runtime-review receipt schema")
    schema_version = _bounded_int(obj["schema_version"], "schema_version", 1, 2)
    if schema_version not in {
        RUNTIME_REVIEW_RECEIPT_SCHEMA_VERSION,
        RUNTIME_REVIEW_RECEIPT_SCHEMA_VERSION_V2,
    }:
        raise ValueError("unexpected runtime-review receipt schema_version")
    if obj["method"] != RUNTIME_REVIEW_METHOD:
        raise ValueError("runtime-review method differs from the frozen selected method")
    if obj["claim_scope"] != RUNTIME_REVIEW_CLAIM_SCOPE:
        raise ValueError("runtime-review claim scope differs from the frozen full-horizon claim")
    status = _status(obj["status"])
    reason = _reason(obj["reason_code"])
    if (status is RuntimeReviewStatus.INSUFFICIENT_EVIDENCE) != (reason is not None):
        raise ValueError("reason_code must be present exactly for INSUFFICIENT_EVIDENCE")
    _configuration_schema(obj["configuration"])
    _campaign_schema(obj["campaign_shape"])
    _decimal_token(obj["required_horizon"], "required_horizon")
    _decimal_token(obj["admitted_prefix"], "admitted_prefix")
    if Decimal(cast(str, obj["admitted_prefix"])) > Decimal(cast(str, obj["required_horizon"])):
        raise ValueError("admitted_prefix exceeds required_horizon")
    _gate_schema(obj["first_failing_gate"])
    witnesses = tuple(_witness(value) for value in _sequence(obj["witnesses"], "witnesses"))
    if [(Decimal(item.time), item.channel_id) for item in witnesses] != sorted(
        (Decimal(item.time), item.channel_id) for item in witnesses
    ):
        raise ValueError("witnesses are not in deterministic time/channel order")
    if len({(item.time, item.channel_id) for item in witnesses}) != len(witnesses):
        raise ValueError("witnesses contain a duplicate time/channel identity")
    _optional_witness(obj["first_decisive_witness"], witnesses, "first_decisive_witness")
    _optional_witness(obj["worst_witness"], witnesses, "worst_witness")
    _witness_counts(obj["witness_counts"], witnesses)
    if list(_sequence(obj["limitations"], "limitations")) != list(RUNTIME_REVIEW_LIMITATIONS):
        raise ValueError("runtime-review limitations differ from the frozen ordered registry")
    _tool_schema(obj["tool"], schema_version=schema_version)
    if schema_version == RUNTIME_REVIEW_RECEIPT_SCHEMA_VERSION_V2:
        _profiles_schema_v2(obj["profiles"])
    return schema_version


def _load_retained_configuration(
    review_root: Path,
    receipt: Mapping[str, CanonicalValue],
    raw: bytes,
) -> tuple[RuntimeReviewConfig, Path]:
    """Read the exact owned config bytes without resolving any original evidence locator."""
    block = _object(receipt["configuration"], "configuration")
    locator = _string(block["locator"], "configuration.locator")
    if locator != "admitted_runtime_review_config.json":
        raise ValueError("configuration locator must name the owned admitted configuration")
    path = review_root / locator
    if hashlib.sha256(raw).hexdigest() != block["raw_sha256"]:
        raise ValueError("owned configuration bytes do not match configuration.raw_sha256")
    primitive = bounded_strict_json_loads(raw, CONFIG_JSON_LIMITS)
    config = RuntimeReviewConfig.from_primitive(primitive)
    if canonical_sha256(config.to_primitive()) != block["semantic_sha256"]:
        raise ValueError("owned configuration semantics do not match configuration.semantic_sha256")
    return config, path


def _load_retained_configuration_v2(
    review_root: Path,
    receipt: Mapping[str, CanonicalValue],
    raw: bytes,
) -> tuple[RuntimeReviewConfigV2, Path]:
    """Decode exact owned role-based config bytes without resolving original cell paths."""
    block = _object(receipt["configuration"], "configuration")
    locator = _string(block["locator"], "configuration.locator")
    if locator != "admitted_runtime_review_config.json":
        raise ValueError("configuration locator must name the owned admitted configuration")
    path = review_root / locator
    if hashlib.sha256(raw).hexdigest() != block["raw_sha256"]:
        raise ValueError("owned configuration bytes do not match configuration.raw_sha256")
    primitive = bounded_strict_json_loads(raw, CONFIG_JSON_LIMITS)
    config = RuntimeReviewConfigV2.from_primitive(primitive)
    if canonical_sha256(config.to_primitive()) != block["semantic_sha256"]:
        raise ValueError("owned configuration semantics do not match configuration.semantic_sha256")
    return config, path


def _profiles_schema_v2(value: object) -> None:
    """Validate the closed role-based receipt profile projection."""
    profiles = _object(value, "profiles")
    _fields(profiles, {"baseline", "candidate", "common_environment"}, "profiles")
    _object(profiles["common_environment"], "profiles.common_environment")
    role_fields = {
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
        "identity_file",
    }
    for role in ("baseline", "candidate"):
        profile = _object(profiles[role], f"profiles.{role}")
        _fields(profile, role_fields, f"profiles.{role}")
        if profile["profile_role"] != role:
            raise ValueError(f"profiles.{role}.profile_role differs from its receipt role")
        _nonempty_string(profile["package_version"], f"profiles.{role}.package_version")
        _nonempty_string(profile["native_version"], f"profiles.{role}.native_version")
        _bounded_int(
            profile["native_version_integer"],
            f"profiles.{role}.native_version_integer",
            1,
            999_999_999,
        )
        require_sha256(
            profile["profile_identity_sha256"],
            f"profiles.{role}.profile_identity_sha256",
        )
        require_sha256(
            profile["runtime_identity_sha256"],
            f"profiles.{role}.runtime_identity_sha256",
        )
        for field in ("mujoco_distribution", "loaded_native_library", "numpy"):
            _object(profile[field], f"profiles.{role}.{field}")
        sentinel = _object(profile["sentinel"], f"profiles.{role}.sentinel")
        _fields(
            sentinel,
            {"status", "sentinel_identity_sha256"},
            f"profiles.{role}.sentinel",
        )
        if sentinel["status"] != "PASS":
            raise ValueError(f"profiles.{role}.sentinel must record PASS")
        require_sha256(
            sentinel["sentinel_identity_sha256"],
            f"profiles.{role}.sentinel.sentinel_identity_sha256",
        )
        if profile["support_tier"] != "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE":
            raise ValueError(
                f"profiles.{role}.support_tier must record current capability admission"
            )
        identity_file = _object(profile["identity_file"], f"profiles.{role}.identity_file")
        _fields(
            identity_file,
            {"locator", "raw_sha256", "size_bytes"},
            f"profiles.{role}.identity_file",
        )
        expected_locator = f"profile_identities/{role}.json"
        if identity_file["locator"] != expected_locator:
            raise ValueError(f"profiles.{role}.identity_file locator is not canonical")
        require_sha256(
            identity_file["raw_sha256"],
            f"profiles.{role}.identity_file.raw_sha256",
        )
        _bounded_int(
            identity_file["size_bytes"],
            f"profiles.{role}.identity_file.size_bytes",
            1,
            4 * 1024 * 1024,
        )


def _parse_owned_profile_identities_v2(
    review_root: Path,
    root_members: Mapping[str, bytes],
    value: CanonicalValue,
    configuration: RuntimeReviewConfigV2,
) -> tuple[OwnedProfileIdentity, OwnedProfileIdentity]:
    """Replay both owned profile identities and bind their authoritative projections."""
    profiles = _object(value, "profiles")
    declarations = (configuration.baseline_profile, configuration.candidate_profile)
    owned: list[OwnedProfileIdentity] = []
    for role, declaration in zip(("baseline", "candidate"), declarations, strict=True):
        profile = _object(profiles[role], f"profiles.{role}")
        identity_file = _object(profile["identity_file"], f"profiles.{role}.identity_file")
        locator = _string(identity_file["locator"], f"profiles.{role}.identity_file.locator")
        if locator != declaration.identity_file:
            raise ValueError(f"profiles.{role} identity locator differs from configuration")
        try:
            raw = root_members[locator]
        except KeyError as exc:
            raise ValueError(f"owned {role} profile identity is missing") from exc
        raw_sha256 = require_sha256(
            identity_file["raw_sha256"], f"profiles.{role}.identity_file.raw_sha256"
        )
        size_bytes = _bounded_int(
            identity_file["size_bytes"],
            f"profiles.{role}.identity_file.size_bytes",
            1,
            4 * 1024 * 1024,
        )
        if len(raw) != size_bytes or hashlib.sha256(raw).hexdigest() != raw_sha256:
            raise ValueError(f"owned {role} profile identity bytes differ from the receipt")
        identity = _load_profile_identity_authority_v2(
            review_root / locator,
            role,
            declaration.profile_identity_sha256,
        )
        if raw != canonical_json_bytes(cast(CanonicalValue, identity)) + b"\n":
            raise ValueError(f"owned {role} profile identity is not canonical JSON")
        projection = _profile_identity_receipt_projection_v2(identity)
        observed = {
            key: member
            for key, member in profile.items()
            if key not in {"identity_file", "runtime_identity_sha256"}
        }
        if projection != observed:
            raise ValueError(f"profiles.{role} differs from its owned profile identity")
        owned.append(
            OwnedProfileIdentity(
                profile_role=role,
                locator=PurePosixPath(locator),
                sha256=raw_sha256,
                size_bytes=size_bytes,
            )
        )
    return cast(tuple[OwnedProfileIdentity, OwnedProfileIdentity], tuple(owned))


def _load_profile_identity_authority_v2(
    path: Path, role: str, profile_identity_sha256: str
) -> dict[str, object]:
    """Invoke the v2 profile authority without importing it on the legacy route."""
    from . import _native_profile_identity

    loader = getattr(_native_profile_identity, "load_native_profile_identity_v2", None)
    if not callable(loader):
        raise RuntimeError("v2 profile identity authority is unavailable")
    identity = loader(
        path,
        expected_profile_role=role,
        expected_profile_identity_sha256=profile_identity_sha256,
    )
    if type(identity) is not dict:
        raise TypeError("v2 profile identity authority must return an object")
    return cast(dict[str, object], identity)


def _profile_identity_receipt_projection_v2(
    identity: Mapping[str, object],
) -> dict[str, CanonicalValue]:
    """Obtain the closed receipt projection from the v2 profile authority."""
    from . import _native_profile_identity

    projector = getattr(_native_profile_identity, "profile_identity_receipt_projection_v2", None)
    if not callable(projector):
        raise RuntimeError("v2 profile receipt projection authority is unavailable")
    projection = projector(identity)
    if type(projection) is not dict:
        raise TypeError("v2 profile receipt projection must be an object")
    canonical_sha256(cast(CanonicalValue, projection))
    return cast(dict[str, CanonicalValue], projection)


def _parse_owned_evidence_cells(value: CanonicalValue) -> tuple[OwnedEvidenceCell, ...]:
    """Decode the exact canonical slot-to-owned-cell map from the receipt."""
    raw_cells = _sequence(value, "evidence_cells")
    if len(raw_cells) != len(_CANONICAL_SLOTS):
        raise ValueError("evidence_cells must contain exactly twelve cells")
    cells: list[OwnedEvidenceCell] = []
    for index, (raw, slot) in enumerate(zip(raw_cells, _CANONICAL_SLOTS, strict=True)):
        obj = _object(raw, f"evidence_cells[{index}]")
        _fields(
            obj,
            {"profile_role", "step_dt", "repeat_id", "directory", "members"},
            f"evidence_cells[{index}]",
        )
        role, step_dt, repeat_id = slot
        if (
            obj["profile_role"] != role
            or obj["step_dt"] != step_dt
            or obj["repeat_id"] != repeat_id
        ):
            raise ValueError("evidence_cells are not the complete canonical slot order")
        expected_locator = PurePosixPath(
            "evidence", role, _STEP_TOKENS[step_dt], f"repeat_{repeat_id}"
        )
        if obj["directory"] != expected_locator.as_posix():
            raise ValueError("an owned evidence locator differs from its canonical slot")
        members = _parse_owned_members(obj["members"], index)
        cells.append(OwnedEvidenceCell(role, step_dt, repeat_id, expected_locator, members))
    return tuple(cells)


def _parse_owned_members(value: object, cell_index: int) -> tuple[OwnedEvidenceMember, ...]:
    """Decode all six byte identities for one owned evidence cell."""
    raw_members = _sequence(value, f"evidence_cells[{cell_index}].members")
    if len(raw_members) != len(_CELL_MEMBERS):
        raise ValueError("an owned evidence cell must record exactly six members")
    members: list[OwnedEvidenceMember] = []
    for raw, expected_name in zip(raw_members, _CELL_MEMBERS, strict=True):
        obj = _object(raw, "owned evidence member")
        _fields(obj, {"name", "sha256", "size_bytes"}, "owned evidence member")
        if obj["name"] != expected_name:
            raise ValueError("owned evidence members are not in canonical order")
        members.append(
            OwnedEvidenceMember(
                expected_name,
                require_sha256(obj["sha256"], f"{expected_name} SHA-256"),
                _nonnegative_int(obj["size_bytes"], f"{expected_name} size_bytes"),
            )
        )
    return tuple(members)


def _recompute_and_compare(
    receipt: Mapping[str, CanonicalValue],
    configuration: AdmittedRuntimeReviewConfiguration,
    owned_cells: Sequence[OwnedEvidenceCell],
) -> None:
    """Re-admit owned raw traces, rerun the method, and compare stable decision facts."""
    from ._decision import evaluate_runtime_evidence
    from ._evidence import admit_runtime_evidence

    evidence = admit_runtime_evidence(
        configuration,
        cell_directories=configuration.cell_directories,
    )
    decision = evaluate_runtime_evidence(evidence)
    expected: dict[str, CanonicalValue] = {
        "status": decision.status.value,
        "reason_code": decision.reason_code.value if decision.reason_code is not None else None,
        "profiles": dict(evidence.profiles),
        "subject": dict(evidence.subject),
        "workload": dict(evidence.workload),
        "required_horizon": configuration.config.required_horizon,
        "admitted_prefix": decision.admitted_prefix,
        "first_failing_gate": (
            None if decision.first_failing_gate is None else dict(decision.first_failing_gate)
        ),
        "first_decisive_witness": (
            None
            if decision.first_decisive_witness is None
            else decision.first_decisive_witness.to_primitive()
        ),
        "worst_witness": (
            None if decision.worst_witness is None else decision.worst_witness.to_primitive()
        ),
        "witness_counts": dict(decision.witness_counts),
        "witnesses": [item.to_primitive() for item in decision.witnesses],
        "evidence_cells": [item.to_primitive() for item in owned_cells],
    }
    for field, recomputed in expected.items():
        if receipt[field] != recomputed:
            raise ValueError(
                f"receipt {field} does not match the decision independently rebuilt from owned evidence"
            )


def _recompute_and_compare_v2(
    receipt: Mapping[str, CanonicalValue],
    configuration: AdmittedRuntimeReviewConfigurationV2,
    owned_cells: Sequence[OwnedEvidenceCell],
    owned_profiles: Sequence[OwnedProfileIdentity],
) -> None:
    """Rebuild one role-based decision and its portable profile/sentinel bindings."""
    from ._decision import evaluate_runtime_evidence
    from ._evidence import admit_runtime_evidence

    evidence = admit_runtime_evidence(
        configuration,
        cell_directories=configuration.cell_directories,
    )
    decision = evaluate_runtime_evidence(evidence)
    profiles = _profiles_primitive_v2(
        configuration,
        evidence.profiles,
        owned_profiles,
    )
    expected: dict[str, CanonicalValue] = {
        "status": decision.status.value,
        "reason_code": decision.reason_code.value if decision.reason_code is not None else None,
        "profiles": profiles,
        "subject": dict(evidence.subject),
        "workload": dict(evidence.workload),
        "required_horizon": configuration.config.required_horizon,
        "admitted_prefix": decision.admitted_prefix,
        "first_failing_gate": (
            None if decision.first_failing_gate is None else dict(decision.first_failing_gate)
        ),
        "first_decisive_witness": (
            None
            if decision.first_decisive_witness is None
            else decision.first_decisive_witness.to_primitive()
        ),
        "worst_witness": (
            None if decision.worst_witness is None else decision.worst_witness.to_primitive()
        ),
        "witness_counts": dict(decision.witness_counts),
        "witnesses": [item.to_primitive() for item in decision.witnesses],
        "evidence_cells": [item.to_primitive() for item in owned_cells],
    }
    for field, recomputed in expected.items():
        if receipt[field] != recomputed:
            raise ValueError(
                f"receipt {field} does not match the role-based decision rebuilt from owned evidence"
            )


def _configuration_schema(value: object) -> None:
    """Validate the owned configuration locator and both exact identities."""
    obj = _object(value, "configuration")
    _fields(obj, {"locator", "raw_sha256", "semantic_sha256"}, "configuration")
    if _string(obj["locator"], "configuration.locator") != "admitted_runtime_review_config.json":
        raise ValueError("configuration locator differs from the frozen owned name")
    require_sha256(obj["raw_sha256"], "configuration.raw_sha256")
    require_sha256(obj["semantic_sha256"], "configuration.semantic_sha256")


def _campaign_schema(value: object) -> None:
    """Validate the exact three-grid/two-repeat campaign shape."""
    obj = _object(value, "campaign_shape")
    _fields(obj, {"step_dts", "repeat_ids", "cell_count"}, "campaign_shape")
    if _sequence(obj["step_dts"], "campaign_shape.step_dts") != ["0.004", "0.002", "0.001"]:
        raise ValueError("campaign_shape.step_dts differs from the frozen grid")
    if _sequence(obj["repeat_ids"], "campaign_shape.repeat_ids") != [0, 1]:
        raise ValueError("campaign_shape.repeat_ids differs from the frozen repeats")
    if _bounded_int(obj["cell_count"], "campaign_shape.cell_count", 12, 12) != 12:
        raise ValueError("campaign_shape.cell_count must be twelve")


def _witness(value: object) -> StableWitness:
    """Decode one exact stable decision witness."""
    obj = _object(value, "witness")
    _fields(obj, _WITNESS_FIELDS, "witness")
    time = _decimal_token(obj["time"], "witness.time")
    tolerance = _decimal_token(obj["tolerance"], "witness.tolerance")
    if Decimal(tolerance) <= 0:
        raise ValueError("witness.tolerance must be greater than zero")
    return StableWitness(
        channel_id=_nonempty_string(obj["channel_id"], "witness.channel_id"),
        classification=_nonempty_string(obj["classification"], "witness.classification"),
        kind=_nonempty_string(obj["kind"], "witness.kind"),
        semantic_type=_nonempty_string(obj["semantic_type"], "witness.semantic_type"),
        time=time,
        tolerance=tolerance,
        decision_input_sha256=require_sha256(
            obj["decision_input_sha256"], "witness.decision_input_sha256"
        ),
    )


def _optional_witness(
    value: object,
    witnesses: Sequence[StableWitness],
    field: str,
) -> None:
    """Require an optional named witness to equal one row in the complete stable list."""
    if value is None:
        return
    parsed = _witness(value)
    if parsed not in witnesses:
        raise ValueError(f"{field} is not one of the complete stable witnesses")


def _witness_counts(value: object, witnesses: Sequence[StableWitness]) -> None:
    """Recompute all three classification counts from the complete witness array."""
    obj = _object(value, "witness_counts")
    _fields(obj, set(_WITNESS_CLASSIFICATIONS), "witness_counts")
    expected = {
        classification: sum(item.classification == classification for item in witnesses)
        for classification in _WITNESS_CLASSIFICATIONS
    }
    actual = {
        classification: _nonnegative_int(obj[classification], f"witness_counts.{classification}")
        for classification in _WITNESS_CLASSIFICATIONS
    }
    if actual != expected:
        raise ValueError("witness_counts do not count the complete witness array")


def _gate_schema(value: object) -> None:
    """Validate one optional stable first-failing-gate object."""
    if value is None:
        return
    obj = _object(value, "first_failing_gate")
    _fields(obj, {"channel_id", "detail", "status", "time"}, "first_failing_gate")
    _nonempty_string(obj["channel_id"], "first_failing_gate.channel_id")
    _string(obj["detail"], "first_failing_gate.detail")
    status = _nonempty_string(obj["status"], "first_failing_gate.status")
    if status not in {code.value for code in RuntimeReviewReasonCode} - {
        RuntimeReviewReasonCode.PREFIX_TOO_SHORT.value
    }:
        raise ValueError("first_failing_gate status is outside the frozen gate registry")
    _decimal_token(obj["time"], "first_failing_gate.time")


def _tool_schema(value: object, *, schema_version: int) -> None:
    """Validate current v2 or self-consistent historical v1 tool provenance."""
    obj = _object(value, "tool")
    _fields(
        obj,
        {
            "metrifid_version",
            "distribution_identity",
            "distribution_identity_sha256",
            "evaluator",
        },
        "tool",
    )
    metrifid_version = _nonempty_string(obj["metrifid_version"], "tool.metrifid_version")
    distribution = _object(obj["distribution_identity"], "tool.distribution_identity")
    if canonical_sha256(cast(CanonicalValue, distribution)) != require_sha256(
        obj["distribution_identity_sha256"], "tool.distribution_identity_sha256"
    ):
        raise ValueError("tool distribution identity SHA-256 does not match its complete object")
    if distribution.get("distribution_version") != metrifid_version:
        raise ValueError("tool distribution identity differs from its recorded product version")
    if (
        schema_version == RUNTIME_REVIEW_RECEIPT_SCHEMA_VERSION_V2
        and metrifid_version != __version__
    ):
        raise ValueError("v2 receipt tool version differs from the validating product version")
    evaluator = _object(obj["evaluator"], "tool.evaluator")
    _fields(
        evaluator,
        {
            "python_implementation",
            "python_version",
            "python_cache_tag",
            "platform",
            "system",
            "machine",
        },
        "tool.evaluator",
    )
    for field, member in evaluator.items():
        _string(member, f"tool.evaluator.{field}")


def _status(value: object) -> RuntimeReviewStatus:
    """Decode one completed status from the closed public enum."""
    try:
        return RuntimeReviewStatus(_nonempty_string(value, "status"))
    except ValueError as exc:
        raise ValueError("status is outside the frozen runtime-review registry") from exc


def _reason(value: object) -> RuntimeReviewReasonCode | None:
    """Decode one optional insufficient-evidence reason from the closed public enum."""
    if value is None:
        return None
    try:
        return RuntimeReviewReasonCode(_nonempty_string(value, "reason_code"))
    except ValueError as exc:
        raise ValueError("reason_code is outside the frozen runtime-review registry") from exc


def _decimal_token(value: object, field: str) -> str:
    """Require one canonical nonnegative ordinary-decimal token."""
    if type(value) is not str:
        raise TypeError(f"{field} must be a string")
    token = value
    try:
        ExactRational.from_decimal_token(token)
        decimal = Decimal(token)
    except (ValueError, InvalidOperation) as exc:
        raise ValueError(f"{field} must be a canonical decimal token") from exc
    if decimal < 0:
        raise ValueError(f"{field} must be nonnegative")
    return token


__all__ = ["load_and_validate_runtime_review_receipt", "validate_runtime_review_tree"]
