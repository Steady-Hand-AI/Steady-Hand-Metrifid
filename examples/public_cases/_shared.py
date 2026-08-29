"""Shared execution for the complaint-backed public case gallery.

Every case runs the same three-stage installed Metrifid journey — Certify, an unbound-candidate
Model Review discovery pass, and a candidate-bound declared review built from that discovery — and
then one case-specific direct-MuJoCo control that observes the compiled mechanism the complaint
described.

The generated ALLOW policy is a mechanical demonstration of the two-pass workflow. It is not release
authority: a human maintainer must inspect and justify every declared rule before a policy like this
one is used to approve a real model change.

Nothing here calls a subprocess, opens a socket, installs a package, discovers an environment, or
registers a callback.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import mujoco  # type: ignore[import-untyped]
import numpy as np

import metrifid
from metrifid.certify import (
    CertifyStatus,
    certify_models,
    load_and_validate_certification_receipt,
)
from metrifid.model_release import (
    ModelReleaseStatus,
    load_and_validate_model_release_receipt,
    review_model_release,
)

CASE_RESULT_SCHEMA = "metrifid.public_case_result"
CASE_RESULT_SCHEMA_VERSION = 1
POLICY_SCHEMA = "metrifid.model_release_policy"
POLICY_SCHEMA_VERSION = 1
_COMMON_EXPECTED_PRODUCT: Mapping[str, object] = MappingProxyType(
    {
        "certify_status": CertifyStatus.NOT_CERTIFIED_COMPILED_DIFFERS.value,
        "certify_exit_code": 40,
        "discovery_status": ModelReleaseStatus.REVIEW_REQUIRED.value,
        "discovery_exit_code": 40,
    }
)
FROZEN_EXPECTED_PRODUCT_BY_CASE: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        case_id: MappingProxyType(
            {
                **_COMMON_EXPECTED_PRODUCT,
                "declared_status": declared_status,
                "declared_exit_code": declared_exit_code,
            }
        )
        for case_id, declared_status, declared_exit_code in (
            (
                "collision_filtering.mask_flattening",
                ModelReleaseStatus.WITHIN_DECLARED_POLICY.value,
                0,
            ),
            (
                "collision_filtering.explicit_pair_loss",
                ModelReleaseStatus.REVIEW_REQUIRED.value,
                40,
            ),
            (
                "collision_filtering.exclusion_loss",
                ModelReleaseStatus.REVIEW_REQUIRED.value,
                40,
            ),
            (
                "mesh_inertia.mode_change",
                ModelReleaseStatus.WITHIN_DECLARED_POLICY.value,
                0,
            ),
            (
                "actuator_transmission.frame_change",
                ModelReleaseStatus.WITHIN_DECLARED_POLICY.value,
                0,
            ),
            (
                "sensor_attachment.site_change",
                ModelReleaseStatus.WITHIN_DECLARED_POLICY.value,
                0,
            ),
        )
    }
)
_OPAQUE_OBJECT_TYPE = "opaque"
_COMPILED_FIELD_OBJECT_TYPE = "compiled_field"
_SELECTOR_FIELDS = ("object_type", "object_name", "field", "change_kind")
_CHECKSUM_MANIFEST_NAME = "CHECKSUMS.sha256"


def canonical_json_bytes(value: object) -> bytes:
    """Encode one value as the gallery's canonical JSON bytes.

    Args:
        value: A JSON-compatible value. Raw ``NaN`` and ``Infinity`` are refused.

    Returns:
        UTF-8 bytes with sorted keys, tight separators, and a trailing newline.
    """
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return text.encode("utf-8") + b"\n"


def write_new_bytes(path: Path, payload: bytes) -> None:
    """Write one new file, refusing to replace anything that already exists.

    Args:
        path: Destination file that must not already exist.
        payload: Exact bytes to write.

    Raises:
        RuntimeError: If the destination already exists.
    """
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite an existing gallery path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_new_json(path: Path, value: object) -> None:
    """Write one new canonical JSON document.

    Args:
        path: Destination file that must not already exist.
        value: JSON-compatible value to encode canonically.
    """
    write_new_bytes(path, canonical_json_bytes(value))


def require_absent_directory(path: Path) -> Path:
    """Resolve one output root that must not exist yet, and create it.

    An existing empty directory is refused too, so a gallery run never merges into, reuses, or
    deletes anything a caller already had.

    The created root is returned fully resolved. Metrifid opens its output directories one component
    at a time and refuses to follow a symbolic link, so a caller who names a path through a symlinked
    parent must still hand the product the real path.

    Args:
        path: Requested output root.

    Returns:
        The created output root, absolute and free of symbolic-link components.

    Raises:
        RuntimeError: If the path already exists.
    """
    requested = path.expanduser().absolute()
    if requested.exists() or requested.is_symlink():
        raise RuntimeError(f"--output must name an absent path: {requested}")
    requested.mkdir(parents=True)
    return requested.resolve(strict=True)


def load_case_manifest(case_directory: Path) -> dict[str, object]:
    """Read and lightly admit one frozen case manifest.

    Args:
        case_directory: Directory holding ``case_manifest.json``.

    Returns:
        The manifest mapping.

    Raises:
        RuntimeError: If the manifest is missing, malformed, or not the expected schema.
    """
    manifest_path = case_directory / "case_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"case manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError(f"case manifest is not an object: {manifest_path}")
    if manifest.get("schema") != "metrifid.public_case":
        raise RuntimeError(f"case manifest has an unexpected schema: {manifest_path}")
    if manifest.get("schema_version") != 1:
        raise RuntimeError(f"case manifest has an unexpected schema version: {manifest_path}")
    return manifest


def build_exact_allow_rules(
    discovery_receipt: Mapping[str, object],
) -> list[dict[str, object]]:
    """Turn every non-opaque discovery row into one exact ALLOW rule, in receipt order.

    This mirrors the accepted two-pass Model Review example: nothing is collapsed, broadened, or
    inferred, and the before/after digests are copied exactly so the declared policy constrains the
    same values the discovery pass observed.

    Args:
        discovery_receipt: A validated discovery receipt.

    Returns:
        One ALLOW rule per retained row, numbered ``allow-0001`` upward.

    Raises:
        RuntimeError: If the receipt shape is wrong, a retained row is not ``UNDECLARED``, or no
            non-opaque row exists.
    """
    raw_changes = discovery_receipt.get("changes")
    if not isinstance(raw_changes, list):
        raise RuntimeError("discovery receipt changes is not an array")
    rules: list[dict[str, object]] = []
    for index, raw_change in enumerate(raw_changes):
        if not isinstance(raw_change, Mapping):
            raise RuntimeError(f"discovery changes[{index}] is not an object")
        selector = raw_change.get("selector")
        if not isinstance(selector, Mapping):
            raise RuntimeError(f"discovery changes[{index}].selector is not an object")
        if selector.get("object_type") == _OPAQUE_OBJECT_TYPE:
            continue
        if raw_change.get("classification") != "UNDECLARED":
            raise RuntimeError(
                f"discovery changes[{index}] is not UNDECLARED; the discovery pass must leave "
                "every non-opaque row undeclared"
            )
        exact_selector: dict[str, object] = {}
        for field in _SELECTOR_FIELDS:
            value = selector.get(field)
            if not isinstance(value, str):
                raise RuntimeError(f"discovery changes[{index}].selector.{field} is not a string")
            exact_selector[field] = value
        rules.append(
            {
                "id": f"allow-{len(rules) + 1:04d}",
                "effect": "ALLOW",
                "selector": exact_selector,
                "before_sha256": _optional_digest(raw_change.get("before_sha256"), index, "before"),
                "after_sha256": _optional_digest(raw_change.get("after_sha256"), index, "after"),
            }
        )
    if not rules:
        raise RuntimeError("the discovery pass produced no explained, non-opaque change to declare")
    return rules


class CaseExpectationError(RuntimeError):
    """One case completed every stage but a decision differed from the frozen expectation.

    This is raised only after the case has published its complete evidence, so a caller can record
    the divergence, keep running the remaining cases, and still hand a reviewer every receipt.
    """

    def __init__(self, case_id: str, mismatches: Sequence[str]) -> None:
        self.case_id = case_id
        self.mismatches = tuple(mismatches)
        super().__init__(f"{case_id}: " + "; ".join(mismatches))


def _require_frozen_expectation(case_id: str, expected: Mapping[str, object]) -> None:
    """Require the case manifest to declare exactly the frozen product journey.

    The journey is frozen by the public case specification, not by the manifest. Checking the
    manifest against the specification first means a tampered or drifted manifest cannot quietly
    redefine what the gallery considers a pass.
    """
    frozen = FROZEN_EXPECTED_PRODUCT_BY_CASE.get(case_id)
    if frozen is None:
        raise RuntimeError(f"{case_id}: no frozen product journey is registered")
    if dict(expected) != dict(frozen):
        raise RuntimeError(
            f"{case_id}: case_manifest.expected_product is not its frozen product journey "
            f"{frozen!r}"
        )


def _expectation_mismatches(
    case_id: str, product_results: Mapping[str, Mapping[str, object]]
) -> list[str]:
    """Return one message per stage whose observed decision differs from this case's contract."""
    expected = FROZEN_EXPECTED_PRODUCT_BY_CASE[case_id]
    mismatches: list[str] = []
    for stage, status_key, exit_key in (
        ("certify", "certify_status", "certify_exit_code"),
        ("discovery", "discovery_status", "discovery_exit_code"),
        ("declared", "declared_status", "declared_exit_code"),
    ):
        row = product_results[stage]
        wanted_status = expected[status_key]
        wanted_exit = expected[exit_key]
        if row["status"] != wanted_status or row["completed_exit_code"] != wanted_exit:
            mismatches.append(
                f"{stage} decided {row['status']} (exit {row['completed_exit_code']}) "
                f"instead of the frozen {wanted_status} (exit {wanted_exit})"
            )
    return mismatches


def _undeclared_residual_reasons(receipt: Mapping[str, object]) -> list[str]:
    """Return the residual reasons the product published for an undeclared opaque row, if any."""
    changes = receipt.get("changes")
    if not isinstance(changes, list):
        return []
    for raw_change in changes:
        if not isinstance(raw_change, Mapping):
            continue
        selector = raw_change.get("selector")
        if not isinstance(selector, Mapping):
            continue
        if selector.get("object_type") != _OPAQUE_OBJECT_TYPE:
            continue
        if raw_change.get("classification") != "UNDECLARED":
            continue
        details = raw_change.get("details")
        if isinstance(details, Mapping):
            reasons = details.get("reasons")
            if isinstance(reasons, list):
                return [str(reason) for reason in reasons]
    return []


def execute_case(case_directory: Path, output_directory: Path) -> dict[str, object]:
    """Run the complete three-stage product journey and control for one case.

    Every stage runs and every artifact is published before any expectation is judged, so a case
    whose decision diverges still leaves a reviewer the complete receipts that show why.

    Args:
        case_directory: Tracked case directory holding the manifest and both model roots.
        output_directory: Absent per-case output directory to publish into.

    Returns:
        The canonical case result mapping that was written to ``case_result.json``.

    Raises:
        CaseExpectationError: Every stage completed but a decision differed from the frozen journey.
        RuntimeError: A receipt failed validation, a required field subset or semantic selector was
            absent, or the independent control broke its contract.
    """
    manifest = load_case_manifest(case_directory)
    case_id = _require_text(manifest.get("case_id"), "case_manifest.case_id")
    expected = manifest.get("expected_product")
    if not isinstance(expected, Mapping):
        raise RuntimeError(f"{case_id}: case_manifest.expected_product is not an object")
    _require_frozen_expectation(case_id, expected)

    baseline = case_directory / _require_text(
        manifest.get("baseline_entrypoint"), "case_manifest.baseline_entrypoint"
    )
    candidate = case_directory / _require_text(
        manifest.get("candidate_entrypoint"), "case_manifest.candidate_entrypoint"
    )
    output_directory.mkdir(parents=True, exist_ok=False)

    certify_models(
        str(baseline),
        str(candidate),
        str(output_directory / "certification"),
        baseline_root=str(baseline.parent),
        candidate_root=str(candidate.parent),
    )
    certify_receipt_path = output_directory / "certification" / "certification.json"
    certify_receipt = load_and_validate_certification_receipt(certify_receipt_path.read_bytes())
    baseline_mjb = _mjb_sha256(certify_receipt, "baseline")
    candidate_mjb = _mjb_sha256(certify_receipt, "candidate")

    discovery_policy = output_directory / "discovery-policy.json"
    write_new_json(discovery_policy, _policy(baseline_mjb, None, []))
    review_model_release(
        str(baseline),
        str(candidate),
        str(discovery_policy),
        str(output_directory / "discovery-review"),
        baseline_root=str(baseline.parent),
        candidate_root=str(candidate.parent),
    )
    discovery_receipt_path = output_directory / "discovery-review" / "model_release.json"
    discovery_receipt = load_and_validate_model_release_receipt(discovery_receipt_path.read_bytes())

    rules = build_exact_allow_rules(discovery_receipt)
    declared_policy = output_directory / "declared-policy.json"
    write_new_json(declared_policy, _policy(baseline_mjb, candidate_mjb, rules))
    review_model_release(
        str(baseline),
        str(candidate),
        str(declared_policy),
        str(output_directory / "declared-review"),
        baseline_root=str(baseline.parent),
        candidate_root=str(candidate.parent),
    )
    declared_receipt_path = output_directory / "declared-review" / "model_release.json"
    declared_receipt = load_and_validate_model_release_receipt(declared_receipt_path.read_bytes())

    control = run_independent_control(case_id, case_directory)
    write_new_json(output_directory / "independent_control.json", control)

    changed_fields, selectors = _discovery_projection(discovery_receipt)
    _require_subset(
        case_id,
        "required_compiled_fields",
        manifest.get("required_compiled_fields"),
        changed_fields,
    )
    _require_selectors(case_id, manifest.get("required_semantic_selectors"), selectors)

    product_results = {
        "certify": _product_row(certify_receipt, certify_receipt_path, output_directory),
        "discovery": _product_row(discovery_receipt, discovery_receipt_path, output_directory),
        "declared": _product_row(declared_receipt, declared_receipt_path, output_directory),
    }

    result = {
        "schema": CASE_RESULT_SCHEMA,
        "schema_version": CASE_RESULT_SCHEMA_VERSION,
        "case_id": case_id,
        "runtime_identity": _runtime_identity(),
        "source_identity": {
            "baseline_source_closure_sha256": _closure_sha256(certify_receipt, "baseline"),
            "candidate_source_closure_sha256": _closure_sha256(certify_receipt, "candidate"),
            "baseline_complete_mjb_sha256": baseline_mjb,
            "candidate_complete_mjb_sha256": candidate_mjb,
        },
        "product_results": product_results,
        "changed_public_fields": changed_fields,
        "semantic_selectors": selectors,
        "independent_control": control,
        "claim_boundary": _require_text(
            manifest.get("claim_boundary"), "case_manifest.claim_boundary"
        ),
    }
    write_new_json(output_directory / "case_result.json", result)

    mismatches = _expectation_mismatches(case_id, product_results)
    if mismatches:
        reasons = _undeclared_residual_reasons(declared_receipt)
        if reasons:
            mismatches.append(
                "the declared review left an undeclared opaque complete-MJB residual for "
                f"{reasons}, which no policy rule can clear"
            )
        raise CaseExpectationError(case_id, mismatches)
    return result


def run_independent_control(case_id: str, case_directory: Path) -> dict[str, object]:
    """Run the one direct-MuJoCo control that observes this case's compiled mechanism.

    These controls deliberately bypass Metrifid so the gallery has an independent observation of the
    mechanism beside the product decision.

    Args:
        case_id: Frozen case identifier.
        case_directory: Tracked case directory.

    Returns:
        The control record, always containing at least ``kind`` and ``classification``.

    Raises:
        RuntimeError: If the case identifier is unknown or the control contract fails.
    """
    manifest = load_case_manifest(case_directory)
    declared = manifest.get("independent_control")
    if not isinstance(declared, Mapping):
        raise RuntimeError(f"{case_id}: case_manifest.independent_control is not an object")
    baseline = case_directory / _require_text(
        manifest.get("baseline_entrypoint"), "case_manifest.baseline_entrypoint"
    )
    candidate = case_directory / _require_text(
        manifest.get("candidate_entrypoint"), "case_manifest.candidate_entrypoint"
    )
    controls = {
        "collision_filtering.mask_flattening": _control_mask_eligibility,
        "collision_filtering.explicit_pair_loss": _control_explicit_pairs,
        "collision_filtering.exclusion_loss": _control_exclusions,
        "mesh_inertia.mode_change": _control_mass_properties,
        "actuator_transmission.frame_change": _control_actuator_moment,
        "sensor_attachment.site_change": _control_sensor_attachment,
    }
    runner = controls.get(case_id)
    if runner is None:
        raise RuntimeError(f"no independent control is defined for case {case_id!r}")
    control = runner(baseline, candidate, declared, case_directory)
    expected = declared.get("expected")
    if control["classification"] != expected:
        raise RuntimeError(
            f"{case_id}: independent control classified {control['classification']!r}, but the "
            f"case manifest requires {expected!r}"
        )
    return control


def write_checksum_manifest(root: Path) -> Path:
    """Write one SHA-256 manifest covering every regular file below a published root.

    Args:
        root: Published output root.

    Returns:
        The manifest path.
    """
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == _CHECKSUM_MANIFEST_NAME:
            continue
        rows.append(f"{_file_sha256(path)}  {relative}")
    rows.sort(key=lambda row: row.split("  ", 1)[1].encode("utf-8"))
    manifest = root / _CHECKSUM_MANIFEST_NAME
    write_new_bytes(manifest, ("\n".join(rows) + "\n").encode("utf-8"))
    return manifest


# --- private helpers ---------------------------------------------------------------------------


def _optional_digest(value: object, index: int, edge: str) -> str | None:
    """Return one discovery digest constraint that is either an exact string or null."""
    if value is None or isinstance(value, str):
        return value
    raise RuntimeError(f"discovery changes[{index}].{edge}_sha256 is not a string or null")


def _require_text(value: object, label: str) -> str:
    """Return one required string field or fail with the field name."""
    if not isinstance(value, str):
        raise RuntimeError(f"{label} is not a string")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    """Return one required nested receipt object or fail with the field name."""
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} is not an object")
    return value


def _policy(
    baseline_sha256: str, candidate_sha256: str | None, rules: list[dict[str, object]]
) -> dict[str, object]:
    """Build one complete public model-release policy document."""
    return {
        "schema": POLICY_SCHEMA,
        "schema_version": POLICY_SCHEMA_VERSION,
        "baseline_compiled_sha256": baseline_sha256,
        "candidate_compiled_sha256": candidate_sha256,
        "rules": rules,
    }


def _mjb_sha256(certification_receipt: Mapping[str, object], role: str) -> str:
    """Read one exact complete-MJB identity from a validated Certify receipt."""
    role_receipt = _mapping(certification_receipt.get(role), f"certification.{role}")
    artifact = _mapping(
        role_receipt.get("compiled_artifact"), f"certification.{role}.compiled_artifact"
    )
    return _require_text(
        artifact.get("mjb_sha256"), f"certification.{role}.compiled_artifact.mjb_sha256"
    )


def _closure_sha256(certification_receipt: Mapping[str, object], role: str) -> str:
    """Read one exact source-closure identity from a validated Certify receipt."""
    role_receipt = _mapping(certification_receipt.get(role), f"certification.{role}")
    return _require_text(
        role_receipt.get("source_closure_sha256"),
        f"certification.{role}.source_closure_sha256",
    )


def _product_row(
    receipt: Mapping[str, object], receipt_path: Path, output_directory: Path
) -> dict[str, object]:
    """Project one validated receipt into the frozen per-stage result row."""
    return {
        "status": _require_text(receipt.get("status"), "receipt.status"),
        "completed_exit_code": _require_int(
            receipt.get("completed_exit_code"), "receipt.completed_exit_code"
        ),
        "receipt_sha256": _require_text(receipt.get("receipt_sha256"), "receipt.receipt_sha256"),
        "relative_receipt_path": receipt_path.relative_to(output_directory).as_posix(),
    }


def _require_int(value: object, label: str) -> int:
    """Return one required integer field, refusing bool."""
    if type(value) is not int:
        raise RuntimeError(f"{label} is not an integer")
    return value


def _discovery_projection(
    discovery_receipt: Mapping[str, object],
) -> tuple[list[str], list[dict[str, str]]]:
    """Return the changed public field names and the exact selectors from a discovery receipt.

    A compiled public field row names the field itself in ``selector.object_name``; its
    ``selector.field`` is the constant ``value``, because the whole compiled array is the subject.
    The selector list keeps every non-opaque row — compiled and semantic alike — so it is exactly the
    set that :func:`build_exact_allow_rules` turns into declared rules.
    """
    raw_changes = discovery_receipt.get("changes")
    if not isinstance(raw_changes, list):
        raise RuntimeError("discovery receipt changes is not an array")
    fields: set[str] = set()
    selectors: dict[str, dict[str, str]] = {}
    for index, raw_change in enumerate(raw_changes):
        selector = _mapping(raw_change, f"changes[{index}]").get("selector")
        selector = _mapping(selector, f"changes[{index}].selector")
        if selector.get("object_type") == _OPAQUE_OBJECT_TYPE:
            continue
        exact = {
            field: _require_text(selector.get(field), f"selector.{field}")
            for field in _SELECTOR_FIELDS
        }
        if exact["object_type"] == _COMPILED_FIELD_OBJECT_TYPE:
            fields.add(exact["object_name"])
        selectors[json.dumps(exact, sort_keys=True)] = exact
    ordered = [selectors[key] for key in sorted(selectors)]
    return sorted(fields), ordered


def _require_subset(case_id: str, label: str, required: object, observed: Sequence[str]) -> None:
    """Require every manifest-declared public field to appear in the discovery projection."""
    if not isinstance(required, list):
        raise RuntimeError(f"{case_id}: case_manifest.{label} is not an array")
    missing = [name for name in required if name not in observed]
    if missing:
        raise RuntimeError(
            f"{case_id}: discovery did not report required {label} {missing!r}; observed "
            f"{list(observed)!r}"
        )


def _require_selectors(
    case_id: str, required: object, observed: Sequence[Mapping[str, str]]
) -> None:
    """Require every manifest-declared semantic selector to appear in the discovery projection."""
    if not isinstance(required, list):
        raise RuntimeError(f"{case_id}: case_manifest.required_semantic_selectors is not an array")
    seen = {json.dumps(dict(item), sort_keys=True) for item in observed}
    missing = [item for item in required if json.dumps(dict(item), sort_keys=True) not in seen]
    if missing:
        raise RuntimeError(
            f"{case_id}: discovery did not report required semantic selectors {missing!r}"
        )


def _runtime_identity() -> dict[str, object]:
    """Record the exact runtime this gallery run observed."""
    return {
        "metrifid_version": metrifid.__version__,
        "mujoco_package_version": mujoco.__version__,
        "mujoco_native_version": mujoco.mj_versionString(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _file_sha256(path: Path) -> str:
    """Return the lowercase hexadecimal SHA-256 of one file's exact bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _float_tokens(values: Sequence[float]) -> dict[str, list[str]]:
    """Return canonical decimal and binary64 hexadecimal tokens for finite measured values."""
    numbers = [float(value) for value in values]
    for number in numbers:
        if not np.isfinite(number):
            raise RuntimeError("an independent control measured a non-finite value")
    return {
        "decimal": [repr(number) for number in numbers],
        "binary64_hex": [number.hex() for number in numbers],
    }


def _compile(entrypoint: Path) -> Any:
    """Compile one case model directly through MuJoCo for an independent control."""
    return mujoco.MjModel.from_xml_path(str(entrypoint))


def _authored_masks(entrypoint: Path) -> dict[str, dict[str, int]]:
    """Read authored contype/conaffinity integers for every named geom in one case model."""
    import xml.etree.ElementTree as ElementTree

    root = ElementTree.parse(entrypoint).getroot()
    masks: dict[str, dict[str, int]] = {}
    for geom in root.iter("geom"):
        name = geom.get("name")
        if name is None:
            continue
        masks[name] = {
            "contype": int(geom.get("contype", "1")),
            "conaffinity": int(geom.get("conaffinity", "1")),
        }
    return masks


def _eligible_pairs(masks: Mapping[str, Mapping[str, int]]) -> list[list[str]]:
    """Return the sorted unordered geom pairs that the authored masks make collision-eligible."""
    names = sorted(masks)
    pairs: list[list[str]] = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_mask, right_mask = masks[left], masks[right]
            eligible = (left_mask["contype"] & right_mask["conaffinity"]) != 0 or (
                right_mask["contype"] & left_mask["conaffinity"]
            ) != 0
            if eligible:
                pairs.append([left, right])
    return pairs


def _control_mask_eligibility(
    baseline: Path, candidate: Path, declared: Mapping[str, object], case_directory: Path
) -> dict[str, object]:
    """Compare authored collision-mask pair eligibility between the two roles."""
    baseline_masks = _authored_masks(baseline)
    candidate_masks = _authored_masks(candidate)
    baseline_pairs = _eligible_pairs(baseline_masks)
    candidate_pairs = _eligible_pairs(candidate_masks)
    if baseline_pairs == candidate_pairs:
        raise RuntimeError(
            "mask flattening control observed identical eligible pair sets in both roles"
        )
    return {
        "kind": _require_text(declared.get("kind"), "independent_control.kind"),
        "classification": "ELIGIBLE_PAIR_SET_DIFFERS",
        "baseline_masks": {name: dict(value) for name, value in sorted(baseline_masks.items())},
        "candidate_masks": {name: dict(value) for name, value in sorted(candidate_masks.items())},
        "baseline_eligible_pairs": baseline_pairs,
        "candidate_eligible_pairs": candidate_pairs,
    }


def _control_explicit_pairs(
    baseline: Path, candidate: Path, declared: Mapping[str, object], case_directory: Path
) -> dict[str, object]:
    """Compare the compiled explicit collision-pair count between the two roles."""
    baseline_model = _compile(baseline)
    candidate_model = _compile(candidate)
    baseline_npair = int(baseline_model.npair)
    candidate_npair = int(candidate_model.npair)
    if baseline_npair != 1 or candidate_npair != 0:
        raise RuntimeError(
            f"explicit pair control expected baseline npair 1 and candidate npair 0, observed "
            f"{baseline_npair} and {candidate_npair}"
        )
    return {
        "kind": _require_text(declared.get("kind"), "independent_control.kind"),
        "classification": "BASELINE_PRESENT_CANDIDATE_ABSENT",
        "baseline_npair": baseline_npair,
        "candidate_npair": candidate_npair,
        "baseline_pair_geom_ids": [
            [int(baseline_model.pair_geom1[index]), int(baseline_model.pair_geom2[index])]
            for index in range(baseline_npair)
        ],
        "candidate_pair_geom_ids": [],
    }


def _control_exclusions(
    baseline: Path, candidate: Path, declared: Mapping[str, object], case_directory: Path
) -> dict[str, object]:
    """Compare the compiled body-exclusion count between the two roles."""
    baseline_model = _compile(baseline)
    candidate_model = _compile(candidate)
    baseline_nexclude = int(baseline_model.nexclude)
    candidate_nexclude = int(candidate_model.nexclude)
    if baseline_nexclude != 1 or candidate_nexclude != 0:
        raise RuntimeError(
            f"exclusion control expected baseline nexclude 1 and candidate nexclude 0, observed "
            f"{baseline_nexclude} and {candidate_nexclude}"
        )
    return {
        "kind": _require_text(declared.get("kind"), "independent_control.kind"),
        "classification": "BASELINE_PRESENT_CANDIDATE_ABSENT",
        "baseline_nexclude": baseline_nexclude,
        "candidate_nexclude": candidate_nexclude,
    }


def _control_mass_properties(
    baseline: Path, candidate: Path, declared: Mapping[str, object], case_directory: Path
) -> dict[str, object]:
    """Compare compiled mass and inertia for the shared frozen mesh between the two roles."""
    _run_mesh_builder(case_directory)
    frozen = _require_text(declared.get("mesh_sha256"), "independent_control.mesh_sha256")
    baseline_mesh = case_directory / "baseline" / "concave_shape.obj"
    candidate_mesh = case_directory / "candidate" / "concave_shape.obj"
    for role, mesh in (("baseline", baseline_mesh), ("candidate", candidate_mesh)):
        observed = _file_sha256(mesh)
        if observed != frozen:
            raise RuntimeError(
                f"{role} mesh digest {observed} is not the manifest's frozen digest {frozen}"
            )
    if baseline_mesh.read_bytes() != candidate_mesh.read_bytes():
        raise RuntimeError("the two roles must share byte-identical mesh geometry")

    measurements: dict[str, dict[str, object]] = {}
    for role, entrypoint in (("baseline", baseline), ("candidate", candidate)):
        model = _compile(entrypoint)
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "inertia_body")
        if body < 0:
            raise RuntimeError(f"{role} model does not define body 'inertia_body'")
        measurements[role] = {
            "body_mass": _float_tokens([model.body_mass[body]]),
            "body_ipos": _float_tokens(list(model.body_ipos[body])),
            "body_inertia": _float_tokens(list(model.body_inertia[body])),
        }
    differs = any(
        _hex_tokens(measurements["baseline"][key]) != _hex_tokens(measurements["candidate"][key])
        for key in ("body_mass", "body_inertia")
    )
    if not differs:
        raise RuntimeError(
            "mesh inertia control observed identical compiled mass and inertia in both roles"
        )
    return {
        "kind": _require_text(declared.get("kind"), "independent_control.kind"),
        "classification": "MASS_OR_INERTIA_DIFFERS",
        "mesh_sha256": frozen,
        "baseline": measurements["baseline"],
        "candidate": measurements["candidate"],
    }


def _run_mesh_builder(case_directory: Path) -> None:
    """Execute the tracked mesh verifier in-process and require it to succeed."""
    import runpy

    builder = case_directory / "build_mesh.py"
    if not builder.is_file():
        raise RuntimeError(f"mesh builder is missing: {builder}")
    saved = list(sys.argv)
    sys.argv = [str(builder)]
    try:
        runpy.run_path(str(builder), run_name="__main__")
    except SystemExit as exit_status:  # the tracked builder reports success through its exit code
        if exit_status.code not in (0, None):
            raise RuntimeError(f"mesh builder failed with exit code {exit_status.code}") from None
    finally:
        sys.argv = saved


def _control_actuator_moment(
    baseline: Path, candidate: Path, declared: Mapping[str, object], case_directory: Path
) -> dict[str, object]:
    """Compare the compiled actuator moment and generalized force at one frozen pose."""
    quaternion = [
        float(cast("str", token)) for token in _require_list(declared.get("qpos_wxyz"), "qpos_wxyz")
    ]
    control_value = float(_require_text(declared.get("control"), "independent_control.control"))
    measurements: dict[str, dict[str, object]] = {}
    for role, entrypoint in (("baseline", baseline), ("candidate", candidate)):
        model = _compile(entrypoint)
        data = mujoco.MjData(model)
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rotor_joint")
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive")
        if joint < 0 or actuator < 0:
            raise RuntimeError(
                f"{role} model does not define joint 'rotor_joint' and actuator 'drive'"
            )
        address = int(model.jnt_qposadr[joint])
        for offset, component in enumerate(quaternion):
            data.qpos[address + offset] = component
        data.ctrl[actuator] = control_value
        mujoco.mj_forward(model, data)
        # The bindings expose the moment as (nu, nv) but flatten it for a single actuator, so
        # reshape by nu before selecting this actuator's row.
        moment = np.asarray(data.actuator_moment, dtype=float).reshape(model.nu, -1)[actuator]
        measurements[role] = {
            "actuator_moment": _float_tokens([float(value) for value in moment]),
            "qfrc_actuator": _float_tokens([float(value) for value in data.qfrc_actuator]),
        }
    differs = any(
        _hex_tokens(measurements["baseline"][key]) != _hex_tokens(measurements["candidate"][key])
        for key in ("actuator_moment", "qfrc_actuator")
    )
    if not differs:
        raise RuntimeError(
            "actuator transmission control observed identical moment and generalized force"
        )
    return {
        "kind": _require_text(declared.get("kind"), "independent_control.kind"),
        "classification": "ACTUATOR_MOMENT_OR_GENERALIZED_FORCE_DIFFERS",
        "qpos_wxyz": [repr(component) for component in quaternion],
        "control": repr(control_value),
        "baseline": measurements["baseline"],
        "candidate": measurements["candidate"],
    }


def _hex_tokens(measurement: object) -> list[str]:
    """Return the binary64 hexadecimal tokens of one recorded measurement.

    Roles are compared on these tokens rather than on decimal text, because two different binary64
    values can share a rounded decimal spelling while their exact bytes differ.
    """
    mapping = _mapping(measurement, "measurement")
    tokens = mapping.get("binary64_hex")
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise RuntimeError("measurement binary64_hex is not a list of strings")
    return [str(token) for token in tokens]


def _require_list(value: object, label: str) -> list[object]:
    """Return one required array field or fail with the field name."""
    if not isinstance(value, list):
        raise RuntimeError(f"independent_control.{label} is not an array")
    return value


def _control_sensor_attachment(
    baseline: Path, candidate: Path, declared: Mapping[str, object], case_directory: Path
) -> dict[str, object]:
    """Resolve which compiled site each named force/torque sensor is attached to in each role.

    This observes compiled attachment only. It makes no claim that the two models produce comparable
    sensor readings; public sensor comparison is outside the current product.
    """
    sensor_names = [
        _require_text(name, "independent_control.sensor_names[]")
        for name in _require_list(declared.get("sensor_names"), "sensor_names")
    ]
    attachments: dict[str, dict[str, str]] = {}
    for role, entrypoint in (("baseline", baseline), ("candidate", candidate)):
        model = _compile(entrypoint)
        role_attachments: dict[str, str] = {}
        for sensor_name in sensor_names:
            sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            if sensor < 0:
                raise RuntimeError(f"{role} model does not define sensor {sensor_name!r}")
            object_type = int(model.sensor_objtype[sensor])
            if object_type != int(mujoco.mjtObj.mjOBJ_SITE):
                raise RuntimeError(
                    f"{role} sensor {sensor_name!r} is attached to object type {object_type}, "
                    "not mjOBJ_SITE"
                )
            site_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_SITE, int(model.sensor_objid[sensor])
            )
            if not isinstance(site_name, str):
                raise RuntimeError(f"{role} sensor {sensor_name!r} has an unnamed site target")
            role_attachments[sensor_name] = site_name
        attachments[role] = role_attachments
    baseline_sites = set(attachments["baseline"].values())
    candidate_sites = set(attachments["candidate"].values())
    if baseline_sites != {"sensor_frame"} or candidate_sites != {"grip_frame"}:
        raise RuntimeError(
            f"sensor attachment control expected baseline 'sensor_frame' and candidate "
            f"'grip_frame', observed {sorted(baseline_sites)} and {sorted(candidate_sites)}"
        )
    return {
        "kind": _require_text(declared.get("kind"), "independent_control.kind"),
        "classification": "BASELINE_SENSOR_FRAME_CANDIDATE_GRIP_FRAME",
        "sensor_object_type": "mjOBJ_SITE",
        "baseline_attachment": attachments["baseline"],
        "candidate_attachment": attachments["candidate"],
    }
