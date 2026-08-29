#!/usr/bin/env python3
"""Demonstrate the public, two-pass static Model Change Gate workflow.

Run this file against an installed Metrifid distribution. It first uses public Certify to bind
the exact baseline and candidate MJB identities. An unbound-candidate discovery review must ask
for review. The script then turns every explained, non-opaque change into an exact ALLOW rule and
runs a fresh, candidate-bound review that must fall within the declared policy.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path

from metrifid.certify import CertifyStatus, certify_models
from metrifid.model_release import ModelReleaseStatus, review_model_release

EXAMPLE_ROOT = Path(__file__).resolve().parent
BASELINE = EXAMPLE_ROOT / "baseline" / "model.xml"
CANDIDATE = EXAMPLE_ROOT / "candidate" / "model.xml"
_SELECTOR_FIELDS = ("object_type", "object_name", "field", "change_kind")


def _canonical_json(value: object) -> str:
    """Encode one JSON-compatible value with the public artifact's canonical JSON settings."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _write_json(path: Path, value: object) -> None:
    """Write a fresh canonical JSON document."""
    path.write_text(_canonical_json(value) + "\n", encoding="utf-8")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    """Return a nested public receipt object or fail with a useful example error."""
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} is not an object")
    if any(not isinstance(key, str) for key in value):
        raise RuntimeError(f"{label} has a non-string key")
    return value


def _text(value: object, label: str) -> str:
    """Return one required public receipt string."""
    if not isinstance(value, str):
        raise RuntimeError(f"{label} is not a string")
    return value


def _optional_hash(value: object, label: str) -> str | None:
    """Return one required digest-or-null policy constraint."""
    if value is not None and not isinstance(value, str):
        raise RuntimeError(f"{label} is not a string or null")
    return value


def _mjb_sha256(certification_receipt: Mapping[str, object], role: str) -> str:
    """Read one public complete-MJB identity from a Certify receipt."""
    role_receipt = _mapping(certification_receipt.get(role), f"certification.{role}")
    artifact = _mapping(
        role_receipt.get("compiled_artifact"),
        f"certification.{role}.compiled_artifact",
    )
    return _text(artifact.get("mjb_sha256"), f"certification.{role}.mjb_sha256")


def _policy(
    baseline_sha256: str,
    candidate_sha256: str | None,
    rules: list[dict[str, object]],
) -> dict[str, object]:
    """Build the complete public policy schema."""
    return {
        "schema": "metrifid.model_release_policy",
        "schema_version": 1,
        "baseline_compiled_sha256": baseline_sha256,
        "candidate_compiled_sha256": candidate_sha256,
        "rules": rules,
    }


def _exact_allow_rules(receipt: Mapping[str, object]) -> list[dict[str, object]]:
    """Convert every non-opaque discovery change into one exact ALLOW declaration."""
    raw_changes = receipt.get("changes")
    if not isinstance(raw_changes, list):
        raise RuntimeError("model-release receipt changes is not an array")

    rules: list[dict[str, object]] = []
    for index, raw_change in enumerate(raw_changes, start=1):
        change = _mapping(raw_change, f"changes[{index - 1}]")
        selector = _mapping(change.get("selector"), f"changes[{index - 1}].selector")
        if selector.get("object_type") == "opaque":
            continue
        if change.get("classification") != "UNDECLARED":
            raise RuntimeError("the discovery pass unexpectedly classified a non-opaque change")
        exact_selector = {
            field: _text(selector.get(field), f"selector.{field}") for field in _SELECTOR_FIELDS
        }
        rules.append(
            {
                "id": f"allow-{len(rules) + 1:04d}",
                "effect": "ALLOW",
                "selector": exact_selector,
                "before_sha256": _optional_hash(
                    change.get("before_sha256"), "change.before_sha256"
                ),
                "after_sha256": _optional_hash(change.get("after_sha256"), "change.after_sha256"),
            }
        )
    if not rules:
        raise RuntimeError("the changed example produced no explained, non-opaque changes")
    return rules


def _assert_mass_and_derived_changes(rules: list[dict[str, object]]) -> None:
    """Keep the example honest about one source edit expanding into derived compiled changes."""
    selectors = [_mapping(rule["selector"], "rule.selector") for rule in rules]
    has_mass = any(
        selector.get("object_type") == "body"
        and selector.get("object_name") == "arm_link"
        and selector.get("field") == "mass"
        for selector in selectors
    )
    has_compiled_field = any(
        selector.get("object_type") == "compiled_field" for selector in selectors
    )
    if not has_mass or not has_compiled_field:
        raise RuntimeError("the mass example did not expose both semantic and derived changes")


def _assert_no_absolute_sources(*receipts: Mapping[str, object]) -> None:
    """Confirm product receipts do not serialize either absolute source path or model root."""
    forbidden = {
        str(BASELINE.resolve()),
        str(CANDIDATE.resolve()),
        str(BASELINE.parent.resolve()),
        str(CANDIDATE.parent.resolve()),
    }
    for receipt in receipts:
        encoded = _canonical_json(receipt)
        if any(path in encoded for path in forbidden):
            raise RuntimeError("a product receipt contains an absolute source path")


def _workspace(raw: str | None) -> Path:
    """Create or admit one empty workspace outside both model roots."""
    if raw is None:
        return Path(tempfile.mkdtemp(prefix="metrifid-model-release-example-")).resolve()
    path = Path(raw).expanduser().resolve()
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise SystemExit("--workspace must name an absent or empty directory")
    else:
        path.mkdir(parents=True)
    return path


def main() -> int:
    """Run Certify, discovery review, and an exact candidate-bound declared review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        help="absent or empty artifact directory; default: a retained temporary directory",
    )
    arguments = parser.parse_args()
    workspace = _workspace(arguments.workspace)

    certification = certify_models(
        str(BASELINE),
        str(CANDIDATE),
        str(workspace / "certification"),
        baseline_root=str(BASELINE.parent),
        candidate_root=str(CANDIDATE.parent),
    )
    if certification.status is not CertifyStatus.NOT_CERTIFIED_COMPILED_DIFFERS:
        raise RuntimeError("the example mass edit did not change the complete MJB")
    baseline_sha256 = _mjb_sha256(certification.receipt, "baseline")
    candidate_sha256 = _mjb_sha256(certification.receipt, "candidate")

    discovery_policy_path = workspace / "discovery-policy.json"
    _write_json(discovery_policy_path, _policy(baseline_sha256, None, []))
    discovery = review_model_release(
        str(BASELINE),
        str(CANDIDATE),
        str(discovery_policy_path),
        str(workspace / "discovery-review"),
        baseline_root=str(BASELINE.parent),
        candidate_root=str(CANDIDATE.parent),
    )
    if discovery.status is not ModelReleaseStatus.REVIEW_REQUIRED:
        raise RuntimeError("an unbound-candidate discovery pass did not require review")
    linked_certification = _mapping(
        discovery.receipt.get("certification_receipt"),
        "model-release certification_receipt",
    )
    if _mjb_sha256(linked_certification, "baseline") != baseline_sha256:
        raise RuntimeError("the review is not linked to the certified baseline MJB")
    if _mjb_sha256(linked_certification, "candidate") != candidate_sha256:
        raise RuntimeError("the review is not linked to the certified candidate MJB")

    rules = _exact_allow_rules(discovery.receipt)
    _assert_mass_and_derived_changes(rules)
    declared_policy_path = workspace / "declared-policy.json"
    _write_json(
        declared_policy_path,
        _policy(baseline_sha256, candidate_sha256, rules),
    )
    declared = review_model_release(
        str(BASELINE),
        str(CANDIDATE),
        str(declared_policy_path),
        str(workspace / "declared-review"),
        baseline_root=str(BASELINE.parent),
        candidate_root=str(CANDIDATE.parent),
    )
    if declared.status is not ModelReleaseStatus.WITHIN_DECLARED_POLICY:
        raise RuntimeError("the exact candidate-bound policy did not cover every change")

    _assert_no_absolute_sources(
        certification.receipt,
        discovery.receipt,
        declared.receipt,
    )
    print(f"workspace       : {workspace}")
    print(f"certify         : {certification.status.value} -> {certification.certification_json}")
    print(f"discovery       : {discovery.status.value} -> {discovery.model_release_json}")
    print(f"declared policy : {len(rules)} exact ALLOW rules -> {declared_policy_path}")
    print(f"declared review : {declared.status.value} -> {declared.model_release_json}")
    print(f"markdown        : {declared.model_release_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
