"""Unit coverage for stable Runtime Review receipts and explanatory Markdown."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast
from unittest.mock import Mock

import pytest

from metrifid.json_values import (
    CanonicalValue,
    canonical_json_bytes,
    compute_self_hash,
    validate_self_hash,
)
from metrifid.runtime_review._config import (
    AdmittedRuntimeReviewConfiguration,
    ExpectedSubjectConfig,
    ExpectedWorkloadConfig,
    RuntimeProfileConfig,
    RuntimeReviewCellConfig,
    RuntimeReviewConfig,
)
from metrifid.runtime_review._markdown import render_runtime_review_markdown
from metrifid.runtime_review._owned_output import OwnedEvidenceCell, OwnedEvidenceMember
from metrifid.runtime_review._receipt import (
    RUNTIME_REVIEW_RECEIPT_KEYS,
    build_runtime_review_receipt,
)
from metrifid.runtime_review._receipt_validation import (
    _recompute_and_compare,
    _validate_document_schema,
)
from metrifid.runtime_review._status import RuntimeReviewStatus
from metrifid.runtime_review._witness import StableWitness
from metrifid.version import __version__

_SHA = "0" * 64


@dataclass(frozen=True, slots=True)
class _MeasuredCampaign:
    """Supply stable admitted campaign facts to the focused receipt test."""

    profiles: dict[str, CanonicalValue]
    subject: dict[str, CanonicalValue]
    workload: dict[str, CanonicalValue]


@dataclass(frozen=True, slots=True)
class _CompletedDecision:
    """Supply one completed public decision without private diagnostics."""

    status: RuntimeReviewStatus
    reason_code: None
    admitted_prefix: str
    first_failing_gate: None
    first_decisive_witness: StableWitness
    worst_witness: StableWitness
    witness_counts: dict[str, CanonicalValue]
    witnesses: tuple[StableWitness, ...]


def _admitted_configuration(tmp_path: Path) -> AdmittedRuntimeReviewConfiguration:
    """Build one semantically complete configuration with inert unit-test paths."""
    cells = tuple(
        RuntimeReviewCellConfig(role, step, repeat, f"inputs/{role}/{step}/{repeat}")
        for role in ("baseline", "candidate")
        for step in ("0.004", "0.002", "0.001")
        for repeat in (0, 1)
    )
    config = RuntimeReviewConfig(
        "metrifid.runtime_review_config",
        1,
        RuntimeProfileConfig("A_3.10.0", "3.10.0"),
        RuntimeProfileConfig("B_3.11.0", "3.11.0"),
        ExpectedSubjectConfig("subject", _SHA, _SHA),
        ExpectedWorkloadConfig(_SHA, _SHA, _SHA),
        "1",
        ("0.004", "0.002", "0.001"),
        (0, 1),
        cells,
        "output",
    )
    raw = canonical_json_bytes(config.to_primitive()) + b"\n"
    return AdmittedRuntimeReviewConfiguration(
        config=config,
        path=tmp_path / "runtime_review.json",
        base_dir=tmp_path,
        raw_bytes=raw,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        semantic_sha256=hashlib.sha256(canonical_json_bytes(config.to_primitive())).hexdigest(),
        cell_directories=tuple(tmp_path / cell.directory for cell in config.cells),
        output_dir=tmp_path / "output",
    )


def _owned_cells() -> tuple[OwnedEvidenceCell, ...]:
    """Build the canonical complete owned-cell identity list."""
    tokens = {"0.004": "0p004", "0.002": "0p002", "0.001": "0p001"}
    names = (
        "CHECKSUMS.sha256",
        "fixture.xml",
        "input_manifest.json",
        "model.mjb",
        "result.json",
        "trace.npz",
    )
    return tuple(
        OwnedEvidenceCell(
            role,
            step,
            repeat,
            PurePosixPath("evidence", role, tokens[step], f"repeat_{repeat}"),
            tuple(OwnedEvidenceMember(name, _SHA, 0) for name in names),
        )
        for role in ("baseline", "candidate")
        for step in ("0.004", "0.002", "0.001")
        for repeat in (0, 1)
    )


def _completed_authority() -> tuple[_MeasuredCampaign, _CompletedDecision]:
    """Build the evidence and decision that authoritatively back one test receipt."""
    witness = StableWitness(
        channel_id="arm.position",
        classification="OUTSIDE_DECLARED_MIGRATION_ENVELOPE",
        kind="SCALAR",
        semantic_type="CONTINUOUS_SCALAR",
        time="0.25",
        tolerance="0.000001",
        decision_input_sha256="1" * 64,
    )
    decision = _CompletedDecision(
        RuntimeReviewStatus.OUTSIDE_DECLARED_MIGRATION_ENVELOPE,
        None,
        "1",
        None,
        witness,
        witness,
        {
            "WITHIN_DECLARED_MIGRATION_ENVELOPE": 0,
            "UNRESOLVED_NEAR_BOUNDARY": 0,
            "OUTSIDE_DECLARED_MIGRATION_ENVELOPE": 1,
        },
        (witness,),
    )
    evidence = _MeasuredCampaign(
        profiles={"baseline": {"profile_id": "A_3.10.0"}, "candidate": {"profile_id": "B_3.11.0"}},
        subject={"fixture_id": "subject"},
        workload={"semantic_sha256": _SHA},
    )
    return evidence, decision


def _completed_receipt(tmp_path: Path) -> dict[str, CanonicalValue]:
    """Build one outside receipt from stable authoritative evidence and decision facts."""
    evidence, decision = _completed_authority()
    distribution: dict[str, CanonicalValue] = {
        "distribution_version": __version__,
        "schema": "metrifid.installed_distribution_identity",
        "schema_version": 1,
    }
    tool: dict[str, CanonicalValue] = {
        "metrifid_version": __version__,
        "distribution_identity": distribution,
        "distribution_identity_sha256": hashlib.sha256(
            canonical_json_bytes(distribution)
        ).hexdigest(),
        "evaluator": {
            "python_implementation": "CPython",
            "python_version": "3.12.0",
            "python_cache_tag": "cpython-312",
            "platform": "test-platform",
            "system": "TestOS",
            "machine": "test-machine",
        },
    }
    return build_runtime_review_receipt(
        configuration=_admitted_configuration(tmp_path),
        evidence=evidence,
        decision=decision,
        evidence_cells=_owned_cells(),
        tool=tool,
    )


def _assert_reconstruction_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt: dict[str, CanonicalValue],
    expected_field: str,
) -> None:
    """Prove a structurally valid reseal still fails evidence-backed reconstruction."""
    from metrifid.runtime_review import _decision, _evidence

    evidence, decision = _completed_authority()
    monkeypatch.setattr(_evidence, "admit_runtime_evidence", Mock(return_value=evidence))
    monkeypatch.setattr(_decision, "evaluate_runtime_evidence", Mock(return_value=decision))
    _validate_document_schema(receipt)
    validate_self_hash(receipt, "receipt_sha256")
    with pytest.raises(ValueError, match=rf"receipt {expected_field} does not match"):
        _recompute_and_compare(
            receipt,
            _admitted_configuration(tmp_path),
            _owned_cells(),
        )


def test_receipt_has_exact_decision_fields_and_canonical_self_hash(tmp_path: Path) -> None:
    """Bind the closed root contract and self-hash to stable decision content."""
    receipt = _completed_receipt(tmp_path)

    assert set(receipt) == RUNTIME_REVIEW_RECEIPT_KEYS
    assert receipt["receipt_sha256"] == compute_self_hash(receipt, "receipt_sha256")
    assert receipt["first_decisive_witness"] == receipt["worst_witness"]
    witness = receipt["first_decisive_witness"]
    assert isinstance(witness, dict)
    assert set(witness) == {
        "channel_id",
        "classification",
        "kind",
        "semantic_type",
        "time",
        "tolerance",
        "decision_input_sha256",
    }
    assert b"difference_interval" not in canonical_json_bytes(receipt)
    assert b'"ratio"' not in canonical_json_bytes(receipt)


def test_legacy_receipt_admits_self_consistent_historical_tool_provenance(
    tmp_path: Path,
) -> None:
    """Validate immutable v1 evidence without relabeling its historical producer."""
    receipt = _completed_receipt(tmp_path)
    tool = receipt["tool"]
    assert isinstance(tool, dict)
    distribution = tool["distribution_identity"]
    assert isinstance(distribution, dict)
    tool["metrifid_version"] = "0.6.0.dev0"
    distribution["distribution_version"] = "0.6.0.dev0"
    tool["distribution_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(distribution)
    ).hexdigest()
    receipt["receipt_sha256"] = compute_self_hash(receipt, "receipt_sha256")

    assert _validate_document_schema(receipt) == 1


def test_independent_reconstruction_rejects_a_resealed_decision_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a status substitution even after its receipt is canonically resealed."""
    receipt = _completed_receipt(tmp_path)
    receipt["status"] = "WITHIN_DECLARED_MIGRATION_ENVELOPE"
    receipt["receipt_sha256"] = compute_self_hash(receipt, "receipt_sha256")

    _assert_reconstruction_refuses(tmp_path, monkeypatch, receipt, "status")


def test_independent_reconstruction_rejects_a_resealed_evidence_identity_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject substituted owned-cell identity facts even after canonical receipt resealing."""
    receipt = _completed_receipt(tmp_path)
    cells = cast(list[CanonicalValue], receipt["evidence_cells"])
    first_cell = cast(dict[str, CanonicalValue], cells[0])
    members = cast(list[CanonicalValue], first_cell["members"])
    first_member = cast(dict[str, CanonicalValue], members[0])
    first_member["sha256"] = "2" * 64
    receipt["receipt_sha256"] = compute_self_hash(receipt, "receipt_sha256")

    _assert_reconstruction_refuses(tmp_path, monkeypatch, receipt, "evidence_cells")


def test_markdown_explains_full_horizon_and_the_decisive_witness(tmp_path: Path) -> None:
    """Keep the human report explanatory while the JSON remains decision-bearing."""
    markdown = render_runtime_review_markdown(_completed_receipt(tmp_path))

    assert "entire declared horizon" in markdown
    assert "First decisive witness" in markdown
    assert "arm.position" in markdown
    assert "non-normative" in markdown
    assert "not an authenticity proof" in markdown
