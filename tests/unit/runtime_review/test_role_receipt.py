"""Unit coverage for role-bound Runtime Review receipt construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pytest

from metrifid.json_values import CanonicalValue, canonical_json_bytes, compute_self_hash
from metrifid.runtime_review import _receipt_validation as receipt_validation
from metrifid.runtime_review._config import (
    AdmittedRuntimeReviewConfigurationV2,
    ExpectedSubjectConfig,
    ExpectedWorkloadConfig,
    RuntimeProfileConfigV2,
    RuntimeReviewCellConfig,
    RuntimeReviewConfigV2,
)
from metrifid.runtime_review._owned_output import (
    OwnedEvidenceCell,
    OwnedEvidenceMember,
    OwnedProfileIdentity,
)
from metrifid.runtime_review._receipt import (
    RUNTIME_REVIEW_RECEIPT_KEYS,
    build_runtime_review_receipt_v2,
)
from metrifid.runtime_review._receipt_validation import _validate_document_schema
from metrifid.runtime_review._status import RuntimeReviewStatus
from metrifid.runtime_review._witness import StableWitness
from metrifid.version import __version__

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
SHA_0 = "0" * 64
SHA_1 = "1" * 64


@dataclass(frozen=True, slots=True)
class _MeasuredRoleCampaign:
    """Supply exact role projections plus stable subject and workload facts."""

    profiles: dict[str, CanonicalValue]
    subject: dict[str, CanonicalValue]
    workload: dict[str, CanonicalValue]


@dataclass(frozen=True, slots=True)
class _CompletedRoleDecision:
    """Supply one completed decision without platform-sensitive diagnostics."""

    status: RuntimeReviewStatus
    reason_code: None
    admitted_prefix: str
    first_failing_gate: None
    first_decisive_witness: StableWitness
    worst_witness: StableWitness
    witness_counts: dict[str, CanonicalValue]
    witnesses: tuple[StableWitness, ...]


def _admitted_role_configuration(tmp_path: Path) -> AdmittedRuntimeReviewConfigurationV2:
    """Build one role-based configuration with inert exact test paths."""
    cells = tuple(
        RuntimeReviewCellConfig(role, step, repeat, f"inputs/{role}/{step}/{repeat}")
        for role in ("baseline", "candidate")
        for step in ("0.004", "0.002", "0.001")
        for repeat in (0, 1)
    )
    config = RuntimeReviewConfigV2(
        "metrifid.runtime_review_config",
        2,
        RuntimeProfileConfigV2(
            "baseline",
            "3.12.0.post2+baseline.1",
            "3.12.0",
            3_012_000,
            SHA_A,
            "profile_identities/baseline.json",
        ),
        RuntimeProfileConfigV2(
            "candidate",
            "3.12.0+candidate.1",
            "3.12.0",
            3_012_000,
            SHA_B,
            "profile_identities/candidate.json",
        ),
        ExpectedSubjectConfig("subject", SHA_C, SHA_D),
        ExpectedWorkloadConfig(SHA_E, SHA_F, SHA_0),
        "1",
        ("0.004", "0.002", "0.001"),
        (0, 1),
        cells,
        "output",
    )
    raw = canonical_json_bytes(config.to_primitive()) + b"\n"
    return AdmittedRuntimeReviewConfigurationV2(
        config=config,
        path=tmp_path / "runtime_review.json",
        base_dir=tmp_path,
        raw_bytes=raw,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        semantic_sha256=hashlib.sha256(canonical_json_bytes(config.to_primitive())).hexdigest(),
        cell_directories=tuple(tmp_path / cell.directory for cell in config.cells),
        profile_identity_paths=(
            tmp_path / "profile_identities" / "baseline.json",
            tmp_path / "profile_identities" / "candidate.json",
        ),
        profile_identity_file_sha256=(SHA_C, SHA_D),
        output_dir=tmp_path / "output",
    )


def _owned_cells() -> tuple[OwnedEvidenceCell, ...]:
    """Build all canonical owned evidence locators and byte identities."""
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
            tuple(OwnedEvidenceMember(name, SHA_0, 0) for name in names),
        )
        for role in ("baseline", "candidate")
        for step in ("0.004", "0.002", "0.001")
        for repeat in (0, 1)
    )


def _owned_profiles() -> tuple[OwnedProfileIdentity, OwnedProfileIdentity]:
    """Build baseline-then-candidate owned profile file identities."""
    return (
        OwnedProfileIdentity(
            "baseline", PurePosixPath("profile_identities/baseline.json"), SHA_C, 101
        ),
        OwnedProfileIdentity(
            "candidate", PurePosixPath("profile_identities/candidate.json"), SHA_D, 102
        ),
    )


def _profile_binding(role: str) -> dict[str, CanonicalValue]:
    """Return one exact evidence-derived role projection before owned-file injection."""
    baseline = role == "baseline"
    package = "3.12.0.post2+baseline.1" if baseline else "3.12.0+candidate.1"
    return {
        "profile_role": role,
        "package_version": package,
        "native_version": "3.12.0",
        "native_version_integer": 3_012_000,
        "profile_identity_sha256": SHA_A if baseline else SHA_B,
        "runtime_identity_sha256": SHA_E if baseline else SHA_F,
        "mujoco_distribution": {
            "name": "mujoco",
            "version": package,
            "payload_sha256": SHA_A if baseline else SHA_B,
            "record_bound_payload_sha256": SHA_C if baseline else SHA_D,
        },
        "loaded_native_library": {
            "resolved_path": f"/profiles/{role}/libmujoco.so",
            "size_bytes": 100,
            "sha256": SHA_E if baseline else SHA_F,
        },
        "numpy": {
            "python_version": "2.4.0",
            "distribution": {
                "name": "numpy",
                "version": "2.4.0",
                "payload_sha256": SHA_0,
                "record_bound_payload_sha256": SHA_1,
            },
        },
        "sentinel": {
            "status": "PASS",
            "sentinel_identity_sha256": SHA_1 if baseline else SHA_0,
        },
        "support_tier": "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE",
    }


def _authority() -> tuple[_MeasuredRoleCampaign, _CompletedRoleDecision]:
    """Build the evidence and completed decision behind one role-based receipt."""
    witness = StableWitness(
        channel_id="arm.position",
        classification="OUTSIDE_DECLARED_MIGRATION_ENVELOPE",
        kind="SCALAR",
        semantic_type="CONTINUOUS_SCALAR",
        time="0.25",
        tolerance="0.000001",
        decision_input_sha256=SHA_1,
    )
    evidence = _MeasuredRoleCampaign(
        profiles={
            "baseline": _profile_binding("baseline"),
            "candidate": _profile_binding("candidate"),
            "common_environment": {"numpy_distribution_sha256": SHA_0},
        },
        subject={"fixture_id": "subject"},
        workload={"semantic_sha256": SHA_0},
    )
    decision = _CompletedRoleDecision(
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
    return evidence, decision


def _tool() -> dict[str, CanonicalValue]:
    """Return a minimal internally consistent installed-tool projection."""
    distribution: dict[str, CanonicalValue] = {
        "distribution_version": __version__,
        "schema": "metrifid.installed_distribution_identity",
        "schema_version": 1,
    }
    return {
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


def _receipt(tmp_path: Path) -> dict[str, CanonicalValue]:
    """Build one completed role-based receipt."""
    evidence, decision = _authority()
    return build_runtime_review_receipt_v2(
        configuration=_admitted_role_configuration(tmp_path),
        evidence=evidence,
        decision=decision,
        evidence_cells=_owned_cells(),
        profile_identities=_owned_profiles(),
        tool=_tool(),
    )


def test_role_receipt_binds_owned_profile_and_sentinel_authority(tmp_path: Path) -> None:
    """Bind existing science plus exact profile, runtime, sentinel, and owned-file facts."""
    receipt = _receipt(tmp_path)

    assert set(receipt) == RUNTIME_REVIEW_RECEIPT_KEYS
    assert receipt["schema_version"] == 2
    assert receipt["receipt_sha256"] == compute_self_hash(receipt, "receipt_sha256")
    profiles = receipt["profiles"]
    assert isinstance(profiles, dict)
    baseline = profiles["baseline"]
    candidate = profiles["candidate"]
    assert isinstance(baseline, dict)
    assert isinstance(candidate, dict)
    assert baseline["identity_file"] == {
        "locator": "profile_identities/baseline.json",
        "raw_sha256": SHA_C,
        "size_bytes": 101,
    }
    sentinel = candidate["sentinel"]
    assert isinstance(sentinel, dict)
    assert sentinel["status"] == "PASS"
    assert _validate_document_schema(receipt) == 2


def test_role_receipt_refuses_historical_tool_provenance(tmp_path: Path) -> None:
    """Keep the production v2 route bound to the current installed product."""
    receipt = _receipt(tmp_path)
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

    with pytest.raises(ValueError, match="v2 receipt tool version"):
        _validate_document_schema(receipt)


def test_role_receipt_refuses_profile_or_sentinel_substitution(tmp_path: Path) -> None:
    """Reject evidence bindings that differ from configuration or lack sentinel PASS."""
    evidence, decision = _authority()
    baseline = evidence.profiles["baseline"]
    assert isinstance(baseline, dict)
    baseline["package_version"] = "3.11.0"
    with pytest.raises(ValueError, match="configuration declaration"):
        build_runtime_review_receipt_v2(
            configuration=_admitted_role_configuration(tmp_path),
            evidence=evidence,
            decision=decision,
            evidence_cells=_owned_cells(),
            profile_identities=_owned_profiles(),
            tool=_tool(),
        )

    evidence, decision = _authority()
    baseline = evidence.profiles["baseline"]
    assert isinstance(baseline, dict)
    sentinel = baseline["sentinel"]
    assert isinstance(sentinel, dict)
    sentinel["status"] = "FAIL"
    with pytest.raises(ValueError, match="sentinel must bind one PASS"):
        build_runtime_review_receipt_v2(
            configuration=_admitted_role_configuration(tmp_path),
            evidence=evidence,
            decision=decision,
            evidence_cells=_owned_cells(),
            profile_identities=_owned_profiles(),
            tool=_tool(),
        )


def test_role_receipt_refuses_release_validation_tier(tmp_path: Path) -> None:
    """Do not let an exact-profile label bypass current live admission."""
    evidence, decision = _authority()
    baseline = evidence.profiles["baseline"]
    assert isinstance(baseline, dict)
    baseline["support_tier"] = "VALIDATED_EXACT_PROFILE"

    with pytest.raises(ValueError, match="current capability admission"):
        build_runtime_review_receipt_v2(
            configuration=_admitted_role_configuration(tmp_path),
            evidence=evidence,
            decision=decision,
            evidence_cells=_owned_cells(),
            profile_identities=_owned_profiles(),
            tool=_tool(),
        )

    receipt = _receipt(tmp_path)
    profiles = receipt["profiles"]
    assert isinstance(profiles, dict)
    retained_baseline = profiles["baseline"]
    assert isinstance(retained_baseline, dict)
    retained_baseline["support_tier"] = "VALIDATED_EXACT_PROFILE"
    receipt["receipt_sha256"] = compute_self_hash(receipt, "receipt_sha256")
    with pytest.raises(ValueError, match="current capability admission"):
        _validate_document_schema(receipt)


def test_role_receipt_refuses_owned_identity_changed_after_admission(tmp_path: Path) -> None:
    """Bind copied profile bytes to the raw-file hashes admitted with the configuration."""
    identities = list(_owned_profiles())
    identities[0] = OwnedProfileIdentity(
        "baseline",
        PurePosixPath("profile_identities/baseline.json"),
        SHA_E,
        101,
    )

    evidence, decision = _authority()
    with pytest.raises(ValueError, match="differ from configuration admission"):
        build_runtime_review_receipt_v2(
            configuration=_admitted_role_configuration(tmp_path),
            evidence=evidence,
            decision=decision,
            evidence_cells=_owned_cells(),
            profile_identities=identities,
            tool=_tool(),
        )


def test_role_receipt_refuses_a_partial_campaign(tmp_path: Path) -> None:
    """A role-based receipt cannot publish with fewer than all twelve exact cells."""
    evidence, decision = _authority()

    with pytest.raises(ValueError, match="exactly twelve"):
        build_runtime_review_receipt_v2(
            configuration=_admitted_role_configuration(tmp_path),
            evidence=evidence,
            decision=decision,
            evidence_cells=_owned_cells()[:-1],
            profile_identities=_owned_profiles(),
            tool=_tool(),
        )


def test_role_receipt_refuses_sentinel_hash_different_from_owned_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt re-seal cannot substitute the sentinel above owned profile bytes."""
    receipt = _receipt(tmp_path)
    profiles = receipt["profiles"]
    assert isinstance(profiles, dict)
    root_members: dict[str, bytes] = {}
    identities: dict[str, dict[str, object]] = {}
    for role in ("baseline", "candidate"):
        profile = profiles[role]
        assert isinstance(profile, dict)
        identity_file = profile["identity_file"]
        assert isinstance(identity_file, dict)
        identity: dict[str, object] = {
            "profile_role": role,
            "profile_identity_sha256": SHA_A if role == "baseline" else SHA_B,
        }
        raw = canonical_json_bytes(identity) + b"\n"  # type: ignore[arg-type]
        locator = f"profile_identities/{role}.json"
        root_members[locator] = raw
        identities[role] = identity
        identity_file["raw_sha256"] = hashlib.sha256(raw).hexdigest()
        identity_file["size_bytes"] = len(raw)

    def load_identity(_path: Path, role: str, _expected_sha256: str) -> dict[str, object]:
        """Return the exact compact owned identity selected by semantic role."""
        return identities[role]

    def substituted_projection(identity: object) -> dict[str, CanonicalValue]:
        """Return the receipt projection with one owned sentinel hash substituted."""
        assert isinstance(identity, dict)
        role = identity["profile_role"]
        assert isinstance(role, str)
        profile = profiles[role]
        assert isinstance(profile, dict)
        projection = {
            key: value
            for key, value in profile.items()
            if key not in {"identity_file", "runtime_identity_sha256"}
        }
        if role == "baseline":
            sentinel = projection["sentinel"]
            assert isinstance(sentinel, dict)
            sentinel = dict(sentinel)
            sentinel["sentinel_identity_sha256"] = SHA_F
            projection["sentinel"] = sentinel
        return projection

    monkeypatch.setattr(
        receipt_validation,
        "_load_profile_identity_authority_v2",
        load_identity,
    )
    monkeypatch.setattr(
        receipt_validation,
        "_profile_identity_receipt_projection_v2",
        substituted_projection,
    )

    with pytest.raises(ValueError, match="differs from its owned profile identity"):
        receipt_validation._parse_owned_profile_identities_v2(
            tmp_path,
            root_members,
            profiles,
            _admitted_role_configuration(tmp_path).config,
        )


@pytest.mark.parametrize("schema_version", [0, 3, True, "2"])
def test_receipt_dispatch_refuses_unknown_or_ambiguous_versions(
    tmp_path: Path, schema_version: object
) -> None:
    """Never select a receipt-validation route from an unsupported version token."""
    receipt = _receipt(tmp_path)
    receipt["schema_version"] = schema_version  # type: ignore[assignment]
    receipt["receipt_sha256"] = compute_self_hash(receipt, "receipt_sha256")

    with pytest.raises((TypeError, ValueError), match="schema_version"):
        _validate_document_schema(receipt)


def test_role_receipt_schema_rejects_swapped_profile_binding(tmp_path: Path) -> None:
    """Reject a re-sealed role substitution before any owned evidence replay."""
    receipt = _receipt(tmp_path)
    profiles = receipt["profiles"]
    assert isinstance(profiles, dict)
    baseline = profiles["baseline"]
    assert isinstance(baseline, dict)
    baseline["profile_role"] = "candidate"
    receipt["receipt_sha256"] = compute_self_hash(receipt, "receipt_sha256")

    with pytest.raises(ValueError, match="differs from its receipt role"):
        _validate_document_schema(receipt)
