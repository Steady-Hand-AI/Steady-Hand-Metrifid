"""Semantic tests for versioned static model-review surface coverage."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import mujoco  # type: ignore[import-untyped]
import pytest

from metrifid.model_release._public_field_registry_catalog import (
    characterized_registry,
    coherent_runtime_base_version,
)
from metrifid.model_release._run import _snapshot_failure
from metrifid.model_release._snapshot import (
    SnapshotRefusal,
    _public_field_facts,
    measure_public_field_surface,
)
from metrifid.operational import (
    OperationalReasonCode,
    OperationalToolObservation,
)

_CATALOG_CASES = (
    (
        "3.9.0",
        "a270838d7ea9371f99fbf831a495ae2a2320d8dfd39e469c06d06bb543d02aab",
        721,
        "d2659517d9515588122c086f801bd1ed96984f31437a9c1d225ac359c344be02",
        675,
    ),
    (
        "3.10.0",
        "a270838d7ea9371f99fbf831a495ae2a2320d8dfd39e469c06d06bb543d02aab",
        721,
        "d2659517d9515588122c086f801bd1ed96984f31437a9c1d225ac359c344be02",
        675,
    ),
    (
        "3.11.0",
        "db4b643579ddd9f57f3ec01b65a8de2789819b1157c92db0c3465a663306c2f6",
        743,
        "7fdcfeb200258a05cac058a98d1926590dcd266046f188a4c65cc26fa747b656",
        697,
    ),
    (
        "3.12.0",
        "0b56690844f789b9a68dc18e4243f6549d01f5059a18922d574469b463633440",
        745,
        "23bb9b012fe1b838e33ba50848c35afa6d9ac5e76a15bc8e679ad46233c783cd",
        699,
    ),
)


@pytest.mark.parametrize(
    ("base_version", "surface_sha256", "surface_count", "registry_sha256", "registry_count"),
    _CATALOG_CASES,
    ids=("oldest_validated", "prior_validated", "current_validated", "latest_validated"),
)
def test_catalog_retains_independently_repeated_surface_identities(
    base_version: str,
    surface_sha256: str,
    surface_count: int,
    registry_sha256: str,
    registry_count: int,
) -> None:
    """Bind each semantic profile role to its exact twice-measured surface tuple."""
    entry = characterized_registry(base_version)
    assert entry is not None
    assert entry.full_public_surface_sha256 == surface_sha256
    assert entry.full_public_surface_count == surface_count
    assert entry.comparable_registry_sha256 == registry_sha256
    assert entry.comparable_registry_count == registry_count
    assert entry.measurement_process_count == 2
    assert entry.measurements_identical is True


def test_cataloged_public_surface_reconstructs_the_receipt_registry() -> None:
    """Reconstruct the installed runtime's cataloged full and comparable identities."""
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    measurement = measure_public_field_surface(model)
    base_version = coherent_runtime_base_version(
        str(mujoco.__version__),
        str(mujoco.mj_versionString()),
        int(mujoco.mj_version()),
    )
    expected = characterized_registry(base_version)
    assert expected is not None
    assert measurement.full_public_surface_sha256 == expected.full_public_surface_sha256
    assert measurement.full_public_surface_count == expected.full_public_surface_count
    assert measurement.comparable_registry_sha256 == expected.comparable_registry_sha256
    assert measurement.comparable_registry_count == expected.comparable_registry_count
    assert measurement.opaque_member_paths == ()


@pytest.mark.parametrize(
    "package_version",
    ["3.12.0.post2", "3.12.0+vendor.1", "3.12.0.post2+vendor.1"],
    ids=("post_suffix", "local_suffix", "combined_suffix"),
)
def test_pure_receipt_binding_derives_stable_suffix_base(package_version: str) -> None:
    """Treat admitted post/local package suffixes as one exact native base identity."""
    assert coherent_runtime_base_version(package_version, "3.12.0", 3_012_000) == "3.12.0"


@pytest.mark.parametrize(
    ("package_version", "native_version", "native_integer"),
    [
        ("3.12.0rc1", "3.12.0", 3_012_000),
        ("3.12.0", "3.11.0", 3_011_000),
        ("3.12.0", "3.12.0", 3_011_000),
    ],
    ids=("prerelease", "base_mismatch", "integer_mismatch"),
)
def test_pure_receipt_binding_rejects_incoherent_identity(
    package_version: str,
    native_version: str,
    native_integer: int,
) -> None:
    """Reject malformed, cross-version, and algorithmically inconsistent receipt facts."""
    with pytest.raises(ValueError):
        coherent_runtime_base_version(package_version, native_version, native_integer)


def test_uncharacterized_model_review_surface_is_refused() -> None:
    """Never reuse an older comparable projection for an unknown future runtime."""
    model = cast(Any, SimpleNamespace(comparable_value=1))
    with pytest.raises(SnapshotRefusal) as captured:
        _public_field_facts(
            model,
            "baseline",
            "99.0.0",
            {"operation": "STATIC_MODEL_REVIEW"},
        )
    assert captured.value.issue == "uncharacterized_public_model_surface"
    assert captured.value.evidence["observed_full_public_surface_sha256"]
    assert captured.value.evidence["observed_registry_sha256"]
    assert "older typed public-field projection" in str(captured.value.evidence["exact_reason"])


def test_unreadable_nested_holder_becomes_typed_surface_mismatch() -> None:
    """A removed nested holder changes coverage instead of raising an attribute error."""
    model = cast(Any, SimpleNamespace(comparable_value=1))
    with pytest.raises(SnapshotRefusal) as captured:
        _public_field_facts(
            model,
            "candidate",
            "3.10.0",
            {"operation": "STATIC_MODEL_REVIEW"},
        )
    assert captured.value.issue == "public_model_surface_mismatch"
    assert captured.value.evidence["claim_risk"] == (
        "the cataloged typed model projection is no longer complete"
    )


def test_opaque_public_data_is_never_silently_excluded() -> None:
    """Refuse a readable noncallable value with no admitted stable representation."""
    model = cast(Any, SimpleNamespace(future_payload=object()))
    with pytest.raises(SnapshotRefusal) as captured:
        _public_field_facts(
            model,
            "baseline",
            "99.0.0",
            {"operation": "STATIC_MODEL_REVIEW"},
        )
    assert captured.value.issue == "opaque_public_model_members"
    assert captured.value.evidence["opaque_member_paths"] == ["future_payload"]


def test_callable_growth_and_comparable_kind_changes_alter_the_surface() -> None:
    """Fingerprint excluded callables and stable kinds even though their values are not hashed."""
    original = measure_public_field_surface(cast(Any, SimpleNamespace(value=1)))
    callable_growth = measure_public_field_surface(
        cast(Any, SimpleNamespace(value=1, future_hook=lambda: None))
    )
    changed_kind = measure_public_field_surface(cast(Any, SimpleNamespace(value="1")))
    assert callable_growth.full_public_surface_count == original.full_public_surface_count + 1
    assert callable_growth.full_public_surface_sha256 != original.full_public_surface_sha256
    assert changed_kind.full_public_surface_sha256 != original.full_public_surface_sha256
    assert changed_kind.comparable_registry_sha256 != original.comparable_registry_sha256


def test_surface_refusal_uses_typed_feature_coverage_reason() -> None:
    """Convert changed model surfaces through the dedicated semantic-coverage refusal ABI."""
    refusal = SnapshotRefusal(
        "baseline",
        "uncharacterized_public_model_surface",
        observed_registry_sha256="0" * 64,
    )
    tool = OperationalToolObservation(
        "0.0.0",
        "VERIFIED_INSTALLED_DISTRIBUTION",
        "1" * 64,
    )
    error = _snapshot_failure(tool, refusal, None)
    assert error.failure.reason.code is OperationalReasonCode.MUJOCO_FEATURE_COVERAGE_INCOMPLETE
    assert error.failure.operation == "review-model"
