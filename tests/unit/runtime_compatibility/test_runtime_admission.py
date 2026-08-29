"""Unit coverage for rolling MuJoCo runtime and feature admission."""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

import mujoco  # type: ignore[import-untyped]
import pytest

from metrifid._model_closure import ModelAdmissionRefusal
from metrifid._mujoco_runtime import (
    MUJOCO_CAPABILITY_INVENTORY,
    MujocoClaimSurface,
    MujocoRuntimeAdmission,
    MujocoSupportTier,
    _native_integer,
    _parse_stable_version,
    admit_model_feature_coverage,
    admit_mujoco_runtime,
    measure_model_feature_facts,
)
from metrifid.operational import OperationalReasonCode


class _RuntimeProxy:
    """Delegate an otherwise real runtime while allowing one test to override attributes."""

    def __init__(self, **overrides: Any) -> None:
        """Retain declared override values on the proxy instance."""
        self.__dict__.update(overrides)

    def __getattr__(self, name: str) -> object:
        """Resolve every non-overridden capability from the installed MuJoCo module."""
        return getattr(mujoco, name)


def _runtime_admission(
    operation: MujocoClaimSurface,
    base_version: str = "3.13.0",
) -> MujocoRuntimeAdmission:
    """Build a coherent typed admission for isolated model-feature tests."""
    major, minor, patch = (int(component) for component in base_version.split("."))
    support_tier = (
        MujocoSupportTier.VALIDATED_EXACT_PROFILE
        if base_version == "3.10.0"
        else MujocoSupportTier.ADMITTED_CAPABILITY_COMPATIBLE_PROFILE
    )
    return MujocoRuntimeAdmission(
        base_version,
        base_version,
        base_version,
        major * 1_000_000 + minor * 1_000 + patch,
        support_tier,
        operation,
        ("MjModel",),
        (("nactuator", True),),
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("3.9.0", (3, 9, 0), id="oldest_validated"),
        pytest.param("3.12.0.post2", (3, 12, 0), id="post_build"),
        pytest.param("3.12.0+vendor.1", (3, 12, 0), id="local_build"),
        pytest.param("3.12.0+Vendor.1", (3, 12, 0), id="mixed_case_local_build"),
        pytest.param("3.13.4.post0+vendor-2", (3, 13, 4), id="future_compatible"),
    ],
)
def test_stable_package_grammar_preserves_the_base_triplet(
    value: str, expected: tuple[int, int, int]
) -> None:
    """Parse every admitted stable suffix without changing its native base identity."""
    assert _parse_stable_version(value, package=True) == expected


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("3.12", id="missing_component"),
        pytest.param("03.12.0", id="ambiguous_leading_zero"),
        pytest.param("3.12.0rc1", id="release_candidate"),
        pytest.param("3.12.0.dev1", id="development_build"),
        pytest.param("3.12.0.POST1", id="noncanonical_post_marker"),
        pytest.param("3.12.0.post", id="empty_post_number"),
        pytest.param("3.12.0+", id="empty_local_token"),
        pytest.param("3.12.0+\u212a", id="unicode_local_token"),
        pytest.param("3.1000.0", id="unencodable_component"),
    ],
)
def test_unstable_or_ambiguous_package_tokens_do_not_parse(value: str) -> None:
    """Refuse every token outside the bounded stable three-component grammar."""
    assert _parse_stable_version(value, package=True) is None


def test_fixed_seed_version_kernel_matches_the_native_integer_algorithm() -> None:
    """Stress the parser/encoder identity boundary with deterministic bounded triplets."""
    randomizer = random.Random(0x4D554A4F434F)
    for _ in range(500):
        triplet = tuple(randomizer.randrange(0, 999) for _component in range(3))
        token = ".".join(str(component) for component in triplet)
        parsed = _parse_stable_version(token, package=True)
        assert parsed == triplet
        assert parsed is not None
        major, minor, patch = parsed
        assert _native_integer(parsed) == major * 1_000_000 + minor * 1_000 + patch


@pytest.mark.parametrize(
    ("package", "expected"),
    [
        pytest.param(
            "3.10.0",
            MujocoSupportTier.ADMITTED_CAPABILITY_COMPATIBLE_PROFILE,
            id="prior_evidence_base",
        ),
        pytest.param(
            "3.10.0.post1",
            MujocoSupportTier.ADMITTED_CAPABILITY_COMPATIBLE_PROFILE,
            id="post_build",
        ),
        pytest.param(
            "3.10.0+vendor.1",
            MujocoSupportTier.ADMITTED_CAPABILITY_COMPATIBLE_PROFILE,
            id="local_build",
        ),
        pytest.param(
            "3.13.0",
            MujocoSupportTier.ADMITTED_CAPABILITY_COMPATIBLE_PROFILE,
            id="future_compatible",
        ),
    ],
)
def test_coherent_stable_runtimes_receive_their_truthful_support_tier(
    package: str, expected: MujocoSupportTier
) -> None:
    """Keep every live coherent runtime at the capability-compatible support tier."""
    parsed = _parse_stable_version(package, package=True)
    assert parsed is not None
    native = ".".join(str(component) for component in parsed)
    proxy = _RuntimeProxy(
        __version__=package,
        mj_versionString=lambda: native,
        mj_version=lambda: _native_integer(parsed),
    )
    admission = admit_mujoco_runtime(MujocoClaimSurface.COMPILED_ARTIFACT, runtime_module=proxy)
    assert admission.support_tier is expected
    assert admission.package_base_version == native


def test_package_and_native_base_versions_must_agree() -> None:
    """Return the mismatch reason when stable package and native identities diverge."""
    proxy = _RuntimeProxy(
        __version__="3.12.0",
        mj_versionString=lambda: "3.11.0",
        mj_version=lambda: 3_011_000,
    )
    with pytest.raises(ModelAdmissionRefusal) as caught:
        admit_mujoco_runtime(MujocoClaimSurface.COMPILED_ARTIFACT, runtime_module=proxy)
    assert caught.value.reason is OperationalReasonCode.MUJOCO_PYTHON_NATIVE_VERSION_MISMATCH
    assert caught.value.evidence["detected_native_version_integer"] == 3_011_000


def test_native_integer_must_be_derived_from_the_reported_triplet() -> None:
    """Return the mismatch reason when the native integer contradicts its stable string."""
    proxy = _RuntimeProxy(
        __version__="3.12.0",
        mj_versionString=lambda: "3.12.0",
        mj_version=lambda: 3_011_000,
    )
    with pytest.raises(ModelAdmissionRefusal) as caught:
        admit_mujoco_runtime(MujocoClaimSurface.COMPILED_ARTIFACT, runtime_module=proxy)
    assert caught.value.reason is OperationalReasonCode.MUJOCO_PYTHON_NATIVE_VERSION_MISMATCH


def test_missing_dynamic_callable_refuses_only_the_affected_inventory() -> None:
    """Keep a replay-only missing callable from blocking complete compiled artifacts."""
    proxy = _RuntimeProxy(mj_step=None)
    compiled = admit_mujoco_runtime(MujocoClaimSurface.COMPILED_ARTIFACT, runtime_module=proxy)
    assert compiled.operation is MujocoClaimSurface.COMPILED_ARTIFACT
    with pytest.raises(ModelAdmissionRefusal) as caught:
        admit_mujoco_runtime(MujocoClaimSurface.DYNAMIC_REPLAY, runtime_module=proxy)
    assert caught.value.reason is OperationalReasonCode.MUJOCO_RUNTIME_CAPABILITY_MISSING
    assert caught.value.evidence["missing_capabilities"] == ("mj_step",)
    assert caught.value.evidence["operation"] == "DYNAMIC_REPLAY"
    assert "support_tier" not in caught.value.evidence


def test_missing_native_package_locator_refuses_before_receipt_identity() -> None:
    """Treat the shared-library locator as a common capability of every native claim."""
    proxy = _RuntimeProxy(__file__=None)
    for surface in MujocoClaimSurface:
        with pytest.raises(ModelAdmissionRefusal) as caught:
            admit_mujoco_runtime(surface, runtime_module=proxy)
        assert caught.value.reason is OperationalReasonCode.MUJOCO_RUNTIME_CAPABILITY_MISSING
        assert caught.value.evidence["missing_capabilities"] == ("__file__",)
        assert caught.value.evidence["operation"] == surface.value


def test_capability_inventories_are_operation_specific_and_lazy() -> None:
    """Bind serialization and stepping APIs only to the claim surfaces that invoke them."""
    names = {
        operation: {requirement.name for requirement in requirements}
        for operation, requirements in MUJOCO_CAPABILITY_INVENTORY.items()
    }
    assert "mj_saveModel" in names[MujocoClaimSurface.COMPILED_ARTIFACT]
    assert "mj_saveModel" not in names[MujocoClaimSurface.DYNAMIC_REPLAY]
    assert "mj_step" in names[MujocoClaimSurface.DYNAMIC_REPLAY]
    assert "mj_step" not in names[MujocoClaimSurface.STATIC_MODEL_REVIEW]
    assert "mjtJoint.mjJNT_HINGE" in names[MujocoClaimSurface.DYNAMIC_REPLAY]
    assert "mjtJoint.mjJNT_HINGE" not in names[MujocoClaimSurface.STATIC_MODEL_REVIEW]
    assert "mjtDyn.mjDYN_FILTER" in names[MujocoClaimSurface.DYNAMIC_REPLAY]
    assert "mjtDyn.mjDYN_FILTER" not in names[MujocoClaimSurface.STATIC_MODEL_REVIEW]


def test_legacy_and_explicit_siso_signatures_measure_equivalently() -> None:
    """Measure old implicit and new explicit one-input/one-output actuator layouts."""
    legacy = SimpleNamespace(nu=2)
    explicit = SimpleNamespace(
        nactuator=2,
        nu=2,
        nout=2,
        actuator_ctrlnum=[1, 1],
        actuator_outnum=[1, 1],
        actuator_ctrlspec=[0, 0],
    )
    legacy_facts = measure_model_feature_facts(legacy)
    explicit_facts = measure_model_feature_facts(explicit)
    assert [
        (item.control_inputs, item.force_outputs) for item in legacy_facts.actuator_signatures
    ] == [
        (1, 1),
        (1, 1),
    ]
    assert [
        (item.control_inputs, item.force_outputs) for item in explicit_facts.actuator_signatures
    ] == [
        (1, 1),
        (1, 1),
    ]


def test_legacy_signature_fallback_is_limited_to_pre_modern_runtime_bases() -> None:
    """Refuse a future claim that omits explicit actuator widths instead of guessing one-to-one."""
    legacy_model = SimpleNamespace(nu=1)
    admitted = admit_model_feature_coverage(
        legacy_model,
        _runtime_admission(MujocoClaimSurface.DYNAMIC_REPLAY, "3.10.0"),
        "candidate",
    )
    assert admitted.legacy_implicit_signature is True

    for surface in (
        MujocoClaimSurface.STATIC_MODEL_REVIEW,
        MujocoClaimSurface.DYNAMIC_REPLAY,
    ):
        with pytest.raises(ModelAdmissionRefusal) as caught:
            admit_model_feature_coverage(
                legacy_model,
                _runtime_admission(surface),
                "candidate",
            )
        assert caught.value.reason is OperationalReasonCode.MUJOCO_FEATURE_COVERAGE_INCOMPLETE
        issues = caught.value.evidence["unsupported_features"]["measurement_issues"]
        assert issues == ("modern_actuator_signature_fields_absent",)


def test_multi_input_actuator_refuses_semantic_claim_before_descriptor_indexing() -> None:
    """Refuse a control signature the one-control action contract cannot represent."""
    model = SimpleNamespace(
        nactuator=1,
        nu=2,
        nout=1,
        actuator_ctrlnum=[2],
        actuator_outnum=[1],
        actuator_ctrlspec=[3],
    )
    with pytest.raises(ModelAdmissionRefusal) as caught:
        admit_model_feature_coverage(
            model, _runtime_admission(MujocoClaimSurface.DYNAMIC_REPLAY), "candidate"
        )
    assert caught.value.reason is OperationalReasonCode.MUJOCO_FEATURE_COVERAGE_INCOMPLETE
    assert caught.value.evidence["unsupported_features"]["actuator_signatures"] == (
        {
            "actuator_index": 0,
            "control_inputs": 2,
            "force_outputs": 1,
            "control_spec": 3,
        },
    )


def test_complete_artifact_records_but_does_not_interpret_new_model_features() -> None:
    """Allow exact compiled bytes while retaining MIMO, surface-velocity, and adhesion facts."""
    model = SimpleNamespace(
        nactuator=1,
        nu=0,
        nout=2,
        actuator_ctrlnum=[0],
        actuator_outnum=[2],
        actuator_ctrlspec=[0],
        geom_surfacevel=[[0.5, 0.0]],
        geom_adhesion=[1.0],
    )
    facts = admit_model_feature_coverage(
        model, _runtime_admission(MujocoClaimSurface.COMPILED_ARTIFACT), "baseline"
    )
    assert facts.unsupported_actuator_signatures()
    assert facts.active_surface_velocity_fields == ("geom_surfacevel",)
    assert facts.active_adhesion_fields == ("geom_adhesion",)


def test_joint_state_replay_does_not_blanket_refuse_new_contact_fields() -> None:
    """Admit a representable SISO trace while retaining surface-velocity and adhesion presence."""
    model = SimpleNamespace(
        nactuator=1,
        nu=1,
        nout=1,
        actuator_ctrlnum=[1],
        actuator_outnum=[1],
        actuator_ctrlspec=[0],
        geom_surfacevel=[[0.5, 0.0]],
        geom_adhesion=[1.0],
    )
    facts = admit_model_feature_coverage(
        model, _runtime_admission(MujocoClaimSurface.DYNAMIC_REPLAY), "candidate"
    )
    assert facts.unsupported_actuator_signatures() == ()
    assert facts.active_surface_velocity_fields == ("geom_surfacevel",)
    assert facts.active_adhesion_fields == ("geom_adhesion",)


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            SimpleNamespace(
                nactuator=0,
                nu=1,
                nout=0,
                actuator_ctrlnum=[],
                actuator_outnum=[],
                actuator_ctrlspec=[],
            ),
            id="incoherent_empty_signature",
        ),
        pytest.param(
            SimpleNamespace(
                nactuator=1,
                nu=1,
                nout=1,
                actuator_ctrlnum=[1.5],
                actuator_outnum=[1],
                actuator_ctrlspec=[0],
            ),
            id="noninteger_signature",
        ),
        pytest.param(
            SimpleNamespace(
                nactuator=1.5,
                nu=1,
                nout=1,
                actuator_ctrlnum=[1],
                actuator_outnum=[1],
                actuator_ctrlspec=[0],
            ),
            id="noninteger_actuator_count",
        ),
    ],
)
def test_malformed_actuator_dimensions_fail_closed_before_semantic_projection(
    model: SimpleNamespace,
) -> None:
    """Refuse incoherent or noninteger actuator signatures instead of coercing them."""
    with pytest.raises(ModelAdmissionRefusal) as caught:
        admit_model_feature_coverage(
            model,
            _runtime_admission(MujocoClaimSurface.STATIC_MODEL_REVIEW),
            "baseline",
        )
    assert caught.value.reason is OperationalReasonCode.MUJOCO_FEATURE_COVERAGE_INCOMPLETE
    assert caught.value.evidence["unsupported_features"]["measurement_issues"]
