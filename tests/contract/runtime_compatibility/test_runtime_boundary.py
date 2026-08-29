"""Contract tests for the private rolling-runtime admission boundary."""

from __future__ import annotations

import inspect

import metrifid
from metrifid import _model_admission, _model_compile
from metrifid._mujoco_runtime import (
    MujocoClaimSurface,
    MujocoRuntimeAdmission,
    MujocoSupportTier,
)


def test_runtime_admission_remains_private_to_native_product_paths() -> None:
    """Keep rolling-runtime types and tier tokens out of the public root SDK."""
    for name in (
        "MujocoClaimSurface",
        "MujocoRuntimeAdmission",
        "MujocoSupportTier",
        "admit_mujoco_runtime",
    ):
        assert name not in metrifid.__all__
        assert not hasattr(metrifid, name)


def test_private_facade_exposes_typed_runtime_evidence_interfaces() -> None:
    """Provide one stable private seam to native orchestrators without public export."""
    assert _model_admission.MujocoClaimSurface is MujocoClaimSurface
    assert _model_admission.MujocoRuntimeAdmission is MujocoRuntimeAdmission
    assert _model_admission.MujocoSupportTier is MujocoSupportTier
    admission = _model_admission.require_supported_runtime(MujocoClaimSurface.COMPILED_ARTIFACT)
    assert isinstance(admission, MujocoRuntimeAdmission)
    evidence = admission.to_evidence()
    assert evidence["operation"] == "COMPILED_ARTIFACT"
    assert evidence["package_base_version"] == admission.native_version_string
    assert evidence["support_tier"] == "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE"


def test_compile_module_contains_no_exact_minor_runtime_wall() -> None:
    """Delegate version policy instead of retaining the removed exact-regex implementation."""
    source = inspect.getsource(_model_compile)
    assert "_MUJOCO_PACKAGE_PATTERN" not in source
    assert "native_string !=" not in source
    assert "3010000" not in source


def test_callback_accessors_are_names_resolved_inside_the_compile_boundary() -> None:
    """Avoid eager optional callback binding while retaining the complete callback registry."""
    assert all(
        isinstance(getter_name, str) for _name, getter_name in _model_compile._CALLBACK_ACCESSORS
    )
    assert all(
        getter_name.startswith("get_") for _name, getter_name in _model_compile._CALLBACK_ACCESSORS
    )
