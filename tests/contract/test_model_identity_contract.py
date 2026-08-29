"""Collect model identity contract scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

import copy
import inspect
import os
import sys
import xml.etree.ElementTree as ET
from collections import namedtuple
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import pytest

import metrifid
from metrifid import _model_admission as admission
from metrifid import _model_closure as closure
from metrifid import _model_compile as model_compile
from metrifid import _model_descriptor_builders as descriptor_builders
from metrifid import _model_descriptors as model_descriptors
from metrifid import _model_identity as identity
from metrifid._model_dependencies import first_complete_root_element as _first_root
from metrifid.json_values import canonical_sha256, freeze_canonical
from metrifid.operational import OperationalReasonCode
from metrifid.schemas import (
    ActuatorAliasEndpoint,
    ActuatorAliasPair,
    AliasArtifact,
    JointAliasPair,
    TargetReference,
)
from metrifid.version import __version__ as CURRENT_VERSION

SHA = "a" * 64
_SOURCE_ROOT = os.environ.get("METRIFID_CANDIDATE_SOURCE_ROOT")
ROOT = (
    Path(_SOURCE_ROOT).resolve()
    if _SOURCE_ROOT is not None
    else Path(__file__).resolve().parents[2]
)


def _joint(name: str = "joint") -> admission.JointDescriptor:
    """Construct one named hinge descriptor with deterministic layout addresses."""
    return admission.JointDescriptor(name, "HINGE", 1, 1, 0, 0)


def _actuator(name: str | None = "act", control: int = 0) -> admission.ActuatorDescriptor:
    """Construct a joint-transmission actuator descriptor at one control index."""
    return admission.ActuatorDescriptor(
        name,
        "JOINT",
        (TargetReference("JOINT", "joint"),),
        "NONE",
        0,
        control,
        None,
    )


def _compiled(
    *,
    actuators: tuple[admission.ActuatorDescriptor, ...] = (_actuator(),),
) -> admission.CompiledModelIdentity:
    """Finalize a minimal compiled identity with canonically sorted actuators."""
    return admission.CompiledModelIdentity(
        admission.CompiledModelIdentity._SCHEMA,
        admission.CompiledModelIdentity._SCHEMA_VERSION,
        None,
        SHA,
        1,
        1,
        len(actuators),
        0,
        (_joint(),),
        tuple(sorted(actuators, key=model_descriptors._actuator_sort_key)),
    ).finalized()


def _refusal_reason(
    exc: pytest.ExceptionInfo[closure.ModelAdmissionRefusal],
) -> OperationalReasonCode:
    """Extract the stable reason code from a captured model-admission refusal."""
    return exc.value.reason


def test_model_admission_remains_internal_to_the_public_api() -> None:
    """Verify that model-admission types stay behind the documented public imports."""
    expected = {
        "__version__",
        "Binary64",
        "ExactRational",
        "canonical_json_bytes",
        "canonical_sha256",
        "strict_json_loads",
        "ComparisonStatus",
        "EngineThreadpoolState",
        "LimitationCode",
        "OperationalExitCode",
        "ReasonCode",
        "ReasonRecord",
        "STATUS_PRECEDENCE",
        "REASON_REGISTRY",
        "ComparisonConfig",
        "ComparisonContractIdentity",
        "ComparisonReceipt",
        "finalize_receipt",
        "validate_receipt",
        "OperationalStage",
        "OperationalReasonCode",
        "OperationalFailure",
        "OperationalToolObservation",
        "InputDigestCode",
        "InputDigest",
        "write_state_artifact",
        "write_actions_artifact",
    }
    assert set(metrifid.__all__) == expected
    assert not any(
        name.startswith("Model") or name.startswith("Semantic") for name in metrifid.__all__
    )
    assert metrifid.__version__ == CURRENT_VERSION


_Version = namedtuple("_Version", "major minor micro releaselevel serial")


def _set_python(monkeypatch: pytest.MonkeyPatch, major: int, minor: int) -> None:
    """Present one interpreter version tuple to the shared runtime gate."""
    monkeypatch.setattr(model_compile.sys, "version_info", _Version(major, minor, 0, "final", 0))


def _admit_platform(monkeypatch: pytest.MonkeyPatch, system: str = "Linux") -> None:
    """Present a POSIX operating system the shared runtime gate accepts."""
    monkeypatch.setattr(model_compile.os, "name", "posix")
    monkeypatch.setattr(model_compile.platform, "system", lambda: system)


def test_python_below_the_minimum_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse Python 3.10 and report the measured version and the declared minimum."""
    _set_python(monkeypatch, 3, 10)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        admission.require_supported_runtime()
    assert _refusal_reason(exc) is OperationalReasonCode.UNSUPPORTED_PYTHON_VERSION
    assert exc.value.evidence["minimum_python"] == "3.11"


@pytest.mark.parametrize("minor", [11, 14])
def test_supported_and_future_minors_are_not_refused_by_version(
    monkeypatch: pytest.MonkeyPatch, minor: int
) -> None:
    """Admit 3.11 and a later minor: the gate has no upper bound and no minor allowlist.

    This asserts the gate's own decision only. Real validation of a Python minor comes from the
    compatibility matrix lane that runs the installed wheel on that interpreter.
    """
    _set_python(monkeypatch, 3, minor)
    _admit_platform(monkeypatch)
    admission.require_supported_runtime()


def test_interpreter_implementation_name_is_never_inspected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admit the runtime regardless of the reported interpreter implementation name."""
    _admit_platform(monkeypatch)
    monkeypatch.setattr(model_compile.platform, "python_implementation", lambda: "SomeOtherPython")
    admission.require_supported_runtime()


@pytest.mark.parametrize("system", ["Linux", "Darwin"])
@pytest.mark.parametrize("machine", ["x86_64", "X86_64", "arm64", "aarch64", "riscv64"])
def test_posix_systems_are_admitted_on_every_architecture(
    monkeypatch: pytest.MonkeyPatch, system: str, machine: str
) -> None:
    """Admit Linux and Darwin on any machine string: architecture is never a gate."""
    _admit_platform(monkeypatch, system)
    monkeypatch.setattr(model_compile.platform, "machine", lambda: machine)
    admission.require_supported_runtime()


@pytest.mark.parametrize("system", ["Windows", "Java", "FreeBSD", "", "linux"])
def test_unsupported_operating_systems_refuse(monkeypatch: pytest.MonkeyPatch, system: str) -> None:
    """Refuse any operating system outside Linux and Darwin, reporting the measured name."""
    monkeypatch.setattr(model_compile.os, "name", "posix")
    monkeypatch.setattr(model_compile.platform, "system", lambda: system)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        admission.require_supported_runtime()
    assert _refusal_reason(exc) is OperationalReasonCode.UNSUPPORTED_PLATFORM
    assert exc.value.evidence["system"] == system


def test_native_windows_refuses_on_a_non_posix_os_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse a non-POSIX ``os.name`` and record it as evidence."""
    monkeypatch.setattr(model_compile.os, "name", "nt")
    monkeypatch.setattr(model_compile.platform, "system", lambda: "Windows")
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        admission.require_supported_runtime()
    assert _refusal_reason(exc) is OperationalReasonCode.UNSUPPORTED_PLATFORM
    assert exc.value.evidence["os_name"] == "nt"


@pytest.mark.parametrize(
    ("attribute", "removed", "expected"),
    [
        ("supports_dir_fd", "open", "dir_fd:open"),
        ("supports_dir_fd", "unlink", "dir_fd:unlink"),
        ("supports_follow_symlinks", "stat", "follow_symlinks:stat"),
        ("supports_fd", "scandir", "fd:scandir"),
    ],
)
def test_each_missing_posix_capability_set_refuses(
    monkeypatch: pytest.MonkeyPatch, attribute: str, removed: str, expected: str
) -> None:
    """Refuse when any required dir_fd, follow_symlinks, or fd capability is absent."""
    _admit_platform(monkeypatch)
    present = set(getattr(model_compile.os, attribute)) - {getattr(model_compile.os, removed)}
    monkeypatch.setattr(model_compile.os, attribute, present)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        admission.require_supported_runtime()
    assert _refusal_reason(exc) is OperationalReasonCode.UNSUPPORTED_PLATFORM
    assert expected in exc.value.evidence["missing_posix_capabilities"]


def test_a_missing_required_callable_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse when a required POSIX callable is not provided by this interpreter."""
    _admit_platform(monkeypatch)
    monkeypatch.setattr(model_compile.os, "pread", None, raising=False)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        admission.require_supported_runtime()
    assert _refusal_reason(exc) is OperationalReasonCode.UNSUPPORTED_PLATFORM
    assert "callable:pread" in exc.value.evidence["missing_posix_capabilities"]


def test_a_missing_required_open_flag_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse when a required open flag is absent or zero."""
    _admit_platform(monkeypatch)
    monkeypatch.setattr(model_compile.os, "O_NOFOLLOW", 0, raising=False)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        admission.require_supported_runtime()
    assert _refusal_reason(exc) is OperationalReasonCode.UNSUPPORTED_PLATFORM
    assert "flag:O_NOFOLLOW" in exc.value.evidence["missing_posix_capabilities"]


@pytest.mark.parametrize(
    ("package_version", "native_version", "native_integer"),
    [
        pytest.param("3.9.0", "3.9.0", 3_009_000, id="oldest_validated"),
        pytest.param("3.10.0", "3.10.0", 3_010_000, id="prior_validated"),
        pytest.param("3.11.0", "3.11.0", 3_011_000, id="current_validated"),
        pytest.param("3.12.0", "3.12.0", 3_012_000, id="latest_validated"),
        pytest.param("3.12.0.post1", "3.12.0", 3_012_000, id="post_build"),
        pytest.param("3.13.0+local", "3.13.0", 3_013_000, id="future_compatible"),
    ],
)
def test_stable_coherent_mujoco_runtime_is_admitted(
    monkeypatch: pytest.MonkeyPatch,
    package_version: str,
    native_version: str,
    native_integer: int,
) -> None:
    """Admit stable coherent validated, rebuilt, and future-capable runtime identities."""
    _admit_platform(monkeypatch)
    monkeypatch.setattr(model_compile.mujoco, "__version__", package_version)
    monkeypatch.setattr(model_compile.mujoco, "mj_versionString", lambda: native_version)
    monkeypatch.setattr(model_compile.mujoco, "mj_version", lambda: native_integer)
    admission.require_supported_runtime()


@pytest.mark.parametrize(
    "package_version",
    [
        pytest.param("3.10.0rc1", id="unstable_rc"),
        pytest.param("3.10.0a1", id="unstable_alpha"),
        pytest.param("3.10.0b2", id="unstable_beta"),
        pytest.param("3.10.0.dev0", id="development"),
        pytest.param("3.10.0.post1.dev0", id="post_development"),
        pytest.param("3.8.9", id="below_floor"),
        pytest.param("3.10", id="missing_patch"),
        pytest.param("", id="empty_token"),
    ],
)
def test_unstable_malformed_or_below_floor_mujoco_packages_refuse(
    monkeypatch: pytest.MonkeyPatch, package_version: str
) -> None:
    """Refuse prerelease, development, malformed, and below-floor package identities."""
    _admit_platform(monkeypatch)
    monkeypatch.setattr(model_compile.mujoco, "__version__", package_version)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        admission.require_supported_runtime()
    assert _refusal_reason(exc) is OperationalReasonCode.UNSUPPORTED_MUJOCO_VERSION


def test_a_non_string_mujoco_version_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse a MuJoCo distribution whose reported version is not a string."""
    _admit_platform(monkeypatch)
    monkeypatch.setattr(model_compile.mujoco, "__version__", None)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        admission.require_supported_runtime()
    assert _refusal_reason(exc) is OperationalReasonCode.UNSUPPORTED_MUJOCO_VERSION
    assert exc.value.evidence["mujoco_python_version"] is None


def test_package_and_native_engine_mismatch_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse an admitted package whose loaded native engine reports another base."""
    _admit_platform(monkeypatch)
    monkeypatch.setattr(model_compile.mujoco, "__version__", "3.10.0.post1")
    monkeypatch.setattr(model_compile.mujoco, "mj_versionString", lambda: "3.9.0")
    monkeypatch.setattr(model_compile.mujoco, "mj_version", lambda: 3090000)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        admission.require_supported_runtime()
    assert _refusal_reason(exc) is OperationalReasonCode.MUJOCO_PYTHON_NATIVE_VERSION_MISMATCH
    assert exc.value.evidence["native_version_integer"] == 3090000


def test_the_shared_gate_is_the_only_exported_runtime_admission() -> None:
    """Expose one shared runtime gate and no alias for the removed Certify-only name."""
    assert hasattr(admission, "require_supported_runtime")
    assert not hasattr(admission, "require_supported_certify_runtime")
    assert "require_supported_runtime" in admission.__all__


def test_compile_and_admission_role_guards_and_parse_error(tmp_path: Path) -> None:
    """Guard role-local reasons and translate malformed entrypoints into compile refusals."""
    assert model_compile._compile_reason("baseline", warning=True) is (
        OperationalReasonCode.BASELINE_MODEL_COMPILE_WARNING
    )
    assert model_compile._compile_reason("candidate", warning=False) is (
        OperationalReasonCode.CANDIDATE_MODEL_COMPILE_ERROR
    )
    with pytest.raises(ValueError):
        model_compile._compile_reason("comparison", warning=False)
    with pytest.raises(ValueError):
        admission.compile_snapshot_model(tmp_path / "missing.xml", "comparison")
    with pytest.raises(ValueError):
        admission.admit_compiled_model(SimpleNamespace(), "comparison")  # type: ignore[arg-type]
    malformed = tmp_path / "root"
    malformed.mkdir()
    (malformed / "model.xml").write_text("<mujoco>", encoding="utf-8")
    with closure.create_model_closure_snapshot(malformed, "model.xml", "baseline") as snapshot:
        with pytest.raises(closure.ModelAdmissionRefusal) as exc:
            model_compile._require_mjcf_root(snapshot, "baseline")
    assert _refusal_reason(exc) is OperationalReasonCode.BASELINE_MODEL_COMPILE_ERROR
    assert exc.value.evidence["issue"] == "no_complete_top_level_element"


def _admission_model(**updates: Any) -> SimpleNamespace:
    """Build a callback-free model namespace whose admission fields can be overridden."""
    values: dict[str, Any] = {
        "nplugin": 0,
        "npluginstate": 0,
        "body_plugin": None,
        "geom_plugin": None,
        "actuator_plugin": None,
        "sensor_plugin": None,
        "nhistory": 0,
        "actuator_historyadr": None,
        "sensor_historyadr": None,
        "nmocap": 0,
        "body_mocapid": [-1],
        "nu": 0,
        "actuator_dyntype": [],
        "actuator_gaintype": [],
        "actuator_biastype": [],
        "nsensor": 0,
        "sensor_type": [],
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_plugin_sensor_branch_without_plugin_reference_array() -> None:
    """Refuse plugin sensors even when the plugin-reference array is absent."""
    model = _admission_model(
        nsensor=1,
        sensor_type=[int(mujoco.mjtSensor.mjSENS_PLUGIN)],
    )
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        admission.admit_compiled_model(model, "candidate")  # type: ignore[arg-type]
    assert _refusal_reason(exc) is OperationalReasonCode.UNSUPPORTED_PLUGIN_STATE


def test_compiled_identity_rejects_unsorted_actuators() -> None:
    """Reject compiled identities whose actuator descriptors are not canonically ordered."""
    first = _actuator("z", 0)
    second = _actuator("a", 1)
    with pytest.raises(ValueError):
        admission.CompiledModelIdentity(
            admission.CompiledModelIdentity._SCHEMA,
            admission.CompiledModelIdentity._SCHEMA_VERSION,
            None,
            SHA,
            1,
            1,
            2,
            0,
            (_joint(),),
            (first, second),
        )


def _target_model() -> SimpleNamespace:
    """Build the minimal object-count surface used for actuator target resolution."""
    return SimpleNamespace(njnt=1, ntendon=1, nsite=2, nbody=2)


def test_target_validation_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse invalid, unnamed, ambiguous, and unknown actuator targets."""
    model = _target_model()
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        model_descriptors._resolve_actuator_target_reference(
            model, int(mujoco.mjtObj.mjOBJ_JOINT), -1, "baseline", 0
        )  # type: ignore[arg-type]
    assert _refusal_reason(exc) is OperationalReasonCode.ACTUATOR_TARGET_MISMATCH

    monkeypatch.setattr(model_compile.mujoco, "mj_id2name", lambda *args: None)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        model_descriptors._resolve_actuator_target_reference(
            model, int(mujoco.mjtObj.mjOBJ_JOINT), 0, "baseline", 0
        )  # type: ignore[arg-type]
    assert _refusal_reason(exc) is OperationalReasonCode.ACTUATOR_TARGET_MISMATCH

    monkeypatch.setattr(model_compile.mujoco, "mj_id2name", lambda *args: "dup")
    monkeypatch.setattr(descriptor_builders, "_count_model_object_names", lambda *args: {"dup": 2})
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        model_descriptors._resolve_actuator_target_reference(
            model, int(mujoco.mjtObj.mjOBJ_JOINT), 0, "baseline", 0
        )  # type: ignore[arg-type]
    assert _refusal_reason(exc) is OperationalReasonCode.ACTUATOR_IDENTITY_AMBIGUOUS

    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        model_descriptors._actuator_targets(
            SimpleNamespace(actuator_trnid=[[0, -1]]),  # type: ignore[arg-type]
            0,
            "baseline",
            "UNKNOWN",
        )
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_LAYOUT_INCOMPATIBLE


def test_joint_and_actuator_descriptor_extraction_rejections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject duplicate names and unknown native types during descriptor extraction."""
    duplicate_joint_model = SimpleNamespace(
        njnt=2,
        jnt_type=[int(mujoco.mjtJoint.mjJNT_HINGE)] * 2,
        jnt_qposadr=[0, 1],
        jnt_dofadr=[0, 1],
    )
    monkeypatch.setattr(model_compile.mujoco, "mj_id2name", lambda *args: "same")
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        model_descriptors._joint_descriptors(duplicate_joint_model, "baseline")  # type: ignore[arg-type]
    assert _refusal_reason(exc) is OperationalReasonCode.JOINT_NAME_DUPLICATE

    unknown_joint_model = SimpleNamespace(
        njnt=1,
        jnt_type=[999],
        jnt_qposadr=[0],
        jnt_dofadr=[0],
    )
    monkeypatch.setattr(model_compile.mujoco, "mj_id2name", lambda *args: "joint")
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        model_descriptors._joint_descriptors(unknown_joint_model, "candidate")  # type: ignore[arg-type]
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_LAYOUT_INCOMPATIBLE

    duplicate_actuator_model = SimpleNamespace(nu=2)
    monkeypatch.setattr(model_compile.mujoco, "mj_id2name", lambda *args: "same")
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        model_descriptors._actuator_descriptors(duplicate_actuator_model, "baseline")  # type: ignore[arg-type]
    assert _refusal_reason(exc) is OperationalReasonCode.ACTUATOR_IDENTITY_AMBIGUOUS

    unknown_actuator_model = SimpleNamespace(
        nu=1,
        actuator_trntype=[999],
        actuator_dyntype=[999],
    )
    monkeypatch.setattr(model_compile.mujoco, "mj_id2name", lambda *args: "act")
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        model_descriptors._actuator_descriptors(unknown_actuator_model, "candidate")  # type: ignore[arg-type]
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_LAYOUT_INCOMPATIBLE


def test_layout_coverage_failure_branches() -> None:
    """Require joint and actuator spans to cover each compiled state layout exactly."""
    assert model_descriptors._covers(0, ())
    assert not model_descriptors._covers(2, ((0, 1),))
    assert model_descriptors._covers(2, ((1, 1), (0, 1)))
    assert not model_descriptors._covers(1, ((0, 0),))
    model = SimpleNamespace(nq=2, nv=1, nu=0, na=0)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        model_descriptors._validate_layout(model, (_joint(),), (), "baseline")  # type: ignore[arg-type]
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_LAYOUT_INCOMPATIBLE


def test_semantic_alignment_type_sort_and_binding_parser_branches() -> None:
    """Round-trip bound aliases while rejecting malformed or unsorted alignment components."""
    valid = identity.align_compiled_models(_compiled(), _compiled())
    _assert_semantic_alignment_type_and_order_refusals(valid)
    binding = freeze_canonical(
        {
            "kind": "JOINT",
            "canonical_name": "joint",
            "baseline_name": "joint",
            "candidate_name": "joint",
        }
    )
    artifact = AliasArtifact(
        "metrifid.aliases",
        1,
        SHA,
        SHA,
        (JointAliasPair("joint", "joint", "joint"),),
        (),
    )
    with_binding = replace(
        valid,
        semantic_alignment_sha256=None,
        aliases_raw_sha256=SHA,
        aliases_semantic_sha256=canonical_sha256(artifact.to_primitive()),
        alias_bindings=(binding,),  # type: ignore[arg-type]
    ).finalized()
    assert identity.SemanticAlignment.from_primitive(with_binding.to_primitive()) == with_binding


def _assert_semantic_alignment_type_and_order_refusals(
    valid: identity.SemanticAlignment,
) -> None:
    """Exercise joint, actuator, and canonical-order validation branches."""
    with pytest.raises(TypeError):
        identity.SemanticAlignment(
            identity.SemanticAlignment._SCHEMA,
            identity.SemanticAlignment._SCHEMA_VERSION,
            None,
            None,
            None,
            (object(),),  # type: ignore[arg-type]
            valid.actuators,
            (),
        )
    with pytest.raises(TypeError):
        identity.SemanticAlignment(
            identity.SemanticAlignment._SCHEMA,
            identity.SemanticAlignment._SCHEMA_VERSION,
            None,
            None,
            None,
            valid.joints,
            (object(),),  # type: ignore[arg-type]
            (),
        )
    with pytest.raises(ValueError):
        identity.SemanticAlignment(
            identity.SemanticAlignment._SCHEMA,
            identity.SemanticAlignment._SCHEMA_VERSION,
            None,
            None,
            None,
            tuple(reversed((*valid.joints, replace(valid.joints[0], canonical_name="z")))),
            valid.actuators,
            (),
        )


def test_duplicate_actuator_alias_consumption_refuses() -> None:
    """Refuse two aliases that consume the same actuator endpoint."""
    endpoint = ActuatorAliasEndpoint("NAMED", "act", None, (), None, None)
    aliases = AliasArtifact(
        "metrifid.aliases",
        1,
        SHA,
        SHA,
        (),
        (
            ActuatorAliasPair("a", endpoint, endpoint),
            ActuatorAliasPair("b", endpoint, endpoint),
        ),
    )
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity.align_compiled_models(_compiled(), _compiled(), aliases)
    assert _refusal_reason(exc) is OperationalReasonCode.ALIAS_BINDING_DUPLICATE


def test_component_validation_hidden_branches() -> None:
    """Reject dot components and strings whose encoded bytes change unexpectedly."""
    with pytest.raises(closure.ModelAdmissionRefusal):
        closure._validate_component(".", "baseline", ".")

    class ChangedEncoding(str):
        """Represent changed encoding."""

        def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
            """Return bytes inconsistent with this string's visible path component."""
            del encoding, errors
            return b"different"

    with pytest.raises(closure.ModelAdmissionRefusal):
        closure._validate_component(ChangedEncoding("same"), "candidate", "same")


def test_completed_object_hash_mutation_attacks_are_total() -> None:
    """Reject missing or forged self-hashes on completed identities and alignments."""
    compiled = _compiled()
    alignment = identity.align_compiled_models(compiled, compiled)
    for parser, primitive, hash_field in (
        (
            admission.CompiledModelIdentity.from_primitive,
            compiled.to_primitive(),
            "compiled_identity_sha256",
        ),
        (
            identity.SemanticAlignment.from_primitive,
            alignment.to_primitive(),
            "semantic_alignment_sha256",
        ),
    ):
        for replacement in (None, "0" * 64):
            changed = copy.deepcopy(primitive)
            changed[hash_field] = replacement
            with pytest.raises(ValueError):
                parser(changed)


def test_no_native_index_selector_exists() -> None:
    """Prevent unstable native indices from entering the public actuator alias schema."""
    parameters = inspect.signature(ActuatorAliasEndpoint).parameters
    assert "index" not in parameters
    assert "native_index" not in parameters
    assert set(parameters) == {
        "kind",
        "name",
        "transmission_type",
        "targets",
        "activation_family",
        "activation_width",
    }
    # No Python-minor allowlist is asserted here: the supported range is 3.11 with no upper
    # bound, and the alias schema does not vary by interpreter minor.
    assert sys.version_info >= (3, 11)


# ==============================================================================================
# Bounded main-root precheck.
#
# The precheck reads the measured entrypoint through the same no-follow, size- and hash-verified
# path as dependency discovery, requires the first complete top-level element to be `mujoco`, and
# ignores trailing bytes. The admitted stable MuJoCo runtime remains the syntax and compile authority.
# ==============================================================================================


_VALID_MODEL_BODY = (
    '<worldbody><body name="b"><joint name="j" type="hinge"/>'
    '<geom name="g" type="sphere" size="0.1"/></body></worldbody>'
)


def _write_model_root(tmp_path: Path, text: str, name: str = "root") -> Path:
    """Write one model entrypoint beneath a named temporary closure root."""
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.xml").write_text(text, encoding="utf-8")
    return root


def _precheck(root: Path) -> None:
    """Run the bounded first-root check against a measured closure snapshot."""
    with closure.create_model_closure_snapshot(root, "model.xml", "baseline") as snapshot:
        model_compile._require_mjcf_root(snapshot, "baseline")


def test_trailing_bytes_after_first_root_pass_the_precheck(tmp_path: Path) -> None:
    """19. a first complete `mujoco` root passes even when a strict full-document parse fails."""
    text = f"<mujoco>{_VALID_MODEL_BODY}</mujoco>\n</mujoco>\n"
    with pytest.raises(ET.ParseError):
        ET.fromstring(text)
    _precheck(_write_model_root(tmp_path, text))
    # The same document must reach native MuJoCo and compile.
    root = _write_model_root(tmp_path, text, name="compiles")
    with closure.create_model_closure_snapshot(root, "model.xml", "baseline") as snapshot:
        assert int(admission.compile_snapshot_model(snapshot, "baseline").nu) == 0


def test_non_mujoco_first_root_refuses(tmp_path: Path) -> None:
    """20. a non-`mujoco` first root refuses with the existing compile-error behavior."""
    root = _write_model_root(tmp_path, f"<robot>{_VALID_MODEL_BODY}</robot>")
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        _precheck(root)
    assert _refusal_reason(exc) is OperationalReasonCode.BASELINE_MODEL_COMPILE_ERROR
    assert exc.value.evidence["root_element"] == "robot"
    assert exc.value.evidence["required_root_element"] == "mujoco"


def test_no_complete_root_refuses(tmp_path: Path) -> None:
    """21. a document with no complete top-level element refuses."""
    for index, text in enumerate(("<mujoco>", "<mujoco><worldbody>", "", "   \n")):
        root = _write_model_root(tmp_path, text, name=f"case{index}")
        with pytest.raises(closure.ModelAdmissionRefusal) as exc:
            _precheck(root)
        assert _refusal_reason(exc) is OperationalReasonCode.BASELINE_MODEL_COMPILE_ERROR
        assert exc.value.evidence["issue"] == "no_complete_top_level_element"


def test_22a_mutated_entrypoint_refuses_before_native_compile(tmp_path: Path) -> None:
    """22. a mutated measured entrypoint refuses before native compile."""
    root = _write_model_root(tmp_path, f"<mujoco>{_VALID_MODEL_BODY}</mujoco>")
    with closure.create_model_closure_snapshot(root, "model.xml", "baseline") as snapshot:
        (snapshot.snapshot_root / "model.xml").write_text(
            f"<mujoco>{_VALID_MODEL_BODY}<!-- tampered --></mujoco>", encoding="utf-8"
        )
        with pytest.raises(closure.ModelAdmissionRefusal) as exc:
            model_compile._require_mjcf_root(snapshot, "baseline")
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID


def test_22b_missing_entrypoint_refuses_before_native_compile(tmp_path: Path) -> None:
    """22. a removed measured entrypoint refuses before native compile."""
    root = _write_model_root(tmp_path, f"<mujoco>{_VALID_MODEL_BODY}</mujoco>")
    with closure.create_model_closure_snapshot(root, "model.xml", "baseline") as snapshot:
        (snapshot.snapshot_root / "model.xml").unlink()
        with pytest.raises(closure.ModelAdmissionRefusal) as exc:
            model_compile._require_mjcf_root(snapshot, "baseline")
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID


def test_22c_symlinked_and_nonregular_entrypoints_refuse(tmp_path: Path) -> None:
    """22. a symlinked or non-regular measured entrypoint refuses before native compile."""
    real = tmp_path / "real.xml"
    real.write_text(f"<mujoco>{_VALID_MODEL_BODY}</mujoco>", encoding="utf-8")
    root = _write_model_root(tmp_path, f"<mujoco>{_VALID_MODEL_BODY}</mujoco>", name="swap")
    with closure.create_model_closure_snapshot(root, "model.xml", "baseline") as snapshot:
        target = snapshot.snapshot_root / "model.xml"
        target.unlink()
        target.symlink_to(real)
        with pytest.raises(closure.ModelAdmissionRefusal) as exc:
            model_compile._require_mjcf_root(snapshot, "baseline")
        assert _refusal_reason(exc) is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID
        target.unlink()
        try:
            os.mkfifo(target)
        except (AttributeError, NotImplementedError, OSError):
            pytest.skip("fifo creation unavailable on this platform")
        with pytest.raises(closure.ModelAdmissionRefusal) as fifo_exc:
            model_compile._require_mjcf_root(snapshot, "baseline")
        assert _refusal_reason(fifo_exc) is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID


def test_native_mujoco_remains_the_compile_authority(tmp_path: Path) -> None:
    """23. after the root precheck passes, native MuJoCo still reports a real compile error."""
    root = _write_model_root(
        tmp_path,
        '<mujoco><worldbody><body name="b"><joint name="j" type="hinge"/>'
        '<geom name="g" type="sphere" size="-1"/></body></worldbody></mujoco>',
    )
    with closure.create_model_closure_snapshot(root, "model.xml", "baseline") as snapshot:
        model_compile._require_mjcf_root(snapshot, "baseline")
        with pytest.raises(closure.ModelAdmissionRefusal) as exc:
            admission.compile_snapshot_model(snapshot, "baseline")
    assert _refusal_reason(exc) is OperationalReasonCode.BASELINE_MODEL_COMPILE_ERROR
    assert "message" in exc.value.evidence


def test_first_complete_root_element_is_not_a_recovering_parser() -> None:
    """The bounded reader must not repair broken markup before the first complete element."""
    assert _first_root(b"<mujoco/>").tag == "mujoco"
    assert _first_root(b"<mujoco></mujoco>trailing").tag == "mujoco"
    assert _first_root(b"<mujoco>") is None
    assert _first_root(b"") is None
    assert _first_root(b"<<<not xml") is None
    # A broken element before the first complete one is not silently skipped.
    assert _first_root(b"<a><b></a></b>") is None
